from __future__ import annotations

import ctypes
import os

from pathlib import Path

import torch
import torch.distributed as dist

from optimization.megatron_fused_linear_cross_entropy.nvshmem.nvshmem_workspace import _wrap_tensor


def _load_bridge():
    path = Path(__file__).with_name("libliger_nvshmem_device.so")
    bridge = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    bridge.liger_nvshmem_uniqueid_size.restype = ctypes.c_size_t
    bridge.liger_nvshmem_get_uniqueid.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    bridge.liger_nvshmem_init.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    bridge.liger_nvshmem_malloc.argtypes = [ctypes.c_size_t]
    bridge.liger_nvshmem_malloc.restype = ctypes.c_void_p
    bridge.liger_nvshmem_free.argtypes = [ctypes.c_void_p]
    bridge.liger_nvshmem_device_zero.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    reduction_args = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_size_t,
    ]
    bridge.liger_nvshmem_device_float_max_reduce.argtypes = reduction_args
    bridge.liger_nvshmem_device_float_sum_reduce.argtypes = reduction_args
    bridge.liger_nvshmem_device_bfloat16_sum_reduce.argtypes = reduction_args
    bridge.liger_nvshmem_device_float16_sum_reduce.argtypes = reduction_args
    return bridge


class DeviceNvshmemReductionWorkspace:
    """Symmetric buffers reduced by GPU-initiated NVSHMEM kernels."""

    def __init__(self, max_tokens, hidden_size, process_group=None, device=None, dx_dtype=torch.float32):
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized first")
        self.process_group = process_group if process_group is not None else dist.group.WORLD
        if self.process_group is not dist.group.WORLD:
            raise ValueError("the experimental NVSHMEM workspace requires the world process group")
        self.rank = dist.get_rank(self.process_group)
        self.world_size = dist.get_world_size(self.process_group)
        if self.world_size > 16:
            raise ValueError("the experimental device NVSHMEM backend supports at most 16 ranks")
        self.device = torch.device(device if device is not None else f"cuda:{self.rank}")
        self.max_tokens = int(max_tokens)
        self.hidden_size = int(hidden_size)
        if dx_dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise TypeError(f"dx_dtype must be float32, bfloat16, or float16, got {dx_dtype}")
        self.dx_dtype = dx_dtype
        os.environ.setdefault("NVSHMEM_BOOTSTRAP", "UID")
        os.environ.setdefault("NVSHMEM_DISABLE_NVLS", "1")
        self._bridge = _load_bridge()
        self._pointers = []
        self._owners = []
        self.allocated_bytes = 0
        self._epoch = 0
        self._closed = False

        uniqueid_size = self._bridge.liger_nvshmem_uniqueid_size()
        uniqueid = torch.empty(uniqueid_size, device=self.device, dtype=torch.uint8)
        if self.rank == 0:
            host_uniqueid = (ctypes.c_ubyte * uniqueid_size)()
            status = self._bridge.liger_nvshmem_get_uniqueid(host_uniqueid, uniqueid_size)
            if status != 0:
                raise RuntimeError(f"nvshmemx_get_uniqueid failed with status {status}")
            uniqueid.copy_(torch.frombuffer(host_uniqueid, dtype=torch.uint8))
        dist.broadcast(uniqueid, src=0, group=self.process_group)
        host_uniqueid = (ctypes.c_ubyte * uniqueid_size).from_buffer_copy(uniqueid.cpu().numpy())
        status = self._bridge.liger_nvshmem_init(
            self.rank,
            self.world_size,
            host_uniqueid,
            uniqueid_size,
        )
        if status != 0:
            raise RuntimeError(f"NVSHMEM initialization failed with status {status}")

        self.max_source = self._allocate((self.max_tokens,), torch.float32)
        self.max_destination = self._allocate((self.max_tokens,), torch.float32)
        self.stats_source = self._allocate((2 * self.max_tokens,), torch.float32)
        self.stats_destination = self._allocate((2 * self.max_tokens,), torch.float32)
        self.dx_source = self._allocate((self.max_tokens, self.hidden_size), self.dx_dtype)
        # Reduce-scatter owns disjoint vector chunks, so the aligned path can
        # overwrite and allgather those chunks in the source allocation.
        self._inplace_dx = self.hidden_size % 8 == 0 and self.world_size <= 16
        self.dx_destination = (
            self.dx_source if self._inplace_dx else self._allocate((self.max_tokens, self.hidden_size), self.dx_dtype)
        )
        self._signals = self._allocate_raw(self.world_size * ctypes.sizeof(ctypes.c_uint64))
        stream = torch.cuda.current_stream(self.device)
        status = self._bridge.liger_nvshmem_device_zero(
            self._signals,
            self.world_size * ctypes.sizeof(ctypes.c_uint64),
            stream.cuda_stream,
        )
        if status != 0:
            raise RuntimeError(f"NVSHMEM signal initialization failed with CUDA status {status}")
        torch.cuda.synchronize(self.device)
        self.overlap_stream = torch.cuda.Stream(device=self.device)
        self.dx_ready = torch.cuda.Event()

    def _allocate_raw(self, size):
        pointer = self._bridge.liger_nvshmem_malloc(size)
        if not pointer:
            raise RuntimeError("nvshmem_malloc returned null")
        self._pointers.append(pointer)
        self.allocated_bytes += size
        return pointer

    def _allocate(self, shape, dtype):
        numel = 1
        for size in shape:
            numel *= size
        pointer = self._allocate_raw(numel * dtype.itemsize)
        tensor, owner = _wrap_tensor(pointer, shape, self.device.index, dtype)
        self._owners.append(owner)
        return tensor

    def _next_epoch(self):
        self._epoch += 2
        return self._epoch

    def _check_tokens(self, num_tokens):
        if not 0 < num_tokens <= self.max_tokens:
            raise ValueError(f"num_tokens must be in [1, {self.max_tokens}], got {num_tokens}")

    def max_buffers(self, num_tokens):
        self._check_tokens(num_tokens)
        return self.max_source[:num_tokens], self.max_destination[:num_tokens]

    def stats_buffers(self, num_tokens):
        self._check_tokens(num_tokens)
        source = self.stats_source[: 2 * num_tokens]
        destination = self.stats_destination[: 2 * num_tokens]
        return (
            source[:num_tokens],
            source[num_tokens:],
            destination[:num_tokens],
            destination[num_tokens:],
        )

    def dx_buffers(self, num_tokens, hidden_size):
        self._check_tokens(num_tokens)
        if hidden_size != self.hidden_size:
            raise ValueError(f"hidden size must be {self.hidden_size}, got {hidden_size}")
        return self.dx_source[:num_tokens], self.dx_destination[:num_tokens]

    def max_reduce(self, source, destination):
        stream = torch.cuda.current_stream(self.device)
        status = self._bridge.liger_nvshmem_device_float_max_reduce(
            destination.data_ptr(),
            source.data_ptr(),
            source.numel(),
            self._signals,
            self._next_epoch(),
            stream.cuda_stream,
        )
        if status != 0:
            raise RuntimeError(f"device NVSHMEM MAX reduction launch failed with CUDA status {status}")

    def sum_reduce(self, source, destination):
        stream = torch.cuda.current_stream(self.device)
        reduction = {
            torch.float32: self._bridge.liger_nvshmem_device_float_sum_reduce,
            torch.bfloat16: self._bridge.liger_nvshmem_device_bfloat16_sum_reduce,
            torch.float16: self._bridge.liger_nvshmem_device_float16_sum_reduce,
        }[source.dtype]
        status = reduction(
            destination.data_ptr(),
            source.data_ptr(),
            source.numel(),
            self._signals,
            self._next_epoch(),
            stream.cuda_stream,
        )
        if status != 0:
            raise RuntimeError(f"device NVSHMEM SUM reduction launch failed with CUDA status {status}")

    def close(self):
        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.process_group)
        for name in (
            "max_source",
            "max_destination",
            "stats_source",
            "stats_destination",
            "dx_source",
            "dx_destination",
        ):
            setattr(self, name, None)
        self.overlap_stream = None
        self.dx_ready = None
        for pointer in reversed(self._pointers):
            self._bridge.liger_nvshmem_free(pointer)
        self._pointers.clear()
        self._owners.clear()
        dist.barrier(group=self.process_group)
        self._bridge.liger_nvshmem_finalize()
        self._closed = True
