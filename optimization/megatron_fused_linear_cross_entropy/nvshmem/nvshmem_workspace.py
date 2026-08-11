from __future__ import annotations

import ctypes

from pathlib import Path

import torch
import torch.distributed as dist


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DLDeleter),
]
_NOOP_DELETER = _DLDeleter(lambda _: None)
_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]


def _load_bridge():
    path = Path(__file__).with_name("libliger_nvshmem.so")
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
    reduction_args = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    bridge.liger_nvshmem_float_max_reduce.argtypes = reduction_args
    bridge.liger_nvshmem_float_sum_reduce.argtypes = reduction_args
    bridge.liger_nvshmem_barrier.argtypes = [ctypes.c_size_t]
    return bridge


def _wrap_tensor(pointer, shape, device, dtype):
    dtype_config = {
        torch.float32: (2, 32),
        torch.float16: (2, 16),
        torch.bfloat16: (4, 16),
    }
    if dtype not in dtype_config:
        raise TypeError(f"unsupported DLPack dtype: {dtype}")
    dtype_code, dtype_bits = dtype_config[dtype]
    shape_storage = (ctypes.c_int64 * len(shape))(*shape)
    strides = []
    stride = 1
    for size in reversed(shape):
        strides.append(stride)
        stride *= size
    stride_storage = (ctypes.c_int64 * len(shape))(*reversed(strides))
    managed = _DLManagedTensor(
        dl_tensor=_DLTensor(
            data=pointer,
            device=_DLDevice(device_type=2, device_id=device),
            ndim=len(shape),
            dtype=_DLDataType(code=dtype_code, bits=dtype_bits, lanes=1),
            shape=shape_storage,
            strides=stride_storage,
            byte_offset=0,
        ),
        manager_ctx=None,
        deleter=_NOOP_DELETER,
    )
    capsule = _PyCapsule_New(ctypes.addressof(managed), b"dltensor", None)
    tensor = torch.utils.dlpack.from_dlpack(capsule)
    return tensor, (shape_storage, stride_storage, managed)


def _wrap_float32(pointer, shape, device):
    return _wrap_tensor(pointer, shape, device, torch.float32)


class NativeNvshmemReductionWorkspace:
    """Symmetric FP32 buffers for experimental FLCE reductions."""

    def __init__(self, max_tokens, hidden_size, process_group=None, device=None):
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized first")
        self.process_group = process_group if process_group is not None else dist.group.WORLD
        if self.process_group is not dist.group.WORLD:
            raise ValueError("the experimental NVSHMEM workspace requires the world process group")
        self.rank = dist.get_rank(self.process_group)
        self.world_size = dist.get_world_size(self.process_group)
        self.device = torch.device(device if device is not None else f"cuda:{self.rank}")
        self.max_tokens = int(max_tokens)
        self.hidden_size = int(hidden_size)
        self._bridge = _load_bridge()
        self._pointers = []
        self._owners = []
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

        self.max_source = self._allocate((self.max_tokens,))
        self.max_destination = self._allocate((self.max_tokens,))
        self.stats_source = self._allocate((2 * self.max_tokens,))
        self.stats_destination = self._allocate((2 * self.max_tokens,))
        self.dx_source = self._allocate((self.max_tokens, self.hidden_size))
        self.dx_destination = self._allocate((self.max_tokens, self.hidden_size))
        self.overlap_stream = torch.cuda.Stream(device=self.device)
        self.dx_ready = torch.cuda.Event()

    def _allocate(self, shape):
        numel = 1
        for size in shape:
            numel *= size
        pointer = self._bridge.liger_nvshmem_malloc(numel * 4)
        if not pointer:
            raise RuntimeError("nvshmem_malloc returned null")
        tensor, owner = _wrap_float32(pointer, shape, self.device.index)
        self._pointers.append(pointer)
        self._owners.append(owner)
        return tensor

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
        status = self._bridge.liger_nvshmem_float_max_reduce(
            destination.data_ptr(),
            source.data_ptr(),
            source.numel(),
            stream.cuda_stream,
        )
        if status != 0:
            raise RuntimeError(f"NVSHMEM MAX reduction failed with status {status}")
        torch.cuda.synchronize(self.device)

    def sum_reduce(self, source, destination):
        stream = torch.cuda.current_stream(self.device)
        status = self._bridge.liger_nvshmem_float_sum_reduce(
            destination.data_ptr(),
            source.data_ptr(),
            source.numel(),
            stream.cuda_stream,
        )
        if status != 0:
            raise RuntimeError(f"NVSHMEM SUM reduction failed with status {status}")
        torch.cuda.synchronize(self.device)

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
        for pointer in reversed(self._pointers):
            self._bridge.liger_nvshmem_free(pointer)
        self._pointers.clear()
        self._owners.clear()
        self.overlap_stream = None
        self.dx_ready = None
        dist.barrier(group=self.process_group)
        self._bridge.liger_nvshmem_finalize()
        self._closed = True
