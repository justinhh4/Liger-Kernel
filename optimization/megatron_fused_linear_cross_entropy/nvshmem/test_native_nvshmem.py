from __future__ import annotations

import ctypes
import os
import tempfile

from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

BRIDGE = Path(__file__).with_name("libliger_nvshmem.so")


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
    bridge = ctypes.CDLL(str(BRIDGE), mode=ctypes.RTLD_GLOBAL)
    bridge.liger_nvshmem_uniqueid_size.restype = ctypes.c_size_t
    bridge.liger_nvshmem_get_uniqueid.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    bridge.liger_nvshmem_init.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    bridge.liger_nvshmem_register_symmetric.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    bridge.liger_nvshmem_register_symmetric.restype = ctypes.c_void_p
    bridge.liger_nvshmem_unregister_symmetric.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    bridge.liger_nvshmem_malloc.argtypes = [ctypes.c_size_t]
    bridge.liger_nvshmem_malloc.restype = ctypes.c_void_p
    bridge.liger_nvshmem_free.argtypes = [ctypes.c_void_p]
    bridge.liger_nvshmem_float_max_reduce.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    bridge.liger_nvshmem_float_sum_reduce.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    return bridge


def _wrap_float32(pointer, shape, device):
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
            dtype=_DLDataType(code=2, bits=32, lanes=1),
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


def _symmetric_float32(bridge, shape, device):
    numel = 1
    for size in shape:
        numel *= size
    pointer = bridge.liger_nvshmem_malloc(numel * 4)
    if not pointer:
        raise RuntimeError("nvshmem_malloc returned null")
    tensor, owner = _wrap_float32(pointer, shape, device)
    return tensor, pointer, owner


def _worker(rank, world_size, rendezvous):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)
    bridge = _load_bridge()

    uniqueid_size = bridge.liger_nvshmem_uniqueid_size()
    uniqueid = torch.empty(uniqueid_size, device="cuda", dtype=torch.uint8)
    if rank == 0:
        host_uniqueid = (ctypes.c_ubyte * uniqueid_size)()
        status = bridge.liger_nvshmem_get_uniqueid(host_uniqueid, uniqueid_size)
        if status != 0:
            raise RuntimeError(f"nvshmemx_get_uniqueid failed with status {status}")
        uniqueid.copy_(torch.frombuffer(host_uniqueid, dtype=torch.uint8))
    dist.broadcast(uniqueid, src=0)
    host_uniqueid = (ctypes.c_ubyte * uniqueid_size).from_buffer_copy(uniqueid.cpu().numpy())
    status = bridge.liger_nvshmem_init(rank, world_size, host_uniqueid, uniqueid_size)
    if status != 0:
        raise RuntimeError(f"NVSHMEM initialization failed with status {status}")

    source, source_pointer, source_owner = _symmetric_float32(bridge, (8192,), rank)
    maximum, maximum_pointer, maximum_owner = _symmetric_float32(bridge, (8192,), rank)
    total, total_pointer, total_owner = _symmetric_float32(bridge, (8192,), rank)
    source.copy_(torch.arange(source.numel(), device=source.device) / source.numel() + rank)
    tensors = (source, maximum, total)
    pointers = (source_pointer, maximum_pointer, total_pointer)
    owners = (source_owner, maximum_owner, total_owner)

    stream = torch.cuda.current_stream().cuda_stream
    status = bridge.liger_nvshmem_float_max_reduce(pointers[1], pointers[0], source.numel(), stream)
    if status != 0:
        raise RuntimeError(f"NVSHMEM MAX reduction failed with status {status}")
    status = bridge.liger_nvshmem_float_sum_reduce(pointers[2], pointers[0], source.numel(), stream)
    if status != 0:
        raise RuntimeError(f"NVSHMEM SUM reduction failed with status {status}")
    torch.cuda.synchronize()
    torch.testing.assert_close(maximum, source if rank == world_size - 1 else source + world_size - 1 - rank)
    expected_total = torch.arange(source.numel(), device=source.device) * world_size / source.numel()
    expected_total += world_size * (world_size - 1) / 2
    torch.testing.assert_close(total, expected_total)

    dist.barrier()
    torch.cuda.synchronize()
    for pointer in pointers:
        bridge.liger_nvshmem_free(pointer)
    del tensors, owners
    dist.barrier()
    bridge.liger_nvshmem_finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile() as rendezvous:
        mp.spawn(_worker, args=(2, rendezvous.name), nprocs=2, join=True)
