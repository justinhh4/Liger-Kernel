from __future__ import annotations

import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from optimization.megatron_fused_linear_cross_entropy.nvshmem.device_nvshmem_workspace import (
    DeviceNvshmemReductionWorkspace,
)


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
    workspace = DeviceNvshmemReductionWorkspace(
        max_tokens=32,
        hidden_size=257,
        device=torch.device("cuda", rank),
    )

    source, maximum = workspace.max_buffers(32)
    source.copy_(torch.arange(source.numel(), device=source.device) / source.numel() + rank)
    workspace.max_reduce(source, maximum)

    dx_source, total = workspace.dx_buffers(32, 257)
    dx_source.copy_(torch.arange(dx_source.numel(), device=source.device).reshape_as(dx_source) / dx_source.numel())
    dx_source.add_(rank)
    workspace.sum_reduce(dx_source, total)
    torch.cuda.synchronize()

    expected_max = torch.arange(source.numel(), device=source.device) / source.numel() + world_size - 1
    expected_total = torch.arange(dx_source.numel(), device=source.device).reshape_as(dx_source)
    expected_total = expected_total * world_size / dx_source.numel()
    expected_total += world_size * (world_size - 1) / 2
    torch.testing.assert_close(maximum, expected_max)
    torch.testing.assert_close(total, expected_total)

    del source, maximum, dx_source, total, expected_max, expected_total
    workspace.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile() as rendezvous:
        mp.spawn(_worker, args=(2, rendezvous.name), nprocs=2, join=True)
