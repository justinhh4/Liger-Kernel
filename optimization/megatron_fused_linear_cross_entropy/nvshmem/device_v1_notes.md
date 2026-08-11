# Device NVSHMEM Variant 1: Direct Peer Loads

## Hypothesis

The host-driven prototype is synchronization-bound rather than
communication-bandwidth-bound. Moving readiness synchronization and peer
access into CUDA kernels should remove three host-visible CUDA completion
fences. On the fully connected B200 NVLink topology, direct peer loads should
reduce full-step latency by 5-15% versus production NCCL at TP4.

## Changes

- Compile CUDA communication kernels with RDC and NVSHMEM device linkage.
- Use GPU-side symmetric readiness signals with monotonically increasing
  epochs.
- Reduce small forward statistics in one resident block.
- Reduce large dX buffers in parallel after a device-side readiness handshake.
- Prefer `nvshmem_ptr` direct peer loads and fall back to `nvshmem_float_g`.
- Keep all production FLCE compute and public APIs unchanged.

## Expected Outcome

Loss and all gradients should match the materialized reference on TP2 and TP4.
The variant must beat NCCL full-step latency without a pass regressing by more
than 10%.

## Actual Results

TP2 smoke correctness passed. On TP4 B200:

| BT | Global V | NCCL full (ms) | Device NVSHMEM full (ms) | Change |
|---:|---:|---:|---:|---:|
| 512 | 32,000 | 0.5519 | 0.4573 | -17.1% |
| 512 | 128,256 | 0.5633 | 0.5186 | -7.9% |
| 2,048 | 32,000 | 0.5734 | 0.7865 | +37.2% |
| 2,048 | 128,256 | 1.3132 | 1.5196 | +15.7% |

The single-block forward reductions scale poorly beyond 1,024 values. The
direct-load dX all-reduce also moves `TP - 1` full copies into every rank and
loses to NCCL for the 32 MiB per-rank dX buffer.

## Verdict

**REJECTED.** The large-token regressions violate the 5% full-step and 10%
cross-pass guardrails. Kernel-side signaling is validated, and the small-token
speedups justify testing an optimized device collective.
