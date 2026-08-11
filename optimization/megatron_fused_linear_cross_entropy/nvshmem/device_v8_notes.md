# Device NVSHMEM Variant 8: Forward Collective Block Sweep

## Hypothesis

Variant 7 makes dX faster than NCCL at every tested shape, but its 256-thread
forward team reductions are 4-5% slower at BT=2,048. Sweeping 128, 256, and 512
threads should identify a better balance between per-thread work and collective
coordination for 2,048-4,096 FP32 elements.

## Changes

- Benchmark block-scoped forward MAX/SUM collectives at 128, 256, and 512
  threads.
- Keep Variant 7's 2,048-block reduce-scatter/allgather unchanged.
- Keep communication entirely device-side NVSHMEM.

## Expected Outcome

Reduce BT=2,048 forward latency by at least 4% while preserving the BT=512
full-step wins.

## Actual Results

The 512-thread candidate failed smoke correctness with CUDA launch status 701
(too many resources for the inlined NVSHMEM collective). The 128-thread
candidate passed correctness but increased BT=2,048 forward latency to
0.292-0.580 ms, versus 0.271-0.562 ms for 256 threads.

## Verdict

**REJECTED.** The existing 256-thread block is the only efficient valid
configuration.
