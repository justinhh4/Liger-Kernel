# Device NVSHMEM Variant 4: Hoisted Vector Peer Loads

## Hypothesis

Variant 3 calls `nvshmem_ptr` once per dX element and peer, adding millions of
invariant device calls. Resolve peer mappings once on the host launcher and
load aligned `float4` vectors directly across NVLink. This should cut dX
reduction time by 20-40% and remove the BT=2,048 regressions.

## Changes

- Resolve up to 16 peer mappings once per reduction launch.
- Pass the mapping table by value to the CUDA kernel.
- Reduce four aligned FP32 values per thread.
- Retain scalar NVSHMEM gets only when direct peer mapping is unavailable.

## Expected Outcome

Keep BT=512 full-step wins and bring both BT=2,048 full-step cases to parity or
better than NCCL.

## Actual Results

TP4 correctness passed. Full-step latency improved by 17.3% and 12.5% at
BT=512. At BT=2,048, the regressions fell from Variant 3's 21.9%/6.9% to
6.7%/2.1%. Backward was within 4.5% of NCCL in every case.

## Verdict

**MIXED.** Three of four cases meet the speed guardrail, but BT=2,048 with
global vocab 32,000 remains 6.7% slower. The next variant targets excess CUDA
block scheduling in the direct-load kernel.
