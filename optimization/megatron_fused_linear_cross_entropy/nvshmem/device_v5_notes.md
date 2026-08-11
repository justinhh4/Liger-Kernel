# Device NVSHMEM Variant 5: Persistent Bandwidth Grid

## Hypothesis

Variant 4 launches one block per 1,024 bytes, producing 8,192 blocks for the
32 MiB dX buffer. A capped 1,024-block grid with grid-stride `float4` loops
should retain enough occupancy to saturate NVLink while reducing block
scheduling and kernel-tail overhead.

## Changes

- Cap the direct-load reduction grid at 1,024 blocks.
- Process additional vectors with a grid-stride loop.
- Keep the hoisted peer mappings and vectorized loads from Variant 4.

## Expected Outcome

Improve BT=2,048 backward by at least 5% without regressing BT=512, making all
four TP4 full-step cases guardrail-compliant.

## Actual Results

TP4 correctness passed, but the capped grid regressed the best Variant 4
results. BT=2,048 full-step latency was 8.8% slower at global vocab 32,000 and
1.0% slower at 128,256. The smaller grid did not saturate peer bandwidth as
effectively as the uncapped launch.

## Verdict

**REJECTED.** Grid scheduling was not the bottleneck; reducing communication
volume is required.
