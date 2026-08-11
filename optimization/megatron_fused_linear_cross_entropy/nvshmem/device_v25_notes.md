# Device NVSHMEM Variant 25: Shape-Gated CuTile dX

## Hypothesis

**Target bottleneck:** Variant 24 shows a clear dX implementation crossover:
CuTile improves BT512/2048 but regresses BT16K. A token-count gate can preserve
both regimes without changing numerical behavior.

## Changes

- Use the tuned CuTile dX kernel for at most 2,048 tokens.
- Keep cuBLAS `torch.mm(out=...)` above 2,048 tokens.
- Retain Variant 22's fused rank-specialized NVSHMEM communication.

## Expected Outcome

Preserve Variant 24's 3-6% low-BT gains and Variant 22's realistic throughput
results.

## Actual Results

TP4/V32K full-step latency was 0.4344 ms at BT512, 0.4667 ms at BT2048,
2.5284 ms at BT16K, and 4.9314 ms at BT32K. The low-BT points retain the
CuTile gain while the realistic points remain statistically consistent with
Variant 22.

## Verdict

**ACCEPTED.** Final winner.
