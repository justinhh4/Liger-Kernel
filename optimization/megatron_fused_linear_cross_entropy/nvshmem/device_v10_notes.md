# Device NVSHMEM Variant 10: Low-Precision dX

## Hypothesis

The realistic-shape dX all-reduce is bandwidth-bound, so communicating the
model dtype instead of FP32 should halve remote traffic.

## Changes

- Added BF16 and FP16 symmetric tensors and bridge entry points.
- Accumulated peer values in FP32 and rounded once on output.
- Wrote the dX GEMM directly into the low-precision symmetric source.

## Actual Results

At TP4 BF16, Variant 9's 0.1-6.5% realistic regressions became 0.0-2.2% wins.
BF16 correctness passed at TP2, TP4, and TP8.

## Verdict

**ACCEPTED.**
