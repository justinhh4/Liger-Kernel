# Device NVSHMEM Variant 34: Dual TP8 Accumulators

## Hypothesis

**Target bottleneck:** v32 retains one dependent FP32 add chain across seven
peer vectors. Two independent even/odd peer accumulators should expose more
load and arithmetic instruction-level parallelism without reducing
vector-level parallelism.

## Changes

- Restore v32's `.cg` peer traffic.
- For TP8 only, accumulate even and odd peer ranks independently.
- Combine the two FP32 accumulators once before rounding.
- Keep TP2/TP4 unchanged.

## Expected Outcome

Reduce TP8 data-kernel latency by 2-5% without changing traffic or occupancy.

## Actual Results

TP8 full-step latency improved from v32's 1.6830 to 1.6517 ms at BT16K,
but regressed from 3.0475 to 3.0712 ms at BT32K. The extra accumulator helps
when each thread owns one vector, then hurts once each thread loops over two.

## Verdict

**MIXED.** Gate by communication work per thread.
