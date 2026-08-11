# Device NVSHMEM Variant 23: 2,048-CTA Overlap

## Hypothesis

**Target bottleneck:** Variant 22 improves TP4 and TP8 with 4,096 CTAs,
confirming that overlap quality matters more than minimum isolated collective
latency. A 2,048-CTA grid may further reduce dW contention, but each thread
must process four vectors at TP4/BT16K.

## Changes

- Reduce the low-precision collective grid cap from 4,096 to 2,048 blocks.
- Keep fused broadcast and TP/rank specialization unchanged.

## Expected Outcome

Determine the lower side of the overlap optimum. Accept only if full-step
latency improves over Variant 22 without a small-shape regression.

## Actual Results

At TP4/V32K, full-step latency was 2.5512 ms at BT16K and 4.9195 ms
at BT32K, versus 2.5483 and 4.9217 ms for Variant 22. The differences are
within run-to-run noise and the smaller grid does not improve both shapes.

## Verdict

**REJECTED.** Restore the 4,096-block cap.
