# Device NVSHMEM Variant 22: Balanced 4,096-CTA Overlap

## Hypothesis

**Target bottleneck:** Variant 21 reduces fused communication to about 681 us,
nearly identical to the overlapped dW duration. Compared with Variant 19, dW
grows from about 615 to 675 us while the more aggressive communication kernel
is active, indicating SM/L2 contention.

Using 4,096 CTAs with two vectors per thread at BT16K/TP4 may reduce scheduling
and cache pressure enough to shorten dW while keeping communication close to
its current duration.

## Changes

- Reduce the low-precision collective grid cap from 8,192 to 4,096 blocks.
- Keep TP/rank specialization, fused broadcast, and 256-thread CTAs.

## Expected Outcome

Reduce the overlapped `max(dW, communication)` region by 2-4% and full-step
latency by 0.5-1.5%.

## Actual Results

TP4/V32K full-step latency reached 2.5483 ms at BT16K and 4.9217 ms at
BT32K. TP8 reached 1.7159 ms at BT16K and 3.1542 ms at BT32K. All points
improved over the 8,192-CTA Variant 21/v18 measurements.

## Verdict

**ACCEPTED.** 4,096 CTAs is the best overlap configuration.
