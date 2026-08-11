# Device NVSHMEM Variant 21: Rank-Specialized Peer Loops

## Hypothesis

**Target bottleneck:** TP-count specialization reduced TP4 V32K backward by
7.2% versus Variant 19. The unrolled loop still compares every constant peer
index against runtime `peers.me`, and chunk ownership remains dynamic.

Specializing both TP count and local rank should eliminate all self-peer
branches, make every peer pointer index constant, and constant-fold the
reduce-scatter chunk offset.

## Changes

- Add local rank as a compile-time template parameter.
- Dispatch rank-specialized TP2, TP4, and TP8 kernels.
- Preserve a dynamic fallback for other world sizes.

## Expected Outcome

Reduce fused communication by another 2-5% and improve full-step V32K latency
by 1-2%.

## Actual Results

TP4/BT16K/V32K backward improved from 1.5623 to 1.5290 ms and full-step
improved from 2.5766 to 2.5625 ms. Nsight Systems measured the fused kernel at
about 681 us, down from about 887 us before TP/rank specialization.

## Verdict

**ACCEPTED.**
