# Device NVSHMEM Variant 19: Fused Reduce-and-Broadcast

## Hypothesis

**Target bottleneck:** The TP4/BT16K/V32K profile shows a 744 us dX
reduce-scatter followed by a 162 us allgather. dW ends about 135 us before the
reduce-scatter, leaving the allgather entirely on the critical path.

Each vector has exactly one reduce-scatter owner. Once that owner has loaded
all source values, it can safely write the completed result to every peer's
destination because no other rank reads that vector. Fusing those stores into
the reduction should overlap broadcast traffic with reduction work and remove
the separate allgather kernel.

## Changes

- Add mutable direct-peer destination mappings to the low-precision launch.
- After FP32 accumulation and one low-precision rounding, store each owned
  vector locally and directly to every peer destination.
- Replace the reduce-scatter barrier plus allgather with one final readiness
  barrier that guarantees remote-store visibility.
- Preserve the existing pairwise and generic fallbacks.

## Expected Outcome

Reduce exposed TP4 V32K backward latency by 80-160 us (5-10%) and full-step
latency by 3-6%, without changing communication volume or numerical results.

## Actual Results

The fused kernel took about 887 us versus roughly 913 us for Variant 18's
reduce-scatter, barrier, and allgather sequence. TP4/BT16K/V32K full-step
latency improved from 2.7049 to 2.6897 ms (0.6%).

## Verdict

**ACCEPTED AS A BASE FOR SPECIALIZATION.** Fusion is correct but remote stores
mostly move the allgather cost into the reduction kernel.
