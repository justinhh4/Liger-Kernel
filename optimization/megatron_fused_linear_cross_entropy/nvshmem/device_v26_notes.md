# Device NVSHMEM Variant 26: Specialized 32-Byte Vectors

## Hypothesis

**Target bottleneck:** In the v25 TP8/BT16K/V32K profile, dW takes about
309 us while fused dX communication takes about 531 us, leaving roughly
220 us exposed. TP4 communication is already hidden.

The old dynamic-loop vector16 experiment regressed TP4, but v25 now
specializes TP count and rank at compile time. Doubling each direct peer
transaction from 16 to 32 bytes should halve loop/index work and improve TP8
NVLink transaction efficiency without the old dynamic-loop overhead.

## Changes

- Widen the low-precision vector from eight to sixteen elements.
- Keep FP32 accumulation, TP/rank specialization, fused broadcast, 256-thread
  CTAs, and the 4,096-block cap.

## Expected Outcome

Reduce TP8 communication by 8-15% and full-step latency by 2-5%. Reject if
TP4 realistic latency regresses by more than 1%.

## Actual Results

Correctness passed, but TP8 full-step latency regressed from 1.7213 to
2.0382 ms at BT16K and from 3.1672 to 3.5116 ms at BT32K. The wider vector
increased register pressure and reduced useful concurrency.

## Verdict

**REJECTED.**
