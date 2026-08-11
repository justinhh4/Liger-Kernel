# Device NVSHMEM Variant 27: TP8 Subwarp Cooperation

## Hypothesis

**Target bottleneck:** v25 TP8 communication takes about 531 us while dW
takes about 309 us. One thread currently performs seven peer loads, a
dependent FP32 accumulation chain, and seven peer stores for each vector.

Assigning one eight-thread subwarp per vector lets each lane load one rank's
16-byte vector concurrently. A width-eight shuffle tree reduces the eight
values in FP32, then each lane writes the result to one rank concurrently.
Traffic volume is unchanged, but peer-operation dependency depth falls from
seven to one.

## Changes

- Restore eight-element/16-byte vectors.
- Add a TP8-only eight-thread cooperative reduce-and-broadcast kernel.
- Retain the per-thread TP2/TP4 specialized kernels.
- Keep the 4,096-block cap and 256-thread CTAs.

## Expected Outcome

Reduce TP8 communication by 15-30% and full-step latency by 5-10%, while
leaving TP4 unchanged.

## Actual Results

Correctness passed, but TP8 full-step latency regressed to 2.0325 ms at
BT16K and 3.9409 ms at BT32K. Eight threads per vector reduced available
vector-level parallelism too aggressively, and shuffle/loop overhead exceeded
the shorter peer dependency chain.

## Verdict

**REJECTED.**
