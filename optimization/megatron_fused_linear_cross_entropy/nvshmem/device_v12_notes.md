# Device NVSHMEM Variant 12: Eight-Element dX Vectors

## Hypothesis

The low-precision peer loop is instruction-bound after halving traffic.
Loading eight values per thread should reduce indexing and loop overhead.

## Changes

- Replaced pairwise peer loads with aligned 16-byte/eight-element loads.
- Kept FP32 accumulation and reduce-scatter/allgather partitioning.

## Actual Results

TP4 full-step latency improved to 2.7452/9.4374 ms at BT16K and
5.4181/19.5299 ms at BT32K for V32K/V128K.

## Verdict

**ACCEPTED.** This was the largest post-BF16 communication improvement.
