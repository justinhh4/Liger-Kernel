# Device NVSHMEM Variant 11: Parallel Forward MAX

## Hypothesis

A single-block team MAX underutilizes B200 for BT16K/32K token vectors.

## Changes

- Retained the block collective through 8,192 elements.
- Used parallel `float4` direct-peer loads for larger MAX reductions.

## Actual Results

TP4 full-step latency improved over Variant 10 at three of four realistic
shapes and reached 2.9711 ms at BT16K/V32K.

## Verdict

**ACCEPTED.**
