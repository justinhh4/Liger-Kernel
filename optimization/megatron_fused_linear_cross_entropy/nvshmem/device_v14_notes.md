# Device NVSHMEM Variant 14: 8,192-Block Grid

## Hypothesis

BT32K still has sufficient work to benefit from another grid-cap increase.

## Changes

- Increased the low-precision collective cap to 8,192 blocks.

## Actual Results

TP4 full-step latency reached 2.7096/9.4460 ms at BT16K and
5.3209/19.2796 ms at BT32K for V32K/V128K. TP8 was 6.7-15.7% faster than
NCCL across the same realistic matrix.

## Verdict

**ACCEPTED.** This is the winning launch configuration.
