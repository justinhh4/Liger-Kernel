# Device NVSHMEM Variant 15: 16,384-Block Grid

## Hypothesis

More CTAs may further expose peer-memory latency.

## Changes

- Increased the low-precision collective cap to 16,384 blocks.

## Actual Results

BT16K regressed to 2.7579/9.5098 ms and BT32K/V128K regressed to 19.6547 ms.

## Verdict

**REJECTED.** Launch and scheduling overhead exceeded the latency-hiding gain.
