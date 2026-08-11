# Device NVSHMEM Variant 16: 512 Threads

## Hypothesis

Larger CTAs may issue more direct-peer loads concurrently.

## Changes

- Doubled low-precision collective blocks from 256 to 512 threads.

## Actual Results

The realistic matrix regressed versus the 256-thread Variant 14, including
2.7756 ms at BT16K/V32K and 19.5739 ms at BT32K/V128K.

## Verdict

**REJECTED.**
