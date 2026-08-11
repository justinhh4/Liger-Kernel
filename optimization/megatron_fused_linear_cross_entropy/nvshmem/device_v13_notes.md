# Device NVSHMEM Variant 13: 4,096-Block Grid

## Hypothesis

The 2,048-block cap leaves too much serial work per thread at BT32K.

## Changes

- Increased the low-precision reduce-scatter/allgather cap to 4,096 blocks.

## Actual Results

All four realistic TP4 cases improved, including BT32K/V128K from 19.5299 to
19.3368 ms.

## Verdict

**ACCEPTED.**
