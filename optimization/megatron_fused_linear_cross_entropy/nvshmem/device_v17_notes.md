# Device NVSHMEM Variant 17: Sixteen-Element dX Vectors

## Hypothesis

Doubling each peer load to 32 bytes may further reduce loop overhead.

## Changes

- Tested 32-byte/sixteen-element vectors with the 8,192-block grid.

## Actual Results

In a reversed-order TP4 run, BT16K/V32K regressed from the vector8 result to
2.7689 ms. The larger-vocabulary result was communication-hidden and did not
justify the communication-sensitive regression.

## Verdict

**REJECTED.** Restored 16-byte/eight-element vectors.
