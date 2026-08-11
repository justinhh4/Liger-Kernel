# Device NVSHMEM Variant 18: In-Place dX All-Reduce

## Hypothesis

Each reduce-scatter rank writes only its owned vector chunk, while peers read
different chunks from that rank. The reduced chunks can therefore overwrite
the source and be allgathered in-place.

## Changes

- Alias dX destination to source for aligned hidden sizes and at most 16 ranks.
- Retain a separate destination for the generic unaligned fallback.

## Actual Results

BF16 and FP16 correctness passed. Final reversed-order BF16 speedups were
3.5-12.9% at TP4 and 6.5-15.7% at TP8. The alias removes one 128 MiB symmetric
buffer at BT16K/H4096 and one 256 MiB buffer at BT32K/H4096.

## Verdict

**ACCEPTED.** Final design.
