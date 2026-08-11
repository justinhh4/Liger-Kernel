# Device NVSHMEM Variant 28: TP8 Four-Lane Cooperation

## Hypothesis

**Target bottleneck:** Variant 27 proves that eight lanes per vector
over-parallelizes the work. Four lanes preserve twice as much vector-level
parallelism: each lane handles two peer loads, a four-lane shuffle reduction,
and two peer stores.

This reduces peer dependency depth from seven to two without Variant 27's
eightfold reduction in independent vectors.

## Changes

- Generalize the TP8 cooperative kernel to a compile-time group width.
- Use four lanes per vector, with two peers assigned to each lane.
- Keep TP2/TP4 and all other v25 behavior unchanged.

## Expected Outcome

Improve TP8 full-step latency by 2-6% relative to v25 while preserving TP4.

## Actual Results

Correctness passed, but TP8 full-step latency regressed to 2.2096 ms at
BT16K and 4.2681 ms at BT32K. Four-lane cooperation was even slower than
eight-lane cooperation because each group looped over more vectors while
still paying shuffle and duplicate result-packing overhead.

## Verdict

**REJECTED.**
