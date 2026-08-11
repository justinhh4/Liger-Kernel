# Device NVSHMEM Variant 29: TP8 Two-Lane Cooperation

## Hypothesis

**Target bottleneck:** Four- and eight-lane cooperation lose too much
vector-level parallelism. Two lanes are the least disruptive cooperative
form: each lane processes four peers, then one shuffle exchange combines the
partial sums and each lane writes four destinations.

This halves peer dependency depth while retaining half of v25's independent
vector threads.

## Changes

- Set the TP8 cooperative group width to two.
- Keep all other v25 settings unchanged.

## Expected Outcome

Determine whether any cooperative peer decomposition can beat the v25
one-thread-per-vector kernel.

## Actual Results

Correctness passed, but TP8 full-step latency regressed to 2.6670 ms at
BT16K and 5.1969 ms at BT32K. All cooperative widths were slower than v25;
the one-thread-per-vector kernel provides the best vector-level parallelism.

## Verdict

**REJECTED.**
