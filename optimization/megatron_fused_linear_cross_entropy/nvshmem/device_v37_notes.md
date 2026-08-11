# Device NVSHMEM Variant 37: L2-Only Peer Reads

## Hypothesis

**Target bottleneck:** v31 bundled `.cg` loads and stores. Remote source
vectors are read once, while broadcast stores do not benefit from a load cache
policy and may pay extra L2 handling.

Using `.cg` only for peer reads may retain the throughput gain with cheaper
normal direct stores.

## Changes

- Restore v35's dual accumulator.
- Keep `__ldcg` for throughput peer reads.
- Use ordinary 16-byte stores for peer broadcasts.

## Expected Outcome

Improve TP8 latency by up to 2% relative to v35/v32.

## Actual Results

TP8 full-step latency was 1.6798 ms at BT16K and regressed to 3.0896 ms
at BT32K. Removing `.cg` stores loses most of v32's large-message benefit.

## Verdict

**REJECTED.**
