# Device NVSHMEM Variant 38: L2-Only Peer Stores

## Hypothesis

**Target bottleneck:** Variant 37 shows that `.cg` broadcast stores matter
more than expected. Ordinary peer reads plus `.cg` stores may retain the gain
without forcing source traffic through L2-only loads.

## Changes

- Use ordinary 16-byte peer reads.
- Restore `__stcg` for throughput peer broadcasts.
- Keep v35's accumulator and shape gates.

## Expected Outcome

Match or beat v35; otherwise retain combined `.cg` loads/stores.

## Actual Results

TP8 full-step latency was 1.6755 ms at BT16K and regressed to 3.1356 ms
at BT32K. Combined `.cg` reads and stores remain faster.

## Verdict

**REJECTED.** Restore v35's combined `.cg` path.
