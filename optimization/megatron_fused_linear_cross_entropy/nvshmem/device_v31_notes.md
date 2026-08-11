# Device NVSHMEM Variant 31: L2-Only Peer Vectors

## Hypothesis

**Target bottleneck:** After v30, TP8's fused data kernel still takes about
533 us. Each source and destination vector is touched once, so caching direct
peer traffic in L1 can evict useful local state without reuse.

Explicit 128-bit `.cg` loads/stores should bypass L1, retain coalesced
transactions, and use L2/NVLink more efficiently.

## Changes

- Bit-cast each 16-byte low-precision vector to `uint4`.
- Use CUDA `__ldcg` and `__stcg` for direct peer mappings.
- Preserve the NVSHMEM scalar fallback and all v30 synchronization.

## Expected Outcome

Reduce TP8 data-kernel latency by 3-8% and full-step latency by 1-3%.

## Actual Results

TP8 full-step latency improved from v30's 1.6843 to 1.6738 ms at BT16K
and from 3.1340 to 3.0564 ms at BT32K. BT512 regressed from 0.4438 to
0.5843 ms, showing a clear cache-policy crossover.

## Verdict

**MIXED.** Accept `.cg` only for throughput shapes.
