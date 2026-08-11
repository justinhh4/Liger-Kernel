# Device NVSHMEM Variant 33: Streaming Peer Vectors

## Hypothesis

**Target bottleneck:** Variant 32 confirms that bypassing L1 helps one-use
throughput traffic. CUDA's `.cs` policy also marks lines as streaming/evict
first, potentially reducing L2 pollution versus `.cg`.

## Changes

- Replace throughput-shape `__ldcg`/`__stcg` with
  `__ldcs`/`__stcs`.
- Keep the v32 volume gate and all other behavior unchanged.

## Expected Outcome

Improve realistic TP8 latency by up to 2%; reject if it loses to `.cg`.

## Actual Results

TP8 full-step latency regressed to 1.7276 ms at BT16K and was effectively
tied at 3.0584 ms at BT32K. Evict-first behavior hurts the active peer stream.

## Verdict

**REJECTED.** Restore `.cg`.
