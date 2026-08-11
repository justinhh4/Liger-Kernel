# Device NVSHMEM Variant 35: Work-Per-Thread Accumulator Gate

## Hypothesis

**Target bottleneck:** Variant 34's dual accumulator wins at BT16K but loses
at BT32K. With TP8 and 4,096x256 threads, BT16K assigns one vector per thread
while BT32K assigns two. Compile-time gating can avoid carrying dual
accumulator registers into the looping regime.

## Changes

- Add a compile-time dual-accumulator kernel parameter.
- Enable it only for TP8 throughput counts up to BT16K/H4096 volume.
- Use v32's single accumulator at low BT and above that volume.

## Expected Outcome

Retain v34 at BT16K and v32 at BT32K.

## Actual Results

TP8 full-step latency was 1.6697 ms at BT16K and 3.0444 ms at BT32K.
The gate retains the dual-chain BT16K improvement while restoring the
single-chain BT32K result.

## Verdict

**ACCEPTED.**
