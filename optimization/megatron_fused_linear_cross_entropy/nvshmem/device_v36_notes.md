# Device NVSHMEM Variant 36: Four TP8 Accumulators

## Hypothesis

**Target bottleneck:** At TP8/BT16K each thread handles exactly one vector, so
there is no loop-carried vector work. Four peer accumulators reduce each FP32
chain to at most two ranks and may expose more remote-load MLP.

## Changes

- Replace v35's two TP8 accumulators with four in the gated BT16K regime.
- Keep all other v35 behavior unchanged.

## Expected Outcome

Improve BT16K by up to 2%; reject if register pressure erases the ILP gain.

## Actual Results

TP8/BT16K full-step latency regressed to 1.6802 ms versus 1.6697 ms for
v35. The extra registers erase the shorter add chains.

## Verdict

**REJECTED.**
