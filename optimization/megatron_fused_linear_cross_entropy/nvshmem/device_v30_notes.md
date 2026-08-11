# Device NVSHMEM Variant 30: Parallel Readiness Handshake

## Hypothesis

**Target bottleneck:** v25 TP8 uses two readiness kernels around dX
communication. Each currently has one thread serially signal and wait on seven
peers, costing roughly 20-28 us per barrier.

Assigning one thread per peer should parallelize signal puts, quiet operations,
and waits. This targets 25-40 us of full-step latency without changing the
data path.

## Changes

- Restore v25's one-thread-per-vector TP8 communication kernel.
- Fence once, then assign one readiness-kernel thread to each peer.
- Keep the same epoch and signal layout.

## Expected Outcome

Improve TP8 full-step latency by 1-2% and TP4 by up to 0.5%, with identical
numerical behavior.

## Actual Results

TP8 full-step latency improved from v25's 1.7213 to 1.6843 ms at BT16K
and from 3.1672 to 3.1340 ms at BT32K. Profiling showed small forward
readiness kernels fall to 4-6 us and dX readiness kernels to roughly
13-20 us.

## Verdict

**ACCEPTED.**
