# Device NVSHMEM Variant 39: One-Pass Local CE

## Hypothesis

**Target bottleneck:** v35/v32 forward reads materialized logits once for a
local-max kernel and again for CE. At TP8/BT16K/V32K the max pass costs about
23 us; at TP4 it costs about 44 us.

For local vocabularies up to 32K, one Triton row program can retain logits in
registers long enough to compute local max, `exp(logit - local_max)`, local
sum, and raw target logit in one pass. After global MAX, only the token-sized
sum vector needs rescaling.

## Changes

- Fuse local max, exponentials, local sum, and target extraction.
- Scale local sums after the device MAX using one token-vector kernel.
- Save locally shifted exponentials; apply
  `exp(local_max - global_max)` in CE backward.
- Gate the path to NVSHMEM and local vocabulary at most 32K.
- Preserve the existing fallback for larger vocabularies and non-NVSHMEM use.

## Expected Outcome

Reduce forward by 20-45 us and full-step latency by 1-2.5% without increasing
memory.

## Actual Results

TP8 V32K full-step latency was:

| BT | v35 | v39 | Change |
|---:|---:|---:|---:|
| 16K | 1.6697 ms | 1.7027 ms | 2.0% slower |
| 32K | 3.0444 ms | 3.1400 ms | 3.1% slower |

Forward improved by 10.7 us at BT16K and 25.9 us at BT32K, validating the
one-pass read reduction. Backward regressed by 17.2 us and 121.5 us because
every element had to apply the local-to-global exponential scale.

## Verdict

**REJECTED.** Retain the one-pass forward idea, but convert the token-sized
global denominator into the local-shift basis before saving it so the existing
fast CE backward can be reused.
