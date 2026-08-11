# Device NVSHMEM Variant 40: Local-Basis Saved Denominator

## Hypothesis

**Target bottleneck:** v39's one-pass forward is faster, but its backward
rescales every saved exponential and loses 17-122 us.

After loss has consumed the global denominator, convert that token-sized
denominator to the local-max basis:

`denom_local = denom_global * exp(global_max - local_max)`.

Then the existing CE backward computes exactly the same global softmax from
the locally shifted exponentials without per-element scale arithmetic.

## Changes

- Retain v39's one-pass local-max/CE forward.
- Add one token-sized post-loss denominator conversion.
- Reuse the original optimized CE backward unchanged.
- Remove v39's custom per-element backward.

## Expected Outcome

Retain most of v39's 11-26 us forward improvement while restoring v35
backward performance.

## Actual Results

TP8 V32K full-step latency was:

| BT | v35 | v40 | Change |
|---:|---:|---:|---:|
| 16K | 1.6697 ms | 1.7170 ms | 2.8% slower |
| 32K | 3.0444 ms | 3.1260 ms | 2.7% slower |

The token-sized conversion did not recover the proven path's end-to-end
latency. Although forward remained 8-26 us faster than v35, the full backward
was 8-86 us slower and the added stream work increased total latency further.

## Verdict

**REJECTED.** Restore v35.
