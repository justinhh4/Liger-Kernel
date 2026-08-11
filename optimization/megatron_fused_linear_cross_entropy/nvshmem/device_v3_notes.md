# Device NVSHMEM Variant 3: Size-Aware Hybrid

## Hypothesis

Variant 2's device team collective removes the BT=2,048 single-block forward
bottleneck, while Variant 1's parallel direct peer loads are dramatically
faster for multi-megabyte dX. Selecting by message size should combine both
strengths and isolate dX as the only remaining large-token bottleneck.

## Changes

- Use block-scoped device team SUM/MAX for messages up to 8,192 FP32 values.
- Use the readiness-handshake plus parallel direct peer loads above that
  threshold.
- Continue overlapping large dX reduction with dW.

## Expected Outcome

Match Variant 1's 8-17% wins at BT=512, improve BT=2,048 forward by at least
20%, and reduce its full-step regression to under 15%.

## Actual Results

TP4 correctness passed. Full-step latency improved by 16.5% and 10.4% at
BT=512. At BT=2,048, forward was within 3.9% of NCCL, but direct-read dX made
full-step latency 21.9% and 6.9% slower for global vocab 32,000 and 128,256.

## Verdict

**REJECTED.** The two BT=2,048 cases violate the 5% speed guardrail. The
remaining target is exclusively the large dX direct-read kernel.
