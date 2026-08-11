# Device NVSHMEM Variant 9: Direct Symmetric Local Max

## Hypothesis

The NVSHMEM forward path currently launches PyTorch `amax`, casts its result to
FP32, and copies it into a symmetric source buffer before communication.
Reducing each row directly into that symmetric FP32 buffer should remove one
kernel launch and one read/write pass over the token vector, closing Variant
7's remaining 1.5% worst-case full-step gap.

## Changes

- Add a Triton row-max reduction over materialized BF16/FP16 logits.
- Write FP32 maxima directly to the NVSHMEM symmetric source tensor.
- Use 8 warps through 8K local vocabulary and 16 warps above it.
- Keep the 256-thread device MAX/SUM collectives and Variant 7 dX algorithm.

## Expected Outcome

Improve forward latency without changing numerical behavior, making the pure
NVSHMEM path at least as fast as NCCL at all four TP4 shapes.

## Actual Results

TP4 BF16 correctness passed. A 20-iteration repeat with reversed provider order
measured:

| BT | Global V | NCCL full (ms) | Device NVSHMEM full (ms) | Improvement |
|---:|---:|---:|---:|---:|
| 512 | 32,000 | 0.4306 | 0.4229 | 1.8% |
| 512 | 128,256 | 0.4897 | 0.4701 | 4.0% |
| 2,048 | 32,000 | 0.5574 | 0.5355 | 3.9% |
| 2,048 | 128,256 | 1.3707 | 1.3091 | 4.5% |

Forward improved by 3.5-28.1%. Backward was 2.4% slower only at the smallest
shape and 0.2-3.2% faster elsewhere.

## Verdict

**ACCEPTED.** The pure device-side NVSHMEM path is faster in every tested TP4
full-step case and satisfies all correctness and performance guardrails.
