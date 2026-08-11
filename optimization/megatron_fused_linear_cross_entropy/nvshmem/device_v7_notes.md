# Device NVSHMEM Variant 7: Wider Collective Grid

## Hypothesis

Variant 6's 1,024-block cap forces two reduce-scatter rounds and eight
allgather rounds for a 32 MiB dX buffer. Doubling the grid should reduce loop
serialization while retaining enough work per block to amortize launch cost.

## Changes

- Increase reduce-scatter and allgather caps from 1,024 to 2,048 blocks.
- Keep 256 threads per block and `float4` peer accesses.
- Keep communication entirely in device-side NVSHMEM for every shape.

## Expected Outcome

Improve BT=2,048 backward by at least 3% without regressing BT=512, making all
four TP4 cases faster than NCCL.

## Actual Results

TP4 correctness passed. At BT=2,048, NVSHMEM backward became 1.8% and 2.2%
faster than NCCL. Full-step latency was 1.5% slower at global vocab 32,000 and
0.3% faster at 128,256. BT=512 retained 17.0% and 5.4% wins.

## Verdict

**ACCEPTED.** This is the best large-message kernel so far and all full-step
results satisfy the 5% guardrail. The remaining optimization target is the
BT=2,048 forward device collectives.
