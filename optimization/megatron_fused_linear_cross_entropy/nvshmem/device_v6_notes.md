# Device NVSHMEM Variant 6: Reduce-Scatter and Allgather

## Hypothesis

Direct all-peer reduction reads `(TP - 1) * N` remote values per rank. Assigning
one output chunk to each rank, reducing only that chunk, then gathering the
reduced chunks cuts TP4 remote traffic from `3N` to `1.5N`. The extra device
readiness kernel should be amortized for 8-32 MiB dX buffers.

## Changes

- Partition vectorized dX evenly across tensor-parallel ranks.
- Reduce each rank's owned chunk from all symmetric sources.
- Signal completion entirely on device.
- Gather completed chunks directly from each owner.
- Retain the prior direct algorithm as a non-vectorizable fallback.

## Expected Outcome

Improve BT=2,048 backward by at least 8% over Variant 4 while preserving the
BT=512 gains, yielding a speedup over NCCL for all four TP4 cases.

## Actual Results

Pending.

## Verdict

Pending.
