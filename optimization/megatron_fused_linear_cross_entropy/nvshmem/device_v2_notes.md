# Device NVSHMEM Variant 2: Device Team Collectives

## Hypothesis

Variant 1 proves that device-side synchronization removes the host-fence
penalty, but its direct peer reads use an unoptimized all-gather-like traffic
pattern. NVSHMEM's device team reductions should use a scalable collective
algorithm and preserve the small-message launch advantage. Parallelizing
forward reductions above 1,024 values should remove the BT=2,048 bottleneck.

## Changes

- Replace direct peer-load SUM/MAX with NVSHMEM block-scoped device team
  collectives where supported.
- Keep direct peer loads as a fallback for unsupported message regimes.
- Preserve symmetric GEMM outputs, the dW/dX overlap schedule, and the
  host-asynchronous execution model.

## Expected Outcome

Retain at least 5% full-step improvement at BT=512 while bringing both BT=2,048
cases within 5% of NCCL or better.

## Actual Results

Correctness passed with `NVSHMEM_DISABLE_NVLS=1`; without it, the runtime
selects an unsupported NVLS algorithm and asserts. TP4 forward latency improved
to 0.192-0.556 ms, but block-scoped dX reduction took 14.9 ms for 8 MiB and
61-62 ms for 32 MiB.

## Verdict

**REJECTED.** Device team collectives are useful only for the small forward
statistics in this runtime. Their large-message algorithm is over 50x slower
than NCCL and cannot be used for dX.
