# NVSHMEM v18 Bottleneck Profile

## Scope

- GPU: 4x NVIDIA B200
- Shape: TP4, BT16,384, H4096, global V32,000, BF16
- Tool: Nsight Systems 2026.1.3
- Target: saved-logit device-NVSHMEM forward and backward

## Representative v18 Backward Timeline

| Stage | Duration |
|---|---:|
| Triton CE backward | 126 us |
| cuBLAS dX | 539 us |
| cuBLAS dW, overlap stream | 609 us |
| NVSHMEM reduce-scatter | 744 us |
| Device readiness barrier | 7 us |
| NVSHMEM allgather | 162 us |

dW starts with the dX collective but finishes about 135 us before
reduce-scatter. Allgather then adds another 162 us to the critical path. The
communication sequence is therefore about 913 us and is the dominant
post-dX stage.

## Diagnosis

The collective is not primarily bandwidth-volume limited: reduce-scatter and
allgather already move the minimum `1.5N` TP4 remote traffic. Its dominant
cost is instruction and dependency overhead from runtime peer loops plus the
separate allgather phase.

Recommended strategy order:

1. Fuse owner-computed reduced vectors with direct peer broadcasts.
2. Specialize common TP counts so peer loops fully unroll.
3. Specialize rank to remove self-peer branches and constant-fold ownership.
4. Retune CTA count for concurrent dW rather than isolated collective latency.
5. Revisit local dX GEMM only after communication is hidden.

## Post-Optimization Observation

Variant 21's TP4/rank-specialized fused kernel takes about 681 us and ends
within a few microseconds of concurrent dW. Variant 22's 4,096-CTA cap further
improves end-to-end overlap. At that point communication is effectively hidden
and dX/dW compute becomes the next bottleneck.
