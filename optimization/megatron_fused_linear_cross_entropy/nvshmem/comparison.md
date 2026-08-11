# Device NVSHMEM Variant Comparison

| Variant | Strategy | Realistic-shape outcome | Verdict |
|---|---|---|---|
| v9 | FP32 vectorized dX | Up to 6.5% slower at TP4 | Superseded |
| v10 | BF16/FP16 dX, FP32 accumulation | Converted all TP4 points to ties/wins | Accepted |
| v11 | Parallel large-message MAX | Improved three of four TP4 points | Accepted |
| v12 | Eight-element peer loads | Largest dX speedup | Accepted |
| v13 | 4,096-block grid | Improved all TP4 points | Accepted |
| v14 | 8,192-block grid | Best launch configuration | Accepted |
| v15 | 16,384-block grid | Regressed three points | Rejected |
| v16 | 512 threads per block | Regressed versus 256 threads | Rejected |
| v17 | Sixteen-element peer loads | Regressed communication-sensitive case | Rejected |
| v18 | In-place reduce-scatter/allgather | Same speed, half the dX symmetric heap | Prior winner |
| v19 | Fused reduce-and-broadcast | 0.6% faster; moved allgather work into reduction | Accepted |
| v20 | TP-specialized peer loops | 4.2% faster than v19 at TP4/V32K | Accepted |
| v21 | TP/rank-specialized peer loops | Fused kernel reduced to about 681 us | Accepted |
| v22 | 4,096-CTA overlap balance | Best realistic communication configuration | Accepted |
| v23 | 2,048-CTA overlap balance | Tied/worse than v22 | Rejected |
| v24 | CuTile dX at all shapes | Low-BT win, realistic regression | Rejected |
| v25 | Shape-gated CuTile dX | Low-BT win with v22 throughput | **Winner** |

Variants 1-18 established device-side synchronization, direct peer mappings,
low-precision reduce-scatter/allgather, dW overlap, vectorization, and in-place
dX. Variants 19-25 remove the separate allgather, specialize common TP/rank
topologies, balance communication against dW, and select the local dX GEMM by
shape. Individual results remain in `device_v*_notes.md`.
