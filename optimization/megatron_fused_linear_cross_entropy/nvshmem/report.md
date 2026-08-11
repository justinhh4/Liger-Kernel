# Device NVSHMEM Megatron FLCE Optimization Report

## Result

The final saved-logit backend is faster than NCCL at every tested B200 shape:
**4.9-19.2% at TP4** and **9.6-18.3% at TP8** for BT16K/32K and global
vocabulary 32K/128K. Compared with Variant 18, the new design is another
0.9-6.7% faster. BT512/2048 guardrails improve too.

This is the communication-optimized form of the saved-logit path previously
called "CuTe DSL v3." With its default Torch communication backend, that path
selects the materialized cuBLAS projection plus Triton CE implementation used
by the `liger` baseline here. The store-free CuTe DSL path is a different,
slower algorithm that recomputes projection work in backward.

The experiment remains under `optimization/`. CuTe DSL cannot directly
device-link arbitrary NVSHMEM calls, so production integration would require
the separate CUDA RDC bridge, NVSHMEM lifecycle management, symmetric
allocations, and world-group constraints used here.

## Final Algorithm

Forward:

1. Materialize the local vocabulary projection with cuBLAS.
2. Reduce each row directly into symmetric FP32 max storage with Triton.
3. Use a block NVSHMEM MAX through 8K tokens and parallel `float4` direct-peer
   loads above 8K.
4. Convert logits to saved low-precision exponentials with the fused Triton CE
   kernel.
5. Pack predicted logits and exponent sums into one FP32 device-side SUM.

Backward:

1. Convert saved exponentials to dlogits in-place.
2. Write dX directly into a BF16/FP16 symmetric buffer, using tuned CuTile
   GEMM through BT2048 and cuBLAS above that crossover.
3. Launch dW concurrently on a second CUDA stream.
4. Assign each rank a disjoint dX vector chunk, accumulate 16-byte/eight-value
   peer loads in FP32, and immediately broadcast each rounded result to peer
   destinations.
5. Use TP/rank-specialized kernels for TP2/4/8, 256-thread CTAs, and a
   4,096-block cap selected for overlap with dW.
6. Signal completion on device once all direct peer stores are visible.

The dX algorithm halves TP4 remote traffic from `3N` to `1.5N` relative to a
direct all-reduce and halves element size relative to the original FP32 path.
In-place partitioning removes the second dX symmetric allocation. Fused
owner-computes broadcast removes the standalone allgather kernel; compile-time
TP/rank specialization reduces the TP4 communication phase from roughly
913 us to about 681 us, making it overlap almost completely with dW.

## B200 BF16 Results

Median full forward+backward latency, 30 measurement iterations per sample,
five samples, reversed provider order:

| TP | BT | Global V | NCCL | Device NVSHMEM | Improvement |
|---:|---:|---:|---:|---:|---:|
| 4 | 16,384 | 32,000 | 3.1276 ms | 2.5284 ms | 19.16% |
| 4 | 16,384 | 128,000 | 9.8560 ms | 9.2380 ms | 6.27% |
| 4 | 32,768 | 32,000 | 6.0581 ms | 4.9314 ms | 18.60% |
| 4 | 32,768 | 128,000 | 19.9727 ms | 19.0000 ms | 4.87% |
| 8 | 16,384 | 32,000 | 2.0652 ms | 1.7213 ms | 16.65% |
| 8 | 16,384 | 128,000 | 5.3983 ms | 4.8799 ms | 9.60% |
| 8 | 32,768 | 32,000 | 3.8763 ms | 3.1672 ms | 18.29% |
| 8 | 32,768 | 128,000 | 10.6304 ms | 9.6070 ms | 9.63% |

Data:

- `device_v25_hybrid_realistic_tp4_bf16.csv`
- `device_v25_final_tp4_v128k_bf16.csv`
- `device_v25_final_tp8_bf16.csv`

The strongest gains occur at V32K, where dX communication is exposed. At
V128K, the longer dW GEMM hides more of the collective, so the remaining
full-step opportunity is primarily projection compute rather than NVSHMEM.

## Guardrails

| TP | BT | Global V | NCCL | Device NVSHMEM | Improvement |
|---:|---:|---:|---:|---:|---:|
| 4 | 512 | 32,000 | 0.5285 ms | 0.4344 ms | 17.80% |
| 4 | 2,048 | 32,000 | 0.5576 ms | 0.4667 ms | 16.30% |

BF16 correctness passed at TP2, TP4, and TP8. FP16 correctness passed at TP2
and TP4; the FP16 bridge uses the same templated vector algorithm as BF16.

## Memory

Torch allocator measurements do not include the symmetric NVSHMEM heap. For
BF16/FP16 with aligned H, the final symmetric allocation is approximately
`BT * H * 2 + BT * 24` bytes: one dX buffer plus FP32 MAX and packed-stat
source/destination buffers. At H4096 this is 128.4 MiB for BT16K and 256.8 MiB
for BT32K. Variant 18 removes another dX buffer of 128 MiB or 256 MiB,
respectively.

## Requirements

- NVIDIA GPUs with direct peer access; validated on 8x B200.
- CUDA 13, Torch `2.13.0.1+cu130`, NCCL 2.28.9.
- `nvidia-nvshmem-cu13==3.4.5`.
- World process group with at most 16 ranks.
- `NVSHMEM_DISABLE_NVLS=1` for this installed runtime.

Build:

```bash
optimization/megatron_fused_linear_cross_entropy/nvshmem/build_device_bridge.sh
```

The script enables relocatable device code, runs the NVSHMEM device-link step,
and creates `libliger_nvshmem_device.so`.
