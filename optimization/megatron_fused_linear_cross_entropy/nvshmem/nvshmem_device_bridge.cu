#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <nvshmem.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kMaxPes = 16;
constexpr int kMaxCollectiveBlocks = 2048;
constexpr int kMaxLowPrecisionBlocks = 4096;
constexpr int kLowPrecisionThreads = 256;
constexpr int kSmallCollectiveThreads = 256;

struct PeerSources {
    const float* pointers[kMaxPes];
    int npes;
    int me;
};

template <typename T>
struct TypedPeerSources {
    const T* pointers[kMaxPes];
    int npes;
    int me;
};

template <typename T>
struct TypedPeerDestinations {
    T* pointers[kMaxPes];
};

template <typename T>
struct LowPrecisionOps;

template <typename T>
struct alignas(16) LowPrecisionVector {
    typename LowPrecisionOps<T>::Packed values[4];
};

template <typename T>
union LowPrecisionVectorBits {
    LowPrecisionVector<T> vector;
    uint4 bits;
};

template <>
struct LowPrecisionOps<__nv_bfloat16> {
    using Packed = __nv_bfloat162;

    __device__ static float2 to_float2(Packed value) {
        return __bfloat1622float2(value);
    }

    __device__ static Packed from_float2(float2 value) {
        return __floats2bfloat162_rn(value.x, value.y);
    }

    __device__ static __nv_bfloat16 from_bits(unsigned short value) {
        return __ushort_as_bfloat16(value);
    }
};

template <>
struct LowPrecisionOps<__half> {
    using Packed = __half2;

    __device__ static float2 to_float2(Packed value) {
        return __half22float2(value);
    }

    __device__ static Packed from_float2(float2 value) {
        return __floats2half2_rn(value.x, value.y);
    }

    __device__ static __half from_bits(unsigned short value) {
        return __ushort_as_half(value);
    }
};

__device__ void signal_and_wait(std::uint64_t* signals, std::uint64_t epoch) {
    const int me = nvshmem_my_pe();
    const int npes = nvshmem_n_pes();

    if (threadIdx.x == 0) {
        __threadfence_system();
    }
    __syncthreads();
    const int pe = threadIdx.x;
    if (pe < npes && pe != me) {
        nvshmem_uint64_p(signals + me, epoch, pe);
        nvshmem_quiet();
        nvshmem_uint64_wait_until(signals + pe, NVSHMEM_CMP_GE, epoch);
    }
    __syncthreads();
}

__global__ void readiness_kernel(std::uint64_t* signals, std::uint64_t epoch) {
    signal_and_wait(signals, epoch);
}

__global__ void parallel_sum_kernel(
    float* destination,
    const float* source,
    std::size_t count,
    PeerSources peers
) {
    const std::size_t first_index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = gridDim.x * blockDim.x;
    const std::size_t vector_count = count / 4;
    for (std::size_t vector_index = first_index; vector_index < vector_count; vector_index += stride) {
        float4 value = reinterpret_cast<const float4*>(source)[vector_index];
        for (int pe = 0; pe < peers.npes; ++pe) {
            if (pe == peers.me) {
                continue;
            }
            const float* peer_source = peers.pointers[pe];
            if (peer_source != nullptr) {
                const float4 peer_value = reinterpret_cast<const float4*>(peer_source)[vector_index];
                value.x += peer_value.x;
                value.y += peer_value.y;
                value.z += peer_value.z;
                value.w += peer_value.w;
            } else {
                const std::size_t base = vector_index * 4;
                value.x += nvshmem_float_g(source + base, pe);
                value.y += nvshmem_float_g(source + base + 1, pe);
                value.z += nvshmem_float_g(source + base + 2, pe);
                value.w += nvshmem_float_g(source + base + 3, pe);
            }
        }
        reinterpret_cast<float4*>(destination)[vector_index] = value;
    }

    for (std::size_t tail_index = vector_count * 4 + first_index; tail_index < count; tail_index += stride) {
        float value = source[tail_index];
        for (int pe = 0; pe < peers.npes; ++pe) {
            if (pe == peers.me) {
                continue;
            }
            const float* peer_source = peers.pointers[pe];
            value += peer_source == nullptr
                ? nvshmem_float_g(source + tail_index, pe)
                : peer_source[tail_index];
        }
        destination[tail_index] = value;
    }
}

__global__ void parallel_max_kernel(
    float* destination,
    const float* source,
    std::size_t count,
    PeerSources peers
) {
    const std::size_t first_index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = gridDim.x * blockDim.x;
    const std::size_t vector_count = count / 4;
    for (std::size_t vector_index = first_index; vector_index < vector_count; vector_index += stride) {
        float4 value = reinterpret_cast<const float4*>(source)[vector_index];
        for (int pe = 0; pe < peers.npes; ++pe) {
            if (pe == peers.me) {
                continue;
            }
            const float* peer_source = peers.pointers[pe];
            float4 peer_value;
            if (peer_source != nullptr) {
                peer_value = reinterpret_cast<const float4*>(peer_source)[vector_index];
            } else {
                const std::size_t base = vector_index * 4;
                peer_value.x = nvshmem_float_g(source + base, pe);
                peer_value.y = nvshmem_float_g(source + base + 1, pe);
                peer_value.z = nvshmem_float_g(source + base + 2, pe);
                peer_value.w = nvshmem_float_g(source + base + 3, pe);
            }
            value.x = fmaxf(value.x, peer_value.x);
            value.y = fmaxf(value.y, peer_value.y);
            value.z = fmaxf(value.z, peer_value.z);
            value.w = fmaxf(value.w, peer_value.w);
        }
        reinterpret_cast<float4*>(destination)[vector_index] = value;
    }

    for (std::size_t tail_index = vector_count * 4 + first_index; tail_index < count; tail_index += stride) {
        float value = source[tail_index];
        for (int pe = 0; pe < peers.npes; ++pe) {
            if (pe == peers.me) {
                continue;
            }
            const float* peer_source = peers.pointers[pe];
            const float peer_value = peer_source == nullptr
                ? nvshmem_float_g(source + tail_index, pe)
                : peer_source[tail_index];
            value = fmaxf(value, peer_value);
        }
        destination[tail_index] = value;
    }
}

__global__ void reduce_scatter_kernel(
    float* destination,
    const float* source,
    std::size_t count,
    PeerSources peers
) {
    const std::size_t vector_count = count / 4;
    const std::size_t chunk_size = (vector_count + peers.npes - 1) / peers.npes;
    const std::size_t chunk_begin = peers.me * chunk_size;
    const std::size_t chunk_end = min(chunk_begin + chunk_size, vector_count);
    const std::size_t first_index = chunk_begin + blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = gridDim.x * blockDim.x;

    for (std::size_t vector_index = first_index; vector_index < chunk_end; vector_index += stride) {
        float4 value = reinterpret_cast<const float4*>(source)[vector_index];
        for (int pe = 0; pe < peers.npes; ++pe) {
            if (pe == peers.me) {
                continue;
            }
            const float* peer_source = peers.pointers[pe];
            if (peer_source != nullptr) {
                const float4 peer_value = reinterpret_cast<const float4*>(peer_source)[vector_index];
                value.x += peer_value.x;
                value.y += peer_value.y;
                value.z += peer_value.z;
                value.w += peer_value.w;
            } else {
                const std::size_t base = vector_index * 4;
                value.x += nvshmem_float_g(source + base, pe);
                value.y += nvshmem_float_g(source + base + 1, pe);
                value.z += nvshmem_float_g(source + base + 2, pe);
                value.w += nvshmem_float_g(source + base + 3, pe);
            }
        }
        reinterpret_cast<float4*>(destination)[vector_index] = value;
    }
}

__global__ void allgather_kernel(
    float* destination,
    std::size_t count,
    PeerSources destination_peers
) {
    const std::size_t vector_count = count / 4;
    const std::size_t chunk_size = (vector_count + destination_peers.npes - 1) / destination_peers.npes;
    const std::size_t first_index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = gridDim.x * blockDim.x;

    for (std::size_t vector_index = first_index; vector_index < vector_count; vector_index += stride) {
        const int owner = min(
            static_cast<int>(vector_index / chunk_size),
            destination_peers.npes - 1
        );
        if (owner == destination_peers.me) {
            continue;
        }
        const float* peer_destination = destination_peers.pointers[owner];
        float4 value;
        if (peer_destination != nullptr) {
            value = reinterpret_cast<const float4*>(peer_destination)[vector_index];
        } else {
            const std::size_t base = vector_index * 4;
            value.x = nvshmem_float_g(destination + base, owner);
            value.y = nvshmem_float_g(destination + base + 1, owner);
            value.z = nvshmem_float_g(destination + base + 2, owner);
            value.w = nvshmem_float_g(destination + base + 3, owner);
        }
        reinterpret_cast<float4*>(destination)[vector_index] = value;
    }
}

template <typename T>
__device__ typename LowPrecisionOps<T>::Packed load_peer_pair(
    const T* source,
    const T* peer_source,
    std::size_t pair_index,
    int pe
);

template <>
__device__ __nv_bfloat162 load_peer_pair(
    const __nv_bfloat16* source,
    const __nv_bfloat16* peer_source,
    std::size_t pair_index,
    int pe
) {
    if (peer_source != nullptr) {
        return reinterpret_cast<const __nv_bfloat162*>(peer_source)[pair_index];
    }
    const std::size_t base = pair_index * 2;
    return __halves2bfloat162(
        __ushort_as_bfloat16(
            nvshmem_ushort_g(reinterpret_cast<const unsigned short*>(source) + base, pe)
        ),
        __ushort_as_bfloat16(
            nvshmem_ushort_g(reinterpret_cast<const unsigned short*>(source) + base + 1, pe)
        )
    );
}

template <>
__device__ __half2 load_peer_pair(
    const __half* source,
    const __half* peer_source,
    std::size_t pair_index,
    int pe
) {
    if (peer_source != nullptr) {
        return reinterpret_cast<const __half2*>(peer_source)[pair_index];
    }
    const std::size_t base = pair_index * 2;
    return __halves2half2(
        __ushort_as_half(
            nvshmem_ushort_g(reinterpret_cast<const unsigned short*>(source) + base, pe)
        ),
        __ushort_as_half(
            nvshmem_ushort_g(reinterpret_cast<const unsigned short*>(source) + base + 1, pe)
        )
    );
}

template <typename T>
__device__ T load_peer_scalar(const T* source, const T* peer_source, std::size_t index, int pe) {
    if (peer_source != nullptr) {
        return peer_source[index];
    }
    return LowPrecisionOps<T>::from_bits(
        nvshmem_ushort_g(reinterpret_cast<const unsigned short*>(source) + index, pe)
    );
}

template <typename T>
__device__ LowPrecisionVector<T> load_peer_vector(
    const T* source,
    const T* peer_source,
    std::size_t vector_index,
    int pe
) {
    if (peer_source != nullptr) {
        LowPrecisionVectorBits<T> value;
        value.bits = __ldcg(
            reinterpret_cast<const uint4*>(peer_source) + vector_index
        );
        return value.vector;
    }
    LowPrecisionVector<T> value;
    const std::size_t first_pair = vector_index * 4;
    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        value.values[pair] = load_peer_pair(
            source,
            peer_source,
            first_pair + pair,
            pe
        );
    }
    return value;
}

template <typename T>
__device__ LowPrecisionVector<T> load_peer_vector_l2(
    const T* source,
    const T* peer_source,
    std::size_t vector_index,
    int pe
) {
    if (peer_source != nullptr) {
        return reinterpret_cast<const LowPrecisionVector<T>*>(peer_source)[vector_index];
    }
    LowPrecisionVector<T> value;
    const std::size_t first_pair = vector_index * 4;
    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        value.values[pair] = load_peer_pair(
            source,
            peer_source,
            first_pair + pair,
            pe
        );
    }
    return value;
}

template <typename T>
__device__ void store_peer_vector(
    T* destination,
    T* peer_destination,
    std::size_t vector_index,
    int pe,
    LowPrecisionVector<T> value
) {
    if (peer_destination != nullptr) {
        LowPrecisionVectorBits<T> stored;
        stored.vector = value;
        __stcg(
            reinterpret_cast<uint4*>(peer_destination) + vector_index,
            stored.bits
        );
        return;
    }
    const std::size_t first_element = vector_index * 8;
    const auto* elements = reinterpret_cast<const unsigned short*>(&value);
    #pragma unroll
    for (int element = 0; element < 8; ++element) {
        nvshmem_ushort_p(
            reinterpret_cast<unsigned short*>(destination) + first_element + element,
            elements[element],
            pe
        );
    }
}

template <typename T>
__device__ void store_peer_vector_l2(
    T* destination,
    T* peer_destination,
    std::size_t vector_index,
    int pe,
    LowPrecisionVector<T> value
) {
    if (peer_destination != nullptr) {
        reinterpret_cast<LowPrecisionVector<T>*>(peer_destination)[vector_index] = value;
        return;
    }
    const std::size_t first_element = vector_index * 8;
    const auto* elements = reinterpret_cast<const unsigned short*>(&value);
    #pragma unroll
    for (int element = 0; element < 8; ++element) {
        nvshmem_ushort_p(
            reinterpret_cast<unsigned short*>(destination) + first_element + element,
            elements[element],
            pe
        );
    }
}

template <typename T>
__global__ void parallel_low_precision_sum_kernel(
    T* destination,
    const T* source,
    std::size_t count,
    TypedPeerSources<T> peers
) {
    using Ops = LowPrecisionOps<T>;
    using Packed = typename Ops::Packed;
    const std::size_t first_index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = gridDim.x * blockDim.x;
    const std::size_t pair_count = count / 2;

    for (std::size_t pair_index = first_index; pair_index < pair_count; pair_index += stride) {
        float2 value = Ops::to_float2(
            reinterpret_cast<const Packed*>(source)[pair_index]
        );
        for (int pe = 0; pe < peers.npes; ++pe) {
            if (pe == peers.me) {
                continue;
            }
            const float2 peer_value = Ops::to_float2(
                load_peer_pair(source, peers.pointers[pe], pair_index, pe)
            );
            value.x += peer_value.x;
            value.y += peer_value.y;
        }
        reinterpret_cast<Packed*>(destination)[pair_index] = Ops::from_float2(value);
    }

    if ((count & 1) != 0 && first_index == 0) {
        float value = static_cast<float>(source[count - 1]);
        for (int pe = 0; pe < peers.npes; ++pe) {
            if (pe != peers.me) {
                value += static_cast<float>(
                    load_peer_scalar(source, peers.pointers[pe], count - 1, pe)
                );
            }
        }
        destination[count - 1] = static_cast<T>(value);
    }
}

template <typename T>
__global__ void low_precision_reduce_scatter_kernel(
    T* destination,
    const T* source,
    std::size_t count,
    TypedPeerSources<T> peers
) {
    using Ops = LowPrecisionOps<T>;
    using Packed = typename Ops::Packed;
    const std::size_t pair_count = count / 2;
    const std::size_t chunk_size = (pair_count + peers.npes - 1) / peers.npes;
    const std::size_t chunk_begin = peers.me * chunk_size;
    const std::size_t chunk_end = min(chunk_begin + chunk_size, pair_count);
    const std::size_t first_index = chunk_begin + blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = gridDim.x * blockDim.x;

    for (std::size_t pair_index = first_index; pair_index < chunk_end; pair_index += stride) {
        float2 value = Ops::to_float2(
            reinterpret_cast<const Packed*>(source)[pair_index]
        );
        for (int pe = 0; pe < peers.npes; ++pe) {
            if (pe == peers.me) {
                continue;
            }
            const float2 peer_value = Ops::to_float2(
                load_peer_pair(source, peers.pointers[pe], pair_index, pe)
            );
            value.x += peer_value.x;
            value.y += peer_value.y;
        }
        reinterpret_cast<Packed*>(destination)[pair_index] = Ops::from_float2(value);
    }
}

template <typename T, int Npes, int Me, bool L2Only, bool DualAccum>
__global__ void vectorized_low_precision_reduce_broadcast_kernel(
    T* destination,
    const T* source,
    std::size_t count,
    TypedPeerSources<T> peers,
    TypedPeerDestinations<T> destinations
) {
    using Ops = LowPrecisionOps<T>;
    const int npes = Npes == 0 ? peers.npes : Npes;
    const int me = Me < 0 ? peers.me : Me;
    const std::size_t vector_count = count / 8;
    const std::size_t chunk_size = (vector_count + npes - 1) / npes;
    const std::size_t chunk_begin = me * chunk_size;
    const std::size_t chunk_end = min(chunk_begin + chunk_size, vector_count);
    const std::size_t first_index = chunk_begin + blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = gridDim.x * blockDim.x;

    for (std::size_t vector_index = first_index; vector_index < chunk_end; vector_index += stride) {
        LowPrecisionVector<T> packed = reinterpret_cast<const LowPrecisionVector<T>*>(source)[vector_index];
        float2 values[4];
        #pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            values[pair] = Ops::to_float2(packed.values[pair]);
        }
        if constexpr (Npes == 8 && DualAccum) {
            float2 alternate[4] = {};
            #pragma unroll
            for (int pe = 0; pe < 8; pe += 2) {
                if (pe != me) {
                    LowPrecisionVector<T> peer_value;
                    if constexpr (L2Only) {
                        peer_value = load_peer_vector_l2(
                            source,
                            peers.pointers[pe],
                            vector_index,
                            pe
                        );
                    } else {
                        peer_value = load_peer_vector(
                            source,
                            peers.pointers[pe],
                            vector_index,
                            pe
                        );
                    }
                    #pragma unroll
                    for (int pair = 0; pair < 4; ++pair) {
                        const float2 converted = Ops::to_float2(peer_value.values[pair]);
                        values[pair].x += converted.x;
                        values[pair].y += converted.y;
                    }
                }
            }
            #pragma unroll
            for (int pe = 1; pe < 8; pe += 2) {
                if (pe != me) {
                    LowPrecisionVector<T> peer_value;
                    if constexpr (L2Only) {
                        peer_value = load_peer_vector_l2(
                            source,
                            peers.pointers[pe],
                            vector_index,
                            pe
                        );
                    } else {
                        peer_value = load_peer_vector(
                            source,
                            peers.pointers[pe],
                            vector_index,
                            pe
                        );
                    }
                    #pragma unroll
                    for (int pair = 0; pair < 4; ++pair) {
                        const float2 converted = Ops::to_float2(peer_value.values[pair]);
                        alternate[pair].x += converted.x;
                        alternate[pair].y += converted.y;
                    }
                }
            }
            #pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                values[pair].x += alternate[pair].x;
                values[pair].y += alternate[pair].y;
            }
        } else {
            #pragma unroll
            for (int pe = 0; pe < npes; ++pe) {
                if (pe == me) {
                    continue;
                }
                LowPrecisionVector<T> peer_value;
                if constexpr (L2Only) {
                    peer_value = load_peer_vector_l2(
                        source,
                        peers.pointers[pe],
                        vector_index,
                        pe
                    );
                } else {
                    peer_value = load_peer_vector(
                        source,
                        peers.pointers[pe],
                        vector_index,
                        pe
                    );
                }
                #pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    const float2 converted = Ops::to_float2(peer_value.values[pair]);
                    values[pair].x += converted.x;
                    values[pair].y += converted.y;
                }
            }
        }
        #pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            packed.values[pair] = Ops::from_float2(values[pair]);
        }
        reinterpret_cast<LowPrecisionVector<T>*>(destination)[vector_index] = packed;
        #pragma unroll
        for (int pe = 0; pe < npes; ++pe) {
            if (pe != me) {
                if constexpr (L2Only) {
                    store_peer_vector_l2(
                        destination,
                        destinations.pointers[pe],
                        vector_index,
                        pe,
                        packed
                    );
                } else {
                    store_peer_vector(
                        destination,
                        destinations.pointers[pe],
                        vector_index,
                        pe,
                        packed
                    );
                }
            }
        }
    }
}

template <typename T, int Me, int GroupWidth>
__global__ void tp8_grouped_low_precision_reduce_broadcast_kernel(
    T* destination,
    const T* source,
    std::size_t count,
    TypedPeerSources<T> peers,
    TypedPeerDestinations<T> destinations
) {
    using Ops = LowPrecisionOps<T>;
    constexpr int group_width = GroupWidth;
    const int lane = threadIdx.x % group_width;
    const int group_in_block = threadIdx.x / group_width;
    const int groups_per_block = blockDim.x / group_width;
    const std::size_t vector_count = count / 8;
    const std::size_t chunk_size = (vector_count + group_width - 1) / group_width;
    const std::size_t chunk_begin = Me * chunk_size;
    const std::size_t chunk_end = min(chunk_begin + chunk_size, vector_count);
    const std::size_t first_index = chunk_begin + blockIdx.x * groups_per_block + group_in_block;
    const std::size_t stride = gridDim.x * groups_per_block;

    for (std::size_t vector_index = first_index; vector_index < chunk_end; vector_index += stride) {
        float2 values[4] = {};
        #pragma unroll
        for (int pe = lane; pe < 8; pe += group_width) {
            const LowPrecisionVector<T> peer_value = load_peer_vector(
                source,
                peers.pointers[pe],
                vector_index,
                pe
            );
            #pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const float2 converted = Ops::to_float2(peer_value.values[pair]);
                values[pair].x += converted.x;
                values[pair].y += converted.y;
            }
        }

        const unsigned int active_mask = __activemask();
        #pragma unroll
        for (int offset = group_width / 2; offset > 0; offset /= 2) {
            #pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                values[pair].x += __shfl_down_sync(
                    active_mask,
                    values[pair].x,
                    offset,
                    group_width
                );
                values[pair].y += __shfl_down_sync(
                    active_mask,
                    values[pair].y,
                    offset,
                    group_width
                );
            }
        }

        LowPrecisionVector<T> reduced;
        #pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 broadcast = {
                __shfl_sync(active_mask, values[pair].x, 0, group_width),
                __shfl_sync(active_mask, values[pair].y, 0, group_width),
            };
            reduced.values[pair] = Ops::from_float2(broadcast);
        }
        #pragma unroll
        for (int pe = lane; pe < 8; pe += group_width) {
            store_peer_vector(
                destination,
                destinations.pointers[pe],
                vector_index,
                pe,
                reduced
            );
        }
    }
}

template <typename T>
__global__ void low_precision_allgather_kernel(
    T* destination,
    std::size_t count,
    TypedPeerSources<T> destination_peers
) {
    using Packed = typename LowPrecisionOps<T>::Packed;
    const std::size_t pair_count = count / 2;
    const std::size_t chunk_size = (pair_count + destination_peers.npes - 1) / destination_peers.npes;
    const std::size_t first_index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = gridDim.x * blockDim.x;

    for (std::size_t pair_index = first_index; pair_index < pair_count; pair_index += stride) {
        const int owner = min(
            static_cast<int>(pair_index / chunk_size),
            destination_peers.npes - 1
        );
        if (owner == destination_peers.me) {
            continue;
        }
        reinterpret_cast<Packed*>(destination)[pair_index] = load_peer_pair(
            destination,
            destination_peers.pointers[owner],
            pair_index,
            owner
        );
    }
}

template <typename T>
__global__ void vectorized_low_precision_allgather_kernel(
    T* destination,
    std::size_t count,
    TypedPeerSources<T> destination_peers
) {
    const std::size_t vector_count = count / 8;
    const std::size_t chunk_size = (vector_count + destination_peers.npes - 1) / destination_peers.npes;
    const std::size_t first_index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = gridDim.x * blockDim.x;

    for (std::size_t vector_index = first_index; vector_index < vector_count; vector_index += stride) {
        const int owner = min(
            static_cast<int>(vector_index / chunk_size),
            destination_peers.npes - 1
        );
        if (owner == destination_peers.me) {
            continue;
        }
        reinterpret_cast<LowPrecisionVector<T>*>(destination)[vector_index] = load_peer_vector(
            destination,
            destination_peers.pointers[owner],
            vector_index,
            owner
        );
    }
}

__global__ void block_max_reduce_kernel(float* destination, const float* source, std::size_t count) {
    nvshmemx_float_max_reduce_block(
        NVSHMEM_TEAM_WORLD,
        destination,
        source,
        count
    );
}

__global__ void block_sum_reduce_kernel(float* destination, const float* source, std::size_t count) {
    nvshmemx_float_sum_reduce_block(
        NVSHMEM_TEAM_WORLD,
        destination,
        source,
        count
    );
}

template <typename T>
int launch_low_precision_sum_reduce(
    void* destination,
    const void* source,
    std::size_t count,
    void* signals,
    std::uint64_t epoch,
    std::uintptr_t stream
) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    readiness_kernel<<<1, 32, 0, cuda_stream>>>(
        static_cast<std::uint64_t*>(signals),
        epoch
    );
    TypedPeerSources<T> peers{};
    peers.npes = nvshmem_n_pes();
    peers.me = nvshmem_my_pe();
    for (int pe = 0; pe < peers.npes; ++pe) {
        peers.pointers[pe] = static_cast<const T*>(nvshmem_ptr(source, pe));
    }
    constexpr int threads = kLowPrecisionThreads;
    if (count % 2 == 0 && peers.npes <= kMaxPes) {
        const bool vectorized = count % 8 == 0;
        const std::size_t work_count = count / (vectorized ? 8 : 2);
        const std::size_t chunk_size = (work_count + peers.npes - 1) / peers.npes;
        const int reduce_blocks = std::min(
            static_cast<int>((chunk_size + threads - 1) / threads),
            kMaxLowPrecisionBlocks
        );
        if (vectorized) {
            const bool l2_only = count > 2048ULL * 4096ULL;
            const bool dual_accum = peers.npes == 8 && count <= 16384ULL * 4096ULL;
            TypedPeerDestinations<T> destinations{};
            for (int pe = 0; pe < peers.npes; ++pe) {
                destinations.pointers[pe] = static_cast<T*>(
                    nvshmem_ptr(destination, pe)
                );
            }
            #define LAUNCH_REDUCE_BROADCAST(NPES, ME) \
                if (l2_only) { \
                    if (dual_accum) { \
                        vectorized_low_precision_reduce_broadcast_kernel<T, NPES, ME, true, true> \
                            <<<reduce_blocks, threads, 0, cuda_stream>>>( \
                                static_cast<T*>(destination), \
                                static_cast<const T*>(source), \
                                count, \
                                peers, \
                                destinations \
                            ); \
                    } else { \
                        vectorized_low_precision_reduce_broadcast_kernel<T, NPES, ME, true, false> \
                            <<<reduce_blocks, threads, 0, cuda_stream>>>( \
                                static_cast<T*>(destination), \
                                static_cast<const T*>(source), \
                                count, \
                                peers, \
                                destinations \
                            ); \
                    } \
                } else { \
                    vectorized_low_precision_reduce_broadcast_kernel<T, NPES, ME, false, false> \
                        <<<reduce_blocks, threads, 0, cuda_stream>>>( \
                            static_cast<T*>(destination), \
                            static_cast<const T*>(source), \
                            count, \
                            peers, \
                            destinations \
                        ); \
                }
            switch (peers.npes) {
                case 2:
                    if (peers.me == 0) {
                        LAUNCH_REDUCE_BROADCAST(2, 0);
                    } else {
                        LAUNCH_REDUCE_BROADCAST(2, 1);
                    }
                    break;
                case 4:
                    switch (peers.me) {
                        case 0: LAUNCH_REDUCE_BROADCAST(4, 0); break;
                        case 1: LAUNCH_REDUCE_BROADCAST(4, 1); break;
                        case 2: LAUNCH_REDUCE_BROADCAST(4, 2); break;
                        default: LAUNCH_REDUCE_BROADCAST(4, 3); break;
                    }
                    break;
                case 8:
                    switch (peers.me) {
                        case 0: LAUNCH_REDUCE_BROADCAST(8, 0); break;
                        case 1: LAUNCH_REDUCE_BROADCAST(8, 1); break;
                        case 2: LAUNCH_REDUCE_BROADCAST(8, 2); break;
                        case 3: LAUNCH_REDUCE_BROADCAST(8, 3); break;
                        case 4: LAUNCH_REDUCE_BROADCAST(8, 4); break;
                        case 5: LAUNCH_REDUCE_BROADCAST(8, 5); break;
                        case 6: LAUNCH_REDUCE_BROADCAST(8, 6); break;
                        default: LAUNCH_REDUCE_BROADCAST(8, 7); break;
                    }
                    break;
                default:
                    LAUNCH_REDUCE_BROADCAST(0, -1);
                    break;
            }
            #undef LAUNCH_REDUCE_BROADCAST
        } else {
            low_precision_reduce_scatter_kernel<<<reduce_blocks, threads, 0, cuda_stream>>>(
                static_cast<T*>(destination),
                static_cast<const T*>(source),
                count,
                peers
            );
        }
        readiness_kernel<<<1, 32, 0, cuda_stream>>>(
            static_cast<std::uint64_t*>(signals),
            epoch + 1
        );
        if (vectorized) {
            return static_cast<int>(cudaGetLastError());
        }
        TypedPeerSources<T> destination_peers{};
        destination_peers.npes = peers.npes;
        destination_peers.me = peers.me;
        for (int pe = 0; pe < destination_peers.npes; ++pe) {
            destination_peers.pointers[pe] = static_cast<const T*>(
                nvshmem_ptr(destination, pe)
            );
        }
        const int gather_blocks = std::min(
            static_cast<int>((work_count + threads - 1) / threads),
            kMaxLowPrecisionBlocks
        );
        low_precision_allgather_kernel<<<gather_blocks, threads, 0, cuda_stream>>>(
            static_cast<T*>(destination),
            count,
            destination_peers
        );
    } else {
        const std::size_t work_items = (count + 1) / 2;
        const int blocks = std::min(
            static_cast<int>((work_items + threads - 1) / threads),
            kMaxLowPrecisionBlocks
        );
        parallel_low_precision_sum_kernel<<<blocks, threads, 0, cuda_stream>>>(
            static_cast<T*>(destination),
            static_cast<const T*>(source),
            count,
            peers
        );
    }
    return static_cast<int>(cudaGetLastError());
}

}  // namespace

extern "C" {

int liger_nvshmem_device_zero(void* pointer, std::size_t size, std::uintptr_t stream) {
    return static_cast<int>(
        cudaMemsetAsync(pointer, 0, size, reinterpret_cast<cudaStream_t>(stream))
    );
}

int liger_nvshmem_device_float_max_reduce(
    void* destination,
    const void* source,
    std::size_t count,
    void* signals,
    std::uint64_t epoch,
    std::uintptr_t stream
) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (count <= 8192) {
        block_max_reduce_kernel<<<1, kSmallCollectiveThreads, 0, cuda_stream>>>(
            static_cast<float*>(destination),
            static_cast<const float*>(source),
            count
        );
    } else {
        readiness_kernel<<<1, 32, 0, cuda_stream>>>(
            static_cast<std::uint64_t*>(signals),
            epoch
        );
        PeerSources peers{};
        peers.npes = nvshmem_n_pes();
        peers.me = nvshmem_my_pe();
        for (int pe = 0; pe < peers.npes; ++pe) {
            peers.pointers[pe] = static_cast<const float*>(nvshmem_ptr(source, pe));
        }
        constexpr int threads = 256;
        const std::size_t work_items = (count + 3) / 4;
        const int blocks = std::min(
            static_cast<int>((work_items + threads - 1) / threads),
            kMaxCollectiveBlocks
        );
        parallel_max_kernel<<<blocks, threads, 0, cuda_stream>>>(
            static_cast<float*>(destination),
            static_cast<const float*>(source),
            count,
            peers
        );
    }
    return static_cast<int>(cudaGetLastError());
}

int liger_nvshmem_device_float_sum_reduce(
    void* destination,
    const void* source,
    std::size_t count,
    void* signals,
    std::uint64_t epoch,
    std::uintptr_t stream
) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (count <= 8192) {
        block_sum_reduce_kernel<<<1, kSmallCollectiveThreads, 0, cuda_stream>>>(
            static_cast<float*>(destination),
            static_cast<const float*>(source),
            count
        );
    } else {
        readiness_kernel<<<1, 32, 0, cuda_stream>>>(
            static_cast<std::uint64_t*>(signals),
            epoch
        );
        PeerSources peers{};
        peers.npes = nvshmem_n_pes();
        peers.me = nvshmem_my_pe();
        for (int pe = 0; pe < peers.npes; ++pe) {
            peers.pointers[pe] = static_cast<const float*>(
                nvshmem_ptr(source, pe)
            );
        }
        constexpr int threads = 256;
        if (count % 4 == 0 && peers.npes <= kMaxPes) {
            const std::size_t vector_count = count / 4;
            const std::size_t chunk_size = (vector_count + peers.npes - 1) / peers.npes;
            const int reduce_blocks = std::min(
                static_cast<int>((chunk_size + threads - 1) / threads),
                kMaxCollectiveBlocks
            );
            reduce_scatter_kernel<<<reduce_blocks, threads, 0, cuda_stream>>>(
                static_cast<float*>(destination),
                static_cast<const float*>(source),
                count,
                peers
            );
            readiness_kernel<<<1, 32, 0, cuda_stream>>>(
                static_cast<std::uint64_t*>(signals),
                epoch + 1
            );
            PeerSources destination_peers{};
            destination_peers.npes = peers.npes;
            destination_peers.me = peers.me;
            for (int pe = 0; pe < destination_peers.npes; ++pe) {
                destination_peers.pointers[pe] = static_cast<const float*>(
                    nvshmem_ptr(destination, pe)
                );
            }
            const int gather_blocks = std::min(
                static_cast<int>((vector_count + threads - 1) / threads),
                kMaxCollectiveBlocks
            );
            allgather_kernel<<<gather_blocks, threads, 0, cuda_stream>>>(
                static_cast<float*>(destination),
                count,
                destination_peers
            );
        } else {
            const std::size_t work_items = (count + 3) / 4;
            const int blocks = static_cast<int>((work_items + threads - 1) / threads);
            parallel_sum_kernel<<<blocks, threads, 0, cuda_stream>>>(
                static_cast<float*>(destination),
                static_cast<const float*>(source),
                count,
                peers
            );
        }
    }
    return static_cast<int>(cudaGetLastError());
}

int liger_nvshmem_device_bfloat16_sum_reduce(
    void* destination,
    const void* source,
    std::size_t count,
    void* signals,
    std::uint64_t epoch,
    std::uintptr_t stream
) {
    return launch_low_precision_sum_reduce<__nv_bfloat16>(
        destination,
        source,
        count,
        signals,
        epoch,
        stream
    );
}

int liger_nvshmem_device_float16_sum_reduce(
    void* destination,
    const void* source,
    std::size_t count,
    void* signals,
    std::uint64_t epoch,
    std::uintptr_t stream
) {
    return launch_low_precision_sum_reduce<__half>(
        destination,
        source,
        count,
        signals,
        epoch,
        stream
    );
}

}
