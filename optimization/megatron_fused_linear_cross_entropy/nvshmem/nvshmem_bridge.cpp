#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

extern "C" {

std::size_t liger_nvshmem_uniqueid_size() {
    return sizeof(nvshmemx_uniqueid_t);
}

int liger_nvshmem_get_uniqueid(void* output, std::size_t size) {
    if (size != sizeof(nvshmemx_uniqueid_t)) {
        return -1;
    }
    nvshmemx_uniqueid_t uniqueid;
    const int status = nvshmemx_get_uniqueid(&uniqueid);
    if (status == 0) {
        std::memcpy(output, &uniqueid, sizeof(uniqueid));
    }
    return status;
}

int liger_nvshmem_init(int rank, int world_size, const void* input, std::size_t size) {
    if (size != sizeof(nvshmemx_uniqueid_t)) {
        return -1;
    }
    nvshmemx_uniqueid_t uniqueid;
    std::memcpy(&uniqueid, input, sizeof(uniqueid));
    nvshmemx_init_attr_t attr;
    std::memset(&attr, 0, sizeof(attr));
    int status = nvshmemx_set_attr_uniqueid_args(rank, world_size, &uniqueid, &attr);
    if (status != 0) {
        return status;
    }
    return nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
}

void liger_nvshmem_finalize() {
    nvshmemx_hostlib_finalize();
}

void* liger_nvshmem_malloc(std::size_t size) {
    return nvshmem_malloc(size);
}

void liger_nvshmem_free(void* pointer) {
    nvshmem_free(pointer);
}

void* liger_nvshmem_register_symmetric(void* pointer, std::size_t size) {
    return nvshmemx_buffer_register_symmetric(pointer, size, 0);
}

int liger_nvshmem_unregister_symmetric(void* pointer, std::size_t size) {
    return nvshmemx_buffer_unregister_symmetric(pointer, size);
}

int liger_nvshmem_float_max_reduce(
    void* destination,
    const void* source,
    std::size_t count,
    std::uintptr_t stream
) {
    return nvshmemx_float_max_reduce_on_stream(
        NVSHMEM_TEAM_WORLD,
        static_cast<float*>(destination),
        static_cast<const float*>(source),
        count,
        reinterpret_cast<cudaStream_t>(stream)
    );
}

int liger_nvshmem_float_sum_reduce(
    void* destination,
    const void* source,
    std::size_t count,
    std::uintptr_t stream
) {
    return nvshmemx_float_sum_reduce_on_stream(
        NVSHMEM_TEAM_WORLD,
        static_cast<float*>(destination),
        static_cast<const float*>(source),
        count,
        reinterpret_cast<cudaStream_t>(stream)
    );
}

int liger_nvshmem_barrier(std::uintptr_t stream) {
    return nvshmemx_barrier_on_stream(
        NVSHMEM_TEAM_WORLD,
        reinterpret_cast<cudaStream_t>(stream)
    );
}

}
