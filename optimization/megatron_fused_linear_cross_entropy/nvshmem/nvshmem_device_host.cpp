#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstddef>
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
    return nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
}

void liger_nvshmem_finalize() {
    nvshmem_finalize();
}

void* liger_nvshmem_malloc(std::size_t size) {
    return nvshmem_malloc(size);
}

void liger_nvshmem_free(void* pointer) {
    nvshmem_free(pointer);
}

}
