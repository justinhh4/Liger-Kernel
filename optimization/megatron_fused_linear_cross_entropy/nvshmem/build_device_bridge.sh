#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
DIR="$ROOT/optimization/megatron_fused_linear_cross_entropy/nvshmem"
PYTHON="$ROOT/.venv/bin/python"
SITE_PACKAGES="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
NVSHMEM="$SITE_PACKAGES/nvidia/nvshmem"
INCLUDE="$NVSHMEM/include"
LIB="$NVSHMEM/lib"
HOST_OBJECT="$DIR/.nvshmem_device_host.o"
DEVICE_OBJECT="$DIR/.nvshmem_device_bridge.o"
DLINK_OBJECT="$DIR/.nvshmem_device_dlink.o"

nvcc -std=c++17 -Xcompiler -fPIC -I"$INCLUDE" \
  -c "$DIR/nvshmem_device_host.cpp" -o "$HOST_OBJECT"
nvcc -std=c++17 -Xcompiler -fPIC -rdc=true -I"$INCLUDE" \
  -c "$DIR/nvshmem_device_bridge.cu" -o "$DEVICE_OBJECT"
nvcc -std=c++17 -Xcompiler -fPIC -rdc=true -dlink "$DEVICE_OBJECT" \
  -o "$DLINK_OBJECT" -L"$LIB" -lnvshmem_device
nvcc -shared -Xcompiler -fPIC \
  "$HOST_OBJECT" "$DEVICE_OBJECT" "$DLINK_OBJECT" \
  -o "$DIR/libliger_nvshmem_device.so" \
  -L"$LIB" -Xlinker=-rpath -Xlinker="$LIB" \
  -Xlinker=-l:libnvshmem_host.so.3 -lnvshmem_device -lcudart -lcuda

rm -f "$HOST_OBJECT" "$DEVICE_OBJECT" "$DLINK_OBJECT"
