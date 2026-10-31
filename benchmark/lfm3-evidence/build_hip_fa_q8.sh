#!/usr/bin/env bash
# Run ONLY on xwing. Incremental, isolated build of a HIP Flash Attention variant.
# This intentionally never installs packages or overwrites production/baseline binaries.
set -Eeuo pipefail
base=/home/scott/lfm3-research
source="$base/llama-cpu-8ca4060-source"
build="$base/build-hip-gfx1151-fa-q8"
log="$base/hip-fa-q8-build.log"
config="$base/hip-fa-q8-config.log"
test "$(hostname -s)" = xwing || { echo "SKIP: not xwing" >&2; exit 78; }
command -v cmake >/dev/null
command -v ninja >/dev/null
command -v hipcc >/dev/null
command -v flock >/dev/null
command -v timeout >/dev/null
command -v ionice >/dev/null
command -v rocminfo >/dev/null
rocminfo 2>/dev/null | grep 'Name: *gfx1151' >/dev/null || { echo 'SKIP: gfx1151 unavailable'; exit 78; }
avail=$(awk '/MemAvailable:/ { print int($2/1024) }' /proc/meminfo)
disk=$(df -Pm "$base" | awk 'END { print $4 }')
echo "preflight available_memory_mib=$avail disk_free_mib=$disk"
test "$avail" -ge 6144 && test "$disk" -ge 32768 || { echo 'SKIP: capacity gate'; exit 78; }
test -d "$source" || { echo "missing $source" >&2; exit 2; }
# Selected tracked source files, pinned to llama.cpp 8ca4060. Not whole-tree attestation.
while read -r digest rel; do
  actual=$(sha256sum "$source/$rel" | awk '{print $1}')
  test "$actual" = "$digest" || { echo "source mismatch: $rel" >&2; exit 2; }
done <<'HASHES'
40da6c2cce09fc79798952e38e0f9d33f9616c5f3a39f89188ff2c1e9198b7ef CMakeLists.txt
73ce1788faa7531554a7a1bbf4e6db63807c067b02adc5e5697968a70818cd04 src/llama-model.cpp
f76532e7d7c510426bb4f96f1c1bcf4ca5321790b07b56ce9446d793feef8d35 src/llama-context.cpp
644205340d2c421abffcf2750508007fb8a168f9d01a1cd6000131a858e5f274 ggml/src/ggml-hip/CMakeLists.txt
0c7ac5176786975fdd66709b5f9a2b092ce4d6a57456356e67c4d9f9193203c2 ggml/src/ggml-vulkan/CMakeLists.txt
8eed21016aabf399c7223f1c8588840edc1d15398e87f2048056d381b8a182ec tools/llama-bench/llama-bench.cpp
HASHES
exec 9>"$base/.hip-fa-q8-build.lock"
flock -n 9 || { echo 'SKIP: another HIP FA build holds the lock' >&2; exit 78; }
cmake -S "$source" -B "$build" -G Ninja   -DGGML_HIP=ON -DGGML_VULKAN=OFF -DGGML_CUDA_FA=ON   -DGGML_CUDA_FA_ALL_QUANTS=OFF   '-DGGML_CUDA_FA_QUANTS=f16-f16;q8_0-q8_0'   -DCMAKE_HIP_ARCHITECTURES=gfx1151   -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF   -DLLAMA_BUILD_SERVER=OFF -DCMAKE_BUILD_TYPE=Release   -DLLAMA_CURL=OFF >"$config" 2>&1
echo "configured $build"
# 300s hard timeout. Exit 124 explicitly means still building; keep incremental artifacts.
nice -n 16 ionice -c3 timeout 300   cmake --build "$build" --target llama-bench -j2 >"$log" 2>&1 || {
  rc=$?
  echo "build incomplete/failure exit=$rc; log=$log; source and baseline untouched" >&2
  exit "$rc"
}
test -x "$build/bin/llama-bench"
echo 'HIP FA Q8 binary ready:'
env LD_LIBRARY_PATH="$build/bin" "$build/bin/llama-bench" --version
