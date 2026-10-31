#!/usr/bin/env bash
# Locally generated synthetic-quality harness; never alters production runtimes.
set -Eeuo pipefail
base=/home/scott/lfm3-research
source_root="$base/llama-cpu-8ca4060-source"
template_source="$source_root/examples/simple/simple.cpp"
src="$base/lfm3-quality-simple.cpp"
binary="$base/lfm3-quality-simple"
candidate="$base/lfm3-quality-simple.rebuilt"
target_lib="$base/build-hip-gfx1151-fa-q8/bin"
baseline_lib="$base/build-hip-gfx1151-bench/bin"

test "$(hostname -s)" = xwing || { echo 'SKIP: xwing only'; exit 78; }
test "$(awk '/MemAvailable:/ {print int($2/1024)}' /proc/meminfo)" -ge 6144 || { echo 'SKIP: memory'; exit 78; }
test "$(df -Pm "$base" | awk 'END{print $4}')" -ge 32768 || { echo 'SKIP: disk'; exit 78; }
test -f "$template_source" && test -d "$target_lib" && test -d "$baseline_lib"
source_sha=$(sha256sum "$template_source" | awk '{print $1}')
test "$source_sha" = '5f3160b48b5c0adff984dfdc6ad39cf9ee113b4d63600596f2127f9a0b9a869e' || {
    echo 'Source identity mismatch'; exit 2;
}
python3 - <<'PY'
from pathlib import Path
base=Path('/home/scott/lfm3-research')
original=(base/'llama-cpu-8ca4060-source/examples/simple/simple.cpp').read_text()
replacement={
    '    const llama_vocab * vocab = llama_model_get_vocab(model);':
'''    const char * chat_template = llama_model_chat_template(model, nullptr);
    if (chat_template) {
        FILE * out_template = fopen("/home/scott/lfm3-research/lfm3-chat-template.j2", "wb");
        if (out_template) {
            fwrite(chat_template, 1, strlen(chat_template), out_template);
            fclose(out_template);
        }
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);''',
    'ctx_params.n_ctx = n_prompt + n_predict - 1;': 'ctx_params.n_ctx = 4096;',
    '    ctx_params.n_batch = n_prompt;':
'''    ctx_params.n_batch = n_prompt;
    ctx_params.type_k = GGML_TYPE_Q8_0;
    ctx_params.type_v = GGML_TYPE_Q8_0;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;''',
    '    // prepare a batch for the prompt':
'''    printf("\\n===OUTPUT===\\n");
    fflush(stdout);

    // prepare a batch for the prompt''',
}
for before, after in replacement.items():
    if original.count(before)!=1:
        raise SystemExit('Unexpected source shape: '+repr(before))
    original=original.replace(before,after,1)
(base/'lfm3-quality-simple.cpp').write_text(original)
PY
nice -n 12 timeout 60 g++ -std=c++17 -O2     -I"$source_root/include" -I"$source_root/ggml/include"     "$src" -L"$target_lib" -Wl,-rpath,"$target_lib"     -lllama -lggml -lggml-base -pthread -o "$candidate"

if test -f "$binary" && ! cmp -s "$candidate" "$binary"; then
    echo "FAIL: compiled harness differs from previously attested binary; leaving original intact" >&2
    sha256sum "$binary" "$candidate"
    exit 2
fi
mv -f "$candidate" "$binary"
test -x "$binary"
for libdir in "$target_lib" "$baseline_lib"; do
    linked=$(LD_LIBRARY_PATH="$libdir" ldd "$binary")
    grep -Fq "libggml-hip.so.0 => $libdir/libggml-hip.so.0" <<<"$linked" || {
        echo "Expected isolated ROCm loader path absent: $libdir"; exit 2;
    }
done
echo 'Synthetic test harness built and both HIP backends resolve:'
sha256sum "$src" "$binary"
