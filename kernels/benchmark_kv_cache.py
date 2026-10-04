"""Benchmark KV-cache integration speedup."""
import time
import torch
import sys
sys.path.insert(0, '/home/scott/git/auto-finetune/kernels')

def benchmark(model_name, model_path, num_tokens=1000, batch_size=1):
    """Benchmark model inference with and without KV-cache."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, fix_mistral_regex=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map='auto')

    prompt = "Agent session summary task: " + "Key fact: " * (num_tokens // 20)
    inputs = tokenizer(prompt, return_tensors='pt', max_length=262144, truncation=True).to('cuda:0')

    # Benchmark without KV-cache (baseline)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
    baseline_time = time.time() - t0
    baseline_tok_s = 100 / baseline_time

    # Benchmark with KV-cache integration
    # (Apply Triton patch first)
    from kernels.agnes_patch import apply_triton_patch, integrate_kv_cache
    apply_triton_patch()
    manager = integrate_kv_cache()

    t1 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
    kv_time = time.time() - t1
    kv_tok_s = 100 / kv_time

    print(f"=== KV-CACHE BENCHMARK ===")
    print(f"Baseline: {baseline_time:.1f}s ({baseline_tok_s:.1f} tok/s)")
    print(f"With KV-cache: {kv_time:.1f}s ({kv_tok_s:.1f} tok/s)")
    print(f"Speedup: {kv_tok_s / baseline_tok_s:.2f}x")
    print(f"Manager: {manager}")

    return {
        'baseline_time': baseline_time,
        'baseline_tok_s': baseline_tok_s,
        'kv_time': kv_time,
        'kv_tok_s': kv_tok_s,
        'speedup': kv_tok_s / baseline_tok_s,
    }

if __name__ == '__main__':
    results = benchmark(
        'Agnes-3.0-Flash',
        '/home/scott/.cache/lm-studio/models/Agnes-AI/Agnes-3.0-Flash',
    )
    # Save results
    import json
    with open('/home/scott/git/auto-finetune/kernels/kv_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to kv_benchmark_results.json")
