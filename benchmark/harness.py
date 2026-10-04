"""Agnes-3.0-Flash Before/After Benchmark Harness"""
import time, json, sys
sys.path.insert(0, '/home/scott/git/auto-finetune/src')
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = '/home/scott/.cache/lm-studio/models/Agnes-AI/Agnes-3.0-Flash'
AGENT_PROMPT = "Agent session summary task: " + "Key fact: " * 500
RESULTS = {}

def load_model():
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, fix_mistral_regex=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map='auto')
    load_time = time.time() - t0
    return tokenizer, model, load_time

def benchmark_gen(tokenizer, model, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors='pt', max_length=262144, truncation=True).to('cuda:0')
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.7, do_sample=True)
    gen_time = time.time() - t0
    tok_s = max_new_tokens / gen_time
    return gen_time, tok_s, tokenizer.decode(out[0], skip_special_tokens=True)

def coherence_score(response):
    keywords = ['action', 'risk', 'summary', 'flag', 'anomaly', 'resource']
    return sum(1 for kw in keywords if kw.lower() in response.lower()) / len(keywords)

def main():
    print("=== AGNES-3.0-FLASH BENCHMARK HARNESS ===")
    tokenizer, model, load_time = load_model()
    RESULTS['load_time'] = load_time
    print(f"Load time: {load_time:.1f}s")
    gen_time, tok_s, response = benchmark_gen(tokenizer, model, AGENT_PROMPT)
    RESULTS['gen_time'] = gen_time
    RESULTS['tok_s'] = tok_s
    RESULTS['coherence'] = coherence_score(response)
    print(f"Gen time: {gen_time:.1f}s | tok/s: {tok_s:.1f} | coherence: {RESULTS['coherence']:.2f}")
    full_gen_time, full_tok_s, full_response = benchmark_gen(tokenizer, model, AGENT_PROMPT, max_new_tokens=20)
    RESULTS['full_gen_time'] = full_gen_time
    RESULTS['full_tok_s'] = full_tok_s
    print(f"Full ctx gen: {full_gen_time:.1f}s | {full_tok_s:.1f} tok/s")
    with open('/home/scott/git/auto-finetune/benchmark/results.json', 'w') as f:
        json.dump(RESULTS, f, indent=2)
    print("Results saved to benchmark/results.json")

if __name__ == '__main__':
    main()
