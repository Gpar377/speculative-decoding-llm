import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.decoding import SpeculativeDecoder

def run_benchmarks():
    target_id = "Qwen/Qwen2.5-1.5B-Instruct"
    draft_id = "Qwen/Qwen2.5-0.5B-Instruct"
    prompt = "Write a comprehensive article explaining quantum computing for children, focusing on qubits, superposition, and entanglement."
    max_tokens = 100
    lookahead_k = 4
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== Running Speculative Decoding Performance Profile ===")
    print(f"Device: {device} | Prompt Length: {len(prompt)} chars")
    print(f"Target: {target_id} | Draft: {draft_id}\n")

    tokenizer = AutoTokenizer.from_pretrained(target_id)
    
    # Load models
    target_model = AutoModelForCausalLM.from_pretrained(target_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
    draft_model = AutoModelForCausalLM.from_pretrained(draft_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)

    # ------------------ Profile Standard Autoregressive Generation ------------------
    print("Running Baseline Autoregressive Generation...")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # Warmup
    _ = target_model.generate(input_ids, max_new_tokens=10, use_cache=True)
    if device == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    outputs = target_model.generate(
        input_ids, 
        max_new_tokens=max_tokens, 
        use_cache=True,
        do_sample=False # Greedy for baseline comparison
    )
    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    t_baseline = end_time - start_time
    baseline_tokens = outputs.shape[1] - input_ids.shape[1]
    baseline_throughput = baseline_tokens / t_baseline

    print(f"  Baseline time: {t_baseline:.2f} seconds")
    print(f"  Baseline output: {baseline_tokens} tokens")
    print(f"  Baseline speed: {baseline_throughput:.2f} tokens/sec\n")

    # ------------------ Profile Speculative Decoding ------------------
    print("Running Speculative Decoding Generation...")
    decoder = SpeculativeDecoder(target_model, draft_model, tokenizer)
    
    # Warmup
    _ = decoder.generate(prompt, max_new_tokens=10, lookahead_k=lookahead_k)
    if device == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    spec_output_text = decoder.generate(
        prompt=prompt,
        max_new_tokens=max_tokens,
        lookahead_k=lookahead_k
    )
    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    t_spec = end_time - start_time
    spec_tokens = len(tokenizer(spec_output_text).input_ids) - input_ids.shape[1]
    spec_throughput = spec_tokens / t_spec

    print(f"  Speculative time: {t_spec:.2f} seconds")
    print(f"  Speculative output: {spec_tokens} tokens")
    print(f"  Speculative speed: {spec_throughput:.2f} tokens/sec\n")

    # ------------------ Performance Summary ------------------
    speedup = spec_throughput / baseline_throughput
    print("--- Benchmark Summary ---")
    print(f"Latency Speedup: {speedup:.2f}x faster throughput")
    print(f"Throughput difference: +{spec_throughput - baseline_throughput:.2f} tokens/sec")
    print("-------------------------")

if __name__ == "__main__":
    run_benchmarks()
