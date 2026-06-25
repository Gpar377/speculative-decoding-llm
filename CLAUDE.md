# Claude Code Guidelines - Speculative Decoding LLM

## Project Overview
This repository contains a PyTorch-based speculative decoding implementation to accelerate LLM inference using draft-target model verification.

## Technology Stack
*   **Python 3.10+**
*   **PyTorch** (CUDA-enabled)
*   **HuggingFace Transformers / Accelerate**

## Coding Standards & Conventions
*   Avoid standard autoregressive loop bottlenecks: reuse KV-caches efficiently to avoid redundant tensor allocations.
*   Clearly modularize the verification sampling logic to allow testing of greedy vs. temperature-based speculative decoding.
*   Document state management in the KV-cache rollback logic. Ensure cached tensors are properly truncated along the sequence dimension upon token rejection.
*   Log acceptance stats (e.g., mean accepted tokens per step, time saved) during inference runs.

## Workflow Rules & Commands
*   **Run Speculative Generation:** `python generate.py --prompt "Your prompt" --draft "meta-llama/Llama-3.2-1B" --target "meta-llama/Llama-3-8B"`
*   **Run Tests:** `pytest tests/` (includes mathematical validation of output distribution match)
*   **Benchmark Performance:** `python scripts/benchmark.py`
