# Speculative Decoding Engine
A high-performance LLM speculative decoding implementation designed to accelerate LLM inference. It uses a small, fast "draft" model to generate a sequence of token proposals, which a larger "target" model validates in a single forward pass, significantly reducing overall latency while maintaining target model distribution guarantees.

## Proposed Git Repo Name
`speculative-decoding-llm`

## Architecture & Scope
*   **Draft and Target Model Pipelines:** Inference runners utilizing PyTorch/HuggingFace Transformers (e.g., Llama-3-8B as target, Llama-3-Draft-1B as draft).
*   **KV-Caching Engine:** Custom Key-Value (KV) cache manager supporting speculative validation. The draft model advances its cache, and if the target model rejects draft tokens, the KV cache is rolled back to the last accepted token index.
*   **Acceptance Sampling Kernel:** Implementation of speculative sampling logic:
    *   Compare probabilities $p(x)$ (target) and $q(x)$ (draft).
    *   If $p(x) \geq q(x)$, accept the token.
    *   If $p(x) < q(x)$, accept with probability $p(x)/q(x)$, otherwise reject and sample from a corrected distribution.
*   **Benchmarking Harness:** Framework measuring token throughput (tokens per second), latency per token, and acceptance rates across different contexts and prompt domains.

## Target Milestones
1. Base inference pipeline setup for target and draft models with raw HuggingFace models.
2. Custom KV Cache implementation supporting checkpointing and rollback.
3. Speculative acceptance/rejection loop with probability matching and correction sampling.
4. Benchmarking harness verifying output distribution equivalence (via KL-divergence checking) and measuring latency speedups.
