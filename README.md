# Speculative Decoding LLM Inference Engine

A high-performance implementation of speculative decoding designed to accelerate LLM inference. The engine uses a lightweight "draft" model to generate token proposals autoregressively, which a larger "target" model validates in parallel in a single forward pass, maintaining output equivalence while significantly reducing latency.

---

## 🏗️ Architecture Design

*   **KV-Cache Rollback (`src/kv_cache.py`):** Custom Key-Value cache manager interface supporting dynamic sequence length truncation. When speculative tokens are rejected, the cache is rolled back to the first rejected index to preserve consistency.
*   **Speculative Sampling (`src/decoding.py`):** Implementation of acceptance sampling math where proposed draft tokens are evaluated against target token probability distributions. Corrected replacement tokens are sampled upon rejections.

---

## 🛠️ Installation and Setup

### 1. Set Up Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/WSL
.venv\Scripts\activate     # Windows

# Install modeling dependencies
pip install torch transformers accelerate
```

### 2. Run Inference CLI
Generates text using custom speculative loops:
```bash
python main.py --prompt "Explain the concept of neural networks in 3 sentences."
```

### 3. Run Benchmark Profiler
Measures exact generation throughput (tokens/sec) and speedup ratios of Speculative Decoding vs. standard Autoregressive Generation:
```bash
python benchmark.py
```
