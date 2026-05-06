# Token-Generation Latency Benchmarking in LLaMA
## CECS530-TokenGenerationLatency
### Mac M4 16GB vs Mac M2 8GB — Apple Silicon Unified Memory Study

---

## Overview

This project benchmarks token-generation latency in LLaMA-family models across two Apple Silicon devices: Mac M4 16GB and Mac M2 8GB. The workload, model family, batch size, backend, and benchmark scripts are held constant to study how Apple Silicon memory bandwidth and related hardware-generation factors affect autoregressive decode latency.

The main result is that M4 shows an observed **~1.5x per-token latency advantage** over M2 under this benchmark setup. Memory bandwidth is the primary explanatory variable, but GPU core count, cache hierarchy, RAM capacity, OS version, thermal state, and scheduler behavior are acknowledged as residual confounders that could not be independently controlled.

---

## Evidence Labels

| Label | Meaning |
|-------|---------|
| `measured` | Directly timed using `time.perf_counter()` with explicit MPS/CUDA synchronization |
| `analytical` | Computed from model architecture, such as KV-cache size formula |
| `estimated` | Derived from architecture-level proportions calibrated to measured PTL |
| `modeled` | Projected from measured F16 baselines using bandwidth-scaling assumptions |

**Q8_0 and Q4_K_M speedups are `modeled`** unless GGUF files are supplied and benchmarked through the `llama.cpp` path (`bench-07-llamacpp`).

---

## Hardware

| Device   | Chip     | RAM  | Memory Bandwidth | GPU Cores | Backend |
|----------|----------|------|-----------------|-----------|---------|
| Mac M4   | Apple M4 | 16GB | 120 GB/s        | 10        | MPS     |
| Mac M2   | Apple M2 | 8GB  | 100 GB/s        | 8         | MPS     |

> M4 and M2 differ in more than memory bandwidth. GPU core count, cache hierarchy, RAM capacity, and OS version (macOS 15 vs 14) are not independently controlled and are treated as residual confounders.

---

## Model

**TinyLlama-1.1B-Chat-v1.0** (`TinyLlama/TinyLlama-1.1B-Chat-v1.0` on HuggingFace)

- 1.1B parameters · 22 layers · 32 attention heads · 4 KV heads (GQA) · float16
- Fits within 8GB RAM; preserves LLaMA-family KV-cache growth and bandwidth behavior
- Absolute latency values are lower than 7B models; relative M4/M2 trends generalize directionally

---

## Key Findings

| Metric | Mac M4 16GB | Mac M2 8GB | Evidence |
|--------|------------|------------|----------|
| TTFT @ 128 tokens | 26.9 ms | 45.0 ms | measured |
| PTL @ 512 context | 30.1 ms | 45.8 ms | measured |
| Throughput @ 128 tokens | ~44 tok/s | ~29 tok/s | measured |
| Cold start (model load) | ~3.6 s | ~4.3 s | measured |
| Q4_K_M speedup vs F16 | ~2.2× | ~2.0× | **modeled** |

- PTL grows **152%** from context 32→1024 tokens due to KV-cache memory traffic (measured)
- Estimated effective bandwidth utilization reaches **~82%** of M4's 120 GB/s ceiling (derived)
- M2 8GB system-level RAM pressure begins around ~1,950 tokens in F16; Q4_K_M shifts this to ~7,800 tokens (analytical estimate — includes model weights, Metal pools, OS overhead, not KV-cache bytes alone)
- p99 latency is **+33% above median** on M4 at 512-token context (measured, n=50 trials)

---

## Repository Structure

```
CECS530-TokenGenerationLatency/
├── benchmarks/
│   ├── utils.py                          # Shared utilities (device, sync, I/O, seed)
│   ├── run_smoke_test.py                 # Quick grader verification (no GGUF needed)
│   ├── 01_ttft_vs_prompt_length.py
│   ├── 02_per_token_latency_vs_context.py
│   ├── 03_e2e_latency_vs_output_length.py
│   ├── 04_throughput_vs_prompt_length.py
│   ├── 05_inter_token_latency_timeline.py
│   ├── 06_cold_vs_warm_run.py
│   ├── 07_quantization_speedup.py        # modeled by default; measured via llamacpp path
│   ├── 08_kvcache_size_vs_context.py     # analytical; no GPU needed
│   └── 09_latency_decomposition.py
├── analysis/
│   └── generate_research_graphs.py
├── results/
│   ├── README.md
│   ├── Mac_M4_16GB/
│   └── Mac_M2_8GB/
├── graphs/
│   └── README.md
├── tests/
│   ├── test_kvcache_formula.py
│   ├── test_result_schema.py
│   └── test_smoke.py
├── docs/
│   └── reproducibility.md
├── optimization/
│   └── kv_cache_proposal.md
├── report/
│   └── findings_report.md
├── README.md
├── requirements.txt
├── Makefile
├── CITATION.cff
└── LICENSE
```

---

## Setup

```bash
git clone https://github.com/Vinayk393/CECS530-TokenGenerationLatency.git
cd CECS530-TokenGenerationLatency

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Minimal Reproduction Path

```bash
# 1. Unit tests (no model download needed):
python -m pytest tests/test_kvcache_formula.py -v

# 2. Smoke test (downloads ~2.2GB TinyLlama on first run):
python benchmarks/run_smoke_test.py

# 3. Full benchmark suite:
make bench

# 4. Generate figures:
make graphs

# 5. Smoke test + pytest together:
make verify
```

Expected outputs:
- `results/smoke_test.json` — from smoke test
- `results/Mac_M4_16GB/` — benchmark CSVs
- `graphs/` — generated PNGs

---

## Running Benchmarks

```bash
make bench                          # All 9 benchmarks (default model)
make bench-07-modeled               # Q4/Q8 projected from measured F16 (default)
make bench-07-llamacpp              # Q4/Q8 via real GGUF files (requires llama.cpp)
make smoke                          # Fast smoke test only
make verify                         # Smoke test + pytest
make graphs                         # Generate figures from existing CSVs

# Override model or bandwidth:
make bench MODEL=meta-llama/Llama-3.2-1B
make bench-04 PEAK_BW=100           # M2 bandwidth
```

---

## Quantization Note

Script 07 has two modes:

```bash
# Default — measures F16, then projects Q4/Q8 (labeled measurement_type=modeled):
make bench-07-modeled

# For actual measured speedup — requires llama.cpp and GGUF model files:
python benchmarks/07_quantization_speedup.py \
    --backend llamacpp \
    --llama_bench /path/to/llama-bench \
    --models Q4=/path/to/model-Q4_K_M.gguf \
             Q8=/path/to/model-Q8_0.gguf \
             F16=/path/to/model-F16.gguf
```

Q8_0 and Q4_K_M speedup values in the paper are **modeled** projections calibrated from measured F16 PTL.

---

## Limitations

- M4 and M2 differ in more than memory bandwidth; bandwidth is the **primary explanatory variable**, not the only causal factor
- Q8_0 and Q4_K_M speedups are **modeled** unless GGUF models are run via `llama.cpp`
- MPS does not expose hardware performance counters; latency decomposition is **estimated**
- Absolute latencies vary with OS version, thermal state, and background processes
- TinyLlama-1.1B absolute values do not generalize to 7B+ models; directional trends do

---

## Third-Party Software and Model Credits

- **[PyTorch](https://pytorch.org/)** — tensor execution and MPS backend
- **[HuggingFace Transformers](https://github.com/huggingface/transformers)** — model loading and autoregressive generation
- **[TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)** — benchmark model
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** — optional GGUF quantized-model benchmarking
- **NumPy, pandas, matplotlib, seaborn** — analysis and visualization
- **pytest** — unit testing

No third-party source code is copied into this repository beyond standard library and API usage.

---

## References

- Touvron et al., "LLaMA: Open and Efficient Foundation Language Models," arXiv 2023
- Grattafiori et al., "The Llama 3 Herd of Models," arXiv 2024
- Gerganov, "llama.cpp," GitHub, 2023
- Williams et al., "Roofline: An Insightful Visual Performance Model," CACM 2009
- Dettmers & Zettlemoyer, "The Case for 4-bit Precision," ICML 2023
- Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention," SOSP 2023
- Zhang et al., "TinyLlama: An Open-Source Small Language Model," arXiv 2024
