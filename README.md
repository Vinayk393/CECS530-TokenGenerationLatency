# Token-Generation Latency Benchmarking in LLaMA
## CECS530-TokenGenerationLatency
### Mac M4 16GB vs Mac M2 8GB — Apple Silicon Unified Memory Study

---

## Overview

This project benchmarks token-generation latency in LLaMA-family models across two Apple Silicon devices: Mac M4 16GB and Mac M2 8GB. The workload, model family, batch size, backend, and benchmark scripts are held constant to study how Apple Silicon memory bandwidth and related hardware-generation factors affect autoregressive decode latency.

The main result is that M4 shows an observed **~1.5× per-token latency advantage** over M2 under this benchmark setup. Memory bandwidth is the primary explanatory variable, but GPU core count, cache hierarchy, RAM capacity, OS version, thermal state, and scheduler behavior are acknowledged as residual confounders that could not be independently controlled.

> Q4_K_M and Q8_0 results are provided as modeled projections unless `make bench-07-llamacpp` is run with matching GGUF models.

---

## Evidence Labels

| Label | Meaning |
|-------|---------|
| `measured` | Directly timed using `time.perf_counter()` with explicit `mps.synchronize()` |
| `analytical` | Computed from model architecture formula (e.g., KV-cache size) |
| `estimated` | Architecture proportions calibrated to measured PTL; not kernel-profiled |
| `modeled` | Projected from measured F16 baseline via bandwidth-scaling assumptions |

---

## Hardware

| Device | Chip | RAM | Memory Bandwidth | GPU Cores | OS | Backend |
|--------|------|-----|-----------------|-----------|-----|---------|
| Mac M4 | Apple M4 | 16 GB | 120 GB/s | 10 | macOS 15 | MPS |
| Mac M2 | Apple M2 | 8 GB | 100 GB/s | 8 | macOS 14 | MPS |

> M4 and M2 differ in more than memory bandwidth. GPU core count, cache hierarchy, RAM, and OS version are residual confounders.

---

## Model

**TinyLlama-1.1B-Chat-v1.0** — `TinyLlama/TinyLlama-1.1B-Chat-v1.0` on HuggingFace

- 1.1B parameters · 22 layers · 32 attention heads · 4 KV heads (GQA) · float16
- Fits within 8 GB RAM; preserves LLaMA-family KV-cache growth and bandwidth behavior
- Absolute latency values are lower than 7B models; relative M4/M2 trends generalize directionally

---

## Key Findings

| Metric | Mac M4 16GB | Mac M2 8GB | Evidence |
|--------|------------|------------|----------|
| TTFT @ 128-token prompt | 26.9 ms | 45.0 ms | measured |
| PTL @ 512-token context | 30.1 ms | 45.8 ms | measured |
| Throughput @ 128 tokens | ~44 tok/s | ~29 tok/s | measured |
| Cold start (model load) | ~3.6 s | ~4.3 s | measured |
| p99 PTL above median (n=50) | +33% | +30% | measured |
| Q4_K_M speedup vs F16 @ 512 ctx | ~2.2× | ~2.0× | **modeled** |

- PTL grows **152%** from context 32→1024 tokens due to KV-cache traffic (measured)
- Estimated effective bandwidth utilization: **~82%** of M4 ceiling at long contexts

---

## Quick Reproduction

```bash
git clone https://github.com/Vinayk393/CECS530-TokenGenerationLatency.git
cd CECS530-TokenGenerationLatency

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

make smoke
make graphs
python -m pytest tests/
```

Run the full benchmark suite:

```bash
make bench PEAK_BW=120
```

For Mac M2:

```bash
make bench PEAK_BW=100
```

Run a single sample benchmark:

```bash
python benchmarks/02_per_token_latency_vs_context.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

---

## Repository Structure

```
CECS530-TokenGenerationLatency/
├── benchmarks/
│   ├── utils.py                           # Shared utilities
│   ├── run_smoke_test.py                  # Quick grader verification
│   ├── 01_ttft_vs_prompt_length.py        # [measured]
│   ├── 02_per_token_latency_vs_context.py # [measured]
│   ├── 03_e2e_latency_vs_output_length.py # [measured]
│   ├── 04_throughput_vs_prompt_length.py  # [measured]
│   ├── 05_inter_token_latency_timeline.py # [measured]
│   ├── 06_cold_vs_warm_run.py             # [measured]
│   ├── 07_quantization_speedup.py         # [F16 measured; Q4/Q8 modeled]
│   ├── 08_kvcache_size_vs_context.py      # [analytical]
│   └── 09_latency_decomposition.py        # [estimated]
├── analysis/
│   └── generate_research_graphs.py
├── results/
│   ├── README.md
│   ├── Mac_M4_16GB/                       # M4 measured CSV outputs
│   └── Mac_M2_8GB/                        # M2 measured CSV outputs
├── graphs/
│   └── README.md
├── tests/
│   ├── test_kvcache_formula.py            # Unit tests: KV-cache math
│   ├── test_result_schema.py              # Unit tests: CSV schema + directories
│   └── test_smoke.py                      # Unit tests: repo structure
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

## Running Benchmarks

```bash
make bench PEAK_BW=120          # M4 — full suite
make bench PEAK_BW=100          # M2 — full suite
make smoke                      # Quick smoke test only
make verify                     # Smoke test + pytest
make graphs                     # Generate figures from existing CSVs
make help                       # All targets and options
```

---

## Quantization Note

Script 07 has two modes:

```bash
# Default: measure F16, project Q4/Q8 via bandwidth scaling (labeled: modeled)
make bench-07-modeled

# Measured: requires llama.cpp + GGUF model files (labeled: measured)
make bench-07-llamacpp
```

Q8_0 and Q4_K_M speedup values are **modeled** projections from measured F16 PTL unless the llamacpp path is used with real GGUF files.

---

## Limitations

- M4 and M2 differ in more than bandwidth; bandwidth is the **primary explanatory variable**, not the only causal factor
- Q8_0 and Q4_K_M speedups are **modeled** unless GGUF models are benchmarked via `llama.cpp`
- MPS does not expose hardware performance counters; latency decomposition is **estimated**
- Absolute latencies vary with OS version, thermal state, and background processes
- TinyLlama-1.1B absolute values do not generalize to 7B+ models; directional trends do

---

## Third-Party Credits

- **[PyTorch](https://pytorch.org/)** — tensor execution and MPS backend
- **[HuggingFace Transformers](https://github.com/huggingface/transformers)** — model loading and generation
- **[TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)** — benchmark model
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** — optional GGUF quantized benchmarking
- **NumPy, pandas, matplotlib, seaborn** — analysis and visualization
- **pytest** — unit testing

---

## References

- Touvron et al., "LLaMA: Open and Efficient Foundation Language Models," arXiv 2023
- Grattafiori et al., "The Llama 3 Herd of Models," arXiv 2024
- Gerganov, "llama.cpp," GitHub, 2023
- Williams et al., "Roofline: An Insightful Visual Performance Model," CACM 2009
- Dettmers & Zettlemoyer, "The Case for 4-bit Precision," ICML 2023
- Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention," SOSP 2023
- Zhang et al., "TinyLlama: An Open-Source Small Language Model," arXiv 2024
