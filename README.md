# Token-Generation Latency Benchmarking in LLaMA
# CECS530-TokenGenerationLatency
### Mac M4 16GB vs Mac M2 8GB — Apple Silicon Unified Memory Study


---

## Overview

This project benchmarks token-generation latency in LLaMA-family models across two Apple Silicon devices — Mac M4 (16GB, 120 GB/s) and Mac M2 (8GB, 100 GB/s) — using the MPS backend. The central finding is that autoregressive LLM decode is **memory-bandwidth-bound**, and the 20 GB/s bandwidth gap between M4 and M2 produces a consistent ~1.5× per-token latency advantage that quantization can further reduce by up to 2.3×.

No prior study directly compares two Apple Silicon chips within the same unified-memory architecture family for LLM inference latency.

---

## Hardware

| Device       | Chip      | RAM  | Memory Bandwidth | Backend |
|--------------|-----------|------|-----------------|---------|
| Mac M4       | Apple M4  | 16GB | 120 GB/s         | MPS     |
| Mac M2       | Apple M2  | 8GB  | 100 GB/s         | MPS     |

---

## Model

**TinyLlama-1.1B-Chat-v1.0** (HuggingFace: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)

- 1.1B parameters · 22 layers · 32 attention heads · 4 KV heads (GQA) · float16
- Chosen to fit comfortably within 8GB RAM while preserving transformer inference behavior
- All latency trends generalize directionally to larger LLaMA-family models

---

## Key Findings

| Metric | Mac M4 16GB | Mac M2 8GB | M2/M4 Ratio |
|--------|------------|------------|------------|
| TTFT @ 128 tokens | 26.9 ms | 45.0 ms | 1.67× |
| PTL @ 512 context | 30.1 ms | 45.8 ms | 1.52× |
| Throughput @ 128 tokens | ~44 tok/s | ~29 tok/s | — |
| Cold start (model load) | ~3.6 s | ~4.3 s | — |
| Q4_K_M speedup vs F16 | 2.2× | 2.0× | — |

- PTL grows **152%** from context 32→1024 tokens due to KV-cache memory traffic
- Memory bandwidth utilization reaches **~82%** of peak at long contexts on M4
- M2 8GB approaches RAM pressure at ~1,950 tokens with F16 — Q4_K_M strongly recommended
- p99 latency is **+33% above median** — tail latency matters for real-time applications

---

## Repository Structure

```
llama-latency-m4-vs-m2/
├── benchmarks/                   # 9 runnable benchmark scripts
│   ├── 01_ttft_vs_prompt_length.py
│   ├── 02_per_token_latency_vs_context.py
│   ├── 03_e2e_latency_vs_output_length.py
│   ├── 04_throughput_vs_prompt_length.py
│   ├── 05_inter_token_latency_timeline.py
│   ├── 06_cold_vs_warm_run.py
│   ├── 07_quantization_speedup.py
│   ├── 08_kvcache_size_vs_context.py
│   └── 09_latency_decomposition.py
│
├── analysis/
│   └── generate_research_graphs.py   # produces all 9 graphs from CSVs
│
├── results/
│   ├── Mac_M4_16GB/                  # 17 CSVs from M4 benchmark runs
│   └── Mac_M2_8GB/                   # 17 CSVs from M2 benchmark runs
│
├── graphs/                           # 9 publication-quality PNGs (180 DPI)
│
├── optimization/
│   └── kv_cache_proposal.md          # written optimization proposal
│
├── report/
│   └── findings_report.md            # complete findings with tables
│
├── docs/
│   └── apple_silicon_architecture.md # unified memory architecture analysis
│
├── README.md
├── requirements.txt
├── .gitignore
└── Makefile
```

---

## Setup

```bash
git clone https://github.com/<your-username>/llama-latency-m4-vs-m2.git
cd llama-latency-m4-vs-m2

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Running Benchmarks

Run all benchmarks in order (uses TinyLlama by default, auto-downloads ~2.2GB):

```bash
make bench
```

Run a single benchmark:

```bash
python benchmarks/01_ttft_vs_prompt_length.py
python benchmarks/05_inter_token_latency_timeline.py --prompt_lengths 128 512
```

Use a different model:

```bash
python benchmarks/01_ttft_vs_prompt_length.py --model meta-llama/Llama-3.2-1B
```

Pass your device's peak bandwidth for accurate BW utilization calculation:

```bash
python benchmarks/04_throughput_vs_prompt_length.py --peak_bw 120   # M4
python benchmarks/04_throughput_vs_prompt_length.py --peak_bw 100   # M2
```

---

## Generating Graphs

```bash
make graphs
```

Or directly:

```bash
python analysis/generate_research_graphs.py
```

Graphs are saved to `graphs/` as 180 DPI PNGs, ready for presentation.

---

## Results CSV Reference

Every CSV in `results/Mac_M4_16GB/` includes a `measurement_type` or `data_source` column:

| Value | Meaning |
|---|---|
| `measured` | Directly timed with `perf_counter` + MPS sync |
| `derived_from_bandwidth_scaling_model` | Estimated from bandwidth ratios |
| `estimated_from_arch_proportions_calibrated_to_observed_ptl` | Decomposition proportions, not hook-measured |
| `analytical` | Calculated from model architecture (KV-cache sizing) |

See `results/Mac_M4_16GB/12_experimental_design.csv` for full parameter documentation.

---

## Optimization

KV-cache quantization is the most impactful single optimization for Apple Silicon:

```bash
# With llama.cpp (recommended):
llama-bench -m model-Q4_K_M.gguf -p 512 -n 128 -r 5
```

Expected speedup: **~2.2× on M4**, **~2.0× on M2** at 512-token context.  
See `optimization/kv_cache_proposal.md` for full analysis.

---

## References

- Touvron et al., "LLaMA: Open and Efficient Foundation Language Models," 2023
- Grattafiori et al., "The Llama 3 Herd of Models," 2024
- Gerganov, "llama.cpp," GitHub, 2023
- Williams et al., "Roofline: An Insightful Visual Performance Model," CACM 2009
- Dettmers & Zettlemoyer, "The Case for 4-bit Precision," ICML 2023
