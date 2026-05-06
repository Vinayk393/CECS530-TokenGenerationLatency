# Token-Generation Latency Benchmarking in LLaMA
## CECS530-TokenGenerationLatency
### Mac M4 16GB vs Mac M2 8GB — Apple Silicon Unified Memory Study

---

## Overview

This project benchmarks token-generation latency in LLaMA-family models across two Apple Silicon devices: Mac M4 (16GB, 120 GB/s) and Mac M2 (8GB, 100 GB/s). The workload, model family, batch size, backend, and benchmark scripts are held constant to study how Apple Silicon memory bandwidth and related hardware-generation factors affect autoregressive decode latency.

The main result is that M4 shows an observed **~1.5× per-token latency advantage** over M2 under this benchmark setup. Memory bandwidth is the primary explanatory variable, but GPU core count, cache hierarchy, RAM capacity, OS version, thermal state, and scheduler behavior are acknowledged as residual confounders that could not be independently controlled.

This project also evaluates KV-cache quantization as an architectural optimization: measuring the decode bottleneck, then proposing the lowest-risk intervention that directly reduces memory traffic.

---

## Hardware

| Device   | Chip      | RAM  | Memory Bandwidth | GPU Cores | Backend |
|----------|-----------|------|-----------------|-----------|---------|
| Mac M4   | Apple M4  | 16GB | 120 GB/s        | 10        | MPS     |
| Mac M2   | Apple M2  | 8GB  | 100 GB/s        | 8         | MPS     |

> **Note:** M4 and M2 differ in more than memory bandwidth. GPU core count, cache hierarchy, RAM capacity, and OS version (macOS 15 vs 14) are not independently controlled. Bandwidth is treated as the primary explanatory variable throughout.

---

## Model

**TinyLlama-1.1B-Chat-v1.0** (`TinyLlama/TinyLlama-1.1B-Chat-v1.0` on HuggingFace)

- 1.1B parameters · 22 layers · 32 attention heads · 4 KV heads (GQA) · float16
- Chosen to fit comfortably within 8GB RAM while preserving the transformer architecture and KV-cache growth behavior of the LLaMA family
- Absolute latency values are lower than 7B models; relative M4/M2 trends generalize directionally

---

## Evidence Labels

This repository separates direct measurements from analytical or model-based values. Every result CSV and script output carries an explicit label.

| Label | Meaning |
|-------|---------|
| `measured` | Directly timed using `time.perf_counter()` with explicit `mps.synchronize()` before each checkpoint |
| `analytical` | Computed from model architecture (e.g., KV-cache size formula) |
| `estimated` | Derived from architecture-level proportions calibrated to measured PTL; not kernel-profiled |
| `modeled` | Projected from measured F16 baselines using bandwidth-scaling assumptions |

**Q8\_0 and Q4\_K\_M speedups are `modeled`** unless GGUF model files are supplied and benchmarked through the `llama.cpp` path (see `bench-07-llamacpp`).

---

## Key Findings

| Metric | Mac M4 16GB | Mac M2 8GB | M2/M4 Ratio | Evidence |
|--------|------------|------------|------------|----------|
| TTFT @ 128 tokens | 26.9 ms | 45.0 ms | 1.67× | measured |
| PTL @ 512 context | 30.1 ms | 45.8 ms | 1.52× | measured |
| Throughput @ 128 tokens | ~44 tok/s | ~29 tok/s | — | measured |
| Cold start (model load) | ~3.6 s | ~4.3 s | — | measured |
| Q4\_K\_M speedup vs F16, modeled from F16 baseline | ~2.2× | ~2.0× | — | modeled |

- PTL grows **152%** from context 32→1024 tokens due to KV-cache memory traffic (measured)
- Estimated effective bandwidth utilization reaches **~82%** of M4's 120 GB/s ceiling at long contexts (derived)
- M2 8GB approaches system-level RAM pressure at ~1,950 tokens with F16 — Q4\_K\_M shifts this to ~7,800 tokens (analytical estimate; system RAM includes model weights, runtime, Metal pools, OS overhead)
- p99 latency is **+33% above median** on M4 at 512-token context (measured, n=50 trials)

---

## Repository Structure

```
CECS530-TokenGenerationLatency/
├── benchmarks/
│   ├── utils.py                          # Shared utilities (device, sync, I/O, seed)
│   ├── run_smoke_test.py                 # Quick grader verification test
│   ├── 01_ttft_vs_prompt_length.py
│   ├── 02_per_token_latency_vs_context.py
│   ├── 03_e2e_latency_vs_output_length.py
│   ├── 04_throughput_vs_prompt_length.py
│   ├── 05_inter_token_latency_timeline.py
│   ├── 06_cold_vs_warm_run.py
│   ├── 07_quantization_speedup.py        # modeled unless llama.cpp GGUF path used
│   ├── 08_kvcache_size_vs_context.py     # analytical; no GPU needed
│   └── 09_latency_decomposition.py
│
├── analysis/
│   └── generate_research_graphs.py       # generates all figures from CSVs
│
├── results/
│   ├── README.md                         # evidence label reference
│   ├── Mac_M4_16GB/                      # CSVs from M4 benchmark runs
│   └── Mac_M2_8GB/                       # CSVs from M2 benchmark runs
│
├── graphs/
│   └── README.md                         # figure-to-script reference
│
├── tests/
│   ├── test_kvcache_formula.py           # unit test: KV-cache math
│   ├── test_result_schema.py             # unit test: CSV schema
│   └── test_smoke.py                     # unit test: smoke output exists
│
├── docs/
│   └── reproducibility.md               # environment notes and known limitations
│
├── optimization/
│   └── kv_cache_proposal.md             # written optimization proposal
│
├── report/
│   └── findings_report.md               # complete findings with tables
│
├── README.md
├── requirements.txt
├── Makefile
├── CITATION.cff
├── LICENSE
└── .gitignore
```

---

## Minimal Reproduction Path

```bash
git clone https://github.com/Vinayk393/CECS530-TokenGenerationLatency.git
cd CECS530-TokenGenerationLatency

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Quick verification (no model download needed for schema/formula tests):
python -m pytest tests/

# Smoke test (downloads ~2.2GB TinyLlama on first run):
python benchmarks/run_smoke_test.py

# Full benchmark suite:
make bench

# Generate all graphs from CSVs:
make graphs
```

**Expected outputs:**
- Benchmark CSVs in `results/<device_name>/` (e.g., `results/Mac_M4_16GB/`)
- Generated figures in `graphs/`
- Console summary showing: device, model, precision, measurement type, output path
- `results/smoke_test.json` from the smoke test

---

## Running Benchmarks

Run all benchmarks (downloads TinyLlama ~2.2GB on first run):

```bash
make bench
```

Run individual benchmarks:

```bash
python benchmarks/01_ttft_vs_prompt_length.py --device_name Mac_M4_16GB
python benchmarks/02_per_token_latency_vs_context.py --device_name Mac_M4_16GB
python benchmarks/05_inter_token_latency_timeline.py --prompt_lengths 128 512
```

Use a different model:

```bash
python benchmarks/01_ttft_vs_prompt_length.py --model meta-llama/Llama-3.2-1B --device_name Mac_M4_16GB
```

Set device peak bandwidth for BW utilization calculation:

```bash
python benchmarks/04_throughput_vs_prompt_length.py --peak_bw 120 --device_name Mac_M4_16GB  # M4
python benchmarks/04_throughput_vs_prompt_length.py --peak_bw 100 --device_name Mac_M2_8GB   # M2
```

---

## Quantization Benchmarking

Script 07 has two modes:

```bash
# Modeled (default) — projects Q4/Q8 speedup from measured F16 baseline:
make bench-07-modeled

# Measured via llama.cpp (requires GGUF files):
make bench-07-llamacpp GGUF_DIR=models/
# or directly:
python benchmarks/07_quantization_speedup.py \
    --backend llamacpp \
    --llama_bench /path/to/llama-bench \
    --models Q4=/path/to/model-Q4_K_M.gguf Q8=/path/to/model-Q8_0.gguf F16=/path/to/model-F16.gguf
```

Q8\_0 and Q4\_K\_M speedups in the paper are **modeled** projections from F16 unless the `llamacpp` path is used with real GGUF files.

---

## Generating Graphs

```bash
make graphs
```

Or with verification (checks required CSVs exist before plotting):

```bash
python analysis/generate_research_graphs.py --check_only   # verify inputs
python analysis/generate_research_graphs.py                # generate all figures
```

---

## Results CSV Reference

Every CSV includes a `measurement_type` or `data_source` column. See `results/README.md` for the full reference.

| Value | Meaning |
|-------|---------|
| `measured` | Directly timed with `perf_counter` + MPS sync |
| `modeled` | Projected via bandwidth-scaling from measured F16 baseline |
| `estimated` | Architecture-based proportions calibrated to measured PTL |
| `analytical` | Computed from model architecture (KV-cache sizing formula) |

---

## Limitations

- M4 and M2 differ in more than memory bandwidth; this repository treats bandwidth as the **primary explanatory variable**, not the only causal factor.
- Q8\_0 and Q4\_K\_M speedups are **modeled** unless GGUF models are benchmarked through `llama.cpp`.
- MPS does not expose full hardware performance counters; latency decomposition is **estimated**.
- Absolute latencies vary with OS version, thermal state, background processes, and memory pressure.
- TinyLlama-1.1B absolute latency values do not generalize directly to 7B+ models; directional trends are expected to transfer.
- Five trials per TTFT configuration and 20-token PTL windows are sufficient for trend analysis but not production-grade tail characterization.

---

## Third-Party Software and Model Credits

This project uses:
- **[PyTorch](https://pytorch.org/)** for tensor execution and MPS backend support
- **[HuggingFace Transformers](https://github.com/huggingface/transformers)** for model loading and autoregressive generation
- **[TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)** as the benchmark model
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** for optional GGUF quantized-model benchmarking
- **NumPy, pandas, matplotlib, seaborn** for analysis and visualization
- **pytest** for unit testing

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
