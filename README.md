# Token-Generation Latency Benchmarking in LLaMA
## CECS530-TokenGenerationLatency
### Mac M4 16GB vs Mac M2 8GB — Apple Silicon Unified Memory Study

![CI](https://github.com/Vinayk393/CECS530-TokenGenerationLatency/actions/workflows/ci.yml/badge.svg)

---

## Overview

This project benchmarks token-generation latency in LLaMA-family models across
two Apple Silicon devices: Mac M4 16GB and Mac M2 8GB. Model, backend, batch
size, benchmark scripts, and software configuration are held constant. Memory
bandwidth is the primary explanatory variable; GPU core count, cache hierarchy,
RAM capacity, OS version, and thermal state are acknowledged as residual confounders.
Project metadata (model, hardware, software versions) is recorded in `metadata.json`.

The main result is that M4 shows an observed **~1.5× per-token latency advantage**
over M2 under this benchmark setup (paper Section 5.2, Table 5).

> **Quantization:** Q4_K_M and Q8_0 results are **modeled projections** from
> measured F16 baselines using bandwidth scaling. Use `make bench-07-llamacpp`
> with GGUF files for physically measured quantization speedup.

---

## Evidence Labels

| Label | Meaning |
|-------|---------|
| `measured` | Directly timed with `time.perf_counter()` + `mps.synchronize()` |
| `analytical` | Computed from architecture formula (paper Eq. 4) |
| `estimated` | Architecture proportions calibrated to measured PTL; not kernel-profiled |
| `modeled` | Projected from measured F16 baseline via bandwidth-scaling assumptions |

> Evidence labels are schema-validated by `pytest tests/test_result_schema.py` —
> run `pytest tests/` to verify all CSVs carry correct `measurement_type` values.

---

## Hardware

| Device | Chip | RAM | Bandwidth | GPU Cores | OS | Backend |
|--------|------|-----|-----------|-----------|-----|---------|
| Mac M4 | Apple M4 | 16 GB | 120 GB/s | 10 | macOS 15 | MPS |
| Mac M2 | Apple M2 | 8 GB | 100 GB/s | 8 | macOS 14 | MPS |

---

## Key Findings (Paper Tables 4, 5 + Result Sections)

| Metric | M4 16GB | M2 8GB | Evidence |
|--------|---------|--------|----------|
| TTFT @ 128-tok prompt | 26.9 ms | 45.0 ms | measured |
| PTL @ 512-tok context | 30.1 ms | 45.8 ms | measured |
| Throughput @ 128-tok | ~44 tok/s | ~29 tok/s | measured |
| Cold start (load) | ~3.6 s | ~4.3 s | measured |
| p99 PTL vs median (n=50) | +33% | +30% | measured |
| Q4_K_M speedup @ 512-tok | ~2.2× | ~2.0× | **modeled** |

---

## Quick Reproduction

Install dependencies:
```bash
pip install -r requirements.txt
```

Generate all paper figures from the submitted CSV files (no model download):
```bash
make graphs
```

Run the full benchmark suite on the current machine:
```bash
make bench PEAK_BW=120 DEVICE=Mac_M4_16GB   # on M4
make bench PEAK_BW=100 DEVICE=Mac_M2_8GB    # on M2
```

Run a single sample benchmark:
```bash
python benchmarks/02_per_token_latency_vs_context.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Reproduce quantization results with physical measurement (replaces modeled projection):
```bash
bash download_gguf_models.sh                               # ~4 GB: F16 + Q8_0 + Q4_K_M
make bench-07-llamacpp GGUF_DIR=models/ DEVICE=Mac_M4_16GB
```
Output: `results/Mac_M4_16GB/07_quantization_llamacpp.csv` with `measurement_type=measured`.
Requires `llama-bench` on PATH or at `models/llama-bench`
(build from https://github.com/ggerganov/llama.cpp).

Run tests:
```bash
pytest tests/
```

Full setup from scratch:
```bash
git clone https://github.com/Vinayk393/CECS530-TokenGenerationLatency.git
cd CECS530-TokenGenerationLatency

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

make smoke                                   # sample test case (~2.2 GB download first run)
make graphs                                  # figures from submitted CSVs (no download needed)
pytest tests/                                # unit tests
make bench PEAK_BW=120 DEVICE=Mac_M4_16GB   # full suite on M4
make bench PEAK_BW=100 DEVICE=Mac_M2_8GB    # full suite on M2
```

Expected outputs:
- `results/smoke_test.json` — smoke test result
- `results/Mac_M4_16GB/*.csv` — benchmark CSVs
- `graphs/*.png` — 9 publication-quality figures

See `REPRODUCIBILITY.md` for complete step-by-step guide.

---

## Methodology Alignment with Paper

### TTFT (Section 4.5)
Timed as a **raw prefill forward pass** — `model(input_ids=..., use_cache=True)` — not `model.generate()`, to avoid HuggingFace generation overhead contaminating the prefill measurement.

### PTL (Section 4.3, 4.5)
Decode loop runs `N_TRANSIENT_SKIP + N_DECODE_TOKENS = 22` steps. Samples collected only from `i >= N_TRANSIENT_SKIP = 2` (paper: "First 2 tokens excluded to eliminate transients").

```python
for i in range(n_tokens + 2):    # +2 for excluded transients
    torch.mps.synchronize()
    t0 = time.perf_counter()
    out = model(input_ids=next_tok, past_key_values=past_kv, use_cache=True)
    torch.mps.synchronize()
    ptl_ms = (time.perf_counter() - t0) * 1000
    if i >= 2:                    # skip first two transient tokens
        samples.append(ptl_ms)
```

---

## Repository Structure

```
CECS530-TokenGenerationLatency/
├── benchmarks/
│   ├── utils.py                           # Shared: device, sync, I/O, KV formula
│   ├── run_smoke_test.py                  # Sample test case (rubric §3.1)
│   ├── 01_ttft_vs_prompt_length.py        # [measured] raw prefill timing
│   ├── 02_per_token_latency_vs_context.py # [measured] skips first 2 tokens
│   ├── 03_e2e_latency_vs_output_length.py # [measured]
│   ├── 04_throughput_vs_prompt_length.py  # [measured]
│   ├── 05_inter_token_latency_timeline.py # [measured]
│   ├── 06_cold_vs_warm_run.py             # [measured]
│   ├── 07_quantization_speedup.py         # [F16 measured; Q4/Q8 modeled]
│   ├── 08_kvcache_size_vs_context.py      # [analytical — no GPU needed]
│   └── 09_latency_decomposition.py        # [estimated]
├── analysis/
│   └── generate_research_graphs.py        # Repo-relative paths only
├── results/
│   ├── README.md                          # Evidence label reference
│   ├── Mac_M4_16GB/                       # M4 benchmark CSV outputs
│   └── Mac_M2_8GB/                        # M2 benchmark CSV outputs
├── graphs/                                # 9 publication-quality PNGs (180 DPI)
├── tests/
│   ├── test_kvcache_formula.py            # Paper Eq. 4+5, PTL methodology
│   ├── test_result_schema.py              # CSV evidence labels
│   └── test_smoke.py                      # Repo structure + path independence
├── docs/
│   └── reproducibility.md
├── optimization/
│   └── kv_cache_proposal.md
├── report/
│   └── findings_report.md
├── workflows/
│   ├── ci.yml                             # CI: runs pytest on push
│   └── verify.yml                         # Verifies CSVs + graph generation
├── README.md
├── REPRODUCIBILITY.md
├── requirements.txt
├── Makefile
├── metadata.json                          # Project metadata (model, hw, sw versions)
├── download_gguf_models.sh                # Downloads GGUF files for bench-07-llamacpp
├── models/                                # GGUF files land here (git-ignored, ~4 GB)
├── CITATION.cff
└── LICENSE
```

---

## Limitations

- M4 and M2 differ beyond bandwidth; bandwidth is the **primary explanatory variable**
- Q8_0 and Q4_K_M speedups are **modeled** unless benchmarked via llama.cpp + GGUF
- MPS lacks hardware performance counters; decomposition is **estimated**
- TinyLlama absolute values do not generalize to 7B+ models; trends do

---

## Third-Party Credits

- **[PyTorch](https://pytorch.org/)** — tensor execution, MPS backend
- **[HuggingFace Transformers](https://github.com/huggingface/transformers)** — model loading
- **[TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)** — model
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** — optional GGUF benchmarking
- **NumPy, pandas, matplotlib, seaborn** — analysis and visualization
- **pytest** — unit testing

---

## References

See paper `report/Adv_Arc_Final_Report.pdf` for full references [1]–[21].