# Reproducibility Guide

Complete step-by-step commands to regenerate every CSV, figure, and result
from scratch. All paths are repo-relative; no absolute paths required.

---

## Prerequisites

- Python 3.11 (3.10+ works)
- macOS 14+ with Apple Silicon (M1/M2/M3/M4) for MPS-accurate results
- ~4 GB free disk (model ~2.2 GB + results)

---

## Step 1 — Clone and install

```bash
git clone https://github.com/Vinayk393/CECS530-TokenGenerationLatency.git
cd CECS530-TokenGenerationLatency

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

---

## Step 2 — Run unit tests (no model download)

```bash
python -m pytest tests/ -v
```

Formula tests and schema tests run instantly. Smoke test requires Step 3.

---

## Step 3 — Sample test case (smoke test)

Downloads TinyLlama-1.1B (~2.2 GB) on first run.

```bash
make smoke
# or: python benchmarks/run_smoke_test.py
```

Expected output on M4 16GB:
```
  TTFT:       ~16–27 ms
  Mean PTL:   ~19–22 ms
  Throughput: ~45–52 tok/s
[PASS] Smoke test complete.
Output → results/smoke_test.json
```

Expected output on M2 8GB:
```
  TTFT:       ~27–45 ms
  Mean PTL:   ~28–35 ms
  Throughput: ~28–36 tok/s
```

---

## Step 4 — Generate figures from submitted CSVs (no download needed)

```bash
make graphs
# or: python analysis/generate_research_graphs.py
```

Verify CSVs exist first:
```bash
python analysis/generate_research_graphs.py --check_only
```

---

## Step 5 — Full benchmark suite

On M4 (120 GB/s):
```bash
make bench PEAK_BW=120
```

On M2 (100 GB/s):
```bash
make bench PEAK_BW=100
```

Individual scripts:
```bash
python benchmarks/01_ttft_vs_prompt_length.py
python benchmarks/02_per_token_latency_vs_context.py
python benchmarks/08_kvcache_size_vs_context.py   # analytical, no GPU
```

---

## Step 6 — Verify everything

```bash
make verify   # smoke test + pytest
```

---

## Methodology alignment with paper

| Paper claim (§4.3) | Code |
|--------------------|------|
| TTFT: "one full forward pass" | `model(input_ids=..., use_cache=True)` (not `generate()`) |
| PTL exclude: "First 2 tokens" | `N_TRANSIENT_SKIP=2; if i >= 2: samples.append(...)` |
| PTL window: "20 tokens" | `N_DECODE_TOKENS=20` |
| Timer: "mps.synchronize() before perf_counter()" | `sync_device()` wraps every timer event |

---

## Evidence labels

| Label | Scripts | Meaning |
|-------|---------|---------|
| `measured` | 01–06, smoke | Direct timing with perf_counter + mps.synchronize() |
| `analytical` | 08 | KV-cache formula M_KV = 2·L·H_kv·d_head·n_ctx·b |
| `estimated` | 09 | Architecture proportions calibrated to measured PTL |
| `modeled` | 07 | Bandwidth-scaling projection from measured F16 baseline |

---

## Quantization: measured vs modeled

Default (`bench-07-modeled`): measures F16 PTL directly, projects Q4/Q8 via
bandwidth scaling. Output labeled `measurement_type=modeled`.

For measured Q4/Q8 speedup, install llama.cpp and provide GGUF files:
```bash
make bench-07-llamacpp
```
