# Results Directory

This directory contains benchmark output files from:
- `Mac_M4_16GB/` — Apple M4 (16 GB, 120 GB/s) benchmark runs
- `Mac_M2_8GB/`  — Apple M2 (8 GB, 100 GB/s) benchmark runs
- `smoke_test.json` — output from `run_smoke_test.py`

---

## Evidence Labels

Every CSV and JSON in this directory includes a `measurement_type` or
`data_source` column. **Do not compare modeled values against measured
values without noting the evidence label difference.**

| Value | Meaning | Scripts |
|-------|---------|---------|
| `measured` | Directly timed with `time.perf_counter()` + `mps.synchronize()` before each checkpoint | 01–06, smoke test |
| `analytical` | Computed from model architecture formula (no timing involved) | 08 |
| `estimated` | Architecture-level proportions calibrated to measured PTL; not kernel-profiled | 09 (decomposition) |
| `modeled` | Projected from measured F16 baseline using bandwidth-scaling assumptions | 07 (default mode) |
| `derived_from_bandwidth_scaling_model` | Legacy label — equivalent to `modeled` | some older CSVs |
| `estimated_from_arch_proportions_calibrated_to_observed_ptl` | Legacy label — equivalent to `estimated` | some older CSVs |

---

## File Reference

| File | Script | Evidence |
|------|--------|----------|
| `01_ttft_vs_prompt_length.csv` | `benchmarks/01_ttft_vs_prompt_length.py` | measured |
| `02_per_token_latency_vs_context.csv` | `benchmarks/02_per_token_latency_vs_context.py` | measured |
| `03_e2e_latency_vs_output_length.csv` | `benchmarks/03_e2e_latency_vs_output_length.py` | measured |
| `04_throughput_vs_prompt_length.csv` | `benchmarks/04_throughput_vs_prompt_length.py` | measured |
| `05_inter_token_latency_timeline.csv` | `benchmarks/05_inter_token_latency_timeline.py` | measured |
| `06_cold_vs_warm_run.csv` | `benchmarks/06_cold_vs_warm_run.py` | measured |
| `07_quantization_speedup.csv` | `benchmarks/07_quantization_speedup.py` | **modeled** (F16 row: measured) |
| `08_kvcache_size_vs_context.csv` | `benchmarks/08_kvcache_size_vs_context.py` | analytical |
| `09_latency_decomposition.csv` | `benchmarks/09_latency_decomposition.py` | estimated |

---

## CSV Schema

Every CSV should include these columns at minimum:

| Column | Type | Description |
|--------|------|-------------|
| `measurement_type` or `data_source` | string | Evidence label (see table above) |
| metric column (e.g., `ptl_ms`, `ttft_ms`) | float | The measured or derived value |
| `device_name` | string | Hardware label (e.g., `Mac_M4_16GB`) |
| `model` | string | HuggingFace model ID |
| `precision` | string | `float16`, `Q4_K_M`, etc. |

---

## Reproducing Results

```bash
# Run all benchmarks (downloads ~2.2 GB model on first run)
make bench DEVICE=Mac_M4_16GB PEAK_BW=120

# On M2:
make bench DEVICE=Mac_M2_8GB PEAK_BW=100

# Quick smoke test only:
python benchmarks/run_smoke_test.py
```

See `docs/reproducibility.md` for full environment details and known limitations.
