# Results Directory

This directory contains saved CSV outputs used to generate the paper figures.

- `Mac_M4_16GB/`: benchmark outputs collected on Apple M4 16 GB.
- `Mac_M2_8GB/`: benchmark outputs collected on Apple M2 8 GB.

Each CSV is labeled as one of:

- `measured`: directly timed using `perf_counter` with MPS synchronization.
- `analytical`: computed from model architecture.
- `modeled`: projected from measured F16 baselines.
- `estimated`: calibrated architectural estimate, not kernel-profiled.

To regenerate figures from these CSVs:

```bash
make graphs
```

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
| `smoke_test.json` | `benchmarks/run_smoke_test.py` | measured |
