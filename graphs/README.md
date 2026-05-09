# Graphs

Nine publication-quality PNGs (180 DPI) generated from CSV results in `results/`.

Run `make graphs` to regenerate all figures from submitted CSV data:
```bash
make graphs
# or directly:
python analysis/generate_research_graphs.py
```

## Figure Index

| File | Paper Figure | Paper Section | Evidence type |
|------|-------------|---------------|---------------|
| `01_ttft_vs_prompt_length.png` | Figure 2 | Section 5.1 | measured |
| `02_ptl_vs_context_length.png` | Figure 3 | Section 5.2 | measured |
| `03_inter_token_timeline.png` | Figure 4 | Section 5.3 | measured |
| `04_latency_decomposition.png` | Figure 6 | Section 5.5 | estimated |
| `05_quantization_speedup.png` | Figure 8 | Section 5.7 | F16 measured; Q4/Q8 modeled |
| `06_cold_vs_warm.png` | Figure 7 | Section 5.6 | measured |
| `07_latency_variance_distribution.png` | Figure 9 | Section 5.8 | measured |
| `08_model_scaling.png` | Figure 10 | Section 5.9 | TinyLlama measured; 1B/3B modeled |
| `09_cross_platform_summary.png` | Figure 5 | Section 5.4 | measured |

## Evidence Labels

| Label | Meaning |
|-------|---------|
| `measured` | Directly timed with `perf_counter` + `mps.synchronize()` |
| `estimated` | Architecture proportions calibrated to measured PTL |
| `modeled` | Projected from measured F16 baseline via bandwidth scaling |