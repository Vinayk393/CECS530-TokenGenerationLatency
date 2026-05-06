# Graphs

Publication-quality figures generated from benchmark CSV outputs.

## Generating Figures

```bash
# Generate all figures from existing CSVs:
python analysis/generate_research_graphs.py

# Verify required CSVs exist before generating:
python analysis/generate_research_graphs.py --check_only

# Via Makefile:
make graphs
make verify
```

Figures are saved as 180 DPI PNGs suitable for inclusion in the report.

---

## Figure Reference

| File | Report Figure | Script | Data Source |
|------|--------------|--------|-------------|
| `fig1_ttft_vs_prompt_length.png` | Figure 1 | `01_ttft_vs_prompt_length.py` | measured |
| `fig2_ptl_vs_context.png` | Figure 2 | `02_per_token_latency_vs_context.py` | measured |
| `fig3_inter_token_timeline.png` | Figure 3 | `05_inter_token_latency_timeline.py` | measured |
| `fig4_throughput_vs_prompt.png` | Figure 4 | `04_throughput_vs_prompt_length.py` | measured |
| `fig5_latency_decomposition.png` | Figure 5 | `09_latency_decomposition.py` | estimated |
| `fig6_cold_warm_run.png` | Figure 6 | `06_cold_vs_warm_run.py` | measured |
| `fig7_quantization_speedup.png` | Figure 7 | `07_quantization_speedup.py` | modeled (F16: measured) |
| `fig8_tail_latency.png` | Figure 8 | `02_per_token_latency_vs_context.py` | measured (n=50) |
| `fig9_model_scaling.png` | Figure 9 | `09_latency_decomposition.py` | TinyLlama measured; 1B/3B modeled |

---

## Evidence Labels in Figures

Figures that display modeled or estimated data include `[modeled]` or
`[estimated]` in their title or caption. Figures from directly timed
measurements include `[measured]`.

See `results/README.md` for the full evidence label reference.
