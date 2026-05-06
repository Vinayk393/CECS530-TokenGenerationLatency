# Results Directory

`Mac_M4_16GB/` — benchmark outputs from Apple M4 16 GB (120 GB/s).
`Mac_M2_8GB/`  — benchmark outputs from Apple M2 8 GB (100 GB/s).

Each CSV includes a `measurement_type` column:

| Value | Meaning |
|-------|---------|
| `measured` | Directly timed with `perf_counter` + `mps.synchronize()` |
| `analytical` | Computed from model architecture formula (paper Eq. 4) |
| `estimated` | Architecture proportions calibrated to measured PTL |
| `modeled` | Projected from measured F16 baseline via bandwidth scaling |

Figures in `graphs/` are generated from these CSVs:
```bash
make graphs
```
