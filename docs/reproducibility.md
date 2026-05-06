# Reproducibility Notes

## Tested Environment

| Component | Value |
|-----------|-------|
| Python | 3.11 |
| macOS (M4) | macOS 15 |
| macOS (M2) | macOS 14 |
| PyTorch | 2.2.x (MPS backend) |
| HuggingFace Transformers | 4.40.x |
| Model | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| Precision | float16 |
| Batch size | 1 |
| Backend | MPS (Metal Performance Shaders) |

## Reproducing Results

### Minimal path (schema and formula tests only — no model download)
```bash
python -m pytest tests/test_kvcache_formula.py tests/test_result_schema.py -v
```

### Smoke test (~2.2 GB model download on first run)
```bash
python benchmarks/run_smoke_test.py
```

### Full benchmark suite
```bash
make bench DEVICE=Mac_M4_16GB PEAK_BW=120
make graphs
```

### On M2 (change device label and bandwidth):
```bash
make bench DEVICE=Mac_M2_8GB PEAK_BW=100
```

## Evidence Labels

Every result CSV and JSON in this repository carries a `measurement_type` field:

| Label | Meaning | Scripts |
|-------|---------|---------|
| `measured` | Directly timed with `perf_counter` + `mps.synchronize()` | 01–06, 09 |
| `analytical` | Computed from model architecture formula | 08 |
| `estimated` | Architecture proportions calibrated to measured PTL | 09 (decomposition) |
| `modeled` | Projected from measured F16 baseline via bandwidth scaling | 07 (default) |

## Known Limitations

### Hardware confounders
M4 and M2 differ in more than memory bandwidth:
- GPU cores: M4 has 10, M2 has 8
- RAM capacity: M4 has 16 GB, M2 has 8 GB
- Cache hierarchy: differs between generations
- OS: macOS 15 (M4) vs macOS 14 (M2)
- Thermal state: not controlled

Memory bandwidth is treated as the **primary explanatory variable**. The other
differences are acknowledged as residual confounders.

### MPS profiling
MPS (Metal Performance Shaders) does not expose CUDA-style per-kernel profiling
counters. Latency decomposition (script 09) is therefore estimated from
architectural proportions calibrated to measured end-to-end PTL, not
kernel-profiled. Component shares are marked `estimated` in output CSVs.

### Quantization speedup
Q8\_0 and Q4\_K\_M speedups reported in the paper are **modeled** projections
from the measured F16 baseline using bandwidth-scaling assumptions. They are not
physical benchmark measurements unless real GGUF model files are benchmarked
through the `llama.cpp` path (`bench-07-llamacpp`).

To obtain measured speedups:
1. Install [llama.cpp](https://github.com/ggerganov/llama.cpp)
2. Download GGUF variants of TinyLlama-1.1B (Q4\_K\_M, Q8\_0, F16)
3. Run: `make bench-07-llamacpp GGUF_DIR=/path/to/models`

### Variance sources
Latency measurements vary with:
- OS scheduler preemptions (not suppressed — real deployed behavior)
- Garbage collection (Python GC)
- Background processes and memory pressure
- Thermal throttling (not measured in sustained runs)

We handle these by running multiple trials and reporting mean, p90, and p99.

### Statistical coverage
- TTFT: 5 trials per configuration (mean and σ reported)
- PTL: 20-token windows with first 2 excluded (steady-state mean)
- Tail latency (p90/p99): 50-trial dedicated runs at 512-token context only

### Absolute latency generalization
TinyLlama-1.1B absolute PTL values are lower than 7B+ models. Relative M4/M2
trends and KV-cache inflection behavior generalize directionally.

## KV-Cache Formula Reference

```
M_KV = 2 × L × H_kv × d_head × context × bytes_per_element
```

For TinyLlama-1.1B (L=22, H\_kv=4, d\_head=64, F16=2 bytes):

| Context | KV-cache (F16) | KV-cache (Q4\_K\_M) |
|---------|---------------|-------------------|
| 512 tokens | 11.5 MB | 2.9 MB |
| 1024 tokens | 23.1 MB | 5.8 MB |
| 2048 tokens | 46.1 MB | 11.5 MB |

**Note:** The raw KV-cache is modest in MB. RAM pressure on M2 8 GB is driven
by the combined footprint: model weights (~2.2 GB) + KV-cache + Metal memory
pools + framework allocations + OS reservation + transient buffers. The
~1,950-token system-level crossover estimate accounts for this full overhead,
not KV-cache bytes alone.
