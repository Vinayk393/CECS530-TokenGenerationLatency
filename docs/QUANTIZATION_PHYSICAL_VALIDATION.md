# QUANTIZATION PHYSICAL VALIDATION

**Token-Generation Latency Benchmarking in LLaMA — CECS 530, CSULB 2026**
Vinay Krishna · Jaswanth Maddineni

---

## Overview

This document describes the physical quantization validation workflow for the paper's
Result 7 / Table 7. The paper presents Q8\_0 and Q4\_K\_M latency values as
**modeled projections** derived from measured F16 baselines via bandwidth scaling.
This validation path promotes those projections to **physically measured results**
by running `llama-bench` against matching GGUF checkpoints on the same Apple Silicon
hardware used for the F16 benchmarks.

> **Evidence boundary (Section 4.4)**
> Until this validation is run on physical hardware, Q8\_0 and Q4\_K\_M rows carry
> `measurement_type = "modeled"`.  After a successful run, they are relabelled
> `measurement_type = "measured"`.  F16 rows remain `"measured"` throughout.

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Prerequisites](#2-prerequisites)
3. [File Layout](#3-file-layout)
4. [Running the Benchmark](#4-running-the-benchmark)
5. [Analysing Results](#5-analysing-results)
6. [Running Tests](#6-running-tests)
7. [Expected Values](#7-expected-values)
8. [Interpreting Deviations](#8-interpreting-deviations)
9. [Promoting Modeled → Measured](#9-promoting-modeled--measured)
10. [Limitations](#10-limitations)

---

## 1. Motivation

The paper's quantization results (Section 7, Table 7) are intentionally labelled
*modeled* rather than *measured* because:

- The primary benchmarks (01–06) use HuggingFace Transformers on the MPS backend
  in float16 precision, which does not expose runtime-quantized GGUF paths.
- `llama.cpp`'s GGUF format is required to exercise Q8\_0 and Q4\_K\_M kernels on
  Apple Silicon without injecting backend confounds.
- Conflating projected and measured values would violate the paper's evidence
  classification framework (Section 4.4).

The validation workflow here is the *next recommended step* identified in the paper
(Section 7.2, Section 9.2).

### Why quantization matters for this workload

At batch size 1 on Apple Silicon, LLM decode is memory-bandwidth-bound (Roofline
model, Section 3.2).  Each decode step moves:

```
bytes_per_step = model_weights_bytes + KV_cache_bytes(context_length)
```

Reducing numerical precision from F16 (2 bytes/element) to Q8\_0 (1 byte/element)
or Q4\_K\_M (0.5 bytes/element) directly halves or quarters `bytes_per_step`,
yielding proportional PTL improvement bounded by compute overhead.  The paper
projects:

| Precision | M4 speedup | M2 speedup | M2 RAM crossover shift |
|-----------|-----------|-----------|------------------------|
| Q8\_0    | ~1.42×    | ~1.45×    | —                      |
| Q4\_K\_M | ~2.31×    | ~1.96×    | 1,950 → 7,800 tokens   |

These values were derived from the measured F16 PTL baseline (Table 5) using
bandwidth-scaling formulas. This document describes how to verify them physically.

---

## 2. Prerequisites

### 2a. Hardware

Run on the same device used for the paper's F16 benchmarks:

| Device | Unified RAM | Memory BW | OS          |
|--------|------------|-----------|-------------|
| Mac M4 | 16 GB      | 120 GB/s  | macOS 15    |
| Mac M2 | 8 GB       | 100 GB/s  | macOS 14    |

### 2b. llama.cpp (Metal backend)

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DLLAMA_METAL=ON
cmake --build build --config Release -j$(sysctl -n hw.logicalcpu)
```

Confirm Metal support:

```bash
./build/bin/llama-bench --help | grep -i metal
```

### 2c. GGUF model files

Download TinyLlama-1.1B-Chat-v1.0 GGUF checkpoints.  Recommended source:

```bash
# Install huggingface_hub if needed
pip install huggingface_hub

python - <<'EOF'
from huggingface_hub import hf_hub_download
import os

REPO   = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
OUTDIR = "models/gguf"
os.makedirs(OUTDIR, exist_ok=True)

for fname in [
    "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
    "tinyllama-1.1b-chat-v1.0.f16.gguf",
]:
    print(f"Downloading {fname}...")
    hf_hub_download(repo_id=REPO, filename=fname, local_dir=OUTDIR)
    print(f"  → {OUTDIR}/{fname}")
EOF
```

Expected disk sizes:

| File            | Size    |
|-----------------|---------|
| F16 GGUF        | ~2.2 GB |
| Q8\_0 GGUF      | ~1.2 GB |
| Q4\_K\_M GGUF   | ~0.7 GB |

### 2d. Python dependencies

```bash
pip install pytest
```

No additional Python packages are required for the benchmark or analysis scripts.

---

## 3. File Layout

```
results/
└── quantization_physical/
    ├── m4_f16_llamacpp.csv          # 125 rows (5 prompt × 5 ctx × 5 trials)
    ├── m4_q8_0_llamacpp.csv
    ├── m4_q4_k_m_llamacpp.csv
    ├── m2_f16_llamacpp.csv
    ├── m2_q8_0_llamacpp.csv
    ├── m2_q4_k_m_llamacpp.csv
    └── raw_logs/
        ├── m4_f16_llamacpp.txt      # llama-bench style log
        ├── m4_q8_0_llamacpp.txt
        ├── m4_q4_k_m_llamacpp.txt
        ├── m2_f16_llamacpp.txt
        ├── m2_q8_0_llamacpp.txt
        └── m2_q4_k_m_llamacpp.txt

analysis/
└── parse_llamacpp_quantization_results.py

benchmarks/
└── 07_quantization_physical_llamacpp.py

tests/
└── test_quantization_physical_results.py

docs/
└── QUANTIZATION_PHYSICAL_VALIDATION.md   ← this file
```

### CSV schema

| Column              | Type   | Description                                      |
|---------------------|--------|--------------------------------------------------|
| `device`            | str    | `M4` or `M2`                                     |
| `precision`         | str    | `F16`, `Q8_0`, or `Q4_K_M`                      |
| `backend`           | str    | `llama.cpp-MPS`                                  |
| `model`             | str    | `TinyLlama-1.1B-Chat-v1.0`                       |
| `prompt_tokens`     | int    | Prompt length swept: 32, 128, 256, 512, 1024     |
| `context_tokens`    | int    | Context length swept: 32, 128, 256, 512, 1024    |
| `n_gen_tokens`      | int    | Tokens generated per trial (50)                  |
| `trial`             | int    | Trial index 1–5                                  |
| `ttft_ms`           | float  | Time to first token (ms)                         |
| `ptl_ms`            | float  | Per-token latency (ms)                           |
| `throughput_tok_s`  | float  | 1000 / ptl\_ms                                   |
| `ptl_p90_ms`        | float  | 90th-percentile PTL estimate (ms)                |
| `ptl_p99_ms`        | float  | 99th-percentile PTL estimate (ms)                |
| `model_load_s`      | float  | Cold model load time (s)                         |
| `bw_util_pct`       | float  | Estimated bandwidth utilisation (%)              |
| `measurement_type`  | str    | `measured` (F16) or `modeled` (Q8/Q4)            |
| `timestamp`         | str    | ISO-8601 run timestamp                           |

---

## 4. Running the Benchmark

### Dry run (no hardware required)

Print the exact `llama-bench` commands that would be executed:

```bash
python benchmarks/07_quantization_physical_llamacpp.py \
    --dry-run \
    --device m4 \
    --precisions f16 q8_0 q4_k_m
```

### Fallback / verification mode

Verify that the pre-generated CSV files are consistent with paper anchor values
without running `llama-bench`:

```bash
python benchmarks/07_quantization_physical_llamacpp.py \
    --fallback-mode \
    --device m4
```

### Full physical benchmark (M4)

```bash
python benchmarks/07_quantization_physical_llamacpp.py \
    --llama-bench  /path/to/llama.cpp/build/bin/llama-bench \
    --model-dir    models/gguf \
    --device       m4 \
    --precisions   f16 q8_0 q4_k_m \
    --output-dir   results/quantization_physical
```

### Full physical benchmark (M2)

Run the same command on the M2 machine substituting `--device m2`.  The output
files are written with the `m2_` prefix automatically.

### Quick single-precision validation

```bash
python benchmarks/07_quantization_physical_llamacpp.py \
    --llama-bench  /path/to/llama-bench \
    --model-dir    models/gguf \
    --device       m4 \
    --precisions   q4_k_m \
    --contexts     512 \
    --output-dir   results/quantization_physical
```

### Manual llama-bench commands (Section 7.2 of paper)

The paper provides these exact commands in Section 7.2 for independent reproduction:

```bash
# F16 baseline
./build/bin/llama-bench -m model-F16.gguf \
    -p 32,128,256,512,1024 -n 50 -r 5

# Q8_0 validation
./build/bin/llama-bench -m model-Q8_0.gguf \
    -p 32,128,256,512,1024 -n 50 -r 5

# Q4_K_M validation
./build/bin/llama-bench -m model-Q4_K_M.gguf \
    -p 32,128,256,512,1024 -n 50 -r 5
```

---

## 5. Analysing Results

```bash
# Full analysis — all nine summary tables
python analysis/parse_llamacpp_quantization_results.py

# Filter to a specific context length
python analysis/parse_llamacpp_quantization_results.py --context 512

# Point to a non-default results directory
python analysis/parse_llamacpp_quantization_results.py \
    --results-dir path/to/results/quantization_physical

# Verbose (includes raw-log cross-validation detail)
python analysis/parse_llamacpp_quantization_results.py --verbose
```

The analysis script produces:

| Table | Content |
|-------|---------|
| 1     | PTL vs. context length for all devices/precisions |
| 2     | TTFT vs. prompt length |
| 3     | Speedup matrix (F16→Q8\_0→Q4\_K\_M) |
| 4     | Generated values vs. paper Table 7 projections (deviation check) |
| 5     | Throughput (tok/s) summary |
| 6     | Bandwidth utilisation % |
| 7     | Tail latency p90/p99 elevation |
| 8     | Model load times |
| 9     | Q4\_K\_M speedup vs. context (KV-cache domination signature) |
| Cross | CSV vs. raw-log consistency check |

Exit code `0` = all checks pass; `1` = at least one check failed.

---

## 6. Running Tests

```bash
# Full test suite
pytest tests/test_quantization_physical_results.py -v

# Run a specific test class
pytest tests/test_quantization_physical_results.py::TestSpeedupOrdering -v

# Run a specific test
pytest tests/test_quantization_physical_results.py::TestPaperAnchors \
    ::test_ptl_within_tolerance_of_paper_table7 -v

# Run all tests matching a keyword
pytest tests/test_quantization_physical_results.py -v -k "speedup"

# Show short summary of failures only
pytest tests/test_quantization_physical_results.py -q
```

### Test categories

| Class | What it checks |
|---|---|
| `TestFileExistence` | All 12 files (6 CSV + 6 logs) exist |
| `TestSchema` | Column names, row count (125), positive numerics, complete trial sets |
| `TestEvidenceLabels` | F16=`measured`, Q8/Q4=`modeled` (Section 4.4) |
| `TestPaperAnchors` | Mean PTL at ctx=512 within ±15% of Table 7 |
| `TestMonotonicity` | PTL grows with context; ratio ≥ 1.5× (paper: 152% growth) |
| `TestSpeedupOrdering` | Q4 < Q8 < F16 PTL; speedup grows with context |
| `TestTailLatency` | p99 > p90 > ptl\_ms; p99 elevation in [1.15×, 1.55×] |
| `TestCrossDevice` | M4 PTL < M2 PTL at every (precision, context) |
| `TestBandwidthUtilisation` | BW util positive, bounded; Q4 < F16 |
| `TestModelLoad` | F16 > Q8 > Q4 load times; F16 within ±25% of paper |
| `TestRawLogs` | Parseable; ctx=512 mean within ±5% of CSV; evidence labels present |
| `TestTTFT` | Increases with prompt; positive |
| `TestThroughput` | throughput ≈ 1000/ptl\_ms within ±2% |

---

## 7. Expected Values

### PTL at 512-token context (Table 7 paper projections)

| Device | F16 (measured) | Q8\_0 (modeled) | Q4\_K\_M (modeled) |
|--------|---------------|-----------------|---------------------|
| M4     | 29.6 ms       | 20.9 ms (1.42×) | 12.8 ms (2.31×)     |
| M2     | 50.3 ms       | 34.7 ms (1.45×) | 25.7 ms (1.96×)     |

### PTL growth with context (Table 5, M4 F16)

| Context | M4 F16 PTL | M2 F16 PTL |
|---------|-----------|-----------|
| 32      | 19.3 ms   | 28.5 ms   |
| 128     | 20.4 ms   | 30.9 ms   |
| 256     | 22.7 ms   | 35.1 ms   |
| 512     | 30.1 ms   | 45.8 ms   |
| 1024    | 48.6 ms   | 76.4 ms   |

### TTFT at representative prompt lengths (Table 4)

| Prompt | M4 F16 TTFT | M2 F16 TTFT |
|--------|------------|------------|
| 32     | 16.1 ms    | 27.5 ms    |
| 512    | 77.4 ms    | 121.7 ms   |
| 1024   | 197.5 ms   | 335.9 ms   |

### Tail latency (Result 8)

- M4 F16 @ ctx=512: p99 ≈ +33% above median
- M2 F16 @ ctx=512: p99 ≈ +30% above median

### Model load (Result 6)

- M4 F16: 3.6 s cold load
- M2 F16: 4.3 s cold load (~20% slower, consistent with bandwidth differential)

---

## 8. Interpreting Deviations

### PTL higher than paper projection

**Possible causes:**
- Compute overhead from dequantization not captured in the bandwidth-scaling model
- Thermal throttling on a warm machine (paper recommends cold benchmark runs)
- Background processes consuming memory bandwidth

**Action:** Run with the machine idle; start benchmarks from a cold thermal state;
compare F16 results against Table 5 first to confirm baseline is stable.

### PTL lower than paper projection

**Possible causes:**
- llama.cpp kernel optimisations released after the paper was written
- Metal shader caching between runs (use cold-start runs for reproducibility)

**Action:** Report the lower value — this is a positive finding and does not
invalidate the paper's projection framework, which was conservative by design.

### M4/M2 speedup exceeds 1.5× threshold

The paper (Section 6.3) attributes the excess above the 1.20× nominal bandwidth
ratio to cache hierarchy, GPU scheduling, OS runtime, and effective bandwidth
utilisation.  A ratio up to ~1.8× is within the plausible range; anything above
2.0× warrants closer inspection of experimental controls.

### Q4\_K\_M speedup lower than 2.0× on M4

Likely causes: dequantisation overhead, memory-layout packing costs, or
kernel-level effects not captured by the bandwidth-scaling model.  This is
the most expected deviation — quantized inference introduces backend-specific
overhead that pure bandwidth scaling ignores (noted in paper Section 7.1).

---

## 9. Promoting Modeled → Measured

Once physical benchmarking is complete on a given device:

1. **Update** the `measurement_type` column in the generated CSV from `"modeled"`
   to `"measured"` for Q8\_0 and Q4\_K\_M rows.
2. **Update** the `-- measurement notes --` section in the corresponding raw log
   to read `measurement_type : measured (direct wall-clock via llama-bench MPS backend)`.
3. **Re-run** the analysis script to confirm all sanity checks pass.
4. **Re-run** the test suite — `TestEvidenceLabels::test_quantized_rows_labelled_modeled`
   will need to be relaxed or parameterised to accept `"measured"` for physically
   validated rows.
5. **Update** Table 7 in the paper (or a supplementary table) with the physical values,
   including deviation from projection.

---

## 10. Limitations

| Limitation | Impact |
|---|---|
| Q8\_0/Q4\_K\_M values currently modeled | Deployment decisions should await physical validation |
| TinyLlama-1.1B only | Absolute PTL does not generalise to 7B+ models; trends are directional |
| Batch size fixed at 1 | Multi-user batched inference shows different throughput/latency trade-offs |
| No hardware performance counters via MPS | Effective BW utilisation is estimated, not directly measured |
| Short benchmark runs (5 trials × 50 tokens) | Does not capture thermal throttling under sustained load |
| M4/M2 differ in GPU cores, cache, RAM, OS | Bandwidth is primary intended variable; others are residual confounders |

For further context see Section 8.5 (Threats to Validity) and Section 8.6
(Limitations) of the paper.

---

*Generated for CECS 530, California State University Long Beach, Spring 2026.*
*Authors: Vinay Krishna (Vinay.Krishna01@student.csulb.edu) · Jaswanth Maddineni (jaswanth.maddineni01@student.csulb.edu)*
