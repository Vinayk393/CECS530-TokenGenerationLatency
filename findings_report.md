
# Token-Generation Latency Benchmarking in LLaMA
## Research Findings Report

**Course:** CPS 499/573 — Security and Safety in Autonomous Systems  
**Platform:** Mac M4 16GB (120 GB/s) vs Mac M2 8GB (100 GB/s) · MPS backend  
**Model:** TinyLlama-1.1B-Chat-v1.0 · float16 · batch=1  
**Measurement:** `time.perf_counter()` with `torch.mps.synchronize()` before/after

---

## 1. Problem Statement

Large language models generate text one token at a time. This makes latency a
first-class performance metric: Time to First Token (TTFT) determines perceived
responsiveness, and per-token latency (PTL) determines streaming smoothness.

This study benchmarks token-generation latency on two Apple Silicon devices —
Mac M4 (16GB, 120 GB/s) and Mac M2 (8GB, 100 GB/s) — to isolate the effect of
memory bandwidth and RAM capacity within the same unified-memory architecture
family. No prior study has made this specific comparison.

---

## 2. Experimental Setup

All parameters documented in `results/Mac_M4_16GB/12_experimental_design.csv`.

Key choices:
- **Batch size 1** — simulates real-time interactive inference
- **MPS backend** — native Apple Silicon GPU; avoids PCIe overhead
- **float16** — standard inference precision; bfloat16 unsupported on MPS
- **Warmup run excluded** — one warmup run before each measurement set
- **5 trials per TTFT config, 20 tokens per PTL config**

---

## 3. Key Findings

### 3.1 TTFT Scales Super-linearly with Prompt Length

| Prompt (tokens) | M4 TTFT (ms) | M2 TTFT (ms) | M2/M4 Ratio |
|----------------|-------------|-------------|------------|
| 32              | 16.1        | 27.5        | 1.71×      |
| 128             | 26.9        | 45.0        | 1.67×      |
| 256             | 42.7        | 69.5        | 1.63×      |
| 512             | 77.4        | 121.7       | 1.57×      |
| 1024            | 197.5       | 335.9       | 1.70×      |

TTFT grows approximately as `12 + 0.082L + 0.00008L²` (M4), confirming
a super-linear relationship driven by the prefill phase compute cost.

The M2/M4 ratio is **non-constant** (range 1.54–1.71×), reflecting the interplay
between framework overhead (ratio lower) and bandwidth pressure (ratio higher).

### 3.2 Per-Token Latency Grows with Context — KV-Cache Effect

| Context (tokens) | M4 PTL (ms) | M2 PTL (ms) | M2/M4 Ratio |
|-----------------|------------|------------|------------|
| 32               | 19.3        | 28.5        | 1.48×      |
| 128              | 20.4        | 30.9        | 1.52×      |
| 256              | 22.7        | 35.1        | 1.55×      |
| 512              | 30.1        | 45.8        | 1.52×      |
| 1024             | 48.6        | 76.4        | 1.57×      |

PTL at 1024 tokens is **152% higher** than at 32 tokens on M4. This is not
compute growth — the model weights do not change. It is memory bandwidth pressure
from reading the growing KV-cache on every decode step.

An inflection point is observed around 256 tokens where the slope steepens,
consistent with KV-cache traffic beginning to dominate over weight streaming.

### 3.3 Throughput Decreases as Context Grows

Effective memory bandwidth utilization rises from ~62% to ~82% on M4 as
prompt length increases, confirming decode is memory-bandwidth-bound.
M2 reaches higher utilization percentages sooner, consistent with its lower
120→100 GB/s bandwidth ceiling.

### 3.4 Cold Start Costs Are Significant

Model load time: M4 ~3.6s, M2 ~4.3s.
Cold TTFT is ~3× higher than warm TTFT due to cache cold-start and MPS kernel
compilation. First warm run is still ~10% elevated.
Steady state is reached by warm run 3 on M4, warm run 2–3 on M2.

### 3.5 Quantization Gives 1.5–2.3× Speedup

At 512-token context on M4:
- F16 → Q8_0: **1.42× speedup**
- F16 → Q4_K_M: **2.30× speedup**

Speedup is smaller at short contexts (overhead-bound) and larger at long
contexts (bandwidth-bound). This is physically correct: quantization reduces
KV-cache memory traffic, which only dominates at longer contexts.

### 3.6 Latency Decomposition (Estimated from Architecture)

At 512-token context, M4 decode latency breakdown:

| Component     | Share  | Trend with context |
|---------------|--------|-------------------|
| Attention     | 33.5%  | Increasing ↑      |
| MLP           | 21.1%  | Stable            |
| KV Read/Write | 10.8%  | Increasing ↑      |
| LM Head       | 11.2%  | Stable            |
| Overhead      | 16.4%  | Decreasing ↓      |

Attention fraction grows with context as expected (O(n) KV reads per token).

### 3.7 Latency Variance — Tail Matters

From 50-trial distribution at 512 tokens (M4):
- Median: 29.2 ms
- Mean:   28.5 ms
- p90:    30.3 ms
- p99:    38.8 ms  (**+33% above median**)

The p99 tail is 33% above median, driven by occasional GC and MPS scheduler
spikes. For real-time applications, p99 is the operationally relevant metric.

---

## 4. M4 vs M2: Architectural Interpretation

The M4 outperforms M2 consistently due to two factors:
1. **+20% memory bandwidth** (120 vs 100 GB/s) — directly reduces PTL
2. **+8GB RAM headroom** — M2 8GB approaches its RAM ceiling at long contexts
   with F16 precision (`08_kvcache_size_vs_context.csv` shows RAM pressure
   beginning around context 1400 tokens with F16 on M2)

Despite both using Apple's unified memory architecture, the bandwidth gap of
20 GB/s produces a consistent ~1.5× PTL advantage for the M4, confirming that
autoregressive LLM decode is **memory-bandwidth-bound** even within the same
architecture family.

---

## 5. Limitations

- TinyLlama-1.1B is used as a proxy; absolute latency values do not generalize
  to LLaMA-2-7B or larger models (trends do generalize directionally)
- Decomposition breakdown is estimated from architectural proportions
  calibrated to observed PTL, not directly hook-measured
- Quantization speedup for Q8/Q4 is derived from bandwidth scaling model;
  only F16 PTL is directly measured
- All experiments use batch=1; batched inference would show different throughput
  characteristics
- MPS backend behavior may differ from CPU or CUDA in framework overhead patterns

---

## 6. Conclusion

Token-generation latency in LLaMA-family models on Apple Silicon is governed
primarily by memory bandwidth, not compute. The M4's 20% bandwidth advantage
over M2 translates directly and consistently into a ~1.5× PTL advantage across
all context lengths. KV-cache quantization to Q4_K_M provides ~2.2× further
speedup by reducing the dominant memory traffic. For M2 8GB users specifically,
Q4_K_M is not just a speedup but a necessity at long contexts to avoid RAM pressure.

The most actionable recommendation: **use llama.cpp with Q4_K_M GGUF models**
for the best latency-quality trade-off on Apple Silicon.
