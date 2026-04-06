
# Apple Silicon Unified Memory Architecture
## Why It Matters for LLM Inference

---

## The Core Difference

Traditional computing separates CPU RAM from GPU VRAM, connected via PCIe:

```
CPU ──[PCIe 16x, ~32 GB/s]── GPU
RAM                           VRAM
(DDR5, ~90 GB/s)              (HBM3, ~3 TB/s on H100)
```

Apple Silicon eliminates this separation:

```
CPU + GPU + Neural Engine
        │
   Unified DRAM
(LPDDR5X, 100–273 GB/s)
```

Both the CPU and GPU cores access the same physical memory pool at the same
bandwidth. There is no model weight copy from RAM to VRAM — weights live in
one place and both processors access them directly.

---

## What This Means for LLM Decode

Autoregressive LLM decode is a **memory-bandwidth-bound** workload. Each new
token requires streaming the full model weight matrix plus the KV-cache from
DRAM. The roofline ceiling is set by DRAM bandwidth, not TFLOPS.

For a 1B parameter model in float16:
- Model weights: ~2 GB
- At 20ms/token on M4: bandwidth usage ≈ 2 GB / 0.020 s = **100 GB/s**
- M4 peak: 120 GB/s → **~83% utilization**

This means the M4 is running close to its physical bandwidth limit during
decode. No algorithmic optimization changes this — only reducing the bytes
transferred (quantization) or increasing the bandwidth ceiling (better hardware).

---

## M4 vs M2: The 20 GB/s Gap

| Spec             | Mac M4 16GB   | Mac M2 8GB    |
|------------------|---------------|---------------|
| Memory bandwidth | 120 GB/s      | 100 GB/s      |
| RAM              | 16 GB         | 8 GB          |
| GPU cores        | 10            | 8             |
| Theoretical PTL  | ∝ 1/120       | ∝ 1/100       |

For a pure bandwidth-bound workload, theoretical PTL ratio = 120/100 = **1.20×**.

Our measured PTL ratio is ~**1.50×** — higher than the theoretical bandwidth
ratio. The gap (1.50 vs 1.20) is explained by:

1. **Framework overhead** — MPS kernel scheduling is slightly heavier on M2
   (fewer GPU cores, older Metal version)
2. **Cache effects** — M2 has less L2/L3 cache, causing more DRAM fetches
   per token even at short contexts
3. **RAM pressure at long contexts** — M2 8GB hits memory pressure earlier,
   causing OS swap activity that inflates latency

---

## The 8GB RAM Constraint (M2 Specific)

Total memory footprint during inference:
```
Total = Model weights + KV-cache + OS + Framework overhead
      ≈ 2.2 GB       + KV(ctx) + ~1.5 GB
```

For TinyLlama-1.1B in float16, the KV-cache at context length L:
```
KV size = 2 × 22 layers × 4 kv_heads × 64 head_dim × L × 2 bytes
        = 0.0022 GB × L
```

**Crossover point on M2 8GB (F16):**
Available for KV = 8 - 2.2 - 1.5 = 4.3 GB
Max context before pressure = 4.3 / 0.0022 ≈ **~1,950 tokens**

Beyond this, macOS starts using swap, causing latency spikes that appear as
unpredictable outliers in the inter-token timeline.

With Q4_K_M (4× smaller KV-cache), the crossover moves to ~7,800 tokens —
effectively eliminating RAM pressure for typical use cases.

---

## Why PCIe Absence Helps — But Has Limits

On an NVIDIA discrete GPU system:
1. Model loads from disk → CPU RAM (~10 GB/s)
2. CPU RAM → GPU VRAM via PCIe (~32 GB/s) — the transfer bottleneck
3. GPU VRAM → GPU compute (~1–3 TB/s on high-end GPUs)

Apple Silicon skips step 2 entirely. This is a significant advantage for
**model loading** (cold start is faster) and for workloads that require
frequent CPU-GPU data sharing.

However, Apple Silicon's peak bandwidth (100–273 GB/s) is far below HBM3
(~3 TB/s on H100). For long-running decode at large batch sizes, datacenter
GPUs outperform Apple Silicon by 10–30× on throughput (tok/s). Our study
operates at batch=1 where Apple Silicon's cold-start and efficiency advantages
are most relevant.

---

## Practical Implications

**For M4 16GB users:**
- F16 inference is viable up to ~1,700 token contexts without RAM pressure
- Q4_K_M recommended for sustained long-context generation
- MPS backend outperforms CPU by ~3× for LLaMA-size models

**For M2 8GB users:**
- Q4_K_M is strongly recommended — F16 hits RAM pressure at ~1,950 tokens
- Avoid models larger than ~4B parameters in F16 (won't fit)
- LLaMA-3.2-1B Q4_K_M fits comfortably and runs at ~30 tok/s

**For both:**
- Cold start is ~3× slower than warm — keep model loaded for interactive use
- Batch size 1 is optimal for single-user interactive scenarios
- Throughput scales almost linearly with memory bandwidth across the Apple Silicon family
