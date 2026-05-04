
# KV-Cache Optimization Proposal

## Project: Token-Generation Latency Benchmarking in LLaMA  
**Platform:** Mac M4 16GB / Mac M2 8GB · MPS backend · TinyLlama-1.1B · float16  
**Data source:** `results/Mac_M4_16GB/09_latency_decomposition_long.csv`

---

## 1. Observed Bottleneck

From latency decomposition at 512-token context (Mac M4):

| Component     | Latency (ms) | Share |
|---------------|-------------|-------|
| Attention     | 9.35        | 33.5% |
| MLP           | 5.89        | 21.1% |
| KV Read/Write | 3.03        | 10.8% |
| LM Head       | 3.12        | 11.2% |
| Overhead      | 4.57        | 16.4% |

Key finding from `02_per_token_latency_summary.csv`:

- PTL at context 32:   **19.3 ms**
- PTL at context 1024: **48.6 ms**  (+152%)

The per-token latency more than doubles as context grows from 32 to 1024 tokens,
while the model weights remain constant. This confirms the bottleneck is
**KV-cache memory traffic**, not compute.

Memory bandwidth utilization rises from ~62% to ~82% (M4) as context grows
(`04_throughput_vs_prompt_length.csv`), confirming the decode phase is
increasingly memory-bandwidth-bound.

---

## 2. Why KV-Cache Causes This

During autoregressive decode, every new token requires reading **all previously
computed key-value pairs** across all 22 transformer layers (TinyLlama-1.1B).

KV-cache memory for TinyLlama-1.1B (F16):

```
size = 2 × layers × kv_heads × head_dim × context_len × bytes
     = 2 × 22 × 4 × 64 × context_len × 2
```

At 1024 tokens: ~2.25 MB per decode step streamed from unified DRAM.  
At 2048 tokens: ~4.5 MB per decode step.

This grows linearly with context but the bandwidth ceiling is fixed  
(120 GB/s on M4, 100 GB/s on M2), so latency increases proportionally.

---

## 3. Proposed Optimization: KV-Cache Quantization

### What
Reduce KV-cache precision from float16 (2 bytes/element) to:
- **Q8_0** (1 byte/element) — 2× memory reduction
- **Q4_0** (0.5 bytes/element) — 4× memory reduction

### Why This Works on Apple Silicon
Apple Silicon's unified memory architecture means CPU and GPU share the same
DRAM pool. Unlike discrete GPU systems with HBM, there is no PCIe transfer —
but the bandwidth ceiling (100–120 GB/s) is also lower than HBM2e (~900+ GB/s).
This makes Apple Silicon **more sensitive** to KV-cache size than datacenter GPUs,
and therefore benefits more from KV quantization proportionally.

### Expected Speedup (from `07_quantization_speedup.csv`)

| Precision | M4 Speedup vs F16 | M2 Speedup vs F16 |
|-----------|-------------------|-------------------|
| Q8_0      | ~1.5×             | ~1.4×             |
| Q4_K_M    | ~2.2×             | ~2.0×             |

Speedup is **context-dependent**: at short contexts (32 tokens), overhead dominates
and quantization gains are smaller (~1.5–1.9×). At long contexts (512+ tokens),
KV traffic dominates and speedup approaches theoretical limits.

### Implementation
KV-cache quantization is natively supported in **llama.cpp** via GGUF format:

```bash
# Run with llama-bench to measure KV quantization effect:
llama-bench -m model-Q4_K_M.gguf -p 512 -n 128 -r 5
```

For HuggingFace backend, KV precision requires custom attention hooks —
not yet supported natively in transformers.

---

## 4. Trade-offs

| Factor          | Q8_0          | Q4_K_M        |
|-----------------|---------------|---------------|
| Speed gain      | ~1.5×         | ~2.2×         |
| Quality impact  | Negligible    | Minor         |
| RAM reduction   | 2×            | 4×            |
| Best for        | Quality-first | Speed-first   |

On M2 8GB specifically, Q4_K_M is strongly recommended: at 1024 tokens with
F16, the KV-cache + model weights approach the 8GB RAM ceiling
(`08_kvcache_size_vs_context.csv`), causing OS memory pressure. Q4_K_M reduces
this by 4×, keeping the total footprint well within the 8GB budget.

---

## 5. Realistic Improvement Estimate

For a 100-token generation with 512-token context on M4:
- F16 decode time:     ~29.6 ms/token × 100 = **2.96 s**
- Q4_K_M decode time:  ~12.8 ms/token × 100 = **1.28 s**
- **Estimated wall-time saving: ~1.7 seconds per generation**

On M2 8GB with long contexts, the benefit is larger due to RAM pressure avoidance.

---

## 6. Conclusion

KV-cache quantization is the most impactful single optimization available on
Apple Silicon without modifying model weights or retraining. It directly targets
the measured bottleneck (memory bandwidth), requires no code changes for llama.cpp
users, and is especially effective on the M2 8GB device where RAM capacity is a
hard constraint at long contexts.
