"""
08_kvcache_size_vs_context.py
-------------------------------
Calculates (analytically) KV-cache memory usage as a function of context length
for different model configs and precisions. No GPU/model loading needed —
this is a pure math + visualization script.

Also estimates the RAM pressure crossover point for 8GB vs 16GB devices,
showing exactly when the KV-cache + model weights exceed available RAM.

Requirements:
    pip install matplotlib numpy

Usage:
    python 08_kvcache_size_vs_context.py
    python 08_kvcache_size_vs_context.py --model_gb 2.2 --num_layers 22 --num_heads 32 --head_dim 64

    TinyLlama-1.1B:     --model_gb 2.2  --num_layers 22 --num_heads 32 --head_dim 64
    LLaMA 3.2-1B:       --model_gb 2.0  --num_layers 16 --num_heads 32 --head_dim 64
    LLaMA 3.2-3B:       --model_gb 6.0  --num_layers 28 --num_heads 24 --head_dim 128

Output:
    results/08_kvcache_size_vs_context.json
    plots/08_kvcache_size_vs_context.png
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_DIR = "results"
PLOTS_DIR   = "plots"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

# Default: TinyLlama-1.1B (safe and realistic for student hardware)
DEFAULT_MODEL_GB   = 2.2
DEFAULT_NUM_LAYERS = 22
DEFAULT_NUM_HEADS  = 32
DEFAULT_HEAD_DIM   = 64
DEFAULT_NUM_KV_HEADS = 4   # GQA: TinyLlama uses 4 KV heads

CONTEXT_LENGTHS = list(range(64, 2049, 64))

PRECISIONS = {
    "F16":    2,   # bytes per element
    "Q8_0":   1,
    "Q4_K_M": 0.5,
}

PRECISION_COLORS = {
    "F16":    "#EF4444",
    "Q8_0":   "#10B981",
    "Q4_K_M": "#3B82F6",
}

RAM_CONFIGS = {
    "M2 8GB":  8,
    "M4 16GB": 16,
}
RAM_COLORS = {
    "M2 8GB":  "#8B5CF6",
    "M4 16GB": "#3B82F6",
}


def kv_cache_gb(context_len: int, num_layers: int, num_kv_heads: int,
                head_dim: int, bytes_per_elem: float) -> float:
    """
    KV-cache size in GB.
    Formula: 2 (K+V) × num_layers × num_kv_heads × head_dim × context_len × bytes
    """
    size_bytes = 2 * num_layers * num_kv_heads * head_dim * context_len * bytes_per_elem
    return size_bytes / 1e9


def crossover_context(model_gb: float, ram_gb: float,
                      num_layers: int, num_kv_heads: int,
                      head_dim: int, bytes_per_elem: float) -> int | None:
    """
    Find the context length at which KV-cache + model weights exceed available RAM.
    Returns the crossover context length, or None if it never crosses within 4096 tokens.
    """
    available_for_kv = ram_gb - model_gb
    if available_for_kv <= 0:
        return 0   # model already doesn't fit
    bytes_per_ctx = 2 * num_layers * num_kv_heads * head_dim * bytes_per_elem
    crossover = int(available_for_kv * 1e9 / bytes_per_ctx)
    return crossover if crossover < 8192 else None


def main(args):
    print(f"Model config: {args.num_layers} layers, {args.num_kv_heads} KV heads, "
          f"head_dim={args.head_dim}, model size={args.model_gb} GB")

    results = {}
    for prec, bpe in PRECISIONS.items():
        sizes = [
            kv_cache_gb(cl, args.num_layers, args.num_kv_heads, args.head_dim, bpe)
            for cl in CONTEXT_LENGTHS
        ]
        results[prec] = [{"context": cl, "kv_gb": round(s, 4)}
                         for cl, s in zip(CONTEXT_LENGTHS, sizes)]

    # Crossover points
    crossovers = {}
    for ram_name, ram_gb in RAM_CONFIGS.items():
        crossovers[ram_name] = {}
        for prec, bpe in PRECISIONS.items():
            cx = crossover_context(args.model_gb, ram_gb, args.num_layers,
                                   args.num_kv_heads, args.head_dim, bpe)
            crossovers[ram_name][prec] = cx
            if cx is not None:
                print(f"  {ram_name} + {prec}: RAM pressure at context ≥ {cx} tokens")
            else:
                print(f"  {ram_name} + {prec}: No RAM pressure within 8192 tokens")

    out_json = os.path.join(RESULTS_DIR, "08_kvcache_size_vs_context.json")
    with open(out_json, "w") as f:
        json.dump({
            "model_gb": args.model_gb,
            "num_layers": args.num_layers,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "kv_cache_by_precision": results,
            "ram_crossover_tokens": crossovers,
        }, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"KV-Cache Memory vs Context Length\n"
        f"{args.num_layers}-layer model · {args.model_gb:.1f} GB weights",
        fontsize=13, fontweight="bold"
    )

    # Plot 1: KV-cache size by precision
    ax = axes[0]
    for prec, data in results.items():
        ctxs  = [d["context"] for d in data]
        sizes = [d["kv_gb"]   for d in data]
        ax.plot(ctxs, sizes, linewidth=2, color=PRECISION_COLORS[prec], label=prec)
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("KV-cache size (GB)")
    ax.set_title("KV-cache growth by precision")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Total memory (model + KV) vs RAM limits
    ax = axes[1]
    for prec, data in results.items():
        ctxs       = [d["context"]               for d in data]
        total_gb   = [d["kv_gb"] + args.model_gb for d in data]
        ax.plot(ctxs, total_gb, linewidth=2, color=PRECISION_COLORS[prec], label=prec)

    for ram_name, ram_gb in RAM_CONFIGS.items():
        ax.axhline(ram_gb, linestyle="--", linewidth=1.5,
                   color=RAM_COLORS[ram_name], label=f"{ram_name} limit")

    ax.fill_between(CONTEXT_LENGTHS, 0, args.model_gb,
                    alpha=0.08, color="#888", label="Model weights")
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Total memory (GB)")
    ax.set_title("Model + KV-cache vs device RAM")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Crossover heatmap (RAM × Precision)
    ax = axes[2]
    ram_names  = list(RAM_CONFIGS.keys())
    prec_names = list(PRECISIONS.keys())
    matrix = np.zeros((len(ram_names), len(prec_names)))
    for ri, rn in enumerate(ram_names):
        for pi, pn in enumerate(prec_names):
            cx = crossovers[rn][pn]
            matrix[ri, pi] = cx if cx is not None else 8192

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=4096)
    ax.set_xticks(range(len(prec_names)))
    ax.set_yticks(range(len(ram_names)))
    ax.set_xticklabels(prec_names)
    ax.set_yticklabels(ram_names)
    ax.set_title("RAM pressure crossover (tokens)\nGreen = more headroom")
    plt.colorbar(im, ax=ax, label="Context tokens until RAM pressure")

    for ri in range(len(ram_names)):
        for pi in range(len(prec_names)):
            val = crossovers[ram_names[ri]][prec_names[pi]]
            label = f"{val:,}" if val is not None else ">8192"
            ax.text(pi, ri, label, ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white" if (val or 9999) < 1024 else "black")

    plt.tight_layout()
    out_png = os.path.join(PLOTS_DIR, "08_kvcache_size_vs_context.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_png}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KV-cache size vs context length")
    parser.add_argument("--model_gb",     type=float, default=DEFAULT_MODEL_GB)
    parser.add_argument("--num_layers",   type=int,   default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--num_heads",    type=int,   default=DEFAULT_NUM_HEADS)
    parser.add_argument("--num_kv_heads", type=int,   default=DEFAULT_NUM_KV_HEADS)
    parser.add_argument("--head_dim",     type=int,   default=DEFAULT_HEAD_DIM)
    main(parser.parse_args())
