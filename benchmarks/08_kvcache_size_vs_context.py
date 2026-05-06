"""
08_kvcache_size_vs_context.py
-------------------------------
Analytically calculates KV-cache memory usage as a function of context length
for different model configs and precisions. No GPU or model loading needed —
this is a pure computation + visualization script.

Evidence label: analytical
  KV-cache sizes are computed from the formula:
      M_KV = 2 × L × H_kv × d_head × context × bytes_per_element

  For TinyLlama-1.1B (L=22, H_kv=4, d_head=64, F16):
      M_KV(1000 tokens) = 2 × 22 × 4 × 64 × 1000 × 2 = 22,528,000 bytes ≈ 22.5 MB

RAM pressure crossover estimates are system-level estimates, NOT raw KV-cache
alone. Practical memory pressure on M2 8GB includes: model weights (~2.2GB),
framework allocations, Metal memory pools, activation buffers, OS-reserved
memory, and transient allocation spikes. The crossover token counts are
empirical estimates, not closed-form results.

Requirements:
    pip install matplotlib numpy pandas

Usage:
    python 08_kvcache_size_vs_context.py
    python 08_kvcache_size_vs_context.py \\
        --model_gb 2.2 --num_layers 22 --num_kv_heads 4 --head_dim 64

    TinyLlama-1.1B:   --model_gb 2.2 --num_layers 22 --num_kv_heads 4 --head_dim 64
    LLaMA 3.2-1B:     --model_gb 2.0 --num_layers 16 --num_kv_heads 8 --head_dim 64
    LLaMA 3.2-3B:     --model_gb 6.0 --num_layers 28 --num_kv_heads 8 --head_dim 128

Output:
    results/<device_name>/08_kvcache_size_vs_context.csv
    results/<device_name>/08_kvcache_size_vs_context_metadata.json
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    ensure_output_dir,
    save_csv,
    save_json,
    save_metadata,
)

# ── Defaults: TinyLlama-1.1B ──────────────────────────────────────────────────
DEFAULT_MODEL_GB    = 2.2
DEFAULT_NUM_LAYERS  = 22
DEFAULT_NUM_HEADS   = 32   # query heads (unused in formula, for reference)
DEFAULT_NUM_KV_HEADS = 4   # GQA KV heads
DEFAULT_HEAD_DIM    = 64

CONTEXT_LENGTHS = list(range(64, 2113, 64))  # 64 to 2048 in steps of 64

PRECISIONS = {
    "F16":    2.0,    # bytes per element
    "Q8_0":   1.0,
    "Q4_K_M": 0.5,
}

PRECISION_COLORS = {
    "F16":    "#EF4444",
    "Q8_0":   "#10B981",
    "Q4_K_M": "#3B82F6",
}

RAM_CONFIGS = {
    "M2 8GB":  8.0,
    "M4 16GB": 16.0,
}
RAM_COLORS = {
    "M2 8GB":  "#8B5CF6",
    "M4 16GB": "#F59E0B",
}

# System-level RAM pressure overhead beyond model weights + raw KV-cache.
# Accounts for: Metal pools, framework allocations, OS pressure, transient buffers.
# This is an empirical estimate calibrated to observed swap onset behavior.
SYSTEM_OVERHEAD_GB = 1.5


def kv_cache_mb(layers: int, kv_heads: int, head_dim: int,
                context: int, bytes_per_elem: float) -> float:
    """
    KV-cache size in MB.
    Formula: 2 (K+V) × layers × kv_heads × head_dim × context × bytes_per_elem
    """
    size_bytes = 2 * layers * kv_heads * head_dim * context * bytes_per_elem
    return size_bytes / 1e6  # bytes → MB


def kv_cache_gb(layers: int, kv_heads: int, head_dim: int,
                context: int, bytes_per_elem: float) -> float:
    return kv_cache_mb(layers, kv_heads, head_dim, context, bytes_per_elem) / 1024.0


def system_level_crossover(model_gb: float, ram_gb: float,
                            layers: int, kv_heads: int,
                            head_dim: int, bytes_per_elem: float) -> int | None:
    """
    Estimate the context length at which total system memory pressure
    (model weights + KV-cache + system overhead) exceeds RAM.

    This is a system-level estimate, not a closed-form analytical result.
    The SYSTEM_OVERHEAD_GB term accounts for runtime allocations, Metal pools,
    OS reservation, and transient buffers that are not captured by the
    raw KV-cache formula alone.

    Returns the estimated crossover token count, or None if > 16384 tokens.
    """
    available_for_kv = ram_gb - model_gb - SYSTEM_OVERHEAD_GB
    if available_for_kv <= 0:
        return 0  # model + overhead already fills RAM
    bytes_per_token = 2 * layers * kv_heads * head_dim * bytes_per_elem
    crossover = int(available_for_kv * 1e9 / bytes_per_token)
    return crossover if crossover <= 16384 else None


def main(args):
    print(f"\nKV-cache sizing: {args.num_layers} layers, {args.num_kv_heads} KV heads, "
          f"head_dim={args.head_dim}, model_weights={args.model_gb} GB")
    print(f"System overhead allowance: {SYSTEM_OVERHEAD_GB} GB (Metal, runtime, OS)")

    out_dir = ensure_output_dir(args.output_dir, args.device_name)

    # ── Compute KV-cache sizes ────────────────────────────────────────────────
    results_by_prec: dict[str, list[dict]] = {}
    all_rows: list[dict] = []

    for prec, bpe in PRECISIONS.items():
        rows = []
        for ctx in CONTEXT_LENGTHS:
            kv_mb = kv_cache_mb(args.num_layers, args.num_kv_heads,
                                 args.head_dim, ctx, bpe)
            kv_gb_ = kv_mb / 1024.0
            rows.append({
                "precision": prec,
                "context_tokens": ctx,
                "kv_cache_mb": round(kv_mb, 3),
                "kv_cache_gb": round(kv_gb_, 6),
                "model_weights_gb": args.model_gb,
                "measurement_type": "analytical",
                "data_source": "formula: 2*L*H_kv*d_head*ctx*bpe",
            })
        results_by_prec[prec] = rows
        all_rows.extend(rows)

    # Verify formula at 1000 tokens (F16)
    check_mb = kv_cache_mb(args.num_layers, args.num_kv_heads,
                            args.head_dim, 1000, 2.0)
    print(f"\n  Formula check — F16 @ 1000 tokens: {check_mb:.2f} MB (expect ~22.5 MB for TinyLlama)")

    save_csv(all_rows, out_dir / "08_kvcache_size_vs_context.csv")

    # ── Crossover estimates ───────────────────────────────────────────────────
    crossovers: dict[str, dict[str, int | None]] = {}
    print("\n  System-level RAM pressure crossover estimates:")
    print(f"  (model_weights={args.model_gb}GB + KV-cache + {SYSTEM_OVERHEAD_GB}GB overhead > RAM)")
    for ram_name, ram_gb in RAM_CONFIGS.items():
        crossovers[ram_name] = {}
        for prec, bpe in PRECISIONS.items():
            cx = system_level_crossover(args.model_gb, ram_gb,
                                        args.num_layers, args.num_kv_heads,
                                        args.head_dim, bpe)
            crossovers[ram_name][prec] = cx
            label = f"{cx:,} tokens" if cx is not None else ">16384 tokens"
            print(f"    {ram_name} + {prec}: {label}")

    # Save summary JSON
    summary = {
        "model_gb": args.model_gb,
        "num_layers": args.num_layers,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "system_overhead_gb": SYSTEM_OVERHEAD_GB,
        "formula_note": "M_KV = 2 * L * H_kv * d_head * context * bytes_per_elem",
        "formula_check_f16_1000tok_mb": round(check_mb, 2),
        "ram_pressure_crossover_tokens": crossovers,
        "measurement_type": "analytical",
        "crossover_note": (
            "Crossover estimates include system_overhead_gb for Metal pools, "
            "runtime allocations, OS reservation, and transient buffers. "
            "Raw KV-cache alone is much smaller than the total RAM footprint."
        ),
    }
    save_json(summary, out_dir / "08_kvcache_size_vs_context_summary.json")
    save_metadata(
        model_name=f"arch:L={args.num_layers},H_kv={args.num_kv_heads},d={args.head_dim}",
        device="none (analytical)",
        device_name=args.device_name,
        precision="F16/Q8_0/Q4_K_M (analytical)",
        measurement_type="analytical",
        output_dir=out_dir,
        filename="08_kvcache_size_vs_context_metadata.json",
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"KV-Cache Memory vs Context Length  [analytical]\n"
        f"{args.num_layers}-layer model · {args.model_gb:.1f} GB weights · "
        f"GQA {args.num_kv_heads} KV heads · d={args.head_dim}",
        fontsize=12, fontweight="bold"
    )

    # Plot 1: KV-cache size by precision (MB)
    ax = axes[0]
    for prec, rows in results_by_prec.items():
        ctxs = [r["context_tokens"] for r in rows]
        mbs  = [r["kv_cache_mb"]    for r in rows]
        ax.plot(ctxs, mbs, linewidth=2, color=PRECISION_COLORS[prec], label=prec)
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("KV-cache size (MB)")
    ax.set_title("KV-cache size by precision [analytical]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.97, "Formula: 2·L·H_kv·d_head·ctx·bpe",
            transform=ax.transAxes, fontsize=7, va="top", color="#555")

    # Plot 2: System-level total memory vs RAM limits
    ax = axes[1]
    for prec, rows in results_by_prec.items():
        ctxs     = [r["context_tokens"] for r in rows]
        total_gb = [r["kv_cache_gb"] + args.model_gb + SYSTEM_OVERHEAD_GB
                    for r in rows]
        ax.plot(ctxs, total_gb, linewidth=2, color=PRECISION_COLORS[prec], label=prec)

    for ram_name, ram_gb in RAM_CONFIGS.items():
        ax.axhline(ram_gb, linestyle="--", linewidth=1.5,
                   color=RAM_COLORS[ram_name], label=f"{ram_name} limit")

    ax.fill_between(CONTEXT_LENGTHS, 0, args.model_gb + SYSTEM_OVERHEAD_GB,
                    alpha=0.08, color="#888",
                    label=f"Model + overhead ({args.model_gb + SYSTEM_OVERHEAD_GB:.1f} GB)")
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("System memory estimate (GB)")
    ax.set_title("System-level memory vs RAM [estimate]")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.97,
            f"Includes {SYSTEM_OVERHEAD_GB} GB overhead\n(Metal, runtime, OS)",
            transform=ax.transAxes, fontsize=7, va="top", color="#555")

    # Plot 3: Crossover heatmap
    ax = axes[2]
    ram_names  = list(RAM_CONFIGS.keys())
    prec_names = list(PRECISIONS.keys())
    matrix = np.zeros((len(ram_names), len(prec_names)))
    for ri, rn in enumerate(ram_names):
        for pi, pn in enumerate(prec_names):
            cx = crossovers[rn][pn]
            matrix[ri, pi] = cx if cx is not None else 16384

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=8192)
    ax.set_xticks(range(len(prec_names)))
    ax.set_yticks(range(len(ram_names)))
    ax.set_xticklabels(prec_names)
    ax.set_yticklabels(ram_names)
    ax.set_title("System-level RAM pressure crossover\n(tokens) — estimate")
    plt.colorbar(im, ax=ax, label="Crossover context (tokens)")
    for ri in range(len(ram_names)):
        for pi in range(len(prec_names)):
            val = crossovers[ram_names[ri]][prec_names[pi]]
            lbl = f"{val:,}" if val is not None else ">16384"
            ax.text(pi, ri, lbl, ha="center", va="center", fontsize=9,
                    fontweight="bold",
                    color="white" if (val or 99999) < 1500 else "black")

    plt.tight_layout()
    out_png = out_dir / "08_kvcache_size_vs_context.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {out_png}")
    plt.close()

    print("\n  measurement_type: analytical")
    print("  NOTE: KV-cache sizes are from the formula only.")
    print("        RAM crossover estimates include system overhead and are empirical.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KV-cache size vs context length (analytical)")
    parser.add_argument("--model_gb",      type=float, default=DEFAULT_MODEL_GB,
                        help="Model weight size in GB (default: 2.2 for TinyLlama-1.1B F16)")
    parser.add_argument("--num_layers",    type=int,   default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--num_heads",     type=int,   default=DEFAULT_NUM_HEADS,
                        help="Query attention heads (reference only, not used in formula)")
    parser.add_argument("--num_kv_heads",  type=int,   default=DEFAULT_NUM_KV_HEADS,
                        help="KV heads (GQA: 4 for TinyLlama)")
    parser.add_argument("--head_dim",      type=int,   default=DEFAULT_HEAD_DIM)
    parser.add_argument("--device_name",   default="Mac_M4_16GB",
                        help="Device label for output path")
    parser.add_argument("--output_dir",    default="results")
    main(parser.parse_args())
