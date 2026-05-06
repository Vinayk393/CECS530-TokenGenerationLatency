"""
analysis/generate_research_graphs.py
======================================
Generates all paper figures from benchmark CSV outputs.

PATH INDEPENDENCE: All paths derived from this file's location.
No hardcoded absolute paths. Works from any directory on any machine.

    REPO_ROOT = Path(__file__).resolve().parents[1]
    RESULTS   = REPO_ROOT / "results"
    GRAPHS    = REPO_ROOT / "graphs"

Usage:
    python analysis/generate_research_graphs.py          # generate all figures
    python analysis/generate_research_graphs.py --check_only   # verify CSVs exist
    make graphs

Figure-to-paper mapping:
    fig1_ttft_vs_prompt_length.png      → Figure 2 (TTFT, measured)
    fig2_ptl_vs_context.png             → Figure 3 (PTL, measured)
    fig3_inter_token_timeline.png       → Figure 4 (timeline, measured)
    fig4_throughput_bw_utilization.png  → Figure 5 (throughput, measured)
    fig5_latency_decomposition.png      → Figure 6 (decomp, estimated)
    fig6_cold_warm_run.png              → Figure 7 (cold/warm, measured)
    fig7_quantization_speedup.png       → Figure 8 (quant, modeled)
    fig8_tail_latency.png               → Figure 9 (tail, measured)
    fig9_model_scaling.png              → Figure 10 (scaling, modeled)
"""

import argparse, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── Repo-relative paths — NO hardcoded /mnt/... paths ─────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS   = REPO_ROOT / "results"
GRAPHS    = REPO_ROOT / "graphs"
GRAPHS.mkdir(exist_ok=True)

DEVICES = {
    "Mac_M4_16GB": {"color": "#2563EB", "label": "M4 16GB (120 GB/s)", "bw": 120},
    "Mac_M2_8GB":  {"color": "#DC2626", "label": "M2 8GB (100 GB/s)",  "bw": 100},
}
DPI = 180   # publication-quality (paper §1.2)


def load(device: str, filename: str) -> pd.DataFrame | None:
    p = RESULTS / device / filename
    return pd.read_csv(p) if p.exists() and p.stat().st_size > 0 else None


def load_both(filename: str) -> dict:
    return {d: df for d in DEVICES if (df := load(d, filename)) is not None}


def check_required_csvs() -> bool:
    required = [
        "01_ttft_vs_prompt_length.csv",
        "02_per_token_latency_vs_context.csv",
        "04_throughput_vs_prompt_length.csv",
        "05_inter_token_latency_timeline.csv",
        "06_cold_vs_warm_run.csv",
    ]
    ok = True
    print("\nChecking required CSVs:")
    for dev in DEVICES:
        for fn in required:
            p = RESULTS / dev / fn
            status = "[OK]     " if p.exists() else "[MISSING]"
            print(f"  {status} {dev}/{fn}")
            if not p.exists():
                ok = False
    print()
    return ok


def style(ax, title, xlabel, ylabel, legend=True):
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    if legend:
        ax.legend(fontsize=8)


def save_fig(fig, name):
    out = GRAPHS / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → graphs/{name}")


# ── Figure 1: TTFT vs prompt length (paper Figure 2) ─────────────────────────

def fig1_ttft(data):
    if not data: return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for dev, df in data.items():
        cfg = DEVICES[dev]
        col = "ttft_mean_ms" if "ttft_mean_ms" in df.columns else "ttft_ms"
        ax1.plot(df["prompt_length"], df[col], "o-",
                 color=cfg["color"], lw=2, label=cfg["label"], ms=5)
        if "ttft_std_ms" in df.columns:
            ax1.fill_between(df["prompt_length"],
                             df[col]-df["ttft_std_ms"], df[col]+df["ttft_std_ms"],
                             alpha=0.12, color=cfg["color"])
        ax2.semilogy(df["prompt_length"], df[col], "o-",
                     color=cfg["color"], lw=2, label=cfg["label"], ms=5)
    style(ax1, "TTFT vs Prompt Length [measured]", "Prompt length (tokens)", "TTFT (ms)")
    style(ax2, "TTFT vs Prompt Length (log scale) [measured]", "Prompt length (tokens)", "TTFT (ms)")
    fig.tight_layout()
    save_fig(fig, "fig1_ttft_vs_prompt_length.png")


# ── Figure 2: PTL vs context length (paper Figure 3) ─────────────────────────

def fig2_ptl(data):
    if not data: return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for dev, df in data.items():
        cfg = DEVICES[dev]
        col = "ptl_mean_ms" if "ptl_mean_ms" in df.columns else "ptl_ms"
        ax1.plot(df["context_length"], df[col], "s-",
                 color=cfg["color"], lw=2, label=cfg["label"], ms=5)
        if "ptl_std_ms" in df.columns:
            ax1.fill_between(df["context_length"],
                             df[col]-df["ptl_std_ms"], df[col]+df["ptl_std_ms"],
                             alpha=0.12, color=cfg["color"])
        # Marginal cost per token
        diff = df[col].diff()
        ctx_mid = (df["context_length"].values[:-1] + df["context_length"].values[1:]) / 2
        ax2.bar(ctx_mid, diff.values[1:], color=cfg["color"], alpha=0.6,
                label=cfg["label"], width=50)

    ax1.axvline(256, color="#888", linestyle=":", lw=1.5, label="~256 tok inflection")
    style(ax1, "PTL vs Context Length [measured]", "Context length (tokens)", "PTL (ms)")
    style(ax2, "Marginal Latency Cost per Context Step", "Mid-point context (tokens)", "ΔPTL (ms)")
    fig.tight_layout()
    save_fig(fig, "fig2_ptl_vs_context.png")


# ── Figure 3: Inter-token timeline (paper Figure 4) ──────────────────────────

def fig3_timeline(data):
    if not data: return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    idx = 0
    for dev, df in data.items():
        cfg = DEVICES[dev]
        for pl_label in ["128", "512"]:
            if idx >= 4: break
            # Filter by prompt_length if column exists
            sub = df[df["prompt_length"]==int(pl_label)] if "prompt_length" in df.columns else df
            if len(sub) == 0:
                sub = df
            ax = axes[idx]
            ax.plot(sub["token_index"].values, sub["ptl_ms"].values,
                    color=cfg["color"], lw=0.9, alpha=0.85)
            mean_val = sub["ptl_ms"].mean()
            ax.axhline(mean_val, color="#555", lw=1, linestyle="--",
                       label=f"mean={mean_val:.1f}ms")
            ax.set_title(f"{cfg['label']} — Prompt {pl_label} tok [measured]", fontsize=9)
            ax.set_xlabel("Token index", fontsize=8)
            ax.set_ylabel("PTL (ms)", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            idx += 1
    for i in range(idx, 4):
        axes[i].set_visible(False)
    fig.suptitle("Every-Token Timeline — Three Decode Phases", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig3_inter_token_timeline.png")


# ── Figure 4: Throughput & BW utilization (paper Figure 5) ───────────────────

def fig4_throughput(data):
    if not data: return
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    for dev, df in data.items():
        cfg = DEVICES[dev]
        pl  = df["prompt_length"] if "prompt_length" in df.columns else df.iloc[:,0]
        thr = df["throughput_tok_s"] if "throughput_tok_s" in df.columns else df.iloc[:,1]
        ax1.bar(pl + (10 if "M4" in dev else -10), thr, width=18,
                color=cfg["color"], alpha=0.85, label=cfg["label"])
        if "ptl_ms" in df.columns:
            ax2.plot(pl, df["ptl_ms"], "o-", color=cfg["color"], lw=2,
                     label=cfg["label"], ms=5)
        if "bw_utilization_pct" in df.columns:
            ax3.plot(pl, df["bw_utilization_pct"], "^-", color=cfg["color"],
                     lw=2, label=cfg["label"], ms=5)

    style(ax1, "Decode Throughput [measured]", "Prompt length (tokens)", "Throughput (tok/s)")
    style(ax2, "PTL vs Prompt Length [measured]", "Prompt length (tokens)", "PTL (ms)")
    style(ax3, "BW Utilization [estimated]", "Prompt length (tokens)", "BW Utilization (%)")
    ax3.axhline(82, color="#888", linestyle="--", lw=1.2, label="~82% M4 ceiling")
    ax3.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, "fig4_throughput_bw_utilization.png")


# ── Figure 5: Latency decomposition (paper Figure 6) ─────────────────────────

def fig5_decomp(data):
    if not data: return
    dev, df = next(iter(data.items()))
    ctx_vals = sorted(df["context_tokens"].unique()) if "context_tokens" in df.columns else [512]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, 6))

    for ctx in ctx_vals[:4]:
        sub = df[df["context_tokens"]==ctx] if "context_tokens" in df.columns else df
        comps = sub["component"].tolist()
        vals  = sub["ms"].tolist()
        bottom = 0
        for i, (c, v) in enumerate(zip(comps, vals)):
            ax1.bar(ctx, v, bottom=bottom, color=colors[i%6], alpha=0.85,
                    label=c if ctx==ctx_vals[0] else "")
            bottom += v

    ax1.set_title(f"Latency Decomposition — {dev} [estimated; not kernel-profiled]",
                  fontsize=9, fontweight="bold")
    ax1.set_xlabel("Context length (tokens)")
    ax1.set_ylabel("Latency (ms)")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Pie at 512 tokens
    sub512 = df[df["context_tokens"]==512] if "context_tokens" in df.columns else df
    ax2.pie(sub512["ms"].values, labels=sub512["component"].values,
            colors=colors[:len(sub512)], autopct="%1.1f%%", textprops={"fontsize":7})
    ax2.set_title("Component Share @ 512 tokens [estimated]", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "fig5_latency_decomposition.png")


# ── Figure 6: Cold vs warm run (paper Figure 7) ───────────────────────────────

def fig6_cold_warm(data):
    if not data: return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for dev, df in data.items():
        cfg = DEVICES[dev]
        run_col  = "run_index" if "run_index" in df.columns else df.columns[0]
        ttft_col = "ttft_ms" if "ttft_ms" in df.columns else df.columns[1]
        ax1.plot(df[run_col], df[ttft_col], "o-", color=cfg["color"],
                 lw=2, label=cfg["label"], ms=6)
        warm = df[df[run_col] >= 2][ttft_col].mean() if len(df) > 2 else df[ttft_col].mean()
        cold = df[df[run_col] == 0][ttft_col].values[0] if len(df) > 0 else 0
        ax2.bar([dev.replace("_", "\n")], [cold/warm], color=cfg["color"], alpha=0.8,
                label=f"cold/warm ≈ {cold/warm:.1f}×")

    style(ax1, "Cold vs Warm TTFT [measured]", "Run index", "TTFT (ms)")
    ax1.axvline(1.5, color="#888", linestyle=":", lw=1.2, label="steady-state")
    ax1.legend(fontsize=7)
    ax2.set_title("Cold/Warm TTFT Ratio [measured]", fontsize=9, fontweight="bold")
    ax2.set_ylabel("cold TTFT / warm TTFT")
    ax2.axhline(3.0, color="#888", linestyle="--", lw=1.2, label="paper: ~3×")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "fig6_cold_warm_run.png")


# ── Figure 7: Quantization speedup (paper Figure 8) ──────────────────────────

def fig7_quant(data):
    if not data: return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    prec_colors = {"F16": "#EF4444", "Q8_0": "#10B981", "Q4_K_M": "#3B82F6"}
    dev, df = next(iter(data.items()))
    prec_col = "precision" if "precision" in df.columns else df.columns[0]
    ptl_col  = "ptl_ms"   if "ptl_ms"   in df.columns else df.columns[2]
    pl_col   = "prompt_length" if "prompt_length" in df.columns else df.columns[1]

    for prec, grp in df.groupby(prec_col):
        mtype  = grp.get("measurement_type", pd.Series(["unknown"])).iloc[0]
        style_ = "-o" if mtype == "measured" else "--s"
        ax1.plot(grp[pl_col], grp[ptl_col], style_,
                 color=prec_colors.get(prec, "#888"),
                 lw=2, label=f"{prec} [{mtype}]", ms=5)
        if "speedup_vs_f16" in grp.columns and prec != "F16":
            ax2.plot(grp[pl_col], grp["speedup_vs_f16"], style_,
                     color=prec_colors.get(prec, "#888"),
                     lw=2, label=f"{prec} [{mtype}]", ms=5)

    style(ax1, "PTL by Precision [F16: measured; Q4/Q8: modeled]",
          "Prompt length (tokens)", "PTL (ms)")
    ax2.axhline(1.0, color="#EF4444", linestyle="--", lw=1.5, label="F16 baseline")
    style(ax2, "Speedup vs F16 [modeled from F16 baseline]",
          "Prompt length (tokens)", "Speedup (×)")
    fig.tight_layout()
    save_fig(fig, "fig7_quantization_speedup.png")


# ── Figure 8: Tail latency distribution (paper Figure 9) ─────────────────────

def fig8_tail(data):
    if not data: return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for dev, df in data.items():
        cfg     = DEVICES[dev]
        ptl_col = "ptl_ms" if "ptl_ms" in df.columns else df.columns[1]
        vals    = sorted(df[ptl_col].dropna().tolist())
        pcts    = [100*i/len(vals) for i in range(len(vals))]
        median  = np.median(vals)
        p90     = np.percentile(vals, 90)
        p99     = np.percentile(vals, 99)

        ax1.plot(vals, pcts, color=cfg["color"], lw=2, label=cfg["label"])
        ax1.axvline(p99, color=cfg["color"], linestyle="--", lw=1,
                    label=f"p99={p99:.0f}ms", alpha=0.7)
        ax2.boxplot(vals, positions=[list(DEVICES.keys()).index(dev)],
                    patch_artist=True,
                    boxprops=dict(facecolor=cfg["color"], alpha=0.6))

    ax1.axhline(99, color="#888", linestyle=":", lw=1)
    style(ax1, "Tail Latency Distribution [measured, n=50]",
          "PTL (ms)", "Percentile (%)")
    ax2.set_title("PTL Box Plot [measured, n=50]", fontsize=9, fontweight="bold")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["M4 16GB", "M2 8GB"], fontsize=8)
    ax2.set_ylabel("PTL (ms)", fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "fig8_tail_latency_distribution.png")


# ── Figure 9: Model size scaling (paper Figure 10) ───────────────────────────

def fig9_scaling(data):
    # Modeled projections from paper Results 9
    # TinyLlama measured, 1B/3B modeled via bandwidth scaling
    model_sizes = [1.1, 1.0, 3.0]
    ptl_m4 = {"measured": [30.1], "modeled": [35.0, 82.0]}  # @512 tok
    ptl_m2 = {"measured": [45.8], "modeled": [53.0, 161.0]}
    ctx    = 512

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

    for dev, ptl_d, cfg in [("Mac_M4_16GB", ptl_m4, DEVICES["Mac_M4_16GB"]),
                              ("Mac_M2_8GB",  ptl_m2, DEVICES["Mac_M2_8GB"])]:
        ax1.plot([1.1], ptl_d["measured"], "o", color=cfg["color"], ms=8,
                 label=f"{cfg['label']} measured")
        ax1.plot([1.0, 3.0], ptl_d["modeled"], "--s", color=cfg["color"],
                 ms=6, alpha=0.7, label=f"{cfg['label']} modeled")

    ax1.axhline(1000/6, color="#888", linestyle=":", lw=1.2,
                label="~6 tok/s threshold")
    style(ax1, "PTL vs Model Size [solid: measured; dashed: modeled]",
          "Model size (B params)", "PTL @ 512-tok context (ms)")
    ax1.set_xticks([1.0, 1.1, 3.0])
    ax1.set_xticklabels(["1B", "1.1B\n(TinyLlama)", "3B"], fontsize=7)

    # TTFT scaling
    ttft_sizes = [1.1, 1.0, 3.0]
    ttft_m4_m = [77.4]; ttft_m4_md = [65.0, 190.0]
    ttft_m2_m = [121.7]; ttft_m2_md = [105.0, 310.0]
    for ptl_m, ptl_md, cfg in [(ttft_m4_m, ttft_m4_md, DEVICES["Mac_M4_16GB"]),
                                 (ttft_m2_m, ttft_m2_md, DEVICES["Mac_M2_8GB"])]:
        ax2.plot([1.1], ptl_m, "o", color=cfg["color"], ms=8, label=cfg["label"])
        ax2.plot([1.0,3.0], ptl_md, "--s", color=cfg["color"], ms=6, alpha=0.7)
    style(ax2, "TTFT vs Model Size @ 512-tok prompt [measured+modeled]",
          "Model size (B params)", "TTFT (ms)")

    # Throughput
    tput_m4 = {"measured":[28.6], "modeled":[24.0, 9.5]}
    tput_m2 = {"measured":[18.9], "modeled":[15.0, 5.2]}
    for ptl_d, cfg in [(tput_m4, DEVICES["Mac_M4_16GB"]),
                        (tput_m2, DEVICES["Mac_M2_8GB"])]:
        ax3.plot([1.1], ptl_d["measured"], "o", color=cfg["color"], ms=8, label=cfg["label"])
        ax3.plot([1.0,3.0], ptl_d["modeled"], "--s", color=cfg["color"], ms=6, alpha=0.7)
    ax3.axhline(6, color="#888", linestyle=":", lw=1.2, label="6 tok/s threshold")
    style(ax3, "Throughput vs Model Size [measured+modeled]",
          "Model size (B params)", "Throughput (tok/s)")

    fig.tight_layout()
    save_fig(fig, "fig9_model_scaling.png")


# ── Main ────────────────────────────────────────────────────────────────────────

def main(args):
    print(f"\nGraph generator")
    print(f"  Repo root:    {REPO_ROOT}")
    print(f"  Results dir:  {RESULTS}")
    print(f"  Graphs dir:   {GRAPHS}")

    if args.check_only:
        ok = check_required_csvs()
        sys.exit(0 if ok else 1)

    check_required_csvs()
    print("Generating figures...\n")

    fig1_ttft(load_both("01_ttft_vs_prompt_length.csv"))
    fig2_ptl(load_both("02_per_token_latency_vs_context.csv"))
    fig3_timeline(load_both("05_inter_token_latency_timeline.csv"))
    fig4_throughput(load_both("04_throughput_vs_prompt_length.csv"))
    fig5_decomp(load_both("09_latency_decomposition.csv"))
    fig6_cold_warm(load_both("06_cold_vs_warm_run.csv"))
    fig7_quant(load_both("07_quantization_speedup.csv"))
    fig8_tail(load_both("10_tail_latency_distribution.csv"))
    fig9_scaling({})  # uses hardcoded paper values

    pngs = list(GRAPHS.glob("*.png"))
    print(f"\n{len(pngs)} figure(s) saved to graphs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paper figures from CSVs")
    parser.add_argument("--check_only", action="store_true",
                        help="Only verify required CSVs exist")
    main(parser.parse_args())
