"""
generate_research_graphs.py
----------------------------
Research-quality plots — matplotlib + seaborn only.
Muted professional palette, white background, publication-ready.
"""

import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
IN   = "/mnt/user-data/uploads"
IN2  = "/mnt/user-data/outputs"      # fixed 07 and 11
OUT  = "/mnt/user-data/outputs/graphs_v2"
os.makedirs(OUT, exist_ok=True)

def read(fname, alt=None):
    for base in [IN, IN2, alt]:
        if base is None: continue
        path = os.path.join(base, fname)
        if os.path.exists(path):
            with open(path) as f:
                return list(csv.DictReader(f))
    raise FileNotFoundError(fname)

# ── Design constants ──────────────────────────────────────────────────────────
M4  = "#1f77b4"   # matplotlib default blue
M2  = "#ff7f0e"   # matplotlib default orange

# Component palette (colorblind-safe, muted)
COMP_COLORS = {
    "self_attn":    "#4878d0",
    "mlp":          "#6acc65",
    "kv_read_write":"#d65f5f",
    "layernorm":    "#b47cc7",
    "lm_head":      "#c4ad66",
    "overhead":     "#77bedb",
}
COMP_LABELS = {
    "self_attn":    "Attention",
    "mlp":          "MLP",
    "kv_read_write":"KV Read/Write",
    "layernorm":    "LayerNorm",
    "lm_head":      "LM Head",
    "overhead":     "Overhead",
}

PREC_COLORS = {"F16": "#d62728", "Q8_0": "#ff7f0e", "Q4_K_M": "#2ca02c"}
MODEL_COLORS = {
    "TinyLlama-1.1B": "#1f77b4",
    "LLaMA-3.2-1B":   "#2ca02c",
    "LLaMA-3.2-3B":   "#d62728",
}

# ── Global style ──────────────────────────────────────────────────────────────
sns.set_theme(style="white", font_scale=1.0)
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.facecolor":     "white",
    "figure.facecolor":   "white",
    "axes.grid":          True,
    "grid.linestyle":     "--",
    "grid.alpha":         0.4,
    "grid.color":         "#cccccc",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.titlesize":     14,
    "axes.titleweight":   "bold",
    "axes.titlepad":      10,
    "axes.labelsize":     12,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "legend.framealpha":  0.85,
    "legend.edgecolor":   "#cccccc",
    "lines.linewidth":    2.0,
    "lines.markersize":   7,
})

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  {name}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  TTFT vs Prompt Length
# ─────────────────────────────────────────────────────────────────────────────
print("01 · TTFT vs Prompt Length")
rows = read("01_ttft_summary.csv")
pl   = [int(r["prompt_length_tokens"]) for r in rows]
m4m  = [float(r["m4_mean_ms"])  for r in rows]
m4s  = [float(r["m4_stdev_ms"]) for r in rows]
m4p  = [float(r["m4_p90_ms"])   for r in rows]
m2m  = [float(r["m2_mean_ms"])  for r in rows]
m2s  = [float(r["m2_stdev_ms"]) for r in rows]
m2p  = [float(r["m2_p90_ms"])   for r in rows]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, yscale in zip(axes, ["linear", "log"]):
    ax.plot(pl, m4m, color=M4, marker="o", label="Mac M4 16GB — mean")
    ax.fill_between(pl,
        [a-b for a,b in zip(m4m,m4s)],
        [a+b for a,b in zip(m4m,m4s)],
        color=M4, alpha=0.15, label="M4 ±1 σ")
    ax.plot(pl, m4p, color=M4, linestyle=":", linewidth=1.4,
            marker="", label="M4 p90")

    ax.plot(pl, m2m, color=M2, marker="s", label="Mac M2 8GB — mean")
    ax.fill_between(pl,
        [a-b for a,b in zip(m2m,m2s)],
        [a+b for a,b in zip(m2m,m2s)],
        color=M2, alpha=0.15, label="M2 ±1 σ")
    ax.plot(pl, m2p, color=M2, linestyle=":", linewidth=1.4,
            marker="", label="M2 p90")

    if yscale == "log":
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_title("TTFT vs Prompt Length (log scale)")
    else:
        ax.set_title("TTFT vs Prompt Length")

    ax.set_xlabel("Prompt Length (tokens)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_xticks(pl)
    ax.legend(ncol=2)

plt.tight_layout()
save(fig, "01_ttft_vs_prompt_length.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Per-Token Latency vs Context Length
# ─────────────────────────────────────────────────────────────────────────────
print("02 · Per-Token Latency vs Context Length")
rows = read("02_per_token_latency_summary.csv")
ctx  = [int(r["context_length_tokens"])  for r in rows]
m4m  = [float(r["m4_mean_ptl_ms"])       for r in rows]
m4s  = [float(r["m4_stdev_ptl_ms"])      for r in rows]
m4p  = [float(r["m4_p90_ptl_ms"])        for r in rows]
m2m  = [float(r["m2_mean_ptl_ms"])       for r in rows]
m2s  = [float(r["m2_stdev_ptl_ms"])      for r in rows]
m2p  = [float(r["m2_p90_ptl_ms"])        for r in rows]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: mean + band + p90
ax = axes[0]
ax.plot(ctx, m4m, color=M4, marker="o", label="Mac M4 — mean")
ax.fill_between(ctx,
    [a-b for a,b in zip(m4m,m4s)],
    [a+b for a,b in zip(m4m,m4s)],
    color=M4, alpha=0.15, label="M4 ±1 σ")
ax.plot(ctx, m4p, color=M4, linestyle=":", linewidth=1.4, label="M4 p90")

ax.plot(ctx, m2m, color=M2, marker="s", label="Mac M2 — mean")
ax.fill_between(ctx,
    [a-b for a,b in zip(m2m,m2s)],
    [a+b for a,b in zip(m2m,m2s)],
    color=M2, alpha=0.15, label="M2 ±1 σ")
ax.plot(ctx, m2p, color=M2, linestyle=":", linewidth=1.4, label="M2 p90")

ax.axvline(256, color="grey", linewidth=1.2, linestyle="--", alpha=0.7)
ax.text(265, max(m2m)*0.92, "KV-cache\ninflection\n~256 tok",
        fontsize=9, color="grey", va="top")
ax.set_xlabel("Context Length (tokens)")
ax.set_ylabel("Per-Token Latency (ms)")
ax.set_title("Per-Token Latency vs Context Length")
ax.set_xticks(ctx)
ax.legend(ncol=2)

# Right: marginal cost
ax = axes[1]
deltas4 = [m4m[i]-m4m[i-1] for i in range(1, len(m4m))]
deltas2 = [m2m[i]-m2m[i-1] for i in range(1, len(m2m))]
mid = [(ctx[i]+ctx[i-1])/2 for i in range(1, len(ctx))]
x   = np.arange(len(mid))
w   = 0.35
ax.bar(x - w/2, deltas4, w, color=M4, alpha=0.75, label="Mac M4")
ax.bar(x + w/2, deltas2, w, color=M2, alpha=0.75, label="Mac M2")
ax.set_xticks(x)
ax.set_xticklabels([f"{int(m)}" for m in mid])
ax.set_xlabel("Mid-point Context (tokens)")
ax.set_ylabel("Latency Increase per Step (ms)")
ax.set_title("Marginal Latency Cost per Context Step")
ax.legend()

plt.tight_layout()
save(fig, "02_ptl_vs_context_length.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Inter-Token Latency Timeline
# ─────────────────────────────────────────────────────────────────────────────
print("03 · Inter-Token Latency Timeline")
r128 = read("05_inter_token_timeline_prompt128.csv")
r512 = read("05_inter_token_timeline_prompt512.csv")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.subplots_adjust(hspace=0.38, wspace=0.3)

for row_i, (rows, pl_label) in enumerate([(r128,"128"), (r512,"512")]):
    toks   = [int(r["token_index"])        for r in rows]
    m4_lat = [float(r["m4_latency_ms"])    for r in rows]
    m2_lat = [float(r["m2_latency_ms"])    for r in rows]
    m4_cum = [float(r["m4_cumulative_ms"]) for r in rows]
    m2_cum = [float(r["m2_cumulative_ms"]) for r in rows]

    W = 7
    def ma(v):
        return np.convolve(v, np.ones(W)/W, mode="valid").tolist()
    ma_x = toks[W-1:]

    # Left: per-token bars + MA
    ax = axes[row_i][0]
    ax.bar(toks, m4_lat, color=M4, alpha=0.30, width=0.8, label="M4 tokens")
    ax.bar(toks, m2_lat, color=M2, alpha=0.18, width=0.8, label="M2 tokens")
    ax.plot(ma_x, ma(m4_lat), color=M4, linewidth=2.0, label=f"M4 {W}-tok avg")
    ax.plot(ma_x, ma(m2_lat), color=M2, linewidth=2.0, label=f"M2 {W}-tok avg")

    ss_m4 = np.mean(m4_lat[2:12])
    ax.axhline(ss_m4, color=M4, linewidth=1.0, linestyle="--", alpha=0.6,
               label=f"M4 steady ≈{ss_m4:.1f} ms")

    ax.annotate(f"TTFT  {m4_lat[0]:.0f} ms",
                xy=(1, m4_lat[0]), xytext=(6, m4_lat[0]*0.82),
                fontsize=9, color=M4,
                arrowprops=dict(arrowstyle="->", color=M4, lw=1.0))

    ax.set_xlabel("Token Index")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Token Latency Timeline — Prompt {pl_label} tokens")
    ax.legend(fontsize=9, ncol=2)

    # Right: cumulative
    ax = axes[row_i][1]
    ax.plot(toks, m4_cum, color=M4, linewidth=2.0, label="Mac M4")
    ax.fill_between(toks, 0, m4_cum, color=M4, alpha=0.07)
    ax.plot(toks, m2_cum, color=M2, linewidth=2.0, label="Mac M2")
    ax.fill_between(toks, 0, m2_cum, color=M2, alpha=0.07)

    ax.annotate(f"{m4_cum[-1]/1000:.2f} s",
                xy=(toks[-1], m4_cum[-1]),
                xytext=(toks[-1]*0.65, m4_cum[-1]*0.72),
                fontsize=9, color=M4,
                arrowprops=dict(arrowstyle="->", color=M4, lw=1.0))
    ax.annotate(f"{m2_cum[-1]/1000:.2f} s",
                xy=(toks[-1], m2_cum[-1]),
                xytext=(toks[-1]*0.65, m2_cum[-1]*0.92),
                fontsize=9, color=M2,
                arrowprops=dict(arrowstyle="->", color=M2, lw=1.0))

    ax.set_xlabel("Token Index")
    ax.set_ylabel("Cumulative Time (ms)")
    ax.set_title(f"Cumulative Wall Time — Prompt {pl_label} tokens")
    ax.legend()

plt.tight_layout()
save(fig, "03_inter_token_timeline.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Latency Decomposition
# ─────────────────────────────────────────────────────────────────────────────
print("04 · Latency Decomposition")
rows  = read("09_latency_decomposition_long.csv")
comps = ["self_attn","mlp","kv_read_write","layernorm","lm_head","overhead"]
pls   = [64, 128, 256, 512]

def gv(pl, dev, comp):
    key = "m4_latency_ms" if dev=="m4" else "m2_latency_ms"
    for r in rows:
        if int(r["prompt_length_tokens"])==pl and r["component"]==comp:
            return float(r[key])
    return 0.0

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

# Stacked bars M4
ax = axes[0]
x  = np.arange(len(pls))
bot = np.zeros(len(pls))
for comp in comps:
    vals = [gv(pl,"m4",comp) for pl in pls]
    ax.bar(x, vals, 0.5, bottom=bot,
           color=COMP_COLORS[comp], label=COMP_LABELS[comp], alpha=0.9)
    bot += np.array(vals)
ax.set_xticks(x)
ax.set_xticklabels([f"{p} tok" for p in pls])
ax.set_xlabel("Prompt Length (tokens)")
ax.set_ylabel("Per-Token Latency (ms)")
ax.set_title("Latency Breakdown — Mac M4")
ax.legend(fontsize=8.5, loc="upper left")

# Stacked bars M2
ax = axes[1]
bot = np.zeros(len(pls))
for comp in comps:
    vals = [gv(pl,"m2",comp) for pl in pls]
    ax.bar(x, vals, 0.5, bottom=bot,
           color=COMP_COLORS[comp], alpha=0.9)
    bot += np.array(vals)
ax.set_xticks(x)
ax.set_xticklabels([f"{p} tok" for p in pls])
ax.set_xlabel("Prompt Length (tokens)")
ax.set_ylabel("Per-Token Latency (ms)")
ax.set_title("Latency Breakdown — Mac M2")

# Donut @ 512 tok M4
ax = axes[2]
vals512 = [gv(512,"m4",c) for c in comps]
colors  = [COMP_COLORS[c] for c in comps]
labels  = [COMP_LABELS[c] for c in comps]
wedges, _, autotexts = ax.pie(
    vals512, colors=colors, startangle=90,
    autopct="%1.1f%%", pctdistance=0.78,
    wedgeprops=dict(width=0.48, edgecolor="white", linewidth=1.8))
for at in autotexts:
    at.set_fontsize(8.5)
    at.set_fontweight("bold")
ax.set_title("Component Share — M4 @ 512 tokens")
ax.legend(wedges, labels, loc="lower center",
          bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8.5)

plt.tight_layout()
save(fig, "04_latency_decomposition.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Quantization Speedup
# ─────────────────────────────────────────────────────────────────────────────
print("05 · Quantization Speedup")
rows = read("07_quantization_speedup.csv", alt=IN2)
pls  = sorted(set(int(r["prompt_length_tokens"]) for r in rows))

def qv(pl, dev, field, prec):
    for r in rows:
        if int(r["prompt_length_tokens"])==pl and r["precision"]==prec:
            return float(r[f"{dev}_{field}"])
    return 0.0

precs  = ["F16","Q8_0","Q4_K_M"]
plabels= ["F16","Q8_0","Q4_K_M"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# PTL lines
ax = axes[0]
for prec in precs:
    ptl4 = [qv(pl,"m4","ptl_ms",prec) for pl in pls]
    ptl2 = [qv(pl,"m2","ptl_ms",prec) for pl in pls]
    ax.plot(pls, ptl4, color=PREC_COLORS[prec], marker="o",
            linewidth=2.0, label=f"M4 {prec}")
    ax.plot(pls, ptl2, color=PREC_COLORS[prec], marker="s",
            linewidth=1.6, linestyle="--", alpha=0.65, label=f"M2 {prec}")
ax.set_xticks(pls)
ax.set_xlabel("Prompt Length (tokens)")
ax.set_ylabel("PTL (ms)")
ax.set_title("Per-Token Latency by Precision")
ax.legend(ncol=2, fontsize=8.5)

# Speedup bars — M4
ax = axes[1]
x  = np.arange(len(pls))
w  = 0.22
for i, prec in enumerate(precs):
    sp = [qv(pl,"m4","speedup_vs_f16",prec) for pl in pls]
    bars = ax.bar(x + (i-1)*w, sp, w*0.92,
                  color=PREC_COLORS[prec], alpha=0.85, label=prec)
    for bar, v in zip(bars, sp):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.02,
                f"{v:.2f}×", ha="center", va="bottom",
                fontsize=8, color="#444")
ax.axhline(1.0, color="grey", linewidth=1.2, linestyle="--", alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels([f"{p} tok" for p in pls])
ax.set_xlabel("Prompt Length (tokens)")
ax.set_ylabel("Speedup (× vs F16)")
ax.set_title("Quantization Speedup — Mac M4")
ax.legend()

# M4 vs M2 Q4 speedup
ax = axes[2]
sp4 = [qv(pl,"m4","speedup_vs_f16","Q4_K_M") for pl in pls]
sp2 = [qv(pl,"m2","speedup_vs_f16","Q4_K_M") for pl in pls]
x   = np.arange(len(pls))
ax.bar(x - 0.2, sp4, 0.38, color=M4, alpha=0.85, label="Mac M4")
ax.bar(x + 0.2, sp2, 0.38, color=M2, alpha=0.85, label="Mac M2")
ax.axhline(1.0, color="grey", linewidth=1.2, linestyle="--", alpha=0.7)
for i, (a, b) in enumerate(zip(sp4, sp2)):
    ax.text(i-0.2, a+0.02, f"{a:.2f}×", ha="center", va="bottom", fontsize=8)
    ax.text(i+0.2, b+0.02, f"{b:.2f}×", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([f"{p} tok" for p in pls])
ax.set_xlabel("Prompt Length (tokens)")
ax.set_ylabel("Speedup (× vs F16)")
ax.set_title("Q4_K_M Speedup: M4 vs M2")
ax.legend()

plt.tight_layout()
save(fig, "05_quantization_speedup.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Cold vs Warm Run
# ─────────────────────────────────────────────────────────────────────────────
print("06 · Cold vs Warm Run")
rows  = read("06_cold_vs_warm_run.csv")
cold  = next(r for r in rows if r["run_type"]=="cold")
warms = [r for r in rows if r["run_type"]=="warm"]
run_x = list(range(1, len(warms)+1))

c4t  = float(cold["m4_ttft_ms"])
c2t  = float(cold["m2_ttft_ms"])
ld4  = float(cold["m4_model_load_ms"])
ld2  = float(cold["m2_model_load_ms"])
w4t  = [float(r["m4_ttft_ms"]) for r in warms]
w2t  = [float(r["m2_ttft_ms"]) for r in warms]
w4e  = [float(r["m4_e2e_ms"])  for r in warms]
w2e  = [float(r["m2_e2e_ms"])  for r in warms]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# TTFT runs
ax = axes[0]
ax.axhline(c4t, color=M4, linewidth=1.5, linestyle="-.", alpha=0.6,
           label=f"M4 cold TTFT ({c4t:.0f} ms)")
ax.axhline(c2t, color=M2, linewidth=1.5, linestyle="-.", alpha=0.6,
           label=f"M2 cold TTFT ({c2t:.0f} ms)")
ax.plot(run_x, w4t, color=M4, marker="o", linewidth=2.0, label="M4 warm runs")
ax.plot(run_x, w2t, color=M2, marker="s", linewidth=2.0, label="M2 warm runs")
ax.axhline(np.mean(w4t), color=M4, linewidth=1.0, linestyle=":",
           label=f"M4 warm mean ({np.mean(w4t):.1f} ms)")
ax.axhline(np.mean(w2t), color=M2, linewidth=1.0, linestyle=":",
           label=f"M2 warm mean ({np.mean(w2t):.1f} ms)")
ax.set_xlabel("Warm Run Index")
ax.set_ylabel("TTFT (ms)")
ax.set_title("Cold vs Warm TTFT")
ax.set_xticks(run_x)
ax.legend(fontsize=8.5)

# E2E warm runs
ax = axes[1]
ax.plot(run_x, w4e, color=M4, marker="o", linewidth=2.0, label="Mac M4")
ax.plot(run_x, w2e, color=M2, marker="s", linewidth=2.0, label="Mac M2")
ax.axhline(np.mean(w4e), color=M4, linewidth=1.0, linestyle="--",
           alpha=0.7, label=f"M4 mean {np.mean(w4e):.0f} ms")
ax.axhline(np.mean(w2e), color=M2, linewidth=1.0, linestyle="--",
           alpha=0.7, label=f"M2 mean {np.mean(w2e):.0f} ms")
ax.set_xlabel("Warm Run Index")
ax.set_ylabel("E2E Latency (ms)")
ax.set_title("End-to-End Latency — Warm Runs")
ax.set_xticks(run_x)
ax.legend(fontsize=8.5)

# Summary bars
ax = axes[2]
labels = ["M4\nModel Load", "M4\nCold TTFT",
          f"M4 Warm\nMean TTFT\n({np.mean(w4t):.1f} ms)",
          "M2\nModel Load", "M2\nCold TTFT",
          f"M2 Warm\nMean TTFT\n({np.mean(w2t):.1f} ms)"]
values = [ld4, c4t, np.mean(w4t), ld2, c2t, np.mean(w2t)]
bar_colors = [M4, M4, M4, M2, M2, M2]
alphas     = [0.45, 0.75, 1.0, 0.45, 0.75, 1.0]
bars = [ax.bar(i, v, 0.6, color=c, alpha=a)
        for i,(v,c,a) in enumerate(zip(values, bar_colors, alphas))]
for i, v in enumerate(values):
    ax.text(i, v + max(values)*0.015, f"{v:.0f}",
            ha="center", va="bottom", fontsize=9)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Time (ms)")
ax.set_title("Cold Start Summary")
patches = [mpatches.Patch(color=M4, label="Mac M4"),
           mpatches.Patch(color=M2, label="Mac M2")]
ax.legend(handles=patches)

plt.tight_layout()
save(fig, "06_cold_vs_warm.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Latency Variance Distribution
# ─────────────────────────────────────────────────────────────────────────────
print("07 · Latency Variance Distribution")
rows = read("10_latency_variance_distribution.csv")
pls  = [int(r["prompt_length_tokens"]) for r in rows]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, dev, dcolor, dlabel in [
        (axes[0], "m4", M4, "Mac M4 16GB"),
        (axes[1], "m2", M2, "Mac M2 8GB")]:

    mn   = [float(r[f"{dev}_min_ms"])    for r in rows]
    med  = [float(r[f"{dev}_median_ms"]) for r in rows]
    mean = [float(r[f"{dev}_mean_ms"])   for r in rows]
    p90  = [float(r[f"{dev}_p90_ms"])    for r in rows]
    p99  = [float(r[f"{dev}_p99_ms"])    for r in rows]
    mx   = [float(r[f"{dev}_max_ms"])    for r in rows]

    x = np.arange(len(pls))

    for i in range(len(pls)):
        # Min–max whisker
        ax.vlines(x[i], mn[i], mx[i], color=dcolor, linewidth=1.2, alpha=0.4)
        ax.hlines([mn[i], mx[i]], x[i]-0.12, x[i]+0.12,
                  color=dcolor, linewidth=1.2, alpha=0.4)
        # IQR box (min to p90)
        ax.bar(x[i], p90[i]-mn[i], 0.42, bottom=mn[i],
               color=dcolor, alpha=0.15)
        # Median
        ax.hlines(med[i], x[i]-0.21, x[i]+0.21,
                  color=dcolor, linewidth=2.5)

    # Mean diamonds
    ax.scatter(x, mean, color=dcolor, marker="D", s=50, zorder=5,
               label="Mean", linewidths=0.8, edgecolors="white")
    # p99 triangles
    ax.scatter(x, p99, color="grey", marker="^", s=45, zorder=5,
               label="p99", linewidths=0.8, edgecolors="white")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p} tok" for p in pls])
    ax.set_xlabel("Prompt Length (tokens)")
    ax.set_ylabel("Per-Token Latency (ms)")
    ax.set_title(f"Latency Distribution — {dlabel}\n(n=50 trials per config)")

    handles = [
        mpatches.Patch(color=dcolor, alpha=0.15, label="Min–p90 range"),
        plt.Line2D([0],[0], color=dcolor, linewidth=2.5, label="Median"),
        plt.Line2D([0],[0], color=dcolor, marker="D", linestyle="",
                   markersize=6, label="Mean"),
        plt.Line2D([0],[0], color="grey", marker="^", linestyle="",
                   markersize=6, label="p99"),
    ]
    ax.legend(handles=handles, fontsize=9)

plt.tight_layout()
save(fig, "07_latency_variance_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Model Scaling
# ─────────────────────────────────────────────────────────────────────────────
print("08 · Model Scaling")
rows   = read("11_model_scaling.csv", alt=IN2)
mnames = ["TinyLlama-1.1B","LLaMA-3.2-1B","LLaMA-3.2-3B"]
pls4   = sorted(set(int(r["prompt_length_tokens"]) for r in rows))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# PTL vs prompt
ax = axes[0]
for mname in mnames:
    mr   = sorted([r for r in rows if r["model_name"]==mname],
                  key=lambda r: int(r["prompt_length_tokens"]))
    xp   = [int(r["prompt_length_tokens"]) for r in mr]
    ptl4 = [float(r["m4_ptl_ms"]) for r in mr]
    ptl2 = [float(r["m2_ptl_ms"]) for r in mr]
    src  = mr[0]["data_source"]
    ls   = "-" if src=="measured" else "--"
    c    = MODEL_COLORS[mname]
    ax.plot(xp, ptl4, color=c, marker="o", linestyle=ls,
            linewidth=2.0, label=f"{mname} M4")
    ax.plot(xp, ptl2, color=c, marker="s", linestyle=":",
            linewidth=1.5, alpha=0.6, label=f"{mname} M2")
ax.set_xticks(pls4)
ax.set_xlabel("Prompt Length (tokens)")
ax.set_ylabel("PTL (ms)")
ax.set_title("PTL vs Prompt Length by Model")
ax.legend(fontsize=8, ncol=1)
ax.text(0.98, 0.03, "─  measured    --  estimated",
        transform=ax.transAxes, fontsize=8, color="grey", ha="right")

# TTFT vs prompt
ax = axes[1]
for mname in mnames:
    mr   = sorted([r for r in rows if r["model_name"]==mname],
                  key=lambda r: int(r["prompt_length_tokens"]))
    xp    = [int(r["prompt_length_tokens"]) for r in mr]
    ttft4 = [float(r["m4_ttft_ms"]) for r in mr]
    ttft2 = [float(r["m2_ttft_ms"]) for r in mr]
    src   = mr[0]["data_source"]
    ls    = "-" if src=="measured" else "--"
    c     = MODEL_COLORS[mname]
    ax.plot(xp, ttft4, color=c, marker="o", linestyle=ls,
            linewidth=2.0, label=f"{mname} M4")
    ax.plot(xp, ttft2, color=c, marker="s", linestyle=":",
            linewidth=1.5, alpha=0.6, label=f"{mname} M2")
ax.set_xticks(pls4)
ax.set_xlabel("Prompt Length (tokens)")
ax.set_ylabel("TTFT (ms)")
ax.set_title("TTFT vs Prompt Length by Model")
ax.legend(fontsize=8, ncol=1)

# Throughput at 256 tok
ax = axes[2]
pl_t = 256
m4_tput = [float(r["m4_throughput_tok_s"])
            for r in rows if int(r["prompt_length_tokens"])==pl_t]
m2_tput = [float(r["m2_throughput_tok_s"])
            for r in rows if int(r["prompt_length_tokens"])==pl_t]
mnames_256 = [r["model_name"]
              for r in rows if int(r["prompt_length_tokens"])==pl_t]
x = np.arange(len(mnames_256))
ax.bar(x - 0.2, m4_tput, 0.38, color=M4, alpha=0.85, label="Mac M4")
ax.bar(x + 0.2, m2_tput, 0.38, color=M2, alpha=0.85, label="Mac M2")
for i,(a,b) in enumerate(zip(m4_tput, m2_tput)):
    ax.text(i-0.2, a+0.3, f"{a:.1f}", ha="center", va="bottom", fontsize=8.5)
    ax.text(i+0.2, b+0.3, f"{b:.1f}", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(x)
ax.set_xticklabels([n.replace("-"," ").replace("LLaMA ","LLaMA\n")
                    for n in mnames_256], fontsize=9)
ax.set_ylabel("Throughput (tok/s)")
ax.set_title(f"Throughput by Model Size @ {pl_t} tokens")
ax.legend()

plt.tight_layout()
save(fig, "08_model_scaling.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Cross-Platform Summary
# ─────────────────────────────────────────────────────────────────────────────
print("09 · Cross-Platform Summary")
rows = read("00_cross_platform_summary.csv")
pls  = [int(r["prompt_length_tokens"])    for r in rows]
m4t  = [float(r["m4_ttft_ms"])            for r in rows]
m2t  = [float(r["m2_ttft_ms"])            for r in rows]
m4p  = [float(r["m4_ptl_ms"])             for r in rows]
m2p  = [float(r["m2_ptl_ms"])             for r in rows]
m4k  = [float(r["m4_throughput_tok_s"])   for r in rows]
m2k  = [float(r["m2_throughput_tok_s"])   for r in rows]
rat  = [float(r["ptl_ratio_m2_over_m4"])  for r in rows]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# TTFT
ax = axes[0]
x  = np.arange(len(pls))
ax.bar(x - 0.2, m4t, 0.38, color=M4, alpha=0.85, label="Mac M4 16GB")
ax.bar(x + 0.2, m2t, 0.38, color=M2, alpha=0.85, label="Mac M2 8GB")
ax.set_xticks(x)
ax.set_xticklabels([str(p) for p in pls])
ax.set_xlabel("Prompt Length (tokens)")
ax.set_ylabel("TTFT (ms)")
ax.set_title("Time to First Token — M4 vs M2")
ax.legend()

# Throughput
ax = axes[1]
ax.plot(pls, m4k, color=M4, marker="o", linewidth=2.0, label="Mac M4 16GB")
ax.plot(pls, m2k, color=M2, marker="s", linewidth=2.0, label="Mac M2 8GB")
ax.fill_between(pls, m4k, m2k, color="grey", alpha=0.08)
ax.set_xticks(pls)
ax.set_xlabel("Prompt Length (tokens)")
ax.set_ylabel("Throughput (tok/s)")
ax.set_title("Decode Throughput — M4 vs M2")
ax.legend()

# PTL ratio
ax = axes[2]
bars = ax.bar(range(len(pls)), rat, 0.5,
              color=[M4 if r < 1.55 else M2 for r in rat], alpha=0.75)
for i, v in enumerate(rat):
    ax.text(i, v + 0.01, f"{v:.3f}×",
            ha="center", va="bottom", fontsize=9.5)
ax.axhline(np.mean(rat), color="grey", linewidth=1.2, linestyle="--",
           label=f"Mean ratio {np.mean(rat):.3f}×")
ax.set_xticks(range(len(pls)))
ax.set_xticklabels([f"{p} tok" for p in pls])
ax.set_ylim(1.2, 1.9)
ax.set_xlabel("Prompt Length (tokens)")
ax.set_ylabel("M2 PTL / M4 PTL")
ax.set_title("M2 / M4 Latency Ratio (PTL)")
ax.legend()

plt.tight_layout()
save(fig, "09_cross_platform_summary.png")


print(f"\nAll 9 graphs saved → {OUT}")
for f in sorted(os.listdir(OUT)):
    if f.endswith(".png"):
        kb = os.path.getsize(os.path.join(OUT,f))/1024
        print(f"  {f:<48}  {kb:6.1f} KB")
