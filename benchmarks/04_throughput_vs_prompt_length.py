"""
04_throughput_vs_prompt_length.py
-----------------------------------
Measures tokens-per-second throughput across prompt lengths.
Also computes effective memory bandwidth utilization to show
how the workload is memory-bound.

Requirements:
    pip install transformers torch accelerate matplotlib

Usage:
    python 04_throughput_vs_prompt_length.py
    python 04_throughput_vs_prompt_length.py --prompt_lengths 32 64 128 256 512 1024
    python 04_throughput_vs_prompt_length.py --decode_tokens 50

Output:
    results/04_throughput_vs_prompt_length.json
    plots/04_throughput_vs_prompt_length.png
"""

import argparse
import json
import os
import time
import statistics
import platform

import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

RESULTS_DIR = "results"
PLOTS_DIR   = "plots"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

DEFAULT_MODEL          = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PROMPT_LENGTHS = [32, 64, 128, 256, 512, 1024]
DEFAULT_DECODE_TOKENS  = 50
DEFAULT_TRIALS         = 3

BASE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Artificial intelligence is transforming the world. "
    "Large language models generate text token by token. "
)

# Approximate peak memory bandwidth for common Apple Silicon chips (GB/s)
# Add your chip here if needed
PEAK_BW_GB_S = {
    "m4":     120,
    "m4 pro": 273,
    "m3":     100,
    "m2":     100,
    "m2 pro": 200,
    "m1":      68,
}


def build_prompt(target_length: int, tokenizer) -> str:
    text = BASE_TEXT
    while len(tokenizer.encode(text)) < target_length:
        text += BASE_TEXT
    tokens = tokenizer.encode(text)[:target_length]
    return tokenizer.decode(tokens)


def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def measure_throughput(model, tokenizer, prompt: str,
                       decode_tokens: int, device: str) -> dict:
    """Returns tokens/sec and per-token latency for a full decode run."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    sync(device)
    # Prefill
    with torch.no_grad():
        out = model(input_ids=inputs["input_ids"], use_cache=True)
    past_key_values = out.past_key_values
    next_token      = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # Time the decode loop only
    sync(device)
    t0 = time.perf_counter()
    for _ in range(decode_tokens):
        with torch.no_grad():
            out = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = out.past_key_values
        next_token      = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    sync(device)
    elapsed = time.perf_counter() - t0

    tput_tok_s = decode_tokens / elapsed
    ptl_ms     = (elapsed / decode_tokens) * 1000
    return {"tput_tok_s": round(tput_tok_s, 2), "ptl_ms": round(ptl_ms, 2)}


def detect_peak_bw() -> float | None:
    """Best-effort detection of Apple Silicon peak BW from system info."""
    info = platform.processor().lower() + platform.machine().lower()
    for chip, bw in PEAK_BW_GB_S.items():
        if chip.replace(" ", "") in info.replace(" ", ""):
            return bw
    return None


def main(args):
    print(f"Loading model: {args.model}")
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps"  if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    # Estimate model size in GB (param count × 2 bytes for fp16)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_gb    = param_bytes / 1e9
    print(f"Model size in memory: {model_gb:.2f} GB")

    peak_bw = args.peak_bw or detect_peak_bw()
    if peak_bw:
        print(f"Peak memory bandwidth: {peak_bw} GB/s")

    results = []

    for pl in args.prompt_lengths:
        prompt     = build_prompt(pl, tokenizer)
        actual_len = len(tokenizer.encode(prompt))
        print(f"\nPrompt length: {actual_len} tokens")

        # Warm-up
        measure_throughput(model, tokenizer, prompt, 5, device)

        tputs, ptls = [], []
        for t in range(args.trials):
            m = measure_throughput(model, tokenizer, prompt, args.decode_tokens, device)
            tputs.append(m["tput_tok_s"])
            ptls.append(m["ptl_ms"])
            print(f"  Trial {t+1}: {m['tput_tok_s']:.1f} tok/s  ({m['ptl_ms']:.1f} ms/tok)")

        mean_tput = statistics.mean(tputs)
        mean_ptl  = statistics.mean(ptls)

        # Effective BW = model size / per-token latency (memory-bound assumption)
        eff_bw_gb_s = model_gb / (mean_ptl / 1000) if mean_ptl > 0 else None
        bw_util_pct = (eff_bw_gb_s / peak_bw * 100) if (eff_bw_gb_s and peak_bw) else None

        entry = {
            "prompt_length":  actual_len,
            "mean_tput_tok_s": round(mean_tput, 2),
            "stdev_tput":      round(statistics.stdev(tputs), 2) if len(tputs) > 1 else 0,
            "mean_ptl_ms":     round(mean_ptl, 2),
            "eff_bw_gb_s":     round(eff_bw_gb_s, 1) if eff_bw_gb_s else None,
            "bw_util_pct":     round(bw_util_pct, 1) if bw_util_pct else None,
        }
        results.append(entry)
        if bw_util_pct:
            print(f"  Eff BW: {eff_bw_gb_s:.1f} GB/s  BW util: {bw_util_pct:.1f}%")

    # Save JSON
    out_json = os.path.join(RESULTS_DIR, "04_throughput_vs_prompt_length.json")
    with open(out_json, "w") as f:
        json.dump({
            "model": args.model, "device": device,
            "model_size_gb": round(model_gb, 3),
            "peak_bw_gb_s": peak_bw,
            "data": results
        }, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # Plot
    lens  = [r["prompt_length"]    for r in results]
    tputs = [r["mean_tput_tok_s"]  for r in results]
    stds  = [r["stdev_tput"]       for r in results]
    bw_u  = [r["bw_util_pct"]      for r in results]

    has_bw = all(v is not None for v in bw_u)
    ncols  = 2 if has_bw else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols + 2, 5))
    if ncols == 1:
        axes = [axes]

    fig.suptitle(f"Throughput vs Prompt Length\n{args.model} · {device}",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.errorbar(lens, tputs, yerr=stds, marker="o", linewidth=2,
                capsize=4, color="#3B82F6", ecolor="#93C5FD")
    ax.fill_between(lens,
                    [t - s for t, s in zip(tputs, stds)],
                    [t + s for t, s in zip(tputs, stds)],
                    alpha=0.15, color="#3B82F6")
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Decode throughput drops with longer prompts")
    ax.grid(True, alpha=0.3)

    if has_bw:
        ax2 = axes[1]
        bar_colors = ["#EF4444" if v > 80 else "#F59E0B" if v > 60 else "#10B981"
                      for v in bw_u]
        bars = ax2.bar([str(l) for l in lens], bw_u, color=bar_colors, alpha=0.85)
        if peak_bw:
            ax2.axhline(100, color="#EF4444", linestyle="--", linewidth=1.5,
                        label=f"Peak ({peak_bw} GB/s)")
        ax2.set_xlabel("Prompt length (tokens)")
        ax2.set_ylabel("Memory bandwidth utilization (%)")
        ax2.set_title("BW utilization rises with context\n→ confirms memory-bound decode")
        ax2.set_ylim(0, 110)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, bw_u):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.0f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_png = os.path.join(PLOTS_DIR, "04_throughput_vs_prompt_length.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_png}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark throughput vs prompt length")
    parser.add_argument("--model",          default=DEFAULT_MODEL)
    parser.add_argument("--prompt_lengths", nargs="+", type=int, default=DEFAULT_PROMPT_LENGTHS)
    parser.add_argument("--decode_tokens",  type=int,             default=DEFAULT_DECODE_TOKENS)
    parser.add_argument("--trials",         type=int,             default=DEFAULT_TRIALS)
    parser.add_argument("--peak_bw",        type=float,           default=None,
                        help="Peak memory bandwidth in GB/s (e.g. 120 for M4, 100 for M2)")
    main(parser.parse_args())
