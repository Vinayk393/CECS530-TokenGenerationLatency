"""
06_cold_vs_warm_run.py
------------------------
Compares cold (first run, model just loaded) vs warm (model cached in memory)
inference latency. Isolates the true cost of model loading vs inference.
Run this immediately after loading the model to capture the real cold-start cost.

Requirements:
    pip install transformers torch accelerate matplotlib psutil

Usage:
    python 06_cold_vs_warm_run.py
    python 06_cold_vs_warm_run.py --prompt_length 128 --warm_runs 10

Output:
    results/06_cold_vs_warm_run.json
    plots/06_cold_vs_warm_run.png
"""

import argparse
import json
import os
import time
import statistics
import gc

import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

RESULTS_DIR = "results"
PLOTS_DIR   = "plots"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

DEFAULT_MODEL        = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PROMPT_LEN   = 128
DEFAULT_WARM_RUNS    = 10
DEFAULT_OUTPUT_TOKS  = 30

BASE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Artificial intelligence is transforming the world. "
    "Large language models generate text token by token. "
)


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


def run_generation(model, tokenizer, prompt: str,
                   output_tokens: int, device: str) -> dict:
    """Full generation run — returns TTFT and E2E latency."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    sync(device)
    t_start = time.perf_counter()

    with torch.no_grad():
        out = model(input_ids=inputs["input_ids"], use_cache=True)
    sync(device)
    t_first = time.perf_counter()

    past_key_values = out.past_key_values
    next_token      = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    for _ in range(output_tokens - 1):
        with torch.no_grad():
            out = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = out.past_key_values
        next_token      = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    sync(device)
    t_end = time.perf_counter()

    return {
        "ttft_ms": round((t_first - t_start) * 1000, 2),
        "e2e_ms":  round((t_end   - t_start) * 1000, 2),
        "decode_ms": round((t_end - t_first) * 1000, 2),
        "tput_tok_s": round(output_tokens / (t_end - t_start), 2),
    }


def main(args):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps"  if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt    = build_prompt(args.prompt_length, tokenizer)
    actual_pl = len(tokenizer.encode(prompt))
    print(f"Prompt length: {actual_pl} tokens")

    # ---- COLD RUN — load model fresh ----
    print("\nLoading model for cold run...")
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    t_load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    t_load_end = time.perf_counter()
    load_time_ms = (t_load_end - t_load_start) * 1000
    print(f"Model load time: {load_time_ms:.0f} ms")

    print("Cold run (first inference after load)...")
    cold_result = run_generation(model, tokenizer, prompt, args.output_tokens, device)
    print(f"  Cold TTFT: {cold_result['ttft_ms']} ms  |  E2E: {cold_result['e2e_ms']} ms")

    # ---- WARM RUNS ----
    print(f"\nWarm runs (×{args.warm_runs})...")
    warm_results = []
    for i in range(args.warm_runs):
        r = run_generation(model, tokenizer, prompt, args.output_tokens, device)
        warm_results.append(r)
        print(f"  Run {i+1}: TTFT {r['ttft_ms']} ms  |  E2E {r['e2e_ms']} ms")

    warm_ttft_mean = statistics.mean([r["ttft_ms"] for r in warm_results])
    warm_ttft_std  = statistics.stdev([r["ttft_ms"] for r in warm_results]) if len(warm_results) > 1 else 0
    warm_e2e_mean  = statistics.mean([r["e2e_ms"]  for r in warm_results])
    warm_e2e_std   = statistics.stdev([r["e2e_ms"]  for r in warm_results]) if len(warm_results) > 1 else 0
    warm_tput_mean = statistics.mean([r["tput_tok_s"] for r in warm_results])

    print(f"\nSummary:")
    print(f"  Cold TTFT: {cold_result['ttft_ms']} ms  |  Warm mean TTFT: {warm_ttft_mean:.1f} ms")
    print(f"  Speedup (warm/cold): {cold_result['ttft_ms'] / warm_ttft_mean:.2f}×")

    # Save JSON
    out = {
        "model": args.model,
        "device": device,
        "prompt_length": actual_pl,
        "output_tokens": args.output_tokens,
        "model_load_time_ms": round(load_time_ms, 1),
        "cold_run": cold_result,
        "warm_runs": {
            "n": args.warm_runs,
            "mean_ttft_ms": round(warm_ttft_mean, 2),
            "stdev_ttft_ms": round(warm_ttft_std, 2),
            "mean_e2e_ms":  round(warm_e2e_mean, 2),
            "stdev_e2e_ms": round(warm_e2e_std,  2),
            "mean_tput_tok_s": round(warm_tput_mean, 2),
            "raw": warm_results,
        },
    }
    out_json = os.path.join(RESULTS_DIR, "06_cold_vs_warm_run.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"Cold vs Warm Run Comparison\n{args.model} · {device}",
                 fontsize=13, fontweight="bold")

    run_labels    = ["Cold (run 1)"] + [f"Warm {i+1}" for i in range(args.warm_runs)]
    all_ttft      = [cold_result["ttft_ms"]] + [r["ttft_ms"] for r in warm_results]
    all_e2e       = [cold_result["e2e_ms"]]  + [r["e2e_ms"]  for r in warm_results]
    all_tput      = [cold_result["tput_tok_s"]] + [r["tput_tok_s"] for r in warm_results]
    x             = list(range(len(run_labels)))
    cold_color    = "#EF4444"
    warm_color    = "#3B82F6"
    bar_colors    = [cold_color] + [warm_color] * args.warm_runs

    # TTFT across runs
    ax = axes[0]
    bars = ax.bar(x, all_ttft, color=bar_colors, alpha=0.85)
    ax.axhline(warm_ttft_mean, color=warm_color, linestyle="--", linewidth=1.5,
               label=f"Warm mean {warm_ttft_mean:.1f} ms")
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("TTFT per run")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    speedup = cold_result["ttft_ms"] / warm_ttft_mean
    ax.text(0, cold_result["ttft_ms"] * 1.02, f"{speedup:.1f}× slower\nthan warm",
            ha="center", fontsize=8, color=cold_color)

    # E2E across runs
    ax = axes[1]
    ax.bar(x, all_e2e, color=bar_colors, alpha=0.85)
    ax.axhline(warm_e2e_mean, color=warm_color, linestyle="--", linewidth=1.5,
               label=f"Warm mean {warm_e2e_mean:.1f} ms")
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("E2E latency (ms)")
    ax.set_title("End-to-end latency per run")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Summary comparison bar
    ax = axes[2]
    categories  = ["Model\nload", "Cold TTFT", "Warm TTFT\n(mean)", "Warm E2E\n(mean)"]
    values      = [load_time_ms, cold_result["ttft_ms"], warm_ttft_mean, warm_e2e_mean]
    colors_bar  = ["#F59E0B", cold_color, warm_color, "#10B981"]
    bars2       = ax.bar(categories, values, color=colors_bar, alpha=0.85)
    for bar, val in zip(bars2, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"{val:.0f} ms", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Cold vs warm summary")
    ax.grid(True, alpha=0.3, axis="y")

    import matplotlib.patches as mpatches
    legend_els = [
        mpatches.Patch(color=cold_color,    label="Cold"),
        mpatches.Patch(color=warm_color,    label="Warm"),
        mpatches.Patch(color="#F59E0B",     label="Load time"),
        mpatches.Patch(color="#10B981",     label="Warm E2E"),
    ]
    ax.legend(handles=legend_els, fontsize=8)

    plt.tight_layout()
    out_png = os.path.join(PLOTS_DIR, "06_cold_vs_warm_run.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_png}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cold vs warm run comparison")
    parser.add_argument("--model",         default=DEFAULT_MODEL)
    parser.add_argument("--prompt_length", type=int, default=DEFAULT_PROMPT_LEN)
    parser.add_argument("--output_tokens", type=int, default=DEFAULT_OUTPUT_TOKS)
    parser.add_argument("--warm_runs",     type=int, default=DEFAULT_WARM_RUNS)
    main(parser.parse_args())
