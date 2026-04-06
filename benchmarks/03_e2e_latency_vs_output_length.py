"""
03_e2e_latency_vs_output_length.py
-------------------------------------
Measures total end-to-end generation time across different output lengths
for multiple prompt lengths. Confirms the linear growth of E2E latency
with number of tokens generated and shows the fixed prefill cost overhead.

Requirements:
    pip install transformers torch accelerate matplotlib

Usage:
    python 03_e2e_latency_vs_output_length.py
    python 03_e2e_latency_vs_output_length.py --output_lengths 16 32 64 128 256
    python 03_e2e_latency_vs_output_length.py --prompt_lengths 64 256 512

Output:
    results/03_e2e_latency_vs_output_length.json
    plots/03_e2e_latency_vs_output_length.png
"""

import argparse
import json
import os
import time
import statistics

import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

RESULTS_DIR = "results"
PLOTS_DIR   = "plots"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

DEFAULT_MODEL          = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_OUTPUT_LENGTHS = [16, 32, 64, 128, 256]
DEFAULT_PROMPT_LENGTHS = [64, 256, 512]
DEFAULT_TRIALS         = 3
BASE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Artificial intelligence is transforming the world. "
    "Large language models generate text token by token. "
)

COLORS = ["#3B82F6", "#10B981", "#8B5CF6", "#F59E0B"]


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


def measure_e2e(model, tokenizer, prompt: str,
                output_length: int, device: str) -> dict:
    """Returns total E2E latency and TTFT for one run."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    sync(device)
    t_start = time.perf_counter()

    with torch.no_grad():
        # TTFT
        out = model(input_ids=inputs["input_ids"], use_cache=True)
    t_first_token = time.perf_counter()

    past_key_values = out.past_key_values
    next_token      = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # Decode remaining tokens
    for _ in range(output_length - 1):
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
        "ttft_ms": round((t_first_token - t_start) * 1000, 2),
        "e2e_ms":  round((t_end        - t_start) * 1000, 2),
        "decode_ms": round((t_end - t_first_token) * 1000, 2),
    }


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

    all_results = {}

    for pl in args.prompt_lengths:
        prompt     = build_prompt(pl, tokenizer)
        actual_len = len(tokenizer.encode(prompt))
        print(f"\n--- Prompt length: {actual_len} tokens ---")
        pl_results = []

        # Warm-up
        measure_e2e(model, tokenizer, prompt, 8, device)

        for ol in args.output_lengths:
            print(f"  Output length: {ol} tokens")
            runs = []
            for t in range(args.trials):
                m = measure_e2e(model, tokenizer, prompt, ol, device)
                runs.append(m)
                print(f"    Trial {t+1}: E2E {m['e2e_ms']} ms  TTFT {m['ttft_ms']} ms")

            e2e_vals  = [r["e2e_ms"]   for r in runs]
            ttft_vals = [r["ttft_ms"]  for r in runs]
            dec_vals  = [r["decode_ms"] for r in runs]

            pl_results.append({
                "output_length":    ol,
                "mean_e2e_ms":      round(statistics.mean(e2e_vals),   2),
                "mean_ttft_ms":     round(statistics.mean(ttft_vals),  2),
                "mean_decode_ms":   round(statistics.mean(dec_vals),   2),
                "stdev_e2e_ms":     round(statistics.stdev(e2e_vals),  2) if len(e2e_vals) > 1 else 0,
                "mean_tpot_ms":     round(statistics.mean(dec_vals) / max(ol - 1, 1), 2),
            })

        all_results[str(actual_len)] = pl_results

    # Save JSON
    out_json = os.path.join(RESULTS_DIR, "03_e2e_latency_vs_output_length.json")
    with open(out_json, "w") as f:
        json.dump({"model": args.model, "device": device, "data": all_results}, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # Plot: E2E latency lines + stacked TTFT vs decode breakdown
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"End-to-End Latency vs Output Length\n{args.model} · {device}",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    for idx, (pl_key, pl_data) in enumerate(all_results.items()):
        ols   = [r["output_length"]  for r in pl_data]
        e2es  = [r["mean_e2e_ms"]    for r in pl_data]
        stds  = [r["stdev_e2e_ms"]   for r in pl_data]
        color = COLORS[idx % len(COLORS)]
        ax.plot(ols, e2es, "o-", color=color, linewidth=2, label=f"Prompt {pl_key} tok")
        ax.fill_between(ols,
                        [e - s for e, s in zip(e2es, stds)],
                        [e + s for e, s in zip(e2es, stds)],
                        alpha=0.1, color=color)

        # Fit linear trend
        coefs = np.polyfit(ols, e2es, 1)
        trend = np.poly1d(coefs)
        ax.plot(ols, trend(ols), "--", color=color, linewidth=1, alpha=0.5)

    ax.set_xlabel("Output tokens generated")
    ax.set_ylabel("End-to-end latency (ms)")
    ax.set_title("E2E latency — linear growth confirmed")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Stacked bar: TTFT vs decode time for one prompt length
    ax = axes[1]
    base_key  = list(all_results.keys())[0]
    base_data = all_results[base_key]
    ols        = [r["output_length"]  for r in base_data]
    ttfts      = [r["mean_ttft_ms"]   for r in base_data]
    dec_times  = [r["mean_decode_ms"] for r in base_data]

    x = range(len(ols))
    bars1 = ax.bar(x, ttfts,     label="TTFT (prefill)",  color="#3B82F6", alpha=0.85)
    bars2 = ax.bar(x, dec_times, label="Decode time",     color="#10B981", alpha=0.85,
                   bottom=ttfts)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{ol} tok" for ol in ols])
    ax.set_xlabel("Output tokens generated")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"TTFT vs decode breakdown\n(prompt = {base_key} tokens)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_png = os.path.join(PLOTS_DIR, "03_e2e_latency_vs_output_length.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_png}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark E2E latency vs output length")
    parser.add_argument("--model",          default=DEFAULT_MODEL)
    parser.add_argument("--output_lengths", nargs="+", type=int, default=DEFAULT_OUTPUT_LENGTHS)
    parser.add_argument("--prompt_lengths", nargs="+", type=int, default=DEFAULT_PROMPT_LENGTHS)
    parser.add_argument("--trials",         type=int,             default=DEFAULT_TRIALS)
    main(parser.parse_args())
