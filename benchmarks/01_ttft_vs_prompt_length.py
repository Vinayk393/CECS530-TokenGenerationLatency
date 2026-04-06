"""
01_ttft_vs_prompt_length.py
----------------------------
Measures Time To First Token (TTFT) across different prompt lengths.

Requirements:
    pip install transformers torch accelerate

Usage:
    python 01_ttft_vs_prompt_length.py
    python 01_ttft_vs_prompt_length.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python 01_ttft_vs_prompt_length.py --model meta-llama/Llama-3.2-1B --prompt_lengths 32 64 128 256 512

Output:
    results/01_ttft_vs_prompt_length.json
    plots/01_ttft_vs_prompt_length.png
"""

import argparse
import json
import os
import time
import statistics

import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from transformers import AutoTokenizer, AutoModelForCausalLM

RESULTS_DIR = "results"
PLOTS_DIR   = "plots"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

DEFAULT_MODEL          = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PROMPT_LENGTHS = [32, 64, 128, 256, 512]
DEFAULT_TRIALS         = 5
BASE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Artificial intelligence is transforming the world. "
    "Large language models generate text token by token. "
)


def build_prompt(target_length: int, tokenizer) -> str:
    """Repeat base text until tokenized length >= target_length."""
    text = BASE_TEXT
    while len(tokenizer.encode(text)) < target_length:
        text += BASE_TEXT
    tokens = tokenizer.encode(text)[:target_length]
    return tokenizer.decode(tokens)


def measure_ttft(model, tokenizer, prompt: str, device: str) -> float:
    """Return TTFT in milliseconds for a single prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Synchronise before timing
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

    return (time.perf_counter() - t0) * 1000  # ms


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

    results = []

    for pl in args.prompt_lengths:
        prompt = build_prompt(pl, tokenizer)
        actual_len = len(tokenizer.encode(prompt))
        print(f"\nPrompt length: {actual_len} tokens")

        # Warm-up run (not recorded)
        measure_ttft(model, tokenizer, prompt, device)

        latencies = []
        for t in range(args.trials):
            ms = measure_ttft(model, tokenizer, prompt, device)
            latencies.append(ms)
            print(f"  Trial {t+1}: {ms:.1f} ms")

        entry = {
            "prompt_length": actual_len,
            "mean_ms":   round(statistics.mean(latencies),   2),
            "median_ms": round(statistics.median(latencies), 2),
            "stdev_ms":  round(statistics.stdev(latencies),  2) if len(latencies) > 1 else 0,
            "min_ms":    round(min(latencies), 2),
            "max_ms":    round(max(latencies), 2),
            "raw_ms":    [round(v, 2) for v in latencies],
        }
        results.append(entry)
        print(f"  Mean: {entry['mean_ms']} ms  |  Stdev: {entry['stdev_ms']} ms")

    # Save JSON
    out_json = os.path.join(RESULTS_DIR, "01_ttft_vs_prompt_length.json")
    with open(out_json, "w") as f:
        json.dump({"model": args.model, "device": device, "data": results}, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # Plot
    lens   = [r["prompt_length"] for r in results]
    means  = [r["mean_ms"]       for r in results]
    stdevs = [r["stdev_ms"]      for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"TTFT vs Prompt Length\n{args.model} · {device}", fontsize=13, fontweight="bold")

    # Linear scale
    ax = axes[0]
    ax.errorbar(lens, means, yerr=stdevs, marker="o", linewidth=2,
                capsize=4, color="#3B82F6", ecolor="#93C5FD", label="Mean ± stdev")
    ax.fill_between(lens,
                    [m - s for m, s in zip(means, stdevs)],
                    [m + s for m, s in zip(means, stdevs)],
                    alpha=0.15, color="#3B82F6")
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Linear scale")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Log scale
    ax = axes[1]
    ax.errorbar(lens, means, yerr=stdevs, marker="o", linewidth=2,
                capsize=4, color="#8B5CF6", ecolor="#C4B5FD")
    ax.set_yscale("log")
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("TTFT (ms, log scale)")
    ax.set_title("Log scale — reveals growth regime")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    out_png = os.path.join(PLOTS_DIR, "01_ttft_vs_prompt_length.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_png}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark TTFT vs prompt length")
    parser.add_argument("--model",          default=DEFAULT_MODEL)
    parser.add_argument("--prompt_lengths", nargs="+", type=int, default=DEFAULT_PROMPT_LENGTHS)
    parser.add_argument("--trials",         type=int,             default=DEFAULT_TRIALS)
    main(parser.parse_args())
