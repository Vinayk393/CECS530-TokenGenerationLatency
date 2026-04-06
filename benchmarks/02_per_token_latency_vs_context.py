"""
02_per_token_latency_vs_context.py
------------------------------------
Measures per-token decode latency as context grows.
Generates tokens one at a time and records the wall-clock cost of each step,
revealing the KV-cache growth effect and the inflection point where
memory-bandwidth pressure dominates.

Requirements:
    pip install transformers torch accelerate matplotlib

Usage:
    python 02_per_token_latency_vs_context.py
    python 02_per_token_latency_vs_context.py --context_lengths 32 64 128 256 512 1024
    python 02_per_token_latency_vs_context.py --model meta-llama/Llama-3.2-1B

Output:
    results/02_per_token_latency_vs_context.json
    plots/02_per_token_latency_vs_context.png
"""

import argparse
import json
import os
import time
import statistics

import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

RESULTS_DIR = "results"
PLOTS_DIR   = "plots"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

DEFAULT_MODEL           = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_CONTEXT_LENGTHS = [32, 64, 128, 256, 512, 768, 1024]
DEFAULT_DECODE_TOKENS   = 20   # tokens generated after the prompt to measure steady-state PTL
DEFAULT_TRIALS          = 3
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


def measure_ptl_at_context(model, tokenizer, prompt: str,
                            decode_tokens: int, device: str) -> list[float]:
    """
    Returns per-token latency (ms) for each of `decode_tokens` generated tokens.
    The prompt establishes the context length.
    First generated token is excluded (it overlaps with TTFT).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    past_key_values = None
    token_latencies = []

    # --- prefill pass (TTFT, discarded) ---
    sync(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    past_key_values = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    sync(device)

    # --- decode loop ---
    for _ in range(decode_tokens):
        sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
        sync(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        token_latencies.append(elapsed_ms)

        past_key_values = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    return token_latencies


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

    for cl in args.context_lengths:
        prompt     = build_prompt(cl, tokenizer)
        actual_len = len(tokenizer.encode(prompt))
        print(f"\nContext length: {actual_len} tokens")

        # Warm-up
        measure_ptl_at_context(model, tokenizer, prompt, 5, device)

        all_ptls = []
        for t in range(args.trials):
            ptls = measure_ptl_at_context(model, tokenizer, prompt,
                                          args.decode_tokens, device)
            all_ptls.extend(ptls)
            print(f"  Trial {t+1}: mean {statistics.mean(ptls):.1f} ms/tok")

        entry = {
            "context_length": actual_len,
            "mean_ptl_ms":    round(statistics.mean(all_ptls),   2),
            "median_ptl_ms":  round(statistics.median(all_ptls), 2),
            "stdev_ptl_ms":   round(statistics.stdev(all_ptls),  2) if len(all_ptls) > 1 else 0,
            "p90_ptl_ms":     round(sorted(all_ptls)[int(0.9 * len(all_ptls))], 2),
            "raw_ptl_ms":     [round(v, 3) for v in all_ptls],
        }
        results.append(entry)
        print(f"  Mean: {entry['mean_ptl_ms']} ms  |  P90: {entry['p90_ptl_ms']} ms")

    # Save JSON
    out_json = os.path.join(RESULTS_DIR, "02_per_token_latency_vs_context.json")
    with open(out_json, "w") as f:
        json.dump({"model": args.model, "device": device, "data": results}, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # Plot
    ctxs   = [r["context_length"] for r in results]
    means  = [r["mean_ptl_ms"]    for r in results]
    p90s   = [r["p90_ptl_ms"]     for r in results]
    stdevs = [r["stdev_ptl_ms"]   for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Per-Token Latency vs Context Length\n{args.model} · {device}",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(ctxs, means, "o-", color="#3B82F6", linewidth=2, label="Mean PTL")
    ax.plot(ctxs, p90s,  "s--", color="#EF4444", linewidth=1.5, label="P90 PTL")
    ax.fill_between(ctxs,
                    [m - s for m, s in zip(means, stdevs)],
                    [m + s for m, s in zip(means, stdevs)],
                    alpha=0.15, color="#3B82F6")
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Per-token latency (ms)")
    ax.set_title("Mean & P90 latency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Highlight inflection: largest jump between consecutive means
    max_jump_idx = max(range(1, len(means)), key=lambda i: means[i] - means[i-1])
    ax.axvline(ctxs[max_jump_idx], color="#F59E0B", linestyle=":", linewidth=2,
               label=f"Inflection ≈ {ctxs[max_jump_idx]} tok")
    ax.legend()

    # Latency growth rate (derivative)
    ax = axes[1]
    if len(ctxs) > 1:
        deltas = [means[i] - means[i-1] for i in range(1, len(means))]
        mid_ctxs = [(ctxs[i] + ctxs[i-1]) / 2 for i in range(1, len(ctxs))]
        ax.bar(mid_ctxs, deltas, width=[(ctxs[i]-ctxs[i-1])*0.6 for i in range(1,len(ctxs))],
               color="#F59E0B", alpha=0.8)
        ax.set_xlabel("Context length (tokens)")
        ax.set_ylabel("Latency increase vs previous (ms)")
        ax.set_title("Marginal latency cost per context step")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_png = os.path.join(PLOTS_DIR, "02_per_token_latency_vs_context.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_png}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark PTL vs context length")
    parser.add_argument("--model",           default=DEFAULT_MODEL)
    parser.add_argument("--context_lengths", nargs="+", type=int, default=DEFAULT_CONTEXT_LENGTHS)
    parser.add_argument("--decode_tokens",   type=int,             default=DEFAULT_DECODE_TOKENS)
    parser.add_argument("--trials",          type=int,             default=DEFAULT_TRIALS)
    main(parser.parse_args())
