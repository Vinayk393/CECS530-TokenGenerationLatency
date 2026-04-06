"""
05_inter_token_latency_timeline.py
-------------------------------------
Records the latency of every single token during one full generation run.
This is the most visually compelling chart — it shows the TTFT spike,
steady-state decode, and the gradual latency climb as KV-cache grows.

Requirements:
    pip install transformers torch accelerate matplotlib

Usage:
    python 05_inter_token_latency_timeline.py
    python 05_inter_token_latency_timeline.py --prompt_length 256 --max_new_tokens 150
    python 05_inter_token_latency_timeline.py --prompt_lengths 64 256 512

Output:
    results/05_inter_token_latency_timeline.json
    plots/05_inter_token_latency_timeline.png
"""

import argparse
import json
import os
import time
import statistics

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

RESULTS_DIR = "results"
PLOTS_DIR   = "plots"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

DEFAULT_MODEL          = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PROMPT_LENGTHS = [128, 512]
DEFAULT_MAX_NEW_TOKENS = 100

BASE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Artificial intelligence is transforming the world. "
    "Large language models generate text token by token. "
)

COLORS = ["#3B82F6", "#8B5CF6", "#10B981"]


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


def record_token_timeline(model, tokenizer, prompt: str,
                          max_new_tokens: int, device: str) -> dict:
    """
    Returns a dict with:
      - ttft_ms: time to first token
      - token_latencies_ms: list of per-token latencies for tokens 2..N
      - cumulative_ms: cumulative wall time per token
    """
    inputs      = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids   = inputs["input_ids"]
    prompt_len  = input_ids.shape[1]

    timeline_ms = []

    sync(device)

    # TTFT — prefill pass
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    sync(device)
    ttft_ms = (time.perf_counter() - t0) * 1000
    timeline_ms.append(ttft_ms)

    past_key_values = out.past_key_values
    next_token      = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # Decode loop
    for i in range(max_new_tokens - 1):
        sync(device)
        t_tok = time.perf_counter()
        with torch.no_grad():
            out = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
        sync(device)
        tok_ms = (time.perf_counter() - t_tok) * 1000
        timeline_ms.append(tok_ms)

        past_key_values = out.past_key_values
        next_token      = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # Stop on EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

    cumulative = list(np.cumsum(timeline_ms))

    return {
        "prompt_length":       prompt_len,
        "tokens_generated":    len(timeline_ms),
        "ttft_ms":             round(ttft_ms, 2),
        "timeline_ms":         [round(v, 3) for v in timeline_ms],
        "cumulative_ms":       [round(v, 2) for v in cumulative],
        "steady_state_mean_ms": round(statistics.mean(timeline_ms[2:]), 2) if len(timeline_ms) > 2 else None,
        "steady_state_std_ms":  round(statistics.stdev(timeline_ms[2:]), 2) if len(timeline_ms) > 3 else None,
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

    all_results = []

    for pl in args.prompt_lengths:
        prompt = build_prompt(pl, tokenizer)
        actual = len(tokenizer.encode(prompt))
        print(f"\nRunning timeline for prompt length: {actual} tokens ...")

        # Warm-up
        record_token_timeline(model, tokenizer, prompt, 10, device)

        result = record_token_timeline(model, tokenizer, prompt,
                                       args.max_new_tokens, device)
        print(f"  TTFT: {result['ttft_ms']} ms")
        print(f"  Steady-state mean: {result['steady_state_mean_ms']} ms/tok")
        print(f"  Tokens generated: {result['tokens_generated']}")
        all_results.append(result)

    # Save JSON
    out_json = os.path.join(RESULTS_DIR, "05_inter_token_latency_timeline.json")
    with open(out_json, "w") as f:
        json.dump({"model": args.model, "device": device, "data": all_results}, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # Plot
    n = len(all_results)
    fig, axes = plt.subplots(n, 2, figsize=(13, 4 * n))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Inter-Token Latency Timeline\n{args.model} · {device}",
                 fontsize=13, fontweight="bold")

    for row_idx, (result, color) in enumerate(zip(all_results, COLORS)):
        timeline = result["timeline_ms"]
        tokens   = list(range(1, len(timeline) + 1))
        pl       = result["prompt_length"]
        ss_mean  = result["steady_state_mean_ms"]
        ss_std   = result["steady_state_std_ms"] or 0

        # Left: per-token latency
        ax = axes[row_idx][0]
        ax.bar(tokens, timeline, color=color, alpha=0.6, width=0.8)
        # Moving average
        window = max(3, len(timeline) // 10)
        ma = np.convolve(timeline, np.ones(window) / window, mode="valid")
        ma_x = range(window, len(timeline) + 1)
        ax.plot(list(ma_x), list(ma), color=color, linewidth=2, label=f"{window}-tok moving avg")

        # Annotate TTFT spike
        ax.annotate(f"TTFT\n{timeline[0]:.0f} ms",
                    xy=(1, timeline[0]),
                    xytext=(min(5, len(timeline) // 4), timeline[0] * 0.85),
                    fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1),
                    color="gray")

        # Steady-state band
        if ss_mean:
            ax.axhline(ss_mean, color="#EF4444", linestyle="--", linewidth=1.5,
                       label=f"Steady-state mean {ss_mean:.1f} ms")
            ax.axhspan(ss_mean - ss_std, ss_mean + ss_std,
                       alpha=0.08, color="#EF4444")

        # Annotate TTFT region and decode region
        ax.axvspan(0.5, 1.5, alpha=0.1, color="#F59E0B", label="TTFT (prefill)")
        ax.axvspan(1.5, len(tokens) + 0.5, alpha=0.04, color="#10B981", label="Decode phase")

        ax.set_xlabel("Token index")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"Prompt {pl} tok — per-token latency (token-by-token)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

        # Right: cumulative time
        ax = axes[row_idx][1]
        cumulative = result["cumulative_ms"]
        ax.plot(tokens, cumulative, color=color, linewidth=2)
        ax.fill_between(tokens, cumulative, alpha=0.1, color=color)
        ax.set_xlabel("Token index")
        ax.set_ylabel("Cumulative time (ms)")
        ax.set_title(f"Prompt {pl} tok — cumulative wall time")
        ax.grid(True, alpha=0.3)

        total_s = cumulative[-1] / 1000
        ax.annotate(f"Total: {total_s:.2f} s\n{len(timeline)} tokens",
                    xy=(len(tokens), cumulative[-1]),
                    xytext=(len(tokens) * 0.6, cumulative[-1] * 0.5),
                    fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1),
                    color="gray")

    plt.tight_layout()
    out_png = os.path.join(PLOTS_DIR, "05_inter_token_latency_timeline.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_png}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record per-token latency timeline")
    parser.add_argument("--model",          default=DEFAULT_MODEL)
    parser.add_argument("--prompt_lengths", nargs="+", type=int, default=DEFAULT_PROMPT_LENGTHS)
    parser.add_argument("--max_new_tokens", type=int,             default=DEFAULT_MAX_NEW_TOKENS)
    main(parser.parse_args())
