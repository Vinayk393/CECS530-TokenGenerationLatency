"""
09_latency_decomposition.py
-----------------------------
Instruments each transformer component using PyTorch forward hooks to measure
exactly where time is spent during decode: attention, MLP, LayerNorm,
LM head, embedding, and framework overhead.

This is the most technically impressive benchmark — it shows the exact
breakdown that prior teams like Team 27 only approximated.

Requirements:
    pip install transformers torch accelerate matplotlib

Usage:
    python 09_latency_decomposition.py
    python 09_latency_decomposition.py --prompt_length 256 --decode_tokens 30
    python 09_latency_decomposition.py --model meta-llama/Llama-3.2-1B

Output:
    results/09_latency_decomposition.json
    plots/09_latency_decomposition.png
"""

import argparse
import json
import os
import time
import statistics
from collections import defaultdict

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

RESULTS_DIR = "results"
PLOTS_DIR   = "plots"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

DEFAULT_MODEL         = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PROMPT_LEN    = 128
DEFAULT_DECODE_TOKENS = 30

COMPONENT_COLORS = {
    "embed_tokens":   "#94A3B8",
    "self_attn":      "#3B82F6",
    "mlp":            "#10B981",
    "input_layernorm":"#F59E0B",
    "post_attn_norm": "#F97316",
    "lm_head":        "#EF4444",
    "overhead":       "#8B5CF6",
}

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


class ComponentTimer:
    """Attaches forward hooks to named modules and accumulates timing."""

    def __init__(self, device: str):
        self.device  = device
        self.times   = defaultdict(list)  # component_name → [ms, ...]
        self._hooks  = []
        self._starts = {}

    def _pre_hook(self, name):
        def hook(module, input):
            sync(self.device)
            self._starts[name] = time.perf_counter()
        return hook

    def _post_hook(self, name):
        def hook(module, input, output):
            sync(self.device)
            elapsed_ms = (time.perf_counter() - self._starts.get(name, time.perf_counter())) * 1000
            self.times[name].append(elapsed_ms)
        return hook

    def attach(self, model):
        """Attach hooks to relevant submodules."""
        for name, module in model.named_modules():
            # Match key component types
            short = name.split(".")[-1]
            if short in ("self_attn", "mlp", "input_layernorm",
                         "post_attention_layernorm", "embed_tokens", "lm_head"):
                canonical = "post_attn_norm" if short == "post_attention_layernorm" else short
                h1 = module.register_forward_pre_hook(self._pre_hook(canonical))
                h2 = module.register_forward_hook(self._post_hook(canonical))
                self._hooks.extend([h1, h2])

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def summary(self) -> dict:
        """Returns mean time per call for each component (ms)."""
        return {k: round(statistics.mean(v), 3) for k, v in self.times.items() if v}


def measure_decomposition(model, tokenizer, prompt: str,
                           decode_tokens: int, device: str) -> dict:
    inputs   = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    timer = ComponentTimer(device)
    timer.attach(model)

    # ── Prefill pass ──────────────────────────────────────────────────────
    sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    sync(device)
    prefill_total_ms = (time.perf_counter() - t0) * 1000
    prefill_component_totals = timer.summary()

    # Clear accumulated times for decode measurement
    timer.times.clear()

    past_key_values = out.past_key_values
    next_token      = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # ── Decode loop ────────────────────────────────────────────────────────
    token_totals_ms = []
    for _ in range(decode_tokens):
        sync(device)
        t_tok = time.perf_counter()
        with torch.no_grad():
            out = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
        sync(device)
        token_totals_ms.append((time.perf_counter() - t_tok) * 1000)
        past_key_values = out.past_key_values
        next_token      = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    timer.remove()

    decode_components = timer.summary()

    # Overhead = measured total - sum of component times
    mean_token_total = statistics.mean(token_totals_ms)
    component_sum    = sum(decode_components.values())
    overhead_ms      = max(0, mean_token_total - component_sum)
    decode_components["overhead"] = round(overhead_ms, 3)

    return {
        "prefill_total_ms":        round(prefill_total_ms, 2),
        "prefill_components_ms":   prefill_component_totals,
        "decode_mean_total_ms":    round(mean_token_total, 2),
        "decode_components_ms":    decode_components,
        "decode_component_pct": {
            k: round(v / mean_token_total * 100, 1)
            for k, v in decode_components.items()
        },
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

    prompt     = build_prompt(args.prompt_length, tokenizer)
    actual_len = len(tokenizer.encode(prompt))
    print(f"Prompt length: {actual_len} tokens | Decode tokens: {args.decode_tokens}")

    # Warm-up
    print("Warm-up run...")
    _ = measure_decomposition(model, tokenizer, prompt, 5, device)

    # Actual measurement
    print("Measuring decomposition...")
    result = measure_decomposition(model, tokenizer, prompt,
                                   args.decode_tokens, device)

    print(f"\nPrefill total: {result['prefill_total_ms']} ms")
    print(f"Decode mean total: {result['decode_mean_total_ms']} ms/token")
    print("\nDecode component breakdown:")
    for k, v in result["decode_components_ms"].items():
        pct = result["decode_component_pct"].get(k, 0)
        print(f"  {k:25s}: {v:6.2f} ms  ({pct:.1f}%)")

    # Multiple prompt lengths for scaling view
    all_results = [{"prompt_length": actual_len, **result}]

    if args.also_run_scaling:
        for pl in [64, 256, 512]:
            if pl == actual_len:
                continue
            p2 = build_prompt(pl, tokenizer)
            r2 = measure_decomposition(model, tokenizer, p2, args.decode_tokens, device)
            all_results.append({"prompt_length": len(tokenizer.encode(p2)), **r2})
            print(f"\nPrompt {pl}: decode total {r2['decode_mean_total_ms']} ms")

    out_json = os.path.join(RESULTS_DIR, "09_latency_decomposition.json")
    with open(out_json, "w") as f:
        json.dump({"model": args.model, "device": device, "data": all_results}, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────
    base     = all_results[0]
    dec_comp = base["decode_components_ms"]
    dec_pct  = base["decode_component_pct"]

    known_components = ["embed_tokens", "self_attn", "mlp",
                        "input_layernorm", "post_attn_norm", "lm_head", "overhead"]
    present = [c for c in known_components if c in dec_comp]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Latency Decomposition — Decode Phase\n{args.model} · {device} · prompt={actual_len} tok",
        fontsize=13, fontweight="bold"
    )

    # Donut chart
    ax = axes[0]
    sizes  = [dec_comp[c] for c in present]
    colors = [COMPONENT_COLORS.get(c, "#ccc") for c in present]
    labels = [f"{c}\n{dec_pct.get(c,0):.1f}%" for c in present]
    wedges, _ = ax.pie(sizes, colors=colors, startangle=90,
                       wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.5))
    ax.set_title("Decode time share (donut)")
    ax.legend(wedges, labels, loc="lower center", fontsize=7,
              bbox_to_anchor=(0.5, -0.18), ncol=2)

    # Horizontal bar chart
    ax = axes[1]
    y    = range(len(present))
    vals = [dec_comp[c] for c in present]
    pcts = [dec_pct.get(c, 0) for c in present]
    cols = [COMPONENT_COLORS.get(c, "#ccc") for c in present]
    bars = ax.barh(list(y), vals, color=cols, alpha=0.85)
    ax.set_yticks(list(y))
    ax.set_yticklabels(present, fontsize=9)
    ax.set_xlabel("Time per decode token (ms)")
    ax.set_title("Component latency (ms)")
    ax.grid(True, alpha=0.3, axis="x")
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=8)

    # Prefill vs decode total comparison
    ax = axes[2]
    if len(all_results) > 1:
        prompt_lens = [r["prompt_length"]         for r in all_results]
        attn_times  = [r["decode_components_ms"].get("self_attn", 0) for r in all_results]
        mlp_times   = [r["decode_components_ms"].get("mlp", 0)       for r in all_results]
        ovhd_times  = [r["decode_components_ms"].get("overhead", 0)  for r in all_results]
        lmhd_times  = [r["decode_components_ms"].get("lm_head", 0)   for r in all_results]
        x = np.arange(len(prompt_lens))
        w = 0.6
        ax.bar(x, attn_times, w, label="Attention",  color="#3B82F6", alpha=0.85)
        ax.bar(x, mlp_times,  w, label="MLP",        color="#10B981", alpha=0.85, bottom=attn_times)
        ax.bar(x, ovhd_times, w, label="Overhead",   color="#8B5CF6", alpha=0.85,
               bottom=[a+m for a, m in zip(attn_times, mlp_times)])
        ax.bar(x, lmhd_times, w, label="LM head",    color="#EF4444", alpha=0.85,
               bottom=[a+m+o for a, m, o in zip(attn_times, mlp_times, ovhd_times)])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{pl} tok" for pl in prompt_lens])
        ax.set_xlabel("Prompt length")
        ax.set_ylabel("Decode latency (ms/token)")
        ax.set_title("Component scaling with context")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    else:
        # Single prompt: prefill vs decode breakdown side by side
        categories = ["Prefill\n(TTFT)", "Decode\n(per token)"]
        main_vals  = [base["prefill_total_ms"], base["decode_mean_total_ms"]]
        colors_bar = ["#F59E0B", "#3B82F6"]
        ax.bar(categories, main_vals, color=colors_bar, alpha=0.85)
        for i, (cat, val) in enumerate(zip(categories, main_vals)):
            ax.text(i, val + 1, f"{val:.1f} ms", ha="center", fontsize=10)
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Prefill vs per-token decode cost")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_png = os.path.join(PLOTS_DIR, "09_latency_decomposition.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_png}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latency decomposition via forward hooks")
    parser.add_argument("--model",               default=DEFAULT_MODEL)
    parser.add_argument("--prompt_length",        type=int, default=DEFAULT_PROMPT_LEN)
    parser.add_argument("--decode_tokens",        type=int, default=DEFAULT_DECODE_TOKENS)
    parser.add_argument("--also_run_scaling",     action="store_true",
                        help="Also run at 64/256/512 tokens for scaling view")
    main(parser.parse_args())
