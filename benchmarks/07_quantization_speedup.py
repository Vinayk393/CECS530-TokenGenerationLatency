"""
07_quantization_speedup.py
-----------------------------
Benchmarks Q4_K_M vs Q8_0 vs F16 using llama.cpp (llama-bench).
Falls back to HuggingFace if llama.cpp is not installed.

Requirements (Option A — recommended):
    Install llama.cpp: https://github.com/ggerganov/llama.cpp
    Download GGUF model files (Q4, Q8, F16) from HuggingFace

Requirements (Option B — fallback):
    pip install transformers torch accelerate matplotlib
    Only tests F16 and auto-quantized (bitsandbytes) if available

Usage (llama.cpp path):
    python 07_quantization_speedup.py \\
        --backend llamacpp \\
        --llama_bench /path/to/llama.cpp/build/bin/llama-bench \\
        --models q4=/path/to/model-Q4_K_M.gguf q8=/path/to/model-Q8_0.gguf f16=/path/to/model-F16.gguf

Usage (HuggingFace fallback):
    python 07_quantization_speedup.py --backend hf --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

Output:
    results/07_quantization_speedup.json
    plots/07_quantization_speedup.png
"""

import argparse
import json
import os
import re
import subprocess
import statistics
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS_DIR = "results"
PLOTS_DIR   = "plots"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

PROMPT_LENGTHS = [32, 128, 256, 512]
PRECISION_COLORS = {
    "Q4_K_M": "#3B82F6",
    "Q8_0":   "#10B981",
    "F16":    "#EF4444",
}


# ─────────────────────────────────────────────
# llama.cpp backend
# ─────────────────────────────────────────────

def run_llama_bench(llama_bench_path: str, model_path: str,
                    prompt_len: int, gen_len: int = 50) -> dict | None:
    """
    Runs llama-bench and parses output.
    Returns dict with tg (tokens/s decode), pp (tokens/s prefill), and latencies.
    """
    cmd = [
        llama_bench_path,
        "-m", model_path,
        "-p", str(prompt_len),
        "-n", str(gen_len),
        "-r", "3",          # 3 repetitions
        "--output", "json",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  llama-bench error: {result.stderr[:200]}")
            return None
        data = json.loads(result.stdout)
        # llama-bench JSON: list of dicts with keys t_pp_avg, t_tg_avg, etc.
        if isinstance(data, list) and len(data) > 0:
            entry = data[0]
            return {
                "pp_tok_s":   entry.get("pp_ts",    entry.get("t_pp_avg", None)),
                "tg_tok_s":   entry.get("tg_ts",    entry.get("t_tg_avg", None)),
                "ttft_ms":    1000 / entry.get("pp_ts", 1) * prompt_len if entry.get("pp_ts") else None,
                "ptl_ms":     1000 / entry.get("tg_ts", 1) if entry.get("tg_ts") else None,
            }
    except Exception as e:
        print(f"  Error running llama-bench: {e}")
        return None


def run_llamacpp_backend(args):
    """Run benchmarks using llama.cpp llama-bench."""
    model_map = {}
    for item in args.models:
        label, path = item.split("=", 1)
        model_map[label.upper()] = path

    results = {}
    for label, path in model_map.items():
        print(f"\nBenchmarking {label}: {path}")
        prec_results = []
        for pl in PROMPT_LENGTHS:
            print(f"  Prompt length: {pl}")
            m = run_llama_bench(args.llama_bench, path, pl)
            if m:
                prec_results.append({"prompt_length": pl, **m})
                print(f"    TG: {m['tg_tok_s']:.1f} tok/s  PTL: {m['ptl_ms']:.1f} ms")
        results[label] = prec_results

    return results


# ─────────────────────────────────────────────
# HuggingFace fallback backend
# ─────────────────────────────────────────────

def run_hf_backend(args):
    """Run benchmarks using HuggingFace transformers (F16 only, or with bitsandbytes)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    BASE_TEXT = (
        "The quick brown fox jumps over the lazy dog. "
        "Artificial intelligence is transforming the world. "
        "Large language models generate text token by token. "
    )

    def build_prompt(target_length, tokenizer):
        text = BASE_TEXT
        while len(tokenizer.encode(text)) < target_length:
            text += BASE_TEXT
        tokens = tokenizer.encode(text)[:target_length]
        return tokenizer.decode(tokens)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps"  if torch.backends.mps.is_available()
        else "cpu"
    )

    def sync():
        if device == "cuda":   torch.cuda.synchronize()
        elif device == "mps":  torch.mps.synchronize()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    results = {}

    configs = [("F16", {"torch_dtype": torch.float16})]
    try:
        import bitsandbytes  # noqa
        configs.append(("Q8_approx", {"load_in_8bit": True}))
        configs.append(("Q4_approx", {"load_in_4bit": True}))
        print("bitsandbytes found — will test 8-bit and 4-bit quantization")
    except ImportError:
        print("bitsandbytes not found — testing F16 only. Install with: pip install bitsandbytes")

    for label, kwargs in configs:
        print(f"\nLoading {label} model...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, device_map=device, **kwargs
            )
            model.eval()
        except Exception as e:
            print(f"  Skipping {label}: {e}")
            continue

        prec_results = []
        for pl in PROMPT_LENGTHS:
            prompt = build_prompt(pl, tokenizer)
            actual = len(tokenizer.encode(prompt))
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Warm-up
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=5, do_sample=False)

            ttfts, ptls = [], []
            for _ in range(3):
                inputs2 = tokenizer(prompt, return_tensors="pt").to(device)
                sync()
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model(input_ids=inputs2["input_ids"], use_cache=True)
                sync()
                ttfts.append((time.perf_counter() - t0) * 1000)

                past = out.past_key_values
                nxt  = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tok_times = []
                for _ in range(20):
                    sync()
                    t1 = time.perf_counter()
                    with torch.no_grad():
                        out2 = model(input_ids=nxt, past_key_values=past, use_cache=True)
                    sync()
                    tok_times.append((time.perf_counter() - t1) * 1000)
                    past = out2.past_key_values
                    nxt  = out2.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ptls.append(statistics.mean(tok_times))

            entry = {
                "prompt_length": actual,
                "ttft_ms":       round(statistics.mean(ttfts), 2),
                "ptl_ms":        round(statistics.mean(ptls),  2),
                "tg_tok_s":      round(1000 / statistics.mean(ptls), 2),
            }
            prec_results.append(entry)
            print(f"  Prompt {actual} tok — TTFT: {entry['ttft_ms']} ms  PTL: {entry['ptl_ms']} ms")

        results[label] = prec_results
        del model
        if device == "mps": torch.mps.empty_cache()

    return results


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_results(results: dict, model_name: str, device: str):
    labels = list(results.keys())
    if not labels:
        print("No results to plot.")
        return

    prompt_lengths = [r["prompt_length"] for r in next(iter(results.values()))]

    # Compute speedups relative to the most expensive precision (F16 or first)
    baseline_key = "F16" if "F16" in results else labels[-1]
    baseline_ptl = {r["prompt_length"]: r["ptl_ms"] for r in results[baseline_key]}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Quantization Speedup: Q4 vs Q8 vs F16\n{model_name} · {device}",
                 fontsize=13, fontweight="bold")

    # PTL across prompt lengths
    ax = axes[0]
    for label, data in results.items():
        color = PRECISION_COLORS.get(label, "#888")
        pls   = [r["prompt_length"] for r in data]
        ptls  = [r["ptl_ms"]        for r in data]
        ax.plot(pls, ptls, "o-", color=color, linewidth=2, label=label)
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Per-token latency (ms)")
    ax.set_title("PTL vs prompt length")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Speedup vs F16 baseline
    ax = axes[1]
    x  = np.arange(len(prompt_lengths))
    width = 0.8 / max(len(labels) - 1, 1)
    for i, (label, data) in enumerate(results.items()):
        if label == baseline_key:
            continue
        speedups = [baseline_ptl[r["prompt_length"]] / r["ptl_ms"] for r in data]
        color    = PRECISION_COLORS.get(label, "#888")
        offset   = (i - len(labels) / 2) * width
        bars     = ax.bar(x + offset, speedups, width * 0.85, color=color, alpha=0.85, label=label)
        for bar, val in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}×", ha="center", va="bottom", fontsize=8)
    ax.axhline(1.0, color="#EF4444", linestyle="--", linewidth=1.5, label=f"{baseline_key} baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{pl} tok" for pl in prompt_lengths])
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel(f"Speedup (× vs {baseline_key})")
    ax.set_title(f"Speedup vs {baseline_key} (lower PTL = higher speedup)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Throughput comparison at each prompt length
    ax = axes[2]
    for label, data in results.items():
        color = PRECISION_COLORS.get(label, "#888")
        pls   = [r["prompt_length"] for r in data]
        tputs = [r.get("tg_tok_s", 1000 / r["ptl_ms"]) for r in data]
        ax.plot(pls, tputs, "s--", color=color, linewidth=1.8, label=label)
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Decode throughput by precision")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(PLOTS_DIR, "07_quantization_speedup.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_png}")
    plt.show()


def main(args):
    if args.backend == "llamacpp":
        if not args.llama_bench:
            raise ValueError("--llama_bench path required for llamacpp backend")
        if not args.models:
            raise ValueError("--models required: e.g. --models q4=/path/q4.gguf f16=/path/f16.gguf")
        results = run_llamacpp_backend(args)
        device  = "mps/cpu (llama.cpp)"
        model_name = args.models[0].split("=")[-1].split("/")[-1]
    else:
        results = run_hf_backend(args)
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        model_name = args.model

    # Save JSON
    out_json = os.path.join(RESULTS_DIR, "07_quantization_speedup.json")
    with open(out_json, "w") as f:
        json.dump({"model": model_name, "device": device, "data": results}, f, indent=2)
    print(f"\nResults saved → {out_json}")

    plot_results(results, model_name, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantization speedup benchmark")
    parser.add_argument("--backend",     choices=["llamacpp", "hf"], default="hf",
                        help="llamacpp (recommended) or hf (HuggingFace fallback)")
    parser.add_argument("--model",       default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="HuggingFace model ID (hf backend only)")
    parser.add_argument("--llama_bench", default=None,
                        help="Path to llama-bench binary")
    parser.add_argument("--models",      nargs="+", default=None,
                        help="label=path pairs, e.g. Q4=/path/q4.gguf Q8=/path/q8.gguf F16=/path/f16.gguf")
    main(parser.parse_args())
