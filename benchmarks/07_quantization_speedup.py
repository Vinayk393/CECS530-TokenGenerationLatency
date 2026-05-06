"""
07_quantization_speedup.py
-----------------------------
Benchmarks quantization speedup (Q4_K_M vs Q8_0 vs F16).

Two modes:
  --mode modeled   (DEFAULT) Projects Q4/Q8 speedup from the measured F16
                   baseline using bandwidth-scaling assumptions. Output is
                   labeled measurement_type=modeled. No GGUF files needed.

  --backend llamacpp  Runs llama-bench with real GGUF model files.
                      Output is labeled measurement_type=measured.
                      Requires llama.cpp installed and GGUF files.

Evidence label:
  - 'modeled'  when --mode modeled (default)
  - 'measured' when --backend llamacpp with real GGUF files

Usage (modeled — default, for paper reproducibility):
    python 07_quantization_speedup.py --mode modeled
    python 07_quantization_speedup.py --mode modeled --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

Usage (llama.cpp — for actual measured quantization speedup):
    python 07_quantization_speedup.py \\
        --backend llamacpp \\
        --llama_bench /path/to/llama.cpp/build/bin/llama-bench \\
        --models Q4=/path/to/model-Q4_K_M.gguf Q8=/path/to/model-Q8_0.gguf F16=/path/to/model-F16.gguf

Output:
    results/<device_name>/07_quantization_speedup.csv
    results/<device_name>/07_quantization_speedup_metadata.json
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    get_device,
    sync_device,
    load_model_and_tokenizer,
    build_prompt,
    set_seed,
    ensure_output_dir,
    save_csv,
    save_json,
    save_metadata,
    print_run_header,
)

PROMPT_LENGTHS = [32, 128, 256, 512]

# Bandwidth-scaling multipliers relative to F16 (theoretical, not measured)
# Q4_K_M: 4x fewer bytes → up to 4x speedup, bounded by fixed overhead
# Q8_0:   2x fewer bytes → up to 2x speedup
# Gains are context-dependent: grow as KV-cache dominates over weights.
# These are calibrated from paper analysis (Section 7).
MODELED_SPEEDUP = {
    # (prompt_length -> speedup_factor)
    "Q4_K_M": {32: 1.53, 128: 1.82, 256: 2.05, 512: 2.31},
    "Q8_0":   {32: 1.21, 128: 1.32, 256: 1.38, 512: 1.42},
    "F16":    {32: 1.00, 128: 1.00, 256: 1.00, 512: 1.00},
}

PRECISION_COLORS = {
    "Q4_K_M": "#3B82F6",
    "Q8_0":   "#10B981",
    "F16":    "#EF4444",
}


# ─────────────────────────────────────────────
# Modeled mode: measure F16, project Q4/Q8
# ─────────────────────────────────────────────

def run_modeled_backend(args, device: str) -> tuple[dict, str]:
    """
    Measure F16 PTL directly, then project Q4_K_M and Q8_0 speedups using
    bandwidth-scaling multipliers from the paper analysis.

    Returns (results dict, measurement_type string).
    """
    model, tokenizer = load_model_and_tokenizer(args.model, device, torch.float16)

    f16_ptl = {}
    print("\n  Measuring F16 baseline PTL...")
    for pl in PROMPT_LENGTHS:
        prompt = build_prompt(pl, tokenizer)
        actual = len(tokenizer.encode(prompt))
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Warm-up
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=2, do_sample=False)

        ptls = []
        past_kv = None
        inp_ids = inputs["input_ids"]

        # Prefill pass
        sync_device(device)
        with torch.no_grad():
            out = model(input_ids=inp_ids, use_cache=True)
        past_kv = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # Decode 20 tokens, skip first 2 transients
        for i in range(22):
            sync_device(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                out2 = model(input_ids=next_tok, past_key_values=past_kv, use_cache=True)
            sync_device(device)
            elapsed = (time.perf_counter() - t0) * 1000
            past_kv = out2.past_key_values
            next_tok = out2.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if i >= 2:
                ptls.append(elapsed)

        f16_ptl[pl] = round(statistics.mean(ptls), 2)
        print(f"    prompt={pl} tok → F16 PTL = {f16_ptl[pl]:.2f} ms")

    # Project Q4/Q8 from F16 using bandwidth-scaling multipliers
    results = {}
    for prec, speedups in MODELED_SPEEDUP.items():
        prec_rows = []
        for pl in PROMPT_LENGTHS:
            f16 = f16_ptl[pl]
            if prec == "F16":
                ptl = f16
                label = "measured"
            else:
                ptl = round(f16 / speedups[pl], 2)
                label = "modeled"
            prec_rows.append({
                "precision": prec,
                "prompt_length": pl,
                "ptl_ms": ptl,
                "f16_ptl_ms": f16,
                "speedup_vs_f16": round(f16 / ptl, 3),
                "throughput_tok_s": round(1000 / ptl, 2),
                "measurement_type": label,
            })
        results[prec] = prec_rows

    return results, "modeled"


# ─────────────────────────────────────────────
# llama.cpp backend (real measured speedup)
# ─────────────────────────────────────────────

def run_llama_bench(llama_bench_path: str, model_path: str,
                    prompt_len: int, gen_len: int = 50) -> dict | None:
    cmd = [
        llama_bench_path, "-m", model_path,
        "-p", str(prompt_len), "-n", str(gen_len),
        "-r", "3", "--output", "json",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"    llama-bench error: {result.stderr[:200]}")
            return None
        data = json.loads(result.stdout)
        if isinstance(data, list) and len(data) > 0:
            entry = data[0]
            tg = entry.get("tg_ts", entry.get("t_tg_avg"))
            return {
                "tg_tok_s": tg,
                "ptl_ms": round(1000 / tg, 2) if tg else None,
            }
    except Exception as e:
        print(f"    Error: {e}")
        return None


def run_llamacpp_backend(args) -> tuple[dict, str]:
    model_map = {}
    for item in args.models:
        label, path = item.split("=", 1)
        model_map[label.upper()] = path

    results = {}
    for label, path in model_map.items():
        print(f"\n  Benchmarking {label}: {path}")
        rows = []
        for pl in PROMPT_LENGTHS:
            m = run_llama_bench(args.llama_bench, path, pl)
            if m and m["ptl_ms"]:
                rows.append({
                    "precision": label,
                    "prompt_length": pl,
                    "ptl_ms": m["ptl_ms"],
                    "throughput_tok_s": round(m["tg_tok_s"], 2),
                    "measurement_type": "measured",
                })
                print(f"    prompt={pl} → PTL={m['ptl_ms']:.1f} ms")
        results[label] = rows

    # Compute speedup vs F16
    if "F16" in results:
        f16_map = {r["prompt_length"]: r["ptl_ms"] for r in results["F16"]}
        for label, rows in results.items():
            for row in rows:
                f16 = f16_map.get(row["prompt_length"], row["ptl_ms"])
                row["f16_ptl_ms"] = f16
                row["speedup_vs_f16"] = round(f16 / row["ptl_ms"], 3)

    return results, "measured"


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_results(results: dict, model_name: str, device: str,
                 out_dir: Path, measurement_type: str) -> None:
    labels = list(results.keys())
    if not labels:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    title_suffix = f"[{measurement_type}]"
    fig.suptitle(
        f"Quantization Speedup: Q4_K_M vs Q8_0 vs F16  {title_suffix}\n"
        f"{model_name} · {device}",
        fontsize=12, fontweight="bold"
    )

    # PTL vs prompt length
    ax = axes[0]
    for label, rows in results.items():
        pls  = [r["prompt_length"] for r in rows]
        ptls = [r["ptl_ms"]        for r in rows]
        color = PRECISION_COLORS.get(label, "#888")
        style = "-o" if rows[0].get("measurement_type") == "measured" else "--s"
        ax.plot(pls, ptls, style, color=color, linewidth=2, label=label)
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Per-token latency (ms)")
    ax.set_title("PTL vs prompt length")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Speedup vs F16
    ax = axes[1]
    baseline_key = "F16"
    if baseline_key in results:
        f16_map = {r["prompt_length"]: r["ptl_ms"] for r in results[baseline_key]}
        x = np.arange(len(PROMPT_LENGTHS))
        width = 0.35
        for i, (label, rows) in enumerate(results.items()):
            if label == baseline_key:
                continue
            speedups = [f16_map.get(r["prompt_length"], r["ptl_ms"]) / r["ptl_ms"]
                        for r in rows]
            color = PRECISION_COLORS.get(label, "#888")
            offset = (i - 1) * width
            bars = ax.bar(x + offset, speedups, width * 0.85,
                          color=color, alpha=0.85, label=label)
            for bar, val in zip(bars, speedups):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{val:.2f}×", ha="center", va="bottom", fontsize=8)
        ax.axhline(1.0, color="#EF4444", linestyle="--", linewidth=1.5,
                   label="F16 baseline")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{pl} tok" for pl in PROMPT_LENGTHS])
        ax.set_xlabel("Prompt length (tokens)")
        ax.set_ylabel("Speedup (× vs F16)")
        ax.set_title(f"Speedup vs F16  [{measurement_type}]")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_png = out_dir / "07_quantization_speedup.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {out_png}")
    plt.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(args):
    set_seed(args.seed)
    device = get_device()
    out_dir = ensure_output_dir(args.output_dir, args.device_name)

    print_run_header(
        model_name=args.model,
        device=device,
        device_name=args.device_name,
        precision="F16 (baseline measured); Q4/Q8 modeled"
        if args.mode == "modeled" else "varies (llamacpp)",
        output_path=str(out_dir),
    )

    if args.backend == "llamacpp":
        if not args.llama_bench or not args.models:
            raise ValueError(
                "--llama_bench and --models are required for llamacpp backend.\n"
                "Run without --backend (or with --mode modeled) for the modeled projection."
            )
        results, mtype = run_llamacpp_backend(args)
    else:
        results, mtype = run_modeled_backend(args, device)

    # Flatten to rows for CSV
    all_rows = []
    for rows in results.values():
        all_rows.extend(rows)

    save_csv(all_rows, out_dir / "07_quantization_speedup.csv")
    save_metadata(
        model_name=args.model,
        device=device,
        device_name=args.device_name,
        precision="f16_baseline_measured_q4q8_modeled" if mtype == "modeled" else "varies",
        measurement_type=mtype,
        output_dir=out_dir,
        filename="07_quantization_speedup_metadata.json",
    )

    plot_results(results, args.model, device, out_dir, mtype)
    print(f"\n  measurement_type: {mtype}")
    if mtype == "modeled":
        print("  NOTE: Q4_K_M and Q8_0 values are projections from the F16 baseline.")
        print("        To obtain measured speedups, use --backend llamacpp with GGUF files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantization speedup benchmark")
    parser.add_argument(
        "--mode", choices=["modeled"], default="modeled",
        help="modeled: measure F16 and project Q4/Q8 (default)",
    )
    parser.add_argument(
        "--backend", choices=["llamacpp", "hf"], default="hf",
        help="llamacpp: use real GGUF files; hf: use HuggingFace (modeled only)",
    )
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--llama_bench", default=None,
                        help="Path to llama-bench binary (llamacpp backend only)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="label=path pairs (llamacpp backend only)")
    parser.add_argument("--device_name", default="Mac_M4_16GB",
                        help="Device label for output path (e.g., Mac_M4_16GB)")
    parser.add_argument("--output_dir", default="results",
                        help="Root output directory")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
