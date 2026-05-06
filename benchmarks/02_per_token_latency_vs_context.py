"""
benchmarks/02_per_token_latency_vs_context.py
==============================================
Measures per-token latency (PTL) as a function of context length.

Paper alignment (Section 4.3, 4.5):
  "PTL window: 20 tokens; PTL exclude: First 2 tokens (eliminate transient)"

  Exact decode loop from paper (Section 4.5):
      for i in range(n_tokens + 2):   # +2 for excluded transients
          torch.mps.synchronize()
          t0 = time.perf_counter()
          out = model(input_ids=next_tok, past_key_values=past_kv, use_cache=True)
          torch.mps.synchronize()
          ptl_ms = (time.perf_counter() - t0) * 1000
          if i >= 2:  # skip first two transient tokens
              samples.append(ptl_ms)

  N_TRANSIENT_SKIP = 2  (paper: "First two tokens excluded")
  N_DECODE_TOKENS  = 20 (paper: "20 tokens per measurement window")

  Token 0: pipeline boundary with TTFT prefill
  Token 1: MPS pipeline ramp-up transient
  Tokens 2-21: steady-state decode → PTL samples

Evidence label: measured
Output: results/<device_name>/02_per_token_latency_vs_context.csv

Expected results (paper Table 5):
    M4: ctx32→19.3ms, ctx128→20.4ms, ctx256→22.7ms, ctx512→30.1ms, ctx1024→48.6ms
    M2: ctx32→28.5ms, ctx128→30.9ms, ctx256→35.1ms, ctx512→45.8ms, ctx1024→76.4ms
"""

import argparse, statistics, sys, time
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
from utils import (build_prompt, ensure_output_dir, get_device,
                   load_model_and_tokenizer, print_run_header,
                   save_csv, save_metadata, set_seed, sync_device)

# Paper §4.3 — must match paper exactly
CONTEXT_LENGTHS  = [32, 128, 256, 512, 1024]
N_TRANSIENT_SKIP = 2    # "PTL exclude: First 2 tokens"
N_DECODE_TOKENS  = 20   # "PTL window: 20 tokens"


def measure_ptl(model, tokenizer, device, context_length):
    """
    Measure PTL at a given context length using the exact decode loop
    described in paper Section 4.5.

    Loop runs N_TRANSIENT_SKIP + N_DECODE_TOKENS = 22 steps.
    Only steps i >= N_TRANSIENT_SKIP contribute to PTL samples.
    """
    prompt   = build_prompt(context_length, tokenizer)
    actual   = len(tokenizer.encode(prompt))
    inputs   = tokenizer(prompt, return_tensors="pt").to(device)

    # ── Prefill (establishes KV-cache, not timed as PTL) ──────────────────────
    with torch.no_grad():
        out = model(input_ids=inputs["input_ids"], use_cache=True)
    past_kv  = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # ── Decode loop — exactly as paper Section 4.5 ────────────────────────────
    samples      = []
    total_steps  = N_TRANSIENT_SKIP + N_DECODE_TOKENS  # 22 steps

    for i in range(total_steps):
        sync_device(device)               # ← paper: mps.synchronize() before timer
        t0 = time.perf_counter()
        with torch.no_grad():
            out2 = model(input_ids=next_tok, past_key_values=past_kv, use_cache=True)
        sync_device(device)               # ← paper: mps.synchronize() after
        elapsed_ms = (time.perf_counter() - t0) * 1000

        past_kv  = out2.past_key_values
        next_tok = out2.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        if i >= N_TRANSIENT_SKIP:         # ← paper: "if i >= 2: samples.append"
            samples.append(elapsed_ms)

    assert len(samples) == N_DECODE_TOKENS, (
        f"Expected {N_DECODE_TOKENS} samples, got {len(samples)}"
    )
    sorted_s = sorted(samples)
    return {
        "context_length":   actual,
        "ptl_mean_ms":      round(statistics.mean(samples), 2),
        "ptl_std_ms":       round(statistics.stdev(samples), 2),
        "ptl_p90_ms":       round(sorted_s[int(0.9 * len(sorted_s))], 2),
        "ptl_min_ms":       round(min(samples), 2),
        "ptl_max_ms":       round(max(samples), 2),
        "throughput_tok_s": round(1000.0 / statistics.mean(samples), 2),
        "n_samples":        len(samples),
        "n_transient_skip": N_TRANSIENT_SKIP,
        "measurement_type": "measured",
        "timing_method":    "manual_kvcache_decode_loop_paper_sec4.5",
    }


def main(args):
    set_seed(args.seed)
    device  = get_device()
    out_dir = ensure_output_dir(args.output_dir, args.device_name)
    print_run_header(args.model, device, args.device_name, "float16", str(out_dir))
    print(f"  PTL tokens collected:  {N_DECODE_TOKENS}")
    print(f"  Transient tokens skip: {N_TRANSIENT_SKIP}  (paper §4.3)")
    print(f"  Context lengths: {CONTEXT_LENGTHS}\n")

    model, tokenizer = load_model_and_tokenizer(args.model, device, torch.float16)

    rows = []
    for cl in CONTEXT_LENGTHS:
        print(f"  Context {cl:>5} tokens ...", end=" ", flush=True)
        r = measure_ptl(model, tokenizer, device, cl)
        r.update({"device": device, "device_name": args.device_name,
                  "model": args.model, "precision": "float16"})
        rows.append(r)
        print(f"PTL = {r['ptl_mean_ms']:.1f} ± {r['ptl_std_ms']:.1f} ms  "
              f"({r['throughput_tok_s']:.1f} tok/s)")

    save_csv(rows, out_dir / "02_per_token_latency_vs_context.csv")
    save_metadata(args.model, device, args.device_name, "float16",
                  "measured", out_dir, "02_per_token_latency_vs_context_metadata.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PTL vs context length [measured]")
    parser.add_argument("--model",       default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device_name", default="Mac_M4_16GB")
    parser.add_argument("--output_dir",  default=str(REPO_ROOT / "results"))
    parser.add_argument("--seed",        type=int, default=42)
    main(parser.parse_args())
