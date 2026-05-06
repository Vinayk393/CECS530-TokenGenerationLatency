"""
benchmarks/run_smoke_test.py
-----------------------------
Quick grader verification test.

- Loads TinyLlama-1.1B (or specified model) on the best available device
- Runs a 32-token prompt, generates 3 tokens
- Prints device, model, TTFT, and mean PTL
- Saves results/smoke_test.json
- Exits with code 1 if model load or generation fails

Usage:
    python benchmarks/run_smoke_test.py
    python benchmarks/run_smoke_test.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python benchmarks/run_smoke_test.py --seed 0
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    get_device, sync_device, load_model_and_tokenizer,
    build_prompt, set_seed, save_json,
)

SMOKE_OUTPUT = Path("results") / "smoke_test.json"
N_PROMPT_TOKENS = 32
N_GEN_TOKENS = 3
N_WARMUP = 1


def run_smoke_test(model_name: str, seed: int) -> dict:
    set_seed(seed)
    device = get_device()

    print("\n=== Smoke Test ===")
    print(f"  Model:   {model_name}")
    print(f"  Device:  {device}  (MPS: {torch.backends.mps.is_available()})")
    if device == "cpu":
        print("  WARNING: MPS not available. Results will not match Apple Silicon values.")

    try:
        model, tokenizer = load_model_and_tokenizer(model_name, device, torch.float16)
    except RuntimeError as e:
        print(f"\n[FAIL] Model load failed: {e}", file=sys.stderr)
        sys.exit(1)

    prompt = build_prompt(N_PROMPT_TOKENS, tokenizer)
    actual_len = len(tokenizer.encode(prompt))
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print(f"  Warm-up pass...")
    try:
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=1, do_sample=False)
    except Exception as e:
        print(f"\n[FAIL] Warm-up failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  Generating {N_GEN_TOKENS} tokens from {actual_len}-token prompt...")
    ptl_samples = []
    try:
        # Prefill
        sync_device(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids=inputs["input_ids"], use_cache=True)
        sync_device(device)
        ttft_ms = (time.perf_counter() - t0) * 1000
        past_kv = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # Decode
        for _ in range(N_GEN_TOKENS):
            sync_device(device)
            t1 = time.perf_counter()
            with torch.no_grad():
                out2 = model(input_ids=next_tok, past_key_values=past_kv, use_cache=True)
            sync_device(device)
            ptl_samples.append((time.perf_counter() - t1) * 1000)
            past_kv = out2.past_key_values
            next_tok = out2.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    except Exception as e:
        print(f"\n[FAIL] Generation failed: {e}", file=sys.stderr)
        sys.exit(1)

    mean_ptl = sum(ptl_samples) / len(ptl_samples)
    print(f"\n  TTFT:       {ttft_ms:.1f} ms")
    print(f"  Mean PTL:   {mean_ptl:.1f} ms")
    print(f"  Throughput: {1000/mean_ptl:.1f} tok/s")
    print("\n[PASS] Smoke test complete.")

    return {
        "status": "pass",
        "model": model_name,
        "device": device,
        "mps_available": torch.backends.mps.is_available(),
        "torch_version": torch.__version__,
        "prompt_tokens": actual_len,
        "n_gen_tokens": N_GEN_TOKENS,
        "ttft_ms": round(ttft_ms, 2),
        "mean_ptl_ms": round(mean_ptl, 2),
        "throughput_tok_s": round(1000 / mean_ptl, 2),
        "ptl_samples_ms": [round(x, 2) for x in ptl_samples],
        "measurement_type": "measured",
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description="Smoke test — quick environment check")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = run_smoke_test(args.model, args.seed)

    SMOKE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_json(result, SMOKE_OUTPUT)
    print(f"  Output → {SMOKE_OUTPUT}")


if __name__ == "__main__":
    main()
