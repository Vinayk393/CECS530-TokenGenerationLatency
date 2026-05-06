"""
benchmarks/01_ttft_vs_prompt_length.py
=======================================
Measures Time to First Token (TTFT) across prompt lengths.

Paper alignment (Section 4.3, 4.5):
  TTFT measurement: "One full forward pass over the prompt, timed from call
  entry to first token emission. Five warm trials per configuration; first excluded."

  Timing uses raw model() forward pass (NOT model.generate()) to match the
  paper's description of "one full forward pass" without HuggingFace generation
  overhead contaminating the prefill timing.

  Timing window:
      sync_device()            ← GPU idle before timer
      t0 = perf_counter()
      model(input_ids=prompt)  ← pure prefill forward pass
      sync_device()            ← GPU done before timer
      ttft_ms = (perf_counter() - t0) * 1000

Evidence label: measured
Output: results/<device_name>/01_ttft_vs_prompt_length.csv

Expected results (paper Table 4):
    M4: 32→16.1ms, 128→26.9ms, 256→42.7ms, 512→77.4ms, 1024→197.5ms
    M2: 32→27.5ms, 128→45.0ms, 256→69.5ms, 512→121.7ms, 1024→335.9ms
"""

import argparse, statistics, sys, time
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
from utils import (build_prompt, ensure_output_dir, get_device,
                   load_model_and_tokenizer, print_run_header,
                   save_csv, save_metadata, set_seed, sync_device)

# Paper §4.3 parameters
PROMPT_LENGTHS = [32, 128, 256, 512, 1024]   # 7 log-spaced (paper uses these 5)
N_TRIALS       = 5   # "5 per config"
N_EXCLUDE      = 1   # "first excluded" (MPS kernel compile / cold cache)


def measure_ttft(model, tokenizer, device, prompt_length):
    """
    Measure TTFT as raw prefill forward pass.
    Returns dict with mean, std, min, max over warm trials.
    """
    prompt  = build_prompt(prompt_length, tokenizer)
    actual  = len(tokenizer.encode(prompt))
    inputs  = tokenizer(prompt, return_tensors="pt").to(device)

    samples = []
    for trial in range(N_TRIALS):
        sync_device(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids=inputs["input_ids"], use_cache=True)
        sync_device(device)
        samples.append((time.perf_counter() - t0) * 1000)

    warm = samples[N_EXCLUDE:]   # exclude first (MPS shader compile)
    return {
        "prompt_length":    actual,
        "ttft_mean_ms":     round(statistics.mean(warm), 2),
        "ttft_std_ms":      round(statistics.stdev(warm), 2) if len(warm) > 1 else 0.0,
        "ttft_min_ms":      round(min(warm), 2),
        "ttft_max_ms":      round(max(warm), 2),
        "n_warm_trials":    len(warm),
        "measurement_type": "measured",
        "timing_method":    "raw_prefill_forward_pass",
    }


def main(args):
    set_seed(args.seed)
    device  = get_device()
    out_dir = ensure_output_dir(args.output_dir, args.device_name)
    print_run_header(args.model, device, args.device_name, "float16", str(out_dir))
    print(f"  Timing: raw prefill forward pass (not generate())")
    print(f"  Trials: {N_TRIALS} per config, first {N_EXCLUDE} excluded")
    print(f"  Prompt lengths: {PROMPT_LENGTHS}\n")

    model, tokenizer = load_model_and_tokenizer(args.model, device, torch.float16)

    rows = []
    for pl in PROMPT_LENGTHS:
        print(f"  Prompt {pl:>5} tokens ...", end=" ", flush=True)
        r = measure_ttft(model, tokenizer, device, pl)
        r.update({"device": device, "device_name": args.device_name,
                  "model": args.model, "precision": "float16"})
        rows.append(r)
        print(f"TTFT = {r['ttft_mean_ms']:.1f} ms ± {r['ttft_std_ms']:.1f} ms")

    save_csv(rows, out_dir / "01_ttft_vs_prompt_length.csv")
    save_metadata(args.model, device, args.device_name, "float16",
                  "measured", out_dir, "01_ttft_vs_prompt_length_metadata.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTFT vs prompt length [measured]")
    parser.add_argument("--model",       default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device_name", default="Mac_M4_16GB",
                        help="Mac_M4_16GB or Mac_M2_8GB")
    parser.add_argument("--output_dir",  default=str(REPO_ROOT / "results"))
    parser.add_argument("--seed",        type=int, default=42)
    main(parser.parse_args())
