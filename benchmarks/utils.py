"""
benchmarks/utils.py
--------------------
Shared utilities for all benchmark scripts.

Provides: get_device, sync_device, load_model_and_tokenizer, save_csv,
          save_json, save_metadata, set_seed, build_prompt,
          ensure_output_dir, get_device_label, print_run_header.
"""

import csv
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    import transformers
    TRANSFORMERS_VERSION = transformers.__version__
except ImportError:
    TRANSFORMERS_VERSION = "unknown"


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> str:
    """Return the best available device string: 'mps', 'cuda', or 'cpu'."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sync_device(device: str) -> None:
    """Synchronize the device before taking a wall-clock timestamp."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    # CPU: no-op


def print_run_header(model_name: str, device: str, device_name: str,
                     precision: str, output_path: str) -> None:
    """Print a standardized run header for every benchmark script."""
    mps_available = torch.backends.mps.is_available()
    print("=" * 60)
    print(f"  Model:      {model_name}")
    print(f"  Device:     {device}  (MPS available: {mps_available})")
    print(f"  Label:      {device_name}")
    print(f"  Precision:  {precision}")
    print(f"  Output:     {output_path}")
    print("=" * 60)
    if device == "cpu" and mps_available is False:
        print("  WARNING: MPS not available. Falling back to CPU.")
        print("  Results will not match paper values obtained on Apple Silicon.")
        print("=" * 60)


# ── Seed ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_name: str, device: str,
                              dtype=torch.float16):
    """
    Load a HuggingFace causal LM and tokenizer onto `device`.

    Returns (model, tokenizer).
    Raises RuntimeError with a clear message if loading fails.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer '{model_name}': {e}") from e

    print(f"  Loading model ({dtype})...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
        )
        model.eval()
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model '{model_name}' on device '{device}': {e}"
        ) from e

    return model, tokenizer


# ── Prompt builder ────────────────────────────────────────────────────────────

_BASE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Artificial intelligence is transforming the world. "
    "Large language models generate text token by token. "
    "Memory bandwidth is the dominant bottleneck in autoregressive decoding. "
)


def build_prompt(target_length: int, tokenizer) -> str:
    """
    Return a string whose tokenized length is exactly `target_length`.
    Repeats _BASE_TEXT until long enough, then truncates.
    """
    text = _BASE_TEXT
    while len(tokenizer.encode(text)) < target_length:
        text += _BASE_TEXT
    tokens = tokenizer.encode(text)[:target_length]
    return tokenizer.decode(tokens, skip_special_tokens=True)


# ── Output paths ──────────────────────────────────────────────────────────────

def ensure_output_dir(output_dir: str, device_name: str) -> Path:
    """
    Create and return Path(output_dir) / device_name.
    E.g., ensure_output_dir('results', 'Mac_M4_16GB') → Path('results/Mac_M4_16GB')
    """
    path = Path(output_dir) / device_name
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── CSV I/O ───────────────────────────────────────────────────────────────────

def save_csv(rows: list[dict], filepath: str | Path) -> None:
    """
    Write a list of dicts to a CSV file.
    All rows must share the same keys; fieldnames taken from the first row.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"  [save_csv] No rows to write → {filepath}")
        return
    fieldnames = list(rows[0].keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved CSV → {filepath}")


# ── JSON I/O ──────────────────────────────────────────────────────────────────

def save_json(data: dict | list, filepath: str | Path) -> None:
    """Write data as indented JSON."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved JSON → {filepath}")


# ── Metadata ──────────────────────────────────────────────────────────────────

def save_metadata(model_name: str, device: str, device_name: str,
                  precision: str, measurement_type: str,
                  output_dir: str | Path, filename: str = "metadata.json") -> None:
    """
    Write a run-level metadata JSON file alongside benchmark results.
    Useful for audit trails and reproducibility verification.
    """
    meta = {
        "model": model_name,
        "device": device,
        "hardware_label": device_name,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": TRANSFORMERS_VERSION,
        "precision": precision,
        "platform": platform.platform(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "measurement_type": measurement_type,
    }
    save_json(meta, Path(output_dir) / filename)
