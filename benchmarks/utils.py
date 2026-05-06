"""
benchmarks/utils.py
--------------------
Shared utilities for all benchmark scripts.

Provides: get_device, sync_device, load_model_and_tokenizer,
          save_csv, save_json, save_metadata, set_seed,
          build_prompt, ensure_output_dir, print_run_header.

Import in any benchmark script:
    from utils import get_device, sync_device, load_model_and_tokenizer, ...
"""

import csv
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch

try:
    import transformers as _tf
    TRANSFORMERS_VERSION = _tf.__version__
except ImportError:
    TRANSFORMERS_VERSION = "unknown"

_BASE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Artificial intelligence is transforming the world. "
    "Large language models generate text token by token. "
    "Memory bandwidth is the dominant bottleneck in autoregressive decoding. "
)


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> str:
    """Return best available device: 'mps', 'cuda', or 'cpu'."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sync_device(device: str) -> None:
    """Synchronize device before taking a wall-clock timestamp."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def print_run_header(model_name: str, device: str, device_name: str,
                     precision: str, output_path: str) -> None:
    """Print a standardized run header for every benchmark script."""
    print("=" * 60)
    print(f"  Model:     {model_name}")
    print(f"  Device:    {device}  (MPS: {torch.backends.mps.is_available()})")
    print(f"  Label:     {device_name}")
    print(f"  Precision: {precision}")
    print(f"  Output:    {output_path}")
    print("=" * 60)
    if device == "cpu":
        print("  WARNING: MPS not available. Falling back to CPU.")
        print("  Results will not match paper values from Apple Silicon.")
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
    Returns (model, tokenizer). Raises RuntimeError on failure.
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
            model_name, torch_dtype=dtype, device_map=device
        )
        model.eval()
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model '{model_name}' on '{device}': {e}"
        ) from e

    return model, tokenizer


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(target_length: int, tokenizer) -> str:
    """Return a string tokenized to exactly `target_length` tokens."""
    text = _BASE_TEXT
    while len(tokenizer.encode(text)) < target_length:
        text += _BASE_TEXT
    tokens = tokenizer.encode(text)[:target_length]
    return tokenizer.decode(tokens, skip_special_tokens=True)


# ── Output paths ──────────────────────────────────────────────────────────────

def ensure_output_dir(output_dir: str, device_name: str) -> Path:
    """Create and return Path(output_dir) / device_name."""
    path = Path(output_dir) / device_name
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── CSV I/O ───────────────────────────────────────────────────────────────────

def save_csv(rows: list, filepath) -> None:
    """Write list of dicts to CSV. All rows must share the same keys."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"  [save_csv] No rows → {filepath}")
        return
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved → {filepath}")


# ── JSON I/O ──────────────────────────────────────────────────────────────────

def save_json(data, filepath) -> None:
    """Write data as indented JSON."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved → {filepath}")


# ── Metadata ──────────────────────────────────────────────────────────────────

def save_metadata(model_name: str, device: str, device_name: str,
                  precision: str, measurement_type: str,
                  output_dir, filename: str = "metadata.json") -> None:
    """Write a run-level metadata JSON for audit and reproducibility."""
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


# ── KV-cache sizing (analytical) ──────────────────────────────────────────────

def kv_cache_mb(layers: int, kv_heads: int, head_dim: int,
                context: int, bytes_per_elem: float) -> float:
    """
    KV-cache size in MB (analytical formula).
    M_KV = 2 * layers * kv_heads * head_dim * context * bytes_per_elem

    TinyLlama-1.1B: kv_cache_mb(22, 4, 64, 1000, 2.0) ≈ 22.5 MB
    """
    return 2 * layers * kv_heads * head_dim * context * bytes_per_elem / 1e6
