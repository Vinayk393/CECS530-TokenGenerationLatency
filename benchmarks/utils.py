"""
benchmarks/utils.py
====================
Shared utilities for all benchmark scripts.

Methodology alignment with paper (Section 4.3, 4.5):
- sync_device(): calls torch.mps.synchronize() before every perf_counter()
  checkpoint — prevents MPS async execution from causing underestimates.
- build_prompt(): fixed repeated text so only prompt LENGTH varies.
- kv_cache_mb(): M_KV = 2 * L * H_kv * d_head * n_ctx * b  (paper Eq. 4)
  TinyLlama F16 @ 1024 tokens = 2*22*4*64*1024*2 ≈ 22.9 MB (paper Eq. 5).

Import in any benchmark:
    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import get_device, sync_device, load_model_and_tokenizer, ...
"""

import csv, json, platform, random, time
from pathlib import Path
import numpy as np
import torch

try:
    import transformers as _tf
    TRANSFORMERS_VERSION = _tf.__version__
except ImportError:
    TRANSFORMERS_VERSION = "unknown"

# Fixed repeated text (paper §4.3: "isolate length effect")
_BASE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Large language models generate text one token at a time. "
    "Memory bandwidth is the dominant bottleneck in autoregressive decoding. "
    "Apple Silicon uses a unified memory architecture shared by CPU and GPU. "
    "KV-cache grows linearly with context length, increasing DRAM traffic. "
)


# ── Device ─────────────────────────────────────────────────────────────────────

def get_device() -> str:
    """Return best available device: 'mps' | 'cuda' | 'cpu'."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sync_device(device: str) -> None:
    """
    Synchronize device before every perf_counter() call.
    Paper §4.3: 'torch.mps.synchronize() before each perf_counter() checkpoint.'
    Without this, async MPS execution causes systematic latency underestimates.
    """
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def print_run_header(model_name, device, device_name, precision, output_path):
    print("=" * 62)
    print(f"  Model:     {model_name}")
    print(f"  Device:    {device}  (MPS: {torch.backends.mps.is_available()})")
    print(f"  Label:     {device_name}")
    print(f"  Precision: {precision}")
    print(f"  Output:    {output_path}")
    print("=" * 62)
    if device == "cpu":
        print("  WARNING: MPS unavailable — CPU fallback. Results != paper.")
        print("=" * 62)


# ── Reproducibility ─────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Model loading ───────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_name: str, device: str,
                              dtype: torch.dtype = torch.float16):
    """
    Load HuggingFace causal LM + tokenizer.
    Uses float16 (bfloat16 unsupported on MPS — paper §4.2).
    Raises RuntimeError with clear message on failure.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Loading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:
        raise RuntimeError(f"Tokenizer load failed '{model_name}': {exc}") from exc

    print(f"  Loading model [{dtype}] ...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device)
        model.eval()
    except Exception as exc:
        raise RuntimeError(f"Model load failed '{model_name}' on '{device}': {exc}") from exc
    return model, tokenizer


# ── Prompt construction ─────────────────────────────────────────────────────────

def build_prompt(target_length: int, tokenizer) -> str:
    """
    Return string tokenized to exactly `target_length`.
    Paper §4.3: fixed repeated text isolates length as sole variable.
    """
    text = _BASE_TEXT
    while len(tokenizer.encode(text)) < target_length:
        text += _BASE_TEXT
    tokens = tokenizer.encode(text)[:target_length]
    return tokenizer.decode(tokens, skip_special_tokens=True)


# ── Output paths ────────────────────────────────────────────────────────────────

def ensure_output_dir(output_dir: str, device_name: str) -> Path:
    """Create and return Path(output_dir)/device_name. Repo-relative only."""
    path = Path(output_dir) / device_name
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── I/O ─────────────────────────────────────────────────────────────────────────

def save_csv(rows: list, filepath) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved CSV  → {filepath}")


def save_json(data, filepath) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved JSON → {filepath}")


def save_metadata(model_name, device, device_name, precision,
                  measurement_type, output_dir, filename="metadata.json"):
    """Write per-run metadata JSON for full audit trail."""
    meta = {
        "model": model_name, "device": device, "hardware_label": device_name,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": TRANSFORMERS_VERSION,
        "precision": precision, "platform": platform.platform(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "measurement_type": measurement_type,
        "paper": "Token-Generation Latency Benchmarking in LLaMA, CECS 530, 2026",
        "repo": "https://github.com/Vinayk393/CECS530-TokenGenerationLatency",
    }
    save_json(meta, Path(output_dir) / filename)


# ── KV-cache sizing (paper Eq. 4) ───────────────────────────────────────────────

def kv_cache_mb(layers: int, kv_heads: int, head_dim: int,
                context: int, bytes_per_elem: float) -> float:
    """
    KV-cache size in MB. Paper Equation 4:
        M_KV = 2 * L * H_kv * d_head * n_ctx * b

    Factor of 2: both K and V tensors stored per layer.

    TinyLlama-1.1B (L=22, H_kv=4, d_head=64), F16 (b=2):
        kv_cache_mb(22, 4, 64, 1024, 2.0) ≈ 22.9 MB  (paper Eq. 5 ✓)

    Quantization impact (per 1000 tokens):
        F16  b=2.0 → 22.5 MB
        Q8_0 b=1.0 → 11.2 MB  (2× smaller)
        Q4   b=0.5 →  5.6 MB  (4× smaller)
    """
    return 2 * layers * kv_heads * head_dim * context * bytes_per_elem / 1_000_000
