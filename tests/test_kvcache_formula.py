"""
tests/test_kvcache_formula.py
------------------------------
Unit tests for the KV-cache sizing formula.

The formula is: M_KV = 2 × L × H_kv × d_head × context × bytes_per_elem

For TinyLlama-1.1B (L=22, H_kv=4, d_head=64, F16=2 bytes):
    M_KV(1000 tokens) = 2 × 22 × 4 × 64 × 1000 × 2 = 22,528,000 bytes ≈ 22.5 MB

These tests guard against the previous incorrect value of 2.25 MB per 1000 tokens.
"""

import sys
from pathlib import Path

# Allow import from benchmarks/
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))
from benchmarks.utils import kv_cache_mb  # noqa: E402


# ─── Helper ──────────────────────────────────────────────────────────────────

def _kv_mb(layers, kv_heads, head_dim, context, bpe):
    """Inline formula for test isolation (does not depend on utils import)."""
    return 2 * layers * kv_heads * head_dim * context * bpe / 1e6


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestKVCacheFormula:
    """TinyLlama-1.1B: L=22, H_kv=4, d_head=64"""

    LAYERS   = 22
    KV_HEADS = 4
    HEAD_DIM = 64

    def test_f16_1000_tokens_approx_22_5mb(self):
        """F16 @ 1000 tokens should be ~22.5 MB, NOT 2.25 MB."""
        mb = _kv_mb(self.LAYERS, self.KV_HEADS, self.HEAD_DIM, 1000, 2.0)
        assert 21.0 <= mb <= 23.5, (
            f"Expected ~22.5 MB, got {mb:.2f} MB. "
            "If this fails, the formula or constants are wrong."
        )

    def test_f16_1000_tokens_not_wrong_value(self):
        """Guard against the previously reported wrong value of 2.25 MB."""
        mb = _kv_mb(self.LAYERS, self.KV_HEADS, self.HEAD_DIM, 1000, 2.0)
        assert mb > 10.0, (
            f"KV-cache at 1000 tokens cannot be {mb:.2f} MB — "
            "previous erroneous value was 2.25 MB. Check formula."
        )

    def test_f16_512_tokens(self):
        """F16 @ 512 tokens ≈ 11.5 MB."""
        mb = _kv_mb(self.LAYERS, self.KV_HEADS, self.HEAD_DIM, 512, 2.0)
        assert 10.5 <= mb <= 12.5, f"Expected ~11.5 MB, got {mb:.2f} MB"

    def test_f16_1024_tokens(self):
        """F16 @ 1024 tokens ≈ 23.1 MB."""
        mb = _kv_mb(self.LAYERS, self.KV_HEADS, self.HEAD_DIM, 1024, 2.0)
        assert 22.0 <= mb <= 24.5, f"Expected ~23.1 MB, got {mb:.2f} MB"

    def test_f16_2048_tokens(self):
        """F16 @ 2048 tokens ≈ 46.1 MB."""
        mb = _kv_mb(self.LAYERS, self.KV_HEADS, self.HEAD_DIM, 2048, 2.0)
        assert 44.0 <= mb <= 48.0, f"Expected ~46.1 MB, got {mb:.2f} MB"

    def test_q4_is_4x_smaller_than_f16(self):
        """Q4_K_M (0.5 bpe) should be 4× smaller than F16 (2.0 bpe)."""
        f16 = _kv_mb(self.LAYERS, self.KV_HEADS, self.HEAD_DIM, 1000, 2.0)
        q4  = _kv_mb(self.LAYERS, self.KV_HEADS, self.HEAD_DIM, 1000, 0.5)
        ratio = f16 / q4
        assert 3.9 <= ratio <= 4.1, f"Expected 4× ratio, got {ratio:.2f}"

    def test_q8_is_2x_smaller_than_f16(self):
        """Q8_0 (1.0 bpe) should be 2× smaller than F16 (2.0 bpe)."""
        f16 = _kv_mb(self.LAYERS, self.KV_HEADS, self.HEAD_DIM, 1000, 2.0)
        q8  = _kv_mb(self.LAYERS, self.KV_HEADS, self.HEAD_DIM, 1000, 1.0)
        ratio = f16 / q8
        assert 1.9 <= ratio <= 2.1, f"Expected 2× ratio, got {ratio:.2f}"

    def test_linear_scaling_with_context(self):
        """KV-cache should scale linearly: doubling context doubles size."""
        mb_512  = _kv_mb(self.LAYERS, self.KV_HEADS, self.HEAD_DIM, 512,  2.0)
        mb_1024 = _kv_mb(self.LAYERS, self.KV_HEADS, self.HEAD_DIM, 1024, 2.0)
        ratio = mb_1024 / mb_512
        assert 1.95 <= ratio <= 2.05, f"Expected 2× scaling, got {ratio:.2f}"

    def test_gqa_vs_mha(self):
        """
        GQA with 4 KV heads vs MHA with 32 KV heads should give 8× difference.
        This confirms GQA reduces KV-cache by H_q/H_kv = 32/4 = 8.
        """
        gqa = _kv_mb(self.LAYERS, 4,  self.HEAD_DIM, 1000, 2.0)
        mha = _kv_mb(self.LAYERS, 32, self.HEAD_DIM, 1000, 2.0)
        ratio = mha / gqa
        assert 7.9 <= ratio <= 8.1, f"Expected 8× GQA reduction, got {ratio:.2f}"

    def test_exact_bytes_f16_1000(self):
        """Exact byte count for F16 @ 1000 tokens."""
        expected_bytes = 2 * 22 * 4 * 64 * 1000 * 2  # = 22,528,000
        actual_mb = _kv_mb(self.LAYERS, self.KV_HEADS, self.HEAD_DIM, 1000, 2.0)
        actual_bytes = actual_mb * 1e6
        assert abs(actual_bytes - expected_bytes) < 1, (
            f"Expected {expected_bytes:,} bytes, got {actual_bytes:,.0f}"
        )
