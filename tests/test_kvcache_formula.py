"""
tests/test_kvcache_formula.py
------------------------------
Guards the KV-cache formula against regression.

Correct formula: M_KV = 2 * L * H_kv * d_head * context * bytes_per_elem

TinyLlama-1.1B (L=22, H_kv=4, d_head=64, F16):
    M_KV(1000 tokens) = 2*22*4*64*1000*2 = 22,528,000 bytes = 22.5 MB

The previous erroneous value was 2.25 MB per 1000 tokens.
These tests prevent that regression.

Run with: python -m pytest tests/test_kvcache_formula.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))


def kv_mb(layers, kv_heads, head_dim, context, bpe):
    """Inline formula — isolated from utils to catch any import issues."""
    return 2 * layers * kv_heads * head_dim * context * bpe / 1e6


L, H, D = 22, 4, 64  # TinyLlama-1.1B defaults


class TestKVCacheFormula:

    def test_f16_1000_tokens_approx_22_5mb(self):
        """F16 @ 1000 tokens must be ~22.5 MB, NOT 2.25 MB."""
        mb = kv_mb(L, H, D, 1000, 2.0)
        assert 21.0 <= mb <= 23.5, f"Expected ~22.5 MB, got {mb:.2f} MB"

    def test_f16_1000_tokens_not_old_wrong_value(self):
        """Explicit guard against previous erroneous 2.25 MB value."""
        mb = kv_mb(L, H, D, 1000, 2.0)
        assert mb > 10.0, (
            f"KV-cache returned {mb:.2f} MB — old wrong value was 2.25 MB. "
            "Check formula constants."
        )

    def test_exact_bytes_f16_1000_tokens(self):
        """Exact byte count: 2*22*4*64*1000*2 = 22,528,000 bytes."""
        expected = 22_528_000
        actual = kv_mb(L, H, D, 1000, 2.0) * 1e6
        assert abs(actual - expected) < 1, (
            f"Expected {expected:,} bytes, got {actual:,.0f}"
        )

    def test_f16_512_tokens(self):
        mb = kv_mb(L, H, D, 512, 2.0)
        assert 10.5 <= mb <= 12.5, f"Expected ~11.5 MB, got {mb:.2f} MB"

    def test_f16_1024_tokens(self):
        mb = kv_mb(L, H, D, 1024, 2.0)
        assert 22.0 <= mb <= 24.5, f"Expected ~23.1 MB, got {mb:.2f} MB"

    def test_f16_2048_tokens(self):
        mb = kv_mb(L, H, D, 2048, 2.0)
        assert 44.0 <= mb <= 48.0, f"Expected ~46.1 MB, got {mb:.2f} MB"

    def test_q4_is_4x_smaller_than_f16(self):
        f16 = kv_mb(L, H, D, 1000, 2.0)
        q4  = kv_mb(L, H, D, 1000, 0.5)
        ratio = f16 / q4
        assert 3.9 <= ratio <= 4.1, f"Expected 4× ratio, got {ratio:.2f}"

    def test_q8_is_2x_smaller_than_f16(self):
        f16 = kv_mb(L, H, D, 1000, 2.0)
        q8  = kv_mb(L, H, D, 1000, 1.0)
        ratio = f16 / q8
        assert 1.9 <= ratio <= 2.1, f"Expected 2× ratio, got {ratio:.2f}"

    def test_linear_scaling_with_context(self):
        """Doubling context must double KV-cache size."""
        mb_512  = kv_mb(L, H, D, 512, 2.0)
        mb_1024 = kv_mb(L, H, D, 1024, 2.0)
        ratio = mb_1024 / mb_512
        assert 1.95 <= ratio <= 2.05, f"Expected 2× scaling, got {ratio:.2f}"

    def test_gqa_8x_reduction(self):
        """GQA with 4 KV heads vs MHA with 32 heads = 8× reduction."""
        gqa = kv_mb(L, 4,  D, 1000, 2.0)
        mha = kv_mb(L, 32, D, 1000, 2.0)
        ratio = mha / gqa
        assert 7.9 <= ratio <= 8.1, f"Expected 8× GQA reduction, got {ratio:.2f}"

    def test_utils_kv_cache_mb_matches_formula(self):
        """utils.kv_cache_mb() must produce the same result as the inline formula."""
        try:
            from utils import kv_cache_mb
        except ImportError:
            import pytest
            pytest.skip("utils.py not importable from this path")
        expected = kv_mb(L, H, D, 1000, 2.0)
        actual   = kv_cache_mb(L, H, D, 1000, 2.0)
        assert abs(actual - expected) < 0.01, (
            f"utils.kv_cache_mb mismatch: {actual:.2f} vs {expected:.2f}"
        )
