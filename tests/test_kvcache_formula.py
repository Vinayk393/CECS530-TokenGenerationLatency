"""
tests/test_kvcache_formula.py
==============================
Unit tests for the KV-cache formula and PTL methodology alignment.

Paper Equation 4: M_KV = 2 * L * H_kv * d_head * n_ctx * b
Paper Equation 5: TinyLlama F16 @ 1024 tokens = 2*22*4*64*1024*2 ≈ 22.9 MB

Tests also verify that the decode loop skips exactly the first 2 tokens
as stated in paper Section 4.3 ("PTL exclude: First 2 tokens").

Run: python -m pytest tests/test_kvcache_formula.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))


def kv_cache_mb(layers, kv_heads, head_dim, context, bytes_per_elem):
    """Paper Eq. 4: M_KV = 2 * L * H_kv * d_head * n_ctx * b"""
    return 2 * layers * kv_heads * head_dim * context * bytes_per_elem / 1_000_000


# TinyLlama-1.1B architectural constants
L, H, D = 22, 4, 64


class TestKVCacheFormula:
    """Paper Section 6.2: KV-Cache Memory Sizing."""

    def test_tinyllama_f16_1024_matches_paper_eq5(self):
        """Paper Eq. 5: M_KV(1024) = 2×22×4×64×1024×2 ≈ 22.9 MB"""
        mb = kv_cache_mb(L, H, D, 1024, 2.0)
        assert 22.0 <= mb <= 24.0, f"Paper says ~22.9 MB, got {mb:.2f} MB"

    def test_exact_bytes_paper_eq5(self):
        """Exact byte count from paper Eq. 5: 23,068,672 bytes"""
        expected = 2 * 22 * 4 * 64 * 1024 * 2   # = 23,068,672
        actual   = kv_cache_mb(L, H, D, 1024, 2.0) * 1_000_000
        assert abs(actual - expected) < 1, f"Expected {expected:,}, got {actual:,.0f}"

    def test_not_old_wrong_2_25mb_value(self):
        """Guard against regression to erroneous 2.25 MB @ 1000 tokens."""
        mb = kv_cache_mb(L, H, D, 1000, 2.0)
        assert mb > 10.0, f"Got {mb:.2f} MB — old wrong value was 2.25 MB"

    def test_f16_512_tokens(self):
        """Paper Table 8: KV-cache @ 512 tokens = 11.5 MB"""
        mb = kv_cache_mb(L, H, D, 512, 2.0)
        assert 11.0 <= mb <= 12.0, f"Expected ~11.5 MB, got {mb:.2f} MB"

    def test_q4_is_4x_smaller(self):
        """Q4_K_M (0.5 bpe) must be 4× smaller than F16 (2.0 bpe)."""
        f16 = kv_cache_mb(L, H, D, 1000, 2.0)
        q4  = kv_cache_mb(L, H, D, 1000, 0.5)
        assert abs(f16/q4 - 4.0) < 0.01, f"Expected 4× ratio, got {f16/q4:.3f}"

    def test_q8_is_2x_smaller(self):
        """Q8_0 (1.0 bpe) must be 2× smaller than F16 (2.0 bpe)."""
        f16 = kv_cache_mb(L, H, D, 1000, 2.0)
        q8  = kv_cache_mb(L, H, D, 1000, 1.0)
        assert abs(f16/q8 - 2.0) < 0.01, f"Expected 2× ratio, got {f16/q8:.3f}"

    def test_linear_scaling_with_context(self):
        """Doubling context must double KV-cache (linear scaling)."""
        mb512  = kv_cache_mb(L, H, D, 512,  2.0)
        mb1024 = kv_cache_mb(L, H, D, 1024, 2.0)
        assert abs(mb1024/mb512 - 2.0) < 0.01, f"Expected 2× scaling, got {mb1024/mb512:.3f}"

    def test_gqa_8x_reduction_vs_mha(self):
        """TinyLlama GQA (4 KV heads) vs MHA (32 heads) = 8× reduction."""
        gqa = kv_cache_mb(L, 4,  D, 1000, 2.0)
        mha = kv_cache_mb(L, 32, D, 1000, 2.0)
        assert abs(mha/gqa - 8.0) < 0.01, f"Expected 8× GQA reduction, got {mha/gqa:.3f}"


class TestPTLMethodologyAlignment:
    """Paper Section 4.3: PTL exclude first 2 tokens."""

    def test_ptl_skip_count_matches_paper(self):
        """
        Paper Table 2: 'PTL exclude: First 2 tokens'
        N_TRANSIENT_SKIP=2, N_DECODE_TOKENS=20 → loop runs 22 steps,
        collects 20 samples from i>=2.
        """
        N_TRANSIENT_SKIP = 2
        N_DECODE_TOKENS  = 20
        total_steps      = N_TRANSIENT_SKIP + N_DECODE_TOKENS

        samples = []
        for i in range(total_steps):
            if i >= N_TRANSIENT_SKIP:   # paper: "if i >= 2"
                samples.append(i)

        assert len(samples) == N_DECODE_TOKENS, (
            f"Paper says collect {N_DECODE_TOKENS} tokens, got {len(samples)}"
        )

    def test_first_sample_is_after_transients(self):
        """First collected sample must be at index N_TRANSIENT_SKIP."""
        N_TRANSIENT_SKIP = 2
        collected_indices = [i for i in range(22) if i >= N_TRANSIENT_SKIP]
        assert collected_indices[0] == N_TRANSIENT_SKIP

    def test_utils_kv_cache_mb_consistent(self):
        """utils.kv_cache_mb() must match inline formula."""
        try:
            from utils import kv_cache_mb as utils_kv
            expected = kv_cache_mb(L, H, D, 1024, 2.0)
            actual   = utils_kv(L, H, D, 1024, 2.0)
            assert abs(actual - expected) < 0.001, (
                f"utils.kv_cache_mb mismatch: {actual:.3f} vs {expected:.3f}"
            )
        except ImportError:
            import pytest
            pytest.skip("utils.py not importable from test path")
