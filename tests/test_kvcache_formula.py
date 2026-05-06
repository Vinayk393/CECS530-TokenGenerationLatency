"""
tests/test_kvcache_formula.py
------------------------------
Unit tests for the KV-cache memory sizing formula.

Formula: M_KV = 2 * layers * kv_heads * head_dim * context * bytes_per_element

TinyLlama-1.1B (L=22, H_kv=4, d_head=64) in F16 (2 bytes):
    M_KV(1024 tokens) = 2 * 22 * 4 * 64 * 1024 * 2 = 23,068,672 bytes ≈ 22.0 MB

Previous erroneous value: 2.25 MB per 1000 tokens (off by 10x).

Run with: python -m pytest tests/test_kvcache_formula.py -v
"""


def kv_cache_mb(layers, kv_heads, head_dim, context, bytes_per_element):
    """KV-cache size in MB. Uses 1024**2 as MB divisor (mebibytes)."""
    return 2 * layers * kv_heads * head_dim * context * bytes_per_element / (1024 ** 2)


def test_tinyllama_kvcache_1024_f16():
    """F16 @ 1024 tokens must be in range 21–24 MB."""
    mb = kv_cache_mb(22, 4, 64, 1024, 2)
    assert 21.0 <= mb <= 24.0, f"Expected 21–24 MB, got {mb:.2f} MB"


def test_tinyllama_kvcache_1000_f16_not_old_wrong_value():
    """Value must be > 10 MB — guards against the old erroneous 2.25 MB."""
    mb = kv_cache_mb(22, 4, 64, 1000, 2)
    assert mb > 10.0, f"Got {mb:.2f} MB — old erroneous value was 2.25 MB"


def test_q4_is_4x_smaller_than_f16():
    """Q4_K_M (0.5 bpe) must be exactly 4x smaller than F16 (2.0 bpe)."""
    f16 = kv_cache_mb(22, 4, 64, 1000, 2)
    q4  = kv_cache_mb(22, 4, 64, 1000, 0.5)
    assert abs(f16 / q4 - 4.0) < 0.01, f"Expected 4x ratio, got {f16/q4:.3f}"


def test_linear_scaling_with_context():
    """Doubling context must double KV-cache size."""
    mb_512  = kv_cache_mb(22, 4, 64, 512,  2)
    mb_1024 = kv_cache_mb(22, 4, 64, 1024, 2)
    assert abs(mb_1024 / mb_512 - 2.0) < 0.01, f"Expected 2x scaling, got {mb_1024/mb_512:.3f}"
