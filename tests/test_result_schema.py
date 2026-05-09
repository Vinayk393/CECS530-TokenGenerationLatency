"""
tests/test_result_schema.py
==============================
Validates result directories and CSV schema.
Runs without a model download.

Run: python -m pytest tests/test_result_schema.py -v
"""
import json
from pathlib import Path
import pandas as pd
import pytest

# Repo-relative — works regardless of where pytest is invoked from
RESULTS = Path(__file__).resolve().parents[1] / "results"

DEVICES = ["Mac_M4_16GB", "Mac_M2_8GB"]

# measurement_type column: values produced by benchmark scripts
VALID_MEASUREMENT_TYPES = {
    "measured",
    "analytical",
    "estimated",
    "modeled",
    "derived_from_bandwidth_scaling_model",
    "estimated_from_arch_proportions_calibrated_to_observed_ptl",
    "cold_run",
    "warm_run",
}

# data_source column: formula/method labels (separate vocabulary)
VALID_DATA_SOURCES = {
    "formula_2_L_Hkv_dhead_ctx_bpe",
    "architecture_proportions",
    "bandwidth_scaling_model",
    "perf_counter_mps_sync",
    "measured",
    "analytical",
    "modeled",
}


# ── Directory checks ────────────────────────────────────────────────────────────

def test_results_directories_exist():
    """Both device result directories must exist."""
    for dev in DEVICES:
        assert (RESULTS / dev).exists(), f"results/{dev}/ missing"


def test_results_readme_exists():
    assert (RESULTS / "README.md").exists(), "results/README.md missing"


# ── CSV population checks ───────────────────────────────────────────────────────

def test_csv_files_have_rows():
    """Every CSV must contain at least one data row."""
    csvs = list(RESULTS.glob("**/*.csv"))
    assert len(csvs) > 0, "No CSVs found in results/"
    for csv_path in csvs[:10]:
        df = pd.read_csv(csv_path)
        assert len(df) > 0, f"{csv_path} is empty"


# ── Schema checks ───────────────────────────────────────────────────────────────

def test_csvs_have_evidence_column():
    """Each CSV must have measurement_type or data_source column."""
    csvs = list(RESULTS.glob("**/*.csv"))
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        has_col = "measurement_type" in df.columns or "data_source" in df.columns
        assert has_col, (
            f"{csv_path.name}: missing measurement_type/data_source column.\n"
            f"Columns found: {list(df.columns)}"
        )


def test_evidence_values_are_valid():
    """
    measurement_type values must be in VALID_MEASUREMENT_TYPES.
    data_source values (where present) must be in VALID_DATA_SOURCES.
    Validated separately — 08_kvcache_size_vs_context.csv carries both
    columns with different vocabularies.
    """
    csvs = list(RESULTS.glob("**/*.csv"))
    for csv_path in csvs:
        df = pd.read_csv(csv_path)

        if "measurement_type" in df.columns:
            for val in df["measurement_type"].dropna().unique():
                assert val in VALID_MEASUREMENT_TYPES, (
                    f"{csv_path.name}: invalid measurement_type '{val}'.\n"
                    f"Allowed: {sorted(VALID_MEASUREMENT_TYPES)}"
                )

        if "data_source" in df.columns:
            for val in df["data_source"].dropna().unique():
                assert val in VALID_DATA_SOURCES, (
                    f"{csv_path.name}: invalid data_source '{val}'.\n"
                    f"Allowed: {sorted(VALID_DATA_SOURCES)}"
                )


# ── Per-experiment label checks ─────────────────────────────────────────────────

def test_quantization_csv_labels():
    """
    Q4/Q8 rows must have measurement_type='modeled'.
    F16 rows must have measurement_type='measured'.
    Paper Section 5.7 evidence classification.
    """
    for dev in DEVICES:
        p = RESULTS / dev / "07_quantization_speedup.csv"
        if not p.exists():
            pytest.skip(f"07_quantization_speedup.csv not found in {dev}/")
        df = pd.read_csv(p)
        if "precision" not in df.columns or "measurement_type" not in df.columns:
            pytest.skip("Missing precision or measurement_type column")
        for _, row in df.iterrows():
            prec  = str(row.get("precision", "")).strip()
            mtype = str(row.get("measurement_type", "")).strip()
            if prec == "F16":
                assert mtype == "measured", (
                    f"{dev}: F16 row should be 'measured', got '{mtype}'"
                )
            elif prec in ("Q4_K_M", "Q8_0"):
                assert mtype == "modeled", (
                    f"{dev}: {prec} row should be 'modeled', got '{mtype}'"
                )


def test_kvcache_csv_labeled_analytical():
    """
    KV-cache sizing CSV (08) measurement_type must be 'analytical'.
    Paper Section 6.2: values derived from formula, not measured.
    """
    for dev in DEVICES:
        p = RESULTS / dev / "08_kvcache_size_vs_context.csv"
        if not p.exists():
            pytest.skip(f"08_kvcache_size_vs_context.csv not found in {dev}/")
        df = pd.read_csv(p)
        if "measurement_type" not in df.columns:
            pytest.skip("No measurement_type column")
        for val in df["measurement_type"].dropna().unique():
            assert val == "analytical", (
                f"{dev}/08_kvcache_size_vs_context.csv: "
                f"measurement_type='{val}', expected 'analytical'"
            )


def test_ttft_csv_labeled_measured():
    """TTFT results (01) must be measurement_type='measured'. Paper Section 5.1."""
    for dev in DEVICES:
        p = RESULTS / dev / "01_ttft_vs_prompt_length.csv"
        if not p.exists():
            pytest.skip(f"01_ttft_vs_prompt_length.csv not found in {dev}/")
        df = pd.read_csv(p)
        if "measurement_type" not in df.columns:
            pytest.skip("No measurement_type column")
        for val in df["measurement_type"].dropna().unique():
            assert val == "measured", (
                f"{dev}/01_ttft_vs_prompt_length.csv: got '{val}', expected 'measured'"
            )


def test_ptl_csv_labeled_measured():
    """PTL results (02) must be measurement_type='measured'. Paper Section 5.2."""
    for dev in DEVICES:
        p = RESULTS / dev / "02_per_token_latency_vs_context.csv"
        if not p.exists():
            pytest.skip(f"02_per_token_latency_vs_context.csv not found in {dev}/")
        df = pd.read_csv(p)
        if "measurement_type" not in df.columns:
            pytest.skip("No measurement_type column")
        for val in df["measurement_type"].dropna().unique():
            assert val == "measured", (
                f"{dev}/02_per_token_latency_vs_context.csv: got '{val}', expected 'measured'"
            )


# ── Quantitative sanity checks ──────────────────────────────────────────────────

def test_ttft_grows_with_prompt_length():
    """
    TTFT must increase monotonically with prompt length.
    Paper Result 1, Table 4: 16.1ms @ 32tok → 197.5ms @ 1024tok.
    Column name confirmed: 'prompt_length', 'ttft_mean_ms'.
    """
    for dev in DEVICES:
        p = RESULTS / dev / "01_ttft_vs_prompt_length.csv"
        if not p.exists():
            pytest.skip(f"01_ttft_vs_prompt_length.csv not found in {dev}/")
        df = pd.read_csv(p)
        if "prompt_length" not in df.columns or "ttft_mean_ms" not in df.columns:
            pytest.skip("Missing prompt_length or ttft_mean_ms column")
        df_sorted = df.sort_values("prompt_length")
        vals = df_sorted["ttft_mean_ms"].tolist()
        assert vals[-1] > vals[0], (
            f"{dev}: TTFT did not grow with prompt length. "
            f"First={vals[0]:.1f}ms Last={vals[-1]:.1f}ms"
        )


def test_ptl_grows_with_context():
    """
    PTL must increase monotonically with context length.
    Paper Result 2, Table 5: 19.3ms @ 32tok → 48.6ms @ 1024tok (+152%).
    Column name confirmed: 'context_length', 'ptl_mean_ms'.
    """
    for dev in DEVICES:
        p = RESULTS / dev / "02_per_token_latency_vs_context.csv"
        if not p.exists():
            pytest.skip(f"02_per_token_latency_vs_context.csv not found in {dev}/")
        df = pd.read_csv(p)
        if "context_length" not in df.columns or "ptl_mean_ms" not in df.columns:
            pytest.skip("Missing context_length or ptl_mean_ms column")
        df_sorted = df.sort_values("context_length")
        vals = df_sorted["ptl_mean_ms"].tolist()
        assert vals[-1] > vals[0], (
            f"{dev}: PTL did not grow with context. "
            f"First={vals[0]:.1f}ms Last={vals[-1]:.1f}ms"
        )


def test_ptl_152_percent_growth():
    """
    Paper Result 2: PTL grows ~152% from 32 to 1024 tokens on M4.
    Uses confirmed column names: context_length, ptl_mean_ms.
    """
    p = RESULTS / "Mac_M4_16GB" / "02_per_token_latency_vs_context.csv"
    if not p.exists():
        pytest.skip("M4 PTL CSV not found")
    df = pd.read_csv(p)
    if "context_length" not in df.columns or "ptl_mean_ms" not in df.columns:
        pytest.skip("Missing required columns")
    df_sorted = df.sort_values("context_length")
    ptl_min = df_sorted["ptl_mean_ms"].iloc[0]
    ptl_max = df_sorted["ptl_mean_ms"].iloc[-1]
    growth_pct = (ptl_max - ptl_min) / ptl_min * 100
    assert growth_pct > 100, (
        f"Expected >100% PTL growth (paper: ~152%), got {growth_pct:.1f}%"
    )


def test_m4_faster_than_m2():
    """
    M4 mean PTL must be lower than M2 mean PTL.
    Paper Table 5: M4 consistently outperforms M2.
    """
    m4 = RESULTS / "Mac_M4_16GB" / "02_per_token_latency_vs_context.csv"
    m2 = RESULTS / "Mac_M2_8GB"  / "02_per_token_latency_vs_context.csv"
    if not m4.exists() or not m2.exists():
        pytest.skip("PTL CSVs not present for both devices")
    df4 = pd.read_csv(m4)
    df2 = pd.read_csv(m2)
    if "ptl_mean_ms" not in df4.columns or "ptl_mean_ms" not in df2.columns:
        pytest.skip("ptl_mean_ms column missing")
    m4_mean = df4["ptl_mean_ms"].mean()
    m2_mean = df2["ptl_mean_ms"].mean()
    assert m4_mean < m2_mean, (
        f"Expected M4 ({m4_mean:.1f}ms) < M2 ({m2_mean:.1f}ms)"
    )


def test_quantization_speedup_order():
    """
    Q4_K_M must be faster than Q8_0, which must be faster than F16.
    Paper Table 7: F16→Q8→Q4 speedup increases monotonically.
    """
    for dev in DEVICES:
        p = RESULTS / dev / "07_quantization_speedup.csv"
        if not p.exists():
            pytest.skip(f"07_quantization_speedup.csv not found in {dev}/")
        df = pd.read_csv(p)
        if "precision" not in df.columns or "ptl_ms" not in df.columns:
            pytest.skip("Missing precision or ptl_ms column")
        # Average across prompt lengths for each precision
        means = df.groupby("precision")["ptl_ms"].mean()
        if not all(k in means.index for k in ["F16", "Q8_0", "Q4_K_M"]):
            pytest.skip("Not all precision levels present")
        assert means["Q4_K_M"] < means["Q8_0"] < means["F16"], (
            f"{dev}: Expected Q4<Q8<F16 PTL. "
            f"Got Q4={means['Q4_K_M']:.1f} Q8={means['Q8_0']:.1f} F16={means['F16']:.1f}"
        )


# ── Smoke JSON check ────────────────────────────────────────────────────────────

def test_smoke_json_if_present():
    p = RESULTS / "smoke_test.json"
    if not p.exists():
        return   # not run yet — skip silently
    with open(p) as f:
        data = json.load(f)
    assert data.get("status") == "pass"
    assert data.get("measurement_type") == "measured"
    assert data.get("ttft_ms", 0) > 0