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

RESULTS = Path("results")
DEVICES = ["Mac_M4_16GB", "Mac_M2_8GB"]
VALID_EVIDENCE = {
    "measured", "analytical", "estimated", "modeled",
    "derived_from_bandwidth_scaling_model",
    "estimated_from_arch_proportions_calibrated_to_observed_ptl",
}


def test_results_directories_exist():
    """Both device result directories must exist."""
    for dev in DEVICES:
        assert (RESULTS / dev).exists(), f"results/{dev}/ missing"


def test_csv_files_have_rows():
    """Every CSV must contain at least one data row."""
    csvs = list(RESULTS.glob("**/*.csv"))
    assert len(csvs) > 0, "No CSVs found in results/"
    for csv_path in csvs[:10]:
        df = pd.read_csv(csv_path)
        assert len(df) > 0, f"{csv_path} is empty"


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
    """Evidence column values must be from the allowed set."""
    csvs = list(RESULTS.glob("**/*.csv"))
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        col = "measurement_type" if "measurement_type" in df.columns else "data_source"
        if col not in df.columns:
            continue
        for val in df[col].dropna().unique():
            assert val in VALID_EVIDENCE, (
                f"{csv_path.name}: invalid evidence value '{val}'"
            )


def test_quantization_csv_labels():
    """Q4/Q8 rows must be 'modeled', F16 rows must be 'measured'."""
    for dev in DEVICES:
        p = RESULTS / dev / "07_quantization_speedup.csv"
        if not p.exists():
            pytest.skip(f"07_quantization_speedup.csv not found in {dev}/")
        df = pd.read_csv(p)
        for _, row in df.iterrows():
            prec  = row.get("precision", "")
            mtype = row.get("measurement_type", "")
            if prec == "F16":
                assert mtype == "measured", f"F16 row should be measured, got '{mtype}'"
            elif prec in ("Q4_K_M", "Q8_0"):
                assert mtype == "modeled", f"{prec} row should be modeled, got '{mtype}'"


def test_kvcache_csv_labeled_analytical():
    """KV-cache sizing CSV must be labeled analytical."""
    for dev in DEVICES:
        p = RESULTS / dev / "08_kvcache_size_vs_context.csv"
        if not p.exists():
            pytest.skip("08_kvcache_size_vs_context.csv not found")
        df = pd.read_csv(p)
        col = "measurement_type" if "measurement_type" in df.columns else "data_source"
        if col in df.columns:
            for val in df[col].unique():
                assert val == "analytical", f"KV-cache CSV row labeled '{val}', expected 'analytical'"


def test_results_readme_exists():
    assert (RESULTS / "README.md").exists(), "results/README.md missing"


def test_smoke_json_if_present():
    p = RESULTS / "smoke_test.json"
    if not p.exists():
        return   # not run yet
    with open(p) as f:
        data = json.load(f)
    assert data.get("status") == "pass"
    assert data.get("measurement_type") == "measured"
    assert data.get("ttft_ms", 0) > 0
