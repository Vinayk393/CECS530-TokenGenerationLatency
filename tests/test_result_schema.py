"""
tests/test_result_schema.py
----------------------------
Unit tests for result CSV schema correctness.

Each result CSV should include either a `measurement_type` or `data_source`
column to clearly label whether the value is measured, estimated, modeled,
or analytical. This test suite checks CSVs that exist in results/ and
verifies they conform to the expected schema.

Run with:
    python -m pytest tests/test_result_schema.py -v
"""

import csv
import json
from pathlib import Path

import pytest

RESULTS_DIR = Path("results")
DEVICE_DIRS = ["Mac_M4_16GB", "Mac_M2_8GB"]

# Column that must appear in every CSV
REQUIRED_EVIDENCE_COLUMN = {"measurement_type", "data_source"}

# Allowed values for the evidence column
VALID_MEASUREMENT_TYPES = {
    "measured",
    "analytical",
    "estimated",
    "modeled",
    "derived_from_bandwidth_scaling_model",
    "estimated_from_arch_proportions_calibrated_to_observed_ptl",
}

# Required columns per script (subset — at minimum these must be present)
SCHEMA_BY_FILE = {
    "01_ttft_vs_prompt_length.csv": {"prompt_length", "ttft_ms"},
    "02_per_token_latency_vs_context.csv": {"context_length", "ptl_ms"},
    "07_quantization_speedup.csv": {"precision", "prompt_length", "ptl_ms"},
    "08_kvcache_size_vs_context.csv": {"context_tokens", "kv_cache_mb", "precision"},
}


def get_existing_csvs():
    """Return list of (device_dir, csv_path) pairs for existing CSVs."""
    found = []
    for device in DEVICE_DIRS:
        device_path = RESULTS_DIR / device
        if not device_path.exists():
            continue
        for csv_file in device_path.glob("*.csv"):
            found.append((device, csv_file))
    return found


def read_csv_headers(filepath: Path) -> set[str]:
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        return set(reader.fieldnames or [])


def read_csv_rows(filepath: Path) -> list[dict]:
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestResultsDirectoryStructure:
    def test_results_directory_exists(self):
        """results/ directory must exist."""
        assert RESULTS_DIR.exists(), "results/ directory not found"

    def test_results_readme_exists(self):
        """results/README.md should exist."""
        readme = RESULTS_DIR / "README.md"
        assert readme.exists(), "results/README.md is missing"

    def test_smoke_test_json_valid_if_exists(self):
        """If smoke_test.json exists, it must be valid JSON with a 'status' key."""
        smoke = RESULTS_DIR / "smoke_test.json"
        if not smoke.exists():
            pytest.skip("smoke_test.json not found — run run_smoke_test.py first")
        with open(smoke) as f:
            data = json.load(f)
        assert "status" in data, "smoke_test.json missing 'status' key"
        assert data["status"] == "pass", f"Smoke test status is '{data['status']}'"


class TestCSVSchema:
    """Tests applied to every CSV found in results/."""

    @pytest.mark.parametrize("device,csv_path", get_existing_csvs())
    def test_csv_not_empty(self, device, csv_path):
        """Each result CSV must have at least one data row."""
        rows = read_csv_rows(csv_path)
        assert len(rows) > 0, f"{csv_path} is empty"

    @pytest.mark.parametrize("device,csv_path", get_existing_csvs())
    def test_csv_has_evidence_column(self, device, csv_path):
        """Each CSV must have either 'measurement_type' or 'data_source'."""
        headers = read_csv_headers(csv_path)
        has_evidence = bool(headers & REQUIRED_EVIDENCE_COLUMN)
        assert has_evidence, (
            f"{csv_path.name} is missing both 'measurement_type' and 'data_source' columns.\n"
            f"Found columns: {sorted(headers)}"
        )

    @pytest.mark.parametrize("device,csv_path", get_existing_csvs())
    def test_csv_evidence_values_are_valid(self, device, csv_path):
        """Evidence column values must be from the allowed set."""
        rows = read_csv_rows(csv_path)
        for row in rows:
            val = row.get("measurement_type") or row.get("data_source", "")
            assert val in VALID_MEASUREMENT_TYPES, (
                f"{csv_path.name}: invalid measurement_type/data_source value '{val}'\n"
                f"Allowed: {VALID_MEASUREMENT_TYPES}"
            )

    @pytest.mark.parametrize("device,csv_path", get_existing_csvs())
    def test_csv_no_empty_metric_values(self, device, csv_path):
        """Numeric metric columns should not be empty strings."""
        rows = read_csv_rows(csv_path)
        headers = read_csv_headers(csv_path)
        numeric_hints = {h for h in headers if any(
            k in h for k in ["ms", "tok_s", "gb", "mb", "speedup", "ratio"]
        )}
        for i, row in enumerate(rows):
            for col in numeric_hints:
                val = row.get(col, "")
                assert val != "", (
                    f"{csv_path.name} row {i+1}: empty value in column '{col}'"
                )


class TestSpecificCSVSchemas:
    """Schema checks for specific benchmark output files."""

    @pytest.mark.parametrize("filename,required_cols", SCHEMA_BY_FILE.items())
    def test_specific_csv_columns(self, filename, required_cols):
        """Check that specific CSVs have their required columns."""
        found_any = False
        for device in DEVICE_DIRS:
            csv_path = RESULTS_DIR / device / filename
            if not csv_path.exists():
                continue
            found_any = True
            headers = read_csv_headers(csv_path)
            missing = required_cols - headers
            assert not missing, (
                f"{csv_path}: missing required columns {missing}\n"
                f"Found: {sorted(headers)}"
            )
        if not found_any:
            pytest.skip(f"{filename} not found in any device directory — run benchmarks first")


class TestQuantizationEvidenceLabels:
    """Q4/Q8 results must be labeled 'modeled' unless llamacpp path was used."""

    def test_quantization_csv_labels_q4q8_correctly(self):
        found_any = False
        for device in DEVICE_DIRS:
            csv_path = RESULTS_DIR / device / "07_quantization_speedup.csv"
            if not csv_path.exists():
                continue
            found_any = True
            rows = read_csv_rows(csv_path)
            for row in rows:
                prec = row.get("precision", "")
                mtype = row.get("measurement_type", "")
                if prec in ("Q4_K_M", "Q8_0"):
                    assert mtype in ("modeled", "measured"), (
                        f"Q4/Q8 row has unexpected measurement_type='{mtype}'. "
                        "Should be 'modeled' (default) or 'measured' (llamacpp path)."
                    )
        if not found_any:
            pytest.skip("07_quantization_speedup.csv not found — run bench-07-modeled first")
