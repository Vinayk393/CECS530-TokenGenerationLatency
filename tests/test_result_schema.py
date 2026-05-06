"""
tests/test_result_schema.py
----------------------------
Verifies that result CSVs in results/ conform to the expected schema:
  - Each CSV must have a 'measurement_type' or 'data_source' column
  - Values must be from the allowed set
  - Q4/Q8 rows must be labeled 'modeled' or 'measured'

Run with: python -m pytest tests/test_result_schema.py -v
"""

import csv
import json
from pathlib import Path

import pytest

RESULTS_DIR = Path("results")
DEVICE_DIRS = ["Mac_M4_16GB", "Mac_M2_8GB"]

VALID_MEASUREMENT_TYPES = {
    "measured",
    "analytical",
    "estimated",
    "modeled",
    "derived_from_bandwidth_scaling_model",
    "estimated_from_arch_proportions_calibrated_to_observed_ptl",
}


def get_existing_csvs():
    found = []
    for device in DEVICE_DIRS:
        p = RESULTS_DIR / device
        if p.exists():
            for csv_file in p.glob("*.csv"):
                found.append((device, csv_file))
    return found


def read_headers(path):
    with open(path, newline="") as f:
        return set(csv.DictReader(f).fieldnames or [])


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class TestResultsDirectory:

    def test_results_dir_exists(self):
        assert RESULTS_DIR.exists(), "results/ directory not found"

    def test_results_readme_exists(self):
        assert (RESULTS_DIR / "README.md").exists(), "results/README.md missing"

    def test_smoke_json_valid_if_exists(self):
        p = RESULTS_DIR / "smoke_test.json"
        if not p.exists():
            pytest.skip("smoke_test.json not found — run run_smoke_test.py first")
        with open(p) as f:
            data = json.load(f)
        assert data.get("status") == "pass"


@pytest.mark.parametrize("device,csv_path", get_existing_csvs())
class TestCSVSchema:

    def test_not_empty(self, device, csv_path):
        assert len(read_rows(csv_path)) > 0, f"{csv_path} is empty"

    def test_has_evidence_column(self, device, csv_path):
        headers = read_headers(csv_path)
        assert headers & {"measurement_type", "data_source"}, (
            f"{csv_path.name} missing 'measurement_type' or 'data_source'. "
            f"Found: {sorted(headers)}"
        )

    def test_evidence_values_valid(self, device, csv_path):
        for row in read_rows(csv_path):
            val = row.get("measurement_type") or row.get("data_source", "")
            assert val in VALID_MEASUREMENT_TYPES, (
                f"{csv_path.name}: invalid value '{val}'"
            )


class TestQuantizationLabels:

    def test_q4_q8_labeled_modeled_or_measured(self):
        found = False
        for device in DEVICE_DIRS:
            p = RESULTS_DIR / device / "07_quantization_speedup.csv"
            if not p.exists():
                continue
            found = True
            for row in read_rows(p):
                prec  = row.get("precision", "")
                mtype = row.get("measurement_type", "")
                if prec in ("Q4_K_M", "Q8_0"):
                    assert mtype in ("modeled", "measured"), (
                        f"Q4/Q8 row has measurement_type='{mtype}'. "
                        "Expected 'modeled' (default) or 'measured' (llamacpp path)."
                    )
        if not found:
            pytest.skip("07_quantization_speedup.csv not found — run bench-07-modeled first")
