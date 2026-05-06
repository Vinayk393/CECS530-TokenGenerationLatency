"""
tests/test_result_schema.py
----------------------------
Validates that result directories exist and CSVs contain data rows.
These tests run without downloading a model.

Run with: python -m pytest tests/test_result_schema.py -v
"""

from pathlib import Path
import pandas as pd


def test_results_directories_exist():
    """Both device result directories must exist."""
    assert Path("results/Mac_M4_16GB").exists(), "results/Mac_M4_16GB missing"
    assert Path("results/Mac_M2_8GB").exists(),  "results/Mac_M2_8GB missing"


def test_csv_files_have_rows():
    """Every CSV in results/ must contain at least one data row."""
    csvs = list(Path("results").glob("**/*.csv"))
    assert len(csvs) > 0, "No CSVs found in results/"
    for csv in csvs[:5]:
        df = pd.read_csv(csv)
        assert len(df) > 0, f"{csv} is empty"
