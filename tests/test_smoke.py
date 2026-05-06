"""
tests/test_smoke.py
--------------------
Validates that the required directory structure and root files exist.
These tests run without downloading a model.

If smoke_test.json exists (after running run_smoke_test.py), also
validates its content.

Run with: python -m pytest tests/test_smoke.py -v
"""

from pathlib import Path
import json


def test_core_directories_exist():
    """All required project directories must exist."""
    for folder in ["benchmarks", "analysis", "results", "graphs",
                   "optimization", "report", "docs"]:
        assert Path(folder).exists(), f"Directory '{folder}' is missing"


def test_required_root_files_exist():
    """All required root-level files must exist."""
    for f in ["README.md", "requirements.txt", "Makefile", "LICENSE"]:
        assert Path(f).exists(), f"Root file '{f}' is missing"


def test_smoke_json_valid_if_present():
    """If smoke_test.json exists, it must have status=pass."""
    p = Path("results/smoke_test.json")
    if not p.exists():
        return  # Not run yet — acceptable
    with open(p) as f:
        data = json.load(f)
    assert data.get("status") == "pass", (
        f"smoke_test.json status='{data.get('status')}', expected 'pass'"
    )
    assert data.get("measurement_type") == "measured"
