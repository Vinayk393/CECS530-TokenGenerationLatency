"""
tests/test_smoke.py
=====================
Validates repository structure and no hardcoded absolute paths.
Runs without a model download.

Run: python -m pytest tests/test_smoke.py -v
"""
import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).parent.parent


def test_core_directories_exist():
    """All required project directories must exist."""
    for folder in ["benchmarks", "analysis", "results", "graphs",
                   "optimization", "report", "docs"]:
        assert (REPO_ROOT / folder).exists(), f"Directory '{folder}' is missing"


def test_required_root_files_exist():
    """All required root-level files must exist."""
    for f in ["README.md", "requirements.txt", "Makefile", "LICENSE",
              "REPRODUCIBILITY.md", "CITATION.cff"]:
        assert (REPO_ROOT / f).exists(), f"Root file '{f}' is missing"


def test_generate_graphs_in_analysis_not_root():
    """Graph script must live in analysis/, not at repo root (wrong location)."""
    assert (REPO_ROOT / "analysis" / "generate_research_graphs.py").exists(), (
        "analysis/generate_research_graphs.py missing"
    )
    # It should NOT be at root
    root_version = REPO_ROOT / "generate_research_graphs.py"
    assert not root_version.exists(), (
        "generate_research_graphs.py found at repo root — it belongs in analysis/"
    )


def test_no_hardcoded_mnt_paths():
    """No script may contain hardcoded /mnt/user-data/... paths."""
    scripts = list((REPO_ROOT / "benchmarks").glob("*.py"))
    scripts += list((REPO_ROOT / "analysis").glob("*.py"))
    for script in scripts:
        text = script.read_text()
        assert "/mnt/user-data" not in text, (
            f"{script.name} contains hardcoded /mnt/user-data path. "
            "Use Path(__file__).resolve().parents[N] instead."
        )


def test_benchmark_scripts_exist():
    """All 9 paper benchmark scripts must be present."""
    for i in range(1, 10):
        scripts = list((REPO_ROOT / "benchmarks").glob(f"0{i}_*.py"))
        assert len(scripts) >= 1, f"Benchmark script 0{i}_*.py not found"


def test_utils_and_smoke_test_exist():
    """utils.py and run_smoke_test.py must exist in benchmarks/."""
    assert (REPO_ROOT / "benchmarks" / "utils.py").exists()
    assert (REPO_ROOT / "benchmarks" / "run_smoke_test.py").exists()


def test_no_bench07_backend_hf_in_makefile():
    """Makefile must not use --backend hf for bench-07 (wrong quantization path)."""
    mk = REPO_ROOT / "Makefile"
    if not mk.exists():
        pytest.skip("Makefile not found")
    text = mk.read_text()
    assert "--backend hf" not in text, (
        "Makefile bench-07 target uses '--backend hf' — "
        "should use '--mode modeled' instead"
    )
