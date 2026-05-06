from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_root_files_exist():
    required_files = [
        "README.md",
        "requirements.txt",
        "Makefile",
        "LICENSE",
        "CITATION.cff",
        "REPRODUCIBILITY.md",
    ]

    for file_name in required_files:
        assert (ROOT / file_name).exists(), f"Missing required root file: {file_name}"


def test_required_directories_exist():
    required_dirs = [
        "analysis",
        "benchmarks",
        "docs",
        "graphs",
        "optimization",
        "report",
        "results",
        "tests",
    ]

    for dir_name in required_dirs:
        assert (ROOT / dir_name).is_dir(), f"Missing required directory: {dir_name}"


def test_required_benchmark_scripts_exist():
    required_scripts = [
        "01_ttft_vs_prompt.py",
        "02_ptl_vs_context.py",
        "03_memory_bandwidth_estimate.py",
        "04_throughput.py",
        "05_inter_token_timeline.py",
        "06_cold_warm_run.py",
        "07_quantization_speedup.py",
        "08_tail_latency_distribution.py",
        "09_latency_decomposition.py",
        "run_smoke_test.py",
        "utils.py",
    ]

    benchmarks_dir = ROOT / "benchmarks"

    for script_name in required_scripts:
        assert (benchmarks_dir / script_name).exists(), (
            f"Missing benchmark script: benchmarks/{script_name}"
        )


def test_reproducibility_docs_exist():
    assert (ROOT / "REPRODUCIBILITY.md").exists()
    assert (ROOT / "docs" / "reproducibility.md").exists()


def test_results_and_graph_docs_exist():
    assert (ROOT / "results" / "README.md").exists()
    assert (ROOT / "graphs" / "README.md").exists()


def test_makefile_contains_verification_targets():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    required_targets = [
        "smoke:",
        "verify:",
        "bench-07-modeled:",
        "bench-07-llamacpp:",
        "graphs:",
        "clean:",
        "help:",
    ]

    for target in required_targets:
        assert target in makefile, f"Missing Makefile target: {target}"


def test_readme_has_evidence_labels():
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()

    required_terms = [
        "evidence labels",
        "measured",
        "estimated",
        "modeled",
        "q4_k_m",
        "kv-cache",
    ]

    for term in required_terms:
        assert term in readme, f"README missing expected term: {term}"
