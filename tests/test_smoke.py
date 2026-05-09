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
        "01_ttft_vs_prompt_length.py",
        "02_per_token_latency_vs_context.py",
        "03_e2e_latency_vs_output_length.py",
        "04_throughput_vs_prompt_length.py",
        "05_inter_token_latency_timeline.py",
        "06_cold_vs_warm_run.py",
        "07_quantization_speedup.py",
        "08_kvcache_size_vs_context.py",
        "09_latency_decomposition.py",
        "run_smoke_test.py",
        "utils.py",
    ]
    benchmarks_dir = ROOT / "benchmarks"
    for script_name in required_scripts:
        assert (benchmarks_dir / script_name).exists(), (
            f"Missing benchmark script: benchmarks/{script_name}"
        )


def test_required_graph_files_exist():
    required_graphs = [
        "01_ttft_vs_prompt_length.png",
        "02_ptl_vs_context_length.png",
        "03_inter_token_timeline.png",
        "04_latency_decomposition.png",
        "05_quantization_speedup.png",
        "06_cold_vs_warm.png",
        "07_latency_variance_distribution.png",
        "08_model_scaling.png",
        "09_cross_platform_summary.png",
    ]
    graphs_dir = ROOT / "graphs"
    for graph_name in required_graphs:
        assert (graphs_dir / graph_name).exists(), (
            f"Missing graph file: graphs/{graph_name}"
        )


def test_analysis_script_exists():
    assert (ROOT / "analysis" / "generate_research_graphs.py").exists(), (
        "Missing analysis/generate_research_graphs.py"
    )


def test_reproducibility_docs_exist():
    assert (ROOT / "REPRODUCIBILITY.md").exists(), "Missing REPRODUCIBILITY.md"
    assert (ROOT / "docs" / "reproducibility.md").exists(), (
        "Missing docs/reproducibility.md"
    )


def test_results_and_graph_docs_exist():
    assert (ROOT / "results" / "README.md").exists(), "Missing results/README.md"
    assert (ROOT / "graphs" / "README.md").exists(), "Missing graphs/README.md"


def test_results_device_dirs_exist():
    assert (ROOT / "results" / "Mac_M4_16GB").is_dir(), (
        "Missing results/Mac_M4_16GB/"
    )
    assert (ROOT / "results" / "Mac_M2_8GB").is_dir(), (
        "Missing results/Mac_M2_8GB/"
    )


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


def test_no_absolute_paths_in_benchmarks():
    """All benchmark scripts must use relative paths only."""
    benchmarks_dir = ROOT / "benchmarks"
    for py_file in sorted(benchmarks_dir.glob("*.py")):
        content = py_file.read_text(encoding="utf-8")
        assert "/Users/" not in content, (
            f"{py_file.name}: contains absolute /Users/ path"
        )
        assert "/home/" not in content, (
            f"{py_file.name}: contains absolute /home/ path"
        )


def test_requirements_has_core_dependencies():
    reqs = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for dep in ["torch", "transformers", "numpy", "pandas", "matplotlib"]:
        assert dep in reqs, f"requirements.txt missing: {dep}"