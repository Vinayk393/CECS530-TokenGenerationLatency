"""
tests/test_smoke.py
--------------------
Validates results/smoke_test.json if it exists.
Does NOT run the smoke test itself (requires model download).

To produce the output file first:
    python benchmarks/run_smoke_test.py

Run with: python -m pytest tests/test_smoke.py -v
"""

import json
from pathlib import Path

import pytest

SMOKE_OUTPUT = Path("results") / "smoke_test.json"


def load():
    if not SMOKE_OUTPUT.exists():
        pytest.skip("smoke_test.json not found — run 'python benchmarks/run_smoke_test.py' first")
    with open(SMOKE_OUTPUT) as f:
        return json.load(f)


class TestSmokeOutput:

    def test_file_exists(self):
        if not SMOKE_OUTPUT.exists():
            pytest.skip("smoke_test.json not found")
        assert SMOKE_OUTPUT.exists()

    def test_is_valid_json(self):
        assert isinstance(load(), dict)

    def test_status_is_pass(self):
        assert load().get("status") == "pass", (
            f"Expected status='pass', got '{load().get('status')}'"
        )

    def test_required_fields_present(self):
        data = load()
        required = {
            "status", "model", "device", "torch_version",
            "prompt_tokens", "n_gen_tokens", "ttft_ms",
            "mean_ptl_ms", "throughput_tok_s", "measurement_type",
        }
        missing = required - set(data.keys())
        assert not missing, f"smoke_test.json missing fields: {missing}"

    def test_measurement_type_is_measured(self):
        assert load().get("measurement_type") == "measured"

    def test_device_is_valid(self):
        assert load().get("device") in ("mps", "cuda", "cpu")

    def test_ttft_positive(self):
        assert load()["ttft_ms"] > 0

    def test_ptl_positive(self):
        assert load()["mean_ptl_ms"] > 0

    def test_ptl_reasonable_range(self):
        ptl = load()["mean_ptl_ms"]
        assert 1.0 <= ptl <= 5000.0, f"PTL {ptl:.1f} ms out of expected range"

    def test_ptl_samples_count_matches_n_gen_tokens(self):
        data = load()
        assert len(data.get("ptl_samples_ms", [])) == data.get("n_gen_tokens", -1)
