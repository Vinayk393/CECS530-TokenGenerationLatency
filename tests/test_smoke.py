"""
tests/test_smoke.py
--------------------
Unit tests for smoke test output.

These tests verify that if run_smoke_test.py has been executed,
its output JSON is valid and contains expected fields.

Run with:
    python -m pytest tests/test_smoke.py -v

Note: These tests do NOT run the smoke test themselves (that requires
model download). They validate the output JSON file if it already exists.
To actually run the smoke test:
    python benchmarks/run_smoke_test.py
"""

import json
from pathlib import Path

import pytest

SMOKE_OUTPUT = Path("results") / "smoke_test.json"


class TestSmokeOutput:
    def _load(self):
        if not SMOKE_OUTPUT.exists():
            pytest.skip(
                "smoke_test.json not found. Run 'python benchmarks/run_smoke_test.py' first."
            )
        with open(SMOKE_OUTPUT) as f:
            return json.load(f)

    def test_smoke_output_exists(self):
        """smoke_test.json must exist after running run_smoke_test.py."""
        if not SMOKE_OUTPUT.exists():
            pytest.skip("smoke_test.json not found — run run_smoke_test.py first")
        assert SMOKE_OUTPUT.exists()

    def test_smoke_output_is_valid_json(self):
        data = self._load()
        assert isinstance(data, dict)

    def test_smoke_status_is_pass(self):
        """Smoke test must have completed successfully."""
        data = self._load()
        assert data.get("status") == "pass", (
            f"Smoke test status is '{data.get('status')}' — expected 'pass'"
        )

    def test_smoke_has_required_fields(self):
        """All required output fields must be present."""
        data = self._load()
        required = {
            "status", "model", "device", "torch_version",
            "prompt_tokens", "n_gen_tokens", "ttft_ms",
            "mean_ptl_ms", "throughput_tok_s", "measurement_type",
        }
        missing = required - set(data.keys())
        assert not missing, f"smoke_test.json missing fields: {missing}"

    def test_smoke_ttft_is_positive(self):
        data = self._load()
        assert data["ttft_ms"] > 0, f"TTFT should be positive, got {data['ttft_ms']}"

    def test_smoke_ptl_is_positive(self):
        data = self._load()
        assert data["mean_ptl_ms"] > 0, f"PTL should be positive, got {data['mean_ptl_ms']}"

    def test_smoke_throughput_is_positive(self):
        data = self._load()
        assert data["throughput_tok_s"] > 0

    def test_smoke_measurement_type_is_measured(self):
        """Smoke test values are directly timed — must be labeled 'measured'."""
        data = self._load()
        assert data.get("measurement_type") == "measured", (
            f"Expected measurement_type='measured', got '{data.get('measurement_type')}'"
        )

    def test_smoke_device_field_set(self):
        """Device field must be 'mps', 'cuda', or 'cpu'."""
        data = self._load()
        assert data.get("device") in ("mps", "cuda", "cpu"), (
            f"Unexpected device: {data.get('device')}"
        )

    def test_smoke_ptl_samples_match_n_gen_tokens(self):
        """Number of PTL samples must equal n_gen_tokens."""
        data = self._load()
        n = data.get("n_gen_tokens", 0)
        samples = data.get("ptl_samples_ms", [])
        assert len(samples) == n, (
            f"Expected {n} PTL samples, got {len(samples)}"
        )

    def test_smoke_ptl_reasonable_range(self):
        """PTL should be in a plausible range (1 ms – 2000 ms)."""
        data = self._load()
        ptl = data["mean_ptl_ms"]
        assert 1.0 <= ptl <= 2000.0, (
            f"PTL {ptl:.1f} ms is outside expected range [1, 2000] ms. "
            "Check device or model configuration."
        )
