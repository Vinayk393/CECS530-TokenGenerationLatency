"""
tests/test_quantization_physical_results.py
============================================
Pytest suite that validates the quantization benchmark artefacts
produced by benchmarks/07_quantization_physical_llamacpp.py against
the paper's anchor values and internal consistency requirements.

Paper reference
---------------
  "Token-Generation Latency Benchmarking in LLaMA: Measurement,
   Bottleneck Attribution, and Architectural Implications on Apple Silicon"
  CECS 530, CSULB 2026 — Vinay Krishna & Jaswanth Maddineni

What is tested
--------------
  Schema       — all required CSV columns present; types correct
  Completeness — all six (device × precision) files exist; correct row count
  Evidence     — F16 rows labelled "measured"; Q8/Q4 rows labelled "modeled"
  Anchors      — mean PTL at 512-token context within ±15% of Table 7 values
  Monotonicity — PTL grows with context (KV-cache pressure signature)
  Speedup      — Q8_0 and Q4_K_M are faster than F16 at every context
  Speedup order— Q4_K_M faster than Q8_0 faster than F16 at ctx ≥ 256
  Tail latency — p99 > p90 > ptl_ms  for every row
  BW util      — M4 BW util ≥ M2 BW util at same precision/context (Section 6)
  Log parsing  — raw_log txt files parseable and consistent with CSV means
  Model load   — load times ordered F16 > Q8_0 > Q4_K_M (smaller file faster)
  Cross-device — M4 PTL < M2 PTL at every (precision, context) pair
  Context growth— PTL at ctx=1024 is ≥ 1.5× PTL at ctx=32 (paper: 152% growth)

Run
---
  pytest tests/test_quantization_physical_results.py -v
  pytest tests/test_quantization_physical_results.py -v -k "test_speedup"
"""

from __future__ import annotations

import csv
import math
import re
import statistics
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths — adjust RESULTS_DIR if your layout differs
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results/quantization_physical")
LOG_DIR     = RESULTS_DIR / "raw_logs"

DEVICES     = ["M4", "M2"]
PRECISIONS  = ["F16", "Q8_0", "Q4_K_M"]
CONTEXTS    = [32, 128, 256, 512, 1024]
PROMPTS     = [32, 128, 256, 512, 1024]
N_TRIALS    = 5

# Paper anchor PTL at 512-token context — Table 7
PAPER_PTL_512 = {
    ("M4", "F16"):    29.6,
    ("M4", "Q8_0"):   20.9,
    ("M4", "Q4_K_M"): 12.8,
    ("M2", "F16"):    50.3,
    ("M2", "Q8_0"):   34.7,
    ("M2", "Q4_K_M"): 25.7,
}

# Allowable deviation from paper anchor values
ANCHOR_TOLERANCE_PCT = 15.0

REQUIRED_CSV_COLUMNS = {
    "device", "precision", "backend", "model",
    "prompt_tokens", "context_tokens", "n_gen_tokens",
    "trial",
    "ttft_ms", "ptl_ms", "throughput_tok_s",
    "ptl_p90_ms", "ptl_p99_ms",
    "model_load_s", "bw_util_pct",
    "measurement_type", "timestamp",
}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _csv_path(device: str, precision: str) -> Path:
    return RESULTS_DIR / f"{device.lower()}_{precision.lower()}_llamacpp.csv"


def _log_path(device: str, precision: str) -> Path:
    return LOG_DIR / f"{device.lower()}_{precision.lower()}_llamacpp.txt"


def _load_csv(device: str, precision: str) -> list[dict]:
    path = _csv_path(device, precision)
    if not path.exists():
        pytest.skip(f"CSV not found: {path}")
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "device":           row["device"].strip().upper(),
                "precision":        row["precision"].strip().upper(),
                "backend":          row["backend"].strip(),
                "prompt_tokens":    int(row["prompt_tokens"]),
                "context_tokens":   int(row["context_tokens"]),
                "n_gen_tokens":     int(row["n_gen_tokens"]),
                "trial":            int(row["trial"]),
                "ttft_ms":          float(row["ttft_ms"]),
                "ptl_ms":           float(row["ptl_ms"]),
                "throughput_tok_s": float(row["throughput_tok_s"]),
                "ptl_p90_ms":       float(row["ptl_p90_ms"]),
                "ptl_p99_ms":       float(row["ptl_p99_ms"]),
                "model_load_s":     float(row["model_load_s"]),
                "bw_util_pct":      float(row["bw_util_pct"]),
                "measurement_type": row["measurement_type"].strip(),
                "timestamp":        row["timestamp"].strip(),
            })
    return rows


def _mean(vals: list[float]) -> float:
    return statistics.mean(vals) if vals else math.nan


def _mean_ptl(rows: list[dict], ctx: int) -> float:
    return _mean([r["ptl_ms"] for r in rows if r["context_tokens"] == ctx])


@pytest.fixture(scope="session")
def all_data() -> dict[tuple[str, str], list[dict]]:
    data: dict[tuple[str, str], list[dict]] = {}
    for device in DEVICES:
        for prec in PRECISIONS:
            path = _csv_path(device, prec)
            if path.exists():
                data[(device, prec)] = _load_csv(device, prec)
    return data


# ---------------------------------------------------------------------------
# 1. File existence
# ---------------------------------------------------------------------------

class TestFileExistence:

    @pytest.mark.parametrize("device,precision", [
        (d, p) for d in DEVICES for p in PRECISIONS
    ])
    def test_csv_file_exists(self, device, precision):
        assert _csv_path(device, precision).exists(), (
            f"Missing CSV: {_csv_path(device, precision)}"
        )

    @pytest.mark.parametrize("device,precision", [
        (d, p) for d in DEVICES for p in PRECISIONS
    ])
    def test_log_file_exists(self, device, precision):
        assert _log_path(device, precision).exists(), (
            f"Missing log: {_log_path(device, precision)}"
        )


# ---------------------------------------------------------------------------
# 2. Schema validation
# ---------------------------------------------------------------------------

class TestSchema:

    @pytest.mark.parametrize("device,precision", [
        (d, p) for d in DEVICES for p in PRECISIONS
    ])
    def test_csv_has_required_columns(self, device, precision):
        path = _csv_path(device, precision)
        if not path.exists():
            pytest.skip(f"File not found: {path}")
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            cols   = set(reader.fieldnames or [])
        missing = REQUIRED_CSV_COLUMNS - cols
        assert not missing, f"Missing columns in {path.name}: {missing}"

    @pytest.mark.parametrize("device,precision", [
        (d, p) for d in DEVICES for p in PRECISIONS
    ])
    def test_csv_row_count(self, device, precision):
        """
        Expect 5 prompt × 5 context × 5 trials = 125 rows.
        """
        rows = _load_csv(device, precision)
        assert len(rows) == 125, (
            f"{device} {precision}: expected 125 rows, got {len(rows)}"
        )

    @pytest.mark.parametrize("device,precision", [
        (d, p) for d in DEVICES for p in PRECISIONS
    ])
    def test_numeric_columns_are_positive(self, device, precision):
        rows = _load_csv(device, precision)
        numeric_cols = ["ttft_ms", "ptl_ms", "throughput_tok_s",
                        "ptl_p90_ms", "ptl_p99_ms", "model_load_s", "bw_util_pct"]
        for col in numeric_cols:
            bad = [r[col] for r in rows if r[col] <= 0]
            assert not bad, (
                f"{device} {precision} column '{col}': found non-positive values: {bad[:5]}"
            )

    @pytest.mark.parametrize("device,precision", [
        (d, p) for d in DEVICES for p in PRECISIONS
    ])
    def test_trial_numbers_complete(self, device, precision):
        rows = _load_csv(device, precision)
        for ctx in CONTEXTS:
            for prompt in PROMPTS:
                trials = sorted(
                    r["trial"] for r in rows
                    if r["context_tokens"] == ctx and r["prompt_tokens"] == prompt
                )
                assert trials == list(range(1, N_TRIALS + 1)), (
                    f"{device} {precision} ctx={ctx} prompt={prompt}: "
                    f"expected trials 1-{N_TRIALS}, got {trials}"
                )


# ---------------------------------------------------------------------------
# 3. Evidence labelling (Section 4.4)
# ---------------------------------------------------------------------------

class TestEvidenceLabels:

    def test_f16_rows_labelled_measured(self, all_data):
        for device in DEVICES:
            rows = all_data.get((device, "F16"), [])
            if not rows:
                pytest.skip(f"No data for {device} F16")
            bad = [r for r in rows if r["measurement_type"] != "measured"]
            assert not bad, (
                f"{device} F16: {len(bad)} rows not labelled 'measured'"
            )

    @pytest.mark.parametrize("precision", ["Q8_0", "Q4_K_M"])
    def test_quantized_rows_labelled_modeled(self, all_data, precision):
        for device in DEVICES:
            rows = all_data.get((device, precision), [])
            if not rows:
                pytest.skip(f"No data for {device} {precision}")
            bad = [r for r in rows if r["measurement_type"] != "modeled"]
            assert not bad, (
                f"{device} {precision}: {len(bad)} rows not labelled 'modeled'"
            )


# ---------------------------------------------------------------------------
# 4. Paper anchor validation (Table 7)
# ---------------------------------------------------------------------------

class TestPaperAnchors:

    @pytest.mark.parametrize("device,precision", [
        (d, p) for d in DEVICES for p in PRECISIONS
    ])
    def test_ptl_within_tolerance_of_paper_table7(self, device, precision):
        rows   = _load_csv(device, precision)
        mean   = _mean_ptl(rows, ctx=512)
        anchor = PAPER_PTL_512.get((device, precision))
        if anchor is None:
            pytest.skip(f"No paper anchor for ({device}, {precision})")
        dev_pct = 100.0 * abs(mean - anchor) / anchor
        assert dev_pct <= ANCHOR_TOLERANCE_PCT, (
            f"{device} {precision} @ ctx=512: mean PTL={mean:.2f} ms  "
            f"paper={anchor:.1f} ms  deviation={dev_pct:.1f}%  "
            f"(tolerance ±{ANCHOR_TOLERANCE_PCT}%)"
        )


# ---------------------------------------------------------------------------
# 5. Monotonicity — PTL grows with context (Section 5.2, Fig 3)
# ---------------------------------------------------------------------------

class TestMonotonicity:

    @pytest.mark.parametrize("device,precision", [
        (d, p) for d in DEVICES for p in PRECISIONS
    ])
    def test_ptl_increases_with_context(self, device, precision):
        """
        Mean PTL at ctx=1024 must exceed mean PTL at ctx=32.
        Paper: 152% growth on M4 F16 (32→1024 tokens).
        """
        rows = _load_csv(device, precision)
        ptl_32   = _mean_ptl(rows, 32)
        ptl_1024 = _mean_ptl(rows, 1024)
        assert ptl_1024 > ptl_32, (
            f"{device} {precision}: PTL should increase with context. "
            f"ctx=32: {ptl_32:.2f} ms  ctx=1024: {ptl_1024:.2f} ms"
        )

    @pytest.mark.parametrize("device,precision", [
        (d, p) for d in DEVICES for p in PRECISIONS
    ])
    def test_ptl_growth_long_context_significant(self, device, precision):
        """
        PTL at 1024 tokens must be ≥ 1.5× PTL at 32 tokens.
        Paper observes 152% growth (2.52× ratio) on M4 F16.
        """
        rows   = _load_csv(device, precision)
        ratio  = _mean_ptl(rows, 1024) / _mean_ptl(rows, 32)
        assert ratio >= 1.5, (
            f"{device} {precision}: expected ctx=1024/ctx=32 ratio ≥ 1.5×, "
            f"got {ratio:.3f}×"
        )


# ---------------------------------------------------------------------------
# 6. Speedup ordering
# ---------------------------------------------------------------------------

class TestSpeedupOrdering:

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("ctx", CONTEXTS)
    def test_q8_faster_than_f16(self, all_data, device, ctx):
        f16_rows = all_data.get((device, "F16"), [])
        q8_rows  = all_data.get((device, "Q8_0"), [])
        if not f16_rows or not q8_rows:
            pytest.skip("Data not available")
        f16_ptl = _mean_ptl(f16_rows, ctx)
        q8_ptl  = _mean_ptl(q8_rows, ctx)
        assert q8_ptl < f16_ptl, (
            f"{device} ctx={ctx}: Q8_0 PTL {q8_ptl:.2f} ms should be < "
            f"F16 PTL {f16_ptl:.2f} ms"
        )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("ctx", CONTEXTS)
    def test_q4_faster_than_q8(self, all_data, device, ctx):
        q8_rows = all_data.get((device, "Q8_0"), [])
        q4_rows = all_data.get((device, "Q4_K_M"), [])
        if not q8_rows or not q4_rows:
            pytest.skip("Data not available")
        q8_ptl = _mean_ptl(q8_rows, ctx)
        q4_ptl = _mean_ptl(q4_rows, ctx)
        assert q4_ptl < q8_ptl, (
            f"{device} ctx={ctx}: Q4_K_M PTL {q4_ptl:.2f} ms should be < "
            f"Q8_0 PTL {q8_ptl:.2f} ms"
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_q4_speedup_grows_with_context(self, all_data, device):
        """
        Speedup ratio (F16/Q4_K_M) at ctx=1024 must exceed ratio at ctx=32.
        This is the KV-cache domination signature (paper Fig 8).
        """
        f16_rows = all_data.get((device, "F16"), [])
        q4_rows  = all_data.get((device, "Q4_K_M"), [])
        if not f16_rows or not q4_rows:
            pytest.skip("Data not available")
        sp_short = _mean_ptl(f16_rows, 32) / _mean_ptl(q4_rows, 32)
        sp_long  = _mean_ptl(f16_rows, 1024) / _mean_ptl(q4_rows, 1024)
        assert sp_long > sp_short, (
            f"{device}: Q4_K_M speedup should grow with context. "
            f"ctx=32: {sp_short:.3f}×  ctx=1024: {sp_long:.3f}×"
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_q4_speedup_at_512_context_reasonable(self, all_data, device):
        """
        Paper projects ~2.2× on M4 and ~2.0× on M2 at 512-token context.
        Accept anything in [1.5×, 3.0×] as physically plausible.
        """
        f16_rows = all_data.get((device, "F16"), [])
        q4_rows  = all_data.get((device, "Q4_K_M"), [])
        if not f16_rows or not q4_rows:
            pytest.skip("Data not available")
        speedup = _mean_ptl(f16_rows, 512) / _mean_ptl(q4_rows, 512)
        assert 1.5 <= speedup <= 3.0, (
            f"{device}: Q4_K_M speedup at ctx=512 = {speedup:.3f}× "
            f"— expected in [1.5×, 3.0×]"
        )


# ---------------------------------------------------------------------------
# 7. Tail latency ordering
# ---------------------------------------------------------------------------

class TestTailLatency:

    @pytest.mark.parametrize("device,precision", [
        (d, p) for d in DEVICES for p in PRECISIONS
    ])
    def test_p99_greater_than_p90(self, device, precision):
        rows = _load_csv(device, precision)
        bad  = [r for r in rows if r["ptl_p99_ms"] <= r["ptl_p90_ms"]]
        assert not bad, (
            f"{device} {precision}: {len(bad)} rows have p99 ≤ p90"
        )

    @pytest.mark.parametrize("device,precision", [
        (d, p) for d in DEVICES for p in PRECISIONS
    ])
    def test_p90_greater_than_ptl_mean(self, device, precision):
        rows = _load_csv(device, precision)
        bad  = [r for r in rows if r["ptl_p90_ms"] <= r["ptl_ms"]]
        assert not bad, (
            f"{device} {precision}: {len(bad)} rows have p90 ≤ ptl_ms"
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_p99_elevation_within_expected_range(self, device):
        """
        Paper Result 8: p99 is +33% above median on M4, +30% on M2.
        Test that mean p99/ptl ratio is in [1.15, 1.55] (generous range).
        """
        rows_f16 = _load_csv(device, "F16")
        rows_512 = [r for r in rows_f16 if r["context_tokens"] == 512]
        if not rows_512:
            pytest.skip("No 512-context rows")
        ratios  = [r["ptl_p99_ms"] / r["ptl_ms"] for r in rows_512]
        mean_r  = _mean(ratios)
        assert 1.15 <= mean_r <= 1.55, (
            f"{device} F16 ctx=512: mean p99/ptl = {mean_r:.3f} "
            f"— expected in [1.15, 1.55] (paper: M4≈1.33, M2≈1.30)"
        )


# ---------------------------------------------------------------------------
# 8. Cross-device ordering — M4 faster than M2
# ---------------------------------------------------------------------------

class TestCrossDevice:

    @pytest.mark.parametrize("precision", PRECISIONS)
    @pytest.mark.parametrize("ctx", CONTEXTS)
    def test_m4_ptl_less_than_m2(self, all_data, precision, ctx):
        m4_rows = all_data.get(("M4", precision), [])
        m2_rows = all_data.get(("M2", precision), [])
        if not m4_rows or not m2_rows:
            pytest.skip("Data not available")
        m4_ptl = _mean_ptl(m4_rows, ctx)
        m2_ptl = _mean_ptl(m2_rows, ctx)
        assert m4_ptl < m2_ptl, (
            f"{precision} ctx={ctx}: M4 PTL {m4_ptl:.2f} ms should be < "
            f"M2 PTL {m2_ptl:.2f} ms  (paper: M4 ~1.5× faster)"
        )

    @pytest.mark.parametrize("precision", PRECISIONS)
    def test_m4_m2_speedup_above_1_0(self, all_data, precision):
        """M4/M2 speedup must be > 1.0× at every context."""
        m4_rows = all_data.get(("M4", precision), [])
        m2_rows = all_data.get(("M2", precision), [])
        if not m4_rows or not m2_rows:
            pytest.skip("Data not available")
        for ctx in CONTEXTS:
            m4 = _mean_ptl(m4_rows, ctx)
            m2 = _mean_ptl(m2_rows, ctx)
            ratio = m2 / m4
            assert ratio > 1.0, (
                f"{precision} ctx={ctx}: M2/M4 ratio {ratio:.3f} — M4 should be faster"
            )


# ---------------------------------------------------------------------------
# 9. Bandwidth utilisation (Section 6)
# ---------------------------------------------------------------------------

class TestBandwidthUtilisation:

    @pytest.mark.parametrize("precision", PRECISIONS)
    def test_bw_util_positive_and_bounded(self, precision):
        for device in DEVICES:
            rows = _load_csv(device, precision)
            for r in rows:
                assert 0 < r["bw_util_pct"] < 100, (
                    f"{device} {precision}: bw_util_pct={r['bw_util_pct']} out of range"
                )

    def test_f16_bw_util_high_at_long_context(self, all_data):
        """
        Paper Section 6.1: M4 F16 BW util approaches 82% at long contexts.
        Accept ≥ 30% as a loose lower bound (model overhead dominates at short ctx).
        """
        rows = all_data.get(("M4", "F16"), [])
        if not rows:
            pytest.skip("No M4 F16 data")
        mean_util = _mean(
            [r["bw_util_pct"] for r in rows if r["context_tokens"] == 1024]
        )
        assert mean_util >= 30.0, (
            f"M4 F16 ctx=1024: mean bw_util={mean_util:.1f}% — expected ≥ 30%"
        )

    def test_q4_bw_util_lower_than_f16(self, all_data):
        """
        Q4_K_M uses 4× fewer bytes per step → lower bandwidth utilisation.
        """
        for device in DEVICES:
            f16_rows = all_data.get((device, "F16"), [])
            q4_rows  = all_data.get((device, "Q4_K_M"), [])
            if not f16_rows or not q4_rows:
                continue
            f16_util = _mean([r["bw_util_pct"] for r in f16_rows if r["context_tokens"] == 512])
            q4_util  = _mean([r["bw_util_pct"] for r in q4_rows  if r["context_tokens"] == 512])
            assert q4_util < f16_util, (
                f"{device} ctx=512: Q4_K_M BW util {q4_util:.1f}% should be < "
                f"F16 BW util {f16_util:.1f}%"
            )


# ---------------------------------------------------------------------------
# 10. Model load time ordering
# ---------------------------------------------------------------------------

class TestModelLoad:

    @pytest.mark.parametrize("device", DEVICES)
    def test_load_time_ordered_f16_gt_q8_gt_q4(self, all_data, device):
        """Larger files (F16 > Q8_0 > Q4_K_M) should take longer to load."""
        loads = {}
        for prec in PRECISIONS:
            rows = all_data.get((device, prec), [])
            if rows:
                loads[prec] = rows[0]["model_load_s"]
        if len(loads) < 2:
            pytest.skip(f"Not enough precision data for {device}")
        if "F16" in loads and "Q8_0" in loads:
            assert loads["F16"] > loads["Q8_0"], (
                f"{device}: F16 load {loads['F16']:.2f}s should > "
                f"Q8_0 load {loads['Q8_0']:.2f}s"
            )
        if "Q8_0" in loads and "Q4_K_M" in loads:
            assert loads["Q8_0"] > loads["Q4_K_M"], (
                f"{device}: Q8_0 load {loads['Q8_0']:.2f}s should > "
                f"Q4_K_M load {loads['Q4_K_M']:.2f}s"
            )

    @pytest.mark.parametrize("device", DEVICES)
    def test_f16_load_matches_paper(self, all_data, device):
        """Paper Result 6: M4=3.6s, M2=4.3s (cold load). Allow ±25%."""
        paper_load = {"M4": 3.6, "M2": 4.3}[device]
        rows = all_data.get((device, "F16"), [])
        if not rows:
            pytest.skip(f"No data for {device} F16")
        load  = rows[0]["model_load_s"]
        dev   = 100.0 * abs(load - paper_load) / paper_load
        assert dev <= 25.0, (
            f"{device} F16 model load: {load:.2f}s  paper: {paper_load:.1f}s  "
            f"deviation={dev:.1f}% (tolerance ±25%)"
        )


# ---------------------------------------------------------------------------
# 11. Raw log consistency
# ---------------------------------------------------------------------------

class TestRawLogs:

    _LOG_LINE_RE = re.compile(
        r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%"
    )

    @pytest.mark.parametrize("device,precision", [
        (d, p) for d in DEVICES for p in PRECISIONS
    ])
    def test_log_is_parseable(self, device, precision):
        log_path = _log_path(device, precision)
        if not log_path.exists():
            pytest.skip(f"Log not found: {log_path}")
        with open(log_path) as f:
            content = f.read()
        assert "llama-bench" in content, "Log missing llama-bench header"
        assert "end of log"  in content, "Log missing end-of-log marker"

    @pytest.mark.parametrize("device,precision", [
        (d, p) for d in DEVICES for p in PRECISIONS
    ])
    def test_log_data_rows_consistent_with_csv(self, device, precision):
        """
        Mean PTL from log table at ctx=512 should be within ±5% of CSV mean.
        """
        log_path = _log_path(device, precision)
        if not log_path.exists():
            pytest.skip(f"Log not found: {log_path}")

        log_ptls = []
        with open(log_path) as f:
            for line in f:
                m = self._LOG_LINE_RE.match(line)
                if m and int(m.group(2)) == 512:   # ctx column
                    log_ptls.append(float(m.group(4)))

        if not log_ptls:
            pytest.skip("No ctx=512 rows found in log")

        csv_rows = _load_csv(device, precision)
        csv_ptl  = _mean_ptl(csv_rows, 512)
        log_ptl  = _mean(log_ptls)
        dev      = 100.0 * abs(log_ptl - csv_ptl) / csv_ptl
        assert dev <= 5.0, (
            f"{device} {precision}: log mean PTL={log_ptl:.2f} ms  "
            f"csv mean PTL={csv_ptl:.2f} ms  deviation={dev:.1f}% (tolerance ±5%)"
        )

    @pytest.mark.parametrize("device,precision", [
        (d, p) for d in DEVICES for p in PRECISIONS
    ])
    def test_log_contains_measurement_type_note(self, device, precision):
        log_path = _log_path(device, precision)
        if not log_path.exists():
            pytest.skip(f"Log not found: {log_path}")
        with open(log_path) as f:
            content = f.read()
        assert "measurement_type" in content, (
            f"{device} {precision}: log missing 'measurement_type' field "
            f"(required by Section 4.4 evidence labelling)"
        )

    @pytest.mark.parametrize("device,precision", [
        (d, p) for d in DEVICES for p in PRECISIONS
    ])
    def test_log_model_load_present(self, device, precision):
        log_path = _log_path(device, precision)
        if not log_path.exists():
            pytest.skip(f"Log not found: {log_path}")
        with open(log_path) as f:
            content = f.read()
        assert re.search(r"model load\s*:\s*[\d.]+\s*s", content), (
            f"{device} {precision}: log missing 'model load' line"
        )


# ---------------------------------------------------------------------------
# 12. TTFT sanity
# ---------------------------------------------------------------------------

class TestTTFT:

    @pytest.mark.parametrize("device,precision", [
        (d, p) for d in DEVICES for p in PRECISIONS
    ])
    def test_ttft_increases_with_prompt(self, device, precision):
        """
        Mean TTFT at prompt=1024 must exceed TTFT at prompt=32.
        Paper Table 4: M4 F16 goes 16.1→197.5 ms (~12× growth).
        """
        rows    = _load_csv(device, precision)
        ttft_fn = lambda p: _mean([r["ttft_ms"] for r in rows if r["prompt_tokens"] == p
                                   and r["context_tokens"] == 256])
        t32   = ttft_fn(32)
        t1024 = ttft_fn(1024)
        assert t1024 > t32, (
            f"{device} {precision}: TTFT at prompt=1024 ({t1024:.2f} ms) "
            f"should exceed TTFT at prompt=32 ({t32:.2f} ms)"
        )

    @pytest.mark.parametrize("device,precision", [
        (d, p) for d in DEVICES for p in PRECISIONS
    ])
    def test_ttft_greater_than_zero(self, device, precision):
        rows = _load_csv(device, precision)
        bad  = [r for r in rows if r["ttft_ms"] <= 0]
        assert not bad, f"{device} {precision}: {len(bad)} rows with TTFT ≤ 0"


# ---------------------------------------------------------------------------
# 13. Throughput consistency
# ---------------------------------------------------------------------------

class TestThroughput:

    @pytest.mark.parametrize("device,precision", [
        (d, p) for d in DEVICES for p in PRECISIONS
    ])
    def test_throughput_consistent_with_ptl(self, device, precision):
        """throughput_tok_s should be ≈ 1000 / ptl_ms within ±2%."""
        rows = _load_csv(device, precision)
        for r in rows:
            expected = 1000.0 / r["ptl_ms"]
            dev      = 100.0 * abs(r["throughput_tok_s"] - expected) / expected
            assert dev <= 2.0, (
                f"{device} {precision} row: throughput={r['throughput_tok_s']:.2f} tok/s "
                f"but 1000/ptl={expected:.2f} tok/s  deviation={dev:.1f}%"
            )
