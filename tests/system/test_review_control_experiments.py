"""Smoke tests for the review-motivated control experiments.

Both scripts are deterministic given their seeds and run in seconds at
paper scale; they re-derive the headline aggregates and check the structural
claims the paper cites (parity gap, false-hit accounting, dead-only
fusion), so a silent regression in either artifact fails loudly.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run(script: str, extra: list[str], tmp_out: Path) -> dict:
    r = subprocess.run(
        [sys.executable, str(ROOT / "examples/benchmarks" / script)]
        + extra + ["--output", str(tmp_out)],
        capture_output=True, text=True, timeout=300, cwd=ROOT,
        env={"PATH": "/usr/bin:/bin", "HOME": str(ROOT)},
    )
    assert r.returncode == 0, f"{script} failed:\n{r.stdout}\n{r.stderr}"
    return json.loads(tmp_out.read_text(encoding="utf-8"))


def test_trace_compile_control(tmp_path):
    report = _run(
        "trace_compile_control.py", ["--trace-dir",
                                     str(ROOT / "examples/random_task_traces")],
        tmp_path / "ctrl.json")
    agg = report["aggregate"]
    # Deterministic full census: exact pinned totals.
    assert agg["total_steps"] == 792
    assert agg["total_dead_steps"] == 152
    assert agg["total_fusible_pairs_compiler_gate"] == 106
    # The structural claim the paper cites: every compiler-gate
    # fusible pair involves a dead step; no live-live fusion exists.
    assert agg["total_fusible_pairs_both_live"] == 0
    # 792 -> 640 at BASIC/AGGRESSIVE: reduction is dead-step
    # elimination exactly (152 = 106 fusion-absorbed + 46 DCE-removed).
    assert agg["effective"]["BASIC"] == 640
    assert agg["effective"]["AGGRESSIVE"] == 640
    assert agg["fused_instructions"]["BASIC"] == 106
    assert abs(agg["reduction_vs_none"]["BASIC"] - 0.1919) < 5e-5


def test_cache_key_differentiation(tmp_path):
    report = _run(
        "cache_key_differentiation.py", [],
        tmp_path / "keys.json")
    w1 = report["w1_param_reordering"]["aggregate"]
    w2 = report["w2_template_churn"]["aggregate"]
    # W1: the instruction-level key dominates the order-sensitive
    # response key under param reordering (parity breaks).
    assert w1["instruction_key_hit_rate"]["mean"] > 0.90
    assert w1["response_key_hit_rate"]["mean"] < 0.15
    assert (w1["instruction_key_hit_rate"]["mean"]
            - w1["response_key_hit_rate"]["mean"]) > 0.75
    # W2: the default params-only key false-hits under template churn;
    # the strict namespace key never does.
    assert w2["default_key_false_hit_rate"]["mean"] > 0.40
    # strict false hits are not aggregated (absence in output = zero
    # across seeds); check per-seed rows directly.
    for row in report["w2_template_churn"]["per_seed"]:
        assert row["strict_key_false_hits"] == 0
    # Strict still hits (it caches per (task, template)).
    assert w2["strict_key_hit_rate"]["mean"] > 0.80
