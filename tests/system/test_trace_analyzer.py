"""Workload-characterization tests for the trace analyzer (I10).

``examples/analyze_task_traces.py`` characterizes the JSONL trace
corpus produced by ``random_tasks_stress.py``: a least-squares Zipf
fit over the instruction-signature frequency distribution, (type,
params) reuse potential, a dependency-shape census (linear / fan-out
/ fan-in / branch), and adjacent fusable-pair opportunities under
the compiler's fusion gate (same type, both pure, no producer-consumer
edge between the pair).

All numbers below come from small synthetic trace fixtures with
hand-computable expectations; one optional smoke test exercises the
real 100-trace corpus when it is present.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.analyze_task_traces import (  # noqa: E402
    analyze,
    classify_workflow_shape,
    fit_zipf,
    fusion_opportunity,
    instruction_signature,
    params_signature,
    reuse_potential,
)

REAL_TRACE_DIR = (
    Path(__file__).resolve().parents[2] / "examples" / "random_task_traces"
)


# ---------------------------------------------------------------------------
# Synthetic-fixture helpers (trace records in the ell_export JSONL shape)
# ---------------------------------------------------------------------------

def make_step(sid, itype="CONTROL_FLOW", deps=(), outputs=(), pure=True,
              name=None, params=None):
    step = {
        "id": sid,
        "name": name or sid,
        "type": itype,
        "description": "",
        "dependencies": list(deps),
        "outputs": list(outputs),
        "side_effects": not pure,
        "step": 0,
    }
    if params is not None:
        step["params"] = params
    return step


def write_trace(path, steps, name="wf", trailing_messages=0):
    """Write one JSONL trace: a workflow record + N message records."""
    wf = {
        "kind": "workflow", "id": "00000000-0000-0000-0000-000000000000",
        "name": name, "description": "", "steps": steps,
        "metadata": {"instruction_count": len(steps)},
    }
    lines = [json.dumps(wf)]
    for i in range(trailing_messages):
        lines.append(json.dumps({
            "kind": "message", "role": "user", "type": "text",
            "content": f"noise {i}", "metadata": {},
        }))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def linear_chain_ops(ops):
    """Chain steps op0 -> op1 -> ... via v0, v1, ... (all pure)."""
    steps, key = [], None
    for i, op in enumerate(ops):
        out = f"v{i}"
        steps.append(make_step(
            f"{op}_{i}", deps=[] if key is None else [key, ], outputs=[out]))
        key = out
    return steps


# ---------------------------------------------------------------------------
# (1) Zipf fit
# ---------------------------------------------------------------------------

class TestFitZipf:
    def test_perfect_zipf(self):
        """counts [6, 3, 2] lie exactly on freq = 6 * rank**-1."""
        out = fit_zipf({"x": 6, "y": 3, "z": 2})
        assert out["n_categories"] == 3
        assert out["total"] == 11
        assert out["s"] == pytest.approx(1.0)
        assert out["r2"] == pytest.approx(1.0)

    def test_single_category_unfittable(self):
        out = fit_zipf({"x": 5})
        assert out["n_categories"] == 1
        assert out["total"] == 5
        assert out["s"] is None and out["r2"] is None

    def test_uniform_has_zero_exponent(self):
        """Zero-variance distribution: s = 0; r2 = 0.0 (nothing to
        explain -- documented convention)."""
        out = fit_zipf({"a": 4, "b": 4, "c": 4})
        assert out["s"] == pytest.approx(0.0)
        assert out["r2"] == 0.0

    def test_heavy_head_gives_positive_exponent(self):
        out = fit_zipf({"a": 100, "b": 1})
        assert out["s"] > 0
        assert out["r2"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# (type, params) signatures
# ---------------------------------------------------------------------------

class TestSignatures:
    def test_positional_index_segments_are_stripped(self):
        assert params_signature(make_step("filter_warn_5")) == "filter_warn"
        assert params_signature(make_step("guard_0_2")) == "guard"
        assert params_signature(make_step("classify_4")) == "classify"

    def test_names_without_indices_unchanged(self):
        assert params_signature(make_step("ingest")) == "ingest"
        assert params_signature(make_step("write_report")) == "write_report"

    def test_explicit_params_canonicalized_sorted(self):
        a = make_step("s", params={"b": 1, "a": [1, 2]})
        b = make_step("s", params={"a": [1, 2], "b": 1})
        c = make_step("s", params={"a": [1, 3], "b": 1})
        assert params_signature(a) == params_signature(b)
        assert params_signature(a) != params_signature(c)

    def test_instruction_signature_joins_type(self):
        s = make_step("filter_warn_5", itype="CONTROL_FLOW")
        assert instruction_signature(s) == "CONTROL_FLOW:filter_warn"


# ---------------------------------------------------------------------------
# (2) Reuse potential
# ---------------------------------------------------------------------------

class TestReusePotential:
    def test_repeat_ratio_corpus_and_per_workflow(self):
        wf1 = {"steps": linear_chain_ops(["x"] * 2 + ["y"])}
        wf2 = {"steps": linear_chain_ops(["x", "z"])}
        out = reuse_potential([wf1, wf2])
        assert out["total_instances"] == 5
        assert out["distinct_signatures"] == 3
        assert out["repeat_instances"] == 2
        assert out["reuse_ratio"] == pytest.approx(2 / 5)
        # wf1: (3-2)/3, wf2: (2-2)/2 -> mean 1/6
        assert out["per_workflow_mean_reuse_ratio"] == pytest.approx(1 / 6)

    def test_empty_corpus(self):
        out = reuse_potential([])
        assert out["total_instances"] == 0
        assert out["reuse_ratio"] == 0.0
        assert out["per_workflow_mean_reuse_ratio"] == 0.0

    def test_top_list(self):
        wf = {"steps": linear_chain_ops(["x"] * 3 + ["y"])}
        out = reuse_potential([wf])
        assert out["top"][0] == ["CONTROL_FLOW:x", 3]


# ---------------------------------------------------------------------------
# (3) Dependency-shape census
# ---------------------------------------------------------------------------

class TestClassifyWorkflowShape:
    def test_linear_chain(self):
        steps = linear_chain_ops(["a", "b", "c"])
        assert classify_workflow_shape(steps) == "linear"

    def test_fan_out(self):
        steps = [
            make_step("src", outputs=["u"]),
            make_step("left", deps=["u"]),
            make_step("right", deps=["u"]),
        ]
        assert classify_workflow_shape(steps) == "fan-out"

    def test_fan_in(self):
        steps = [
            make_step("s1", outputs=["a"]),
            make_step("s2", outputs=["b"]),
            make_step("merge", deps=["a", "b"]),
        ]
        assert classify_workflow_shape(steps) == "fan-in"

    def test_branch_diamond(self):
        steps = [
            make_step("src", outputs=["u"]),
            make_step("left", deps=["u"], outputs=["l"]),
            make_step("right", deps=["u"], outputs=["r"]),
            make_step("merge", deps=["l", "r"]),
        ]
        assert classify_workflow_shape(steps) == "branch"

    def test_dependencies_by_step_id_resolve(self):
        steps = [
            make_step("a", outputs=[]),
            make_step("b", deps=["a"]),
        ]
        assert classify_workflow_shape(steps) == "linear"


# ---------------------------------------------------------------------------
# (4) Adjacent fusable pairs (compiler gate)
# ---------------------------------------------------------------------------

class TestFusionOpportunity:
    def test_gate_requires_same_type_purity_no_dataflow(self):
        steps = [
            make_step("p0", outputs=[]),                     # pure
            make_step("p1", deps=["ext"], outputs=["v"]),    # fuses w/ p0
            make_step("p2", deps=["v"]),                     # p1's consumer
            make_step("w", itype="STATE_MANAGEMENT",
                      deps=["v"], pure=False),
        ]
        out = fusion_opportunity([{"steps": steps}])
        assert out["total_adjacent_pairs"] == 3
        assert out["fusable_pairs"] == 1
        assert out["fusable_ratio"] == pytest.approx(1 / 3)
        assert out["by_type"] == {"CONTROL_FLOW": 1}

    def test_empty_workflow(self):
        out = fusion_opportunity([{"steps": []}])
        assert out["total_adjacent_pairs"] == 0
        assert out["fusable_ratio"] == 0.0


# ---------------------------------------------------------------------------
# End-to-end analyze() on a synthetic corpus
# ---------------------------------------------------------------------------

def _build_corpus(tmp_path):
    """Two workflows + one message-only file (must be skipped).

    taskA: 13 pure chain steps (x*9, y*3, z*1) + side-effectful writer.
    taskB: fan-out ingest -> {dead guard, tail} -> writer.
    taskC: message records only, no workflow record.
    """
    a_steps = linear_chain_ops(["x"] * 9 + ["y"] * 3 + ["z"])
    a_steps.append(make_step(
        "write_report", itype="STATE_MANAGEMENT", deps=["v12"],
        pure=False))
    write_trace(tmp_path / "taskA.jsonl", a_steps, name="taskA",
                trailing_messages=2)

    b_steps = [
        make_step("ingest", itype="STATE_MANAGEMENT", outputs=["u"],
                  pure=False),
        make_step("guard_0_0", deps=["u"], outputs=[]),
        make_step("tail_1", deps=["u"], outputs=["w"]),
        make_step("write_report", itype="STATE_MANAGEMENT", deps=["w"],
                  pure=False),
    ]
    write_trace(tmp_path / "taskB.jsonl", b_steps, name="taskB")

    (tmp_path / "taskC.jsonl").write_text(
        json.dumps({"kind": "message", "role": "user", "type": "text",
                    "content": "no workflow here", "metadata": {}}) + "\n",
        encoding="utf-8")
    return tmp_path


class TestAnalyze:
    def test_provenance_and_counts(self, tmp_path):
        report = analyze(_build_corpus(tmp_path))
        prov = report["provenance"]
        assert prov["n_traces"] == 2  # taskC skipped
        # provenance lists every scanned file, including the
        # message-only one that contributed no workflow
        assert prov["files"] == ["taskA.jsonl", "taskB.jsonl",
                                 "taskC.jsonl"]
        assert prov["n_steps"] == 18
        assert prov["corpus_sha256"]  # non-empty, recomputed below

    def test_zipf_section(self, tmp_path):
        z = analyze(_build_corpus(tmp_path))["zipf"]
        # x:9 y:3 write_report:2 z/guard/tail/ingest:1 -> 7 categories
        assert z["n_categories"] == 7
        assert z["total"] == 18
        assert z["s"] > 0
        assert 0.0 <= z["r2"] <= 1.0
        assert z["top"][0] == ["CONTROL_FLOW:x", 9]

    def test_reuse_section(self, tmp_path):
        r = analyze(_build_corpus(tmp_path))["reuse"]
        assert r["total_instances"] == 18
        assert r["distinct_signatures"] == 7
        assert r["repeat_instances"] == 11  # x:8, y:2, write_report:1
        assert r["reuse_ratio"] == pytest.approx(11 / 18)
        # taskA 10/14, taskB 0/4
        assert r["per_workflow_mean_reuse_ratio"] == pytest.approx(5 / 14)

    def test_shape_census_section(self, tmp_path):
        s = analyze(_build_corpus(tmp_path))["shapes"]
        assert s["counts"] == {"linear": 1, "fan-out": 1,
                               "fan-in": 0, "branch": 0}
        assert s["dead_end_steps"] == 1  # the planted guard
        assert s["workflows_with_multiple_components"] == 0
        assert s["avg_steps"] == pytest.approx(9.0)

    def test_fusion_section(self, tmp_path):
        f = analyze(_build_corpus(tmp_path))["fusion"]
        assert f["total_adjacent_pairs"] == 16
        assert f["fusable_pairs"] == 1  # (guard_0_0, tail_1)
        assert f["fusable_ratio"] == pytest.approx(1 / 16)

    def test_report_is_deterministic(self, tmp_path):
        _build_corpus(tmp_path)
        assert analyze(tmp_path) == analyze(tmp_path)


# ---------------------------------------------------------------------------
# Optional smoke test on the real corpus
# ---------------------------------------------------------------------------

def _real_corpus_present():
    return REAL_TRACE_DIR.is_dir() and \
        len(list(REAL_TRACE_DIR.glob("*.jsonl"))) == 100


@pytest.mark.skipif(not _real_corpus_present(),
                    reason="real 100-trace corpus not present")
class TestRealCorpusSmoke:
    def test_basic_sanity(self):
        report = analyze(REAL_TRACE_DIR)
        prov = report["provenance"]
        assert prov["n_traces"] == 100
        assert prov["n_steps"] > 0
        z = report["zipf"]
        assert z["n_categories"] >= 5
        assert z["s"] > 0
        assert 0.0 <= z["r2"] <= 1.0
        assert 0.0 <= report["reuse"]["reuse_ratio"] <= 1.0
        assert sum(report["shapes"]["counts"].values()) == 100
        f = report["fusion"]
        assert 0.0 <= f["fusable_ratio"] <= 1.0
