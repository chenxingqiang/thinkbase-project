"""Cost model: linear in the effect count, honest about what is modeled.

The script converts measured avoided-unit counts into seconds, tokens and dollars.
Those conversions are only meaningful if they are linear (a doubled workload
doubles the saving, with no offsets), if a zero effect yields zero saving (no
constant dressed up as a benefit), and if the declared band actually brackets the
point estimate. Each of those is asserted here, plus the provenance rule that the
counts in the payload are the counts in the committed files.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "examples" / "benchmarks" / "cost_model.py"


def _module():
    name = "cost_model"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _payload():
    return _module().build_payload()


def test_conversion_is_linear_in_the_effect_count() -> None:
    cm = _module()
    prices = cm.load_unit_prices(cm.DEFAULT_UNIT_RESULTS)
    single = cm.convert_effect(746.0, prices)
    doubled = cm.convert_effect(1492.0, prices)
    for channel in ("seconds", "input_tokens", "output_tokens", "dollars"):
        if channel == "seconds":
            for stat in ("mean", "p50", "p95"):
                assert doubled["seconds"][stat] == 2 * single["seconds"][stat]
        else:
            assert doubled[channel] == 2 * single[channel]
    assert doubled["unit_cost_dollars"] == single["unit_cost_dollars"]


def test_zero_effect_yields_zero_saving() -> None:
    cm = _module()
    prices = cm.load_unit_prices(cm.DEFAULT_UNIT_RESULTS)
    point = cm.convert_effect(0.0, prices)
    assert point["avoided_units"] == 0.0
    assert point["seconds"] == {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    assert point["input_tokens"] == 0.0
    assert point["output_tokens"] == 0.0
    assert point["dollars"] == 0.0
    band = cm.sensitivity(0.0, prices)
    assert band["dollars"] == {
        "low": 0.0,
        "high": 0.0,
        "corners": band["dollars"]["corners"],
    }
    assert all(corner["dollars"] == 0.0 for corner in band["dollars"]["corners"])
    assert band["input_tokens"] == {"low": 0.0, "high": 0.0}
    assert band["output_tokens"] == {"low": 0.0, "high": 0.0}


def test_band_brackets_the_point_estimate_for_every_mechanism() -> None:
    payload = _payload()
    assert payload["mechanisms"]
    for row in payload["mechanisms"]:
        point = row["point"]
        if point is None:
            # not priceable: the avoided unit is a tool execution, not a model call
            assert row["convertible"] is False and "not_priceable" in row
            continue
        band = row["sensitivity"]
        assert band["dollars"]["low"] <= point["dollars"] <= band["dollars"]["high"]
        assert band["input_tokens"]["low"] <= point["input_tokens"] <= band["input_tokens"]["high"]
        assert band["output_tokens"]["low"] <= point["output_tokens"] <= band["output_tokens"]["high"]
        assert band["seconds"]["low"] <= point["seconds"]["mean"] <= band["seconds"]["high"]
        assert band["seconds"]["low"] == point["seconds"]["p50"]
        assert band["seconds"]["high"] == point["seconds"]["p95"]
        # Price and token multipliers act on dollars only.
        assert band["seconds"]["invariant_to_price"] is True
        assert band["seconds"]["invariant_to_token_count"] is True


def test_dollar_band_is_the_four_corner_envelope() -> None:
    payload = _payload()
    for row in payload["mechanisms"]:
        if row["point"] is None:
            continue
        corners = row["sensitivity"]["dollars"]["corners"]
        assert len(corners) == 4
        assert {(c["price_multiplier"], c["token_multiplier"]) for c in corners} == {
            (0.5, 0.7), (0.5, 1.3), (1.5, 0.7), (1.5, 1.3)
        }
        values = [c["dollars"] for c in corners]
        assert row["sensitivity"]["dollars"]["low"] == min(values)
        assert row["sensitivity"]["dollars"]["high"] == max(values)


def test_payload_declares_itself_modeled_not_observed() -> None:
    payload = _payload()
    assert payload["result_kind"] == "MODELED"
    assert payload["not_observed"] is True
    assert "MODELED" in payload["statement"]
    assert "not observed" in payload["statement"].lower()
    assert payload["price_assumption"]["status"].startswith("assumed")
    assert payload["price_assumption"]["provenance"].endswith("Appendix G (priced-comparison protocol)")
    assert payload["aggregate"]["sum_allowed"] is False
    assert payload["aggregate"]["max_mechanism"] in {"F1", "F2", "F3"}


def test_counts_are_read_from_the_committed_files_not_invented() -> None:
    cm = _module()
    payload = cm.build_payload()
    rows = {row["id"]: row for row in payload["mechanisms"]}

    cache = json.loads(cm.DEFAULT_CACHE_RESULTS.read_text(encoding="utf-8"))
    first = cache["cache_scopes"]["first_trial"]
    assert rows["F1"]["effect_units"] == first["within_run_hits"]
    assert rows["F1"]["effect_source"]["denominator"] == first["calls"]

    template = json.loads(cm.DEFAULT_TEMPLATE_RESULTS.read_text(encoding="utf-8"))
    assert rows["F2"]["effect_units"] == template["interleaved"]["served"]
    assert rows["F2"]["effect_source"]["denominator"] == template["interleaved"]["steps"]

    compound = json.loads(cm.DEFAULT_COMPOUND_RESULTS.read_text(encoding="utf-8"))
    assert rows["F3"]["effect_units"] == compound["reduction_if_substituted"]["steps_removed"]
    assert rows["F3"]["effect_source"]["denominator"] == compound["reduction_if_substituted"]["steps"]

    # Every measured input is fingerprinted, so a payload can be traced to its files.
    assert set(payload["provenance"]["inputs"]) == {
        "examples/benchmarks/isa_real_llm_results.json",
        "examples/benchmarks/real_trace_pilot_tau_bench_results.json",
        "examples/benchmarks/instruction_precipitation_results.json",
        "examples/benchmarks/compound_binding_results.json",
    }
    assert all(len(meta["sha256"]) == 64 for meta in payload["provenance"]["inputs"].values())


def test_unit_prices_come_from_the_gateway_run() -> None:
    cm = _module()
    payload = cm.build_payload()
    prices = payload["unit_prices"]
    assert prices["t_mean_s"] == 2.297587
    assert prices["t_p50_s"] == 2.172433
    assert prices["t_p95_s"] == 3.2974
    assert prices["input_tokens_per_call"] == 21.0
    assert prices["output_tokens_per_call"] == 64.0
    assert prices["output_tokens_per_call"] == float(payload["price_assumption"]["regime"]["max_tokens"])
    assert payload["units"]["unit_price_dollars_per_avoided_unit"] == 0.001023
    assert prices["source"] == "examples/benchmarks/isa_real_llm_results.json"


def test_denominator_guard_rejects_the_corpus_wide_inflation() -> None:
    payload = _payload()
    guard = payload["denominator_guard"]
    assert guard["accepted_denominator"] == 1789
    assert guard["accepted_avoided_calls"] == 746.0
    assert guard["rejected_denominator"] == 14285
    # 0.41699 x 14,285 is the invalid 8x-inflated count; the accepted count is not it.
    assert abs(guard["rejected_avoided_calls"] - 5956.8) < 0.2
    assert guard["rejected_avoided_calls"] > 7.9 * guard["accepted_avoided_calls"]
    assert payload["mechanisms"][0]["effect_units"] != guard["rejected_avoided_calls"]


def test_point_emits_avoided_calls_with_their_basis() -> None:
    payload = _payload()
    rows = {row["id"]: row for row in payload["mechanisms"]}
    # F1 is deliberately left unpriced: a cache hit replays a recorded tool
    # execution, and the model turn that emitted the call still happens.
    assert rows["F1"]["effect_units"] == 746.0
    assert rows["F1"]["convertible"] is False
    assert rows["F1"]["point"] is None
    assert "not_priceable" in rows["F1"]
    assert "tool execution" in rows["F1"]["effect_units_are"]
    for mechanism in ("F2", "F3"):
        point = rows[mechanism]["point"]
        assert point["avoided_calls"] == rows[mechanism]["effect_units"]
        assert "planner bypass" in point["avoided_calls_basis"]


def test_assumption_conditioned_rows_say_so() -> None:
    payload = _payload()
    rows = {row["id"]: row for row in payload["mechanisms"]}
    assert rows["F1"]["assumption_conditioned"] is False
    assert rows["F2"]["assumption_conditioned"] is True
    assert rows["F3"]["assumption_conditioned"] is True
    for mechanism in ("F2", "F3"):
        assert "planner bypass" in rows[mechanism]["conversion_assumption"]


def test_cli_writes_a_deterministic_payload(tmp_path: Path) -> None:
    cm = _module()
    first = tmp_path / "cost_model_results.json"
    second = tmp_path / "again.json"
    assert cm.main(["--output", str(first)]) == 0
    assert cm.main(["--output", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()

    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["schema"] == cm.SCHEMA_VERSION
    assert all(check["ok"] for check in payload["checks"])
    assert payload["aggregate"]["max_mechanism"] == "F2"
