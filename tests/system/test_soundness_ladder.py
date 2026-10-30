"""Cost-of-soundness ladder: the nesting the paper's cost claims depend on.

``soundness_ladder.py`` replays one corpus through two four-rung ladders and
counts hits and unsafe hits at each rung. The sizes are the paper's business;
what this module pins is the structure that makes "each ingredient costs X
points of hit rate and removes Y unsafe hits" a measured statement rather than a
decomposition by assumption:

  1. the main ladder scores one admitted set -- the 1,728 reads -- at every rung,
     so a rung delta cannot be an admission change;
  2. unsafe hits are monotone non-increasing along both ladders, reach zero at
     L2 on the reads-only ladder and stay there at L3;
  3. the mirror ladder does *not* reach zero before its admission rung: with
     writes admitted, a tenant-bound key plus entity invalidation still serves
     repeated writes, and only refusing them clears the total;
  4. every marginal cost is arithmetic on the two rung tables it joins -- never a
     second measurement -- and the rung 3 "admission" cost on the main ladder is
     exactly zero, because admission was given before rung 0;
  5. re-running the script over the corpus reproduces the committed file byte
     for byte.

The payload carries no ``generated_at``-style field: ``provenance`` pins content
hashes, not a clock. Regenerate the committed artifact in place with::

    ./venv/bin/python examples/benchmarks/soundness_ladder.py \\
        --output examples/benchmarks/soundness_ladder_results.json

The results tests skip (rather than fail) when that artifact or the corpus is
absent, matching the convention for the repo's other generated payloads.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "examples" / "benchmarks" / "soundness_ladder.py"
COMMITTED = REPO_ROOT / "examples" / "benchmarks" / "soundness_ladder_results.json"
CORPUS = REPO_ROOT / "examples" / "real_traces" / "tau_bench_historical.jsonl"

BREAKDOWN_FIELDS = ("writes_served", "stale_reads", "cross_tenant")
ADMITTED_READS = 1728
ALL_CALLS = 2275

REGENERATE = (
    "./venv/bin/python examples/benchmarks/soundness_ladder.py "
    "--output examples/benchmarks/soundness_ladder_results.json"
)

pytestmark = pytest.mark.skipif(
    not COMMITTED.exists(),
    reason=f"committed ladder results missing; regenerate with: {REGENERATE}",
)


def _payload() -> dict[str, Any]:
    return json.loads(COMMITTED.read_text(encoding="utf-8"))


def _ladder(name: str) -> dict[str, Any]:
    return _payload()["ladders"][name]


def _rung(ladder: str, rid: str) -> dict[str, Any]:
    for rung in _ladder(ladder)["rungs"]:
        if rung["id"] == rid:
            return rung
    raise AssertionError(f"rung {rid!r} missing from ladder {ladder!r}")


def _first_difference(left: Any, right: Any, path: str = "payload") -> str | None:
    """Describe the first place two JSON payloads disagree, for failure messages."""
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return (f"{path}: keys differ (only committed: "
                    f"{sorted(set(left) - set(right))}; only fresh: "
                    f"{sorted(set(right) - set(left))})")
        for key in sorted(left, key=str):
            found = _first_difference(left[key], right[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for index, (a, b) in enumerate(zip(left, right)):
            found = _first_difference(a, b, f"{path}[{index}]")
            if found:
                return found
        return None
    if left != right:
        return f"{path}: committed {left!r} != fresh {right!r}"
    return None


# ---------------------------------------------------------------------------
# (1) One admitted set, and it is the typed policy's
# ---------------------------------------------------------------------------
def test_main_ladder_scores_one_admitted_set_of_the_1728_reads() -> None:
    payload = _payload()
    main = payload["ladders"]["main"]
    admitted = {rung["admitted_calls"] for rung in main["rungs"]}

    assert admitted == {ADMITTED_READS}, (
        f"main rungs admit {sorted(admitted)} calls, not one set of {ADMITTED_READS}"
    )
    assert main["denominator"] == ADMITTED_READS
    for rung in main["rungs"]:
        assert rung["calls_total"] == ALL_CALLS, "every rung replays the same corpus"
        assert rung["admission"] == "reads"
        assert rung["hits_by_effect"]["write"] == 0
        assert rung["hits_by_effect"]["control"] == 0

    recon = payload["reconciliation"]
    assert recon["admitted_set"]["reads"] == ADMITTED_READS
    assert recon["admitted_set"]["reads_is_the_denominator"] is True
    assert recon["calls_total"] == ALL_CALLS
    assert recon["admitted_set"]["writes"] + recon["admitted_set"]["controls"] == (
        ALL_CALLS - ADMITTED_READS
    )


# ---------------------------------------------------------------------------
# (2) Unsafe falls monotonically, and reaches zero where the design says
# ---------------------------------------------------------------------------
def test_unsafe_hits_are_monotone_on_both_ladders() -> None:
    for ladder, expected_zero_from in (("main", 2), ("mirror", 3)):
        rungs = _ladder(ladder)["rungs"]
        unsafe = [rung["unsafe_hits"] for rung in rungs]
        assert all(a >= b for a, b in zip(unsafe, unsafe[1:])), (
            f"{ladder}: unsafe hits rise along {[r['id'] for r in rungs]}: {unsafe}"
        )
        assert unsafe[0] > 0, f"{ladder}: a zero baseline would make the ladder vacuous"
        assert unsafe[expected_zero_from] == 0, (
            f"{ladder}: rung {rungs[expected_zero_from]['id']} serves {unsafe[expected_zero_from]} "
            "unsafe hits"
        )
        assert all(value == 0 for value in unsafe[expected_zero_from:])
        for rung in rungs:
            assert sum(rung["unsafe_breakdown"].values()) == rung["unsafe_hits"]
            assert set(rung["unsafe_breakdown"]) == set(BREAKDOWN_FIELDS)
            assert rung["unsafe_hits"] <= rung["hits"]


def test_admission_is_what_clears_the_mirror_ladder() -> None:
    m2 = _rung("mirror", "M2")
    m3 = _rung("mirror", "M3")

    assert m2["invalidation"] != "none" and m2["key_binds_tenant"] is True
    assert m2["unsafe_hits"] > 0, (
        "with writes admitted, a tenant-bound key plus entity invalidation must still "
        "serve repeated writes; a zero here would mean the mirror measured nothing"
    )
    assert set(m2["unsafe_breakdown"]) == set(BREAKDOWN_FIELDS)
    assert m2["unsafe_breakdown"]["writes_served"] == m2["unsafe_hits"], (
        "the residue at M2 is served writes, not staleness or cross-tenant reads"
    )
    assert m3["unsafe_hits"] == 0
    assert m3["admitted_calls"] == ADMITTED_READS


def test_rung_three_restates_rung_two_on_the_main_ladder() -> None:
    l2, l3 = _rung("main", "L2"), _rung("main", "L3")

    assert (l2["hits"], l2["unsafe_hits"]) == (l3["hits"], l3["unsafe_hits"])
    assert l2["unsafe_breakdown"] == l3["unsafe_breakdown"]
    cost = _ladder("main")["marginal_costs"][2]
    assert (cost["from_rung"], cost["to_rung"]) == ("L2", "L3")
    assert cost["hit_rate_points_lost"] == 0
    assert cost["unsafe_hits_removed"] == 0
    assert cost["hits_lost"] == 0


# ---------------------------------------------------------------------------
# (3) Costs are arithmetic on the rungs, measured on one denominator
# ---------------------------------------------------------------------------
def test_marginal_costs_are_differences_of_adjacent_rungs() -> None:
    for ladder in ("main", "mirror"):
        doc = _ladder(ladder)
        by_id = {rung["id"]: rung for rung in doc["rungs"]}
        costs = doc["marginal_costs"]
        assert len(costs) == len(doc["rungs"]) - 1

        for cost in costs:
            before = by_id[cost["from_rung"]]
            after = by_id[cost["to_rung"]]
            assert cost["denominator"] == doc["denominator"]
            assert cost["hits_lost"] == before["hits"] - after["hits"]
            assert cost["unsafe_hits_removed"] == before["unsafe_hits"] - after["unsafe_hits"]
            assert cost["hit_rate_before"] == before["hit_rate_on_denominator"]
            assert cost["hit_rate_after"] == after["hit_rate_on_denominator"]
            assert cost["hit_rate_points_lost"] == pytest.approx(
                (cost["hit_rate_before"] - cost["hit_rate_after"]) * 100, abs=0.006
            )
            assert set(cost["unsafe_removed_by_class"]) == set(BREAKDOWN_FIELDS)
            assert sum(cost["unsafe_removed_by_class"].values()) == cost["unsafe_hits_removed"]
            assert str(cost["hit_rate_points_lost"]) in cost["sentence"]

        total = doc["total_cost"]
        assert total["hit_rate_points_lost"] == pytest.approx(
            (total["hit_rate_first"] - total["hit_rate_last"]) * 100, abs=0.006
        )
        assert total["hits_lost"] == doc["rungs"][0]["hits"] - doc["rungs"][-1]["hits"]
        assert total["unsafe_hits_removed"] == (
            doc["rungs"][0]["unsafe_hits"] - doc["rungs"][-1]["unsafe_hits"]
        )


def test_rates_are_shares_of_the_stated_denominator() -> None:
    for ladder in ("main", "mirror"):
        doc = _ladder(ladder)
        for rung in doc["rungs"]:
            assert 0.0 <= rung["hit_rate"] <= 1.0
            assert rung["hit_rate"] == pytest.approx(
                rung["hits"] / rung["admitted_calls"], abs=5e-5
            )
            assert rung["hit_rate_on_denominator"] == pytest.approx(
                rung["hits"] / doc["denominator"], abs=5e-5
            )
            assert rung["hit_rate_of_all_calls"] == pytest.approx(
                rung["hits"] / rung["calls_total"], abs=5e-5
            )


def test_admitting_writes_adds_hits_and_unsafe_hits() -> None:
    added = _payload()["what_admitting_writes_adds"]
    m2, m3 = _rung("mirror", "M2"), _rung("mirror", "M3")

    assert added["hits_added"] == m2["hits"] - m3["hits"] > 0
    assert added["unsafe_hits_added"] == m2["unsafe_hits"] - m3["unsafe_hits"] > 0
    assert added["hits_added_by_effect"]["write"] == m2["unsafe_breakdown"]["writes_served"]
    assert sum(added["hits_added_by_effect"].values()) == added["hits_added"]


# ---------------------------------------------------------------------------
# (4) The committed JSON is reproducible and pinned to its corpus
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not SCRIPT.exists(), reason="ladder producer script missing")
def test_rebuild_reproduces_the_committed_json(tmp_path: Path) -> None:
    out = tmp_path / "soundness_ladder_results.json"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, (
        f"rebuild failed (exit {completed.returncode}): {completed.stderr.strip()}\n"
        f"regenerate with: {REGENERATE}"
    )

    committed_bytes = COMMITTED.read_bytes()
    fresh_bytes = out.read_bytes()
    difference = _first_difference(
        json.loads(committed_bytes), json.loads(fresh_bytes)
    )
    assert fresh_bytes == committed_bytes, (
        "rebuild is not byte-identical to the committed file: "
        + (difference or "payloads parse equal, so only formatting or key order differs")
        + f"; if the change is intended, regenerate with: {REGENERATE}"
    )


@pytest.mark.skipif(not CORPUS.exists(), reason="tau-bench corpus missing")
def test_results_pin_the_corpus_they_were_measured_on() -> None:
    payload = _payload()
    digest = hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    provenance = payload["provenance"]["corpus"]

    assert payload["corpus_sha256"] == digest
    assert provenance["sha256"] == digest
    assert provenance["matches_expected"] is True
    assert provenance["calls"] == ALL_CALLS
    assert provenance["first_trial_only"] is True
    assert payload["checks"]["corpus_matches_committed"]["ok"] is True
    assert all(entry["ok"] for entry in payload["checks"].values()), (
        "the committed payload reports a failed invariant: "
        f"{sorted(name for name, entry in payload['checks'].items() if not entry['ok'])}"
    )
