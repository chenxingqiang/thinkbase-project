"""Does the committed absorption rate survive a task-level holdout?

``compound_binding`` mines and measures on the same 325 tasks, so its 12.53% is
a resubstitution number. ``compound_holdout`` mines compounds on one parity half
of each domain's tasks and measures absorption on the other, in both directions.
These tests check the properties the resulting claim rests on, not the report's
arithmetic: the halves are a partition of the corpus, the recorded mining half
is the half that was actually mined, the holdout rate does not exceed the
in-sample rate (if it does, the claim needs rewording), the mirror direction is
actually run and reported, and the committed JSON is reproducible apart from
provenance.

The corpus is the authority for the split: half membership is recomputed here
from the recorded task ids, and the committed counts are only accepted when
they match that recomputation. The one thing the report may say about itself is
provenance -- git state and timestamps -- so determinism compares everything
else.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "examples" / "benchmarks" / "compound_holdout.py"
COMMITTED = REPO_ROOT / "examples" / "benchmarks" / "compound_holdout_results.json"
CORPUS = REPO_ROOT / "examples" / "real_traces" / "tau_bench_observed.jsonl"
BASELINE = REPO_ROOT / "examples" / "benchmarks" / "compound_binding_results.json"
DOMAINS = ("retail", "airline")

# A run may identify its checkout, clock and machine; it may not change the
# measured payload. Provenance is the only place those fields belong.
TIMESTAMP_KEYS = ("generated_at", "generated", "timestamp", "created_at",
                  "created", "ran_at", "elapsed_s")


def _script():
    name = "compound_holdout"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _committed() -> Dict[str, Any]:
    return json.loads(COMMITTED.read_text(encoding="utf-8"))


def _corpus_by_domain() -> Dict[str, List[Dict[str, Any]]]:
    ho = _script()
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for line in CORPUS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            grouped.setdefault(ho.domain_of(row), []).append(row)
    return grouped


def _blocks(payload: Dict[str, Any]
            ) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """The pooled split and every domain split, under the label each reports."""
    yield "pooled", payload["pooled"]
    for name in sorted(payload["by_domain"]):
        yield name, payload["by_domain"][name]


def _directions(payload: Dict[str, Any]
                ) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    for label, block in _blocks(payload):
        for tag in ("A_to_B", "B_to_A"):
            yield label, tag, block[tag]


def _payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Everything a rerun must reproduce exactly: the payload without provenance."""
    return {key: value for key, value in payload.items()
            if key != "provenance" and key not in TIMESTAMP_KEYS}


def _first_difference(left: Any, right: Any,
                      path: str = "payload") -> Optional[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return (f"{path}: keys differ; only committed "
                    f"{sorted(set(left) - set(right))}, only fresh "
                    f"{sorted(set(right) - set(left))}")
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


def test_halves_partition_each_domains_task_ids_exactly_once() -> None:
    """Every task id belongs to one half; the halves' union is the domain."""
    payload = _committed()
    ho = _script()
    grouped = _corpus_by_domain()

    assert payload["corpus"]["domains"] == sorted(grouped) == sorted(DOMAINS)
    assert payload["corpus"]["tasks"] == sum(len(rows) for rows in grouped.values())
    assert payload["corpus"]["dispatches"] == sum(
        len(row["calls"]) for rows in grouped.values() for row in rows)
    assert "parity" in payload["split"]["rule"]

    for domain, rows in grouped.items():
        ids = {str(row["task_id"]) for row in rows}
        even = {task_id for task_id in ids if int(task_id) % 2 == 0}
        odd = ids - even

        # The partition, checked on task ids rather than on sizes: disjoint and
        # total. Airline and retail reuse the same numbers by design, so this is
        # asserted within a domain, never across domains.
        assert not (even & odd)
        assert even | odd == ids

        rows_a = [row for row in rows if str(row["task_id"]) in even]
        rows_b = [row for row in rows if str(row["task_id"]) in odd]
        # Every recorded trajectory, including both model runs of a task, lands
        # on exactly one side.
        assert len(rows_a) + len(rows_b) == len(rows)
        assert len({str(row["task_id"]) for row in rows_a}) == len(even)
        assert len({str(row["task_id"]) for row in rows_b}) == len(odd)

        import_halves_a, import_halves_b = ho.split_parity(rows)
        assert {str(task["task_id"]) for task in import_halves_a} == even
        assert {str(task["task_id"]) for task in import_halves_b} == odd

        block = payload["by_domain"][domain]
        assert block["tasks"] == len(rows), (
            f"{domain}: report counts {block['tasks']} trajectories, the corpus "
            f"has {len(rows)}")
        assert (block["halves"]["A"]["tasks"],
                block["halves"]["B"]["tasks"]) == (len(rows_a), len(rows_b)), (
            f"{domain}: recorded halves are "
            f"{block['halves']['A']['tasks']}/{block['halves']['B']['tasks']} "
            f"trajectories, the corpus partition is "
            f"{len(rows_a)}/{len(rows_b)}")
        assert block["halves"]["A"]["dispatches"] == sum(
            len(row["calls"]) for row in rows_a), (
            f"{domain}: half A dispatch count does not match the corpus partition")
        assert block["halves"]["B"]["dispatches"] == sum(
            len(row["calls"]) for row in rows_b), (
            f"{domain}: half B dispatch count does not match the corpus partition")
        assert (block["halves"]["A"]["dispatches"]
                + block["halves"]["B"]["dispatches"]) == block["dispatches"]

    # The pooled split is the domain splits laid side by side, not a third one.
    pooled = payload["pooled"]
    assert pooled["tasks"] == sum(block["tasks"] for name, block in _blocks(payload)
                                  if name != "pooled"), (
        "the pooled split is not the sum of the domain splits")
    for half in ("A", "B"):
        assert pooled["halves"][half]["tasks"] == sum(
            block["halves"][half]["tasks"] for name, block in _blocks(payload)
            if name != "pooled"), (
            f"pooled half {half} task count is not the domain sum")
        assert pooled["halves"][half]["dispatches"] == sum(
            block["halves"][half]["dispatches"] for name, block in _blocks(payload)
            if name != "pooled"), (
            f"pooled half {half} dispatch count is not the domain sum")
    assert (pooled["halves"]["A"]["dispatches"]
            + pooled["halves"]["B"]["dispatches"]) == pooled["dispatches"]

    partition_check = {check["name"]: check["ok"] for check in payload["checks"]}
    assert partition_check.get("halves_partition_the_corpus") is True


def test_compounds_are_mined_on_the_recorded_half_not_the_measured_half() -> None:
    """The report names its mining half, and the record is that half's work."""
    payload = _committed()
    ho = _script()
    max_k = payload["split"]["max_k"]
    min_support = payload["split"]["min_support"]

    grouped = _corpus_by_domain()
    tasks_by_label = {"pooled": [row for rows in grouped.values() for row in rows],
                      **grouped}

    for label, block in _blocks(payload):
        tasks = tasks_by_label[label]
        halves = dict(zip(("A", "B"), ho.split_parity(tasks)))
        for tag in ("A_to_B", "B_to_A"):
            cell = block[tag]
            learn_half, measure_half = tag.split("_to_")

            # The direction records which half it mined and which it measured,
            # and they are not the same half.
            assert cell["learn"]["half"] == learn_half, (
                f"{label} {tag}: learn half is {cell['learn']['half']!r}, "
                f"expected {learn_half!r}")
            assert cell["measure"]["half"] == measure_half, (
                f"{label} {tag}: measure half is {cell['measure']['half']!r}, "
                f"expected {measure_half!r}")
            assert cell["learn"]["half"] != cell["measure"]["half"]
            assert cell["learn"]["tasks"] == len(halves[learn_half])
            assert cell["measure"]["tasks"] == len(halves[measure_half])
            assert cell["learn"]["tasks"] + cell["measure"]["tasks"] == block["tasks"]
            assert (cell["learn"]["dispatches"]
                    + cell["measure"]["dispatches"]) == block["dispatches"]

            # Re-mine both halves from the corpus: the recorded count is the
            # learn half's repertoire, so the compounds were not discovered on
            # the half they are measured against.
            mined_learn = ho.CB.mine_compounds(
                halves[learn_half], max_k=max_k, min_support=min_support)
            mined_measure = ho.CB.mine_compounds(
                halves[measure_half], max_k=max_k, min_support=min_support)
            reported = cell["compounds"]["mined_on_learn"]
            assert reported == len(mined_learn), (
                f"{label} {tag}: report says {reported} compounds were mined on "
                f"half {learn_half}, but that half mines {len(mined_learn)}; the "
                "mining half it records is not the half it used")
            assert len(mined_learn) != len(mined_measure), (
                f"{label} {tag}: both halves mine {len(mined_learn)} compounds, so "
                "the recorded count cannot establish which half was mined; the "
                "report needs to record the compound identities, not just counts")
            assert sum(bucket["mined"] for bucket in cell["by_k"].values()) == reported

            # Everything downstream is bounded by that measured repertoire.
            assert (cell["compounds"]["verified_transferred_to_measure"]
                    <= cell["compounds"]["verified_exposed_in_measure"]
                    <= cell["compounds"]["verified_on_learn"] <= reported), (
                f"{label} {tag}: the verification funnel is not a subset chain")


def test_holdout_absorption_does_not_exceed_the_in_sample_rate() -> None:
    """If it does, the paper's claim needs rewording, and this test says so."""
    payload = _committed()
    verdict = payload["verdict"]
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

    # The in-sample number the holdout is compared against is the committed
    # resubstitution rate, not a figure retyped into the report.
    in_sample = baseline["reduction_if_substituted"]["reduction"]
    assert verdict["committed_rate"] == in_sample
    assert payload["baseline"]["in_sample"]["rate"] == in_sample
    assert payload["reproduction"]["matches"] is True

    forward = payload["pooled"]["A_to_B"]["absorption"]["measure_holdout"]
    reverse = payload["pooled"]["B_to_A"]["absorption"]["measure_holdout"]
    removed = forward["dispatches_removed"] + reverse["dispatches_removed"]
    dispatches = forward["dispatches"] + reverse["dispatches"]
    assert dispatches == payload["corpus"]["dispatches"] == verdict["dispatches"]
    assert verdict["holdout_dispatches_removed"] == removed
    assert verdict["holdout_rate"] == round(removed / dispatches, 4)
    assert verdict["holdout_rate"] <= verdict["committed_rate"], (
        f"pooled holdout absorption is {verdict['holdout_rate']:.2%} "
        f"({removed}/{dispatches}), at or above the in-sample "
        f"{verdict['committed_rate']:.2%} it is supposed to undercut: the claim "
        "that the in-sample figure overstates the out-of-sample rate is false as "
        "stated, and the claim needs rewording")

    # The same comparison per direction and per domain: a pooled number cannot
    # hide a direction that got better out of sample.
    for label, tag, cell in _directions(payload):
        holdout = cell["absorption"]["measure_holdout"]
        measured_in_sample = cell["absorption"]["learn_in_sample"]
        assert holdout["dispatches"] == cell["measure"]["dispatches"]
        assert measured_in_sample["dispatches"] == cell["learn"]["dispatches"]
        assert holdout["rate"] == round(
            holdout["dispatches_removed"] / holdout["dispatches"], 4)
        assert measured_in_sample["rate"] == round(
            measured_in_sample["dispatches_removed"]
            / measured_in_sample["dispatches"], 4)
        assert holdout["rate"] <= measured_in_sample["rate"], (
            f"{label} {tag}: holdout absorption {holdout['rate']:.2%} on half "
            f"{cell['measure']['half']} is above the in-sample "
            f"{measured_in_sample['rate']:.2%} on half {cell['learn']['half']}; "
            "the claim needs rewording, because learning and measuring on the "
            "same half did not overstate absorption here")


def test_symmetry_check_runs_and_reports_both_directions() -> None:
    """A to B and B to A are both run, reported, and cover the corpus once."""
    payload = _committed()
    verdict = payload["verdict"]
    assert {"A_to_B", "B_to_A"} <= set(verdict["directional"])

    for label, block in _blocks(payload):
        assert {"A_to_B", "B_to_A"} <= set(block)
        forward = block["A_to_B"]
        reverse = block["B_to_A"]

        assert (forward["learn"]["half"], forward["measure"]["half"]) == ("A", "B"), (
            f"{label}: A_to_B learns on {forward['learn']['half']!r} and measures "
            f"on {forward['measure']['half']!r}, so the symmetry check is not "
            "running the direction it claims")
        assert (reverse["learn"]["half"], reverse["measure"]["half"]) == ("B", "A"), (
            f"{label}: B_to_A learns on {reverse['learn']['half']!r} and measures "
            f"on {reverse['measure']['half']!r}, so the mirror direction is "
            "missing or mislabelled")
        assert forward["learn"]["tasks"] == block["halves"]["A"]["tasks"]
        assert forward["measure"]["tasks"] == block["halves"]["B"]["tasks"]
        assert reverse["learn"]["tasks"] == block["halves"]["B"]["tasks"]
        assert reverse["measure"]["tasks"] == block["halves"]["A"]["tasks"]

        # Both directions report their own absorption denominator, and the two
        # denominators are the two halves, not the same half twice.
        assert forward["measure"]["dispatches"] == block["halves"]["B"]["dispatches"]
        assert reverse["measure"]["dispatches"] == block["halves"]["A"]["dispatches"]
        assert (forward["measure"]["dispatches"]
                + reverse["measure"]["dispatches"]) == block["dispatches"]
        # Each half's in-sample figure is its own, and equals the other
        # direction's measured denominator, so the symmetry check is not
        # comparing two differently-measured numbers.
        assert (forward["absorption"]["learn_in_sample"]["dispatches"]
                == reverse["absorption"]["measure_holdout"]["dispatches"])
        assert (reverse["absorption"]["learn_in_sample"]["dispatches"]
                == forward["absorption"]["measure_holdout"]["dispatches"])

        for cell in (forward, reverse):
            holdout = cell["absorption"]["measure_holdout"]
            assert holdout["dispatches"] > 0
            assert 0.0 <= holdout["rate"] <= 1.0
            assert holdout["dispatches_removed"] <= holdout["dispatches"]

        if label == "pooled":
            assert verdict["directional"]["A_to_B"] == forward[
                "absorption"]["measure_holdout"]["rate"]
            assert verdict["directional"]["B_to_A"] == reverse[
                "absorption"]["measure_holdout"]["rate"]
            spread = round(
                abs(verdict["directional"]["A_to_B"]
                    - verdict["directional"]["B_to_A"]) * 100, 2)
            assert verdict["directional"]["spread_points"] == spread


def test_rerun_reproduces_the_payload_apart_from_provenance(tmp_path) -> None:
    """The committed JSON is a deterministic function of the corpus."""
    output = tmp_path / "compound_holdout_results.json"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--corpus", str(CORPUS),
         "--baseline", str(BASELINE), "--output", str(output)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=25)
    assert completed.returncode == 0, (
        "the committed results cannot be regenerated; the script exited "
        f"{completed.returncode}\n{completed.stderr[-2000:]}")

    fresh = json.loads(output.read_text(encoding="utf-8"))
    committed = _committed()
    difference = _first_difference(_payload(committed), _payload(fresh))
    assert difference is None, (
        f"a fresh run differs from the committed JSON at {difference}; only "
        "provenance (git state, timestamps) may drift between runs")

    # Provenance is allowed to vary, but it must still pin the inputs the
    # numbers came from -- otherwise the determinism check proves nothing.
    assert fresh["provenance"]["corpus_sha256"] == hashlib.sha256(
        CORPUS.read_bytes()).hexdigest()
    assert fresh["provenance"]["corpus_bytes"] == CORPUS.stat().st_size
    assert fresh["provenance"]["baseline_sha256"] == hashlib.sha256(
        BASELINE.read_bytes()).hexdigest()
    assert all(check["ok"] for check in fresh["checks"]), (
        "the regenerated report's own checks fail: "
        f"{[check['name'] for check in fresh['checks'] if not check['ok']]}")
