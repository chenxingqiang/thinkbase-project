"""Cross-model transfer: the wiring is what has to travel, not just the calls.

The experiment's claim is conditional -- "given a compound that reproduces its
own model's traces, does it reproduce another model's?" -- so each part of the
conditional gets an invariant test: eligibility rejects a spec that fails its own
source, a parameter-only compound transfers trivially while an observation-wired
one does not, failures are classified rather than counted as a single bucket,
cross-domain exposure is reported as a denominator instead of a rate, and the
committed results JSON is reproducible from the committed corpus.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "examples" / "benchmarks" / "compound_transfer.py"
CORPUS = REPO_ROOT / "examples" / "real_traces" / "tau_bench_observed.jsonl"
COMMITTED = REPO_ROOT / "examples" / "benchmarks" / "compound_transfer_results.json"


def _module():
    name = "compound_transfer"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FIND_GET = (("find_user", ("name",)), ("get_user", ("user_id",)))
LIST_LIST = (("list_types", ("category",)), ("list_types", ("category",)))


def _task(task_id, source, calls, reward=1.0):
    return {"source": source, "task_id": task_id, "trial": 0, "reward": reward,
            "calls": [[tool, args, obs] for tool, args, obs in calls]}


def _corpus(tmp_path: Path, extra=None) -> Path:
    """Two retail models, each with one wired and one parameter-only compound.

    Every task holds exactly one compound so the mined repertoire is exactly
    ``FIND_GET`` and ``LIST_LIST`` -- no accidental bigram across the boundary.
    """
    rows = [
        # learner: the user id is read out of the find_user result
        _task("1", "gpt-4o-retail", [
            ("find_user", {"name": "a"}, '{"user": {"id": "u1"}}'),
            ("get_user", {"user_id": "u1"}, '{"orders": ["#A"]}'),
        ]),
        _task("2", "gpt-4o-retail", [
            ("find_user", {"name": "b"}, '{"user": {"id": "u2"}}'),
            ("get_user", {"user_id": "u2"}, '{"orders": ["#B"]}'),
        ]),
        # learner: the second category is a fresh value, not a result read
        _task("3", "gpt-4o-retail", [
            ("list_types", {"category": "x"}, "ok"),
            ("list_types", {"category": "y"}, "ok"),
        ]),
        _task("4", "gpt-4o-retail", [
            ("list_types", {"category": "m"}, "ok"),
            ("list_types", {"category": "n"}, "ok"),
        ]),
        # tester: same task ids, a different model, different entities
        _task("1", "sonnet-35-new-retail", [
            ("find_user", {"name": "c"}, '{"user": {"id": "u3"}, "previous": "u7"}'),
            ("get_user", {"user_id": "u7"}, '{"orders": ["#C"]}'),
        ]),
        _task("2", "sonnet-35-new-retail", [
            ("find_user", {"name": "d"}, '{"user": {"id": "u4"}}'),
            ("get_user", {"user_id": "u8"}, '{"orders": ["#D"]}'),
        ]),
        _task("3", "sonnet-35-new-retail", [
            ("list_types", {"category": "p"}, "ok"),
            ("list_types", {"category": "q"}, "ok"),
        ]),
        _task("4", "sonnet-35-new-retail", [
            ("list_types", {"category": "r"}, "ok"),
            ("list_types", {"category": "s"}, "ok"),
        ]),
    ]
    if extra:
        rows.extend(extra)
    path = tmp_path / "toy.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_eligibility_rejects_a_spec_that_fails_its_own_source() -> None:
    ct = _module()
    broken = [
        _task("1", "m1", [
            ("find_user", {"name": "a"}, '{"user": {"id": "u1"}}'),
            ("get_user", {"user_id": "u1"}, "ok"),
        ]),
        _task("2", "m1", [
            ("find_user", {"name": "b"}, '{"accounts": [{"id": "u2"}]}'),
            ("get_user", {"user_id": "u2"}, "ok"),
        ]),
    ]
    eligible, funnel, excluded = ct.eligible_specs(broken, max_k=2, min_support=2)
    assert FIND_GET not in eligible
    assert FIND_GET in excluded
    assert funnel["signature_consistent"] == 1      # one spec, same wiring
    assert funnel["in_source_replay"] == 0          # ... but it does not replay
    assert funnel["recurring"] >= 1


def test_parameter_only_transfers_while_the_wired_compound_does_not(tmp_path) -> None:
    ct = _module()
    path = _corpus(tmp_path)
    report = ct.run(path, max_k=2, min_support=2)
    cell = report["cross_model"]["cells"]["retail"]
    wiring = cell["wiring"]
    assert cell["eligible"] == 2
    assert cell["exposed"] == 2
    # the repeated call with a free category is reproducible ...
    assert wiring["parameter_only"]["transferred"] == 1
    assert wiring["parameter_only"]["exposed"] == 1
    # ... the compound that reads a result is not: the other model read "u7"
    # where the learned path points at "u3"
    wired = wiring["observation_wired_all_specs"]
    assert wired["exposed"] == 1
    assert wired["transferred"] == 0
    assert wired["occurrences"]["tested"] == 2
    assert wired["occurrences"]["exact"] == 0
    assert cell["transferred"] == 1
    assert cell["transfer_rate"] == 0.5


def test_failure_modes_split_ambiguity_from_an_outside_value(tmp_path) -> None:
    ct = _module()
    path = _corpus(tmp_path)
    report = ct.run(path, max_k=2, min_support=2)
    cell = report["cross_model"]["cells"]["retail"]
    # task 1: the value is present in the result, at another path;
    # task 2: the value is nowhere in the result
    assert cell["failures"] == {"ambiguous_read": 1, "external_value": 1}


def test_specs_get_their_free_slots_from_the_window_they_run_on(tmp_path) -> None:
    ct = _module()
    path = _corpus(tmp_path)
    report = ct.run(path, max_k=2, min_support=2)
    cell = report["cross_model"]["cells"]["retail"]
    # every parameter-only window replays exactly: all later arguments are
    # supplied by the caller, so this is a consistency check on the harness
    plain = cell["wiring"]["parameter_only"]
    occurrences = plain["occurrences"]
    assert occurrences["tested"] == 2 and occurrences["exact"] == 2
    assert occurrences["exact_share"] == 1.0 and occurrences["failed"] == 0


def test_cross_domain_reports_a_denominator_not_a_rate(tmp_path) -> None:
    ct = _module()
    extra = [
        _task("1", "gpt-4o-airline", [
            ("book_reservation", {"flight": "F1"}, "ok"),
            ("send_certificate", {"reservation": "R1"}, "ok"),
        ]),
        _task("2", "sonnet-35-new-airline", [
            ("book_reservation", {"flight": "F2"}, "ok"),
            ("send_certificate", {"reservation": "R2"}, "ok"),
        ]),
    ]
    path = _corpus(tmp_path, extra)
    report = ct.run(path, max_k=2, min_support=2)
    domain = report["cross_domain"]
    assert domain["eligible_exposed_in_tester"] == 0
    assert domain["shared_atomic_templates"] == 0
    assert "transfer_rate" not in domain
    text = ct.summarise(report)
    assert "cross-domain retail->airline" in text


def test_halves_are_disjoint_and_cover_every_task() -> None:
    ct = _module()
    tasks = [_task(str(index), "m", []) for index in range(7)]
    first, second = ct.split_halves(tasks)
    assert [task["task_id"] for task in first] == ["0", "2", "4", "6"]
    assert [task["task_id"] for task in second] == ["1", "3", "5"]
    again = ct.split_halves(tasks)
    assert again == (first, second)


def test_run_is_deterministic_on_the_committed_corpus() -> None:
    ct = _module()
    first = ct.run(CORPUS)
    second = ct.run(CORPUS)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_sources_are_disjoint_and_partition_the_corpus() -> None:
    """A transfer cell must never test on a trajectory it learned from."""
    ct = _module()
    tasks = ct.CB.load_tasks(CORPUS)
    sources = ct.load_sources(CORPUS)
    # every record belongs to exactly one source; the partition is total
    assert sum(len(rows) for rows in sources.values()) == len(tasks)
    assert len({task["source"] for task in tasks}) == len(sources)
    # the two members of every cross-model pair are different sources, and no
    # trajectory (source, task, trial) appears on both sides
    for learner, tester in ((ct.LEARNER, ct.TESTER), (ct.TESTER, ct.LEARNER),
                            (ct.LEARNER_AIR, ct.TESTER_AIR),
                            (ct.TESTER_AIR, ct.LEARNER_AIR)):
        assert learner != tester
        left = {(row["source"], row["task_id"], row["trial"]) for row in sources[learner]}
        right = {(row["source"], row["task_id"], row["trial"]) for row in sources[tester]}
        assert not (left & right)
    # the same task id recurs across models by design (same tau-bench tasks), so
    # disjointness is over trajectories, never over task ids
    assert sources[ct.LEARNER][0]["task_id"] is not None
    report = ct.run(CORPUS)
    for name, cell in report["cross_model"]["cells"].items():
        assert cell["learner_tasks"] and cell["tester_tasks"]
        assert cell["label"].split(" -> ")[0] != cell["label"].split(" -> ")[1]


def test_transferred_never_exceeds_eligible() -> None:
    """Every numerator is bounded by its own denominator, in every subset."""
    ct = _module()
    report = ct.run(CORPUS)
    rows = list(report["cross_model"]["cells"].values())
    for cell in rows:
        assert cell["exposed"] <= cell["eligible"]
        assert cell["transferred"] <= cell["exposed"]
        assert cell["transferred"] <= cell["eligible"]
        assert cell["faithful"] <= cell["transferred"]
        assert cell["transfer_rate"] == round(cell["transferred"] / cell["exposed"], 4)
        occ = cell["occurrences"]
        assert occ["exact"] <= occ["tested"]
        subset = cell["wiring"]
        for key in ("observation_wired", "parameter_only",
                    "observation_wired_all_specs"):
            assert subset[key]["transferred"] <= subset[key]["eligible"]
            assert subset[key]["exposed"] <= subset[key]["eligible"]
    for name, pooled in report["controls"]["pooled"].items():
        assert pooled["transferred"] <= pooled["exposed"] <= sum(
            cell["exposed"] for cell in rows)


def test_within_source_control_is_at_least_cross_source_transfer() -> None:
    """Varying the task instance must not cost more than varying the model."""
    ct = _module()
    report = ct.run(CORPUS)
    cells = report["cross_model"]["cells"]
    controlled = report["controls"]["per_direction"]
    assert controlled, "no direction has a within-source control"
    for direction, value in controlled.items():
        within = value["conditions"]["within_source"]
        # matched budget: the cross-source cell learned on the same H1 tasks
        cross = value["conditions"]["same_task"]
        assert within["transferred"] <= within["exposed"]
        assert within["transfer_rate"] >= cross["transfer_rate"], (
            f"{direction}: within-source {within['transfer_rate']} < "
            f"cross-source {cross['transfer_rate']}")
    # pooled, at the same learning budget: H1 -> H2 (same model) vs H1 -> other
    pooled = report["controls"]["pooled"]
    assert pooled["within_source"]["rate"] >= pooled["same_task"]["rate"]
    assert pooled["same_task"]["rate"] >= pooled["unseen_tasks"]["rate"]
    # and the full-corpus control is never below the full-corpus cross-model
    # cell for a direction it covers
    for direction in controlled:
        within = report["controls"]["per_direction"][direction]["conditions"]["within_source"]
        assert within["transfer_rate"] >= cells[direction]["transfer_rate"]


def test_path_drift_and_mismatch_partition_failed_replays() -> None:
    """Drift, mismatch and unbound slots account for every non-exact replay."""
    ct = _module()
    report = ct.run(CORPUS)
    for name, cell in report["cross_model"]["cells"].items():
        occ = cell["occurrences"]
        assert occ["failed"] == occ["tested"] - occ["exact"]
        assert occ["failed"] == (occ["path_drift"] + occ["mismatched"] +
                                occ["unbound_slots"])
        assert sum(cell["failures"].values()) == occ["failed"]
    text = ct.summarise(report)
    assert "path drift" in text and "mismatch" in text
    assert "within-source control" in text


def test_provenance_records_the_corpus_it_reads() -> None:
    ct = _module()
    report = ct.run(CORPUS)
    provenance = report["provenance"]
    import hashlib
    expected = hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    assert provenance["corpus_sha256"] == expected
    assert provenance["corpus_bytes"] == CORPUS.stat().st_size
    assert len(provenance["git_rev"]) == 40


def test_committed_results_match_a_fresh_run() -> None:
    ct = _module()
    committed = json.loads(COMMITTED.read_text(encoding="utf-8"))
    fresh = ct.run(CORPUS)
    for key in ("max_k", "min_support", "sources", "cross_domain"):
        assert fresh[key] == committed[key], f"{key} drift"
    for name, cell in committed["cross_model"]["cells"].items():
        regenerated = fresh["cross_model"]["cells"][name]
        for key in ("eligible", "exposed", "transferred", "faithful",
                    "occurrences", "slots", "failures", "wiring",
                    "excluded_by_in_source_replay"):
            assert regenerated[key] == cell[key], f"{name}.{key} drift"
    assert fresh["controls"]["pooled"] == committed["controls"]["pooled"]
