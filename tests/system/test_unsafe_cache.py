"""Unsafe-cache baselines: the ordering the four cache policies must exhibit.

``unsafe_cache_baselines.py`` replays one committed corpus through four
run-scope caches -- ``blind``, ``blind+invalidate``, ``semantic-ish`` and
``typed`` -- and counts the hits each one serves that are wrong. The sizes are
the paper's business; what this module pins is the structure that makes the
comparison readable:

  1. the blind policy is the unsafe ceiling of the gated policies: adding a
     safety gate to the blind key (invalidation, effect typing) may only remove
     unsafe hits, never add them;
  2. the typed policy serves no unsafe hit of any class, while still hitting;
  3. every policy's unsafe total is exactly its per-class split
     (writes + stale + cross-tenant), because each hit is counted once;
  4. hit rates are shares in [0, 1] computed from the policy's own hits and
     admitted calls, and they order blind >= typed as the design predicts;
  5. re-running the script over the corpus reproduces the committed file byte
     for byte.

``semantic-ish`` is excluded from (1) on purpose, and the exclusion is asserted
rather than assumed: its key normalises arguments and drops identity
(``user_id``/``tenant``), so its key space is *coarser* than the blind key and
can merge two tenants' requests into one memo. In the committed replay it
therefore serves more hits than blind, and more unsafe ones. A policy cannot be
bounded by a policy whose key space it does not share, so the ceiling is
asserted where it applies and the merge property is asserted here.

The payload carries no ``generated_at``-style field: ``provenance`` pins content
hashes (script, identity module, corpus), not a clock. Reproduction is therefore
compared byte for byte with no field excused. Regenerate the committed artifact
in place with::

    ./venv/bin/python examples/benchmarks/unsafe_cache_baselines.py \\
        --output examples/benchmarks/unsafe_cache_results.json

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
SCRIPT = REPO_ROOT / "examples" / "benchmarks" / "unsafe_cache_baselines.py"
COMMITTED = REPO_ROOT / "examples" / "benchmarks" / "unsafe_cache_results.json"
CORPUS = REPO_ROOT / "examples" / "real_traces" / "tau_bench_historical.jsonl"

BREAKDOWN_FIELDS = ("writes_served", "stale_reads", "cross_tenant")
# Fields a rerun may legitimately stamp; this payload has none, which is why the
# reproduction test can demand exact bytes. The names match the other audit
# tests so a future producer that starts stamping is handled in one place.
CLOCK_FIELDS = ("generated_at", "generated", "timestamp", "created_at",
                "created", "ran_at", "elapsed_s")

BLIND = "blind"
TYPED = "typed"
# The only policy whose key is coarser than the blind key: it drops identity and
# normalises arguments, so it is not bounded by the blind replay (see module doc).
COARSER_POLICY = "semantic-ish"

REGENERATE = (
    "./venv/bin/python examples/benchmarks/unsafe_cache_baselines.py "
    "--output examples/benchmarks/unsafe_cache_results.json"
)

pytestmark = pytest.mark.skipif(
    not COMMITTED.exists(),
    reason=f"committed unsafe-cache results missing; regenerate with: {REGENERATE}",
)


def _payload() -> dict[str, Any]:
    return json.loads(COMMITTED.read_text(encoding="utf-8"))


def _policies() -> dict[str, dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for policy in _payload()["policies"]:
        assert policy["name"] not in by_name, f"duplicate policy name {policy['name']!r}"
        by_name[policy["name"]] = policy
    return by_name


def _policy(name: str) -> dict[str, Any]:
    policies = _policies()
    assert name in policies, (
        f"policy {name!r} missing; committed names are {sorted(policies)}")
    return policies[name]


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


def _corpus_first_trial_counts() -> tuple[int, int]:
    """(records, calls) for the slice the policies replay: first trial of each task."""
    records = calls = 0
    for line in CORPUS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("trial") not in (None, 0):
            continue
        records += 1
        calls += sum(len(turn) for turn in record.get("turns", []))
    return records, calls


# ---------------------------------------------------------------------------
# (1) The blind policy bounds the policies that add a gate to it
# ---------------------------------------------------------------------------
def test_blind_policy_is_the_unsafe_ceiling_of_the_gated_policies() -> None:
    policies = _policies()
    blind = policies[BLIND]
    assert blind["unsafe_hits"] > 0, "a zero ceiling would make the ordering vacuous"

    gated = {name: policy for name, policy in policies.items()
             if name not in (BLIND, COARSER_POLICY)}
    assert gated, "no gated policy to compare against"
    for name, policy in gated.items():
        assert blind["unsafe_hits"] >= policy["unsafe_hits"], (
            f"{name!r} serves {policy['unsafe_hits']} unsafe hits, more than blind's "
            f"{blind['unsafe_hits']}, although it adds a safety gate (invalidation "
            "or effect typing) to the blind baseline"
        )


def test_a_coarser_key_is_not_bounded_by_the_blind_policy() -> None:
    """The exclusion in the ceiling test is a property, not a convenience.

    ``semantic-ish`` merges calls the blind key keeps apart (it drops identity and
    normalises arguments), so its hit set must contain the blind key's; a
    violation means the two replays no longer line up or the key stopped being a
    refinement of the published recipe.
    """
    policies = _policies()
    assert COARSER_POLICY in policies
    blind, coarser = policies[BLIND], policies[COARSER_POLICY]
    assert coarser["hits"] >= blind["hits"], (
        "a coarser key can only merge requests, so it cannot hit fewer calls than "
        "the blind key it refines; the two replays disagree"
    )


# ---------------------------------------------------------------------------
# (2) The typed policy is safe in every class
# ---------------------------------------------------------------------------
def test_typed_policy_serves_no_unsafe_hit_of_any_class() -> None:
    typed = _policy(TYPED)

    assert typed["unsafe_hits"] == 0
    for field in BREAKDOWN_FIELDS:
        assert typed["unsafe_breakdown"][field] == 0, f"typed still serves {field}"
    assert set(typed["unsafe_by_class_overlapping"].values()) == {0}
    assert typed["memoises_writes"] is False

    # Zero unsafe hits earned by refusing every call would prove nothing.
    assert typed["hits"] > 0, "typed must still serve reads to be a safe cache"


# ---------------------------------------------------------------------------
# (3) Totals and the per-class split agree
# ---------------------------------------------------------------------------
def test_every_policy_total_equals_its_per_class_split() -> None:
    for policy in _payload()["policies"]:
        parts = policy["unsafe_breakdown"]
        assert set(parts) == set(BREAKDOWN_FIELDS), f"{policy['name']}: {sorted(parts)}"
        assert all(isinstance(value, int) and not isinstance(value, bool)
                   for value in parts.values()), "unsafe counts are integers"
        assert all(value >= 0 for value in parts.values())
        assert sum(parts.values()) == policy["unsafe_hits"], (
            f"{policy['name']}: {parts} sums to {sum(parts.values())}, "
            f"not {policy['unsafe_hits']}"
        )
        assert policy["unsafe_hits"] <= policy["hits"], (
            f"{policy['name']}: unsafe hits cannot exceed hits")


# ---------------------------------------------------------------------------
# (4) Hit rates are shares, and blind >= typed
# ---------------------------------------------------------------------------
def test_hit_rates_are_shares_and_order_blind_above_typed() -> None:
    policies = _payload()["policies"]
    for policy in policies:
        rate = policy["hit_rate"]
        assert isinstance(rate, (int, float)) and not isinstance(rate, bool)
        assert 0.0 <= rate <= 1.0, f"{policy['name']}: hit_rate {rate} is not a share"
        assert 0.0 <= policy["hit_rate_of_all_calls"] <= 1.0
        assert policy["hits"] <= policy["cacheable_calls"] <= policy["calls_total"]
        assert rate == pytest.approx(
            policy["hits"] / policy["cacheable_calls"], abs=5e-5
        ), f"{policy['name']}: hit_rate must be hits over admitted calls"

    assert _policy(BLIND)["hit_rate"] >= _policy(TYPED)["hit_rate"], (
        "the blind policy is the higher-hit, less safe end of the comparison"
    )


# ---------------------------------------------------------------------------
# (5) The committed JSON is reproducible
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not SCRIPT.exists(), reason="unsafe-cache producer script missing")
def test_rebuild_reproduces_the_committed_json(tmp_path: Path) -> None:
    out = tmp_path / "unsafe_cache_results.json"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert completed.returncode == 0, (
        f"rebuild failed (exit {completed.returncode}): {completed.stderr.strip()}\n"
        f"regenerate with: {REGENERATE}"
    )

    committed_bytes = COMMITTED.read_bytes()
    fresh_bytes = out.read_bytes()
    committed = json.loads(committed_bytes)
    fresh = json.loads(fresh_bytes)

    clock = sorted(field for field in CLOCK_FIELDS if field in committed or field in fresh)
    if clock:
        # A stamp is the one thing a rerun may change; drop it and compare the rest.
        for field in clock:
            committed.pop(field, None)
            fresh.pop(field, None)
        difference = _first_difference(committed, fresh)
        assert difference is None, (
            f"rebuild differs from the committed payload at {difference}; "
            f"regenerate with: {REGENERATE}"
        )
        return

    difference = _first_difference(committed, fresh)
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
    assert provenance["first_trial_only"] is True

    records, calls = _corpus_first_trial_counts()
    assert provenance["records"] == records
    assert provenance["calls"] == calls
    assert all(policy["calls_total"] == calls for policy in payload["policies"]), (
        "every policy must replay the same corpus"
    )
