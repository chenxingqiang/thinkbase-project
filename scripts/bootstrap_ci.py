"""Percentile bootstrap confidence intervals for the ISA microbenchmarks.

The published microbench (``examples/benchmarks/isa_microbench_results.json``)
reports ``mean``/``stdev`` per configuration, but only a few configurations
keep the *per-seed raw values* those aggregates were computed from.  This
script recovers a 95% percentile bootstrap CI for every quantity whose raw
per-seed series is actually stored, and a paired difference CI where the
stored series is a per-seed difference of arms that share a seed base.

What is (and is not) computed
-----------------------------
* ``dynamic`` stores one record per seed (``seed`` 0..4) with
  ``original_len`` / ``optimized_len`` / ``shrink_ratio`` /
  ``optimized_at_iteration``; each field is resampled over its 5 per-seed
  values.
* ``paired_ablation.paired_interaction`` stores ``per_seed``, the per-seed
  ``(fusion+dce) - fusion_only - dce_only`` difference on the arms that share
  ``seed_base`` 2000.  It is a paired difference by construction, so it is
  reported as one.
* ``cache``, ``compiler``, ``ablation`` and the four ``paired_ablation`` arms
  store ``mean``/``stdev`` only.  No per-seed vector is committed for them, so
  no CI is emitted: recovering one would mean inventing values.  They are
  listed under ``not_computable`` with the reason.
* Configurations with ``n == 1`` never get a CI (a resample of one value has
  zero width and would misrepresent uncertainty).

Method
------
Percentile bootstrap: draw ``resamples`` samples of size ``n`` with
replacement, take the mean of each, and report the ``alpha/2`` and
``1 - alpha/2`` empirical quantiles (linear interpolation between order
statistics, numpy-free).  The point estimate is the plain arithmetic mean of
the observed values, not the bootstrap average.

Determinism
-----------
No wall clock, no unseeded RNG.  Each metric gets its own ``random.Random``
seeded from ``sha256("<base_seed>:<metric>")``, so CIs do not depend on
iteration order and two runs at the same commit are byte-identical.

Usage
-----
    ./venv/bin/python scripts/bootstrap_ci.py \\
        --input examples/benchmarks/isa_microbench_results.json \\
        --output examples/benchmarks/bootstrap_ci_results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "examples" / "benchmarks" / "isa_microbench_results.json"
DEFAULT_OUTPUT = ROOT / "examples" / "benchmarks" / "bootstrap_ci_results.json"

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20240601
DEFAULT_CONFIDENCE = 0.95

METHOD = "percentile_bootstrap"
SCHEMA = "bootstrap_ci/v1"


# --------------------------------------------------------------------------
# Bootstrap core
# --------------------------------------------------------------------------

def _is_number(value: Any) -> bool:
    """True for real numbers only: booleans and strings are not metrics."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Empirical quantile with linear interpolation between order statistics."""
    if not sorted_values:
        raise ValueError("quantile of empty sample")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _rng_for(metric: str, base_seed: int) -> random.Random:
    """Per-metric RNG so results are independent of iteration order."""
    digest = hashlib.sha256(f"{base_seed}:{metric}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    rng_seed: int = DEFAULT_SEED,
    metric: str = "",
) -> Dict[str, Any]:
    """Percentile bootstrap CI for the mean of ``values``.

    Returns ``{"mean", "ci_low", "ci_high", "n", "method"}``; raises for
    ``n < 2`` rather than emitting a zero-width interval.
    """
    data = [float(v) for v in values]
    n = len(data)
    if n < 2:
        raise ValueError("bootstrap CI needs at least 2 observations")
    if resamples < 1:
        raise ValueError("resamples must be >= 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly inside (0, 1)")

    rng = _rng_for(metric or "unnamed", rng_seed) if metric else random.Random(rng_seed)
    means = [
        statistics.fmean(rng.choices(data, k=n))
        for _ in range(resamples)
    ]
    means.sort()
    alpha = 1.0 - confidence
    return {
        "mean": statistics.fmean(data),
        "ci_low": _quantile(means, alpha / 2.0),
        "ci_high": _quantile(means, 1.0 - alpha / 2.0),
        "n": n,
        "method": METHOD,
    }


def paired_difference_ci(
    arm_a: Sequence[float],
    arm_b: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    rng_seed: int = DEFAULT_SEED,
    metric: str = "",
) -> Dict[str, Any]:
    """Percentile bootstrap CI for the mean of ``arm_a - arm_b``, paired by seed.

    The two arms must be aligned on the same seed order and have the same
    length; the resampling unit is the pair, not the observation.
    """
    a = [float(v) for v in arm_a]
    b = [float(v) for v in arm_b]
    if len(a) != len(b):
        raise ValueError(
            f"paired arms must have equal length (got {len(a)} and {len(b)})")
    diffs = [x - y for x, y in zip(a, b)]
    return bootstrap_mean_ci(
        diffs, resamples=resamples, confidence=confidence,
        rng_seed=rng_seed, metric=metric)


def _rounded(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Round floats for stable, diff-friendly JSON; keep the key order fixed."""
    return {
        "mean": round(entry["mean"], 6),
        "ci_low": round(entry["ci_low"], 6),
        "ci_high": round(entry["ci_high"], 6),
        "n": entry["n"],
        "method": entry["method"],
    }


# --------------------------------------------------------------------------
# Extraction from the committed results file
# --------------------------------------------------------------------------

def _per_seed_records(records: Iterable[Any]) -> Optional[Tuple[List[Any], List[Dict[str, Any]]]]:
    """If ``records`` is one row per seed, return (seeds, rows sorted by seed)."""
    materialized = list(records)
    rows = [r for r in materialized if isinstance(r, dict)]
    if not rows or len(rows) != len(materialized):
        return None
    if any("seed" not in r or not _is_number(r["seed"]) for r in rows):
        return None
    seeds = [r["seed"] for r in rows]
    if len(set(seeds)) != len(seeds):
        return None
    ordered = [r for _, r in sorted(zip(seeds, rows), key=lambda p: p[0])]
    return seeds, ordered


def collect_series(payload: Sequence[Dict[str, Any]]) -> Tuple[
        Dict[str, List[float]], Dict[str, List[float]], List[Dict[str, str]]]:
    """Split the file into per-seed metric series, paired series and skips.

    * plain per-seed fields (rows keyed by ``seed``) -> ``metrics``;
    * nested aggregates carrying a ``per_seed`` list -> ``per_seed_lists``
      (a nested list with a ``definition`` is a per-seed *difference* of arms,
      so the caller routes it to the paired section);
    * everything else -> ``skipped`` with a reason.
    """
    metrics: Dict[str, List[float]] = {}
    paired: Dict[str, List[float]] = {}
    skipped: List[Dict[str, str]] = []

    for experiment in payload:
        if not isinstance(experiment, dict):
            continue
        name = str(experiment.get("experiment", "unknown"))
        records = experiment.get("data")
        if not isinstance(records, list) or not records:
            skipped.append({"experiment": name, "reason": "no data records"})
            continue

        per_seed = _per_seed_records(records)
        if per_seed is not None:
            seeds, rows = per_seed
            fields = sorted({k for r in rows for k in r} - {"seed"})
            found = False
            for field in fields:
                values = [r[field] for r in rows]
                if not all(_is_number(v) for v in values):
                    continue  # e.g. replay_state_identical (bool)
                found = True
                if len(values) < 2:
                    skipped.append({
                        "experiment": name,
                        "metric": f"{name}.{field}",
                        "reason": f"n={len(values)} (1); no CI emitted",
                    })
                    continue
                metrics[f"{name}.{field}"] = [float(v) for v in values]
            if not found:
                skipped.append({
                    "experiment": name,
                    "reason": "per-seed rows carry no numeric metric",
                })
            continue

        found_any = False
        aggregate_rows = 0
        for row in records:
            if not isinstance(row, dict):
                continue
            blocks = [v for v in row.values() if isinstance(v, dict)]
            if blocks and all("per_seed" not in v for v in blocks):
                aggregate_rows += 1
        for field in sorted(experiment):
            block = experiment[field]
            if not isinstance(block, dict):
                continue
            series = block.get("per_seed")
            if not (isinstance(series, list) and series
                    and all(_is_number(v) for v in series)):
                continue
            found_any = True
            key = f"{name}.{field}"
            if len(series) < 2:
                skipped.append({
                    "experiment": name,
                    "metric": key,
                    "reason": f"n={len(series)} (1); no CI emitted",
                })
                continue
            target = paired if "definition" in block else metrics
            target[key] = [float(v) for v in series]

        if not found_any:
            skipped.append({
                "experiment": name,
                "reason": "stores mean/stdev aggregates only; no per-seed raw "
                          "values are committed, so no CI can be computed",
            })
        elif aggregate_rows:
            skipped.append({
                "experiment": name,
                "reason": f"{aggregate_rows} arm row(s) store mean/stdev "
                          "aggregates only; a two-arm paired difference from "
                          "those arms is not computable, so only the committed "
                          "per_seed series is reported",
            })

    return metrics, paired, skipped


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    """Repo-relative when inside the checkout, so the artifact is portable."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def build_report(
    input_path: Path,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
    source_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute the CI report for every quantity with committed per-seed raw data."""
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("expected a JSON list of experiments")

    metrics, paired, skipped = collect_series(payload)

    metric_entries = {
        key: _rounded(bootstrap_mean_ci(
            values, resamples=resamples, confidence=confidence,
            rng_seed=seed, metric=key))
        for key, values in metrics.items()
    }

    paired_entries: Dict[str, Any] = {}
    for key, values in paired.items():
        result = _rounded(bootstrap_mean_ci(
            values, resamples=resamples, confidence=confidence,
            rng_seed=seed, metric=key))
        definition = _definition_for(payload, key)
        if definition:
            result = {"definition": definition, **result}
        result["source"] = "committed per_seed series (paired by seed base)"
        paired_entries[key] = result

    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "source": {
            "path": _display_path(input_path) if source_label is None else source_label,
            "sha256": _sha256(input_path),
            "experiments": [e.get("experiment") for e in payload
                            if isinstance(e, dict)],
        },
        "method": {
            "name": METHOD,
            "resamples": resamples,
            "confidence": confidence,
            "interval": [round((1.0 - confidence) / 2.0, 6),
                         round(1.0 - (1.0 - confidence) / 2.0, 6)],
            "quantile": "linear interpolation between order statistics",
            "rng": "random.Random, per-metric seed sha256(base_seed:metric)[:8]",
            "base_seed": seed,
            "point_estimate": "arithmetic mean of the observed values",
        },
        "metrics": metric_entries,
        "paired_differences": paired_entries,
        "not_computable": skipped,
    }
    return report


def _definition_for(payload: Sequence[Dict[str, Any]], key: str) -> Optional[str]:
    experiment, _, field = key.partition(".")
    for item in payload:
        if isinstance(item, dict) and item.get("experiment") == experiment:
            block = item.get(field)
            if isinstance(block, dict) and isinstance(block.get("definition"), str):
                return block["definition"]
    return None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Percentile bootstrap CIs for per-seed microbench results")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    report = build_report(
        args.input,
        resamples=args.resamples,
        confidence=args.confidence,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    if not args.quiet:
        print(f"source: {report['source']['path']} "
              f"(sha256 {report['source']['sha256'][:12]}...)")
        for key, entry in report["metrics"].items():
            print(f"  {key}: {entry['mean']:.4f} "
                  f"[{entry['ci_low']:.4f}, {entry['ci_high']:.4f}] "
                  f"n={entry['n']}")
        for key, entry in report["paired_differences"].items():
            print(f"  {key} (paired diff): {entry['mean']:.4f} "
                  f"[{entry['ci_low']:.4f}, {entry['ci_high']:.4f}] "
                  f"n={entry['n']}")
        print(f"  skipped: {len(report['not_computable'])} configuration(s)")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
