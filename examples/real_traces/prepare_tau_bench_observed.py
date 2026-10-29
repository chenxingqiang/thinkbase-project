"""Compact tau-bench trajectories *with observations* for compound-instruction analysis.

The pilot corpus (`prepare_tau_bench.py`) keeps only calls, because a result-cache
key is built from a call's inputs. Compound instructions need the missing half:
whether a later call's arguments can be *bound* from earlier calls' results. That
question cannot be asked of an inputs-only corpus, so this compactor keeps, for
each call, its tool name, its arguments, and the observation the environment
returned for it (`tool` messages, joined by `tool_call_id`).

Observations are truncated to OBS_LIMIT characters: the binding test looks for
scalar values that a later argument reuses, and those appear within the first few
hundred characters of these records. Truncation is disclosed because it can only
make the bindable fraction *conservative* -- a value cut off cannot be found.

Deterministic: records and calls stay in source order; the same inputs give the
same bytes.

Usage:
    ./venv/bin/python examples/real_traces/prepare_tau_bench_observed.py \
        --tau-bench-root /path/to/tau-bench \
        --output examples/real_traces/tau_bench_observed.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

SOURCES = [
    "gpt-4o-retail",
    "gpt-4o-airline",
    "sonnet-35-new-retail",
    "sonnet-35-new-airline",
]
OBS_LIMIT = 400


def _arguments(call: Dict[str, Any]) -> Dict[str, Any]:
    raw = (call.get("function") or {}).get("arguments")
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": str(raw)}


def calls_of(record: Dict[str, Any]) -> List[List[Any]]:
    """[[tool, arguments, observation], ...] in call order.

    The observation is the content of the ``tool`` message whose
    ``tool_call_id`` matches the call, or ``None`` when the run has none (the
    final call often does).
    """
    observations: Dict[str, str] = {}
    pending: List[str] = []
    for message in record.get("traj") or []:
        if message.get("role") == "tool":
            key = message.get("tool_call_id") or (pending.pop(0) if pending else "")
            observations[str(key)] = str(message.get("content") or "")[:OBS_LIMIT]
        for call in message.get("tool_calls") or []:
            pending.append(str(call.get("id") or ""))
    out: List[List[Any]] = []
    for message in record.get("traj") or []:
        for call in message.get("tool_calls") or []:
            call_id = str(call.get("id") or "")
            out.append([call["function"]["name"], _arguments(call),
                        observations.get(call_id)])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau-bench-root", type=Path, required=True)
    parser.add_argument("--first-trial-only", action="store_true", default=True)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("tau_bench_observed.jsonl"))
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for source in SOURCES:
        path = args.tau_bench_root / "historical_trajectories" / f"{source}.json"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        seen = set()
        for record in json.loads(path.read_text(encoding="utf-8")):
            task = str(record.get("task_id"))
            if task in seen:
                continue
            seen.add(task)
            calls = calls_of(record)
            if not calls:
                continue
            rows.append({
                "source": source,
                "task_id": task,
                "trial": record.get("trial", 0),
                "reward": record.get("reward"),
                "calls": calls,
            })

    payload = "\n".join(json.dumps(row, separators=(",", ":"), sort_keys=True)
                       for row in rows) + "\n"
    args.output.write_text(payload, encoding="utf-8")

    calls = sum(len(row["calls"]) for row in rows)
    with_obs = sum(1 for row in rows for call in row["calls"] if call[2])
    print(f"tasks: {len(rows)}  calls: {calls}  with observation: {with_obs} "
          f"({with_obs / max(1, calls):.1%})  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
