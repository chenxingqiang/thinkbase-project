"""Compact the tau-bench historical trajectories into the pilot corpus.

Source: sierra-research/tau-bench (MIT), ``historical_trajectories/*.json`` --
four real model runs (GPT-4o and Claude 3.5 Sonnet on the retail and airline
domains). Each file is a JSON list of task records::

    {"task_id": 0, "reward": 1.0, "trial": 0, "traj": [
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "get_order_details",
                          "arguments": "{\\"order_id\\": \\"#W1\\"}"}}]}, ...]}

This script keeps only what the pilot measures -- the ordered tool calls and
their arguments, grouped into turns (one turn = one assistant message, so
several parallel calls stay grouped) -- and drops message prose, system
prompts, and tool observations. Observed arguments are *inputs* to the call,
not results, which is what a result-cache key is built from.

Deterministic: same inputs -> same bytes (records and keys in source order).
The output corpus is committed so the pilot is reproducible without the 49 MB
upstream tarball; the source file digests are recorded in the README.

Usage:
    ./venv/bin/python examples/real_traces/prepare_tau_bench.py \
        --tau-bench-root /path/to/tau-bench
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

SOURCES = [
    "gpt-4o-retail",
    "gpt-4o-airline",
    "sonnet-35-new-retail",
    "sonnet-35-new-airline",
]


def _arguments(call: Dict[str, Any]) -> Dict[str, Any]:
    raw = (call.get("function") or {}).get("arguments")
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    return json.loads(raw)


def turns_of(record: Dict[str, Any]) -> List[List[List[Any]]]:
    """[turn][call] = [tool_name, arguments]."""
    turns: List[List[List[Any]]] = []
    for message in record.get("traj") or []:
        calls = message.get("tool_calls") or []
        if not calls:
            continue
        turns.append([
            [call["function"]["name"], _arguments(call)] for call in calls
        ])
    return turns


def compact(root: Path) -> Iterable[Dict[str, Any]]:
    base = root / "historical_trajectories"
    for source in SOURCES:
        path = base / f"{source}.json"
        for record in json.loads(path.read_text(encoding="utf-8")):
            turns = turns_of(record)
            if not turns:
                continue
            yield {
                "source": source,
                "task_id": record.get("task_id"),
                "trial": record.get("trial"),
                "reward": record.get("reward"),
                "turns": turns,
            }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau-bench-root", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("tau_bench_historical.jsonl"))
    args = parser.parse_args()

    rows = list(compact(args.tau_bench_root))
    payload = "\n".join(
        json.dumps(row, separators=(",", ":"), sort_keys=True) for row in rows
    ) + "\n"
    args.output.write_text(payload, encoding="utf-8")

    calls = sum(len(t) for row in rows for t in row["turns"])
    print(f"trajectories: {len(rows)}  calls: {calls}  -> {args.output}")
    for source in SOURCES:
        n = sum(1 for row in rows if row["source"] == source)
        c = sum(len(t) for row in rows if row["source"] == source for t in row["turns"])
        print(f"  {source:24s} {n:4d} trajectories {c:5d} calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
