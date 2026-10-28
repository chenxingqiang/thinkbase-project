"""Compact real SWE-agent trajectories into the pilot corpus.

Source: ``nebius/SWE-agent-trajectories`` (CC-BY-4.0) -- recorded runs of
SWE-agent on SWE-bench instances (Llama-70B/8B/405B backbones). Each row holds a
``trajectory``: alternating ``ai`` and ``user`` messages. The action is not a
structured tool call here: SWE-agent emits it as the last fenced code block of
the assistant text, and the next user message is its observation.

The parser is deliberately mechanical and disclosed, because it decides the
workload:

  * take the last triple-backtick block of each ``ai`` message -- no block, no
    action (the model only reasoned; SWE-agent would have ended the episode);
  * a block starting with ``str_replace_editor`` is an editor call: its first
    token after the tool name is the subcommand (``view``/``create``/
    ``str_replace``/``insert``), and the first argument after that is the path;
  * anything else is a shell command, with its first line kept as ``cmd``;
  * ``view`` is a read (pure, returns a file); every other editor subcommand and
    every shell command is effectful.

Compaction keeps ``[tool, args]`` per turn and drops prose and observations --
the pilot measures structure and cache keys, and a key is built from a call's
inputs.

Deterministic: rows are taken in file order, so the same shard and limit always
produce the same corpus. One call per turn: SWE-agent acts sequentially.

Usage:
    ./venv/bin/python examples/real_traces/prepare_swe_agent.py \
        --parquet /tmp/swe0.parquet --limit 300
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

FENCE = re.compile(r"```[a-zA-Z0-9_-]*\n(.*?)```", re.DOTALL)


def parse_action(text: str) -> Optional[List[Any]]:
    """Return [tool, args] for one assistant message, or None if it acted not."""
    blocks = FENCE.findall(text or "")
    if not blocks:
        return None
    body = blocks[-1].strip()
    if not body:
        return None
    if body.startswith("str_replace_editor"):
        parts = body.split()
        sub = parts[1] if len(parts) > 1 else "view"
        path = next((p for p in parts[2:] if p.startswith("/")), "")
        return ["str_replace_editor", {"command": sub, "path": path}]
    cmd = body.splitlines()[0].strip()
    return ["bash", {"cmd": cmd}]


def turns_of(record: Dict[str, Any]) -> List[List[List[Any]]]:
    turns: List[List[List[Any]]] = []
    for message in record.get("trajectory") or []:
        if message.get("role") != "ai":
            continue
        action = parse_action(message.get("text") or "")
        if action:
            turns.append([action])
    return turns


def compact(parquet: Path, limit: int) -> Iterable[Dict[str, Any]]:
    import pyarrow.parquet as pq  # imported here so the module stays testable without it

    table = pq.read_table(parquet, columns=["instance_id", "model_name", "trajectory"])
    rows = table.to_pylist()
    for row in rows[:limit]:
        turns = turns_of(row)
        if not turns:
            continue
        yield {
            "source": "swe-agent-" + str(row.get("model_name", "unknown")),
            "task_id": row.get("instance_id"),
            "trial": 0,
            "reward": None,
            "turns": turns,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("swe_agent_sample.jsonl"))
    args = parser.parse_args()

    rows = list(compact(args.parquet, args.limit))
    payload = "\n".join(
        json.dumps(row, separators=(",", ":"), sort_keys=True) for row in rows
    ) + "\n"
    args.output.write_text(payload, encoding="utf-8")

    calls = sum(len(t) for row in rows for t in row["turns"])
    bash = sum(1 for row in rows for t in row["turns"] for n, _ in t if n == "bash")
    views = sum(1 for row in rows for t in row["turns"]
                for n, a in t if n == "str_replace_editor" and a.get("command") == "view")
    print(f"trajectories: {len(rows)}  calls: {calls}  "
          f"(bash {bash}, editor-view {views}, other {calls - bash - views})")
    print(f"mean calls/trajectory: {calls / max(1, len(rows)):.1f}  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
