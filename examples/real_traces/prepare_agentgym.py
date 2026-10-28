"""Compact AgentGym task flows into the pilot corpus (finite action spaces).

Source: ``AgentGym/AgentTraj-L`` (Apache-2.0). Each record holds a short
``conversations`` list in which the agent's turns mix a preamble, a ``Thought:``
and, finally, an ``Action:`` line. Three environments are used here because their
action spaces are *finite vocabularies* rather than free-form tool arguments:

  * ``babyai``  -- grid navigation: turn left/right, move forward, go to <obj>,
    pick up <obj>, go through <door>, toggle and go through <door>;
  * ``alfworld`` -- household manipulation: go to, take <obj> from <rec>,
    put <obj> in/on <rec>, open, close, use, heat, cool, clean, examine, look,
    inventory;
  * ``sqlgym``  -- analytics: one verb family over table names.

Compaction keeps the ordered ``[verb, args]`` actions of each record and drops
prose, thoughts, and observations. Numbers are stripped from entity mentions so
that ``alarmclock 1`` and ``alarmclock 2`` instantiate the same template -- the
identity of an instruction is its operation, not the object instance.

Disclosed limitation: these records are per-step samples (an episode prefix and
its next action), not complete episodes, so they measure action-level
precipitation across tasks -- how quickly the vocabulary closes and how much
later traffic it serves -- not the composition of a whole episode into one
instruction.

Usage:
    ./venv/bin/python examples/real_traces/prepare_agentgym.py \
        --input /tmp/agt --output examples/real_traces/agentgym_sample.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ACTION_MARKER = "action:"
SQL_VERB = re.compile(r"^\s*(select|insert|update|delete|with|create|drop)\b", re.IGNORECASE)
TABLE = re.compile(r"\b(?:from|join|into|update)\s+([a-zA-Z_][\w.]*)", re.IGNORECASE)
TRAILING_ID = re.compile(r"\s+\d+$")


def _strip_instance(text: str) -> str:
    """``alarmclock 1`` -> ``alarmclock``; instance numbers are parameters."""
    return TRAILING_ID.sub("", text.strip()).lower()


def parse_alfworld(action: str) -> Tuple[str, Tuple[str, ...]]:
    text = action.strip().rstrip(".")
    lowered = text.lower()
    for verb in ("go to", "take", "put", "heat", "cool", "clean", "examine",
                 "open", "close", "use", "toggle", "look", "inventory"):
        if lowered.startswith(verb + " ") or lowered == verb:
            rest = text[len(verb):].strip()
            if verb == "take":
                # take <obj> from <rec>
                parts = re.split(r"\s+from\s+", rest, maxsplit=1, flags=re.IGNORECASE)
                obj = _strip_instance(parts[0]) if parts else ""
                rec = _strip_instance(parts[1]) if len(parts) > 1 else ""
                return "take", (obj, rec)
            if verb == "put":
                parts = re.split(r"\s+(?:in|on)\s+", rest, maxsplit=1, flags=re.IGNORECASE)
                obj = _strip_instance(parts[0]) if parts else ""
                rec = _strip_instance(parts[1]) if len(parts) > 1 else ""
                return "put", (obj, rec)
            if verb in ("heat", "cool", "clean"):
                parts = re.split(r"\s+with\s+", rest, maxsplit=1, flags=re.IGNORECASE)
                obj = _strip_instance(parts[0]) if parts else ""
                tool = _strip_instance(parts[1]) if len(parts) > 1 else ""
                return verb, (obj, tool)
            return verb, (_strip_instance(rest),)
    return "other", (lowered.split()[0] if lowered.split() else "",)


BABYAI_ALIASES = {"pickup": "pick up", "drop": "pick up", "toggle": "toggle and go through"}


def parse_babyai(action: str) -> Tuple[str, Tuple[str, ...]]:
    text = action.strip().rstrip(".")
    lowered = text.lower()
    for alias, canonical in BABYAI_ALIASES.items():
        if lowered.startswith(alias):
            text = canonical + text[len(alias):]
            lowered = text.lower()
            break
    for verb in ("toggle and go through", "go through", "pick up", "go to",
                 "turn right", "turn left", "move forward"):
        if lowered.startswith(verb):
            rest = text[len(verb):].strip()
            return verb, (_strip_instance(rest),) if rest else ()
    return "other", (lowered.split()[0] if lowered.split() else "",)


def parse_sqlgym(action: str) -> Tuple[str, Tuple[str, ...]]:
    match = SQL_VERB.match(action)
    verb = match.group(1).lower() if match else "other"
    tables = tuple(sorted({_strip_instance(t.split(".")[-1]) for t in TABLE.findall(action)}))
    return verb, tables


PARSERS = {"alfworld": parse_alfworld, "babyai": parse_babyai, "sqlgym": parse_sqlgym}


def last_action(text: str) -> Optional[str]:
    """The action is whatever follows the last ``Action:`` marker.

    BabyAI puts it on the next line, ALFWorld on the same line, and SQLGym in a
    fenced block, so the remainder is joined after removing code fences. Only
    the *last* marker counts: a thought may mention the word earlier.
    """
    lowered = text.lower()
    index = lowered.rfind(ACTION_MARKER)
    if index < 0:
        return None
    rest = text[index + len(ACTION_MARKER):]
    rest = rest.replace("```sql", " ").replace("```", " ")
    body = " ".join(line.strip() for line in rest.splitlines() if line.strip()).strip()
    return body or None


def actions_of(record: Dict[str, Any], env: str) -> List[List[Any]]:
    turns: List[List[Any]] = []
    parser = PARSERS[env]
    for message in record.get("conversations") or []:
        if message.get("from") not in ("gpt", "assistant"):
            continue
        body = last_action(message.get("value") or "")
        if not body:
            continue
        verb, shape = parser(body)
        turns.append([[verb, {"shape": list(shape)}]])
    return turns


def compact(root: Path, envs: Sequence[str], limit: Optional[int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for env in envs:
        path = root / f"{env}_train.json"
        if not path.exists():
            print(f"  skip {env}: missing {path}")
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        kept = 0
        for row in rows:
            turns = actions_of(row, env)
            if not turns:
                continue
            out.append({
                "source": f"agentgym-{env}",
                "task_id": str(row.get("item_id") or f"{env}-{kept}"),
                "trial": 0,
                "reward": None,
                "turns": turns,
            })
            kept += 1
            if limit and kept >= limit:
                break
        print(f"  {env}: records={len(rows)} kept={kept}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--envs", default="alfworld,babyai,sqlgym")
    parser.add_argument("--limit", type=int, default=0, help="per environment; 0 = all")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("agentgym_sample.jsonl"))
    args = parser.parse_args()

    rows = compact(args.input, [e.strip() for e in args.envs.split(",") if e.strip()],
                   args.limit or None)
    payload = "\n".join(json.dumps(row, separators=(",", ":"), sort_keys=True) for row in rows) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    calls = sum(len(t) for row in rows for t in row["turns"])
    print(f"records: {len(rows)}  actions: {calls}  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
