#!/usr/bin/env python3
"""Render the unsafe-cache safety tables from unsafe_cache_results.json.

Writes markdown to stdout, and also to --output when given. Two tables:

  1. one row per policy: hit rate, unsafe hits, unsafe hits per 1,000 calls,
     and the assumption the policy makes;
  2. the unsafe hits split into writes served, stale reads, and cross-tenant.

Every number is read from the JSON; nothing is defaulted, interpolated, or
estimated. The input schema is deliberately narrow, and every field is
required, so a producer that drops or renames one fails here loudly:

  {
    "policies": [                      # exactly four, names unique; an object
                                       # keyed by policy name is accepted too
      {
        "name": "exact-key",
        "hit_rate": 0.62,              # share in [0, 1], not a percentage
        "unsafe_hits": 41,             # integer
        "calls": 10000,                # integer > 0, the per-1,000 denominator
        "assumes": "byte-identical (tool, args)",
        "unsafe_breakdown": {
          "writes_served": 12,         # the three parts must sum to unsafe_hits
          "stale_reads": 25,
          "cross_tenant": 4
        }
      }
    ]
  }

--output refuses to overwrite a git-tracked file unless --force is passed, so
running the script bare cannot clobber a committed artifact.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BREAKDOWN_FIELDS = ("writes_served", "stale_reads", "cross_tenant")
POLICY_FIELDS = ("name", "hit_rate", "unsafe_hits", "calls", "assumes", "unsafe_breakdown")
MAX_TABLE_LINES = 40

# LaTeX specials; "%" is the one that silently eats text in a paper.
_TEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


class TableError(Exception):
    """A missing, malformed, or internally inconsistent input field."""


def tex(value: str) -> str:
    """Escape a string for LaTeX; escape the pipe too so markdown cells hold."""
    return "".join(_TEX_ESCAPES.get(ch, ch) for ch in value).replace("|", r"\|")


def require(obj: dict, key: str, where: str):
    if not isinstance(obj, dict):
        raise TableError(f"{where}: expected an object, got {type(obj).__name__}")
    if key not in obj:
        raise TableError(f"{where}: missing required field {key!r}")
    return obj[key]


def as_int(value, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TableError(f"{where}: expected an integer, got {value!r}")
    return value


def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise TableError(f"input not found: {path}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TableError(f"{path}: not valid JSON ({exc})") from exc

    policies = require(doc, "policies", str(path))
    if isinstance(policies, dict):  # keyed by policy name is the other natural shape
        converted = []
        for name, fields in policies.items():
            if not isinstance(fields, dict):
                raise TableError(f"{path}: policies[{name!r}] must be an object")
            if "name" in fields and fields["name"] != name:
                raise TableError(f"{path}: policy key {name!r} disagrees with its 'name' {fields['name']!r}")
            converted.append({**fields, "name": name})
        policies = converted
    if not isinstance(policies, list):
        raise TableError(f"{path}: 'policies' must be a list or an object keyed by name, "
                         f"got {type(policies).__name__}")
    if len(policies) != 4:
        raise TableError(f"{path}: 'policies' must hold exactly the four policies, got {len(policies)}")

    rows, seen = [], set()
    for index, policy in enumerate(policies):
        where = f"{path}: policies[{index}]"
        for key in POLICY_FIELDS:
            require(policy, key, where)

        name = policy["name"]
        if not isinstance(name, str) or not name.strip():
            raise TableError(f"{where}: 'name' must be a non-empty string")
        if name in seen:
            raise TableError(f"{where}: duplicate policy name {name!r}")
        seen.add(name)

        hit_rate = policy["hit_rate"]
        if isinstance(hit_rate, bool) or not isinstance(hit_rate, (int, float)):
            raise TableError(f"{where}: 'hit_rate' must be a number, got {hit_rate!r}")
        if not 0.0 <= float(hit_rate) <= 1.0:
            raise TableError(f"{where}: 'hit_rate' is a share in [0, 1], got {hit_rate!r}")

        unsafe_hits = as_int(policy["unsafe_hits"], f"{where}.unsafe_hits")
        calls = as_int(policy["calls"], f"{where}.calls")
        if unsafe_hits < 0:
            raise TableError(f"{where}: 'unsafe_hits' must be >= 0, got {unsafe_hits}")
        if calls <= 0:
            raise TableError(f"{where}: 'calls' must be > 0, got {calls}")
        if unsafe_hits > calls:
            raise TableError(f"{where}: 'unsafe_hits' ({unsafe_hits}) exceeds 'calls' ({calls})")

        assumes = policy["assumes"]
        if not isinstance(assumes, str) or not assumes.strip():
            raise TableError(f"{where}: 'assumes' must be a non-empty string")

        breakdown = policy["unsafe_breakdown"]
        parts = {}
        for key in BREAKDOWN_FIELDS:
            parts[key] = as_int(require(breakdown, key, f"{where}.unsafe_breakdown"),
                                f"{where}.unsafe_breakdown.{key}")
            if parts[key] < 0:
                raise TableError(f"{where}.unsafe_breakdown.{key} must be >= 0, got {parts[key]}")
        if sum(parts.values()) != unsafe_hits:
            raise TableError(
                f"{where}: unsafe_breakdown sums to {sum(parts.values())} but 'unsafe_hits' is {unsafe_hits}")

        rows.append({"name": name, "hit_rate": float(hit_rate), "unsafe_hits": unsafe_hits,
                     "calls": calls, "assumes": assumes, **parts})
    return rows


def markdown(rows: list[dict], source: str) -> str:
    header = ["Policy", "Hit rate", "Unsafe hits", "Unsafe / 1k calls", "Assumes"]
    lines = [f"*Table 1: unsafe cache reuse by policy (source: `{source}`).*", "",
             "| " + " | ".join(header) + " |",
             "|" + "|".join(["---", "---:", "---:", "---:", "---"]) + "|"]
    for row in rows:
        lines.append(
            f"| {tex(row['name'])} | {row['hit_rate'] * 100:.1f}\\% | {row['unsafe_hits']} | "
            f"{row['unsafe_hits'] * 1000.0 / row['calls']:.1f} | {tex(row['assumes'])} |")

    lines += ["", "*Table 2: the same unsafe hits split by failure mode.*", "",
              "| Policy | Writes served | Stale reads | Cross-tenant | Total |",
              "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {tex(row['name'])} | {row['writes_served']} | {row['stale_reads']} | "
                     f"{row['cross_tenant']} | {row['unsafe_hits']} |")

    body = 2 * len(rows) + 4  # both header pairs and both separator lines
    if body > MAX_TABLE_LINES:
        raise TableError(f"table body would be {body} lines, over the {MAX_TABLE_LINES}-line budget")
    return "\n".join(lines) + "\n"


def latex(rows: list[dict]) -> str:
    out = [r"\begin{tabular}{lrrrl}", r"\toprule",
           r"Policy & Hit rate & Unsafe hits & Unsafe / 1k calls & Assumes \\", r"\midrule"]
    for row in rows:
        out.append(f"{tex(row['name'])} & {row['hit_rate'] * 100:.1f}\\% & {row['unsafe_hits']} & "
                   f"{row['unsafe_hits'] * 1000.0 / row['calls']:.1f} & {tex(row['assumes'])} \\\\")
    out += [r"\midrule",
            r"Policy & Writes served & Stale reads & Cross-tenant & Total \\", r"\midrule"]
    for row in rows:
        out.append(f"{tex(row['name'])} & {row['writes_served']} & {row['stale_reads']} & "
                   f"{row['cross_tenant']} & {row['unsafe_hits']} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def is_tracked(path: Path) -> bool:
    try:
        result = subprocess.run(["git", "-C", str(path.parent), "ls-files", "--error-unmatch", path.name],
                                capture_output=True, text=True, check=False)
    except OSError:
        return False
    return result.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render unsafe-cache safety tables from JSON.")
    parser.add_argument("--input", type=Path,
                        default=Path(__file__).with_name("unsafe_cache_results.json"),
                        help="results JSON (default: unsafe_cache_results.json beside this script)")
    parser.add_argument("--output", type=Path, help="also write the markdown here")
    parser.add_argument("--force", action="store_true", help="allow overwriting a tracked --output")
    parser.add_argument("--latex", action="store_true", help="emit a booktabs tabular instead of markdown")
    args = parser.parse_args(argv)

    try:
        rows = load_rows(args.input)
        if args.output is None:
            sys.stdout.write(latex(rows) if args.latex else markdown(rows, args.input.name))
            return 0

        out = args.output.expanduser().resolve()
        if out == args.input.resolve():
            raise TableError(f"--output {out} is the input file; refusing to overwrite it")
        if out.is_dir():
            raise TableError(f"--output {out} is a directory")
        if out.exists() and is_tracked(out) and not args.force:
            raise TableError(f"--output {out} is tracked by git; pass --force to overwrite it")
        if out.exists() and not is_tracked(out):
            print(f"safety_table: note: overwriting existing untracked {out}", file=sys.stderr)

        text = latex(rows) if args.latex else markdown(rows, args.input.name)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        sys.stdout.write(text)
        return 0
    except TableError as exc:
        print(f"safety_table: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
