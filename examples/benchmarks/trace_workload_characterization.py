"""Characterize the real 100-trace workload and record the result.

Experiment runner (E1, trace characterization): executes the I10
analyzer ``examples/analyze_task_traces.py`` as a subprocess over the
committed ``examples/random_task_traces`` corpus, then enriches its
report with the run provenance the experiment requires -- git rev,
UTC timestamp, corpus-generation seeds, scale, and a coverage block
-- and writes ``examples/benchmarks/trace_workload_results.json``.

Discipline: every aggregate number in the output is computed by the
analyzer from the raw corpus; this runner never hand-fills a
statistic. The headline block is derived by reference to the
analyzer's own output, and the corpus-generation seed constants are
machine-extracted from ``examples/random_tasks_stress.py`` (argparse
defaults plus the per-task ``random.Random(10_000 + spec.index)``
corpus seeding line) rather than hand-copied.

The analysis is a deterministic full census of the committed corpus:
the analyzer has no sampling randomness, so no analysis-side seeds
exist and multi-seed averaging does not apply (the recorded seeds
are the *corpus-generation* seeds).

Stdlib only. Usage:
    ./venv/bin/python \\
        examples/benchmarks/trace_workload_characterization.py \\
        [--trace-dir DIR] [--output FILE] [--top N]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYZER = REPO_ROOT / "examples" / "analyze_task_traces.py"
GENERATOR = REPO_ROOT / "examples" / "random_tasks_stress.py"
DEFAULT_TRACE_DIR = REPO_ROOT / "examples" / "random_task_traces"
DEFAULT_OUTPUT = (REPO_ROOT / "examples" / "benchmarks"
                  / "trace_workload_results.json")
RAW_TMP_DIR = REPO_ROOT / "examples" / "benchmarks"

# Session convention: proxy variables poison litellm imports, so the
# analyzer subprocess runs with them scrubbed from the environment.
PROXY_VARS = ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy",
              "HTTPS_PROXY", "https_proxy")

ANALYSES = ("zipf", "reuse", "shapes", "fusion")
SCHEMA = "trace_workload_results/v1"

# Methodological caveats carried from the analyzer's documented
# conventions (see its module docstring); text only, no numbers.
NOTES = (
    "params signature: the traces serialize no instruction params, "
    "so operation identity is recovered from step names by stripping "
    "trailing positional index suffixes (guard_0_2 -> guard); a "
    "non-empty explicit params field, when present, is canonicalized "
    "with sorted keys",
    "dead_end_steps is a single-pass structural count (pure steps "
    "whose outputs no step consumes), not the compiler's AGGRESSIVE "
    "fixpoint liveness analysis",
    "Zipf degenerate-case conventions: fewer than two categories -> "
    "s/r2 null; zero-variance (uniform) distribution -> r2 = 0.0",
    "shape 'branch' means the workflow has both a fan-out and a "
    "fan-in node (e.g. a diamond); the committed corpus contains no "
    "fan-in or branch workflows",
    "the fusion gate mirrors agentflow/core/isa/compiler.py::"
    "_are_fusible on trace dicts: same type, both explicitly pure, "
    "no producer-consumer edge inside the pair",
    "reuse_ratio is the corpus-level share of step instances that "
    "repeat an already-seen (type, params) signature; "
    "per_workflow_mean_reuse_ratio is the mean of the per-workflow "
    "repeat shares",
)


def _git_rev() -> Optional[str]:
    """Short HEAD revision via the one read-only git command the
    experiment brief allows; None if it cannot be resolved."""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip() or None


def _extract_generation_seeds() -> Dict[str, Any]:
    """Machine-extract the corpus seed scheme from the generator
    source, so provenance never relies on hand-copied constants."""
    out: Dict[str, Any] = {
        "generator": "examples/random_tasks_stress.py",
        "master_seed": None,
        "task_count": None,
        "per_task_seed_base": None,
        "per_task_seed_range": None,
        "extraction": ("argparse defaults and the per-task corpus "
                       "seeding line, parsed from the generator "
                       "source at run time"),
    }
    try:
        text = GENERATOR.read_text(encoding="utf-8")
    except OSError:
        out["extraction"] += " (generator source unreadable)"
        return out
    master = re.search(r'"--seed",\s*type=int,\s*default=(\d+)', text)
    tasks = re.search(r'"--tasks",\s*type=int,\s*default=(\d+)', text)
    per_task = re.search(
        r'random\.Random\((\d[\d_]*)\s*\+\s*spec\.index\)', text)
    if master:
        out["master_seed"] = int(master.group(1))
    if tasks:
        out["task_count"] = int(tasks.group(1))
    if per_task:
        base = int(per_task.group(1).replace("_", ""))
        out["per_task_seed_base"] = base
        if out["task_count"]:
            out["per_task_seed_range"] = [
                base, base + out["task_count"] - 1]
    if not (master and tasks and per_task):
        out["extraction"] += " (some constants not found)"
    return out


def _run_analyzer(trace_dir: Path, top: int) -> Dict[str, Any]:
    """Run the analyzer CLI as a subprocess; return its JSON report.

    The raw intermediate report lives in a temp directory under the
    workspace (sandbox-safe) and is deleted with it.
    """
    RAW_TMP_DIR.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in PROXY_VARS}
    with tempfile.TemporaryDirectory(
            prefix=".trace_workload_raw_", dir=str(RAW_TMP_DIR)) as tmp:
        raw = Path(tmp) / "raw_report.json"
        cmd = [sys.executable, str(ANALYZER),
               "--trace-dir", str(trace_dir),
               "--output", str(raw),
               "--top", str(top)]
        done = subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                              capture_output=True, text=True)
        if done.stdout:
            print(done.stdout, end="")
        if done.returncode != 0:
            sys.stderr.write(done.stderr)
            raise SystemExit(
                f"analyzer failed with exit code {done.returncode}")
        return json.loads(raw.read_text(encoding="utf-8"))


def _headline(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Key numbers, derived by reference to the analyzer's output."""
    z, r = report["zipf"], report["reuse"]
    s, f = report["shapes"], report["fusion"]
    return {
        "zipf_s": z["s"],
        "zipf_r2": z["r2"],
        "signature_categories": z["n_categories"],
        "reuse_ratio": r["reuse_ratio"],
        "per_workflow_mean_reuse_ratio":
            r["per_workflow_mean_reuse_ratio"],
        "shape_counts": s["counts"],
        "dead_end_steps": s["dead_end_steps"],
        "fusable_pairs": f["fusable_pairs"],
        "total_adjacent_pairs": f["total_adjacent_pairs"],
        "fusable_ratio": f["fusable_ratio"],
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the I10 trace-workload analyzer over the "
                    "committed corpus and write the enriched results "
                    "JSON (provenance: git rev, timestamp, seeds, "
                    "scale; plus coverage).")
    parser.add_argument(
        "--trace-dir", default=str(DEFAULT_TRACE_DIR),
        help=f"directory of *.jsonl traces (default: {DEFAULT_TRACE_DIR})")
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help=f"results JSON path (default: {DEFAULT_OUTPUT})")
    parser.add_argument(
        "--top", type=int, default=10,
        help="number of top signatures to list (default: 10)")
    args = parser.parse_args(argv)

    trace_dir = Path(args.trace_dir)
    trace_files = sorted(p for p in trace_dir.glob("*.jsonl")
                         if p.is_file())
    report = _run_analyzer(trace_dir, args.top)

    prov = report["provenance"]
    n_files = len(trace_files)
    n_workflows = prov["n_traces"]
    files_without_workflow = n_files - n_workflows
    analyses_run = [k for k in ANALYSES if k in report]

    seeds = _extract_generation_seeds()
    seeds["analysis_seed"] = None
    seeds["analysis_seed_note"] = (
        "the analysis is a deterministic full census of the committed "
        "corpus; the analyzer has no sampling randomness, so "
        "multi-seed averaging does not apply (these are the "
        "corpus-generation seeds)")

    gaps: List[str] = []
    if files_without_workflow:
        gaps.append(f"{files_without_workflow} trace file(s) contained "
                    "no workflow record and were skipped")
    for key in ("master_seed", "task_count", "per_task_seed_base"):
        if seeds[key] is None:
            gaps.append(f"generation seed constant {key!r} could not "
                        "be extracted from the generator source")

    prov["generated_at_utc"] = datetime.now(
        timezone.utc).isoformat(timespec="seconds")
    prov["git_rev"] = _git_rev()
    if prov["git_rev"] is None:
        gaps.append("git_rev could not be resolved")
    prov["analyzer"] = {
        "path": "examples/analyze_task_traces.py",
        "role": "I10 trace analyzer; produced every statistic in "
                "this file",
        "invocation": [
            "./venv/bin/python", "examples/analyze_task_traces.py",
            "--trace-dir", str(trace_dir),
            "--top", str(args.top),
            "--output", "<temp raw report (deleted after read)>",
        ],
    }
    prov["seeds"] = seeds
    prov["scale"] = {
        "n_trace_files": n_files,
        "n_workflows": n_workflows,
        "n_steps": prov["n_steps"],
        "n_signature_categories": report["zipf"]["n_categories"],
        "n_step_instances": report["reuse"]["total_instances"],
        "n_adjacent_pairs": report["fusion"]["total_adjacent_pairs"],
    }
    prov["determinism"] = (
        "the analysis sections are a deterministic function of the "
        "corpus (pinned by corpus_sha256); only generated_at_utc and "
        "git_rev vary between runs of this runner")

    coverage = {
        "status": ("full"
                   if (not gaps and files_without_workflow == 0
                       and analyses_run == list(ANALYSES))
                   else "partial"),
        "component_status": {"I10-trace-analyzer": "shipped"},
        "n_trace_files_found": n_files,
        "n_workflows_parsed": n_workflows,
        "n_files_without_workflow_record": files_without_workflow,
        "analyses_run": analyses_run,
        "analyses_missing": [k for k in ANALYSES if k not in report],
        "provenance_gaps": gaps,
        "notes": list(NOTES),
    }

    output = {
        "schema": SCHEMA,
        **report,
        "coverage": coverage,
        "headline": _headline(report),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n",
                        encoding="utf-8")
    print(f"results written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
