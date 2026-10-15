"""Save the system-task demo's AgentISA instruction stream into ELL.

Runs the service-log-triage stream once (so instruction metrics are
real), exports it into the ELL integration layer as a workflow +
message trace, saves the trace as JSONL, and prints a compact
human-readable rendering.

Usage:
    ./venv/bin/python examples/system_task_ell_export.py \
        --seed 42 --files 6 --lines-per-file 400 \
        --trace examples/system_task_ell_trace.jsonl
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentflow.core.isa.ell_export import (  # noqa: E402
    export_stream_to_ell,
    render_trace,
    save_ell_trace,
)
from agentflow.ell.integration import ell_integration  # noqa: E402
from system_task_demo import (  # noqa: E402
    build_stream,
    generate_corpus,
    inject_sandbox,
    report_sha256,
    run_stream,
    write_corpus,
)
from agentflow.sandbox.local_executor import LocalSandboxExecutor  # noqa: E402


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the demo instruction stream into ELL")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--files", type=int, default=6)
    parser.add_argument("--lines-per-file", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=3,
                        help="repeated realizations (exercises the cache)")
    parser.add_argument("--trace", type=str,
                        default="examples/system_task_ell_trace.jsonl")
    args = parser.parse_args()

    import random
    corpus = generate_corpus(random.Random(args.seed), args.files,
                             args.lines_per_file)
    log_files = sorted(corpus)
    from system_task_demo import RuleClassifierBackend
    backend = RuleClassifierBackend()
    stream = build_stream(backend, log_files)

    # Execute the stream repeatedly so metrics (cache hits, executions)
    # reflect a real run, not zeros.
    sandbox = LocalSandboxExecutor(name="ell-export")
    try:
        write_corpus(sandbox, corpus)
        inject_sandbox(stream, sandbox)
        for _ in range(args.repeats):
            await run_stream(stream)
        report_hash = report_sha256(sandbox)
    finally:
        sandbox.close()

    execution_summary = {
        "realizations": args.repeats,
        "model_calls": backend.model_calls,
        "report_sha256": report_hash[:16],
    }
    workflow, messages = await export_stream_to_ell(
        stream,
        name="service-log-triage",
        description=(
            "AgentISA demo: seven-instruction service-log triage stream "
            "(sandbox I/O, dead guard, pure transforms, cacheable LLM step)"
        ),
        execution_summary=execution_summary,
    )

    path = save_ell_trace(args.trace, workflow, messages)

    stored = ell_integration.get_messages()
    print("=" * 74)
    print(f"ELL workflow '{workflow.name}' registered: "
          f"{len(workflow.steps)} steps, id={workflow.id[:8]}..")
    print(f"messages recorded in ELLIntegration: {len(stored)}")
    print(f"trace saved: {path}")
    print("=" * 74)
    print(render_trace(messages))
    print("=" * 74)
    print(f"workflows registered: {ell_integration.list_workflows()}")
    print(f"stream metrics: {json.dumps(execution_summary)}")


if __name__ == "__main__":
    asyncio.run(main())
