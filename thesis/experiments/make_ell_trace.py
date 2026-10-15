"""Generate the ELL audit trace for the gated pipeline (standalone).

Run after the grid: exports the gated variant's instruction stream to
ELL (workflow + JSONL message trace) for auditability.
"""

import asyncio
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "examples"))

from agentflow.core.isa.ell_export import (  # noqa: E402
    export_stream_to_ell, save_ell_trace,
)
from agentflow.sandbox.local_executor import LocalSandboxExecutor  # noqa: E402
from system_task_demo import inject_sandbox  # noqa: E402

from rsix import data as data_mod  # noqa: E402
from rsix.models import GRU4Rec  # noqa: E402
from rsix.streams import build_experiment_stream  # noqa: E402
from run_experiments import COMMON  # noqa: E402


def main():
    train, test, nu, ni = data_mod.dataset(HERE / "data" / "u.data")
    config = dict(COMMON, variant="gated", seed=0)
    config["generator"] = "consistency"
    stream = build_experiment_stream(
        config, lambda seed: GRU4Rec(ni, dim=64, seed=seed))
    import asyncio
    workflow, messages = asyncio.run(export_stream_to_ell(
        stream, name="rsi-gated",
        description="Fidelity-gated RSI closed loop over ML-100K "
                    "(ingest -> split -> gate fit -> 4 rounds of "
                    "generate/gate/mix/train -> results)"))
    out = save_ell_trace(
        HERE / "results" / "rsi_gated_trace.jsonl", workflow, messages)
    print(f"ELL trace -> {out} ({len(messages)} messages)")


if __name__ == "__main__":
    main()
