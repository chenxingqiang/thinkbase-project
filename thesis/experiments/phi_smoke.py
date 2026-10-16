"""Smoke: real-data gated run with the Phi layer (postconditions) active.

Runs one gated seed on the real ML-100K data with all invariants
enforced (any violation raises), then cross-checks the resulting NDCG
curve against the committed main_results.json per-seed value for the
same seed -- same stream semantics + same seed must reproduce the
committed numbers exactly (determinism cross-validation), while the
INV-1 budget postcondition machine-checks every round's mixture.

Discards the sandbox result; writes nothing to results/.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "examples"))

import torch  # noqa: E402

from agentflow.sandbox.local_executor import LocalSandboxExecutor  # noqa: E402
from system_task_demo import inject_sandbox, run_stream  # noqa: E402

from rsix import data as data_mod  # noqa: E402
from rsix.models import GRU4Rec  # noqa: E402
from rsix.streams import build_experiment_stream  # noqa: E402
from run_experiments import COMMON, SEEDS  # noqa: E402


def main():
    torch.set_num_threads(2)
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    _, _, _, n_items = data_mod.dataset(HERE / "data" / "u.data")
    config = dict(COMMON, variant="gated", seed=seed)
    config["generator"] = "consistency"
    stream = build_experiment_stream(
        config, lambda seed: GRU4Rec(n_items, dim=64, seed=seed))
    n_post = sum(1 for i in stream if i.context.postconditions)
    print(f"stream: {len(stream)} instructions, "
          f"{n_post} carry postcondition invariants")

    sb = LocalSandboxExecutor(name=f"phi-smoke-{seed}")
    try:
        sb.write_file("data/u.data",
                      (HERE / "data" / "u.data").read_bytes())
        inject_sandbox(stream, sb)
        t0 = time.time()
        asyncio.run(run_stream(stream))  # raises on any violation
        wall = time.time() - t0
        hist = json.loads(
            sb.read_file(f"result_gated_{seed}.json").decode())
    finally:
        sb.close()

    rounds = hist["rounds"]
    ndcg = [r["metrics"]["ndcg@10"] for r in rounds]
    # the final round trains but does not gate -> gate_stats is {}
    js = [r["gate_stats"].get("mixed_js_distance") for r in rounds]
    adm = [r["gate_stats"].get("n_admitted") for r in rounds]
    print(f"wall={wall:.0f}s")
    print("ndcg per round:", [round(v, 4) for v in ndcg])
    print("mixed JS per round (INV-1 checked):", js)
    print("admitted per round:", adm)
    assert all(v is None or v <= config.get("eps", 0.08) + 1e-9 for v in js), \
        "INV-1 would have raised inside the stream"

    # determinism cross-check against the committed grid
    mr = json.loads((HERE / "results" / "main_results.json").read_text())
    committed = mr["table"]["gated"]["per_seed_ndcg"][SEEDS.index(seed)]
    print(f"final ndcg: this run={ndcg[-1]:.4f} "
          f"committed(seed {seed})={committed:.4f}")
    status = "IDENTICAL" if abs(ndcg[-1] - committed) < 1e-12 else "DIFF"
    print(f"determinism cross-check: {status}")
    return 0 if status == "IDENTICAL" else 1


if __name__ == "__main__":
    sys.exit(main())
