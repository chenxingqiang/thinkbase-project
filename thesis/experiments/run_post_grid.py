"""Post-grid supplementary runs (executed AFTER the main grid).

Purpose (both required for thesis honesty):
1. ``cons_conf`` ablation cell: consistency generator + confidence-only
   gate, 3 seeds -- completes the generator x gate mechanism grid with
   the missing row (measures the gate's marginal value on a healthy
   candidate pool).
2. Gated drift re-statistics: the main grid ran with the pre-fix
   fidelity module whose reported mixed-JS was the stale value of the
   last *rejected* candidate.  Admission decisions and metrics are
   unaffected, but the reported drift curve must be recomputed with the
   fixed module.  This reruns gated x 3 seeds and REWRITES only the
   gated curves/js entries in main_results.json, keeping all other
   variants' measurements untouched.

Runs are additive to results: ablation.json gains ``cons_conf``;
main_results.json gated js curve becomes the honest admitted-mixture
JS.  Deterministic, seeded, single-process.
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
from run_experiments import COMMON, agg, final_metric, SEEDS  # noqa: E402

OUT = HERE / "results"


def one_run(variant, seed, **over):
    config = dict(COMMON, variant=variant, seed=seed, **over)
    config["generator"] = over.get(
        "generator", "consistency" if variant == "gated" else "topk")

    def factory(seed):
        return GRU4Rec(n_items, dim=64, seed=seed)

    stream = build_experiment_stream(config, factory)
    sb = LocalSandboxExecutor(name=f"post-{variant}-{seed}")
    try:
        sb.write_file("data/u.data", Path(HERE / "data" / "u.data").read_bytes())
        inject_sandbox(stream, sb)
        t0 = time.time()
        asyncio.run(run_stream(stream))
        wall = time.time() - t0
        hist = json.loads(sb.read_file(
            f"result_{variant}_{seed}.json").decode())
    finally:
        sb.close()
    hist["wall_s"] = wall
    return hist


if __name__ == "__main__":
    torch.set_num_threads(2)
    train, test, n_users, n_items = data_mod.dataset(
        Path(HERE / "data" / "u.data"))

    # ---- 1) cons_conf ablation cell (3 seeds) --------------------
    print("== cons_conf (consistency gen + confidence-only gate) ==")
    hists = [one_run("conf_only", s, generator="consistency")
             for s in SEEDS]
    finals = final_metric(hists)
    print("  per-seed:", [round(v, 4) for v in finals],
          "mean:", round(sum(finals) / len(finals), 4))
    ab = json.loads((OUT / "ablation.json").read_text())
    ab["table"]["cons_conf"] = {
        "ndcg@10_mean": sum(finals) / len(finals),
        "per_seed": finals}
    (OUT / "ablation.json").write_text(json.dumps(ab, indent=1))
    print("  -> ablation.json[cons_conf]")

    # ---- 2) gated drift re-statistics (fixed module, 3 seeds) -----
    print("== gated drift re-stats (fixed JS reporting) ==")
    gh = [one_run("gated", s) for s in SEEDS]
    g = agg(gh)
    # sanity: performance must match the main grid (same seeds/decisions)
    mr = json.loads((OUT / "main_results.json").read_text())
    old_final = mr["table"]["gated"]["ndcg@10_mean"]
    new_final = g["ndcg"][-1]
    print(f"  ndcg final: old={old_final:.4f} new={new_final:.4f} "
          f"(decisions identical => should match)")
    if abs(old_final - new_final) > 1e-9:
        print("  !! mismatch -- keeping BOTH, investigate before overwrite")
        mr.setdefault("gated_drift_recheck", {
            "ndcg": g["ndcg"], "js": g["js"],
            "admitted": g["admitted"]})
    else:
        mr["curves"]["gated"]["js"] = g["js"]
        mr["curves"]["gated"]["admitted"] = g["admitted"]
    (OUT / "main_results.json").write_text(json.dumps(mr, indent=1))
    print("  -> main_results.json gated js curve (honest admitted-set JS)")
    print("  gated js:", [round(j, 4) if j else None for j in g["js"]])
