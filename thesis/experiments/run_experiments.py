"""Main experiment driver: every table/figure number in the thesis.

Runs (deterministically, seeded) the full experiment grid and writes
results/ JSON files consumed by scripts/fill_numbers.py and
scripts/make_figures.py:

* main_results.json    -- Table 4.2 (main comparison) + significance
* rounds.json          -- iteration curves (Figure 4.2) per variant
* drift.json           -- JS drift curves (Figure 4.3) per variant
* ablation.json        -- Table 4.3 (module ablation)
* sensitivity.json     -- hyperparameter sensitivity (Table 4.4)
* sparsity.json        -- sparsity-stratified gains (Table 4.5)

Each experimental run is itself an AgentISA instruction stream; the
driver additionally exports one ELL trace per variant for
auditability.  Honesty rules: every number is a measurement; nothing
is extrapolated; failed runs abort the driver.
"""

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))          # repo root (agentflow)
sys.path.insert(0, str(HERE.parents[1] / "examples"))

import torch  # noqa: E402

from agentflow.core.isa.ell_export import (  # noqa: E402
    export_stream_to_ell, save_ell_trace,
)
from agentflow.sandbox.local_executor import LocalSandboxExecutor  # noqa: E402
from system_task_demo import inject_sandbox, run_stream  # noqa: E402

from rsix import data as data_mod  # noqa: E402
from rsix.models import GRU4Rec, BPRMF, ItemPop  # noqa: E402
from rsix.metrics import evaluate, paired_t_test  # noqa: E402
from rsix.streams import build_experiment_stream  # noqa: E402

VARIANTS = ["no_rsi", "naive", "conf_only", "no_adaptive", "gated"]
SEEDS = [0, 1, 2]
ROUNDS = 4
COMMON = dict(
    data_path="data/u.data",
    rounds=ROUNDS, epochs_init=60, epochs_round=5,
    mix_lambda=1.0, n_views=3, tau=0.01, margin=0.002,
)


def one_run(variant: str, seed: int, **over) -> Dict:
    """One deterministic RSI run; returns the round history."""
    config = dict(COMMON, variant=variant, seed=seed, **over)
    config["generator"] = over.get(
        "generator", "consistency" if variant == "gated" else "topk")

    def factory(seed):
        return GRU4Rec(n_items_global, dim=64, seed=seed)

    stream = build_experiment_stream(config, factory)
    sandbox = LocalSandboxExecutor(name=f"run-{variant}-{seed}")
    try:
        sandbox.write_file("data/u.data",
                           data_bytes_global)
        inject_sandbox(stream, sandbox)
        t0 = time.time()
        asyncio.run(run_stream(stream))
        wall = time.time() - t0
        hist = json.loads(sandbox.read_file(
            f"result_{variant}_{seed}.json").decode())
    finally:
        sandbox.close()
    hist["wall_s"] = wall
    return hist


def run_variant(variant: str, seeds=SEEDS, **over) -> List[Dict]:
    return [one_run(variant, s, **over) for s in seeds]


def agg(hists: List[Dict]) -> Dict:
    """Aggregate metric trajectories across seeds (mean per round)."""
    rounds = hists[0]["rounds"]
    out = {"ndcg": [], "hr": [], "admitted": [], "js": []}
    for r in range(len(rounds)):
        out["ndcg"].append(float(sum(
            h["rounds"][r]["metrics"]["ndcg@10"]
            for h in hists) / len(hists)))
        out["hr"].append(float(sum(
            h["rounds"][r]["metrics"]["hr@10"]
            for h in hists) / len(hists)))
        out["admitted"].append(float(sum(
            h["rounds"][r]["gate_stats"].get("n_admitted", 0)
            for h in hists) / len(hists)))
        out["js"].append(float(sum(
            h["rounds"][r]["gate_stats"].get("js", 0.0) or 0.0
            for h in hists) / len(hists)))
    return out


def final_metric(hists: List[Dict], key="ndcg@10") -> List[float]:
    return [h["rounds"][-1]["metrics"][key] for h in hists]


def main() -> None:
    global n_items_global, data_bytes_global
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/u.data")
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--quick", action="store_true",
                        help="smoke mode: 1 seed, 1 round")
    args = parser.parse_args()

    torch.set_num_threads(2)
    random.seed(0)
    out_dir = HERE / args.out
    out_dir.mkdir(exist_ok=True)

    train, test, n_users, n_items = data_mod.dataset(Path(args.data))
    n_items_global = n_items
    data_bytes_global = Path(args.data).read_bytes()
    print(f"dataset: {n_users} users, {n_items} items, "
          f"{sum(len(v) for v in train.values())} train events")

    seeds = SEEDS
    rounds = ROUNDS
    if args.quick:
        seeds, rounds = [0], 1
        COMMON["rounds"] = 1

    # ---- 0. non-RSI baselines (quick, direct) ---------------------
    pop_model = ItemPop(n_items)
    pop_model.fit(train)
    pop_eval = evaluate(pop_model.rank, train, test)
    bpr = BPRMF(n_users, n_items, dim=64, lr=0.01, l2=1e-4, seed=0)
    bpr.fit(train, epochs=60, rng=random.Random(0))
    bpr_eval = evaluate(bpr.rank, train, test)
    (out_dir / "baselines.json").write_text(json.dumps({
        "itempop": pop_eval, "bprmf": bpr_eval}, indent=1))
    print(f"baselines: ItemPop ndcg={pop_eval['ndcg@10']:.4f} "
          f"BPR-MF ndcg={bpr_eval['ndcg@10']:.4f}")

    # ---- 1. main comparison + curves ------------------------------
    print("== main comparison (5 variants x 3 seeds) ==")
    mains: Dict[str, List[Dict]] = {}
    for variant in VARIANTS:
        t0 = time.time()
        mains[variant] = run_variant(variant, seeds=seeds)
        agg_v = agg(mains[variant])
        print(f"  {variant:12s} ndcg "
              + " -> ".join(f"{v:.4f}" for v in agg_v["ndcg"])
              + f"   ({time.time() - t0:.0f}s)")

    baseline_finals = final_metric(mains["no_rsi"])
    table = {}
    for variant, hists in mains.items():
        finals = final_metric(hists)
        sig = paired_t_test(finals, baseline_finals) \
            if variant != "no_rsi" else None
        table[variant] = {
            "ndcg@10_mean": sum(finals) / len(finals),
            "ndcg@10_std": (
                sum((v - sum(finals) / len(finals)) ** 2
                    for v in finals) / (len(finals) - 1)) ** 0.5
            if len(finals) > 1 else 0.0,
            "hr@10_mean": sum(final_metric(hists, "hr@10"))
            / len(finals),
            "per_seed_ndcg": finals,
            "significance_vs_no_rsi": sig,
        }

    (out_dir / "main_results.json").write_text(json.dumps({
        "config": dict(COMMON, seeds=seeds, rounds=rounds),
        "table": table,
        "curves": {v: agg(h) for v, h in mains.items()},
    }, indent=1))
    print("  -> results/main_results.json")

    if args.quick:
        return  # smoke mode stops here

    # ---- 2. ablation ---------------------------------------------
    print("== ablation ==")
    ablations = {
        "full": mains["gated"],
        "no_dist_budget": mains["conf_only"],      # 置信轴 only
        "no_adaptive": mains["no_adaptive"],        # 固定50%准入
        "topk_gen": run_variant("gated", generator="topk"),
        "naive_gen": mains["naive"],
    }
    ab_table = {}
    for name, hists in ablations.items():
        finals = final_metric(hists)
        ab_table[name] = {
            "ndcg@10_mean": sum(finals) / len(finals),
            "per_seed": finals,
        }
    (out_dir / "ablation.json").write_text(json.dumps({
        "table": ab_table}, indent=1))
    print("  -> results/ablation.json")

    # ---- 3. sensitivity (eps, lambda, tau) ------------------------
    print("== sensitivity ==")
    # 敏感性: 形状分析用单种子(seed 0), 论文中如实标注
    sens = {}
    for eps in (0.02, 0.05, 0.08, 0.12):
        h = one_run("gated", 0, eps=eps)
        sens[f"eps={eps}"] = {"final_ndcg":
            h["rounds"][-1]["metrics"]["ndcg@10"]}
    for lam in (0.5, 1.0, 2.0):
        h = one_run("gated", 0, mix_lambda=lam)
        sens[f"lambda={lam}"] = {"final_ndcg":
            h["rounds"][-1]["metrics"]["ndcg@10"]}
    for tau in (0.005, 0.01, 0.02):
        h = one_run("gated", 0, tau=tau)
        sens[f"tau={tau}"] = {"final_ndcg":
            h["rounds"][-1]["metrics"]["ndcg@10"]}
    (out_dir / "sensitivity.json").write_text(json.dumps(
        {"table": sens}, indent=1))
    print("  -> results/sensitivity.json")

    # ---- 4. sparsity stratification -------------------------------
    print("== sparsity stratification ==")
    # 用 gated 主跑的逐用户指标按历史长度分层 —— 重跑一次逐用户记录
    strat = {"sparse": [], "medium": [], "dense": []}
    for u, gain in per_user_gains(train, test, seed=0):
        L = len(train[u])
        bucket = "sparse" if L <= 20 else (
            "medium" if L <= 60 else "dense")
        strat[bucket].append(gain)
    sparsity_table = {
        b: {"mean_ndcg_gain": sum(v) / len(v), "n_users": len(v)}
        for b, v in strat.items()}
    (out_dir / "sparsity.json").write_text(json.dumps(
        {"table": sparsity_table}, indent=1))
    print("  -> results/sparsity.json")

    # ---- 5. ELL trace for the gated pipeline ----------------------
    config = dict(COMMON, variant="gated", seed=0)
    config["generator"] = "consistency"
    stream = build_experiment_stream(
        config, lambda seed: GRU4Rec(n_items, dim=64, seed=seed))
    workflow, messages = asyncio.run(export_stream_to_ell(
        stream, name="rsi-gated",
        description="Fidelity-gated RSI closed loop over ML-100K"))
    save_ell_trace(out_dir / "rsi_gated_trace.jsonl",
                   workflow, messages)
    print("  -> results/rsi_gated_trace.jsonl")


def per_user_gains(train, test, seed=0):
    """Direct (non-stream) computation of per-user NDCG@10 gains.

    Runs the gated and no_rsi trajectories once with the same data and
    seed, then evaluates the FINAL models user by user.  Honest note:
    this recomputes the two trajectories instead of persisting models
    through the stream, keeping the experiment pipeline stateless.
    """
    import numpy as np
    from rsix.generation import consistency_candidates, augment_views
    from rsix import fidelity as fid
    import random as _random

    rng = _random.Random(seed)
    pop = data_mod.popularity(train, n_items_global)
    gate = fid.FidelityGate(eps=0.08, z_min=0.0,
                           mix_lambda=1.0).fit(pop)
    real_n = int(pop.sum())

    def trajectory(variant):
        model = GRU4Rec(n_items_global, dim=64, seed=seed)
        model.fit(train, epochs=60, rng=_random.Random(seed))
        current = train
        for r in range(4):
            if variant == "gated":
                cands, _ = consistency_candidates(
                    model, current, rng=_random.Random(seed * 31 + r))
                admitted, _ = gate.score_and_filter(cands, real_n)
            else:
                admitted = []
            current = {u: list(items) for u, items in train.items()}
            for u, i in admitted:
                current.setdefault(u, []).append(i)
            model.fit(current, epochs=5, rng=_random.Random(seed * 977 + r))
        return model

    m_rsi = trajectory("gated")
    m_base = trajectory("no_rsi")
    from rsix.metrics import rank_of_target
    for u, target in sorted(test.items()):
        seen = train[u]
        r_rsi = rank_of_target(m_rsi.rank(u, seen), seen, target)
        r_base = rank_of_target(m_base.rank(u, seen), seen, target)
        ndcg = lambda r: 1.0 / np.log2(r + 1) if r <= 10 else 0.0
        yield u, ndcg(r_rsi) - ndcg(r_base)


if __name__ == "__main__":
    main()
