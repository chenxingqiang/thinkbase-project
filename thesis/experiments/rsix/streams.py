"""AgentISA orchestration: one RSI experiment = one instruction stream.

Each recursive self-improvement run is expressed as an AgentISA
instruction stream -- ingest (STATE) -> split (CONTROL) -> fit gate
(CONTROL) -> per round [generate (model-backed) -> gate (CONTROL) ->
retrain+eval (CONTROL)] -> write results (STATE) -- executed by the
same runtime as the rest of this repository, with an ELL trace
exported per run.  The orchestration is infrastructure, not a research
claim: it makes every experiment run inspectable and reproducible.
"""

import hashlib
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from agentflow.core.isa.instruction import Instruction, InstructionType as IsaType

from . import data as data_mod
from . import fidelity as fid
from . import rsi as rsi_mod
from .metrics import evaluate


def _instruction(id: str, itype: IsaType, execute: Callable, *,
                 deps: List[str], outputs: List[str], side_effects: bool,
                 description: str = "") -> Instruction:
    """(p, self) -> result factory, mirroring the suites' pattern."""

    class _Step(Instruction):
        async def execute(self, input_params: Dict[str, Any]) -> Any:
            start = time.time()
            result = execute(input_params, self)
            self._update_metrics(start, True)
            return result

    step = _Step(id=id, name=id, type=itype, params={},
                 description=description or id,
                 side_effects=side_effects, outputs=outputs)
    step.dependencies = list(deps)
    step.sandbox = None
    return step


def build_experiment_stream(config: Dict[str, Any],
                            model_factory: Callable) -> List[Instruction]:
    """Materialize one RSI experiment config as an instruction stream."""
    variant = config["variant"]
    rounds = config.get("rounds", 5)
    stream: List[Instruction] = []

    stream.append(_instruction(
        "ingest_dataset", IsaType.STATE_MANAGEMENT,
        lambda p, s: {"interactions": data_mod.load_interactions(
            Path(config["data_path"]))},
        deps=[], outputs=["interactions"], side_effects=True,
        description="Read the rating file (positive interactions)"))

    def _split(p, s):
        triples = p["ctx"]["interactions"]
        compact, n_users, n_items = data_mod.remap(triples)
        train, test = data_mod.leave_one_out(compact, n_users)
        pop = data_mod.popularity(train, n_items)
        return {"train": train, "test": test, "n_items": n_items,
                "pop": pop, "n_users": n_users}

    stream.append(_instruction(
        "build_split", IsaType.CONTROL_FLOW, _split,
        deps=["interactions"], outputs=["train", "test", "n_items",
                                        "pop", "n_users"],
        side_effects=False,
        description="Leave-one-out split + item popularity"))

    def _fit_gate(p, s):
        gate = fid.FidelityGate(
            eps=config.get("eps", 0.08),
            z_min=config.get("z_min", 0.0),
            mix_lambda=config.get("mix_lambda", 0.5),
        ).fit(p["ctx"]["pop"])
        return {"gate": gate}

    stream.append(_instruction(
        "fit_fidelity_gate", IsaType.CONTROL_FLOW, _fit_gate,
        deps=["pop"], outputs=["gate"], side_effects=False,
        description="Fit popularity strata + real distribution"))

    # ---- per-round closed loop -----------------------------------------
    for r in range(rounds + 1):
        rkey = f"r{r}"

        def _train_round(p, s, r=r, rkey=rkey):
            # Recursive self-improvement trains the SAME system: round 0
            # initializes the model; every later round continues from
            # the previous round's weights on the mixed data.
            ctx = p["ctx"]
            if r == 0:
                model = model_factory(seed=config["seed"])
                info = model.fit(ctx["train"], epochs=config["epochs_init"],
                                rng=_rng(config["seed"], 0))
            else:
                model = ctx["last_model"]
                info = model.fit(ctx["train"],
                                 epochs=config["epochs_round"],
                                 rng=_rng(config["seed"], r))
            metrics = evaluate(model.rank, ctx["train"], ctx["test"])
            return {f"model_{rkey}": model,
                    f"metrics_{rkey}": metrics, "last_model": model,
                    "info": info}

        stream.append(_instruction(
            f"train_round_{r}", IsaType.CONTROL_FLOW, _train_round,
            deps=["train", "test"], outputs=[f"model_{rkey}",
                                             f"metrics_{rkey}",
                                             "last_model"],
            side_effects=False,
            description=f"Round {r}: train + evaluate"))

        if r == rounds:
            break

        def _generate(p, s, r=r, rkey=rkey):
            # generator is an orthogonal axis: gated defaults to the
            # FixMatch-style consistency generator; naive keeps the
            # self-reinforcing top-k generator (the motivation case)
            from . import generation as gen_mod
            kind = config.get(
                "generator",
                "consistency" if variant == "gated" else "topk")
            if kind == "consistency":
                cands, gstats = gen_mod.consistency_candidates(
                    p["ctx"]["last_model"], p["ctx"]["train"],
                    n_views=config.get("n_views", 3),
                    tau=config.get("tau", 0.01),
                    margin=config.get("margin", 0.002),
                    rng=_rng(config["seed"] * 31 + r, 7))
                return {"candidates": cands, "gen_stats": gstats}
            cands = gen_mod.topk_candidates(
                p["ctx"]["last_model"], p["ctx"]["train"],
                topk=config.get("topk", 10))
            return {"candidates": cands,
                    "gen_stats": {"users": 0, "candidates": len(cands)}}

        stream.append(_instruction(
            f"generate_{rkey}", IsaType.LLM_INTERACTION, _generate,
            deps=["last_model", "train"], outputs=["candidates"],
            side_effects=False,
            description=f"Round {r}: model self-generates candidates"))

        def _gate(p, s, r=r, rkey=rkey):
            ctx = p["ctx"]
            gate: fid.FidelityGate = ctx["gate"]
            real_n = int(ctx["pop"].sum())
            cands = ctx["candidates"]
            if variant == "naive":
                admitted = [(u, i) for u, i, _, _ in cands]
                dist = fid.stratum_distribution(
                    [i for _, i in admitted], gate.head_of)
                stats = {"n_candidates": len(cands),
                         "n_admitted": len(admitted),
                         "js": fid.js_distance(dist, gate.real_dist)}
            elif variant == "no_rsi":
                admitted, stats = [], {"n_candidates": len(cands),
                                       "n_admitted": 0, "js": 0.0}
            elif variant == "conf_only":
                admitted, st = gate.filter_confidence_only(cands, real_n)
                stats = {"n_candidates": st["n_candidates"],
                         "n_admitted": st["n_admitted"],
                         "js": st["synthetic_js_distance"]}
            elif variant == "no_adaptive":
                admitted, st = gate.filter_no_adaptive(cands, real_n)
                stats = {"n_candidates": st["n_candidates"],
                         "n_admitted": st["n_admitted"],
                         "js": st["synthetic_js_distance"]}
            else:  # gated (ours)
                admitted, st = gate.score_and_filter(cands, real_n)
                stats = {"n_candidates": st["n_candidates"],
                         "n_admitted": st["n_admitted"],
                         "js": st["mixed_js_distance"]}
            return {f"admitted_{rkey}": admitted,
                    f"gate_stats_{rkey}": stats}

        stream.append(_instruction(
            f"gate_{rkey}", IsaType.CONTROL_FLOW, _gate,
            deps=["candidates", "gate", "pop"], outputs=[f"admitted_{rkey}",
                                                        f"gate_stats_{rkey}"],
            side_effects=False,
            description=f"Round {r}: fidelity gate over candidates"))

        def _mix(p, s, r=r, rkey=rkey):
            ctx = p["ctx"]
            admitted = ctx[f"admitted_{rkey}"]
            mixed = {u: list(items) for u, items in ctx["train"].items()}
            reps = max(1, round(ctx["gate"].mix_lambda))
            for u, i in admitted:
                for _ in range(reps):
                    mixed.setdefault(u, []).append(i)
            return {"train": mixed, "last_mix": {
                "real_events": sum(len(v) for v in ctx["train"].values()),
                "synthetic_events": len(admitted) * reps}}

        stream.append(_instruction(
            f"mix_{rkey}", IsaType.CONTROL_FLOW, _mix,
            deps=["train", f"admitted_{rkey}", "gate"],
            outputs=["train", "last_mix"], side_effects=False,
            description=f"Round {r}: mix real + synthetic training data"))

    def _write(p, s):
        ctx = p["ctx"]
        rounds = config.get("rounds", 5)
        history = {
            "variant": variant, "seed": config["seed"],
            "rounds": [
                {"round": r,
                 "metrics": ctx[f"metrics_r{r}"],
                 "gate_stats": ctx.get(f"gate_stats_r{r}", {}),
                 "mix": ctx.get(f"last_mix", {}) if r < rounds else {}}
                for r in range(rounds + 1)
            ],
        }
        body = str(history)  # avoid torch tensors in JSON
        import json
        payload = json.dumps(history, default=str, indent=1)
        s.sandbox.write_file(
            f"result_{variant}_{config['seed']}.json", payload.encode())
        return {"history": history,
                "report_written": hashlib.sha256(
                    payload.encode()).hexdigest()}

    stream.append(_instruction(
        "write_results", IsaType.STATE_MANAGEMENT, _write,
        deps=[f"metrics_r{rounds}"], outputs=[], side_effects=True,
        description="Persist the round-by-round history"))
    return stream


def _rng(seed: int, r: int):
    import random
    return random.Random(seed * 977 + r)
