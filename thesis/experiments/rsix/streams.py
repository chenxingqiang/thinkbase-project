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

import numbers

from agentflow.core.isa.instruction import (Instruction,
                                             InstructionType as IsaType,
                                             TypeConstraint)

from . import data as data_mod
from . import fidelity as fid
from . import rsi as rsi_mod
from .metrics import evaluate


def _instruction(id: str, itype: IsaType, execute: Callable, *,
                 deps: List[str], outputs: List[str], side_effects: bool,
                 description: str = "",
                 post: Optional[Callable[[Dict[str, Any], Any],
                                         Optional[str]]] = None,
                 dep_types: Optional[Dict[str, TypeConstraint]] = None
                 ) -> Instruction:
    """(p, self) -> result factory, mirroring the suites' pattern.

    Three AgentISA layers are wired per instruction:

    * ``T`` (types): ``dep_types`` maps dataflow edges to
      ``TypeConstraint``s; they compose into ONE dependent type on the
      ``ctx`` parameter (Def 3.2: the type of ctx depends on a
      predicate over its contents) and are checked through the
      AgentISA-native ``validate_types`` before execution.
    * ``Phi-pre`` (preconditions): the declared dataflow itself --
      every dep must be present in ctx -- verified through the native
      ``verify_conditions``.
    * ``Phi-post`` (postconditions): ``post`` is an invariant checker
      with signature ``(ctx, result) -> None-or-violation-message``;
      ``None`` means the postcondition holds.  Registered in the
      native context record and enforced by this wrapper (the base
      class only auto-verifies preconditions, so the wrapper is the
      documented evaluation point for the post-state).
    """

    dep_map = dict(dep_types or {})

    def _check_deps(state: Dict[str, Any]) -> bool:
        ctx = state.get("ctx", {})
        return all(d in ctx for d in deps)

    def _ctx_typed(ctx: Dict[str, Any]) -> bool:
        for d, tc in dep_map.items():
            if d not in ctx or not _conforms(ctx[d], tc):
                return False
        return True

    class _Step(Instruction):
        async def execute(self, input_params: Dict[str, Any]) -> Any:
            # T: the typed dataflow contract on ctx (native path).
            if not self.validate_types(input_params):
                ctx = input_params.get("ctx", {})
                why = "ctx is not a dict"
                for d, tc in dep_map.items():
                    if d not in ctx:
                        why = f"missing ctx key '{d}'"
                        break
                    if not _conforms(ctx[d], tc):
                        why = f"ctx['{d}'] violates its type contract"
                        break
                raise RuntimeError(
                    f"[{id}] type check failed: {why}")
            # Phi: precondition = declared dataflow is resolvable.
            if not self.verify_conditions(input_params):
                missing = [d for d in deps
                           if d not in input_params.get("ctx", {})]
                raise RuntimeError(
                    f"[{id}] precondition violated: missing ctx keys "
                    f"{missing} (dataflow declared but not produced)")
            start = time.time()
            result = execute(input_params, self)
            # Phi: postcondition = invariant over the post-state.
            if post is not None:
                why = post(input_params.get("ctx", {}), result)
                if why is not None:
                    raise RuntimeError(
                        f"[{id}] postcondition violated: {why}")
            self._update_metrics(start, True)
            return result

    step = _Step(id=id, name=id, type=itype,
                 params={"ctx": TypeConstraint(dict,
                                               dependent_on=_ctx_typed)},
                 description=description or id,
                 side_effects=side_effects, outputs=outputs)
    step.dependencies = list(deps)
    step.add_precondition(_check_deps)
    if post is not None:
        # Native record for the formal view; evaluated by the wrapper
        # above (the base runtime only pre-verifies).
        step.add_postcondition(
            lambda state, _p=post: _p(state.get("ctx", {}), None) is None)
    step.sandbox = None
    return step


def _conforms(value, tc: TypeConstraint) -> bool:
    """Single-value form of the base class's validate_types logic."""
    try:
        if value is None:
            return bool(tc.nullable)
        if not isinstance(value, tc.base_type):
            return False
        if tc.dependent_on is not None and not tc.dependent_on(value):
            return False
        if tc.validation is not None and not tc.validation(value):
            return False
        return True
    except Exception:
        return False


def _is_int_pair(v) -> bool:
    return (isinstance(v, (tuple, list)) and len(v) == 2
            and all(isinstance(x, numbers.Integral) for x in v))


def _is_int_quad(v) -> bool:
    return (isinstance(v, (tuple, list)) and len(v) == 4
            and isinstance(v[0], numbers.Integral)
            and isinstance(v[1], numbers.Integral)
            and isinstance(v[2], numbers.Real)
            and isinstance(v[3], numbers.Real))


# Shared edge contracts: each declared dependency of a step carries a
# TypeConstraint -- Def 3.2's dependent types realized as predicates
# over the value flowing on that dataflow edge.
T_INTERACTIONS = TypeConstraint(
    list, dependent_on=lambda v: all(_is_int_pair(t) for t in v))
T_SEQUENCES = TypeConstraint(
    dict, dependent_on=lambda v: all(
        isinstance(k, numbers.Integral)
        and isinstance(seq, list)
        and all(isinstance(i, numbers.Integral) for i in seq)
        for k, seq in v.items()))
T_TEST_MAP = TypeConstraint(
    dict, dependent_on=lambda v: all(
        isinstance(k, numbers.Integral) and isinstance(i, numbers.Integral)
        for k, i in v.items()))
T_POP = TypeConstraint(
    np.ndarray, dependent_on=lambda v: v.ndim == 1 and v.size > 0
    and bool(np.all(np.isfinite(v))) and bool(np.all(v >= 0)))
def _gate_fitted(g) -> bool:
    """Total predicate: a fitted gate carries a positive 1-D real
    distribution over strata.  Never raises, including on the
    constructed-but-unfitted gate (real_dist is None)."""
    if not (hasattr(g, "real_dist") and hasattr(g, "score_and_filter")):
        return False
    rd = getattr(g, "real_dist", None)
    if rd is None:
        return False
    try:
        arr = np.asarray(rd)
        return arr.ndim == 1 and arr.dtype.kind in "fi" \
            and float(arr.sum()) > 0.0
    except Exception:
        return False


T_GATE = TypeConstraint(object, dependent_on=_gate_fitted)
T_CANDIDATES = TypeConstraint(
    list, dependent_on=lambda v: all(_is_int_quad(t) for t in v))
T_ADMITTED = TypeConstraint(
    list, dependent_on=lambda v: all(_is_int_pair(t) for t in v))
T_GATE_STATS = TypeConstraint(
    dict, dependent_on=lambda st: (
        isinstance(st.get("n_candidates"), numbers.Integral)
        and isinstance(st.get("n_admitted"), numbers.Integral)))
T_METRICS = TypeConstraint(
    dict, dependent_on=lambda m: all(
        isinstance(m.get(k), numbers.Real) and 0.0 <= m[k] <= 1.0
        for k in ("ndcg@10", "hr@10")))
T_MODEL = TypeConstraint(
    object, dependent_on=lambda m: callable(getattr(m, "fit", None))
    and callable(getattr(m, "rank", None)))
T_MIXBOOK = TypeConstraint(
    dict, dependent_on=lambda b: (
        isinstance(b.get("real_events"), numbers.Integral)
        and isinstance(b.get("synthetic_events"), numbers.Integral)
        and b["real_events"] >= 0 and b["synthetic_events"] >= 0))


def check_gate_invariant(variant: str, eps: float, rkey: str):
    """Postcondition factory for the gate step, per experiment variant.

    For ``gated`` this machine-checks INV-1 -- the mixed distribution
    must stay inside the eps budget on every round of every run -- and
    for the baselines it pins their defining behavior (naive admits
    everything, no_rsi admits nothing).
    """

    def post(ctx, res):
        admitted = res[f"admitted_{rkey}"]
        stats = res[f"gate_stats_{rkey}"]
        n_c, n_a = stats["n_candidates"], stats["n_admitted"]
        if n_a > n_c:
            return f"admitted {n_a} > candidates {n_c}"
        if variant == "naive" and n_a != n_c:
            return f"naive must admit everything ({n_a}/{n_c})"
        if variant == "no_rsi" and n_a != 0:
            return f"no_rsi must admit nothing (got {n_a})"
        if variant in ("conf_only", "no_adaptive"):
            if "synthetic_js_distance" not in stats:
                return f"{variant}: missing synthetic_js_distance stat"
        if variant == "gated":
            js = stats.get("mixed_js_distance")
            if js is None or js > eps + 1e-9:
                return f"INV-1 violated: mixed JS {js} > eps {eps}"
        if len(admitted) != n_a:
            return f"admitted list ({len(admitted)}) != stat ({n_a})"
        return None

    return post


def check_mix_invariant(rkey: str):
    """Postcondition factory for the mix step: mixture bookkeeping.

    The synthetic-event count must equal admitted x repetition factor
    (the checkable form of the mixing guarantee INV-2), and the real
    data must survive the mixture untouched.
    """

    def post(ctx, res):
        book = res["last_mix"]
        n_adm = len(ctx[f"admitted_{rkey}"])
        reps = max(1, round(ctx["gate"].mix_lambda))
        expect = n_adm * reps
        if book["synthetic_events"] != expect:
            return (f"mixture bookkeeping broken: synthetic="
                    f"{book['synthetic_events']} expected={expect}")
        if book["real_events"] <= 0:
            return "mixture lost the real data (real_events<=0)"
        if not res["train"]:
            return "mixed training set is empty"
        return None

    return post


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
        description="Read the rating file (positive interactions)",
        post=lambda ctx, res: None if res["interactions"]
        else "ingested interaction list is empty"))

    def _split(p, s):
        triples = p["ctx"]["interactions"]
        compact, n_users, n_items = data_mod.remap(triples)
        train, test = data_mod.leave_one_out(compact, n_users)
        pop = data_mod.popularity(train, n_items)
        return {"train": train, "test": test, "n_items": n_items,
                "pop": pop, "n_users": n_users}

    def _split_post(ctx, res):
        train, test = res["train"], res["test"]
        if not train:
            return "split produced an empty training set"
        if not 0 < len(test) <= res["n_users"]:
            return f"test size {len(test)} outside (0, n_users]"
        for u, t in test.items():
            if t in train.get(u, []):
                return f"leakage: test item of user {u} is in its train seq"
        return None

    stream.append(_instruction(
        "build_split", IsaType.CONTROL_FLOW, _split,
        deps=["interactions"], outputs=["train", "test", "n_items",
                                        "pop", "n_users"],
        side_effects=False,
        description="Leave-one-out split + item popularity",
        post=_split_post,
        dep_types={"interactions": T_INTERACTIONS}))

    def _fit_gate(p, s):
        gate = fid.FidelityGate(
            eps=config.get("eps", 0.08),
            z_min=config.get("z_min", 0.0),
            mix_lambda=config.get("mix_lambda", 0.5),
        ).fit(p["ctx"]["pop"])
        return {"gate": gate}

    def _gate_fit_post(ctx, res):
        g = res["gate"]
        total = float(np.asarray(g.real_dist).sum())
        if len(np.asarray(g.real_dist)) < 2:
            return "gate strata degenerate (<2 strata)"
        if abs(total - 1.0) > 1e-6:
            return f"gate real distribution not normalized (sum={total})"
        return None

    stream.append(_instruction(
        "fit_fidelity_gate", IsaType.CONTROL_FLOW, _fit_gate,
        deps=["pop"], outputs=["gate"], side_effects=False,
        description="Fit popularity strata + real distribution",
        post=_gate_fit_post,
        dep_types={"pop": T_POP}))

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

        def _train_post(ctx, res, rkey=rkey):
            m = res[f"metrics_{rkey}"]
            for k in ("ndcg@10", "hr@10"):
                v = m.get(k)
                if not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0:
                    return f"metric {k} out of contract: {v!r}"
            return None

        stream.append(_instruction(
            f"train_round_{r}", IsaType.CONTROL_FLOW, _train_round,
            deps=["train", "test"], outputs=[f"model_{rkey}",
                                             f"metrics_{rkey}",
                                             "last_model"],
            side_effects=False,
            description=f"Round {r}: train + evaluate",
            post=_train_post,
            dep_types={"train": T_SEQUENCES, "test": T_TEST_MAP}))

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

        def _gen_post(ctx, res, r=r):
            cands = res["candidates"]
            if not isinstance(cands, list):
                return f"candidates not a list: {type(cands)}"
            # per-user cap depends on the generator: consistency emits
            # at most one pseudo-label per user; topk emits <= topk.
            kind = config.get("generator",
                              "consistency" if variant == "gated"
                              else "topk")
            cap = (len(ctx["train"]) if kind == "consistency"
                   else config.get("topk", 10) * len(ctx["train"]))
            if len(cands) > cap:
                return (f"candidates {len(cands)} exceed per-generator "
                        f"cap {cap} ({kind})")
            for c in cands:
                if len(c) != 4:
                    return f"malformed candidate tuple: {c!r}"
            return None

        stream.append(_instruction(
            f"generate_{rkey}", IsaType.LLM_INTERACTION, _generate,
            deps=["last_model", "train"], outputs=["candidates"],
            side_effects=False,
            description=f"Round {r}: model self-generates candidates",
            post=_gen_post,
            dep_types={"last_model": T_MODEL, "train": T_SEQUENCES}))

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
                         # unambiguous key consumed by the INV-1
                         # postcondition checker
                         "mixed_js_distance": st["mixed_js_distance"],
                         # legacy alias: agg()/figures read "js" over
                         # the committed result JSONs; for gated it IS
                         # the mixed JS
                         "js": st["mixed_js_distance"]}
            return {f"admitted_{rkey}": admitted,
                    f"gate_stats_{rkey}": stats}

        stream.append(_instruction(
            f"gate_{rkey}", IsaType.CONTROL_FLOW, _gate,
            deps=["candidates", "gate", "pop"], outputs=[f"admitted_{rkey}",
                                                        f"gate_stats_{rkey}"],
            side_effects=False,
            description=f"Round {r}: fidelity gate over candidates",
            post=check_gate_invariant(variant, config.get("eps", 0.08),
                                      rkey),
            dep_types={"candidates": T_CANDIDATES,
                       "gate": T_GATE, "pop": T_POP}))

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
            description=f"Round {r}: mix real + synthetic training data",
            post=check_mix_invariant(rkey),
            dep_types={"train": T_SEQUENCES,
                       f"admitted_{rkey}": T_ADMITTED,
                       "gate": T_GATE}))

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

    # Declare every ctx key the writer reads at runtime: the compiler's
    # dataflow graph is built from these edges, and an undeclared read is
    # a latent DCE hazard (the producer could be legally eliminated).
    _write_deps = [f"metrics_r{r}" for r in range(rounds + 1)]
    _write_deps += [f"gate_stats_r{r}" for r in range(rounds)]
    _write_deps += ["last_mix"]
    def _write_post(ctx, res):
        sha = res.get("report_written", "")
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            return f"report sha256 contract broken: {sha!r}"
        return None

    _write_types = {f"metrics_r{r}": T_METRICS for r in range(rounds + 1)}
    _write_types.update({f"gate_stats_r{r}": T_GATE_STATS
                         for r in range(rounds)})
    _write_types["last_mix"] = T_MIXBOOK
    stream.append(_instruction(
        "write_results", IsaType.STATE_MANAGEMENT, _write,
        deps=_write_deps, outputs=[], side_effects=True,
        description="Persist the round-by-round history",
        post=_write_post, dep_types=_write_types))
    return stream


def _rng(seed: int, r: int):
    import random
    return random.Random(seed * 977 + r)
