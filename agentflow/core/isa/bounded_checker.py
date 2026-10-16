"""Bounded verification checker for AgentISA (paper Definition 3.5).

This module ships the *bounded explorer* that Section 3.3.3 of the
AgentISA paper describes.  For every pure (side-effect-free)
instruction it constructs a bounded set of candidate input states from
the instruction's declared parameter constraints and checks the
per-instruction validity obligation

    pre(i, s)  =>  post(i, exec(i, s))

on each state.  Concretely, per candidate state the checker requires:

1. well-typedness -- ``instruction.validate_types(state)``;
2. preconditions -- ``instruction.verify_conditions(state)`` (the
   callables registered via ``add_precondition``) plus any explicit
   ``pre_condition`` callables passed to :func:`verify_instruction`;
3. only for states that pass both, the instruction is *executed* and
   three obligations are evaluated over the post-state (the execution
   result):

   - progress: execution must not raise while the precondition holds;
   - preservation: declared ``outputs`` must be present in the result
     and well-typed where ``context.type_env`` declares a constraint;
   - postcondition: the explicit ``post_condition`` callables plus the
     callables registered via ``add_postcondition`` must hold.

Side-effectful instructions (``side_effects=True``, the conservative
default) are never executed by the checker and are reported as out of
scope: their validity obligation is not discharged here.

State-space construction is deterministic and bounded:

- ``bool`` parameters enumerate ``{True, False}`` and enum parameters
  enumerate all members (exhaustive axes);
- bounded ``int`` parameters probe ``{-1, 0, 1, 2}`` (narrowed by the
  constraint's ``dependent_on``/``validation`` predicates when
  present), ``str`` parameters probe a small fixed set, and list /
  tuple / set / dict / bytes parameters probe short deterministic
  templates (bounded probes, not exhaustive);
- non-enumerable domains such as ``float`` use deterministic seeded
  samples from ``random.Random(seed)`` (sampled axes);
- types the checker cannot construct are reported as unenumerable and
  produce no states (the obligation is *not* discharged);
- if the instruction consumes a ``ctx`` dict (its ``dependencies``
  list), states are built as ``{"ctx": {...}}`` with one probed value
  per required key;
- the cross product of all axes is capped at ``max_states`` (default
  256); when the full product exceeds the cap, a deterministic seeded
  subset is explored and the verdict reports ``truncated``.

Because the enumeration is bounded, the checker is sound but
incomplete on probe/sampled/truncated spaces: a passing verdict means
the obligation was discharged on every *explored* state, and
``exhaustive`` records whether the explored set covered the whole
declared-finite space.  A verdict only counts as ``passed`` when at
least one state reached execution, so vacuous checks are never
reported as success.

This module is intentionally independent of
``agentflow.core.isa.verifier`` and ``agentflow.core.isa.validation``
(the runtime/formal machinery): it depends only on the
:class:`~agentflow.core.isa.instruction.Instruction` base class, the
standard library, and numpy (guarded, for ndarray parameter probes).
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import math
import random
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .instruction import Instruction, TypeConstraint

try:  # numpy is optional; used only to probe ndarray-typed parameters
    import numpy as _np
except Exception:  # pragma: no cover - numpy simply not installed
    _np = None

__all__ = [
    "DEFAULT_MAX_STATES",
    "DEFAULT_SEED",
    "CTX_VALUE_POOL",
    "INT_PROBE",
    "STR_PROBE",
    "Violation",
    "StateSpace",
    "InstructionVerdict",
    "StreamVerdict",
    "enumerate_state_space",
    "generate_state_space",
    "verify_instruction",
    "verify_instruction_async",
    "verify_stream",
    "verify_stream_async",
]

# --- bounded-enumeration constants (all deterministic) ---------------------

DEFAULT_MAX_STATES = 256
DEFAULT_SEED = 0

#: Bounded probe for ``int`` parameters, narrowed by declared constraints.
INT_PROBE: Tuple[int, ...] = (-1, 0, 1, 2)
#: Deterministic widening scan used when narrowing empties the int probe.
INT_WIDEN_SCAN: Tuple[int, ...] = tuple(range(0, 33)) + tuple(range(-1, -33, -1))
#: Small deterministic probe set for ``str`` parameters.
STR_PROBE: Tuple[str, ...] = ("", "a", "abc", "hello")
#: Deterministic widening scan for narrowed-away string probes.
STR_WIDEN_SCAN: Tuple[str, ...] = ("0", "1", "42", "x", "xy", "AgentISA", "hello world", "0123456789")
#: Fixed anchor values for ``float`` parameters (seeded samples are added).
FLOAT_ANCHORS: Tuple[float, ...] = (0.0, -1.5, 1.5)
#: Number of seeded samples drawn per ``float`` axis.
FLOAT_SAMPLE_COUNT = 3
#: Deterministic probe values for untyped ``ctx`` keys (dependencies).
CTX_VALUE_POOL: Tuple[Any, ...] = (1, 0, -1, True, False, "", "abc", None, 1.5, [0, 1])
#: Element pool used to build short list/tuple/set probes.
LIST_ELEMENT_POOL: Tuple[Any, ...] = (0, "a")
#: Maximum values accepted from a widening scan.
WIDEN_LIMIT = 4
#: Maximum seeded draws attempted while widening a float axis.
WIDEN_TRIES = 32

_PRE = Callable[[Dict[str, Any]], bool]


# --- reporting dataclasses --------------------------------------------------


@dataclass
class Violation:
    """One failed obligation on one explored state.

    ``kind`` is one of ``"progress"`` (precondition held but execution
    raised), ``"preservation"`` (declared outputs missing or
    ill-typed), or ``"postcondition"``.  ``state`` is the pristine
    enumerated input state; ``detail`` is a human-readable cause.
    """

    kind: str
    state: Dict[str, Any]
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "state": self.state, "detail": self.detail}


@dataclass
class StateSpace:
    """The bounded set of candidate input states for one instruction."""

    states: List[Dict[str, Any]]
    exhaustive: bool
    sampled: bool
    truncated: bool
    total_candidates: int
    axes: Dict[str, List[Any]]
    modes: Dict[str, str]
    notes: List[str] = field(default_factory=list)


@dataclass
class InstructionVerdict:
    """Result of checking one instruction against Definition 3.5.

    Required summary fields (paper Section 3.3.3):

    - ``checked_states``: states on which the implication was evaluated
      (well-typed, precondition-satisfying, execution attempted);
    - ``exhaustive``: whether the explored states covered the whole
      declared-finite space without truncation;
    - ``progress_violations`` / ``preservation_violations`` /
      ``postcondition_violations``: counts per obligation;
    - ``passed``: all three violation counts are zero *and* at least
      one state reached execution.

    ``skipped`` verdicts (side-effectful instructions) have
    ``passed=False`` and ``skip_reason`` set: out of scope, not
    discharged.
    """

    instruction_id: str
    instruction_name: str
    status: str
    skipped: bool
    passed: bool
    checked_states: int
    exhaustive: bool
    progress_violations: int
    preservation_violations: int
    postcondition_violations: int
    # --- supplementary diagnostics ---
    candidate_states: int = 0
    well_typed_states: int = 0
    precondition_errors: int = 0
    sampled: bool = False
    truncated: bool = False
    skip_reason: Optional[str] = None
    violations: List[Violation] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction_id": self.instruction_id,
            "instruction_name": self.instruction_name,
            "status": self.status,
            "skipped": self.skipped,
            "passed": self.passed,
            "checked_states": self.checked_states,
            "exhaustive": self.exhaustive,
            "progress_violations": self.progress_violations,
            "preservation_violations": self.preservation_violations,
            "postcondition_violations": self.postcondition_violations,
            "candidate_states": self.candidate_states,
            "well_typed_states": self.well_typed_states,
            "precondition_errors": self.precondition_errors,
            "sampled": self.sampled,
            "truncated": self.truncated,
            "skip_reason": self.skip_reason,
            "notes": list(self.notes),
            "violations": [v.to_dict() for v in self.violations],
        }


@dataclass
class StreamVerdict:
    """Result of :func:`verify_stream`: pure instructions checked,
    side-effectful (or otherwise uncheckable) ones listed as skipped."""

    verdicts: List[InstructionVerdict]
    checked: List[InstructionVerdict]
    skipped: List[InstructionVerdict]
    passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction_ids": [v.instruction_id for v in self.verdicts],
            "passed": self.passed,
            "total_checked": len(self.checked),
            "total_skipped": len(self.skipped),
            "checked": [v.to_dict() for v in self.checked],
            "skipped": [v.to_dict() for v in self.skipped],
        }


# --- small helpers ----------------------------------------------------------


def _brief(value: Any) -> str:
    """Deterministic, length-capped repr for violation details."""
    try:
        text = repr(value)
    except Exception:  # pragma: no cover - defensive
        text = f"<unreprable {type(value).__name__}>"
    return text if len(text) <= 120 else text[:117] + "..."


def _as_callables(spec: Any, argname: str) -> List[_PRE]:
    """Normalize None / callable / sequence-of-callables."""
    if spec is None:
        return []
    if callable(spec):
        return [spec]
    if isinstance(spec, (list, tuple)):
        out = []
        for item in spec:
            if not callable(item):
                raise TypeError(f"{argname} must contain callables, got {item!r}")
            out.append(item)
        return out
    raise TypeError(f"{argname} must be a callable, a sequence of callables, or None")


def _copy_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow-copy a state (and its ctx dict) so execution cannot
    mutate the enumerated state used in violation reports."""
    out = dict(state)
    ctx = out.get("ctx")
    if isinstance(ctx, Mapping):
        out["ctx"] = dict(ctx)
    return out


def _passes_constraint(value: Any, constraint: TypeConstraint) -> bool:
    """Single-value mirror of ``Instruction.validate_types`` semantics."""
    if value is None:
        return bool(constraint.nullable)
    try:
        if not isinstance(value, constraint.base_type):
            return False
        if constraint.dependent_on is not None and not constraint.dependent_on(value):
            return False
        if constraint.validation is not None and not constraint.validation(value):
            return False
    except Exception:
        return False
    return True


def _value_meets_constraint(value: Any, constraint: TypeConstraint) -> Tuple[bool, str]:
    """Like :func:`_passes_constraint` but with a failure reason."""
    if value is None:
        if constraint.nullable:
            return True, ""
        return False, "must not be None (constraint is not nullable)"
    try:
        if not isinstance(value, constraint.base_type):
            expected = getattr(constraint.base_type, "__name__", constraint.base_type)
            return False, f"has type {type(value).__name__}, expected {expected}"
        if constraint.dependent_on is not None and not constraint.dependent_on(value):
            return False, "fails its dependent_on predicate"
        if constraint.validation is not None and not constraint.validation(value):
            return False, "fails its validation predicate"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"could not be checked against its constraint ({type(exc).__name__}: {exc})"
    return True, ""


# --- bounded axis enumeration ------------------------------------------------


@dataclass
class _AxisSpec:
    kind: str  # "param" | "ctx"
    key: str
    values: List[Any]
    mode: str  # "exhaustive" | "probe" | "sampled" | "opaque" | "unenumerable"
    note: Optional[str] = None

    @property
    def label(self) -> str:
        return self.key if self.kind == "param" else f"ctx.{self.key}"


def _list_templates(ctor: Callable[[Any], Any]) -> List[Any]:
    """Short deterministic list/tuple/set probes over LIST_ELEMENT_POOL."""
    out: List[Any] = [ctor()]
    for element in LIST_ELEMENT_POOL:
        out.append(ctor([element]))
    for pair in itertools.product(LIST_ELEMENT_POOL, repeat=2):
        out.append(ctor(pair))
    return out


def _base_candidates(base: Any, rng: random.Random) -> Tuple[List[Any], str]:
    """Deterministic candidate values for one base type (pre-narrowing).

    Returns ``(values, mode)`` where mode is ``"exhaustive"`` (the full
    finite domain of the type), ``"probe"`` (a fixed bounded subset of
    an infinite domain), ``"sampled"`` (seeded RNG samples), or
    ``"unenumerable"`` (the checker cannot construct the type).
    """
    # bool first: bool is a subclass of int and must be matched exactly.
    if base is bool:
        return [True, False], "exhaustive"
    if base is type(None):
        return [None], "exhaustive"
    if isinstance(base, type) and issubclass(base, Enum):
        try:
            return list(base), "exhaustive"
        except Exception:  # pragma: no cover - defensive
            return [], "unenumerable"
    if isinstance(base, tuple):
        # A union of types: merge each member's candidates, deduplicated
        # in deterministic order, with the weakest member's mode.
        merged: List[Any] = []
        modes: List[str] = []
        for member in base:
            values, mode = _base_candidates(member, rng)
            for value in values:
                if value not in merged:
                    merged.append(value)
            modes.append(mode)
        if not merged:
            return [], "unenumerable"
        if any(mode == "sampled" for mode in modes):
            return merged, "sampled"
        if modes and all(mode == "exhaustive" for mode in modes):
            return merged, "exhaustive"
        return merged, "probe"
    if base is int:
        return list(INT_PROBE), "probe"
    if base is str:
        return list(STR_PROBE), "probe"
    if base is float:
        samples = [round(rng.uniform(-1.0, 1.0), 6) for _ in range(FLOAT_SAMPLE_COUNT)]
        return list(FLOAT_ANCHORS) + samples, "sampled"
    if base is list:
        return _list_templates(list), "probe"
    if base is tuple:
        return _list_templates(tuple), "probe"
    if base is set:
        return _list_templates(set), "probe"
    if base is frozenset:
        return _list_templates(frozenset), "probe"
    if base is dict:
        return [{}, {"k": 0}, {"k": "v"}], "probe"
    if base in (bytes, bytearray):
        return [base(b""), base(b"x"), base((0, 1))], "probe"
    if base is complex:
        return [0j, 1 + 2j, -1.5 + 0.5j], "probe"
    if _np is not None and base is _np.ndarray:
        return [_np.array([]), _np.array([0, 1])], "probe"
    return [], "unenumerable"


def _narrow_candidates(
    values: List[Any], constraint: TypeConstraint, base: Any, rng: random.Random
) -> Tuple[List[Any], bool]:
    """Narrow candidate values by the constraint's own predicates.

    Returns ``(values, widened)``; when narrowing empties the probe the
    checker deterministically widens (fixed scan for int/str, seeded
    draws for float) rather than reporting an empty domain prematurely.
    """
    if constraint.dependent_on is None and constraint.validation is None:
        return values, False
    kept = [value for value in values if _passes_constraint(value, constraint)]
    if kept:
        return kept, False
    widened: List[Any] = []
    if base is int:
        for value in INT_WIDEN_SCAN:
            if _passes_constraint(value, constraint):
                widened.append(value)
            if len(widened) >= WIDEN_LIMIT:
                break
    elif base is str:
        for value in STR_WIDEN_SCAN:
            if _passes_constraint(value, constraint):
                widened.append(value)
            if len(widened) >= WIDEN_LIMIT:
                break
    elif base is float:
        for _ in range(WIDEN_TRIES):
            value = round(rng.uniform(-1.0, 1.0), 6)
            if _passes_constraint(value, constraint):
                widened.append(value)
            if len(widened) >= WIDEN_LIMIT:
                break
    return widened, bool(widened)


def _param_axis(name: str, constraint: TypeConstraint, rng: random.Random) -> _AxisSpec:
    base = constraint.base_type
    values, mode = _base_candidates(base, rng)
    note: Optional[str] = None
    if mode == "unenumerable":
        note = f"param {name!r}: base type {base!r} is not enumerable by the checker"
        if constraint.nullable:
            # None is the one well-typed value the checker can construct.
            values = [None]
            note += "; only None (nullable) is probed"
    else:
        values, widened = _narrow_candidates(values, constraint, base, rng)
        if widened:
            note = f"param {name!r}: probe widened to satisfy the declared constraint"
        elif not values and not constraint.nullable:
            note = f"param {name!r}: the declared constraint admits none of the probe values"
        if constraint.nullable:
            values = values + [None]
    return _AxisSpec("param", name, values, mode, note)


def _sample_indices(rng: random.Random, total: int, count: int) -> List[int]:
    """Deterministic seeded subset of ``range(total)`` (sorted).

    Uses only ``Random.randrange`` so the selection is stable across
    Python versions (unlike ``Random.sample`` whose internals may vary).
    """
    if count >= total:
        return list(range(total))
    chosen = set()
    while len(chosen) < count:
        chosen.add(rng.randrange(total))
    return sorted(chosen)


def _decode_index(index: int, sizes: Sequence[int]) -> List[int]:
    """Mixed-radix decode of a product index (last axis varies fastest)."""
    combo: List[int] = []
    remaining = index
    for size in reversed(sizes):
        combo.append(remaining % size)
        remaining //= size
    combo.reverse()
    return combo


def _state_from_combo(axes: Sequence[_AxisSpec], combo: Sequence[Any]) -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    ctx: Dict[str, Any] = {}
    for axis, value in zip(axes, combo):
        if axis.kind == "param":
            state[axis.key] = value
        else:
            ctx[axis.key] = value
    if ctx:
        state["ctx"] = ctx
    return state


def enumerate_state_space(
    instruction: Instruction,
    *,
    dependencies: Optional[Sequence[str]] = None,
    ctx_pool: Sequence[Any] = CTX_VALUE_POOL,
    max_states: int = DEFAULT_MAX_STATES,
    seed: int = DEFAULT_SEED,
) -> StateSpace:
    """Build the bounded set of candidate input states for ``instruction``.

    Axes come from the instruction's ``params`` constraints and, when
    the instruction consumes a ``ctx`` dict (its ``dependencies``
    list), from one probed value per required ctx key.  Params whose
    entry is not a :class:`TypeConstraint` are treated as constants and
    omitted (the shipped ``validate_types`` only checks keys it knows).
    The cross product is capped at ``max_states``; beyond the cap a
    deterministic seeded subset is explored.
    """
    if max_states <= 0:
        raise ValueError(f"max_states must be positive, got {max_states}")
    rng = random.Random(seed)

    raw_params = getattr(instruction, "params", None)
    params: Mapping[str, Any] = raw_params if isinstance(raw_params, Mapping) else {}

    axes: List[_AxisSpec] = []
    for pname, constraint in params.items():
        if not isinstance(constraint, TypeConstraint):
            axes.append(
                _AxisSpec(
                    "param",
                    str(pname),
                    [],
                    "opaque",
                    f"param {pname!r} is not a TypeConstraint; treated as a constant and omitted from states",
                )
            )
            continue
        axes.append(_param_axis(str(pname), constraint, rng))

    if dependencies is not None:
        deps = [str(dep) for dep in dependencies]
    else:
        deps = [str(dep) for dep in (getattr(instruction, "dependencies", None) or [])]
    for dep in deps:
        axes.append(
            _AxisSpec("ctx", dep, list(ctx_pool), "probe", f"ctx key {dep!r}: untyped, probed with the default pool")
        )

    productive = [axis for axis in axes if axis.mode != "opaque"]
    notes = [axis.note for axis in axes if axis.note]

    if any(len(axis.values) == 0 for axis in productive):
        # An empty axis makes the whole product empty: no state can be
        # well-typed, so nothing can be discharged.
        states: List[Dict[str, Any]] = []
        total = 0
        truncated = False
        sampled = False
    elif not productive:
        states = [{}]
        total = 1
        truncated = False
        sampled = False
    else:
        sizes = [len(axis.values) for axis in productive]
        total = math.prod(sizes)
        if total <= max_states:
            states = [
                _state_from_combo(productive, combo)
                for combo in itertools.product(*(axis.values for axis in productive))
            ]
            truncated = False
            sampled = False
        else:
            indices = _sample_indices(rng, total, max_states)
            states = [
                _state_from_combo(productive, _decode_index(index, sizes))
                for index in indices
            ]
            truncated = True
            sampled = True

    exhaustive = (not truncated) and all(axis.mode == "exhaustive" for axis in productive)
    sampled = sampled or any(axis.mode == "sampled" for axis in productive)

    return StateSpace(
        states=states,
        exhaustive=exhaustive,
        sampled=sampled,
        truncated=truncated,
        total_candidates=total,
        axes={axis.label: list(axis.values) for axis in axes},
        modes={axis.label: axis.mode for axis in axes},
        notes=notes,
    )


#: The paper's Section 3.3.3 sketches ``generate_state_space``; this
#: alias keeps the shipped API recognizable against that sketch.
generate_state_space = enumerate_state_space


# --- per-state checks ---------------------------------------------------------


def _is_well_typed(instruction: Any, state: Dict[str, Any]) -> bool:
    validate = getattr(instruction, "validate_types", None)
    if not callable(validate):
        return True
    try:
        return bool(validate(state))
    except Exception:
        return False


def _preconditions_hold(
    instruction: Any, state: Dict[str, Any], explicit_pres: Sequence[_PRE]
) -> Tuple[bool, int]:
    """verify_conditions (context preconditions) AND explicit pres.

    A raising precondition cannot establish the antecedent of the
    implication, so it is treated as "precondition does not hold" and
    counted in the returned error count (surfaced as a verdict note,
    not as a Definition 3.5 violation).
    """
    errors = 0
    verify = getattr(instruction, "verify_conditions", None)
    if callable(verify):
        try:
            if not bool(verify(state)):
                return False, errors
        except Exception:
            return False, errors + 1
    for pre in explicit_pres:
        try:
            if not bool(pre(state)):
                return False, errors
        except Exception:
            return False, errors + 1
    return True, errors


def _preservation_violations(
    state: Dict[str, Any],
    outcome: Any,
    outputs: Sequence[str],
    type_env: Mapping[str, Any],
) -> List[Violation]:
    """Check declared outputs against the actual execution result."""
    if not outputs:
        return []
    if not isinstance(outcome, Mapping):
        return [
            Violation(
                "preservation",
                state,
                f"declared outputs {list(outputs)} but execution returned "
                f"{type(outcome).__name__}, not a mapping",
            )
        ]
    violations: List[Violation] = []
    for name in outputs:
        if name not in outcome:
            violations.append(
                Violation(
                    "preservation",
                    state,
                    f"declared output {name!r} missing from result keys "
                    f"{sorted(str(k) for k in outcome.keys())}",
                )
            )
            continue
        constraint = type_env.get(name)
        if isinstance(constraint, TypeConstraint):
            ok, reason = _value_meets_constraint(outcome[name], constraint)
            if not ok:
                violations.append(
                    Violation("preservation", state, f"declared output {name!r} {reason}")
                )
    return violations


def _postcondition_violations(
    state: Dict[str, Any],
    outcome: Any,
    explicit_posts: Sequence[_PRE],
    context_posts: Sequence[_PRE],
) -> List[Violation]:
    """Evaluate every postcondition callable over the post-state.

    A postcondition that returns a falsy value, or that raises on a
    reachable post-state (an ill-specified contract), is a violation.
    """
    violations: List[Violation] = []
    labelled = [("explicit", post) for post in explicit_posts] + [
        ("context", post) for post in context_posts
    ]
    for label, post in labelled:
        name = getattr(post, "__name__", "<callable>")
        try:
            holds = bool(post(outcome))
            detail = (
                None
                if holds
                else f"{label} postcondition {name!r} returned False on post-state {_brief(outcome)}"
            )
        except Exception as exc:
            holds = False
            detail = (
                f"{label} postcondition {name!r} raised {type(exc).__name__}: "
                f"{exc} on post-state {_brief(outcome)}"
            )
        if not holds:
            violations.append(Violation("postcondition", state, detail))
    return violations


# --- instruction- and stream-level verification ------------------------------


async def verify_instruction_async(
    instruction: Any,
    pre_condition: Union[Callable[[Dict[str, Any]], bool], Sequence[Callable], None] = None,
    post_condition: Union[Callable[[Any], bool], Sequence[Callable], None] = None,
    *,
    dependencies: Optional[Sequence[str]] = None,
    ctx_pool: Sequence[Any] = CTX_VALUE_POOL,
    max_states: int = DEFAULT_MAX_STATES,
    seed: int = DEFAULT_SEED,
) -> InstructionVerdict:
    """Check one instruction against Definition 3.5 on a bounded state space.

    This is the coroutine core of the checker; the (paper-shaped) sync
    entry point is :func:`verify_instruction`.
    """
    execute = getattr(instruction, "execute", None)
    if not callable(execute):
        raise TypeError(f"cannot verify an object without a callable execute(): {instruction!r}")

    instruction_id = str(getattr(instruction, "id", getattr(instruction, "name", "<unnamed>")))
    instruction_name = str(getattr(instruction, "name", instruction_id))

    if bool(getattr(instruction, "side_effects", True)):
        return InstructionVerdict(
            instruction_id=instruction_id,
            instruction_name=instruction_name,
            status="skipped_side_effects",
            skipped=True,
            passed=False,
            checked_states=0,
            exhaustive=False,
            progress_violations=0,
            preservation_violations=0,
            postcondition_violations=0,
            skip_reason="side_effects",
            notes=[
                "side-effectful instruction: out of scope for the bounded checker "
                "(never executed, obligation not discharged here)"
            ],
        )

    space = enumerate_state_space(
        instruction,
        dependencies=dependencies,
        ctx_pool=ctx_pool,
        max_states=max_states,
        seed=seed,
    )
    explicit_pres = _as_callables(pre_condition, "pre_condition")
    explicit_posts = _as_callables(post_condition, "post_condition")
    context = getattr(instruction, "context", None)
    context_posts = list(getattr(context, "postconditions", None) or []) if context is not None else []
    outputs = list(getattr(instruction, "outputs", None) or [])
    raw_env = getattr(context, "type_env", None) if context is not None else None
    type_env: Mapping[str, Any] = raw_env if isinstance(raw_env, Mapping) else {}

    candidate_states = 0
    well_typed_states = 0
    checked_states = 0
    precondition_errors = 0
    violations: List[Violation] = []

    for state in space.states:
        candidate_states += 1
        if not _is_well_typed(instruction, state):
            continue
        well_typed_states += 1
        holds, errors = _preconditions_hold(instruction, state, explicit_pres)
        precondition_errors += errors
        if not holds:
            # pre(i, s) is false: the implication is vacuous on this state.
            continue
        checked_states += 1
        try:
            outcome = execute(_copy_state(state))
            if inspect.isawaitable(outcome):
                outcome = await outcome
        except Exception as exc:
            violations.append(
                Violation(
                    "progress",
                    state,
                    f"execution raised {type(exc).__name__}: {exc}",
                )
            )
            continue
        violations.extend(_preservation_violations(state, outcome, outputs, type_env))
        violations.extend(_postcondition_violations(state, outcome, explicit_posts, context_posts))

    progress = sum(1 for violation in violations if violation.kind == "progress")
    preservation = sum(1 for violation in violations if violation.kind == "preservation")
    postcondition = sum(1 for violation in violations if violation.kind == "postcondition")

    notes = list(space.notes)
    if precondition_errors:
        notes.append(
            f"{precondition_errors} explored state(s) had a precondition raise; "
            "treated as 'precondition does not hold'"
        )
    if candidate_states == 0:
        notes.append("no candidate state could be generated; nothing was discharged")
    elif checked_states == 0:
        notes.append(
            "no explored state was both well-typed and precondition-satisfying; "
            "nothing was discharged"
        )

    return InstructionVerdict(
        instruction_id=instruction_id,
        instruction_name=instruction_name,
        status="checked",
        skipped=False,
        passed=checked_states > 0 and not violations,
        checked_states=checked_states,
        exhaustive=space.exhaustive,
        progress_violations=progress,
        preservation_violations=preservation,
        postcondition_violations=postcondition,
        candidate_states=candidate_states,
        well_typed_states=well_typed_states,
        precondition_errors=precondition_errors,
        sampled=space.sampled,
        truncated=space.truncated,
        skip_reason=None,
        violations=violations,
        notes=notes,
    )


def _run_sync(coro):
    """Run a coroutine to completion from synchronous code.

    Uses ``asyncio.run`` when no event loop is running.  When called
    from inside a loop (e.g. an async test), the coroutine is run on a
    private loop in a worker thread, since nesting ``asyncio.run``
    inside a running loop is not allowed.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    box: Dict[str, Any] = {}

    def _runner() -> None:
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # forwarded to the caller thread
            box["error"] = exc

    worker = threading.Thread(target=_runner, name="bounded-checker-loop", daemon=True)
    worker.start()
    worker.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def verify_instruction(
    instruction: Any,
    pre_condition: Union[Callable[[Dict[str, Any]], bool], Sequence[Callable], None] = None,
    post_condition: Union[Callable[[Any], bool], Sequence[Callable], None] = None,
    **kwargs: Any,
) -> InstructionVerdict:
    """Check one pure instruction against Definition 3.5 (synchronous).

    Paper-shaped entry point::

        verify_instruction(instruction, pre_condition, post_condition)

    Keyword arguments are forwarded to :func:`verify_instruction_async`
    (``max_states``, ``seed``, ``ctx_pool``, ``dependencies``).
    """
    return _run_sync(verify_instruction_async(instruction, pre_condition, post_condition, **kwargs))


def _uncheckable_verdict(item: Any) -> InstructionVerdict:
    instruction_id = str(getattr(item, "id", getattr(item, "name", "<unnamed>")))
    return InstructionVerdict(
        instruction_id=instruction_id,
        instruction_name=str(getattr(item, "name", instruction_id)),
        status="skipped_uncheckable",
        skipped=True,
        passed=False,
        checked_states=0,
        exhaustive=False,
        progress_violations=0,
        preservation_violations=0,
        postcondition_violations=0,
        skip_reason="no-execute",
        notes=["stream item has no callable execute(); skipped by the bounded checker"],
    )


def _normalize_condition_spec(spec: Any) -> Tuple[Any, Any]:
    """Accept (pre, post) tuples, {"pre":..., "post":...} dicts, or a
    single callable used as the precondition."""
    if isinstance(spec, Mapping):
        return spec.get("pre"), spec.get("post")
    if isinstance(spec, tuple) and len(spec) == 2:
        return spec[0], spec[1]
    if callable(spec):
        return spec, None
    raise TypeError(f"unsupported condition spec: {spec!r}")


async def verify_stream_async(
    stream: Sequence[Any],
    *,
    conditions: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> StreamVerdict:
    """Check every pure instruction in ``stream``; list the rest as skipped.

    ``conditions`` optionally maps an instruction id to a
    ``(pre_condition, post_condition)`` pair, a
    ``{"pre": ..., "post": ...}`` dict, or a single precondition
    callable.  Instructions lacking these carry only their own
    registered context conditions.  The stream passes only if at least
    one instruction was checked and every checked instruction passed.
    """
    verdicts: List[InstructionVerdict] = []
    for item in stream:
        pre = post = None
        if conditions is not None:
            spec = conditions.get(getattr(item, "id", None))
            if spec is not None:
                pre, post = _normalize_condition_spec(spec)
        if not callable(getattr(item, "execute", None)):
            verdicts.append(_uncheckable_verdict(item))
            continue
        verdicts.append(await verify_instruction_async(item, pre, post, **kwargs))
    checked = [verdict for verdict in verdicts if not verdict.skipped]
    skipped = [verdict for verdict in verdicts if verdict.skipped]
    passed = bool(checked) and all(verdict.passed for verdict in checked)
    return StreamVerdict(verdicts=verdicts, checked=checked, skipped=skipped, passed=passed)


def verify_stream(
    stream: Sequence[Any],
    *,
    conditions: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> StreamVerdict:
    """Synchronous :func:`verify_stream_async` (paper-shaped convenience)."""
    return _run_sync(verify_stream_async(stream, conditions=conditions, **kwargs))
