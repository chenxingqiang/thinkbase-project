"""Unit tests for the bounded verification checker (Definition 3.5).

The checker's whole point is to catch planted violations, so these
tests deliberately include instructions with false postconditions,
raising executions, and missing outputs -- alongside instructions that
must pass exhaustively.  Everything is deterministic: the bounded
state spaces are enumerated from seeded/fixed pools only.
"""

import json
from enum import Enum
from typing import Any, Dict

import pytest

from agentflow.core.isa.bounded_checker import (
    CTX_VALUE_POOL,
    enumerate_state_space,
    verify_instruction,
    verify_instruction_async,
    verify_stream,
)
from agentflow.core.isa.instruction import Instruction, InstructionType, TypeConstraint


class BoolAndInstruction(Instruction):
    """Pure instruction: boolean AND over two fully enumerable params."""

    def __init__(self, instruction_id: str = "bool_and"):
        super().__init__(
            id=instruction_id,
            name=instruction_id,
            type=InstructionType.CONTROL_FLOW,
            params={
                "x": TypeConstraint(base_type=bool),
                "y": TypeConstraint(base_type=bool),
            },
            description="Pure boolean conjunction (x AND y).",
            side_effects=False,
            outputs=["result"],
        )

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        return {"result": bool(input_params["x"] and input_params["y"])}


class BoolOrInstruction(Instruction):
    """Pure instruction: boolean OR over two bool params."""

    def __init__(self):
        super().__init__(
            id="bool_or",
            name="bool_or",
            type=InstructionType.CONTROL_FLOW,
            params={
                "x": TypeConstraint(base_type=bool),
                "y": TypeConstraint(base_type=bool),
            },
            description="Pure boolean disjunction (x OR y).",
            side_effects=False,
            outputs=["result"],
        )

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        return {"result": bool(input_params["x"] or input_params["y"])}


class RaisingInstruction(Instruction):
    """Pure instruction whose execution always raises (bad progress)."""

    def __init__(self):
        super().__init__(
            id="raising",
            name="raising",
            type=InstructionType.CONTROL_FLOW,
            params={"n": TypeConstraint(base_type=int)},
            description="Raises on every execution.",
            side_effects=False,
            outputs=[],
        )

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        raise RuntimeError(f"boom on n={input_params['n']}")


class CountingEffectInstruction(Instruction):
    """Side-effectful instruction that records whether it ran."""

    def __init__(self):
        super().__init__(
            id="effect",
            name="effect",
            type=InstructionType.STATE_MANAGEMENT,
            params={"x": TypeConstraint(base_type=bool)},
            description="Side-effectful instruction (must never be executed).",
            side_effects=True,
            outputs=[],
        )
        self.executions = 0

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        self.executions += 1
        return {"x": input_params["x"]}


class MissingOutputInstruction(Instruction):
    """Declares an output it never produces (preservation violation)."""

    def __init__(self):
        super().__init__(
            id="missing_output",
            name="missing_output",
            type=InstructionType.CONTROL_FLOW,
            params={"x": TypeConstraint(base_type=bool)},
            description="Declares output 'value' but returns another key.",
            side_effects=False,
            outputs=["value"],
        )

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        return {"other": input_params["x"]}


class WrongTypeOutputInstruction(Instruction):
    """Produces a declared output with the wrong type."""

    def __init__(self):
        super().__init__(
            id="wrong_type_output",
            name="wrong_type_output",
            type=InstructionType.CONTROL_FLOW,
            params={"x": TypeConstraint(base_type=bool)},
            description="Declares int output 'count' but returns a str.",
            side_effects=False,
            outputs=["count"],
        )
        self.context.type_env["count"] = TypeConstraint(base_type=int)

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        return {"count": "many"}


class ScalarOutputInstruction(Instruction):
    """Declares outputs but returns a non-mapping result."""

    def __init__(self):
        super().__init__(
            id="scalar_output",
            name="scalar_output",
            type=InstructionType.CONTROL_FLOW,
            params={"x": TypeConstraint(base_type=bool)},
            description="Declares outputs but returns a scalar.",
            side_effects=False,
            outputs=["v"],
        )

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        return 42


class CtxLenInstruction(Instruction):
    """Consumes ctx['v1'] via its dependencies list (run_stream convention)."""

    def __init__(self):
        super().__init__(
            id="ctx_len",
            name="ctx_len",
            type=InstructionType.STATE_MANAGEMENT,
            params={},
            description="len(ctx['v1']) over a declared dependency.",
            side_effects=False,
            outputs=["n"],
        )
        self.dependencies = ["v1"]

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        return {"n": len(input_params["ctx"]["v1"])}


class TwoIntInstruction(Instruction):
    """Two bounded int params: 4 x 4 = 16 candidate states."""

    def __init__(self):
        super().__init__(
            id="two_ints",
            name="two_ints",
            type=InstructionType.CONTROL_FLOW,
            params={
                "a": TypeConstraint(base_type=int),
                "b": TypeConstraint(base_type=int),
            },
            description="Pure integer addition.",
            side_effects=False,
            outputs=["sum"],
        )

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        return {"sum": input_params["a"] + input_params["b"]}


class NarrowIntInstruction(Instruction):
    """Int param narrowed by a validation predicate (v > 0)."""

    def __init__(self):
        super().__init__(
            id="narrow_int",
            name="narrow_int",
            type=InstructionType.CONTROL_FLOW,
            params={"n": TypeConstraint(base_type=int, validation=lambda v: v > 0)},
            description="Pure doubling over positive ints.",
            side_effects=False,
            outputs=["doubled"],
        )

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        return {"doubled": input_params["n"] * 2}


class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class ColorTagInstruction(Instruction):
    """Enum x bool params: 3 x 2 = 6 states, all enumerable."""

    def __init__(self):
        super().__init__(
            id="color_tag",
            name="color_tag",
            type=InstructionType.CONTROL_FLOW,
            params={
                "color": TypeConstraint(base_type=Color),
                "flag": TypeConstraint(base_type=bool),
            },
            description="Builds a tag string from an enum and a flag.",
            side_effects=False,
            outputs=["tag"],
        )

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        return {"tag": f"{input_params['color'].name}-{input_params['flag']}"}


# --- 1. pure instruction, true postcondition, small enumerable space ---------


def test_pure_instruction_small_space_passes_exhaustively():
    """Exhaustive over the bool cross product: zero violations, passed."""
    instr = BoolAndInstruction()
    instr.add_postcondition(lambda post: isinstance(post["result"], bool))

    verdict = verify_instruction(instr, None, lambda post: post["result"] in (True, False))

    assert verdict.status == "checked"
    assert verdict.skipped is False
    assert verdict.passed is True
    assert verdict.exhaustive is True
    assert verdict.sampled is False
    assert verdict.truncated is False
    assert verdict.checked_states == 4  # {True, False} x {True, False}
    assert verdict.candidate_states == 4
    assert verdict.well_typed_states == 4
    assert verdict.progress_violations == 0
    assert verdict.preservation_violations == 0
    assert verdict.postcondition_violations == 0
    assert verdict.violations == []
    # The required summary keys are exposed as a plain dict.
    summary = verdict.to_dict()
    assert summary["checked_states"] == 4
    assert summary["exhaustive"] is True
    assert summary["passed"] is True
    assert summary["postcondition_violations"] == 0


# --- 2. deliberately false postcondition -------------------------------------


def test_deliberately_false_postcondition_is_caught():
    """A planted violation must be reported with its counterexample state."""
    instr = BoolAndInstruction()
    # False exactly on the state where x AND y is True, i.e. (True, True).
    verdict = verify_instruction(instr, None, lambda post: post["result"] is False)

    assert verdict.passed is False
    assert verdict.checked_states == 4
    assert verdict.postcondition_violations == 1
    assert len(verdict.violations) == 1
    assert verdict.violations[0].kind == "postcondition"
    assert verdict.violations[0].state == {"x": True, "y": True}
    assert verdict.progress_violations == 0
    assert verdict.preservation_violations == 0


# --- 3. implication semantics: pre => post, not pre AND post ------------------


def test_implication_semantics_vacuous_states_do_not_violate():
    """A postcondition failing only on precondition-failing states passes.

    pre excludes exactly (False, False); the postcondition also fails
    exactly there.  Checking `pre AND post` instead of `pre => post`
    would report a violation -- the checker must not.
    """
    instr = BoolOrInstruction()
    # Registered context precondition (exercised via verify_conditions).
    instr.add_precondition(
        lambda state: isinstance(state.get("x"), bool) and isinstance(state.get("y"), bool)
    )
    # Explicit precondition: excludes exactly the (False, False) state.
    verdict = verify_instruction(
        instr,
        lambda state: state["x"] or state["y"],
        lambda post: post["result"] is True,
    )

    assert verdict.candidate_states == 4
    assert verdict.checked_states == 3  # (False, False) is vacuous: pre fails
    assert verdict.exhaustive is True
    assert verdict.postcondition_violations == 0
    assert verdict.progress_violations == 0
    assert verdict.preservation_violations == 0
    assert verdict.passed is True


# --- 4. progress violation: execution raises ---------------------------------


def test_progress_violation_when_execution_raises():
    """Precondition holds, execution raises -> progress violation caught."""
    instr = RaisingInstruction()

    verdict = verify_instruction(instr)

    assert verdict.passed is False
    assert verdict.checked_states == 4  # int probe {-1, 0, 1, 2}, no precondition
    assert verdict.progress_violations == 4
    assert verdict.postcondition_violations == 0
    assert verdict.preservation_violations == 0
    assert all(v.kind == "progress" for v in verdict.violations)
    assert all("boom" in v.detail for v in verdict.violations)
    assert all("RuntimeError" in v.detail for v in verdict.violations)
    # ints are a bounded probe, not an exhaustive domain
    assert verdict.exhaustive is False
    assert verdict.sampled is False


def test_progress_violation_on_precondition_satisfying_state_only():
    """Execution raising only where the precondition fails is not a violation."""
    instr = RaisingInstruction()
    # Precondition holds exactly on the non-negative probe values; the
    # instruction raises everywhere, so violations occur only on 0, 1, 2.
    verdict = verify_instruction(instr, lambda state: state["n"] >= 0)

    assert verdict.checked_states == 3
    assert verdict.progress_violations == 3
    assert verdict.passed is False
    offending = sorted(v.state["n"] for v in verdict.violations)
    assert offending == [0, 1, 2]


# --- 5. side-effectful instructions ------------------------------------------


def test_side_effectful_instruction_is_skipped_and_never_executed():
    """side_effects=True -> out of scope, never executed, not discharged."""
    instr = CountingEffectInstruction()
    # Even a deliberately false postcondition must not be evaluated.
    verdict = verify_instruction(instr, None, lambda post: False)

    assert verdict.skipped is True
    assert verdict.status == "skipped_side_effects"
    assert verdict.skip_reason == "side_effects"
    assert verdict.passed is False
    assert verdict.checked_states == 0
    assert verdict.progress_violations == 0
    assert verdict.preservation_violations == 0
    assert verdict.postcondition_violations == 0
    assert instr.executions == 0  # never executed
    assert instr.metrics.total_executions == 0
    assert "out of scope" in " ".join(verdict.notes)


# --- 6. stream-level verification --------------------------------------------


def test_verify_stream_separates_checked_and_skipped():
    """A mixed stream reports checked pure instructions separately from
    skipped side-effectful ones, and never executes the latter."""
    good = BoolAndInstruction(instruction_id="and_good")
    effect = CountingEffectInstruction()
    bad = BoolAndInstruction(instruction_id="and_bad")
    stream = [good, effect, bad]
    conditions = {
        "and_good": (None, lambda post: post["result"] in (True, False)),
        "and_bad": (None, lambda post: post["result"] is False),  # planted violation
    }

    report = verify_stream(stream, conditions=conditions)

    assert [v.instruction_id for v in report.checked] == ["and_good", "and_bad"]
    assert [v.instruction_id for v in report.skipped] == ["effect"]
    assert [v.instruction_id for v in report.verdicts] == ["and_good", "effect", "and_bad"]
    assert effect.executions == 0  # side-effectful member never executed
    assert report.checked[0].passed is True
    assert report.checked[1].passed is False
    assert report.checked[1].postcondition_violations == 1
    assert report.passed is False

    summary = report.to_dict()
    assert summary["total_checked"] == 2
    assert summary["total_skipped"] == 1
    assert summary["skipped"][0]["instruction_id"] == "effect"
    assert summary["skipped"][0]["skip_reason"] == "side_effects"
    assert len(summary["checked"]) == 2


def test_verify_stream_all_passing():
    """A stream of only pure, valid instructions passes overall."""
    stream = [BoolAndInstruction(instruction_id="and_1"), BoolOrInstruction()]
    report = verify_stream(stream)
    assert report.passed is True
    assert len(report.checked) == 2
    assert report.skipped == []


# --- 7. preservation violations ----------------------------------------------


def test_preservation_violation_missing_output():
    instr = MissingOutputInstruction()
    verdict = verify_instruction(instr)
    assert verdict.passed is False
    assert verdict.checked_states == 2  # one bool param
    assert verdict.preservation_violations == 2
    assert all(v.kind == "preservation" for v in verdict.violations)
    assert all("'value'" in v.detail for v in verdict.violations)


def test_preservation_violation_wrong_output_type():
    instr = WrongTypeOutputInstruction()
    verdict = verify_instruction(instr)
    assert verdict.passed is False
    assert verdict.preservation_violations == 2
    detail = verdict.violations[0].detail
    assert "expected int" in detail


def test_preservation_violation_non_mapping_result():
    instr = ScalarOutputInstruction()
    verdict = verify_instruction(instr)
    assert verdict.passed is False
    assert verdict.preservation_violations == 2
    assert "not a mapping" in verdict.violations[0].detail


# --- 8. ctx-consuming instructions (dependencies) ----------------------------


def test_ctx_dependency_states_shape_and_check():
    """States for dependency-consuming instructions are {'ctx': {...}}."""
    instr = CtxLenInstruction()

    space = enumerate_state_space(instr)
    assert len(space.states) == len(CTX_VALUE_POOL)
    assert space.exhaustive is False  # ctx keys are probed, not known exhaustively
    assert all(set(state.keys()) == {"ctx"} for state in space.states)
    assert all(set(state["ctx"].keys()) == {"v1"} for state in space.states)

    # Precondition admits only the list-valued ctx probe; postcondition
    # checks the computed length.
    verdict = verify_instruction(
        instr,
        lambda state: isinstance(state["ctx"]["v1"], list),
        lambda post: post["n"] == 2,
    )
    assert verdict.candidate_states == len(CTX_VALUE_POOL)
    assert verdict.checked_states == 1  # exactly one pool value is a list
    assert verdict.postcondition_violations == 0
    assert verdict.progress_violations == 0
    assert verdict.passed is True


def test_ctx_dependency_progress_violation_on_garbage_ctx():
    """With no precondition filtering, an unhandled ctx value surfaces
    as a progress violation (the precondition was too weak)."""
    instr = CtxLenInstruction()

    verdict = verify_instruction(instr, None, lambda post: post["n"] >= 0)

    # len() succeeds on exactly three pool values: "" (0), "abc" (3),
    # and [0, 1] (2); the other seven raise -> progress violations.
    assert verdict.checked_states == len(CTX_VALUE_POOL)
    assert verdict.progress_violations == len(CTX_VALUE_POOL) - 3
    assert verdict.postcondition_violations == 0  # 0, 3, and 2 all satisfy >= 0
    assert verdict.passed is False


# --- 9. bounded enumeration details ------------------------------------------


def test_max_states_cap_produces_deterministic_sample():
    """Beyond the cap the checker explores a seeded, reproducible subset."""
    instr = TwoIntInstruction()

    space = enumerate_state_space(instr, max_states=8)
    assert space.total_candidates == 16  # 4 x 4 probe cross product
    assert len(space.states) == 8
    assert space.truncated is True
    assert space.sampled is True
    assert space.exhaustive is False

    # Byte-identical enumeration on a fresh run with the same seed.
    again = enumerate_state_space(instr, max_states=8)
    assert space.states == again.states

    verdict = verify_instruction(instr, max_states=8)
    assert verdict.checked_states == 8
    assert verdict.truncated is True
    assert verdict.exhaustive is False
    assert verdict.passed is True

    # Under the default cap the full 16-state probe product is explored.
    full = verify_instruction(instr)
    assert full.checked_states == 16
    assert full.exhaustive is False  # int axes are probes
    assert full.truncated is False
    assert full.passed is True


def test_int_probe_narrowed_by_constraint():
    """A validation predicate narrows the int probe to its passing values."""
    instr = NarrowIntInstruction()
    space = enumerate_state_space(instr)
    assert space.axes["n"] == [1, 2]  # {-1, 0, 1, 2} narrowed by v > 0
    verdict = verify_instruction(instr, None, lambda post: post["doubled"] > 0)
    assert verdict.checked_states == 2
    assert verdict.passed is True


def test_enum_axis_enumerates_all_members():
    instr = ColorTagInstruction()
    verdict = verify_instruction(instr, None, lambda post: isinstance(post["tag"], str))
    assert verdict.exhaustive is True
    assert verdict.checked_states == 6  # 3 enum members x 2 bools
    assert verdict.passed is True


def test_verdicts_are_deterministic():
    """Same instruction + config -> byte-identical verdict dicts."""
    instr = BoolAndInstruction()
    post = lambda post: post["result"] in (True, False)  # noqa: E731
    first = verify_instruction(instr, None, post)
    second = verify_instruction(instr, None, post)
    assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(second.to_dict(), sort_keys=True)

    raising = RaisingInstruction()
    r1 = verify_instruction(raising)
    r2 = verify_instruction(raising)
    assert json.dumps(r1.to_dict(), sort_keys=True) == json.dumps(r2.to_dict(), sort_keys=True)


# --- 10. async API and sync/async interop ------------------------------------


@pytest.mark.asyncio
async def test_async_api_directly():
    instr = RaisingInstruction()
    verdict = await verify_instruction_async(instr)
    assert verdict.progress_violations == 4
    assert verdict.passed is False


@pytest.mark.asyncio
async def test_sync_api_inside_running_event_loop():
    """The sync entry point must work from inside a running loop."""
    instr = BoolAndInstruction()
    verdict = verify_instruction(instr, None, lambda post: isinstance(post["result"], bool))
    assert verdict.passed is True
    assert verdict.checked_states == 4


def test_unenumerable_param_discharges_nothing():
    """A param type the checker cannot construct yields no states and
    an honest failed verdict rather than a vacuous pass."""

    class Opaque:
        pass

    class OpaqueInstruction(Instruction):
        def __init__(self):
            super().__init__(
                id="opaque",
                name="opaque",
                type=InstructionType.CONTROL_FLOW,
                params={"payload": TypeConstraint(base_type=Opaque)},
                description="Takes an unconstructible type.",
                side_effects=False,
                outputs=[],
            )

        async def execute(self, input_params: Dict[str, Any]) -> Any:
            return {}

    instr = OpaqueInstruction()
    verdict = verify_instruction(instr)
    assert verdict.candidate_states == 0
    assert verdict.checked_states == 0
    assert verdict.passed is False
    assert any("not enumerable" in note for note in verdict.notes)
