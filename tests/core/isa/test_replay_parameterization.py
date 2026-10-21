"""Parameterized replay of learned instructions (Section 3.3, stage v).

Composite replay previously re-ran the recorded subsequence but had no
parameter surface: the mined template saved no parameter schema,
caller-supplied parameter values traveling alongside ``ctx`` were
dropped, and nothing type-checked unseen values.  Dialogue-mined
templates, which carry no recorded body, could only refuse.

This file covers the parameterized-replay contract:

    mine -> save operator sequence + parameter schema -> replay
    with *unseen* parameter values == re-running the recorded
    constituents with those values

plus the dialogue-template executable-declaration rule: a template
with an executable declaration executes; one without keeps the scoped
``NotImplementedError`` naming the reason and both ways out.
"""

import asyncio

import pytest
from typing import Any, Dict, List

from agentflow.core.isa.agentic_calling import AgenticTransformer
from agentflow.core.isa.instruction import (
    Instruction,
    InstructionType,
    TypeConstraint,
)
from agentflow.core.isa.instruction_learning import (
    InstructionLearner,
    InstructionTemplate,
    record_trace,
)


# ----------------------------------------------------------------------
# A deterministic, pure, ctx-connected stream parameterized by two
# external inputs (the run_stream convention): every step reads from
# the shared ctx and merges its dict output back into it.
# ----------------------------------------------------------------------

class GenValues(Instruction):
    """Derive the raw value list from the seed (pure arithmetic)."""

    def __init__(self):
        super().__init__(
            id="gen_values", name="gen_values",
            type=InstructionType.STATE_MANAGEMENT,
            params={}, description="Derive raw values from the seed",
            side_effects=False, outputs=["values"],
        )
        self.dependencies = ["seed"]

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        seed = input_params["ctx"]["seed"]
        return {"values": [(seed * (i + 3)) % 97 + 1 for i in range(8)]}


class FilterValues(Instruction):
    """Keep the values at or above the cutoff (pure)."""

    def __init__(self):
        super().__init__(
            id="filter_values", name="filter_values",
            type=InstructionType.CONTROL_FLOW,
            params={}, description="Filter values by cutoff",
            side_effects=False, outputs=["kept"],
        )
        self.dependencies = ["values", "cutoff"]

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        values = input_params["ctx"]["values"]
        cutoff = input_params["ctx"]["cutoff"]
        return {"kept": [v for v in values if v >= cutoff]}


class TotalKept(Instruction):
    """Summarize the kept values (pure)."""

    def __init__(self):
        super().__init__(
            id="total_kept", name="total_kept",
            type=InstructionType.CONTROL_FLOW,
            params={}, description="Total the kept values",
            side_effects=False, outputs=["total", "report"],
        )
        self.dependencies = ["kept"]

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        kept = input_params["ctx"]["kept"]
        total = sum(kept)
        return {"total": total, "report": f"kept={total}"}


STREAM_IDS = ["gen_values", "filter_values", "total_kept"]

# Parameter values used while mining: none of the replay cases below
# may reuse them, so every replayed value is genuinely unseen.
RECORDED_SEEDS = (1, 2, 3)
RECORDED_CUTOFFS = (10, 20)

# 210 unseen (seed, cutoff) combinations.  Cutoffs 40..98 sweep
# selectivities down to the empty-kept edge case (values are <= 97).
UNSEEN_CASES = [
    (100 + i, 40 + (i % 59))
    for i in range(210)
]


def build_two_param_stream() -> List[Instruction]:
    """The three-step stream, rebuilt fresh on every call."""
    return [GenValues(), FilterValues(), TotalKept()]


async def rerun_constituents(
    template: InstructionTemplate, seed: int, cutoff: int
) -> Dict[str, Any]:
    """Oracle: re-run the recorded constituent sequence with new values."""
    reference_trace = await record_trace(
        list(template.components), {"seed": seed, "cutoff": cutoff}
    )
    merged: Dict[str, Any] = {}
    for entry in reference_trace:
        merged.update(entry["outputs"])
    return merged


async def mine_two_param_template():
    """Mine the two-parameter stream from six recorded traces."""
    learner = InstructionLearner(AgenticTransformer())
    traces = [
        await record_trace(
            build_two_param_stream(),
            {"seed": seed, "cutoff": cutoff},
        )
        for seed in RECORDED_SEEDS
        for cutoff in RECORDED_CUTOFFS
    ]
    templates = learner.learn_from_traces(traces)
    assert len(templates) == 1
    return learner, templates[0]


# ----------------------------------------------------------------------
# (a) Parameterized replay: schema capture and unseen-value consistency
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_operator_sequence_and_parameter_schema_are_saved():
    """Promotion saves the operator sequence *and* its parameter schema."""
    learner, template = await mine_two_param_template()

    # The operator sequence is saved (constituent instances, mined order).
    assert [c.id for c in template.components] == STREAM_IDS

    # The parameter schema is saved: one entry per external input, its
    # observed type as the constraint's base type.
    assert set(template.params) == {"seed", "cutoff"}
    for name in ("seed", "cutoff"):
        constraint = template.params[name]
        assert isinstance(constraint, TypeConstraint)
        assert constraint.base_type is int

    # The created instruction exposes the same schema as its params.
    learned = learner.create_instruction(template)
    assert learned.params == template.params

    # The mined pattern keeps the schema for provenance.
    pattern = learner.pattern_recognizer.patterns[
        template.name.removeprefix("learned_")
    ]
    assert pattern.parameter_schema == template.params


@pytest.mark.asyncio
async def test_210_unseen_parameter_values_match_rerun():
    """Replay with unseen parameter values == re-running the sequence.

    210 combinations never recorded during mining, across all three
    calling conventions (wrapped ctx, flat, and schema parameters
    passed alongside an empty ctx).
    """
    learner, template = await mine_two_param_template()
    learned = learner.create_instruction(template)

    for i, (seed, cutoff) in enumerate(UNSEEN_CASES):
        expected = await rerun_constituents(template, seed, cutoff)

        if i % 7 == 3:
            # Schema parameters as siblings of an empty ctx.
            result = await learned.execute(
                {"ctx": {}, "seed": seed, "cutoff": cutoff}
            )
        elif i % 5 == 2:
            # Flat convention.
            result = await learned.execute({"seed": seed, "cutoff": cutoff})
        else:
            # Wrapped (run_stream) convention.
            result = await learned.execute(
                {"ctx": {"seed": seed, "cutoff": cutoff}}
            )

        assert result == expected, (
            f"case {i}: seed={seed} cutoff={cutoff}"
        )


@pytest.mark.asyncio
async def test_sibling_parameters_bind_into_replay():
    """Parameters passed alongside ctx bind into the replay context.

    Previously the wrapped convention took only ctx, so a caller
    supplying the schema parameters as siblings had them silently
    dropped (the replay then died on a missing key).
    """
    learner, template = await mine_two_param_template()
    learned = learner.create_instruction(template)

    seed, cutoff = 4242, 51  # unseen values
    expected = await rerun_constituents(template, seed, cutoff)

    result = await learned.execute(
        {"ctx": {}, "seed": seed, "cutoff": cutoff}
    )
    assert result == expected

    # Partial: one parameter in ctx, one alongside it.
    partial = await learned.execute(
        {"ctx": {"seed": seed}, "cutoff": cutoff}
    )
    assert partial == expected


@pytest.mark.asyncio
async def test_ctx_wins_over_sibling_parameters():
    """On conflict the ctx value wins; siblings only fill gaps."""
    learner, template = await mine_two_param_template()
    learned = learner.create_instruction(template)

    expected = await rerun_constituents(template, seed=5, cutoff=12)
    result = await learned.execute(
        {"ctx": {"seed": 5, "cutoff": 12}, "seed": 999}
    )
    assert result == expected


@pytest.mark.asyncio
async def test_unseen_parameter_values_are_type_checked():
    """Unseen values are validated against the saved schema."""
    learner, template = await mine_two_param_template()
    learned = learner.create_instruction(template)

    # Wrapped convention: the bound value fails the schema constraint.
    with pytest.raises(TypeError, match="seed"):
        await learned.execute({"ctx": {"seed": "not-an-int", "cutoff": 20}})

    # Flat convention: rejected by the standard params check.
    with pytest.raises(TypeError, match="Invalid parameter types"):
        await learned.execute({"seed": "not-an-int", "cutoff": 20})


@pytest.mark.asyncio
async def test_mixed_observed_types_fall_back_to_permissive_schema():
    """An input observed with several types stays permissive (object)."""

    class EchoSeed(Instruction):
        def __init__(self):
            super().__init__(
                id="echo_seed", name="echo_seed",
                type=InstructionType.STATE_MANAGEMENT,
                params={}, description="Echo the seed",
                side_effects=False, outputs=["echo"],
            )
            self.dependencies = ["seed"]

        async def execute(self, input_params: Dict[str, Any]) -> Any:
            return {"echo": repr(input_params["ctx"]["seed"])}

    class DescribeEcho(Instruction):
        def __init__(self):
            super().__init__(
                id="describe_echo", name="describe_echo",
                type=InstructionType.CONTROL_FLOW,
                params={}, description="Describe the echo",
                side_effects=False, outputs=["description"],
            )
            self.dependencies = ["echo"]

        async def execute(self, input_params: Dict[str, Any]) -> Any:
            echo = input_params["ctx"]["echo"]
            return {"description": f"echo({len(echo)})"}

    learner = InstructionLearner(AgenticTransformer())
    traces = [
        await record_trace([EchoSeed(), DescribeEcho()], {"seed": value})
        for value in (1, "one")  # int and str observations
    ]
    templates = learner.learn_from_traces(traces)
    assert len(templates) == 1

    schema = templates[0].params
    assert set(schema) == {"seed"}
    assert isinstance(schema["seed"], TypeConstraint)
    assert schema["seed"].base_type is object

    # A value of a type never observed still replays correctly.
    learned = learner.create_instruction(templates[0])
    result = await learned.execute({"ctx": {"seed": 7.5}})
    assert result == {"echo": "7.5", "description": "echo(3)"}


@pytest.mark.asyncio
async def test_bare_instruction_traces_get_permissive_schema():
    """Traces without recorded inputs still yield a usable schema."""
    learner = InstructionLearner(AgenticTransformer())
    traces = [
        build_two_param_stream()  # bare Instructions, no input records
        for _ in range(2)
    ]
    templates = learner.learn_from_traces(traces)
    assert len(templates) == 1

    schema = templates[0].params
    assert set(schema) == {"seed", "cutoff"}
    for name in ("seed", "cutoff"):
        assert isinstance(schema[name], TypeConstraint)
        assert schema[name].base_type is object

    learned = learner.create_instruction(templates[0])
    expected = await rerun_constituents(templates[0], seed=555, cutoff=42)
    result = await learned.execute({"ctx": {"seed": 555, "cutoff": 42}})
    assert result == expected


# ----------------------------------------------------------------------
# (b) Dialogue templates: executable declaration or scoped error
# ----------------------------------------------------------------------

SAMPLE_DIALOGUES: List[Dict[str, Any]] = [
    {
        "turns": [
            {"role": "system", "content": "Execute test_function"},
            {"role": "user", "content": '{"param1": "test1", "param2": 42}'},
            {"role": "assistant", "content": '{"result": "success1"}'},
        ],
        "success": True,
        "context": {"param1": "test1", "param2": 42},
    },
    {
        "turns": [
            {"role": "system", "content": "Execute test_function"},
            {"role": "user", "content": '{"param1": "test2", "param2": 43}'},
            {"role": "assistant", "content": '{"result": "success2"}'},
        ],
        "success": True,
        "context": {"param1": "test2", "param2": 43},
    },
]


@pytest.mark.asyncio
async def test_dialogue_template_with_executor_executes():
    """A dialogue template carrying an executable declaration runs it."""
    learner = InstructionLearner(AgenticTransformer())
    learner.learn_from_dialogues(SAMPLE_DIALOGUES)
    template = next(iter(learner.instruction_templates.values()))
    assert template.components is None  # dialogue-mined: no recorded body

    calls: List[Dict[str, Any]] = []

    def executor(input_params: Dict[str, Any]) -> Any:
        calls.append(dict(input_params))
        return {"result": f"executed:{input_params['param1']}"}

    template.executor = executor
    instruction = learner.create_instruction(template)

    result = await instruction.execute({"param1": "alpha", "param2": 42})
    assert result == {"result": "executed:alpha"}
    assert calls == [{"param1": "alpha", "param2": 42}]
    assert instruction.metrics.successful_executions == 1


@pytest.mark.asyncio
async def test_dialogue_template_with_async_executor_executes():
    """Async executable declarations are awaited."""
    learner = InstructionLearner(AgenticTransformer())
    template = InstructionTemplate(
        name="async_declared",
        type=InstructionType.LLM_INTERACTION,
        params={},
        preconditions=["requires_param1"],
        postconditions=[],
    )

    async def executor(input_params: Dict[str, Any]) -> Any:
        await asyncio.sleep(0)
        return {"result": f"async:{input_params['param1']}"}

    template.executor = executor
    instruction = learner.create_instruction(template)
    result = await instruction.execute({"param1": "beta"})
    assert result == {"result": "async:beta"}


@pytest.mark.asyncio
async def test_executor_postconditions_are_enforced():
    """Declared produces_X postconditions hold for executor results."""

    def good_executor(input_params: Dict[str, Any]) -> Any:
        return {"result": "ok"}

    def bad_executor(input_params: Dict[str, Any]) -> Any:
        return {"ignored": True}

    learner = InstructionLearner(AgenticTransformer())

    good = InstructionTemplate(
        name="declared_good",
        type=InstructionType.LLM_INTERACTION,
        params={},
        preconditions=[],
        postconditions=["produces_result"],
        executor=good_executor,
    )
    instruction = learner.create_instruction(good)
    assert await instruction.execute({}) == {"result": "ok"}

    bad = InstructionTemplate(
        name="declared_bad",
        type=InstructionType.LLM_INTERACTION,
        params={},
        preconditions=[],
        postconditions=["produces_result"],
        executor=bad_executor,
    )
    instruction = learner.create_instruction(bad)
    with pytest.raises(RuntimeError, match="failed to produce"):
        await instruction.execute({})


@pytest.mark.asyncio
async def test_preconditions_gates_before_executor_runs():
    """Preconditions are checked before the declaration executes."""
    ran = []

    learner = InstructionLearner(AgenticTransformer())
    template = InstructionTemplate(
        name="gated_declared",
        type=InstructionType.LLM_INTERACTION,
        params={},
        preconditions=["requires_param1"],
        postconditions=[],
        executor=lambda p: ran.append(p) or {"result": "ran"},
    )
    instruction = learner.create_instruction(template)

    with pytest.raises(ValueError, match="Preconditions not met"):
        await instruction.execute({})
    assert ran == []


@pytest.mark.asyncio
async def test_bodyless_template_error_names_the_way_out():
    """Without a body or declaration the scoped error explains both."""
    learner = InstructionLearner(AgenticTransformer())
    for components in (None, []):
        bodyless = InstructionTemplate(
            name="bodyless_template",
            type=InstructionType.LLM_INTERACTION,
            params={},
            preconditions=[],
            postconditions=[],
            components=components,
        )
        instruction = learner.create_instruction(bodyless)
        with pytest.raises(NotImplementedError) as excinfo:
            await instruction.execute({})
        message = str(excinfo.value)
        # The scoped error keeps its reason ...
        assert "no constituent" in message
        # ... and names both ways out: attach an executable
        # declaration, or re-mine from instruction traces.
        assert "executor" in message
        assert "learn_from_traces" in message
