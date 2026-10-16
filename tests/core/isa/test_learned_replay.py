"""Composite-replay execution of learned instructions (Section 3.3).

Covers the promotion pipeline end to end on a small deterministic,
pure, ctx-connected instruction stream:

    build stream -> execute (record traces) -> mine -> promote ->
    replay as a LearnedInstruction

Learned instructions replay their mined constituent subsequence and
must reproduce its outputs exactly; promotion must be deterministic
across independent runs; templates promoted without recorded bodies
must refuse to execute; and every declared ``produces_X``
postcondition must hold after replay.
"""

import pytest
from typing import Any, Dict, List, Tuple

from agentflow.core.isa.agentic_calling import AgenticTransformer
from agentflow.core.isa.instruction import Instruction, InstructionType
from agentflow.core.isa.instruction_learning import (
    InstructionLearner,
    InstructionTemplate,
    record_trace,
)


# ----------------------------------------------------------------------
# A small deterministic, pure, ctx-connected instruction stream.
# Every step reads from the shared ctx and merges its dict output
# back into it (the run_stream convention of the system task demo).
# ----------------------------------------------------------------------

class SeedValues(Instruction):
    """Derive the raw value list from the seed (pure arithmetic)."""

    def __init__(self):
        super().__init__(
            id="seed_values", name="seed_values",
            type=InstructionType.STATE_MANAGEMENT,
            params={}, description="Derive raw values from the seed",
            side_effects=False, outputs=["values"],
        )
        self.dependencies = ["seed"]

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        seed = input_params["ctx"]["seed"]
        return {"values": [(seed * (i + 3)) % 97 + 1 for i in range(8)]}


class NormalizeValues(Instruction):
    """Shift the raw values (pure)."""

    def __init__(self):
        super().__init__(
            id="normalize", name="normalize",
            type=InstructionType.CONTROL_FLOW,
            params={}, description="Normalize raw values",
            side_effects=False, outputs=["normalized"],
        )
        self.dependencies = ["values"]

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        values = input_params["ctx"]["values"]
        return {"normalized": [v * 2 - 50 for v in values]}


class ThresholdSelect(Instruction):
    """Keep the values above a fixed cutoff (pure)."""

    def __init__(self):
        super().__init__(
            id="threshold_select", name="threshold_select",
            type=InstructionType.CONTROL_FLOW,
            params={}, description="Select values above the cutoff",
            side_effects=False, outputs=["selected"],
        )
        self.dependencies = ["normalized"]

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        normalized = input_params["ctx"]["normalized"]
        return {"selected": [v for v in normalized if v > 20]}


class ClassifyValues(Instruction):
    """Deterministic rule-based labeler standing in for the LLM."""

    def __init__(self):
        super().__init__(
            id="classify", name="classify",
            type=InstructionType.LLM_INTERACTION,
            params={}, description="Label selected values",
            side_effects=False, outputs=["labels"],
        )
        self.dependencies = ["selected"]

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        selected = input_params["ctx"]["selected"]
        return {"labels": ["high" if v >= 90 else "low" for v in selected]}


class AggregateLabels(Instruction):
    """Count the labels (pure)."""

    def __init__(self):
        super().__init__(
            id="aggregate", name="aggregate",
            type=InstructionType.CONTROL_FLOW,
            params={}, description="Aggregate label counts",
            side_effects=False, outputs=["summary"],
        )
        self.dependencies = ["labels"]

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        labels = input_params["ctx"]["labels"]
        summary: Dict[str, int] = {"total": len(labels)}
        for label in labels:
            summary[label] = summary.get(label, 0) + 1
        return {"summary": summary}


class FormatReport(Instruction):
    """Render the summary deterministically (pure, no file writes)."""

    def __init__(self):
        super().__init__(
            id="format_report", name="format_report",
            type=InstructionType.STATE_MANAGEMENT,
            params={}, description="Render the aggregate as a report",
            side_effects=False, outputs=["report"],
        )
        self.dependencies = ["summary"]

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        summary = input_params["ctx"]["summary"]
        parts = ", ".join(
            f"{key}={value}" for key, value in sorted(summary.items())
        )
        return {"report": f"triage({parts})"}


STREAM_IDS = [
    "seed_values", "normalize", "threshold_select",
    "classify", "aggregate", "format_report",
]

EXPECTED_EDGES: List[Tuple[str, str]] = [
    ("seed_values", "normalize"),
    ("normalize", "threshold_select"),
    ("threshold_select", "classify"),
    ("classify", "aggregate"),
    ("aggregate", "format_report"),
]

EXPECTED_POSTCONDITIONS = [
    "produces_values", "produces_normalized", "produces_selected",
    "produces_labels", "produces_summary", "produces_report",
]


def build_stream() -> List[Instruction]:
    """The six-step ctx-connected stream, rebuilt fresh on every call."""
    return [
        SeedValues(), NormalizeValues(), ThresholdSelect(),
        ClassifyValues(), AggregateLabels(), FormatReport(),
    ]


async def mine_promoted(seed: int = 42, runs: int = 2):
    """Full pipeline: execute the stream ``runs`` times, mine, promote.

    Everything is rebuilt from scratch on every call, so two calls
    share no state -- exactly the independence the determinism test
    needs.
    """
    learner = InstructionLearner(AgenticTransformer())
    traces = [
        await record_trace(build_stream(), {"seed": seed})
        for _ in range(runs)
    ]
    templates = learner.learn_from_traces(traces)
    return learner, templates


@pytest.mark.asyncio
async def test_pattern_carries_constituents_and_dataflow():
    """The mined pattern captures the subsequence and its dataflow."""
    _, templates = await mine_promoted()

    # Two runs of the identical six-step stream make the full stream
    # the single maximal recurring window (support 2, success 1.0).
    assert len(templates) == 1
    template = templates[0]

    assert template.components is not None
    assert [c.id for c in template.components] == STREAM_IDS
    assert template.dataflow_edges == EXPECTED_EDGES
    # The only external input is the seed; everything else is produced
    # inside the subsequence.
    assert template.preconditions == ["requires_seed"]
    assert template.postconditions == EXPECTED_POSTCONDITIONS


@pytest.mark.asyncio
async def test_replay_reproduces_constituent_outputs():
    """(a) Learned replay == executing the constituent subsequence."""
    _, templates = await mine_promoted()
    template = templates[0]
    learner = InstructionLearner(AgenticTransformer())
    learned = learner.create_instruction(template)

    # Mixed constituent types promote to the sequential-composition
    # type used by the composite machinery.
    assert learned.type == InstructionType.CONTROL_FLOW

    # Reference: execute the constituent subsequence on the same input.
    reference_trace = await record_trace(list(template.components), {"seed": 42})
    reference_outputs: Dict[str, Any] = {}
    for entry in reference_trace:
        reference_outputs.update(entry["outputs"])

    # Wrapped (run_stream) convention.
    result = await learned.execute({"ctx": {"seed": 42}})
    assert result == reference_outputs

    # Flat convention must be equivalent: the replay context is built
    # from the bare parameter dict when no "ctx" wrapper is present.
    flat_result = await learned.execute({"seed": 42})
    assert flat_result == result


@pytest.mark.asyncio
async def test_promotion_is_deterministic():
    """(b) Same traces + seed -> same promoted ids, in the same order."""
    learner_a, templates_a = await mine_promoted(seed=42)
    learner_b, templates_b = await mine_promoted(seed=42)

    names_a = [t.name for t in templates_a]
    names_b = [t.name for t in templates_b]
    assert names_a == names_b
    assert len(names_a) > 0
    # Pattern ids are content-derived, not instance counters.
    assert names_a[0].startswith("learned_pattern_")

    ids_a = [learner_a.create_instruction(t).id for t in templates_a]
    ids_b = [learner_b.create_instruction(t).id for t in templates_b]
    assert ids_a == ids_b


@pytest.mark.asyncio
async def test_bodyless_template_refuses_to_execute():
    """(c) A template mined without bodies raises NotImplementedError."""
    learner = InstructionLearner(AgenticTransformer())
    for components in (None, []):
        bodyless = InstructionTemplate(
            name="bodyless_template",
            type=InstructionType.LLM_INTERACTION,
            params={},
            preconditions=["requires_seed"],
            postconditions=[],
            components=components,
        )
        instruction = learner.create_instruction(bodyless)
        with pytest.raises(NotImplementedError, match="no constituent"):
            await instruction.execute({"seed": 42})


@pytest.mark.asyncio
async def test_declared_postconditions_hold_after_replay():
    """(d) Every produces_X postcondition holds after replay."""
    _, templates = await mine_promoted()
    template = templates[0]
    learner = InstructionLearner(AgenticTransformer())
    learned = learner.create_instruction(template)

    result = await learned.execute({"ctx": {"seed": 42}})

    for post in template.postconditions:
        assert post.startswith("produces_")
        assert post.removeprefix("produces_") in result
    # The registered postcondition lambdas hold against the result.
    assert all(check(result) for check in learned.context.postconditions)
    # Declared outputs are advertised on the instruction itself.
    assert sorted(learned.outputs) == sorted(
        p.removeprefix("produces_") for p in template.postconditions
    )

    # Preconditions are enforced on replay: without the external seed
    # input the learned instruction refuses to run.
    with pytest.raises(ValueError, match="Preconditions not met"):
        await learned.execute({"ctx": {}})
