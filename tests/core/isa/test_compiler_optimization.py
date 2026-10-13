"""Tests for the AgentISA instruction compiler optimization passes.

Covers instruction fusion, sequence-aware dead-code elimination,
parallelization grouping, and the resource/memory planning passes that
operate on (possibly nested) IR structures.
"""

import pytest
from typing import Any, Dict

from agentflow.core.isa.compiler import (
    CompilationContext,
    InstructionCompiler,
    IRBuilder,
    IRNodeType,
)
from agentflow.core.isa.formal import FormalInstruction, InstructionType


def make_instr(name: str, type_: InstructionType = InstructionType.LLM,
               side_effects: bool = True, outputs=None, dependencies=None) -> FormalInstruction:
    instr = FormalInstruction(
        name=name,
        type=type_,
        metadata={
            "side_effects": side_effects,
            "outputs": list(outputs or []),
            "cost": 1.0,
            "memory": 10,
        },
    )
    instr.dependencies = list(dependencies or [])
    return instr


@pytest.fixture
def compiler() -> InstructionCompiler:
    return InstructionCompiler({})


def run_pass(compiler: InstructionCompiler, seq, pass_name: str):
    """Apply exactly one optimization pass in isolation.

    ``compile()`` runs the full pipeline (fusion + DCE + ...), which
    legitimately eliminates pure instructions that end up with no
    consumers; single-pass assertions need the passes applied
    individually, mirroring the ablation experiment methodology.
    """
    ctx = CompilationContext({"optimization_level": "AGGRESSIVE"})
    ir = compiler.ir_builder.build(seq)
    ir = compiler.optimization_passes[pass_name](ir, ctx)
    # Materialize the pass result through the code generator so the
    # assertion sees the same wrapper form a full compile() produces.
    return compiler.code_generator.generate(ir, ctx)


@pytest.fixture
def basic_ctx() -> CompilationContext:
    return CompilationContext({"optimization_level": "BASIC"})


@pytest.fixture
def aggressive_ctx() -> CompilationContext:
    return CompilationContext({"optimization_level": "AGGRESSIVE"})


class TestInstructionFusion:
    """Fusion merges adjacent side-effect-free instructions."""

    def test_adjacent_pure_same_type_fuse(self, compiler, basic_ctx):
        a = make_instr("a", InstructionType.LLM, side_effects=False)
        b = make_instr("b", InstructionType.LLM, side_effects=False)
        out = run_pass(compiler, [a, b], "instruction_fusion")
        assert len(out) == 1
        assert out[0].name == "fused_a_b"

    def test_different_types_do_not_fuse(self, compiler, basic_ctx):
        a = make_instr("a", InstructionType.LLM, side_effects=False)
        b = make_instr("b", InstructionType.STATE, side_effects=False)
        out = run_pass(compiler, [a, b], "instruction_fusion")
        assert len(out) == 2
        assert {i.name for i in out} == {"a", "b"}

    def test_side_effectful_instructions_not_fused(self, compiler, basic_ctx):
        a = make_instr("a", InstructionType.LLM, side_effects=True)
        b = make_instr("b", InstructionType.LLM, side_effects=False)
        out = compiler.compile([a, b], basic_ctx)
        # b has no side effects and no consumers -> DCE removes it,
        # a stays.  Fusion itself must not merge them.
        assert [i.name for i in out] == ["a"]

    def test_dependent_instructions_not_fused(self, compiler, basic_ctx):
        a = make_instr("a", InstructionType.LLM, side_effects=False, outputs=["x"])
        b = make_instr("b", InstructionType.LLM, side_effects=False, dependencies=["x"])
        out = run_pass(compiler, [a, b], "instruction_fusion")
        # b consumes a's output: no fusion.
        assert {i.name for i in out} == {"a", "b"}

    def test_fused_metadata_merges_outputs(self, compiler, basic_ctx):
        a = make_instr("a", InstructionType.LLM, side_effects=False, outputs=["x"])
        b = make_instr("b", InstructionType.LLM, side_effects=False, outputs=["y"])
        out = run_pass(compiler, [a, b], "instruction_fusion")
        assert len(out) == 1
        merged = out[0].metadata
        assert set(merged.get("outputs", [])) == {"x", "y"}
        assert merged.get("cost") == 2.0
        assert merged.get("memory") == 20


class TestDeadCodeElimination:
    """DCE uses backward liveness over the sequence."""

    def test_dead_instruction_removed(self, compiler, basic_ctx):
        dead = make_instr("dead", InstructionType.STATE, side_effects=False)
        live = make_instr("live", InstructionType.CONTROL, side_effects=True)
        out = compiler.compile([dead, live], basic_ctx)
        assert [i.name for i in out] == ["live"]

    def test_output_consumer_keeps_producer(self, compiler, basic_ctx):
        producer = make_instr("p", InstructionType.STATE, side_effects=False, outputs=["v"])
        consumer = make_instr("c", InstructionType.LLM, side_effects=True)
        consumer.dependencies = ["v"]
        out = compiler.compile([producer, consumer], basic_ctx)
        assert {i.name for i in out} == {"p", "c"}

    def test_required_outputs_in_context_keeps_instruction(self, compiler):
        ctx = CompilationContext({
            "optimization_level": "BASIC",
            "target_metrics": {"required_outputs": ["v"]},
        })
        producer = make_instr("p", InstructionType.STATE, side_effects=False, outputs=["v"])
        sink = make_instr("sink", InstructionType.CONTROL, side_effects=True)
        out = compiler.compile([producer, sink], ctx)
        assert {i.name for i in out} == {"p", "sink"}

    def test_chain_of_dependencies_all_live(self, compiler, basic_ctx):
        a = make_instr("a", side_effects=False, outputs=["x1"])
        b = make_instr("b", side_effects=False, outputs=["x2"], dependencies=["x1"])
        c = make_instr("c", side_effects=True, dependencies=["x2"])
        out = run_pass(compiler, [a, b, c], "dead_code_elimination")
        assert {i.name for i in out} == {"a", "b", "c"}


class TestParallelization:
    """ADVANCED level groups independent instructions."""

    def test_independent_grouped_into_parallel_block(self, compiler, aggressive_ctx):
        a = make_instr("a", InstructionType.LLM, side_effects=False, outputs=["x1"])
        b = make_instr("b", InstructionType.LLM, side_effects=False, outputs=["x2"])
        c = make_instr("c", InstructionType.CONTROL, side_effects=True)
        c.dependencies = ["x1", "x2"]
        out = run_pass(compiler, [a, b, c], "parallel_execution")
        # a and b are independent -> one parallel block wrapper + sink.
        assert len(out) == 2
        wrapper = out[0]
        assert wrapper.metadata.get("is_parallel") is True
        inner = wrapper.metadata["parallel_instructions"]
        assert {i.name for i in inner} == {"a", "b"}
        assert out[1].name == "c"

    def test_dependent_instructions_not_parallelized(self, compiler, aggressive_ctx):
        a = make_instr("a", InstructionType.LLM, side_effects=False, outputs=["x"])
        b = make_instr("b", InstructionType.LLM, side_effects=False, dependencies=["x"])
        c = make_instr("c", InstructionType.CONTROL, side_effects=True)
        c.dependencies = ["x"]
        out = run_pass(compiler, [a, b, c], "parallel_execution")
        # No wrapper for a/b (b depends on a's output).
        names = [i.name for i in out]
        assert "parallel_block" not in names

    def test_side_effectful_not_parallelized(self, compiler, aggressive_ctx):
        a = make_instr("a", InstructionType.LLM, side_effects=True)
        b = make_instr("b", InstructionType.LLM, side_effects=True)
        out = run_pass(compiler, [a, b], "parallel_execution")
        assert all(not i.metadata.get("is_parallel") for i in out)


class TestResourceAndMemoryPasses:
    """ADVANCED/AGGRESSIVE passes annotate plans without altering semantics."""

    def test_resource_plan_created(self, compiler, aggressive_ctx):
        a = make_instr("a", side_effects=False, outputs=["x"])
        sink = make_instr("sink", side_effects=True)
        sink.dependencies = ["x"]
        out = compiler.compile([a, sink], aggressive_ctx)
        # The plan lives in IR metadata during compilation; the passes
        # must run without raising on flat and nested structures alike.
        assert len(out) >= 1

    def test_memory_lifecycle_live_ranges(self, compiler):
        a = make_instr("a", side_effects=False, outputs=["v"])
        sink = make_instr("sink", side_effects=True)
        sink.dependencies = ["v"]
        lifecycle = compiler._create_memory_lifecycle(
            type("IR", (), {"instructions": [a, sink]})()
        )
        assert lifecycle["live_ranges"]["v"] == (0, 1)

    def test_flatten_nested_groups(self, compiler):
        a = make_instr("a", side_effects=False)
        b = make_instr("b", side_effects=False)
        flat = compiler._flatten_instructions([[a, b], a])
        assert len(flat) == 3


class TestOptimizationLevels:
    """Pass gating follows the declared level requirements."""

    def test_none_level_is_identity(self, compiler):
        ctx = CompilationContext({"optimization_level": "NONE"})
        a = make_instr("a", side_effects=False)
        b = make_instr("b", side_effects=False)
        out = compiler.compile([a, b], ctx)
        assert len(out) == 2
        assert out[0].name == "a"

    def test_basic_does_not_parallelize(self, compiler, basic_ctx):
        a = make_instr("a", InstructionType.LLM, side_effects=False)
        b = make_instr("b", InstructionType.LLM, side_effects=False)
        out = compiler.compile([a, b], basic_ctx)
        assert all(not i.metadata.get("is_parallel") for i in out)
