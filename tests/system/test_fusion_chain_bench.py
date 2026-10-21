"""Tests for the chain-enriched fusion benchmark (``fusion_chain_bench.py``).

E8-fusion-chains responds to the ``"optimization gains are small"``
criticism: the published compiler ladder
(``examples/benchmarks/isa_microbench_results.json``) shows
BASIC == ADVANCED == AGGRESSIVE on instruction reduction and
ADVANCED == AGGRESSIVE on dispatch because its random-slot workload
under-represents chain structure.  These tests pin:

* the chain-enriched workload generator: exact lengths, deterministic
  per seed, forced single-segment structure (fusible runs, dataflow
  chains, parallel batches, dead runs, effectful anchors), and a final
  sink that consumes exactly the leaf outputs;
* the IR-capturing compile wrapper: output identical to the public
  ``InstructionCompiler.compile``, the patched ``generate`` restored,
  and plans captured only at the levels whose passes produce them;
* the metrics: parallel-block unwrapping, declared-cost makespan
  (sequence sums, parallel blocks take the max), fusion survivor
  counting, and memory-plan peak/reuse liveness with the conservative
  production-before-release tie-break;
* the aggregation discipline: every aggregate equals ``statistics``
  applied to the raw per-seed records, single-seed stdev is 0.0;
* the level-differentiation semantics: AGGRESSIVE is structurally
  identical to ADVANCED on every seed under chain enrichment, while
  the memory plan differentiates only AGGRESSIVE;
* harness fidelity: the random-slot control arm reproduces the
  published microbench compiler numbers exactly;
* the generated artifact ``fusion_chain_results.json`` (skipif
  missing) satisfies the schema, aggregation discipline and the
  AGGRESSIVE saturation invariant.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

import pytest

# Make the benchmark script importable (it lives in examples/benchmarks).
BENCH_DIR = Path(__file__).resolve().parents[2] / "examples" / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

import fusion_chain_bench as bench  # noqa: E402  (needs BENCH_DIR on sys.path)

ARTIFACT = BENCH_DIR / "fusion_chain_results.json"
PUBLISHED = BENCH_DIR / "isa_microbench_results.json"
LEVELS = ["NONE", "BASIC", "ADVANCED", "AGGRESSIVE"]


def _forced_weights(name):
    weights = {k: 0.0 for k in
               ("fusion_chain", "parallel_batch", "dataflow_chain",
                "dead_chain", "effectful_anchor")}
    weights[name] = 1.0
    return weights


def _compile(compiler, seq, level):
    from agentflow.core.isa.compiler import CompilationContext
    ctx = CompilationContext({"optimization_level": level})
    return bench.compile_and_capture_ir(compiler, seq, ctx)


# ---------------------------------------------------------------------------
# Workload generation
# ---------------------------------------------------------------------------

class TestWorkloadGeneration:
    def test_exact_length_honored_including_sink(self):
        for length in (10, 30, 97, 200):
            rng = random.Random(7 + length)
            workload = bench.generate_chain_workload(
                rng, length, bench.CHAIN_MIX)
            assert len(workload.instructions) == length

    def test_census_counts_sum_to_length(self):
        rng = random.Random(11)
        workload = bench.generate_chain_workload(rng, 200, bench.CHAIN_MIX)
        segment_total = sum(
            count for key, count in workload.census.items()
            if key.endswith("_instructions"))
        assert segment_total == 200
        assert workload.census["total"] == 200

    def test_all_segment_types_present(self):
        rng = random.Random(23)
        workload = bench.generate_chain_workload(rng, 400, bench.CHAIN_MIX)
        for key in ("fusion_chain_instructions", "parallel_batch_instructions",
                    "dataflow_chain_instructions", "dead_chain_instructions",
                    "effectful_anchor_instructions"):
            assert workload.census[key] > 0

    def test_deterministic_per_seed(self):
        for seed in (0, 1, 4):
            w1 = bench.generate_chain_workload(
                random.Random(5000 + seed), 120, bench.CHAIN_MIX)
            w2 = bench.generate_chain_workload(
                random.Random(5000 + seed), 120, bench.CHAIN_MIX)
            sig1 = [(i.name, i.type, sorted(i.metadata.items(), key=str),
                     i.dependencies) for i in w1.instructions]
            sig2 = [(i.name, i.type, sorted(i.metadata.items(), key=str),
                     i.dependencies) for i in w2.instructions]
            assert sig1 == sig2
            assert w1.census == w2.census

    def test_forced_fusion_chain_run_is_pure_same_type_no_deps(self):
        rng = random.Random(3)
        workload = bench.generate_chain_workload(
            rng, 60, _forced_weights("fusion_chain"))
        body = workload.instructions[:-1]
        for seg, run in workload.runs:
            if seg != "fusion_chain":
                continue
            chunk = body[:run]
            body = body[run:]
            assert len(chunk) == run
            for instr in chunk:
                assert instr.metadata["side_effects"] is False
                assert instr.type == chunk[0].type
                assert instr.dependencies == []
                assert instr.metadata["outputs"]

    def test_forced_dataflow_chain_has_producer_consumer_edges(self):
        rng = random.Random(5)
        workload = bench.generate_chain_workload(
            rng, 60, _forced_weights("dataflow_chain"))
        body = workload.instructions[:-1]
        for seg, run in workload.runs:
            if seg != "dataflow_chain":
                continue
            chunk = body[:run]
            body = body[run:]
            assert len(chunk) == run
            for prev, cur in zip(chunk, chunk[1:]):
                assert cur.dependencies == prev.metadata["outputs"]
                assert prev.metadata["memory"] in (256, 512, 1024, 2048)

    def test_forced_parallel_batch_blocks_pairwise_fusion(self):
        rng = random.Random(9)
        workload = bench.generate_chain_workload(
            rng, 60, _forced_weights("parallel_batch"))
        body = workload.instructions[:-1]
        for seg, run in workload.runs:
            if seg != "parallel_batch":
                continue
            chunk = body[:run]
            body = body[run:]
            assert len(chunk) == run
            for prev, cur in zip(chunk, chunk[1:]):
                assert prev.type != cur.type  # no pairwise fusion
                assert cur.dependencies == []

    def test_forced_dead_chain_has_no_outputs_and_is_pure(self):
        rng = random.Random(13)
        workload = bench.generate_chain_workload(
            rng, 60, _forced_weights("dead_chain"))
        body = workload.instructions[:-1]
        assert body
        for instr in body:
            assert instr.metadata["side_effects"] is False
            assert instr.metadata["outputs"] == []

    def test_sink_is_effectful_and_consumes_leaf_outputs_only(self):
        rng = random.Random(31)
        workload = bench.generate_chain_workload(rng, 150, bench.CHAIN_MIX)
        instructions = workload.instructions
        sink = instructions[-1]
        assert sink.metadata["side_effects"] is True
        all_outputs = [o for i in instructions[:-1]
                       for o in i.metadata.get("outputs", [])]
        consumed = {d for i in instructions[:-1] for d in i.dependencies}
        leaves = [o for o in all_outputs if o not in consumed]
        assert sorted(sink.dependencies) == sorted(set(leaves))
        # Dataflow-internal outputs must NOT be leaves (consumed in-sequence).
        assert any(set(i.dependencies) & set(all_outputs)
                   for i in instructions[:-1])

    def test_every_instruction_declares_cost(self):
        rng = random.Random(17)
        workload = bench.generate_chain_workload(rng, 80, bench.CHAIN_MIX)
        for instr in workload.instructions:
            assert "cost" in instr.metadata
            assert instr.metadata["cost"] >= 0


# ---------------------------------------------------------------------------
# Compile wrapper with IR capture
# ---------------------------------------------------------------------------

class TestCompileCapture:
    def _compiler(self):
        from agentflow.core.isa.compiler import InstructionCompiler
        return InstructionCompiler({})

    def _seq(self, seed=41, length=80):
        return bench.generate_chain_workload(
            random.Random(seed), length, bench.CHAIN_MIX).instructions

    def test_output_matches_public_compile(self):
        from agentflow.core.isa.compiler import CompilationContext
        compiler = self._compiler()
        seq = self._seq()
        for level in LEVELS:
            out_a, _ = _compile(compiler, seq, level)
            out_b = compiler.compile(
                seq, CompilationContext({"optimization_level": level}))
            assert [i.name for i in out_a] == [i.name for i in out_b]

    def test_generate_method_restored(self):
        compiler = self._compiler()
        original = compiler.code_generator.generate
        _compile(compiler, self._seq(), "AGGRESSIVE")
        assert compiler.code_generator.generate == original
        # And a later compile still works normally.
        out, _ = _compile(compiler, self._seq(seed=42), "BASIC")
        assert out

    def test_memory_plan_captured_only_at_aggressive(self):
        compiler = self._compiler()
        seq = self._seq(seed=43)
        for level in LEVELS:
            _, ir = _compile(compiler, seq, level)
            if level == "AGGRESSIVE":
                assert "memory_plan" in ir.metadata
            else:
                assert "memory_plan" not in ir.metadata

    def test_resource_plan_captured_at_advanced_and_above(self):
        compiler = self._compiler()
        seq = self._seq(seed=44)
        for level in LEVELS:
            _, ir = _compile(compiler, seq, level)
            if level in ("ADVANCED", "AGGRESSIVE"):
                assert "resource_plan" in ir.metadata
            else:
                assert "resource_plan" not in ir.metadata


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def _formal(self, name, cost=1.0, **metadata):
        from agentflow.core.isa.formal import FormalInstruction, InstructionType
        meta = {"cost": cost}
        meta.update(metadata)
        return FormalInstruction(name=name, type=InstructionType.LLM,
                                 metadata=meta)

    def _block(self, members):
        from agentflow.core.isa.formal import FormalInstruction, InstructionType
        return FormalInstruction(
            name="parallel_block", type=InstructionType.CONTROL,
            metadata={"parallel_instructions": members, "is_parallel": True})

    def test_effective_count_unwraps_parallel_blocks(self):
        items = [self._formal("a"), self._block([self._formal("b"),
                                                 self._formal("c")]),
                 self._formal("d")]
        assert bench.effective_instruction_count(items) == 4
        assert bench.dispatch_count(items) == 3

    def test_declared_cost_makespan_sums_sequence_costs(self):
        items = [self._formal("a", 0.5), self._formal("b", 1.5),
                 self._formal("c", 2.0)]
        assert bench.declared_cost_makespan(items) == pytest.approx(4.0)

    def test_declared_cost_makespan_parallel_block_takes_max(self):
        items = [self._formal("a", 0.5),
                 self._block([self._formal("b", 1.5), self._formal("c", 2.0)]),
                 self._formal("d", 1.0)]
        assert bench.declared_cost_makespan(items) == pytest.approx(3.5)

    def test_declared_cost_defaults_to_one(self):
        from agentflow.core.isa.formal import FormalInstruction, InstructionType
        plain = FormalInstruction(name="x", type=InstructionType.LLM,
                                  metadata={})
        assert bench.declared_cost_makespan([plain]) == pytest.approx(1.0)

    def test_fused_instruction_count_counts_survivors(self):
        items = [self._formal("fused_a_b"), self._formal("fused_c_d"),
                 self._formal("e"), self._block([self._formal("fused_e_f")])]
        assert bench.fused_instruction_count(items) == 2

    def test_parallel_block_stats(self):
        items = [self._block([self._formal("a"), self._formal("b"),
                             self._formal("c")]), self._formal("d"),
                 self._block([self._formal("e"), self._formal("f")])]
        wrappers, max_width = bench.parallel_block_stats(items)
        assert wrappers == 2
        assert max_width == 3

    def test_memory_metrics_peak_and_reuse(self):
        # Two disjoint live ranges (0,1) and (3,4): peak 1, reuse 1.
        lifecycle = {"live_ranges": {"x": (0, 1), "y": (3, 4)}}
        metrics = bench.memory_plan_metrics(
            {"lifecycle": lifecycle, "allocation": {"total_bytes": 12.0},
             "optimization": ["reuse_disjoint_buffers"]})
        assert metrics["peak_live_outputs"] == 1
        assert metrics["distinct_tracked_outputs"] == 2
        assert metrics["buffer_reuse_potential"] == 1
        assert metrics["total_declared_memory_bytes"] == 12.0
        assert metrics["optimizations"] == ["reuse_disjoint_buffers"]

    def test_memory_metrics_conservative_tie_break(self):
        # y is produced exactly when x is last consumed: both must be
        # counted live at that position (production before release).
        lifecycle = {"live_ranges": {"x": (0, 2), "y": (2, 4)}}
        metrics = bench.memory_plan_metrics({"lifecycle": lifecycle})
        assert metrics["peak_live_outputs"] == 2
        assert metrics["buffer_reuse_potential"] == 0

    def test_memory_metrics_none_when_no_plan(self):
        assert bench.memory_plan_metrics(None) is None


# ---------------------------------------------------------------------------
# Aggregation discipline
# ---------------------------------------------------------------------------

class TestAggregation:
    def test_aggregates_are_statistics_over_raw(self):
        records = [
            {"length": 10, "level": "BASIC", "seed": s,
             "effective_instruction_count": 10 - s,
             "instruction_count_reduction": 0.1 * s,
             "dispatch_count": 5 + s,
             "declared_cost_makespan": 2.0 * s,
             "fused_instruction_count": s,
             "parallel_block_wrappers": s % 2,
             "max_parallel_width": s,
             "memory_plan_present": s >= 3,
             "peak_live_outputs": s if s >= 3 else None,
             "distinct_tracked_outputs": s + 1 if s >= 3 else None,
             "buffer_reuse_potential": s - 1 if s >= 3 else None,
             "memory_plan_optimizations": ["reuse_disjoint_buffers"]
                                          if s >= 3 else None,
             "total_declared_memory_bytes": float(s) if s >= 3 else None}
            for s in range(5)
        ]
        agg = bench.aggregate_records(records)[("10", "BASIC")]
        for key in ("effective_instruction_count",
                    "instruction_count_reduction", "dispatch_count",
                    "declared_cost_makespan", "fused_instruction_count",
                    "parallel_block_wrappers", "max_parallel_width"):
            values = [r[key] for r in records]
            assert agg[key] == {"mean": statistics.mean(values),
                                "stdev": statistics.stdev(values)}
        for key in ("peak_live_outputs", "distinct_tracked_outputs",
                    "buffer_reuse_potential", "total_declared_memory_bytes"):
            values = [r[key] for r in records if r[key] is not None]
            assert agg[key] == {"mean": statistics.mean(values),
                                "stdev": statistics.stdev(values)}
        assert agg["memory_plan_present"] == {
            "seeds_with_plan": 2, "seeds_total": 5}
        assert agg["memory_plan_optimizations"] == {
            "union": ["reuse_disjoint_buffers"]}

    def test_single_seed_stdev_zero(self):
        records = [{"length": 10, "level": "NONE", "seed": 0,
                    "effective_instruction_count": 10,
                    "instruction_count_reduction": 0.0,
                    "dispatch_count": 10, "declared_cost_makespan": 10.0,
                    "fused_instruction_count": 0,
                    "parallel_block_wrappers": 0, "max_parallel_width": 0,
                    "memory_plan_present": False,
                    "peak_live_outputs": None,
                    "distinct_tracked_outputs": None,
                    "buffer_reuse_potential": None,
                    "memory_plan_optimizations": None,
                    "total_declared_memory_bytes": None}]
        agg = bench.aggregate_records(records)[("10", "NONE")]
        for key in ("effective_instruction_count", "dispatch_count",
                    "declared_cost_makespan"):
            assert agg[key]["stdev"] == 0.0

    def test_missing_memory_values_stay_null(self):
        records = [{"length": 10, "level": "BASIC", "seed": 0,
                    "effective_instruction_count": 8,
                    "instruction_count_reduction": 0.2,
                    "dispatch_count": 8, "declared_cost_makespan": 9.0,
                    "fused_instruction_count": 1,
                    "parallel_block_wrappers": 0, "max_parallel_width": 0,
                    "memory_plan_present": False,
                    "peak_live_outputs": None,
                    "distinct_tracked_outputs": None,
                    "buffer_reuse_potential": None,
                    "memory_plan_optimizations": None,
                    "total_declared_memory_bytes": None}]
        agg = bench.aggregate_records(records)[("10", "BASIC")]
        assert agg["peak_live_outputs"] is None
        assert agg["memory_plan_optimizations"] is None


# ---------------------------------------------------------------------------
# Level differentiation on chain-enriched workloads
# ---------------------------------------------------------------------------

class TestChainArmBehavior:
    def _records(self, length=60, seeds=3, seed_base=5000):
        compiler = bench.InstructionCompiler({})
        out = []
        for seed in range(seeds):
            rng = random.Random(seed_base + seed)
            workload = bench.generate_chain_workload(rng, length,
                                                     bench.CHAIN_MIX)
            for level in LEVELS:
                out.append(bench.compile_record(
                    compiler, workload.instructions, level, length, seed,
                    "chain_enriched"))
        return out

    def test_basic_reduces_effective_count(self):
        records = self._records()
        none_eff = [r["effective_instruction_count"] for r in records
                    if r["level"] == "NONE"]
        basic_eff = [r["effective_instruction_count"] for r in records
                     if r["level"] == "BASIC"]
        assert statistics.mean(basic_eff) < statistics.mean(none_eff)

    def test_advanced_dispatch_below_basic(self):
        records = self._records()
        basic_disp = [r["dispatch_count"] for r in records
                      if r["level"] == "BASIC"]
        adv_disp = [r["dispatch_count"] for r in records
                    if r["level"] == "ADVANCED"]
        assert statistics.mean(adv_disp) < statistics.mean(basic_disp)

    def test_aggressive_structurally_identical_to_advanced(self):
        records = self._records(length=80, seeds=5)
        structural = bench.STRUCTURAL_METRIC_KEYS
        by = {(r["seed"], r["level"]): r for r in records}
        for seed in range(5):
            adv, aggr = by[(seed, "ADVANCED")], by[(seed, "AGGRESSIVE")]
            for key in structural:
                assert adv[key] == aggr[key]

    def test_memory_plan_only_aggressive_has_content(self):
        records = self._records(length=80, seeds=5)
        for r in records:
            if r["level"] == "AGGRESSIVE":
                assert r["memory_plan_present"] is True
                assert r["buffer_reuse_potential"] >= 0
            else:
                assert r["memory_plan_present"] is False
                assert r["buffer_reuse_potential"] is None


# ---------------------------------------------------------------------------
# Control-arm fidelity vs the published microbench numbers
# ---------------------------------------------------------------------------

class TestControlArmFidelity:
    @pytest.mark.skipif(not PUBLISHED.exists(),
                        reason="published microbench results not present")
    def test_control_arm_reproduces_published_numbers(self):
        published = json.loads(PUBLISHED.read_text())
        compiler_experiment = next(e for e in published
                                   if e["experiment"] == "compiler")
        expected = {(d["length"], d["level"]): d
                    for d in compiler_experiment["data"]}
        records = bench.run_arm(
            compiler=bench.InstructionCompiler({}),
            workload_factory=bench.control_workload_factory(),
            lengths=[50, 200, 1000], levels=LEVELS, seeds=5,
            seed_base=2000, arm="random_slots_control")
        agg = bench.aggregate_records(records)
        for (length, level), exp in expected.items():
            got = agg[(str(length), level)]
            assert got["instruction_count_reduction"]["mean"] == pytest.approx(
                exp["effective_instruction_count_reduction"]["mean"],
                abs=0.0), (length, level)
            assert got["dispatch_count"]["mean"] == pytest.approx(
                exp["dispatch_count"]["mean"], abs=0.0), (length, level)

    def test_control_workload_factory_matches_microbench_generator(self):
        import isa_microbench as mb
        factory = bench.control_workload_factory()
        rng_a = random.Random(2000 + 2)
        rng_b = random.Random(2000 + 2)
        seq = factory(rng_a, 40)
        ref = mb.synthetic_formal_sequence(rng_b, 40, p_dead=0.25, p_pure=0.55)
        assert [i.name for i in seq] == [i.name for i in ref]


# ---------------------------------------------------------------------------
# Differentiation summary
# ---------------------------------------------------------------------------

class TestDifferentiation:
    def _fake_records(self):
        # Two seeds, two levels, length 10: AGGRESSIVE mirrors ADVANCED
        # structurally but adds a memory plan.
        records = []
        for seed in range(2):
            base = {"length": 10, "seed": seed,
                    "effective_instruction_count": 8,
                    "instruction_count_reduction": 0.2,
                    "dispatch_count": 4, "declared_cost_makespan": 5.0,
                    "fused_instruction_count": 2,
                    "parallel_block_wrappers": 1, "max_parallel_width": 3}
            for level, extra in (
                    ("ADVANCED", {"memory_plan_present": False}),
                    ("AGGRESSIVE", {"memory_plan_present": True,
                                    "peak_live_outputs": 2,
                                    "distinct_tracked_outputs": 5,
                                    "buffer_reuse_potential": 3,
                                    "memory_plan_optimizations": ["x"],
                                    "total_declared_memory_bytes": 9.0})):
                rec = dict(base, level=level, **extra)
                records.append(rec)
        return records

    def test_differentiation_paired_deltas(self):
        records = self._fake_records()
        diff = bench.differentiate(records)[0]
        pairs = {tuple(d["pair"]): d for d in diff["pairs"]}
        adv_aggr = pairs[("ADVANCED", "AGGRESSIVE")]
        assert adv_aggr["deltas"]["dispatch_count"] == 0.0
        assert adv_aggr["deltas"]["effective_instruction_count"] == 0.0
        assert adv_aggr["structural_metrics_identical_per_seed"] is True
        assert adv_aggr["memory_plan_differentiation"] == {
            "lower_seeds_with_plan": 0, "upper_seeds_with_plan": 2,
            "upper_mean_buffer_reuse_potential": 3.0}

    def test_differentiation_flags_structural_delta(self):
        records = self._fake_records()
        # Perturb AGGRESSIVE dispatch on one seed: identity must break.
        records[-1]["dispatch_count"] += 1
        diff = bench.differentiate(records)[0]
        pairs = {tuple(d["pair"]): d for d in diff["pairs"]}
        assert pairs[("ADVANCED", "AGGRESSIVE")][
            "structural_metrics_identical_per_seed"] is False

    def test_verdict_fields_present(self):
        records = self._fake_records()
        verdict = bench.verdict(records)
        assert verdict["aggressive_structurally_identical_to_advanced"] is True
        assert verdict["aggressive_memory_plan_differentiates"] is True


# ---------------------------------------------------------------------------
# Full result building, determinism and CLI
# ---------------------------------------------------------------------------

class TestBuildAndCLI:
    def test_build_results_deterministic_data(self):
        args = dict(lengths=[30], seeds=2, seed_base=5000)
        r1 = bench.build_results(**args)
        r2 = bench.build_results(**args)
        assert r1["data"] == r2["data"]
        assert r1["differentiation"] == r2["differentiation"]
        assert r1["workload_census"] == r2["workload_census"]
        assert r1["coverage"]["status"] == "complete"
        assert r1["coverage"]["cells_run"] == r1["coverage"]["cells_planned"]
        # Provenance varies (timestamp) but keys are all present.
        for key in ("git_rev", "timestamp", "seeds", "scale"):
            assert key in r1["provenance"]
        assert r1["provenance"]["seeds"] == [0, 1]
        assert r1["provenance"]["scale"]["lengths"] == [30]

    def test_main_writes_output(self, tmp_path, monkeypatch):
        target = tmp_path / "out.json"
        argv = ["--output", str(target), "--seeds", "2",
                "--lengths", "30", "--seed-base", "5000"]
        bench.main(argv)
        data = json.loads(target.read_text())
        assert data["experiment"] == "fusion_chain_level_differentiation"
        assert data["provenance"]["seeds"] == [0, 1]
        assert data["coverage"]["status"] == "complete"
        chain = data["data"]["chain_enriched"]
        assert {(d["length"], d["level"]) for d in chain} == {
            (30, level) for level in LEVELS}


# ---------------------------------------------------------------------------
# Generated artifact (skipif missing; produced by the real run)
# ---------------------------------------------------------------------------

class TestArtifact:
    @pytest.mark.skipif(not ARTIFACT.exists(),
                        reason="fusion_chain_results.json not generated yet")
    def test_schema(self):
        data = json.loads(ARTIFACT.read_text())
        prov = data["provenance"]
        assert prov["git_rev"]
        assert prov["seeds"] == [0, 1, 2, 3, 4]
        assert prov["scale"]["lengths"] == [50, 200, 1000]
        assert prov["scale"]["seeds_per_length"] == 5
        assert data["coverage"]["status"] == "complete"
        for arm in ("chain_enriched", "random_slots_control"):
            assert len(data["data"][arm]) == 3 * 4 * 5  # raw per-seed records
            assert len(data["aggregates"][arm]) == 3 * 4
            assert len(data["differentiation"][arm]) == 3

    @pytest.mark.skipif(not ARTIFACT.exists(),
                        reason="fusion_chain_results.json not generated yet")
    def test_aggregation_discipline(self):
        data = json.loads(ARTIFACT.read_text())
        for arm, entries in data["data"].items():
            grouped = {}
            for rec in entries:
                assert "seed" in rec
                grouped.setdefault((rec["length"], rec["level"]),
                                   []).append(rec)
            for (length, level), recs in grouped.items():
                assert len(recs) == 5
                # Aggregates in differentiation/headline are derived from
                # these raw records; here we check the aggregate table
                # exists for every cell.
                assert any(h["length"] == length and h["level"] == level
                           for h in data["aggregates"][arm]), (arm, length,
                                                               level)
            # Mean of raw dispatch counts must equal reported means.
            for cell in data["aggregates"][arm]:
                recs = grouped[(cell["length"], cell["level"])]
                for key in ("effective_instruction_count",
                            "dispatch_count", "declared_cost_makespan"):
                    mean = statistics.mean(r[key] for r in recs)
                    assert cell[key]["mean"] == pytest.approx(mean, abs=1e-9)

    @pytest.mark.skipif(not ARTIFACT.exists(),
                        reason="fusion_chain_results.json not generated yet")
    def test_aggressive_saturation_invariant(self):
        data = json.loads(ARTIFACT.read_text())
        for arm in ("chain_enriched", "random_slots_control"):
            grouped = {}
            for rec in data["data"][arm]:
                grouped.setdefault((rec["length"], rec["seed"]), {})[
                    rec["level"]] = rec
            for (length, seed), by_level in grouped.items():
                for key in bench.STRUCTURAL_METRIC_KEYS:
                    assert by_level["ADVANCED"][key] == \
                        by_level["AGGRESSIVE"][key], (arm, length, seed, key)
            verdict = data["verdicts"][arm]
            assert verdict["aggressive_structurally_identical_to_advanced"] \
                is True

    @pytest.mark.skipif(not ARTIFACT.exists(),
                        reason="fusion_chain_results.json not generated yet")
    def test_chain_arm_memory_content_control_empty(self):
        data = json.loads(ARTIFACT.read_text())
        chain = [r for r in data["data"]["chain_enriched"]
                 if r["level"] == "AGGRESSIVE" and r["length"] == 1000]
        control = [r for r in data["data"]["random_slots_control"]
                   if r["level"] == "AGGRESSIVE" and r["length"] == 1000]
        assert statistics.mean(r["buffer_reuse_potential"]
                               for r in chain) > 0
        assert statistics.mean(r["buffer_reuse_potential"]
                               for r in control) == 0.0
