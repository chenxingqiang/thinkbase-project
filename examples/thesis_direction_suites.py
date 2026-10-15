"""Five master's-thesis-direction task suites, batch-run through AgentISA.

Each direction is materialized as a hand-authored, executable
instruction stream with realistic structure for that research area --
deliberately varying graph shapes (linear+guard, two-branch merge,
wide fan-out, branch convergence, pure system pipeline) so the batch
exercises different parts of the ISA type system and optimizer:

  A. data_pipeline  -- ETL data cleaning        (chain, 1 LLM, 1 guard)
  B. rag_qa         -- retrieval-augmented QA   (two-branch, 2 LLM, 1 guard)
  C. coordination   -- multi-agent task planning (wide fan-out, 0 LLM, 1 guard)
  D. code_audit     -- static security audit    (branch merge, 1 LLM, 1 guard)
  E. eval_pipeline  -- experiment leaderboard   (pure system, 0 LLM, 1 guard)

Every suite runs the same verification battery as the 100-task
stress harness: static compile validity (NONE/AGGRESSIVE with
dataflow preservation), cached repeats (byte-identical artifacts,
zero model calls after the first realization), closed-loop hot-path
adoption consistency, full-stack determinism, and ELL export with
JSONL round-trip.

LLM-typed steps use the deterministic RuleClassifierBackend as a
stand-in model (as in the paper's microbenchmarks: the cost
accounting is what is measured, not the classifier's intelligence).

Usage:
    ./venv/bin/python examples/thesis_direction_suites.py \
        --repeats 4 --hot-threshold 2 \
        --output examples/thesis_direction_results.json
"""

import argparse
import asyncio
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentflow.core.isa.ell_export import (
    export_stream_to_ell, load_ell_trace, save_ell_trace,
)
from agentflow.core.isa.instruction import InstructionType as IsaType
from agentflow.core.isa.instruction_optimization import (
    OptimizationManager, OptimizationMetrics,
)
from agentflow.sandbox.local_executor import LocalSandboxExecutor

from random_tasks_stress import _instruction_factory, compile_check
from system_task_demo import RuleClassifierBackend, inject_sandbox, run_stream

# ---------------------------------------------------------------------------
# Suite A: 数据管道 / ETL 数据清洗 (data_pipeline)
# ---------------------------------------------------------------------------


def gen_etl_corpus(rng: random.Random) -> Dict[str, str]:
    """Messy customer records: duplicates, casing drift, nulls."""
    cities = ["Beijing", "beijing", "Shanghai", "SHANGHAI", "Shenzhen"]
    out = []
    for i in range(300):
        out.append(json.dumps({
            "id": rng.randint(1, 200),  # duplicates by design
            "city": rng.choice(cities),
            "amount": rng.choice([None, rng.randint(1, 9999)]),
            "ts": f"2025-0{rng.randint(1, 9)}-1{rng.randint(0, 9)}",
        }))
    return {"raw_records.jsonl": "\n".join(out) + "\n"}


def build_data_pipeline(backend: RuleClassifierBackend,
                        corpus: Dict[str, str]) -> List[Instruction]:
    stream: List[Instruction] = []
    stream.append(_instruction_factory(
        "ingest_records", IsaType.STATE_MANAGEMENT,
        lambda p, s: {"records": [
            json.loads(l) for l in
            s.sandbox.read_file("raw_records.jsonl").decode().splitlines()
        ]},
        deps=[], outputs=["records"], side_effects=True,
        description="Read raw customer records"))

    stream.append(_instruction_factory(
        "dedupe_records", IsaType.CONTROL_FLOW,
        lambda p, s: {"deduped": [
            json.loads(j) for j in {
                json.dumps(r, sort_keys=True)
                for r in p["ctx"]["records"]
            }
        ]},
        deps=["records"], outputs=["deduped"], side_effects=False,
        description="Deduplicate records by canonical form"))

    stream.append(_instruction_factory(
        "schema_guard", IsaType.CONTROL_FLOW,
        (lambda p, s: (
            None if all("id" in r for r in p["ctx"]["deduped"])
            else _raise("schema guard failed")) or {}),
        deps=["deduped"], outputs=[], side_effects=False,
        description="Defensive schema check (dead code)"))

    stream.append(_instruction_factory(
        "normalize_fields", IsaType.CONTROL_FLOW,
        lambda p, s: {"normalized": [
            {**r, "city": str(r.get("city", "")).title(),
             "amount": r.get("amount") or 0}
            for r in p["ctx"]["deduped"]
        ]},
        deps=["deduped"], outputs=["normalized"], side_effects=False,
        description="Normalize city casing and null amounts"))

    def _enrich(p, instr):
        return _cacheable_compute(
            instr, {"n": len(p["ctx"]["normalized"])},
            lambda: {"enriched": {
                str(r["id"]): backend.infer(f"city:{r['city']}")
                for r in p["ctx"]["normalized"]
            }})

    stream.append(_instruction_factory(
        "enrich_category", IsaType.LLM_INTERACTION, _enrich,
        deps=["normalized"], outputs=["enriched"], side_effects=False,
        description="Cacheable LLM step: link each record to a category",
        cache_config={"max_size": 100_000, "ttl": 3600}))

    stream.append(_instruction_factory(
        "quality_summary", IsaType.CONTROL_FLOW,
        lambda p, s: {"quality": {
            "total": len(p["ctx"]["normalized"]),
            "categories": sorted(set(p["ctx"]["enriched"].values())),
            "zero_amount": sum(
                1 for r in p["ctx"]["normalized"] if r["amount"] == 0),
        }},
        deps=["normalized", "enriched"], outputs=["quality"],
        side_effects=False, description="Aggregate data-quality metrics"))

    stream.append(_report_writer(
        "write_dataset", ["quality", "enriched"], "cleaned.json",
        deps=["quality"], description="Write the cleaned dataset"))
    return stream


# ---------------------------------------------------------------------------
# Suite B: RAG 检索增强问答 (rag_qa)
# ---------------------------------------------------------------------------

TOPIC_WORDS = ["cache", "sandbox", "compiler", "memory", "network",
               "storage", "latency", "retry", "shuffle", "index"]


def gen_rag_corpus(rng: random.Random) -> Dict[str, str]:
    docs = []
    for i in range(120):
        words = [rng.choice(TOPIC_WORDS) for _ in range(6)]
        docs.append(f"doc{i:03d}: " + " ".join(words))
    query = " ".join(rng.choice(TOPIC_WORDS) for _ in range(2))
    return {"corpus.txt": "\n".join(docs) + "\n", "query.txt": query + "\n"}


def build_rag_qa(backend: RuleClassifierBackend,
                 corpus: Dict[str, str]) -> List[Instruction]:
    stream: List[Instruction] = []
    # Branch 1: corpus path
    stream.append(_instruction_factory(
        "ingest_corpus", IsaType.STATE_MANAGEMENT,
        lambda p, s: {"chunks": sorted(
            s.sandbox.read_file("corpus.txt").decode().splitlines())},
        deps=[], outputs=["chunks"], side_effects=True,
        description="Read the document corpus"))
    # Branch 2: query path
    stream.append(_instruction_factory(
        "ingest_query", IsaType.STATE_MANAGEMENT,
        lambda p, s: {"query": s.sandbox.read_file("query.txt")
                      .decode().strip()},
        deps=[], outputs=["query"], side_effects=True,
        description="Read the user query"))

    def _embed(p, instr):
        return _cacheable_compute(
            instr, {"n": len(p["ctx"]["chunks"])},
            lambda: {"embeddings": {c: backend.infer(c.split(":")[0])
                                    for c in p["ctx"]["chunks"]}})

    stream.append(_instruction_factory(
        "embed_corpus", IsaType.LLM_INTERACTION, _embed,
        deps=["chunks"], outputs=["embeddings"], side_effects=False,
        description="Cacheable LLM step: embed every chunk",
        cache_config={"max_size": 100_000, "ttl": 3600}))

    stream.append(_instruction_factory(
        "retrieve_topk", IsaType.CONTROL_FLOW,
        lambda p, s: {"top_chunks": sorted(
            c for c in p["ctx"]["chunks"]
            if any(w in c for w in p["ctx"]["query"].split())
        )[:20]},
        deps=["embeddings", "query"], outputs=["top_chunks"],
        side_effects=False,
        description="Retrieve top-k chunks matching the query"))

    def _rerank(p, instr):
        tops = sorted(set(p["ctx"]["top_chunks"]))
        return _cacheable_compute(
            instr, {"n": len(tops)},
            lambda: {"reranked": sorted(
                tops, key=lambda c: (backend.infer(c), c))})

    stream.append(_instruction_factory(
        "rerank", IsaType.LLM_INTERACTION, _rerank,
        deps=["top_chunks"], outputs=["reranked"], side_effects=False,
        description="Cacheable LLM step: rerank retrieved chunks",
        cache_config={"max_size": 100_000, "ttl": 3600}))

    stream.append(_instruction_factory(
        "synthesize_answer", IsaType.CONTROL_FLOW,
        lambda p, s: {"answer":
                      f"Q: {p['ctx']['query']}\n" + "\n".join(
                          f"[{i+1}] {c}" for i, c in
                          enumerate(p["ctx"]["reranked"][:5]))},
        deps=["reranked", "query"], outputs=["answer"],
        side_effects=False, description="Assemble the cited answer"))

    stream.append(_instruction_factory(
        "citation_guard", IsaType.CONTROL_FLOW,
        (lambda p, s: (
            None if "[" in p["ctx"]["answer"]
            else _raise("citation guard failed")) or {}),
        deps=["answer"], outputs=[], side_effects=False,
        description="Defensive citation check (dead code)"))

    stream.append(_report_writer(
        "write_answer", ["answer"], "answer.md",
        deps=["answer"], description="Write the synthesized answer"))
    return stream


# ---------------------------------------------------------------------------
# Suite C: 多智能体任务协调 (coordination)
# ---------------------------------------------------------------------------


def gen_coordination_corpus(rng: random.Random) -> Dict[str, str]:
    tasks = []
    for i in range(24):
        tasks.append(json.dumps({
            "task": f"job{i:02d}",
            "effort": rng.randint(1, 9),
            "skill": rng.choice(["index", "shuffle", "compact"]),
        }))
    return {"tasks.jsonl": "\n".join(tasks) + "\n"}


def _plan_for(tag: str) -> Tuple[str, Callable]:
    def _plan(p, s):
        mine = [t for t in p["ctx"]["subtasks"]
                if t["skill"] == tag or tag == "index"]
        return {f"plan_{tag}": {
            "assigned": len(mine),
            "load": sum(t["effort"] for t in mine),
            "items": sorted(t["task"] for t in mine)[:20],
        }}
    return tag, _plan


def build_coordination(backend: RuleClassifierBackend,
                       corpus: Dict[str, str]) -> List[Instruction]:
    stream: List[Instruction] = []
    stream.append(_instruction_factory(
        "ingest_tasks", IsaType.STATE_MANAGEMENT,
        lambda p, s: {"tasks": [
            json.loads(l) for l in
            s.sandbox.read_file("tasks.jsonl").decode().splitlines()
        ]},
        deps=[], outputs=["tasks"], side_effects=True,
        description="Read the task backlog"))

    stream.append(_instruction_factory(
        "decompose_tasks", IsaType.CONTROL_FLOW,
        lambda p, s: {"subtasks": [
            {**t, "sub": f"{t['task']}#s{i % 3}"}
            for i, t in enumerate(p["ctx"]["tasks"])
        ]},
        deps=["tasks"], outputs=["subtasks"], side_effects=False,
        description="Split each task into subtasks"))

    # Wide fan-out: three per-agent plans, all pure, all independent.
    for tag in ("a", "b", "c"):
        _, fn = _plan_for(tag)
        stream.append(_instruction_factory(
            f"plan_agent_{tag}", IsaType.CONTROL_FLOW, fn,
            deps=["subtasks"], outputs=[f"plan_{tag}"],
            side_effects=False,
            description=f"Schedule agent {tag.upper()}'s subtasks"))

    stream.append(_instruction_factory(
        "merge_plans", IsaType.CONTROL_FLOW,
        lambda p, s: {"merged_plan": {
            "agents": {k: p["ctx"][f"plan_{k}"]
                       for k in ("a", "b", "c")},
            "total_assigned": sum(
                p["ctx"][f"plan_{k}"]["assigned"] for k in ("a", "b", "c")),
        }},
        deps=["plan_a", "plan_b", "plan_c"], outputs=["merged_plan"],
        side_effects=False,
        description="Merge per-agent plans and detect conflicts"))

    stream.append(_instruction_factory(
        "load_guard", IsaType.CONTROL_FLOW,
        (lambda p, s: (
            None if p["ctx"]["merged_plan"]["total_assigned"] >= 0
            else _raise("load guard failed")) or {}),
        deps=["merged_plan"], outputs=[], side_effects=False,
        description="Defensive load-balance check (dead code)"))

    stream.append(_report_writer(
        "write_schedule", ["merged_plan"], "schedule.md",
        deps=["merged_plan"], description="Write the team schedule"))
    return stream


# ---------------------------------------------------------------------------
# Suite D: 代码静态安全审计 (code_audit)
# ---------------------------------------------------------------------------

SNIPPETS = [
    "password = '{secret}'",
    "cursor.execute(query, params)",
    "eval(user_input)",
    "with open(path) as f: data = f.read()",
    "os.system('rm -rf /tmp/cache')",
    "hashlib.sha256(payload).hexdigest()",
    "requests.get(url, verify=False)",
    "logger.info('structured %s', event)",
    "subprocess.run(cmd, shell=True)",
    "token = jwt.encode(payload, key)",
    "open(tmp_path, 'w').write(body)",
    "random.seed(42)",
]


def gen_audit_corpus(rng: random.Random) -> Dict[str, str]:
    files = {}
    for fi in range(4):
        lines = []
        for li in range(30):
            lines.append(f"line{li:02d}: {rng.choice(SNIPPETS)}")
        files[f"module_{fi}.py"] = "\n".join(lines) + "\n"
    return files


def build_code_audit(backend: RuleClassifierBackend,
                     corpus: Dict[str, str]) -> List[Instruction]:
    stream: List[Instruction] = []
    stream.append(_instruction_factory(
        "ingest_sources", IsaType.STATE_MANAGEMENT,
        lambda p, s: {"sources": {
            f"module_{i}.py": s.sandbox.read_file(f"module_{i}.py")
            .decode().splitlines()
            for i in range(4)
        }},
        deps=[], outputs=["sources"], side_effects=True,
        description="Read the source modules"))

    stream.append(_instruction_factory(
        "parse_stats", IsaType.CONTROL_FLOW,
        lambda p, s: {"stats": {
            name: {"lines": len(lines), "tokens": sum(
                len(l.split()) for l in lines)}
            for name, lines in p["ctx"]["sources"].items()
        }},
        deps=["sources"], outputs=["stats"], side_effects=False,
        description="Per-module size statistics"))

    RISKY = ("eval(", "shell=True", "verify=False", "os.system", "password =")

    stream.append(_instruction_factory(
        "detect_patterns", IsaType.CONTROL_FLOW,
        lambda p, s: {"findings": sorted(
            f"{name}:{i}:{pat}"
            for name, lines in p["ctx"]["sources"].items()
            for i, l in enumerate(lines)
            for pat in RISKY if pat in l
        )},
        deps=["sources"], outputs=["findings"], side_effects=False,
        description="Match risky-code patterns"))

    def _classify(p, instr):
        findings = p["ctx"]["findings"]
        return _cacheable_compute(
            instr, {"n": len(findings)},
            lambda: {"severity": {f: backend.infer(f) for f in findings}})

    stream.append(_instruction_factory(
        "classify_severity", IsaType.LLM_INTERACTION, _classify,
        deps=["findings"], outputs=["severity"], side_effects=False,
        description="Cacheable LLM step: rate each finding's severity",
        cache_config={"max_size": 100_000, "ttl": 3600}))

    stream.append(_instruction_factory(
        "license_guard", IsaType.CONTROL_FLOW,
        (lambda p, s: (
            None if any("line00" in l for ls in p["ctx"]["sources"].values()
                        for l in ls) else _raise("license guard failed")
        ) or {}),
        deps=["sources"], outputs=[], side_effects=False,
        description="Defensive license check (dead code)"))

    # Branch merge: consumes both stats and severity.
    stream.append(_instruction_factory(
        "summarize_audit", IsaType.CONTROL_FLOW,
        lambda p, s: {"audit_summary": {
            "modules": len(p["ctx"]["stats"]),
            "total_lines": sum(v["lines"] for v in p["ctx"]["stats"].values()),
            "findings": len(p["ctx"]["severity"]),
            "high": sorted(f for f, sev in p["ctx"]["severity"].items()
                           if sev == "high")[:10],
        }},
        deps=["stats", "severity"], outputs=["audit_summary"],
        side_effects=False, description="Converge stats and severities"))

    stream.append(_report_writer(
        "write_audit", ["audit_summary"], "audit.md",
        deps=["audit_summary"], description="Write the audit report"))
    return stream


# ---------------------------------------------------------------------------
# Suite E: 实验评测流水线 (eval_pipeline)
# ---------------------------------------------------------------------------


def gen_eval_corpus(rng: random.Random) -> Dict[str, str]:
    configs = []
    for lr in (0.1, 0.01):
        for bs in (16, 32):
            for opt in ("sgd", "adam"):
                configs.append(json.dumps({
                    "config": f"lr{lr}_bs{bs}_{opt}",
                    "seed": rng.randint(0, 9999),
                }))
    return {"configs.jsonl": "\n".join(configs) + "\n"}


def build_eval_pipeline(backend: RuleClassifierBackend,
                        corpus: Dict[str, str]) -> List[Instruction]:
    stream: List[Instruction] = []
    stream.append(_instruction_factory(
        "ingest_configs", IsaType.STATE_MANAGEMENT,
        lambda p, s: {"configs": [
            json.loads(l) for l in
            s.sandbox.read_file("configs.jsonl").decode().splitlines()
        ]},
        deps=[], outputs=["configs"], side_effects=True,
        description="Read the experiment configurations"))

    stream.append(_instruction_factory(
        "expand_grid", IsaType.CONTROL_FLOW,
        lambda p, s: {"runs": [
            {**c, "rep": rep} for c in p["ctx"]["configs"]
            for rep in (0, 1, 2)
        ]},
        deps=["configs"], outputs=["runs"], side_effects=False,
        description="Expand configs into repeated runs"))

    def _simulate(p, s):
        def score(run):
            r = random.Random(run["seed"] + run["rep"])
            return round(
                0.5 + 0.3 * (len(run["config"]) % 7) / 7
                + r.random() * 0.05, 4)
        return {"scores": sorted(
            ({"config": r["config"], "rep": r["rep"], "score": score(r)}
             for r in p["ctx"]["runs"]),
            key=lambda x: (x["config"], x["rep"]))}

    stream.append(_instruction_factory(
        "simulate_runs", IsaType.CONTROL_FLOW, _simulate,
        deps=["runs"], outputs=["scores"], side_effects=False,
        description="Seeded deterministic training simulation"))

    stream.append(_instruction_factory(
        "collect_metrics", IsaType.CONTROL_FLOW,
        lambda p, s: {"metrics": sorted(
            ({"config": c, "mean": round(sum(
                r["score"] for r in p["ctx"]["scores"]
                if r["config"] == c) / 3, 4)}
             for c in {r["config"] for r in p["ctx"]["scores"]}),
            key=lambda m: m["config"])},
        deps=["scores"], outputs=["metrics"], side_effects=False,
        description="Aggregate per-config mean scores"))

    stream.append(_instruction_factory(
        "rank_configs", IsaType.CONTROL_FLOW,
        lambda p, s: {"ranking": [
            {"rank": i + 1, **m}
            for i, m in enumerate(sorted(
                p["ctx"]["metrics"], key=lambda m: (-m["mean"], m["config"])))
        ]},
        deps=["metrics"], outputs=["ranking"], side_effects=False,
        description="Rank configurations by mean score"))

    stream.append(_instruction_factory(
        "significance_guard", IsaType.CONTROL_FLOW,
        (lambda p, s: (
            None if len({r["mean"] for r in p["ctx"]["ranking"]}) > 1
            else _raise("significance guard failed")) or {}),
        deps=["ranking"], outputs=[], side_effects=False,
        description="Defensive significance check (dead code)"))

    stream.append(_report_writer(
        "write_leaderboard", ["ranking"], "leaderboard.md",
        deps=["ranking"], description="Write the experiment leaderboard"))
    return stream


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _raise(msg: str):
    raise ValueError(msg)


def _artifact_sha256(sandbox: LocalSandboxExecutor, filename: str) -> str:
    return hashlib.sha256(sandbox.read_file(filename)).hexdigest()


def _write_corpus(sandbox: LocalSandboxExecutor,
                  corpus: Dict[str, str]) -> None:
    """Write suite inputs at the sandbox root (no logs/ prefix)."""
    for name, content in sorted(corpus.items()):
        sandbox.write_file(name, content.encode("utf-8"))


def _cacheable_compute(instr: Instruction, key_input: Dict[str, Any],
                       compute_fn: Callable[[], Dict[str, Any]]):
    """Shared cacheable-step body: get -> hit? : compute+put.

    ``compute_fn`` must stay lazy: it is only invoked on a cache miss
    (eagerly evaluating the computation at the call site would charge
    the model on every realization even when the cache hits).
    """
    cache_key = instr._get_cache_key(key_input)
    cached = instr.cache.get(cache_key)
    if cached is not None:
        instr._last_hit = True
        return cached
    instr._last_hit = False
    computed = compute_fn()
    instr.cache.put(cache_key, computed)
    return computed


def _report_writer(id: str, keys: List[str], filename: str, *,
                   deps: List[str], description: str) -> Instruction:
    """Final side-effectful step: deterministic JSON artifact + hash."""
    def _write(p, s):
        payload = {k: p["ctx"][k] for k in keys}
        body = json.dumps(payload, sort_keys=True, indent=1)
        s.sandbox.write_file(filename, body.encode())
        return {"report_written": hashlib.sha256(body.encode()).hexdigest()}

    return _instruction_factory(
        id, IsaType.STATE_MANAGEMENT, _write,
        deps=deps, outputs=[], side_effects=True, description=description)


# ---------------------------------------------------------------------------
# Battery (mirrors random_tasks_stress.run_task, for hand-built streams)
# ---------------------------------------------------------------------------

DIRECTIONS: List[Dict[str, Any]] = [
    {
        "key": "data_pipeline", "title": "数据管道 / ETL 数据清洗",
        "angle": ("研究数据清洗流水线的声明式编排：去重、规范化、"
                  "质量度量的可缓存/可消除结构"),
        "artifact": "cleaned.json",
        "corpus": gen_etl_corpus, "builder": build_data_pipeline,
    },
    {
        "key": "rag_qa", "title": "RAG 检索增强问答",
        "angle": ("研究检索-重排-生成的多 LLM 步缓存链与双分支数据流"
                  "（语料路径 ∥ 查询路径）"),
        "artifact": "answer.md",
        "corpus": gen_rag_corpus, "builder": build_rag_qa,
    },
    {
        "key": "coordination", "title": "多智能体任务协调",
        "angle": ("研究扇出式多智能体计划合并：并行纯步骤、依赖汇聚"
                  "与死守卫在协调逻辑中的位置"),
        "artifact": "schedule.md",
        "corpus": gen_coordination_corpus, "builder": build_coordination,
    },
    {
        "key": "code_audit", "title": "代码静态安全审计",
        "angle": ("研究静态分析流水线的分支汇聚：模式检测与严重度"
                  "分类两条支路在汇总步合流"),
        "artifact": "audit.md",
        "corpus": gen_audit_corpus, "builder": build_code_audit,
    },
    {
        "key": "eval_pipeline", "title": "实验评测流水线",
        "angle": ("研究评测编排的纯系统任务：网格展开、种子化模拟、"
                  "排名的确定性可复现结构"),
        "artifact": "leaderboard.md",
        "corpus": gen_eval_corpus, "builder": build_eval_pipeline,
    },
]


async def run_suite(direction: Dict[str, Any], index: int,
                    args) -> Dict[str, Any]:
    key = direction["key"]
    args.artifact = direction["artifact"]
    corpus = direction["corpus"](random.Random(100 + index))
    backend = RuleClassifierBackend()
    stream = direction["builder"](backend, corpus)

    sandbox = LocalSandboxExecutor(name=key)
    try:
        _write_corpus(sandbox, corpus)
        inject_sandbox(stream, sandbox)

        manager = OptimizationManager()
        manager.dynamic_optimizer.hot_path_threshold = args.hot_threshold
        active = stream
        hashes: List[str] = []
        calls: List[int] = []
        per_repeat_len: List[int] = []
        for _ in range(args.repeats):
            per_repeat_len.append(len(active))
            before = backend.model_calls
            await run_stream(active)
            calls.append(backend.model_calls - before)
            hashes.append(_artifact_sha256(sandbox, args.artifact))
            metrics = OptimizationMetrics(
                execution_time=0.01, memory_usage=1.0,
                api_calls=calls[-1], success_rate=1.0)
            result = manager.record_execution(active, metrics)
            if (result is not active and result is not None
                    and len(result) < len(active)):
                active = result

        # full-stack determinism: fresh stream + cache + workspace
        fresh_backend = RuleClassifierBackend()
        fresh = direction["builder"](fresh_backend, corpus)
        fresh_sandbox = LocalSandboxExecutor(name=f"{key}-det")
        try:
            _write_corpus(fresh_sandbox, corpus)
            inject_sandbox(fresh, fresh_sandbox)
            await run_stream(fresh)
            fresh_hash = _artifact_sha256(fresh_sandbox, args.artifact)
        finally:
            fresh_sandbox.close()

        # ELL export + JSONL round-trip
        guards = [i.id for i in stream if i.id.endswith("_guard")]
        workflow, messages = await export_stream_to_ell(
            stream, name=key, description=direction["title"],
            execution_summary={"repeats": args.repeats},
            eliminated=guards)
        trace_path = Path(args.trace_dir) / f"{key}.jsonl"
        save_ell_trace(trace_path, workflow, messages)
        wf_rec, msg_recs = load_ell_trace(trace_path)
        ell_ok = (
            len(messages) == 1 + 2 * len(stream)
            and len(wf_rec["steps"]) == len(stream)
            and len(msg_recs) == len(messages)
        )
    finally:
        sandbox.close()

    guard_set = {i.id for i in stream if i.id.endswith("_guard")}
    compile_out = compile_check(stream, guards=guard_set)
    has_llm = any(i.type == IsaType.LLM_INTERACTION for i in stream)
    llm_steps = [i for i in stream if i.type == IsaType.LLM_INTERACTION]
    cache_ok = (
        not has_llm
        or calls[0] == 0
        or (calls[0] > 0 and all(c == 0 for c in calls[1:]))
    )
    monotone = all(
        b >= c for b, c in zip(per_repeat_len[1:], per_repeat_len[2:]))
    checks = {
        "static_compile_valid": compile_out["pass"],
        "cached_repeats": cache_ok and len(set(hashes)) == 1,
        "closed_loop_consistent": (
            len(set(hashes)) == 1 and monotone
            and per_repeat_len[-1] <= per_repeat_len[0]),
        "deterministic": fresh_hash == hashes[0],
        "ell_export": ell_ok,
    }
    types = {}
    for i in stream:
        types[str(i.type).split(".")[-1]] = types.get(
            str(i.type).split(".")[-1], 0) + 1
    return {
        "key": key, "title": direction["title"],
        "angle": direction["angle"],
        "checks": checks,
        "all_pass": all(checks.values()),
        "stream_length": len(stream),
        "type_counts": types,
        "llm_steps": len(llm_steps),
        "guards": len(guards),
        "aggressive_effective": compile_out["AGGRESSIVE"]["effective"],
        "eliminated": compile_out["AGGRESSIVE"]["eliminated"],
        "per_repeat_length": per_repeat_len,
        "per_repeat_calls": calls,
        "cache_hits": sum(getattr(i.metrics, "cache_hits", 0)
                          for i in llm_steps),
        "adopted": per_repeat_len[-1] < per_repeat_len[0],
        "report_sha256": hashes[0][:12],
    }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-run 5 thesis-direction task suites")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--hot-threshold", type=int, default=2)
    parser.add_argument("--output", type=str,
                        default="examples/thesis_direction_results.json")
    parser.add_argument("--trace-dir", type=str,
                        default="examples/thesis_direction_traces")
    args = parser.parse_args()

    Path(args.trace_dir).mkdir(parents=True, exist_ok=True)
    results = []
    for index, direction in enumerate(DIRECTIONS):
        results.append(await run_suite(direction, index, args))

    summary = {
        "suites": len(results),
        "all_pass": all(r["all_pass"] for r in results),
        "per_check": {
            name: sum(int(r["checks"][name]) for r in results)
            for name in results[0]["checks"]
        },
        "adoption_rate": sum(r["adopted"] for r in results) / len(results),
    }
    if args.output:
        Path(args.output).write_text(json.dumps(
            {"summary": summary, "suites": results}, indent=1))

    print("=" * 76)
    print("Five thesis-direction suites through the AgentISA battery")
    print("=" * 76)
    for r in results:
        flags = " ".join(
            f"{n}={'P' if v else 'F'}" for n, v in r["checks"].items())
        print(f"\n[{r['key']}] {r['title']}")
        print(f"  angle    : {r['angle']}")
        print(f"  stream   : {r['stream_length']} instructions "
              f"({r['type_counts']}), llm={r['llm_steps']}, "
              f"guards={r['guards']}")
        print(f"  compile  : NONE={r['stream_length']} -> "
              f"AGGRESSIVE={r['aggressive_effective']} "
              f"(eliminated: {r['eliminated'] or 'none'})")
        print(f"  repeats  : len {r['per_repeat_length']}, "
              f"calls {r['per_repeat_calls']}, "
              f"cache_hits {r['cache_hits']}, "
              f"adopted={r['adopted']}")
        print(f"  artifact : sha256 {r['report_sha256']}")
        print(f"  checks   : {flags}")
    print("=" * 76)
    print(f"per-check totals : {summary['per_check']}")
    print(f"adoption rate    : {summary['adoption_rate']:.0%}")
    verdict = "PASS" if summary["all_pass"] else "FAIL"
    print(f"VERDICT: {verdict}")
    if args.output:
        print(f"results written to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
