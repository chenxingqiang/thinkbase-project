"""End-to-end AgentISA test over the actual paper, with a real model.

This is the rag_qa thesis suite turned up to real inputs and a real
backend: the corpus is the AgentISA paper itself (``docs/AgentISA.tex``)
and the two LLM-typed steps make REAL gateway round trips
(claude-glm-5.3-flash through the local Anthropic-compatible gateway)
instead of using the counting stand-in.  Retrieval stays a
deterministic local step so the pipeline's non-LLM subgraph is exactly
reproducible; the model is asked for temperature 0.

Battery (real-LLM semantics, stated honestly):

* static_compile_valid      -- NONE keeps the stream; AGGRESSIVE may
                               only drop the dead guard.
* cached_repeats            -- the FIRST realization makes real calls;
                               repeats 2+ make ZERO gateway calls and
                               the artifact is byte-identical.
* closed_loop_consistent    -- hot-path adoption only steps the stream
                               down, artifact unchanged.
* retrieval_deterministic   -- a fresh stack (new stream/cache/sandbox)
                               reproduces the chunks and retrieved
                               ids byte-exactly.
* answer_observation        -- fresh-stack answers are compared with
                               the cached run's answers and REPORTED
                               (real models are not guaranteed
                               byte-deterministic; nothing is claimed).
* ell_export                -- the executed stream serializes to a
                               JSONL trace with expected counts.

Usage:
    ./venv/bin/python examples/paper_real_llm_test.py \
        --model claude-glm-5.3-flash \
        --output examples/paper_real_llm_results.json
"""

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "benchmarks"))

from agentflow.core.isa.ell_export import (
    export_stream_to_ell, load_ell_trace, save_ell_trace,
)
from agentflow.core.isa.instruction import Instruction, InstructionType as IsaType
from agentflow.core.isa.instruction_optimization import (
    OptimizationManager, OptimizationMetrics,
)
from agentflow.sandbox.local_executor import LocalSandboxExecutor

from isa_real_llm_bench import RealLLMBackend
from random_tasks_stress import compile_check
from system_task_demo import inject_sandbox, run_stream

PAPER_PATH = Path(__file__).resolve().parents[1] / "docs" / "AgentISA.tex"

QUERIES = [
    "What does the paper's result-caching mechanism cache, and what hit "
    "rates does the evaluation report under different Zipf skews?",
    "How does the closed-loop trace optimizer decide to adopt an "
    "optimized instruction stream, and what safeguards it?",
    "What does the paper claim about dead-code elimination in the "
    "static compiler, and how is dataflow preservation guaranteed?",
]

# ---------------------------------------------------------------------------
# Deterministic local retrieval subgraph
# ---------------------------------------------------------------------------

STOPWORDS = set("the a an of and or to in on for with is are be as by it "
                "its this that what how does do and".split())


def chunk_paper(text: str, target_chars: int = 1400) -> List[str]:
    """Split the paper's LaTeX source into paragraph-ish chunks."""
    # strip comment lines and excessive whitespace
    lines = [l for l in text.splitlines() if not l.lstrip().startswith("%")]
    body = "\n".join(lines)
    paragraphs = [re.sub(r"\s+", " ", p).strip()
                  for p in re.split(r"\n\s*\n", body)]
    paragraphs = [p for p in paragraphs if len(p) > 80]
    chunks: List[str] = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 1 > target_chars and current:
            chunks.append(current)
            current = p
        else:
            current = f"{current} {p}".strip()
    if current:
        chunks.append(current)
    return chunks


def retrieve(query: str, chunks: List[str], k: int = 3) -> List[int]:
    """Deterministic keyword-overlap retrieval: query -> top-k chunk ids."""
    terms = {t for t in re.findall(r"[a-z]{3,}", query.lower())}
    terms -= STOPWORDS
    scored = []
    for idx, chunk in enumerate(chunks):
        blob = set(re.findall(r"[a-z]{3,}", chunk.lower()))
        overlap = len(terms & blob)
        # density tie-break, then stable by index
        scored.append((-overlap, -len(terms & blob) / max(1, len(blob)), idx))
    scored.sort()
    return [idx for _, _, idx in scored[:k]]


# ---------------------------------------------------------------------------
# Async-capable instruction factory (real gateway calls are awaitable)
# ---------------------------------------------------------------------------


def _ainstruction_factory(
        id: str, itype: IsaType, execute: Callable, *,
        deps: List[str], outputs: List[str], side_effects: bool,
        description: str = "",
        cache_config: Optional[Dict] = None) -> Instruction:
    """Like random_tasks_stress._instruction_factory, but the step body
    may be a coroutine (real gateway round trips)."""

    class _Step(Instruction):
        async def execute(self, input_params: Dict[str, Any]) -> Any:
            start = time.time()
            result = execute(input_params, self)
            if asyncio.iscoroutine(result):
                result = await result
            self._update_metrics(start, True,
                                 cache_hit=getattr(self, "_last_hit", False))
            return result

    step = _Step(
        id=id, name=id, type=itype, params={},
        description=description or id,
        side_effects=side_effects, outputs=outputs,
        cache_config=cache_config,
    )
    step.dependencies = list(deps)
    step.sandbox = None
    step._last_hit = False
    return step


def _cacheable(instr: Instruction, key_input: Dict[str, Any],
               compute_fn: Callable[[], Awaitable[Dict[str, Any]]]):
    """Lazy async cacheable-step body (mirrors the suites' helper)."""
    cache_key = instr._get_cache_key(key_input)

    async def _run():
        cached = instr.cache.get(cache_key)
        if cached is not None:
            instr._last_hit = True
            return cached
        instr._last_hit = False
        computed = await compute_fn()
        instr.cache.put(cache_key, computed)
        return computed

    return _run()


# ---------------------------------------------------------------------------
# The paper-RAG instruction stream
# ---------------------------------------------------------------------------


def build_paper_rag_stream(backend: RealLLMBackend,
                           queries: List[str]) -> List[Instruction]:
    stream: List[Instruction] = []

    stream.append(_ainstruction_factory(
        "ingest_paper", IsaType.STATE_MANAGEMENT,
        lambda p, s: {"paper_text":
                      s.sandbox.read_file("AgentISA.tex").decode()},
        deps=[], outputs=["paper_text"], side_effects=True,
        description="Read the AgentISA paper source"))

    stream.append(_ainstruction_factory(
        "chunk_paper", IsaType.CONTROL_FLOW,
        lambda p, s: {"chunks": chunk_paper(p["ctx"]["paper_text"])},
        deps=["paper_text"], outputs=["chunks"], side_effects=False,
        description="Split the paper into paragraph chunks"))

    stream.append(_ainstruction_factory(
        "retrieve_chunks", IsaType.CONTROL_FLOW,
        lambda p, s: {"retrieved": {
            str(qi): retrieve(q, p["ctx"]["chunks"])
            for qi, q in enumerate(queries)}},
        deps=["chunks"], outputs=["retrieved"], side_effects=False,
        description="Deterministic top-k retrieval per query"))

    def _classify(p, instr):
        chunks = p["ctx"]["chunks"]
        retrieved = p["ctx"]["retrieved"]

        async def _compute():
            kept: Dict[str, List[int]] = {}
            for qi, query in enumerate(queries):
                ids = retrieved[str(qi)]
                excerpts = "\n".join(
                    f"[{i}] {chunks[i][:300]}" for i in ids)
                prompt = (
                    "Question about a paper:\n"
                    f"{query}\n\nCandidate excerpts:\n{excerpts}\n\n"
                    "Which excerpt indices are relevant? Reply with ONLY "
                    "a JSON array of integers, e.g. [0,2]."
                )
                raw = await backend.infer(prompt)
                nums = re.findall(r"\d+", raw)
                keep = [i for i in ids if str(i) in nums] or list(ids)
                kept[str(qi)] = keep
            return {"kept": kept}

        return _cacheable(
            instr, {"queries": len(queries),
                    "retrieved": {k: v for k, v in retrieved.items()}},
            _compute)

    stream.append(_ainstruction_factory(
        "classify_relevance", IsaType.LLM_INTERACTION, _classify,
        deps=["retrieved", "chunks"], outputs=["kept"], side_effects=False,
        description="Real LLM step: filter relevant excerpts (cacheable)",
        cache_config={"max_size": 100, "ttl": 3600}))

    def _synthesize(p, instr):
        chunks = p["ctx"]["chunks"]
        kept = p["ctx"]["kept"]

        async def _compute():
            answers: Dict[str, str] = {}
            for qi, query in enumerate(queries):
                ids = kept[str(qi)]
                excerpts = "\n\n".join(
                    f"[{i}] {chunks[i][:900]}" for i in ids)
                prompt = (
                    "Answer the question using ONLY the excerpts below. "
                    "Cite the excerpt numbers you used like [0]. Keep the "
                    "answer under 120 words.\n\n"
                    f"Question: {query}\n\nExcerpts:\n{excerpts}"
                )
                answers[str(qi)] = await backend.infer(prompt)
            return {"answers": answers}

        return _cacheable(
            instr, {"kept": {k: v for k, v in kept.items()}}, _compute)

    stream.append(_ainstruction_factory(
        "synthesize_answer", IsaType.LLM_INTERACTION, _synthesize,
        deps=["kept", "chunks"], outputs=["answers"], side_effects=False,
        description="Real LLM step: synthesize cited answers (cacheable)",
        cache_config={"max_size": 100, "ttl": 3600}))

    stream.append(_ainstruction_factory(
        "citation_guard", IsaType.CONTROL_FLOW,
        (lambda p, s: (
            None if all("[" in a or len(a) > 40
                        for a in p["ctx"]["answers"].values())
            else _raise("citation guard failed")) or {}),
        deps=["answers"], outputs=[], side_effects=False,
        description="Defensive citation check (dead code)"))

    def _write(p, s):
        payload = {
            "queries": queries,
            "kept": p["ctx"]["kept"],
            "answers": p["ctx"]["answers"],
            "retrieved": p["ctx"]["retrieved"],
        }
        body = json.dumps(payload, sort_keys=True, indent=1)
        s.sandbox.write_file("answers.json", body.encode())
        return {"report_written": hashlib.sha256(body.encode()).hexdigest()}

    stream.append(_ainstruction_factory(
        "write_answers", IsaType.STATE_MANAGEMENT, _write,
        deps=["answers"], outputs=[], side_effects=True,
        description="Write the cited answers artifact"))
    return stream


def _raise(msg: str):
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------


async def run_battery(backend: RealLLMBackend, args,
                      paper_text: Optional[str] = None) -> Dict[str, Any]:
    paper = paper_text if paper_text is not None else PAPER_PATH.read_text()
    stream = build_paper_rag_stream(backend, QUERIES)

    sandbox = LocalSandboxExecutor(name="paper-rag")
    try:
        sandbox.write_file("AgentISA.tex", paper.encode())
        inject_sandbox(stream, sandbox)

        manager = OptimizationManager()
        manager.dynamic_optimizer.hot_path_threshold = args.hot_threshold
        active = stream
        hashes: List[str] = []
        calls: List[int] = []
        per_repeat_len: List[int] = []
        for _ in range(args.repeats):
            per_repeat_len.append(len(active))
            before = backend.api_calls
            await run_stream(active)
            calls.append(backend.api_calls - before)
            artifact = sandbox.read_file("answers.json")
            hashes.append(hashlib.sha256(artifact).hexdigest())
            metrics = OptimizationMetrics(
                execution_time=0.01, memory_usage=1.0,
                api_calls=calls[-1], success_rate=1.0)
            result = manager.record_execution(active, metrics)
            if (result is not active and result is not None
                    and len(result) < len(active)):
                active = result

        # fresh stack: new stream + cache + workspace (same paper)
        fresh = build_paper_rag_stream(backend, QUERIES)
        fresh_sandbox = LocalSandboxExecutor(name="paper-rag-fresh")
        try:
            fresh_sandbox.write_file("AgentISA.tex", paper.encode())
            inject_sandbox(fresh, fresh_sandbox)
            before = backend.api_calls
            await run_stream(fresh)
            fresh_calls = backend.api_calls - before
            fresh_artifact = json.loads(
                fresh_sandbox.read_file("answers.json").decode())
        finally:
            fresh_sandbox.close()

        # ELL export + JSONL round-trip
        guards = {i.id for i in stream if i.id.endswith("_guard")}
        workflow, messages = await export_stream_to_ell(
            stream, name="paper-real-llm",
            description="AgentISA RAG over docs/AgentISA.tex with a real model",
            execution_summary={"repeats": args.repeats,
                               "real_api_calls": calls[0]},
            eliminated=sorted(guards))
        trace_path = Path(args.trace)
        save_ell_trace(trace_path, workflow, messages)
        wf_rec, msg_recs = load_ell_trace(trace_path)
        ell_ok = (
            len(messages) == 1 + 2 * len(stream)
            and len(wf_rec["steps"]) == len(stream)
            and len(msg_recs) == len(messages)
        )
        artifact_json = json.loads(sandbox.read_file("answers.json").decode())
    finally:
        sandbox.close()

    compile_out = compile_check(stream, guards=guards)
    # the deterministic subgraph only: chunking + retrieval.  ``kept``
    # and ``answers`` pass through real model calls and belong to the
    # empirical observation, not to this gate.
    retrieval_identical = (
        fresh_artifact["retrieved"] == artifact_json["retrieved"]
    )
    kept_identical = fresh_artifact["kept"] == artifact_json["kept"]
    answers_identical = (
        fresh_artifact["answers"] == artifact_json["answers"]
    )
    monotone = all(b >= c for b, c
                   in zip(per_repeat_len[1:], per_repeat_len[2:]))
    checks = {
        "static_compile_valid": compile_out["pass"],
        "cached_repeats": (
            calls[0] > 0 and all(c == 0 for c in calls[1:])
            and len(set(hashes)) == 1
        ),
        "closed_loop_consistent": (
            len(set(hashes)) == 1 and monotone
            and per_repeat_len[-1] <= per_repeat_len[0]
        ),
        "retrieval_deterministic": retrieval_identical,
        "ell_export": ell_ok,
    }
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "answer_observation": {
            "fresh_stack_calls": fresh_calls,
            "fresh_kept_identical": kept_identical,
            "fresh_answers_identical": answers_identical,
            "note": ("real-model answers compared empirically; "
                     "temperature 0 was requested but byte-determinism "
                     "is not guaranteed and not claimed"),
        },
        "stream_length": len(stream),
        "aggressive_effective": compile_out["AGGRESSIVE"]["effective"],
        "eliminated": compile_out["AGGRESSIVE"]["eliminated"],
        "per_repeat_length": per_repeat_len,
        "per_repeat_calls": calls,
        "real_usage": {
            "api_calls": backend.api_calls,
            "input_tokens": backend.input_tokens,
            "output_tokens": backend.output_tokens,
            "wall_ms": round(backend.wall_ms, 1),
        },
        "answers": artifact_json["answers"],
        "kept": artifact_json["kept"],
        "report_sha256": hashes[0][:12],
    }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-paper, real-model AgentISA end-to-end test")
    parser.add_argument("--model", type=str, default="claude-glm-5.3-flash")
    parser.add_argument("--base-url", type=str,
                        default="https://127.0.0.1:8443")
    parser.add_argument("--ca-cert", type=str,
                        default="/opt/homebrew/etc/nginx/ssl/witmem-gw.crt")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--hot-threshold", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--output", type=str,
                        default="examples/paper_real_llm_results.json")
    parser.add_argument("--trace", type=str,
                        default="examples/paper_real_llm_trace.jsonl")
    args = parser.parse_args()

    import os
    settings = json.load(open(Path.home() / ".claude" / "settings.json"))
    api_key = settings.get("env", {}).get("ANTHROPIC_AUTH_TOKEN") \
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if not api_key:
        raise SystemExit("no API key: set ANTHROPIC_AUTH_TOKEN")

    backend = RealLLMBackend(
        base_url=args.base_url, api_key=api_key, ca_cert=args.ca_cert,
        model=args.model, max_tokens=args.max_tokens,
        temperature=0.0, max_payload_chars=12000,
    )
    try:
        result = await run_battery(backend, args)
    finally:
        await backend.close()

    Path(args.output).write_text(json.dumps(result, indent=1))

    print("=" * 74)
    print(f"AgentISA RAG over the real paper, real model "
          f"({args.model}), {args.repeats} repeats")
    print("=" * 74)
    for qi, query in enumerate(QUERIES):
        print(f"\nQ{qi + 1}: {query}")
        print(f"  kept excerpts : {result['kept'][str(qi)]}")
        answer = result["answers"][str(qi)].replace("\n", " ")
        print(f"  A: {answer[:600]}")
    print("\n" + "=" * 74)
    print(f"real usage     : {result['real_usage']}")
    print(f"repeats        : len {result['per_repeat_length']}, "
          f"calls {result['per_repeat_calls']}")
    print(f"compile        : NONE={result['stream_length']} -> "
          f"AGGRESSIVE={result['aggressive_effective']} "
          f"(eliminated {result['eliminated'] or 'none'})")
    obs = result["answer_observation"]
    print(f"answer obs     : fresh-stack kept-identical="
          f"{obs['fresh_kept_identical']}, answers-identical="
          f"{obs['fresh_answers_identical']} ({obs['note']})")
    for name, ok in result["checks"].items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    verdict = "PASS" if result["all_pass"] else "FAIL"
    print("=" * 74)
    print(f"VERDICT: {verdict}   (results: {args.output})")


if __name__ == "__main__":
    asyncio.run(main())
