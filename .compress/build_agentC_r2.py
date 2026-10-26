# -*- coding: utf-8 -*-
"""Round-2 compression of .compress/agentC.md.

Protected blocks (table rows + ':' captions, the 5.6 code fence, the five
Research-question sentences, all headings) are extracted byte-exact FROM
agentC.md itself; only explanatory prose is rewritten.
"""
import io, re

PATH = "/Users/chenxingqiang/thinkbase-project/.compress/agentC.md"

with io.open(PATH, "r", encoding="utf-8") as f:
    src = f.read()
lines = src.split("\n")

# ---- extract protected blocks from the current agentC.md ----
def extract_tables(ls):
    """Return list of table blocks: consecutive '|' lines plus following ':' caption lines."""
    tables, i = [], 0
    while i < len(ls):
        if ls[i].startswith("|"):
            j = i
            while j < len(ls) and ls[j].startswith("|"):
                j += 1
            while j < len(ls) and ls[j].startswith(":"):
                j += 1
            tables.append("\n".join(ls[i:j]))
            i = j
        else:
            i += 1
    return tables

def extract_code(ls):
    blocks, i = [], 0
    while i < len(ls):
        if ls[i].strip().startswith("```"):
            j = i + 1
            while j < len(ls) and not ls[j].strip().startswith("```"):
                j += 1
            blocks.append("\n".join(ls[i:j+1]))
            i = j + 1
        else:
            i += 1
    return blocks

def extract_rq(ls):
    return [l for l in ls if l.startswith("**Research question.**")]

T = extract_tables(lines)      # 8 tables in document order
CODE = extract_code(lines)     # 1 code fence
RQ = extract_rq(lines)         # 5 RQ sentences
assert len(T) == 8, f"expected 8 tables, got {len(T)}"
assert len(CODE) == 1, f"expected 1 code block, got {len(CODE)}"
assert len(RQ) == 5, f"expected 5 RQs, got {len(RQ)}"
T1, T2, T3, T4, T5, T6, T7, T8 = T
CODE6 = CODE[0]
RQ2, RQ3, RQ4, RQ7, RQ71 = RQ

doc = f"""# 5. Empirical Evaluation

We evaluate AgentISA with a reproducible microbenchmark suite (`examples/benchmarks/isa_microbench.py`) and a multi-agent functional test suite, under the scope fixed at the end of Section 4.3: Sections 5.2–5.5 run against the mock backend; Sections 5.7–5.8 extend validation to a live LLM gateway and four system-level tasks; Section 5.9 discusses threats to validity.

## 5.1 Methodology

### 5.1.1 Instrumentation: Mock LLM Backend

All experiments route model inference through a `MockLLMBackend` that counts each `infer` call as one simulated LLM API round trip and charges a fixed 80 ms simulated latency per call to a simulated clock. It never sleeps and never contacts a live model, so every "API call" is observable, deterministic given a seed, and reproducible without network access or inference spend. Everything *around* the backend is real: the cache, compiler, and optimizer are the production `agentflow.core.isa` implementations (`InstructionCache`, `InstructionCompiler`, `OptimizationManager`), exercised through the same code paths a deployment uses.

### 5.1.2 Workloads

Workloads are synthetic and seed-controlled:

- **Caching**: 2,000 tasks from a discrete distribution over 50 instruction templates × 20 parameter values, popularity Zipf-distributed with exponent $s \\in \\{{0.5, 1.0, 1.5\\}}$ (larger $s$ concentrates popularity on fewer tasks, increasing reuse potential). Three arms share the same instrumentation (Section 5.2): no cache, the `InstructionCache`, and a naive app-level exact-response cache (a plain dict keyed on the request payload).
- **Static compilation**: random sequences of length 50, 200, and 1,000 from `synthetic_formal_sequence` (`examples/benchmarks/isa_microbench.py`; short enough to audit line by line). Each of the first $n-1$ slots is drawn independently: with probability 0.25 a *dead* instruction (side-effect-free, **no declared outputs, hence no consumers anywhere in the sequence**—removable by construction, planted not inferred); with probability 0.55 a *pure* instruction (side-effect-free, one declared output $v_i$, consumed **only by the final sink**, fusable next to a same-type pure neighbor); otherwise *side-effectful* (always live, never fused or removed). The $n$-th slot is a side-effectful *final sink* depending on exactly the set of all pure outputs, so every pure instruction has a downstream consumer and no pure output is deleted as dead. The attributes are disjoint and jointly exhaustive per slot; the passes find the planted DCE/fusion targets, so the measured reductions are properties of this workload family, not of arbitrary agent programs.
- **Dynamic optimization**: instruction traces of length 100 recorded for up to 200 iterations per seed, hot-path detection active (threshold 50 recordings); the loop terminates at the trigger—iteration 50 in all seeds—so each seed contributes exactly 51 recordings (the 50 counted plus the trigger).
- **Ablations**: two arms. The primary arm is *paired*: it re-compiles the static-compilation sequences (identical seed base 2,000) with fusion and dead-code-elimination toggled individually and jointly—a same-input decomposition of the 34.2% headline. The secondary arm is an *independent* draw, same distribution but seed base 4,000, a fresh-sequence replication (all-passes total 35.0%, within one standard deviation of the headline).

### 5.1.3 Protocol and Statistics

Each synthetic-workload configuration (Sections 5.2–5.5) is run with 5 random seeds; we report mean ± sample standard deviation. These are **descriptive statistics only**: we make no hypothesis-test claims (no p-values, no significance tests) anywhere in this paper. Five seeds characterize variability across the tested workload instances, not statistical significance or population-level generality. Section 5.2's naive response-cache baseline shares the mock-backend instrumentation; a formal *inferential* comparison against full external systems remains out of scope, and no such claim is made.

### 5.1.4 Metric Terminology

Several distinct quantities must not be conflated; each is defined once here:

- **Effective-instruction reduction**—fewer *effective* instructions in the compiled program (Sections 5.3/5.5); an instruction-stream property, not execution time.
- **Dispatch-count reduction**—fewer dispatch operations at runtime, including parallel grouping's folding (Section 5.3); a runtime count.
- **Trace shrinkage**—shorter traces after hot-path re-optimization (Section 5.4); the optimized trace's executed-state identity is verified separately there.
- **Cache hit rate / call reduction**—fraction of calls served from the result cache; here the call-reduction ratio equals the hit rate by construction (Section 5.2).
- **Simulated latency saving**—avoided-call counts × the mock's fixed 80 ms (Section 5.2); derived, not measured.
- **Measured latency/token saving**—wall-clock time and token counts against the real gateway (Sections 5.7–5.7.1); measured.

For context preservation (Theorem 3.3): *extraction-time context preservation* is the copy invariance of the extraction step; *runtime context validation* is the monitor's pre/post-dispatch constraint checking—different properties, different coverage (Section 3.2.2 discloses the runtime side's deletion blind spot).

## 5.2 Experiment 1: Result Caching

{RQ2}

We execute 2,000 tasks per seed, with caching disabled and with the `InstructionCache` enabled, measuring simulated API calls at the backend; the reduction ratio and hit rate coincide by construction (every avoided call is exactly one cache hit).

**Cache-key contract.** The shipped result cache keys an entry on the (instruction, params) pair—deliberately *not* on model identity/version, prompt-template version, tool schema, decoding parameters, or ambient context. Soundness requires those factors held fixed, which every benchmark here guarantees by construction (single model alias, fixed prompt templates, fixed decoding configuration, fresh cache per arm); Section 5.7.1's classifier-keyed step hashes exactly the input dictionary its prompt is built from, under the same discipline. A deployment changing any of these factors must invalidate or namespace the cache (the production cache isolates by TTL, LRU capacity, and never-cached failures but does not detect semantic-configuration drift); cache validity across *varying* model versions or prompts is future work, not exercised by any number in this paper.

{T1}

**Naive response-cache baseline.** A third arm runs an app-level *response* cache keyed on the exact request payload string—what an exact-match cache in front of the backend (GPTCache's exact mode, or an HTTP response cache) keys on—implemented as a plain dict with no AgentISA machinery. It issues 810.4 ± 5.7, 642.0 ± 3.8, and 408.0 ± 9.4 calls at $s = 0.5$, $1.0$, $1.5$: **parity with the `InstructionCache` to the digit**, because on this workload the (instruction, params) key is already canonical. Parity is the finding, stated plainly: at the level of call reduction the instruction-level cache adds nothing over exact-response caching; what it adds is the compile-time transformations of Sections 5.3 and 5.5—fusion and dead-code elimination, reducing the program before any request is issued, which a response cache cannot express because it sees request strings, not instruction semantics. Semantic/embedding caching (GPTCache's approximate mode [46]) remains out of scope: no head-to-head against approximate matching is claimed.

**Interpretation.** The reduction grows monotonically with skew, from 59.5% at $s = 0.5$ to 79.6% at $s = 1.5$, matching the cache-aware expectation: the savings' upper bound is the probability mass of repeated tasks, which the Zipf exponent directly controls. In simulated latency terms the savings are the avoided-call counts × 80 ms ($s = 0.5/1.0/1.5$: 95,168 ± 455 / 108,640 ± 305 / 127,360 ± 750 ms)—derived, not separately measured.

Two caveats keep this honest. First, these are *simulated* calls against a mock backend: the reduction is exact for this workload family, but real deployments face cache-invalidation, prompt-context, and embedding-variability effects the mock abstracts away. Second, the absolute number depends on reuse structure; workloads without repetition would see zero benefit. The experiment establishes the *mechanism and its skew sensitivity*, not a universal constant.

## 5.3 Experiment 2: Static Compilation

{RQ3}

We compile random sequences at four optimization levels and measure two quantities, kept explicit because they are easy to conflate: **effective instruction count**—instructions remaining after fusion and dead-code elimination, a program-size measure—and **dispatch count**—independent dispatch operations the runtime issues, where parallel grouping can fold several independent instructions into one dispatch *without changing the effective instruction count*.

{T2}

**Interpretation.** Three observations:

1. **Effective instruction count reduction is level-stable**: BASIC, ADVANCED, and AGGRESSIVE produce identical reductions (34.2% ± 1.8% at length 200). The fusion and DCE passes at BASIC already capture the available gains on these synthetic sequences; higher levels add *scheduling* improvements (dispatch folding), not elimination—and ADVANCED and AGGRESSIVE coincide here, their extra passes finding no further targets, differing only on streams carrying combinable patterns. We report this as-is rather than claim monotonic gains.

2. **Dispatch count drops further with parallel grouping**: at length 200, dispatch falls from 131.6 (BASIC) to 74.0 (ADVANCED) with effective-instruction reduction unchanged—adjacent side-effect-free instructions with disjoint dependencies and outputs grouped into shared dispatches. The 74.0 figure is a scheduling reduction, not a further program-size reduction. (An earlier implementation folded more aggressively—44.4 at length 200—but could place side-effectful instructions into shared groups whose non-interference was never established; the sounder grouping here folds less, and Appendix C.5's dispatch-folding corollary now matches the implementation exactly.)

3. **Gains are stable across sequence lengths**: 32.8%, 34.2%, 35.4% at lengths 50, 200, 1,000 respectively, tightest variance at length 1,000 (±1.6%); length does not materially change the opportunity structure.

Effective-instruction reduction is again a *program-size* metric: it translates to simulated API-call savings only where eliminated or fused instructions carry model inference; the caching experiment measures the complementary reuse effect.

## 5.4 Experiment 3: Dynamic Hot-Path Optimization

{RQ4}

We record an instruction trace of length 100 for up to 200 iterations per seed with hot-path detection active (Section 5.1.2). Once a path is flagged hot, the runtime forms an optimized trace; **this experiment measures that trace's length and re-executes it**—the replay verification below checks that it reproduces the original trace's final state exactly. Whether a hot-path-optimized stream preserves a task-level artifact is established separately, on realistic streams, by Section 5.8 (byte-identical output).

{T3}

Across seeds, hot-path optimization shrinks the trace by **65.8% ± 4.6%**; the seed-to-seed spread (59.0%–70.0%) reflects which sub-sequences each synthetic trace makes hot, the effect present in all five seeds. As with the static experiments, this is trace-length reduction on synthetic traces—workload-controlled evidence that the detection-formation-reoptimization loop functions, not a wall-clock speedup claim.

**Replay verification.** Each seed's optimized trace *is* re-executed—every instruction reads its declared dependencies from a threaded state dict and returns deterministic bindings for its declared outputs (side-effectful instructions leave per-id markers)—and its final state is compared with the original's, executed the same way. The states are identical in all five seeds (5/5; recorded per seed as `replay_state_identical` in the results file), so the 65.8% shrinkage carries an executed-equality check at trace level—on the deterministic mock, not against a live model; this is the same style of check as Section 5.8's task-level byte-identity, applied to the synthetic traces. Live-model behavior remains untested—no mock experiment can address it.

## 5.5 Ablation: Per-pass Contribution

To attribute the static gains we run the two ablations of Section 5.1.2. The primary one is **paired**: it compiles the *same* sequences as Section 5.3 (identical seed base 2,000 and generator) with fusion and DCE toggled individually and jointly, so per-configuration differences are same-input paired differences—an exact decomposition of the headline, not a replication:

{T4}

The fusion+DCE row is 34.2% ± 1.8%—the headline itself, confirming that these two passes account for the entire static gain at length 200. The **paired interaction**, computed per seed as (fusion+DCE − NONE) − (fusion − NONE) − (DCE − NONE)—34.2 − 21.6 − 24.5 = −11.9 points—is −11.9% ± 2.0% (per-seed −14.5/−9.5/−13.0/−12.0/−10.5 points): the passes overlap—dead slots break fusion chains and fused pairs absorb neighbors DCE would otherwise have removed individually—so the sum of the single-pass gains (46.1 points) overstates the joint effect by 11.9 points.

The secondary ablation is the earlier **independent-draw replication** (seed base 4,000, same distribution): fusion alone 21.9% ± 3.5%, DCE alone 24.1% ± 3.3%, fusion+DCE and all passes 35.0% ± 3.7%. It reproduces the paired structure on fresh sequences (single-pass shares agree within one standard deviation), so the decomposition is not an artifact of one draw. "All passes" equals fusion+DCE on these workloads, consistent with Section 5.3's level stability—no additional pass contributes measurable reduction, and we report that absence rather than dress the higher levels in unearned credit. Because dead instructions are planted at rate 0.25 by construction, DCE's per-pass share is near-tautological for this family: it measures that the pass finds the planted dead code, precisely its function.

## 5.6 Multi-Agent Coordination: Functional Validation

The multi-agent layer—`AgentCoordinator`, `SecurityMonitor`, and the collaborative instruction set—is validated functionally by 43 tests, all passing. The 43 are the coordination-and-security subset; the repository's full suite—861 passing tests, 14 skipped at run time—also covers the ISA core, compiler, and optimizer units of Sections 5.2–5.5. This is **qualitative functional validation**—the tests verify the coordination behaviors exist and are correct, not that they are fast; no performance number should be inferred here.

The coordination flow implemented by `AgentCoordinator`:

{CODE6}

What the 43 tests verify, grouped by subsystem:

1. **Agent registration and role policies** (registration creates policies; permissions and security requirements differ by role; trust levels are bounded and reward declared specializations).
2. **Task coordination** (successful coordination produces a solution; failed tasks are recorded; missing authentication or authorization raises; no suitable agents raises; selection respects a capability threshold; the coordinator is the highest-scoring agent).
3. **Execution planning** (plans have stage structure with coordination rules; every stage has fallback keys; fallback strategies are differentiated; a fallback executes on stage failure; validators are reasoning-heavy non-coordinator agents).
4. **SecurityMonitor** (valid tasks pass; missing authentication or authorization fails; violations are recorded with timestamps; declared data sensitivity is enforced; sensitive categories infer a high sensitivity level; benign categories pass at medium; resource access is open by default, restrictable by configuration).
5. **Collaborative instructions** (barriers wait for the declared participant count and time out on a missing peer; width-one barriers resolve immediately; resource-allocation instructions reject over-allocation and serialize concurrent requests; the workflow state machine accepts valid actions, rejects invalid ones and double starts; a research-team scenario runs end to end).

These tests exercise the same production modules used elsewhere and are fully deterministic (no LLM calls). They establish correctness of the coordination machinery under the tested conditions; what they do not establish is scalability under contention, crash recovery, or adversarial inputs beyond the policy checks above—all future work (Section 6).

## 5.7 Real-LLM End-to-End Validation of Result Caching

{RQ7}

**Setup.** We replace the mock backend with a real model served by a local Anthropic-protocol gateway (TLS with an internal CA; alias `claude-glm-5.3-flash`, a hosted reasoning-capable GLM model; `max_tokens=64`), using sequential HTTP calls (httpx, no proxy, 90 s timeout) with three bounded retries—none occurred. Every other component is the production code path: the same `InstructionCache` protocol, instruction objects, and workload generator, with the same instruction pool (50 templates) and parameter space (20 values) as Section 5.2.

**Scale.** Because each real round trip costs ~2.2 s, we reduce scale from 2,000 tasks × 5 seeds to **200 tasks × 1 seed** per Zipf level (s in {{0.5, 1.0, 1.5}}); the whole run is 1,040 measured round trips (plus one preflight validating connectivity before timing) in 39.4 minutes. The seed pins the task sequence exactly, so the workload is the same one the expectation below is computed from.

**An exactness property.** A cache hit depends only on the (instruction, params) pairs seen so far—not on what the model returns. The production `InstructionCache` is LRU with a TTL, so in general a hit also depends on wall-clock time and capacity; under this benchmark none of that can bind: `max_size` = 100,000 (at most 200 distinct keys per arm), `ttl` = 3,600 s (every cache-enabled arm completes in under seven minutes; the no-cache arms run to 7.9 minutes and never consult a cache), a fresh cache per arm, failed calls never cached. Hit/miss decisions are therefore a pure function of the (instruction, params) sequence, *computable without any backend*; if the mechanism is correct the real run must reproduce that sequence exactly—and at the recorded granularity (per-arm hit rates and call counts) it does:

{T5}

Per-call latency was stable across arms (p50 2.13–2.21 s, p95 3.06–3.54 s). Token savings are gateway-reported `usage` differences: every avoided call saves its input tokens plus the tokens the reasoning model generates before responding (output tokens were the binding constraint of `max_tokens=64` in every call).

**Interpretation.** First, the mechanism transfers unchanged: observed hit rates match the computed expectation exactly at every skew level, so Section 5.2's synthetic results measure the cache, not a mock artifact. Second, avoided calls convert into measured time and tokens—14.1%, 26.6%, and 45.8% of the no-cache arm's wall clock at s = 0.5, 1.0, 1.5—tracking but not identical to the call reduction, since arms ran sequentially and per-call latency drifted slightly (the exact quantity is the call reduction; wall and token figures are measurements, not derivations). Third, the *absolute* hit rates here (8.5–45.5%) are lower than Section 5.2's (59.5–79.6%) purely because repetition accumulates with task count: at N = 200 these are the exact computed values for this workload family, and at N = 2,000 the same computation yields the Section 5.2 numbers—a property of the workload, not the backend.

**Scope of the claim.** This experiment validates the caching mechanism end-to-end—one gateway, one model, one seed, N = 200, sequential requests, no streaming—recording per-arm aggregate hit rates and call counts; per-position identity follows from the deterministic cache decisions above but is not separately logged, and nothing here is a throughput, availability, or dollar-cost claim. Sections 5.3–5.5 are structural, never dispatching to the backend; Section 5.7.1 extends the real-gateway validation to them via a cache-mediated protocol, their validity otherwise resting on the code-path and ablation arguments there.

### 5.7.1 Real-Gateway Validation of Compilation and Hot-Path Optimization

{RQ71}

**Setup.** A dedicated harness (`examples/paper_real_llm_compiler_test.py`; tests in `tests/system/test_real_llm_compiler_test.py`) reuses Section 5.7's gateway configuration (same base URL, internal-CA TLS, alias `claude-glm-5.3-flash`, sequential httpx, 90 s timeout, three bounded retries) but raises `max_tokens` to 512 (Section 5.7 used 64) so completions dominate the priced comparison; replies still bind at the cap, as below. It runs identically online against the gateway or offline against a deterministic same-protocol mock that counts calls, accounts tokens from template sizes, and charges the constant 80 ms simulated latency. The task is Section 5.8's service-log triage shape: a seven-instruction stream with one cacheable LLM-type step (`classify_signature`, one model call per distinct incident-signature batch) and the same planted dead guard (`check_format`), run over repeated realizations from a seeded truncated-Zipf distribution over a fixed universe of 40 task instances.

**Arms and protocol.** Arm (a) is the unoptimized authored stream; it runs first, cold, performing every live round trip. Arm (b) drives the *same* stream through the production machinery—`OptimizationManager` / `optimize_hot_path` records each realization and adopts the shrunken trace once the path goes hot; the static `InstructionCompiler` compiles at NONE vs AGGRESSIVE with the dataflow-preservation assertion—then executes the same draws against the cache arm (a) warmed. The classify step's cache key is the sha256 of exactly the input dictionary its prompt is built from, so arm (b) issuing *zero* new gateway calls proves it computed identical keys—hence identical prompts and consumed values—and its artifacts must come out byte-for-byte equal; any perturbation of the LLM step's inputs would surface as a cache miss (a live call) and/or artifact divergence.

**Scale.** Tiny scale: 50 task realizations × 1 seed × Zipf s = 1.0. The workload contains 22 distinct signature batches, so the entire run is 23 gateway round trips (one preflight plus 22 live calls, all issued by arm (a)) in 4.4 minutes of wall time—a bounded validation run, not the 200 × 3-skew sweep of Section 5.7.

**Results.** All checks pass. Arm (a) made exactly the 22 first-occurrence calls the backend-free computation predicts and served the other 28 realizations from the result cache (hit rate 56.0%), consuming 4,027 prompt and 11,264 gateway-reported completion tokens in 260.2 s of measured call latency. Arm (b)—compiled, hot-path-adopted—made zero new gateway calls (all 50 realizations hit the warmed keys) and reproduced every artifact byte-for-byte. The hot path was adopted at realization 7 (threshold 5; adoption lands two realizations past the threshold because the counter must strictly exceed it and recording follows execution, as in Section 5.8): realizations 1–6 executed the full seven-instruction stream, realizations 7–50 the shrunken six-instruction form with `check_format` absorbed into its successor by fusion—306 instruction executions, against 350 in arm (a), a 12.6% reduction in executed instructions attributable to the adopted shrunken trace (the per-stream shrink is 14.3%, 7→6; the realized saving is diluted by the six full-form realizations before adoption; the static compiler alone reproduces the 7→6 shrink at AGGRESSIVE, so the two mechanisms' contributions are not separated), reported separately from the cache's call savings. Static compilation of the same stream: NONE 7 → AGGRESSIVE 6 effective instructions, dataflow preserved.

**What this does and does not show.** The byte-identity property is cache-mediated by design: the cached classification values entered the artifacts once, through arm (a)—values that on this alias are fallback-derived, as explained below—and arm (b) is proven to compute the same keys and consume the same values. This validates the stream transformation, not model determinism, which is not guaranteed at temperature 0 and is not claimed. On this alias the reasoning model's replies bind at `max_tokens` in every call—as Section 5.7 reported at `max_tokens=64`, and as a diagnostic call at 1,500 tokens still did—so the classify step's severity values come from its deterministic fallback; what is validated against the live model is the mechanism chain (key computation, cache decisions, call and token accounting, artifact identity), not classification quality.

**A priced comparison (costs MODELED—computed from a pricing overlay rather than observed—on measured tokens).** The harness prices the run's measured token counts under three modes: *no optimization* (every realization re-queries at full price), *AgentISA result caching* (first occurrences only—the measured behavior of both arms), and a *provider prompt cache* (MODELED: every realization queries, but repeat prompts bill input tokens as prompt-cache reads at 0.1x the input price—configurable—mirroring Anthropic's published prompt-caching read discount; completions always bill at full price). Two simplifications—omitting Anthropic's 1.25x cache-write premium and 1024-token minimum cacheable prefix—both favor the provider-cache baseline, so the comparison is conservative for us. List prices are parameterized stand-ins ($3/M input, $15/M output by default) because the gateway's actual pricing is unknown. Token counts are measurements, the pricing overlay is a model, and no dollar-savings claim is made:

{T6}

On this run's token mix the result cache cuts modeled cost by 56.0%—it removes whole calls, completions included—while the modeled provider prompt cache cuts only 3.3%, because the reasoning model's completion tokens dominate and a discount on input reads cannot touch them; the ranking of the two caching strategies is a function of the input/output token mix, not a constant. On the deterministic mock at paper scale (200 tasks × 1 seed × 3 Zipf levels, template-accounted tokens, same modes and prices) the result cache cuts modeled cost by 80.1%, 81.6%, and 85.1% at s = 0.5, 1.0, 1.5, while the modeled provider prompt cache cuts 28.9–31.4%—there the short completions do not dominate, so the input-side discount matters more; both columns remain MODELED-on-measured in both cases. GPTCache, an external cache layer, is explicitly out of scope; no head-to-head against it is claimed.

**What the real-gateway experiments do and do not establish.** Three layers, kept separate: (1) *gateway integration correctness*—requests reach the gateway, the cache governs calls, and call/token statistics are recorded faithfully: **established** by Sections 5.7 and 5.7.1. (2) *Workflow semantic preservation*—the optimized stream computes the same cache keys and consumes the same cached values as the unoptimized one, yielding byte-identical artifacts: **established as a cache-mediated identity** (see above), validating the stream transformation, not model determinism. (3) *Real-model task quality*: **not established here**—the reasoning alias returned results at the token ceiling and the classify step fell back to a deterministic rule, so no statement about live-model output quality is made anywhere in this paper; the abstract, highlights, and conclusion state only layers (1) and (2). Both real-gateway runs share the coverage limits above (one gateway, one model, one seed, sequential, no streaming); multi-model, multi-seed, and real-deployment behavior remain untested.

## 5.8 Controlled Four-Task System-Level Characterization

The experiments above measure individual mechanisms on synthetic sequences; this section is a *controlled four-task system-level characterization*—a functional demonstration under a fixed closed-loop protocol (same task, 12 realizations, one cacheable step each), not a generalization experiment; the identical 91.7% hit-rate row in the runtime table below is a property of that protocol, not four independent cache observations. The seed-controlled characterization (`examples/system_task_workload.py`, `tests/system/test_system_task_workload.py`; the original single-task demonstration lives on as `examples/system_task_demo.py`) is built from **four tasks with genuinely different dependency shapes**, each realized end-to-end against sandbox-resident files, making the mechanisms' composition—and Section 3.4.4's closed-loop optimization claims—checkable:

- the original **service-log triage**: a seven-instruction linear stream, detailed below;
- **csv-etl-anomaly**: an eight-instruction fan-out pipeline (ingest CSV shards, schema guard, parse, per-column statistics, per-row signatures and anomaly triage, aggregate, write);
- **report-merge**: a seven-instruction wide-fan-in task (ingest five markdown reports, header guard, dedupe, per-line signatures, cacheable consistency check, merge, write);
- **regression-diff**: an eight-instruction two-branch convergence task (parallel baseline/candidate log ingests, guard, normalize, diff, classify hunks, aggregate, write).

**The triage task's instruction stream** (seven instructions over sandbox-resident files): `ingest_logs` reads six 400-line log files from the sandbox workspace (side-effectful); `check_syntax` is a defensive format guard that never rejects on this workload (a *planted dead guard*, standing in for the belt-and-braces checks real pipelines carry); `normalize_entries` parses and normalizes lines; `extract_incidents` lifts normalized entries into incident records; `classify_signature` triage-classifies each incident (LLM-type, cacheable, backed by a deterministic rule stand-in counting simulated model calls); `aggregate_triage` summarizes by severity class; `write_report` writes the triage report to the sandbox (side-effectful).

**Configuration.** Seed 42; 6 files × 400 lines; 12 repeated realizations of the task; hot-path threshold 5. Every phase is deterministic and re-runnable from the repository (Artifact Availability).

**What the triage walk-through shows.**

- **Static compilation with structural validation.** NONE compiles to 7 effective instructions / 7 dispatches; AGGRESSIVE to 6 / 6 (14.3% reduction): the dead guard fuses with the following pure step rather than being dropped outright. The walk-through *asserts the compiled stream's dataflow validity*—every step whose output is consumed downstream must survive compilation, the only eliminable step being the guard—so the shrinkage cannot silently come from dropping live producers.
- **Reuse across repeated realizations.** The first realization makes 50 (simulated) model calls; all eleven subsequent realizations make zero (result cache hit rate 91.7% on the classification step); all 12 reports are byte-identical.
- **Closed-loop hot-path adoption.** The stream crosses the hot-path threshold at the sixth recording (the counter must strictly exceed it and recording follows execution, so adoption lands at realization threshold+2); the runtime *adopts* the optimized form automatically, repeats 7–12 execute the shrunken 6-instruction stream (instruction executions per realization: `[7,7,7,7,7,7,6,6,6,6,6,6]`), and on a fresh workspace the optimized form reproduces the report byte-for-byte—semantics-preserving on the real task, not only on synthetic sequences.
- **Full-stack determinism.** A completely fresh stack—new instructions, new cache, new workspace, same seed—reproduces the report byte-for-byte (SHA-256 equality).

**Calibrating the synthetic headline.** Section 5.4 reports 65.8% trace shrinkage on synthetic traces with planted dead instructions; the four realistic tasks show **12.5–25.0%** (one planted guard in a seven- or eight-instruction stream, plus one additional live-pure fusion pair in csv-etl-anomaly). We state both figures rather than privilege either: 12.5–25.0% reflects the planted dead-check density (one guard per seven- to eight-instruction stream)—a stated density assumption, not observed ordinary-pipeline waste—while the synthetic figure is an upper bound under the skew/planting parameters we control, not a typical-case estimate. Constructing a realistic (non-synthetic) dependency shape also served the audit purpose, exposing two grouping/fusion soundness gaps—side-effectful instructions opening dispatch groups, fused instructions losing their external dependencies—now fixed with regression tests; Section 5.3's dispatch-folding numbers reflect the corrected (more conservative) grouping.

**The four-task characterization.** All four tasks run the identical closed-loop battery (`examples/system_task_workload.py`, `characterize()`, 12 realizations each, hot-path threshold 5, one planted dead guard and one cacheable LLM-type step each, the latter backed by a deterministic rule stand-in counting simulated model calls):

*Structure (compile-time):*

{T7}

*Execution (runtime, 12 realizations each):*

{T8}

Every task passes the same structural assertions: the guard is never dropped outright (the eliminated set is empty in all four tasks), the reduction being fusion—in csv-etl-anomaly a second adjacent pure pair (per-column statistics with row-signature construction) also fuses, hence its 25.0% shrink with a single planted guard. The first realization pays all model calls and every subsequent realization hits the result cache—the 91.7% hit rate is 11 of 12 realizations by construction (one cacheable step, identical seeded draws), so the four rows share it structurally. Reuse and closed-loop adoption are thus invariant across the four dependency shapes under the fixed protocol (12 seeded realizations, threshold 5, one cacheable step per task), while the guard/fusion shrink varies with stream structure (12.5–25.0%)—the compile shrink is the only shape-dependent quantity, which is what one wants from an infrastructure claim; the characterization's value over the single-task demonstration is evidence that the composition behavior is not an artifact of one stream's structure.

## 5.9 Limitations and Threats to Validity

We enumerate the limitations of the evaluation explicitly:

1. **Mock backend, not a live LLM.** The mock charges a constant 80 ms per simulated call and returns deterministic responses; beyond counting, it covers Section 5.4's replay verification (final-state identity, 5/5 seeds) and Section 5.2's naive-cache baseline (same instrumentation). The real-gateway experiments cover the cache reproducing the computed hit/miss statistics and converting avoided calls into measured time and tokens (Section 5.7), and the compiled, hot-path-adopted stream issuing zero new gateway calls with byte-identical artifacts (Section 5.7.1)—a cache-mediated identity validating the stream transformation, not model determinism. Still not covered: multi-model and multi-seed behavior, real deployment, caching under model nondeterminism *across prompt variations*, streaming responses, provider rate limits, and live-model task quality (the four-task classification steps are deterministic rule-based stand-ins, and the 5.7.1 gateway run fell back to the deterministic classifier).

2. **Synthetic workloads.** Zipf-distributed synthetic tasks are a reasonable model of skewed agent workloads, but a model; Section 5.8's task-level characterization only partially addresses this. The cache-reduction figures (59.5–79.6%) are exact for these workload families and should be read as *skew-sensitivity curves*, not universal savings; uniformly novel tasks would see near-zero benefit. Section 5.8's four-shape characterization calibrates the 65.8% synthetic headline downward with its 12.5–25.0% per-task compile shrink and shows uniform reuse and closed-loop adoption under the fixed protocol (12 seeded realizations, threshold 5)—but the four tasks remain seed-controlled, hand-constructed streams, not a production workload study.

3. **No production end-to-end evaluation.** We have not deployed AgentISA in a production setting; consequently this paper contains no latency-under-load, throughput, availability, or dollar-cost measurements, and we make no such claims.

4. **No real cost data (token usage measured; pricing MODELED on measured tokens).** All latency figures in Sections 5.2–5.5 are simulated (avoided calls × 80 ms); Section 5.7 measures real wall-clock time and gateway-reported tokens at reduced scale, and Section 5.7.1 prices its own measured tokens (the 22-call reference run) under three modes—no optimization, AgentISA result caching, and a modeled provider prompt cache with a parameterized cache-read discount (default 0.1, after Anthropic's published prompt-caching economics). The costs are MODELED-on-measured: tokens are measurements, prices are configurable list-price stand-ins (the gateway's actual pricing is unknown), and no dollar-savings claim is made. Section 5.2's baseline is an *exact-match* response cache (parity reported); head-to-head against *approximate* (semantic/embedding) caching (GPTCache's approximate mode—explicitly out of scope) and a priced deployment study under real pricing remain future work.

5. **Program-size metrics, not wall-clock time.** Effective-instruction and dispatch-count reduction are structural metrics: they correlate with—but do not equal—wall-clock improvements, which depend on per-instruction cost, available parallelism, and dispatch overhead.

6. **Descriptive statistics only.** Synthetic configurations use five seeds (Section 5.1.3); the real-gateway and system-level runs are single-seed by design. No hypothesis tests, confidence intervals, or significance claims are made; standard deviations indicate stability across seeds, nothing more.

7. **Functional, not performance, validation of coordination.** The 43 multi-agent tests verify behavioral correctness under deterministic conditions, not throughput or scalability.

Despite these limitations, the evaluation achieves its purpose: it demonstrates, under fully disclosed and reproducible conditions, that each optimization mechanism is implemented, active, and produces the claimed structural effects on workloads designed to expose them.
"""

with io.open(PATH, "w", encoding="utf-8") as f:
    f.write(doc)
print("written", PATH)
print("total words:", len(doc.split()), "chars:", len(doc))
