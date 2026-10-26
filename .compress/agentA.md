# 1. Introduction

Recent advances in Large Language Models (LLMs) have produced systems with strong natural language understanding, generation, and—through instruction tuning and human feedback—instruction-following capabilities [1, 2, 3, 4, 5, 6], and a rapidly growing body of work couples them with tools, memory, and control loops to build autonomous agents [9, 10, 11, 12]. Yet the engineering substrate beneath these agents remains largely ad hoc: behaviors are specified as prompts and provider-specific function-calling schemas—difficult to specify precisely, analyze formally, or optimize systematically.

## Background and Motivation

Function calling—selecting an API from a declared tool set and emitting structured arguments—has become the dominant interface between LLMs and external systems [13, 14, 15], consolidated by standardization efforts such as the Model Context Protocol [18] and measured by harnesses such as the Berkeley Function Calling Leaderboard [17]. However, the unit of interaction remains the individual API call: each call is dispatched, priced, and forgotten independently, even when an agent repeatedly executes structurally identical work.

This per-call statelessness contrasts with how computing systems historically handle repetitive execution: classical compilers and hardware translate recurring patterns into cheaper forms—instruction fusion, dead-code elimination, caching, and trace-based dynamic optimization [33, 34, 45]. Our motivating hypothesis is that these techniques apply to LLM agent workloads: if agent behavior is expressed as an explicit instruction stream rather than opaque prompts, recurring patterns become visible to a compiler and optimizable with well-understood, semantics-preserving transformations.

## Challenges in Current LLM-based Agent Systems

Five structural challenges motivate AgentISA:

1. **Per-call statelessness of tool interfaces.** Mainstream function-calling and tool-use protocols treat each call as an isolated event [13, 14, 18]: the provider has no notion of a *program* being executed, repeated identical work incurs repeated cost, and caching, where present, is a bolt-on afterthought rather than a semantic property of the execution model.

2. **Experience that does not accumulate as compiler-visible instruction structure.** Paradigms such as Reflexion [10] and Voyager [12] accumulate experience as verbal reflections or inference-time skill libraries—not reified as reusable, statically optimizable instructions that a compiler can fuse, schedule, or eliminate.

3. **Absence of a shared behavioral contract.** Each multi-agent framework defines its own abstractions for roles, messages, and workflows [19, 20, 21, 22, 23, 24]; no instruction-level contract spans frameworks, fragmenting tooling and cross-framework analysis.

4. **Cost pressure at the wrong layer.** FrugalGPT [28] shows LLM inference cost is a first-order concern, and serving systems attack it below the API boundary with paged attention, continuous batching, KV-cache-centric architectures, and attention kernels [38, 39, 40, 41]; what is missing is the complementary, above-the-API layer that reduces the *number and size* of calls an agent issues in the first place.

5. **Integration complexity across providers.** Portability is handled at the library level [26, 27]: logic written against one provider's function-calling surface does not transfer cleanly to another, and capability differences are hard to reason about systematically [17].

## Research Objectives and Contributions

This paper introduces AgentISA, an instruction set architecture that addresses these challenges by making agent behavior a first-class compilable artifact. Our contributions are:

1. **A formal instruction set** for LLM agents with compositional semantics, a type system, and pre/post-conditions on every instruction (Section 3.1). Two deliberately modest sanity results are stated as design rationale: Turing completeness (Theorem 3.1)—negative certification that the instruction set does not constrain expressiveness—and type safety for the pure core (Theorem 3.2), which fixes the well-formedness obligation the offline bounded checker discharges for pure instructions (Section 3.3.3) and the runtime brackets with pre-dispatch context validation.

2. **Agentic Calling**, a mechanism lifting provider-specific function calling into reusable, cacheable, semantically specified agent instructions, with a model-adapter layer for cross-provider portability (Section 3.2).

3. **Trace-based instruction learning**, which mines recurring instruction patterns from interaction traces by frequency-based sequence mining and deterministic behavior grouping, and promotes validated recurring patterns into candidate instructions whose `execute` bodies replay the mined constituent subsequence by sequential composition over a shared context (Section 3.3; replay reproduces the recorded subsequence's outputs on the same inputs and does not generalize beyond them, as the section states).

4. **A compiler and runtime** implementing fusion, dead-code elimination, parallel-grouped scheduling, and dynamic hot-path specialization for agent instruction streams (Sections 3.4 and 4), together with specified tree-pattern instruction selection and cycle-accurate pipeline scheduling (Sections 4.1.1 and 4.1.2), with semantic-preservation arguments for the implemented rewrites (Theorems 3.4–3.6, proofs in Appendix C).

5. **An empirical evaluation** of the implemented system, with full methodology disclosure. On synthetic Zipf-distributed workloads (5 random seeds each), caching reduces simulated API calls by 59.5–79.6% depending on skew, static compilation reduces the effective instruction count by 34.2%±1.8% (sequence length 200), and dynamic optimization shrinks repeatedly executed traces by 65.8%±4.6%. Two real-gateway experiments (Sections 5.7 and 5.7.1) validate the cache and the compiled pipeline against a real LLM gateway, and a four-task system-level characterization (Section 5.8) composes the mechanisms on realistic (non-synthetic) instruction streams.

6. **Multi-agent coordination infrastructure** with capability-based agent selection, execution planning, fallback strategies, and a security monitor enforcing authentication, authorization, resource-access, and data-sensitivity policies (Sections 3.5 and 5.6), validated by 43 passing functional tests.

Our empirical claims are scoped accordingly: Sections 5.2–5.5 exercise the real `agentflow.core.isa` implementation against a mock LLM backend on synthetic workloads; Sections 5.7–5.8 extend the validation at reduced scale on realistic streams; none are production end-to-end measurements (threats to validity in Section 5.9).

Section 2 reviews related work; Sections 3 and 4 present the design and its implementation; Section 5 the empirical evaluation; Sections 6 and 7 discussion and conclusions.

# 2. Related Work

AgentISA draws on eight lines of work, reviewed in turn.

## 2.1 Foundation LLMs and Function Calling

Modern LLMs are autoregressive transformers trained at large scale [1, 2, 3, 4], with instruction tuning and reinforcement learning from human feedback making them reliable instruction followers [5, 6]. Their text distribution is conventionally factorized as:

$$
P(y|x) = \prod_{t=1}^{T} P(y_t|x, y_{<t}; \theta)
$$

where $x$ is the input context, $y$ the generated output, and $\theta$ the model parameters. Function calling constrains this distribution to a structured output space:

$$
F(x) = \text{argmax}_{f \in \mathcal{F}} P(f|x, \mathcal{F})
$$

where $\mathcal{F}$ is the set of declared functions and $F(x)$ the selected call. GPT-4 [3], PaLM [2], and open models such as LLaMA [4] expose this capability with provider-specific schemas; AgentISA does not modify this layer but treats any such interface as a *backend* targeted by a common instruction set, as a compiler targets multiple processor backends [33].

The cost structure of this layer motivates our work. If each call carries latency $\tau_i$ plus a context-switching overhead $\omega_i$ (prompt reconstruction, serialization, cache misses; the symbol $\delta$ is reused in Appendix B for per-instruction transition functions—the two are unrelated), then $n$ similar calls cost:

$$
C(n) = \sum_{i=1}^{n} (\tau_i + \omega_i)
$$

**Definition 2.1** (Function Call Overhead): The overhead of executing $n$ function calls is $C(n)$, the sum of per-call latencies and per-call context reconstruction costs. When repeated calls carry identical semantic inputs and satisfy the cache-validity conditions of Section 3.2, the repeated context-construction cost $\omega_i$ is avoidable—which is precisely the cost AgentISA's caching and fusion are designed to remove. Structural similarity alone does not make a re-invocation waste: if an input's dependencies have changed or the call probes external state, re-execution has semantic meaning, and the cache-validity conditions—not structural identity—are what license reuse.

## 2.2 Reasoning and Agent Paradigms

A second line enhances reasoning at prompt time: chain-of-thought prompting [7] and zero-shot reasoning [8] elicit intermediate steps; ReAct interleaves reasoning traces with actions [9]; Reflexion adds verbal self-critique across episodes [10]; Tree of Thoughts explores deliberative search [11]; Voyager demonstrates open-ended skill acquisition [12]. These paradigms define *what* an agent should do at the cognitive level; AgentISA defines *how* the resulting behaviors are represented, compiled, and executed as instructions. A ReAct loop, for instance, becomes an AgentISA instruction sequence whose tool invocations are fused, cacheable agentic calls.

## 2.3 Tool Calling and Interface Standardization

Toolformer trains models to interleave tool calls with text [13]; Gorilla connects models to large API corpora [14]; ToolLLM scales tool coverage to 16,000+ real-world APIs [15]; HuggingGPT orchestrates model hubs for task solving [16]. On the standardization side, the Model Context Protocol defines an open specification for exposing tools and context [18], and the Berkeley Function Calling Leaderboard provides systematic cross-model evaluation of function-calling accuracy [17]. These efforts standardize the *interface*; AgentISA standardizes the *execution model above it*: an AgentISA program compiled for two providers differs only in the final translation step (Section 3.2.3), and interface-level standards alone cannot fuse or eliminate calls.

## 2.4 Multi-Agent Frameworks

AutoGen frames agent applications as multi-agent conversations [19]; MetaGPT assigns standardized roles and workflows [20]; AgentVerse facilitates multi-agent collaboration and studies emergent behaviors [21]; ChatDev organizes agents as a software-development organization [22]; CAMEL studies communicative agent societies [23]; AutoGPT popularized autonomous goal-driven agents [24]. AgentISA contributes a different artifact to this space: not a framework or orchestration philosophy but an instruction-level substrate—capability profiles, coordination instructions with barrier semantics, and a security monitor—that any such framework could target: capability-based agent selection, execution planning with per-stage fallback strategies, and policy enforcement for authentication, authorization, resource access, and data sensitivity (Section 3.5, Section 5.6).

## 2.5 Prompt Compilation and Cost Optimization

Closest to our compilation perspective is DSPy, which compiles declarative prompt programs into optimized pipelines via prompt search and fine-tuning [25]; LangChain [26] and LlamaIndex [27] provide library-level chaining and indexing abstractions; FrugalGPT reduces cost through model cascades [28]. AgentISA differs in the level at which optimization operates. DSPy optimizes *prompts*; FrugalGPT optimizes *model selection*; AgentISA optimizes *instruction streams*: fusion, dead-code elimination, parallel grouping (the shipped form of pipeline scheduling), result caching, and dynamic hot-path specialization—implemented and measured (Sections 3.4, 4, and 5)—plus tree-pattern instruction selection over pattern covers and cycle-accurate pipeline scheduling, specified but not yet implemented (Sections 4.1.1 and 4.1.2). These classical compiler transformations [33] compose with—rather than replace—prompt-level and model-level optimization.

## 2.6 Benchmarks for Agent Systems

AgentBench evaluates LLMs as agents across diverse environments [29]; GAIA tests general assistant abilities [30]; PlanBench probes planning and reasoning about change [31]; SWE-bench measures resolution of real-world GitHub issues [32]. These benchmarks evaluate *end-to-end capability*; AgentISA's evaluation instead measures *subsystem behavior under controlled conditions*—API-call reduction under synthetic Zipf workloads, instruction-count reduction from compilation, and trace shrinkage from dynamic optimization (Section 5). The two are complementary: instruction-level metrics can attribute end-to-end gains to specific mechanisms when such gains are measured, and are reproducible without live model access.

## 2.7 Classical Compiler and Architecture Techniques

AgentISA's optimization machinery is deliberately classical: tree-pattern instruction selection with dynamic programming, peephole optimization, and data-flow-based dead-code elimination are standard [33]; pipelining, hazard detection, and resource-constrained scheduling follow the classical architecture literature [34]; the RISC-V project demonstrates the value of open, well-specified instruction sets [35]; type safety via progress-and-preservation is the standard technique of programming-language theory [36]; CompCert establishes that compiler correctness itself can be formally verified [37]. Our contribution is not a new compiler technique but the demonstration that this machinery transfers to LLM agent workloads, together with the semantic-preservation arguments adapted to instructions with pre/post-conditions (Appendix C).

## 2.8 LLM Serving Systems

Below the API boundary, the serving community has made inference dramatically cheaper: PagedAttention manages KV memory like virtual memory [38]; Orca introduces continuous batching for transformer serving [39]; FlashAttention redesigns attention for IO-awareness [40]; Mooncake proposes a KVCache-centric disaggregated architecture [41]; energy analyses quantify the inference cost of large models [42]. This line and AgentISA are orthogonal layers of the same stack: serving systems make each call cheaper; AgentISA reduces how many calls an agent makes and how much redundant work each carries. The two compound multiplicatively.

## 2.9 Research Gaps and Our Contributions

In summary, the literature leaves four gaps that AgentISA addresses:

1. **No instruction-level contract for agent behavior.** Interfaces are standardized [13, 14, 18] and frameworks numerous [19, 20, 21, 22, 23, 24], but no shared instruction set spans providers and frameworks.

2. **No compiler for agent behavior.** Prompt-level compilation exists [25] and model-level cost reduction exists [28], and *caching itself is well-trodden ground*—at the response level (GPTCache [46]), at the KV-prefix level (RadixAttention/SGLang [47]), and at the provider prompt-caching level. What none of these layers expose is an *instruction-level* contract: caching there is an optimization of model serving—invisible to the agent program's structure, and without effect on it. Instruction-level fusion, elimination, and scheduling—with cacheability as a declared property of the instruction set—have not, to our knowledge, been demonstrated.

3. **No semantics-preserving account of agent optimization.** Serving work verifies kernels and schedulers, not agent-level transformations; we provide progress-and-preservation-style arguments for ours (Theorems 3.2, 3.4, 3.5, 3.6).

4. **No reproducible instruction-level evaluation.** Existing benchmarks measure end-to-end capability [29, 30, 31, 32]; we contribute a seed-controlled evaluation: a microbenchmark suite with fully disclosed synthetic workloads and mock-backend instrumentation, extended by real-gateway and system-level validations (Section 5).

AgentISA addresses these four gaps directly—the instruction set and Agentic Calling abstraction (Sections 3.1–3.2), instruction-level optimization with semantic-preservation arguments (Section 3.4, Theorems 3.2 and 3.4–3.6, Appendices B–C), trace-derived instruction learning (Section 3.3), and a seed-controlled evaluation (Sections 5.2–5.8); the full contribution list is in Section 1 and is not repeated here.
