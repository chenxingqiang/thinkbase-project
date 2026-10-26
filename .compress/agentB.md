# 3. AgentISA Design

Building on Sections 1 and 2, we present the AgentISA design: the core architecture (Section 3.1), the Agentic Calling framework (Section 3.2), trace-based instruction learning (Section 3.3), instruction organization and optimization (Section 3.4), and the capability-assessment layer (Section 3.5).

## 3.1 Core Architecture

### 3.1.1 Instruction Set Specification

The AgentISA instruction set is formally defined as a tuple:

**Definition 3.1** (AgentISA Core):
$$
\text{AgentISA} = (I, T, \Sigma, \Delta, \Phi)
$$

where:

- $I$ is the set of instructions
- $T$ is the type system
- $\Sigma$ is the state space
- $\Delta$ is the transition function set
- $\Phi$ is the verification function set

Following the conventions of classical instruction set design [33, 34, 35], we define the basic instruction format:

```
INSTRUCTION = {
    opcode: OPCODE,
    args: ARGS[],
    precondition: PRED,
    postcondition: PRED,
    context: CONTEXT
}
```

The instruction set is categorized into four fundamental classes:

1. **Control Flow Instructions**:
   $$\mathcal{C} = \{seq, branch, loop, call\}$$

2. **State Management Instructions**:
   $$\mathcal{S} = \{load, store, update, cache\}$$

3. **LLM Interaction Instructions**:
   $$\mathcal{L} = \{query, response, context\}$$

4. **Resource Management Instructions**:
   $$\mathcal{R} = \{allocate, free, monitor\}$$

Two sanity results anchor the instruction-set definition; neither introduces new machinery (the simulation technique from the compiler literature [33], the type-safety technique from the semantics literature [36]), and we state them for the roles they play: Theorem 3.1 certifies that the instruction set does not artificially constrain what agent programs can express, and Theorem 3.2 fixes the well-formedness obligation that the offline bounded checker discharges for pure instructions (Section 3.3.3) and that the runtime brackets with pre-dispatch context validation (Section 3.2.3).

**Theorem 3.1** (Turing Completeness): The AgentISA instruction set is Turing complete.

*Proof sketch*: We exhibit a translation from WHILE programs—already known to be Turing equivalent—to AgentISA instruction sequences, using a simulation argument over instruction classes: state management instructions encode an unbounded store, control-flow instructions encode sequential composition, branching, and iteration. The full construction is in Appendix A, following standard techniques [33, 36].

### 3.1.2 Type System and Semantics

The type system T extends the simply-typed discipline of Pierce [36] with dependent-type constructions ($\Pi$, $\Sigma$, $\mu$):

**Definition 3.2** (AgentISA Type System):
$$
T ::= \text{Base} \mid \Pi(x:T_1).T_2 \mid \Sigma(x:T_1).T_2 \mid \mu X.T
$$

where:

- Base types include {Bool, Int, String, Float}
- Π represents dependent function types
- Σ represents dependent pair types
- μ represents recursive types

The semantic rules follow operational semantics:

$$
\frac{e_1 \rightarrow e_1'}{E[e_1] \rightarrow E[e_1']} \text{ (Context)} \quad
\frac{}{(\lambda x.e)v \rightarrow e[x:=v]} \text{ (β-reduction)}
$$

The pure core carries a standard type-safety argument—progress and preservation, in the style of Pierce [36]—discharging the well-formedness role fixed in Section 3.1 and extended to the shipped instruction set by the bounded checker of Section 3.3.3:

**Theorem 3.2** (Type Safety for the Pure Core): Well-typed programs in the pure core calculus of Appendix B—the $\lambda$-fragment (base types, dependent $\Pi$/$\Sigma$ types, recursion) plus the composition operators applied to instructions carrying typed pre/post-conditions—cannot "go wrong", under the per-instruction validity obligation of Definition 3.5 (stated in Section 3.3.3 below; pre/post-conditions), with transition functions assumed total on precondition-satisfying states (Appendix B.7). Effectful instruction classes are embedded via their typed pre/post-conditions and covered to the extent their preconditions are checkable.

*Proof sketch*: By the standard two-lemma technique—progress and preservation—as presented in Pierce [36]:

1. Progress: $\forall e, T.\ \vdash e : T \implies \mathrm{value}(e) \lor \exists e'.\ e \to e'$
2. Preservation: $\forall e, e', T.\ \vdash e : T \land e \to e' \implies \vdash e' : T$

Both lemmas are proved by induction over the derivations of the merged judgment system of Appendix B, with a joint argument for the composition cases. For instruction steps, progress follows from the Definition 3.5 validity obligation plus totality, preservation from the postcondition typing; composition operators are handled by the chained invariant $post(i_1) \Rightarrow pre(i_2)$ (Appendix B).

### 3.1.3 LLM Integration Interface

The LLM integration interface is a protocol playing the same role for agent backends that the Model Context Protocol plays for tool exposure [18]; its components correspond to the interface conventions of modern function-calling providers [3, 18]—$\mathcal{P}_{LLM} = (Q, R, C, \psi)$ with $Q \leftrightarrow$ QueryFormat (query space, JSON Schema), $R \leftrightarrow$ ResponseFormat (response space, JSON Schema), $C$ the context-management functions (whose implementation surface today exposes the context window size, ContextWindow: Integer), and $\psi \leftrightarrow$ StateMapping: Map<AgentState, LLMContext>:

```
Protocol LLMInterface {
    QueryFormat: JSON Schema
    ResponseFormat: JSON Schema
    ContextWindow: Integer
    StateMapping: Map<AgentState, LLMContext>
}
```

## 3.2 Agentic Calling Framework

### 3.2.1 Function Calling to Agentic Calling Transformation

The transformation from function calling to Agentic Calling is formalized as:

**Definition 3.3** (Agentic Transformation):
$$
\mathcal{T}: F \rightarrow A
$$

where $F$ is the function call space and $A$ is the agent instruction space.

$\mathcal{T}$ is required to preserve call semantics; this holds by construction for the three-step process below—the synthesized instruction records the function signature verbatim, embeds the context verbatim (Theorem 3.3), and delegates execution to the original call—so we state preservation as a property of the construction rather than a separately proved theorem:

$$
\forall f \in F: \text{sem}(\mathcal{T}(f)) \equiv \text{sem}(f)
$$

The transformation process follows three steps, mirroring the signature-to-call conventions of function-calling interfaces [13, 14]:

1. Function signature analysis
2. Context embedding
3. Instruction synthesis

### 3.2.2 Context Preservation and State Management

State management in AgentISA follows a formal state machine model:

$$
M = (S, s_0, \delta, F)
$$

where:

- S: Set of states
- s₀: Initial state
- δ: State transition function
- F: Set of final states

Extraction-time context preservation is guaranteed by construction (Theorem 3.3); at runtime it is only monitored:

**Theorem 3.3** (Context Preservation): Let $\phi$ be the context interpretation function (defined compositionally over context *contents*—the keys and values the context carries), and $t$ the extraction transformation that embeds a context $c$ verbatim into an instruction's `context` field. Then, for every $c \in C$:

$$
\phi(t(c)) = \phi(c)
$$

*Proof*: By the definition of $t$, the extracted instruction carries a verbatim copy of the source context; since $\phi$ is defined over context contents and the copy is content-identical, $\phi(t(c)) = \phi(c)$ holds by construction—a *design invariant of the extraction pipeline* (the implementation copies rather than re-encodes). At runtime the shipped checks are constraint-based and bracket the transition: `AgenticTransformer.validate_context` checks the registered context constraints pre-dispatch, and `verify_context_preservation(before, after)` re-checks them pairwise across the executed transition and reports violations (mechanics in Section 3.2.3); neither performs structural validation of the carried context. Extraction-time preservation is discharged by the copy construction; preservation across an executed transition is *monitored, not guaranteed*, by the runtime checks. ∎

### 3.2.3 Cross-model Compatibility

Cross-model *protocol* compatibility—not behavioral equivalence of models—is achieved through an abstract interface layer (provider-independent instruction representation with provider-specific adapters):

```
Interface ModelAdapter {
    translate: AgentInstruction → ModelSpecificCall      // shipped: translate_instruction
    normalize: ModelSpecificResponse → AgentResponse    // shipped: normalize_response
    validateContext: Context → Boolean                  // shipped: AgenticTransformer.validate_context
}
```

All three interface operations ship. `translate` is `ModelAdapter.translate_instruction` (with registered type transformations). `normalize` is `ModelAdapter.normalize_response`, which folds the two dominant wire shapes—the Anthropic protocol (typed `content` blocks with `usage.input_tokens`/`output_tokens`) and the OpenAI protocol (`choices[0].message` with `usage.prompt_tokens`/`completion_tokens`)—into a canonical `AgentResponse` envelope (content, tool calls, usage, model, raw), so downstream instruction code never branches on provider syntax. `validateContext` is `AgenticTransformer.validate_context`, the pre-execution complement of the shipped post-hoc `verify_context_preservation(before, after)`: it evaluates the registered context constraints against a single state before dispatch, applying two-argument pairwise predicates reflexively $(v, v)$ (single-argument predicates apply directly) and leaving constraints over absent keys vacuously satisfied—so key *deletion* goes unreported, a blind spot analogous to the non-reflexive case: constraints that are non-reflexive by design (e.g. strict growth) are not detectable pre-execution.

Motivated by cross-model function-calling studies [17] and open protocol efforts [18]—providers differ in surface syntax but can agree on *protocol shape*—the requirement is a **protocol-level contract on the adapter layer**. For $\mathbb{M}$ the set of provider adapters, each $m \in \mathbb{M}$ provides a request normalizer $\text{norm}_{in}(m)$ and a response normalizer $\text{norm}_{out}(m)$ such that:

$$
\begin{aligned}
\forall m \in \mathbb{M}: & \text{norm}_{in}(m), \text{norm}_{out}(m) \\
& \text{ are total on the supported request and response classes,}
\end{aligned}
$$

mapping every supported raw request into the unified instruction-side schema and every supported raw response into the unified result schema, with declared types preserved and errors classified into a common taxonomy. This is deliberately weaker than semantic-output equivalence: nothing implies that two models given the same normalized input return the same content ($m_1 = m_2$ would already demand determinism, which no LLM provides), and cross-model *behavioral* comparison is an empirical question at benchmark level (Section 3.5, [17, 29]).

## 3.3 Trace-Based Instruction Learning

### 3.3.1 Dialogue Pattern Recognition

Pattern recognition in AgentISA is frequency-based and deterministic (action-signature grouping) rather than grammatical; it is implemented in `agentflow.core.isa.pattern_mining` and `agentflow.core.isa.instruction_learning`. To prevent over-reading: the subsystem is a **discovery-and-replay pipeline, not an autonomous learning loop**. Its five stages are (i) trace-derived pattern mining (this section), (ii) dialogue-derived template extraction (§3.3.2—templates carry no recorded instruction body and raise `NotImplementedError` if executed), (iii) candidate validation (support and success-rate thresholds), (iv) *developer-controlled* promotion (mining never edits the optimizer's rule set; a human decides which candidates become instructions), and (v) composite replay (a promoted instruction re-runs its recorded subsequence on the same inputs; it does not generalize to unseen parameter values). Stages (i)–(iii) and (v) ship; stage (iv) is deliberately manual.

**Definition 3.4** (Instruction Sequence Pattern): A pattern $p$ over the instruction alphabet $I$ is a non-empty contiguous sequence $p = (i_1, \dots, i_k) \in I^k$ together with two frequency statistics:

- **Support**:
  $$
  \begin{aligned}
  sup(p) = P(& p \text{ occurs as a contiguous} \\
  & \text{subsequence in a randomly drawn trace})
  \end{aligned}
  $$
- **Confidence**: $conf(p \Rightarrow j) = P(i_{t+1} = j \mid (i_{t-k+1}, \dots, i_t) = p)$

A pattern is *frequent* if $sup(p) \geq \sigma_{min}$ and *predictive* if $\max_j conf(p \Rightarrow j) \geq \gamma_{min}$, for thresholds $\sigma_{min}, \gamma_{min} \in (0, 1]$.

Pattern recognition proceeds in three stages:

1. **Frequent-sequence mining.** Candidate subsequences are enumerated from execution traces with occurrence counts; successor statistics are accumulated alongside, and sequences below $\sigma_{min}$ are pruned together with their statistics—so confidence is estimated only over frequent sequences. The output is the set of frequent, predictive patterns.

2. **Behavior grouping.** The shipped `PatternRecognizer` groups dialogues deterministically by *action signature*—the sorted set of system-turn contents—and promotes each signature bucket with sufficient support (and, for dialogue-derived buckets, an acceptable success rate) into a pattern. An earlier prototype used TF-IDF embeddings with density-based clustering and k-means; we abandoned it for production use because it was non-deterministic across library versions and degenerate on the short traces agents actually provide (a k-means view over instruction-trace features remains available in `pattern_mining`, with fixed `random_state`). Promotions split by source: trace-mined promotions execute by **composite replay**—the mined pattern carries its constituent instruction sequence and the dataflow edges between its members, and the learned instruction's `execute` body re-runs that sequence in mined order against a shared context built from its inputs, merging outputs and returning the keys declared by its `produces_X` postconditions—while dialogue-mined templates, which carry no recorded instruction body, raise `NotImplementedError` rather than silently no-opping. Promotion is deterministic: pattern identity derives from the mined instruction-id sequence, so the same traces promote the same instructions in the same order across runs. Replay is plain sequential composition—nothing smarter—so a learned instruction reproduces the outputs of its recorded subsequence on the same inputs but does not generalize to unseen parameter values; no automated mining-to-optimization path exists yet (promotion is a developer decision).

3. **Pattern validation.** Each candidate is checked for significance—support above $\sigma_{min}$ and, for dialogue-derived patterns, an acceptable success rate—before promotion to a candidate instruction.

This design is deliberately statistical: agent traces are short and noisy, and frequency statistics degrade gracefully where grammar induction would overfit. It also composes with the compiler—a frequent, verified pattern is exactly the candidate for promotion to a fused instruction (Section 3.4.4).

### 3.3.2 Instruction Extraction Pipeline

The shipped extraction process instantiates the Section 3.3.1 pipeline: frequent-sequence mining (pattern detection), deterministic action-signature grouping (instruction formation, carrying the recorded constituent sequence and its dataflow edges), and threshold-gated promotion with the standard optimization passes applied to any resulting stream (optimization and validation).

### 3.3.3 Instruction Optimization and Verification

Optimization is coupled to verification: where verified-compilation efforts such as [37] verify the transformations of a whole compiler, AgentISA's verification targets a different granularity—the validity of individual instructions and the obligations of individual rewrites:

**Definition 3.5** (Instruction Verification): An instruction i is valid if:
$$
\forall s \in S: \text{pre}(i,s) \implies \text{post}(i,\text{exec}(i,s))
$$

For pure instructions with finite, small state spaces, the shipped checker (`agentflow.core.isa.bounded_checker`) enumerates the candidate input states derived from the declared parameter constraints—exhaustively for finite domains such as `bool` and enum parameters—and checks the implication on each state; for larger or non-enumerable spaces it probes a bounded, deterministically sampled set of states (default cap 256), which makes it a *bounded* verification procedure rather than a complete one. For every pure instruction the checker generates the bounded state space, filters it by well-typedness and declared preconditions, executes the instruction, and evaluates three obligations on each surviving state—progress (no raise where the precondition holds), preservation (declared outputs present and well-typed), and the postcondition itself. Side-effectful instructions are never executed by the checker and are reported as out of scope; their obligation is delegated to authoring-time unit tests and review. Theorem 3.2 is stated conditionally on this obligation, Theorems 3.4–3.6 carry the separate rewrite obligations of Appendix C.7, and for state spaces beyond the enumeration bound the condition remains substantive rather than discharged (Appendices B.7 and C.7):

$$
\forall s \in S_{bounded} \subseteq S: \text{pre}(i,s) \implies \text{post}(i,\text{exec}(i,s))
$$

where $S_{bounded}$ is the explored subset; completeness is traded for tractability—the standard bounded-exploration compromise, with the operational-semantics treatment of instruction meaning following [36].

The guarantee therefore comes in two explicitly separate layers. **Layer 1 (theory, conditional):** Theorem 3.2 and the type-safety results of Appendix B hold for every program whose instructions satisfy Definition 3.5 *on all of $S$*, under the total-definedness premise of Appendix B.7—a mathematical statement about a proof obligation, not about the checker. **Layer 2 (implemented verification):** the shipped checker discharges that obligation only on $S_{bounded}$ (exhaustive enumeration for small finite domains; deterministic bounded sampling with cap 256 otherwise) and only for pure instructions; side-effectful instructions are reported out of scope. Success of the bounded check is evidence toward, not a proof of, Layer 1's premise; no claim in this paper equates the two.

**Theorem 3.4** (Optimization Soundness): Every atomic rewrite of the pass pipeline—fusion (Appendix C.2's rule, applied by the combining pass), dead-code elimination, and instruction combining (Section 3.4.4)—preserves the denotation of the instruction sequence it transforms: $\text{sem}(r(i)) \equiv \text{sem}(i)$, read as the partial equivalence of C.1.

*Proof sketch*: By structural induction on instruction composition. The base case shows each atomic rewrite preserves the denotation of a single instruction; the inductive step lifts preservation through each composition operator of Definition 3.6. The full proof, with the per-rewrite case analysis, is in Appendix C.

The shipped checker (`agentflow.core.isa.bounded_checker.verify_instruction`) implements Definition 3.5 on the bounded state space exactly as described above (enumerate, filter by preconditions, execute only pure instructions); it reports violations structurally:

```python
def verify_instruction(instruction, pre_condition, post_condition):
    if instruction.side_effects: return skipped(instruction)  # never executed
    state_space = generate_state_space(instruction)  # bounded: exhaustive or seeded, cap 256
    for state in state_space.states:
        if pre_condition(state):
            next_state = execute(instruction, state)
            check(progress, preservation, post_condition, next_state)  # structured violation reporting
    return verdict  # checked_states, exhaustive, violations by kind, passed
```

## 3.4 Instruction Organization and Optimization

### 3.4.1 Instruction Composition

The AgentISA supports hierarchical instruction composition through a formal composition algebra:

**Definition 3.6** (Instruction Composition): Given instructions $i_1, i_2 \in I$, we define composition operators:

$$
\begin{align*}
&\text{Sequential}: i_1 \circ i_2 \\
&\text{Parallel}: i_1 \parallel i_2 \\
&\text{Conditional}: i_1 \triangleright i_2 \\
&\text{Iterative}: i_1^* \\
\end{align*}
$$

Their denotations (sequential execution; parallel interleaving under disjoint state footprints; guarded execution of $i_2$ on $i_1$'s triggering outcome; zero-or-more iterations of $i_1$) are fixed in Appendix C.1.

Composition follows algebraic properties. Properties 1 and 3 hold for all instructions; Property 2 is a *conditional* law—purity and pairwise footprint disjointness together do *not* suffice, and idempotence of $i_1$ must be assumed separately:

1. **Associativity** (all instructions, up to denotation):
   $$(i_1 \circ i_2) \circ i_3 = i_1 \circ (i_2 \circ i_3)$$

2. **Distributivity** (conditional on idempotent $i_1$ and pairwise-disjoint footprints): the law
   $$i_1 \parallel (i_2 \circ i_3) = (i_1 \parallel i_2) \circ (i_1 \parallel i_3)$$
   holds under two premises that must be stated separately. **(a) Idempotence of $i_1$**: $\delta_{i_1} \circ \delta_{i_1} = \delta_{i_1}$ on the precondition-satisfying states—$i_1$'s writes must not change the values its own subsequent execution reads (a constant overwrite of declared outputs qualifies; a read-modify-write of a cell it also reads does not). **(b) Pairwise footprint disjointness**: the denotations of $i_1, i_2, i_3$ act on pairwise disjoint state components (declared outputs distinct, read sets disjoint from the other members' writes). Under (a), the second execution of $i_1$ on the right-hand side reproduces the first's writes; under (b), $i_2$ and $i_3$ cannot perturb $i_1$'s inputs between its two occurrences; disjoint updates commute, so both sides denote the same net state transformation.

   *Both premises are necessary, and neither follows from the other or from purity.* That purity does not imply idempotence is the sharper failure: a pure `update(+1)` applied twice increments twice (pure means free of *external* side effects; writes to declared state cells are permitted). Taking $i_1$ as such an update and $i_2 = i_3 = \epsilon$ (the identity, empty footprint, so (b) holds vacuously), the left side executes $i_1$ once and the right side twice: from $x = 0$ the two sides yield $x = 1$ and $x = 2$—the law fails for a pure instruction whose footprints are disjoint from every other participant's, and only premise (a) rules this out. Conversely, without (b), a write of $i_2$ into a cell $i_1$ reads changes $i_1$'s second execution even when $i_1$ is idempotent.

3. **Identity**:
   $$i \circ \epsilon = \epsilon \circ i = i$$

   where $\epsilon$ is the empty (identity) instruction.

**Remark** (implementation consequence): distributivity also fails for side-effectful instructions, for the same double-execution reason. Because the law is doubly conditional, the shipped compiler does not rely on it at all: parallel grouping is gated on purity plus disjoint dependencies and disjoint outputs, `side_effects=True` instructions are excluded from all such rewrites, and no distributivity-based simplification is applied. The law is stated for the algebra's completeness, not as a license for rewrites.

### 3.4.2 Instruction Pipeline Design

We formalize the instruction pipeline as a directed acyclic graph (DAG):

**Definition 3.7** (Pipeline DAG):
$$
P = (V, E, \lambda, \delta)
$$

where:

- V: Pipeline stages
- E: Dependencies between stages
- λ: Stage latency function
- δ: Data transfer function

Pipeline stages are organized as:

```
Pipeline {
    FETCH → DECODE → SCHEDULE → EXECUTE → COMMIT
    with
        hazard_detection: HazardGraph
        dependency_tracking: DependencyMatrix
        resource_allocation: ResourceMap
}
```

**Theorem 3.5** (Pipeline Correctness): For any valid instruction sequence $s$ and any schedule $S$ satisfying conditions S1-S4 of Appendix C.5—each instruction dispatched exactly once, dependencies respected, dispatch groups pairwise non-interfering, commit in program order—pipeline execution preserves program semantics.

*Proof*: By induction over the dispatch groups of $S$ in commit order, under the invariant that the architectural state after committing each prefix of groups equals the sequential state after the same instructions; S1–S4 supply exactly-once dispatch, dependency order, group commutativity, and in-order commit (details in Appendix C.5).

### 3.4.3 Instruction Selection and Scheduling

Instruction selection follows a tree-pattern matching approach:

**Definition 3.8** (Selection Pattern):
$$
\pi = (\tau, C, \text{cost}, \text{code})
$$

where:

- $\tau$: a tree pattern over the IR (unrelated to the type-system symbol $T$ of Definition 3.2)
- $C$: context constraints on the surrounding state
- cost: Cost function
- code: Generated code

Given an IR tree $t$ to cover, the selection process is formalized as:

$$
\text{Select}(t) = \argmin_{\pi \in \Pi} \{\text{cost}(\pi) \mid \text{matches}(\pi.\tau, t) \land \text{satisfies}(t, \pi.C)\}
$$

Scheduling optimization problem (with $\text{delay}(i)$ the incurred stall of $i$, $\text{active}(t)$ the instructions running at time $t$, $\text{res}(i)$ the resource demand of $i$, and Capacity the resource bound):

$$
\begin{align*}
&\text{minimize} \sum_{i \in I} \text{delay}(i) \\
&\text{subject to:} \\
&\quad \forall (i_1, i_2) \in \text{Deps}: \text{start}(i_2) \geq \text{end}(i_1) \\
&\quad \forall t: \sum_{i \in \text{active}(t)} \text{res}(i) \leq \text{Capacity}
\end{align*}
$$

### 3.4.4 Static and Dynamic Optimization

**Static optimization.** Static optimization employs several techniques:

1. **Peephole Optimization**:
   $$
   \text{PH}(i_1,...,i_n) = \begin{cases}
   \text{opt}(i_1,...,i_n) & \text{if pattern exists} \\
   i_1,...,i_n & \text{otherwise}
   \end{cases}
   $$

2. **Dead Code Elimination**:
   $$
   \text{Live}(i) = \text{Use}(i) \cup (\text{Live}_{\text{out}}(i) - \text{Def}(i))
   $$
   The two shipped passes approximate this fixpoint at different conservatism levels. The *compiler* pass (formal streams) removes a side-effect-free instruction when no live consumer depends on its id or any declared output; the *trace optimizer* pass (executable instructions) is more conservative—any instruction declaring at least one output is kept, because executable traces lack the formal dependency annotations that make output-consumption analysis exact. The compiler experiments (Section 5.3) exercise the former; the dynamic hot-path machinery (Section 5.4) the latter.

3. **Instruction Combining** (the named candidate rules below are specified and not registered—Section 4.1.3; the shipped optimizer's built-in pairwise compatibility combining is active, and it is this built-in combining that the hot-path trace re-optimization of Section 5.4 applies):

   ```
   Combine Rules {
     // LOAD/STORE/MOVE are textbook mnemonics, illustrative
     // only; they are not shipped opcode names
     LOAD followed by STORE → MOVE
     Multiple CONTEXT_UPDATE → Single UPDATE
     Sequential QUERY → Batch QUERY
   }
   ```

**Dynamic optimization.** Dynamic optimization in the shipped runtime is **repetition-triggered and threshold-based**; it does not (yet) respond to load or performance signals:

1. **Hot Path Detection** (implemented): the `DynamicOptimizer` keeps a monotone counter per exact instruction sequence (`path_id` = the ordered join of instruction ids); a path is flagged hot when its count exceeds a configurable threshold *and* its most recent recorded success rate is at least a stability bound (0.9 by default). There is no windowing and no load signal:
   $$
   \text{Hot}(p) :\!\!\iff \text{count}(p) > \theta \land \text{success}(p) \geq \alpha
   $$

2. **Trace Optimization** (implemented): once hot, the cached sequence is re-optimized by the static pass machinery—peephole, dead-code elimination, instruction combining—bounded to a fixed number of passes; the runtime then serves the shrunken form for subsequent executions (demonstrated end-to-end in Section 5.8).

3. **Trace Formation and Adaptive Recompilation** (specified, not implemented): sub-trace extraction from observed frequency profiles, and performance-triggered recompilation (`should_recompile`: execution count above threshold *and* observed performance below expectation) remain design-level. The shipped runtime re-optimizes the whole cached sequence once per recording after the threshold is crossed, with no performance feedback loop.

Two known consequences of the shipped design: because `path_id` requires exact sequence identity, trace diversity (similar-but-not-identical sequences) never accumulates counts and the mechanism stays cold; and hot-path state (sequences, counters, statistics) is unbounded. Both are discussed with their remedies in Section 6.1.

### 3.4.5 Verification and Analysis

Optimization correctness is argued through:

**Theorem 3.6** (Pass-Pipeline Soundness, extending Theorem 3.4): For any optimization transform T composed of the pass pipeline's rewrites,
$$
\forall p \in \text{Program}: \text{sem}(T(p)) \equiv \text{sem}(p)
$$

**Proof Sketch**:

1. Show semantic preservation for each atomic transformation (the atomic-rewrite lemma, Appendix C.2)
2. Prove composition of transformations preserves semantics (the compositionality lemma, Appendix C.3)
3. Note that hazard-free pipeline execution is *not* part of this theorem—it is Theorem 3.5 (Appendix C.5), invoked separately; this theorem's contribution is the rewrite layer (steps 1–2).

**Performance Model** (specified; the benchmarks of Section 5 measure latency and token counts directly and do not evaluate it):
$$
\text{Perf}(p) = \sum_{i \in p} (\text{base\_cost}(i) + \text{overhead}(i) - \text{optimization\_gain}(i))
$$

where optimization_gain is calculated as:
$$
\text{optimization\_gain}(i) = \sum_{o \in \text{Opts}} \text{benefit}(o,i) \times \text{application\_rate}(o,i)
$$

## 3.5 Capability Assessment

Agents are heterogeneous: an agent may be strong at reasoning but weak at tool use, or vice versa. To make such differences explicit—and to give the multi-agent coordinator of Section 5.6 a principled basis for agent selection—AgentISA includes a capability assessment subsystem, implemented as `CapabilityProfile` in `agentflow.core.isa.capability`.

**Definition 3.9** (Capability Profile): A capability profile of an agent $a$ maps each capability domain $D$ (the shipped enum carries eight domains: reasoning, memory, generation, optimization, specialization, coordination, security, robustness) to a score–weight pair:
$$
\text{Cap}(a) = \{(\text{dom}_k, s_k, w_k) \mid \text{dom}_k \in D\}
$$

where each domain carries a score $s_k \in [0, 1]$ and a weight $w_k \geq 0$ reflecting the domain's importance in the current deployment context.

**Implementation status.** The aggregation and selection mechanisms ship (`CapabilityProfile`, `get_overall_score`, and the coordinator's threshold selection) and are functionally tested in Section 5.6; the instruction-level metric extraction is a specified stub—the seventeen `_measure_*` functions currently return 0.0—so `assess_capabilities` has no production call site yet, and the tests of Section 5.6 inject `DomainScore` values directly. The pipeline's *mechanics* are pinned by unit tests (`tests/core/isa/test_capability_pipeline.py`): end-to-end assessment over a real instruction list produces well-formed results for all eight domains with stub-zero scores, the domain aggregation is the confidence-weighted mean with the weight deliberately absent (guarding against a double-weighting defect observed during development), and the top-level weighted mean applies weights exactly once. An earlier version of the aggregation double-counted—domain weights entered both the per-domain score and the overall mean, capping an end-to-end assessment well below the coordinator's 0.7 threshold; the aggregation has been fixed to apply the weight only at the top level (equations below).

Assessment proceeds in three steps:

1. **Domain metric computation.** For each domain, `assess_capabilities` is *specified* to compute a set of `CapabilityMetric` values from the agent's instruction set—e.g., for the reasoning domain, metrics derived from the prevalence and complexity of reasoning-type instructions. As stated above, this extraction is a specified stub in the shipped code, so exercised assessments derive domain scores from injected metrics.

2. **Domain aggregation.** Metrics are aggregated into a per-domain `DomainScore` as a confidence-weighted mean, matching the shipped `_compute_weighted_score`:
$$
s_k = \frac{\sum_{m \in M_k} v_m \cdot \text{conf}_m}{\sum_{m \in M_k} \text{conf}_m}
$$

where $M_k$ is the metric set of domain $k$, $v_m$ the metric value, and $\text{conf}_m$ its confidence; an empty metric set scores $0$, and zero total confidence degenerates to the unweighted mean. The domain weight $w_k$ is deliberately absent here—it enters only once, at the top level (an earlier shipped form folded it into $s_k$ as well, double-counting it; Section 5.6's tests were unaffected because they inject `DomainScore` values directly).

3. **Overall score.** `get_overall_score` computes the weighted mean across assessed domains:
$$
S(a) = \frac{\sum_{k} w_k \, s_k}{\sum_{k} w_k}
$$

with $S(a)$ falling back to the unweighted mean when all weights are zero.

Three properties of this design deserve emphasis. First, it is **static rather than live-probing**: assessment is designed to be cheap, deterministic, and side-effect-free, though the shipped metric extraction is the stub disclosed above and exercised scores are injected. Second, it is **compositional**: because scores summarize instructions, fusing or eliminating instructions during compilation (Section 3.4) changes the profile predictably, and a production deployment would re-assess after optimization (the shipped coordinator does not yet do so). Third, it directly serves coordination: the agent-selection predicate in Section 5.6 chooses the highest-scoring agent above a capability threshold for each task, subject to the security policies of the `SecurityMonitor`—with the tests injecting scores directly, as disclosed above.

We state the scope of this subsystem honestly: `CapabilityProfile` assesses *declared instruction-level capability*, which correlates with—but does not measure—end-to-end task performance. Cross-model behavioral comparison at benchmark level is the domain of evaluation suites such as BFCL [17] and AgentBench [29]; the two views are complementary.
