# Pre-registered protocol: the end-to-end experiment

**Status: not run.** This document is the design of the experiment Section 5.9 item 2 and
`.drafts/rebuttal_matrix.md` name as the only one that could change the paper's verdict.
It is written before any data exists so that the result cannot be chosen after the fact.
Estimated execution: 2–3 months, dominated by harness work, not by compute.

## 1. Why it cannot be the current harness

Every measurement in the paper replays **recorded** trajectories: a "tool call" is a
lookup of the recorded output, and the model never sees a result and never decides the
next step. An end-to-end claim needs the opposite arrangement — the model chooses the
tool calls, the environment answers them, and success is scored by the environment's own
reward function. The repository has no such loop today (`AgentCoordinator` plans
capability-labelled agents; it does not solve tasks), so the harness is new code, not a
flag.

## 2. Design

Three arms, same task set, same model, same seed schedule, same time budget:

| Arm | Cache | Bypass |
|---|---|---|
| `none` | disabled | — |
| `legacy` | the shipped `InstructionCache` (untyped: no tenant, admits effects, TTL/LRU) | as today |
| `typed` | `SoundResultCache` (tenant key, fail-closed effect gate, footprint eviction) | forced on the first occurrence of every (instruction, params, tenant) |

Randomize task order per seed. All three arms see the identical task sequence for a given
seed, so paired differences are within-task.

## 3. Endpoints (in order of importance)

1. **Task success** (primary). Environment reward, binary per task. Report the paired
   success-rate difference with a task-clustered interval, and the success *cost* of the
   typed cache (a hit that is refused is a call that runs; correctness must not move).
2. **Unsafe serving** (primary, safety). Count of hits that change the observable outcome
   of the trajectory, classified by Section 5.13's D1/D2/D3 classes, adjudicated by the
   environment state rather than by our own key — the strongest available answer to the
   circularity objection.
3. **Latency** (secondary). Wall clock per task, decomposed into model turns and tool
   execution. **The decomposition is the point**: in replay a hit saves a lookup, while
   the measured LLM round trip is 2.2 s (Section 5.7). If tool execution is a small share
   of end-to-end time, a cache hit is worth a small share of end-to-end latency, and the
   paper must say so before a reviewer does.
4. **Tokens** (secondary). Prompt and completion tokens per task, per arm.
5. **Cost** (secondary). Dollars at list prices, with the token mix reported.

## 4. Feasibility, from numbers already measured

| Quantity | Measured value | Source |
|---|---|---|
| LLM round trip, sequential | 2.2 s | Section 5.7 |
| 1,040 round trips | 39.4 min | Section 5.7 |
| Tool calls in the replay corpus | 14,285 (2,275 first-trial) | Section 5.12 |
| Hit rate, untyped cache | 41.70% cross-task | Section 5.12 |
| Hit rate, typed cache | 26.56% (459/1,728) | Section 5.13 |

A τ-bench retail episode is 8–15 model turns; at 2.2 s per turn and one tool call per
turn, a 200-task × 5-seed × 3-arm run is on the order of 10^4–10^5 sequential round trips,
i.e. **tens of hours of wall clock per arm** if run sequentially, which parallelizes across
tasks but not within one. The harness work (environment adapter, scorer, replay-safe
logging, three-arm driver) dominates.

## 5. Power

The unit is the task, not the call. With 200 tasks and a paired design, the detectable
difference in success rate at 80% power is roughly 2.8/sqrt(200) ≈ 6 points (McNemar,
two-sided, alpha 0.05, discordant-pair rate assumed 0.3 — the assumption is stated because
it is not yet measured). **A success-rate effect smaller than ~6 points will not be
resolvable at this scale**, so the protocol commits in advance to reporting the interval
and to calling a smaller point estimate inconclusive rather than positive.

## 6. Decision rules, fixed before the run

- If the typed arm's success rate is **below** `none` by more than the interval allows,
  the paper's claim is wrong and the paper says so: sound caching would cost task success.
- If the typed arm is **indistinguishable** from `none` and the legacy arm is **below**
  both, the paper's thesis is confirmed end to end.
- If all three are indistinguishable and the intervals exclude effects larger than 6
  points, the honest conclusion is that caching does not move task success at this scale,
  and the contribution reduces to latency and correctness — which is what the paper
  already claims, now with a bound.
- No arm is dropped, no seed is added after seeing results, and every arm's raw logs are
  committed.

## 7. Threats we already know

1. **Model non-determinism** at temperature > 0 makes arms differ for reasons unrelated to
   the cache; use temperature 0 where the provider supports it, and report the provider's
   guarantee rather than assuming determinism.
2. **Cache warmth** is a function of task order; randomize and report the order.
3. **Reward hacking** by the model would inflate all arms equally, which is acceptable for
   paired differences but must be disclosed.
4. **Cost**: three arms × 200 tasks is the smallest design with a chance of resolving a
   6-point effect; a smaller run should be reported as a pilot, not as the experiment.
