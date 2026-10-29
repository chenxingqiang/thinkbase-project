# Real agent trajectories (pilot corpus)

The paper's evidence uses two workload families: linear sequences with planted
optimization targets (Section 5.3) and the generator's graph-shaped corpus
(I.4.2). `tau_bench_historical.jsonl` is the third, real family: tool-call
trajectories produced by real models running the τ-bench agent benchmark.

## Source and license

- Repository: `sierra-research/tau-bench`
- Revision: `59a200c6d575` (branch `main`, 2026-03-18)
- License: MIT (Copyright (c) 2024 Sierra)
- Files used: `historical_trajectories/{gpt-4o-retail,gpt-4o-airline,sonnet-35-new-retail,sonnet-35-new-airline}.json`
- Upstream SHA-256:

  ```
  df01707894836168ff0ec9616b0bf08f66c7e5afcf313e5fe4f7a2f5c2ec938b  gpt-4o-retail.json
  e9e6c0297660c537f83d4fd9c476ce7a9a86ecd2784874b7bfc13be598e37bfa  gpt-4o-airline.json
  0df526398e9d2720c32d340815cffb04fe8c4f8a61b1f4f84bf3bb558f760131  sonnet-35-new-retail.json
  fe62fcd514b855b36f156dd4c3c7748597b392b006aff739b53337a9f3ba94d1  sonnet-35-new-airline.json
  ```

## What these trajectories are, and are not

They are **real model behavior**: GPT-4o and Claude 3.5 Sonnet deciding which
tool to call, with which arguments, while solving retail and airline tasks. The
tool *environment* is τ-bench's mock backend, so the traces say nothing about
network behavior, latency, or production traffic. They are the closest public
artifact to "what an agent actually does", not a deployment sample.

A record is one task attempt:

```json
{"source": "gpt-4o-retail", "task_id": 0, "reward": 1.0,
 "turns": [[["get_order_details", {"order_id": "#W2378156"}]]]}
```

`turns[t][c]` is `[tool_name, arguments]`; a turn with several calls means the
model issued them in one assistant message (parallel by construction). Message
prose, system prompts, and tool observations are dropped: the pilot measures
structure and cache keys, and a cache key is built from the *inputs* of a call,
never its result.

## Regenerating

```bash
# upstream tarball (codeload works where github.com HTML does not)
curl -sSL -o tb.tar.gz https://codeload.github.com/sierra-research/tau-bench/tar.gz/59a200c6d575
tar -xzf tb.tar.gz
./venv/bin/python examples/real_traces/prepare_tau_bench.py --tau-bench-root ./tau-bench-59a200c6d575
```

The corpus is committed (1.4 MB) so the pilot reproduces without the 49 MB
upstream tree; the upstream digests above let a reader verify the provenance.

## Scale

| source | trajectories | tool calls |
|---|---|---|
| gpt-4o-retail | 460 | 3,274 |
| gpt-4o-airline | 182 | 1,164 |
| sonnet-35-new-retail | 919 | 7,086 |
| sonnet-35-new-airline | 399 | 2,761 |
| **total** | **1,960** | **14,285** |

Tasks whose trajectory contained no tool call are omitted (28 of them).

## The same trajectories, with observations

`tau_bench_observed.jsonl` is the second reading of the *same* files: each call
keeps the observation the environment returned for it, joined by `tool_call_id`.
It exists because compound instructions cannot be tested on an inputs-only
corpus — asking whether a fused sequence can be instantiated means asking
whether its later arguments come from earlier *results*, and that information is
in the `tool` messages the pilot corpus drops.

```bash
./venv/bin/python examples/real_traces/prepare_tau_bench_observed.py \
    --tau-bench-root ./tau-bench-59a200c6d575 \
    --output examples/real_traces/tau_bench_observed.jsonl
```

Scale: first trial of each task only (325 tasks), 2,275 calls, 2,221 (97.6%)
carrying an observation. Observations are truncated at 400 characters. The
truncation is disclosed because it is not neutral in the direction that matters:
a value cut off cannot be found by the binder, so every bindability figure in
Appendix I.9 is a lower bound on what a longer observation would support.

The committed file is 1.0 MB. It is derived from the digests listed above, so
the provenance argument is the same one.

## Second family: bash-only coding agents

`SWE-agent-trajectories` on Hugging Face (`nebius/SWE-agent-trajectories`,
revision `68195a14`, CC-BY-4.0): recorded SWE-agent runs on SWE-bench
instances. `swe_agent_sample.jsonl` takes the first 300 trajectories of shard
`train-00000-of-00012` in file order (8,150 actions; swe-agent-llama-70b/8b/405b
backbones).

These trajectories carry no structured tool calls: SWE-agent emits its action
as the last fenced block of an assistant message. `prepare_swe_agent.py` parses
it mechanically—a block starting with `str_replace_editor` becomes an editor
call, anything else a shell command with its first line kept. This shard is
bash-only: all 8,150 actions are shell commands, so the family has no read-only
instruction, which is exactly why it is reported separately in I.6.

```bash
# China mirror works where huggingface.co resolves to a poisoned address
curl -L -o swe0.parquet \
  https://hf-mirror.com/datasets/nebius/SWE-agent-trajectories/resolve/main/data/train-00000-of-00012.parquet
./venv/bin/python examples/real_traces/prepare_swe_agent.py --parquet swe0.parquet --limit 300
```

| source | trajectories | tool calls | cacheable calls |
|---|---|---|---|
| swe-agent-llama-70b/8b/405b | 300 | 8,150 | 0 |

## Third family: task flows with explicitly finite action spaces

`AgentGym/AgentTraj-L` (Apache-2.0) ships rollouts in ten environments. Three
are compacted into `agentgym_sample.jsonl` because their action spaces are
explicit vocabularies rather than free-form tool arguments:

| environment | action space | records | actions |
|---|---|---|---|
| `alfworld` (household) | 13 verbs | 2,420 | 32,247 |
| `babyai` (grid) | 8 verbs | 810 | 4,642 |
| `sqlgym` (analytics) | 1 verb | 3,000 | 3,000 |

```bash
./venv/bin/python examples/real_traces/prepare_agentgym.py \
    --input /path/to/AgentTraj-L --output examples/real_traces/agentgym_sample.jsonl
./venv/bin/python examples/benchmarks/instruction_precipitation.py \
    --corpus examples/real_traces/agentgym_sample.jsonl --param-mode shape --max-idle 100 \
    --output examples/benchmarks/instruction_precipitation_agentgym_results.json
```

Disclosed limitation: these files are per-step samples (an episode prefix and
its next action), so they measure action-level precipitation across tasks, not
the composition of a whole episode into one instruction.
