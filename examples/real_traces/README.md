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
