# Reproducing the AgentISA Paper Numbers

This file anchors the paper's reproducibility claims. The numbers in Section 5
are generated from the repository state whose build produces the submitted PDF;
the commit hash is recorded below at submission time.

## Version anchor

- Repository state: the commit tagged for the submission build of
  `docs/AgentISA.pdf` (run `git rev-parse HEAD` in the artifact root; the
  submission snapshot is the parent of any post-camera fix).
- Runtime: Python 3.14, dependencies as pinned in `venv/` and the project
  lock file; pytest 9.x.
- Build: `python3 scripts/build_paper.py` regenerates `docs/AgentISA.tex`
  and `docs/AgentISA.pdf` from `docs/AgentISA.md`.

## Functional tests

```bash
python -m pytest tests/core/isa/test_coordination.py \
                 tests/core/instructions/test_agent_coordination.py -q
# 30 + 13 = 43 passed

python -m pytest tests/system/ -q
```

Note: the coordination tests initialize a local Ray session; on machines
where `$HOME` is not writable, redirect `HOME` to a writable directory first.

## Regenerating Section 5

| Paper section | Command | Raw results |
| --- | --- | --- |
| 5.2–5.5 | `python examples/benchmarks/isa_microbench.py` (full battery; see `--help` for the per-section entry points used in the paper) | `examples/benchmarks/isa_microbench_results.json` |
| 5.2–5.5 (real LLM) | `python examples/benchmarks/isa_real_llm_bench.py --tasks 200 --seeds 1 --zipf 0.5 1.0 1.5 --max-tokens 64` | `examples/benchmarks/isa_real_llm_results.json` |
| 5.6 | covered by the functional tests above | — |
| 5.7 / 5.7.1 | `python examples/paper_real_llm_compiler_test.py --backend real --scale paper` (add `--backend mock` for the offline mock) | `examples/paper_real_llm_compiler_results_mock.json`, `examples/paper_real_llm_compiler_results_real.json` |
| 5.8 | `python examples/system_task_demo.py`; battery: `python examples/system_task_workload.py` | `examples/system_task_results.json` |

## Claim-to-code map

| Paper claim | Implementing module | Test | Raw results |
| --- | --- | --- | --- |
| Instruction set + contracts (§3.1–3.2) | `agentflow/core/isa/instruction.py`, `agentflow/core/isa/formal_instruction.py`, `agentflow/core/isa/types.py` | `tests/core/isa/test_instruction.py`, `tests/core/isa/test_validation.py` | — |
| Bounded verification checker (§3.3.3) | `agentflow/core/isa/bounded_checker.py` | `tests/core/isa/test_bounded_checker.py` | — |
| Optimization passes: fusion / DCE / grouping (§3.4, §4.1.3) | `agentflow/core/isa/instruction_optimization.py`, `agentflow/core/isa/compiler.py` | `tests/core/isa/test_instruction_optimization.py`, `tests/core/isa/test_compiler_optimization.py` | `isa_microbench_results.json` |
| Hot-path detection (§4.2) | `agentflow/core/isa/analyzer.py`, `agentflow/core/isa/agent_isa.py` | `tests/core/isa/test_agent_isa.py` | `isa_microbench_results.json` |
| Result caching (§3.2) | `agentflow/core/isa/instruction.py` (result caching in the dispatch path) | `tests/core/isa/test_instruction.py` | `isa_microbench_results.json`, `isa_real_llm_results.json` |
| Runtime monitoring (§4.3) | `agentflow/core/isa/monitoring.py`, `agentflow/core/isa/validation.py` | `tests/system/` | `paper_real_llm_compiler_results_real.json` |
| Capability aggregation (§3.5) | `agentflow/core/isa/capability.py` | `tests/core/isa/test_coordination.py` (injected scores) | — |
| Multi-agent coordination (§5.6) | `agentflow/core/isa/coordination.py` | `tests/core/isa/test_coordination.py`, `tests/core/instructions/test_agent_coordination.py` | — |

### Specified-only components (not implemented)

These are design specifications in the paper, **not** shipped mechanisms:

- Symbolic-execution equivalence checker (§4.3.1) — no module; obligations
  argued per-pass in Appendix C instead.
- Tree-pattern dynamic-programming instruction selection (§4.1.2) — the
  shipped selector is the greedy priority rule.
- Cycle-accurate pipeline scheduler (§4.1.2) — the shipped schedule loop is
  sequential execution plus parallel grouping.
- Capability metric extraction (§3.5) — the seventeen `_measure_*` functions
  in `capability.py` return 0.0 (stub); §5.6 tests inject `DomainScore`
  values directly.
- Fusion-node executable composite codegen (§4.1.3 / C.2) — the shipped
  fusion emits formal fusion nodes; the recorded body executes constituents.
