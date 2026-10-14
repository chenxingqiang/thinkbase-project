"""Sandbox execution for agent-generated code.

Shows the "instruction -> sandbox execution" chain: an agent produces code
(or a command), the sandbox runs it in isolation, and the result (stdout /
error / timeout) is returned to the agent.

Two backends:
* local -- resource-limited subprocess on this machine (default; use only
  for development/tests, NOT for untrusted production code);
* e2b   -- Firecracker microVM in the E2B cloud (strong isolation;
  set E2B_API_KEY; internet egress disabled by default).

Run:  python examples/sandbox_executor_example.py [e2b|local|auto]
"""

from __future__ import annotations

import os
import sys

# Make the repository root importable when run as a plain script.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentflow.sandbox import SandboxResult, create_sandbox_executor  # noqa: E402


def show(label: str, result: SandboxResult) -> None:
    print(f"--- {label}: {result.status} ({result.execution_time:.3f}s)")
    if result.stdout:
        print("  stdout:", result.stdout.strip().splitlines()[-1])
    if result.error:
        print("  error:", result.error)
    if result.stderr:
        print("  stderr:", result.stderr.strip().splitlines()[-1])


def main(backend: str) -> None:
    # 1. The agent "generates" a piece of code to execute.
    agent_code = (
        "def fibonacci(n):\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n):\n"
        "        a, b = b, a + b\n"
        "    return a\n"
        "print(f'fib(30) = {fibonacci(30)}')\n"
    )
    agent_command = "echo 'agent command executed safely'"

    with create_sandbox_executor(backend) as sandbox:
        show("run_code (agent-generated Python)", sandbox.run_code(agent_code))
        show("run_command (agent-generated shell)", sandbox.run_command(agent_command))

        # 2. The agent can also drop files into the sandbox workspace.
        sandbox.write_file("agent_notes.txt", b"scratch data from the agent")
        data = sandbox.read_file("agent_notes.txt")
        print("--- read_file: agent_notes.txt ->", data)

        # 3. A failing snippet surfaces as an error result (agent can retry).
        show("run_code (agent bug)", sandbox.run_code("raise ValueError('bad step')"))

        # 4. A runaway snippet is killed by the timeout.
        show(
            "run_code (runaway)",
            sandbox.run_code("import time; time.sleep(60)", timeout=2),
        )

    print("\nSandbox closed; workspace torn down.")


if __name__ == "__main__":
    backend = sys.argv[1] if len(sys.argv) > 1 else "local"
    if backend not in ("local", "e2b", "auto"):
        raise SystemExit(f"usage: {sys.argv[0]} [local|e2b|auto]")
    main(backend)
