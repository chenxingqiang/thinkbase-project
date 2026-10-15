"""RSI experiment package: fidelity-gated recursive self-improvement.

All modules are deterministic under a fixed seed (torch CPU, single
thread) so every number in the thesis is reproducible from the command
line.  The experiment driver orchestrates each run as an AgentISA
instruction stream (see ``streams.py``).
"""

from . import data, models, metrics, fidelity, rsi, streams  # noqa: F401
