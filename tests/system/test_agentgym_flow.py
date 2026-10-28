"""AgentGym task flows: the parsers that decide what an "instruction" is.

The cross-family result in Appendix I.8 depends on the mapping from free text to
an operation: if ``take alarmclock 1 from desk 1`` and ``take mug 2 from desk 1``
were different instructions, the household family would look like the SQL family
and the two-regime finding would be an artifact of parsing. These tests pin that
choice -- instance numbers are parameters, not identity -- and the per-environment
grammars of the three action spaces.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "examples" / "real_traces" / "prepare_agentgym.py"


def _module():
    name = "prepare_agentgym"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_instance_numbers_are_parameters_not_identity() -> None:
    """Two calls on different instances share one instruction."""
    m = _module()
    first = m.parse_alfworld("take alarmclock 1 from desk 1.")
    second = m.parse_alfworld("take alarmclock 2 from desk 1.")
    assert first == second == ("take", ("alarmclock", "desk"))


def test_alfworld_grammar_separates_receptacle_arguments() -> None:
    m = _module()
    assert m.parse_alfworld("put mug 1 in cabinet 2") == ("put", ("mug", "cabinet"))
    assert m.parse_alfworld("go to drawer 3") == ("go to", ("drawer",))
    assert m.parse_alfworld("clean plate 1 with sinkbasin 1") == ("clean", ("plate", "sinkbasin"))


def test_babyai_aliases_collapse_to_the_same_opcode() -> None:
    """``pickup`` and ``pick up`` are one instruction in the environment."""
    m = _module()
    assert m.parse_babyai("pickup red ball 1") == m.parse_babyai("pick up red ball 1")
    assert m.parse_babyai("turn left") == ("turn left", ())


def test_sqlgym_template_is_the_verb_plus_table_set() -> None:
    """The shared verb and the differing tables are what make SQL a third regime."""
    m = _module()
    assert m.parse_sqlgym("SELECT name FROM students WHERE id = 1") == ("select", ("students",))
    assert m.parse_sqlgym("select count(*) from A join B on A.id=B.id")[1] == ("a", "b")


def test_only_the_last_action_marker_counts() -> None:
    """A thought that mentions an action earlier must not shadow the real one."""
    m = _module()
    text = "Thought: I will Action: pretend to look.\nAction: go to cabinet 1"
    assert m.last_action(text) == "go to cabinet 1"
    assert m.last_action("Thought: no action here") is None
