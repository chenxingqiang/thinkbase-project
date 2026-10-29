"""The number gate's failure branch must stay last, or checks get swallowed.

Twice in this project a new verification block was appended after the
``if fails:`` branch, so its failures accumulated into a list that nobody read
and the gate kept printing clean. A structural invariant is cheaper than
remembering: the branch is the last statement in the file, and the clean banner
only prints when the list is empty.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE = REPO_ROOT / "scripts" / "number_regression.py"


def test_failure_branch_is_the_last_statement() -> None:
    lines = GATE.read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("if fails:")]
    assert len(starts) == 1, f"expected one failure branch, found {len(starts)}"
    tail = [line for line in lines[starts[0]:] if line.strip()]
    assert tail[0].startswith("if fails:"), tail[0]
    assert any("sys.exit(1)" in line for line in tail), "failure branch must exit nonzero"
    assert any("REGRESSION-CLEAN" in line for line in tail), "clean banner must follow the branch"
    assert tail.index(next(line for line in tail if "REGRESSION-CLEAN" in line)) > 0
    for line in tail:
        assert not line.startswith("# ----"), (
            "a check section was appended after the failure branch; "
            "move the branch to the end of the file")


def test_clean_banner_follows_the_failure_branch() -> None:
    text = GATE.read_text(encoding="utf-8")
    branch = text.index("if fails:")
    banner = text.index("REGRESSION-CLEAN")
    assert branch < banner
    assert "sys.exit(1)" in text[branch:banner]
