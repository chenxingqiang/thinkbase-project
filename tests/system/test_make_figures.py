"""Smoke test: scripts/make_figures.py regenerates all six paper figures,
and its manifest records data consistent with the committed results JSONs.

The figure script itself asserts every plotted value against its source
artifact and fails loudly on mismatch, so a clean exit IS the data check;
this test additionally pins the manifest's shape so silent figure loss
(e.g. a renamed asset) cannot slip through the suite.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "docs" / "figures"
SIX = [
    "wide_1_architecture.pdf",
    "2_compile_pipeline.pdf",
    "3_cache_exactness.pdf",
    "4_token_mix_sweep.pdf",
    "5_quality_per_class.pdf",
    "6_ceiling_census.pdf",
]


def test_make_figures_regenerates_all_six():
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts/make_figures.py")],
        capture_output=True, text=True, timeout=300, cwd=ROOT,
    )
    assert r.returncode == 0, f"make_figures failed:\n{r.stdout}\n{r.stderr}"
    for name in SIX:
        assert (FIGS / name).exists(), f"missing figure: {name}"
        assert (FIGS / name).stat().st_size > 1000, f"suspiciously small: {name}"


def test_figures_manifest_pins_expected_data():
    m = json.loads((FIGS / "figures_manifest.json").read_text(encoding="utf-8"))
    assert set(m["figures"]) == set(SIX)
    f3 = m["figures"]["3_cache_exactness.pdf"]
    assert f3["all_cells_exact"] is True
    assert len(f3["families"]) == len(f3["sources"]) == len(f3["cells"])
    assert m["figures"]["6_ceiling_census.pdf"]["glm_512"] == [22, 22]
    assert m["figures"]["6_ceiling_census.pdf"]["minimax_512"] == [101, 104]
    assert m["figures"]["6_ceiling_census.pdf"]["minimax_1500"] == [68, 104]
    assert m["figures"]["6_ceiling_census.pdf"]["doubao_1500"] == [0, 104]
    assert m["figures"]["4_token_mix_sweep.pdf"]["cells"] == 195
    assert m["figures"]["2_compile_pipeline.pdf"]["steps_compiled"] == 6
