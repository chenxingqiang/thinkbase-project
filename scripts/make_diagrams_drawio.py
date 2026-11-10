#!/usr/bin/env python3
"""The five AgentISA schematics, authored as draw.io files and exported to PDF.

The .drawio sources in docs/diagrams/ are the editable ground truth: open them
in diagrams.net (or the draw.io desktop app) to move a box, then re-run
``./venv/bin/python scripts/make_diagrams_drawio.py`` to regenerate the PDFs in
docs/figures/ that the papers include.

House style, one place: rounded panels filled with a light tint of their accent
and edged in the accent itself, 1 px orthogonal edges with small block
arrowheads, Helvetica, no in-figure titles, no decorative borders. The accent
wheel is the NPG (Nature Publishing Group) palette; see PALETTE.

Export is headless: ``draw.io --no-sandbox --disable-gpu --export``. The
desktop CLI can hang if a GUI instance is already open, so this script closes
any first.

Page geometry: draw.io exports 1 px ~= 0.73 pt, so each canvas is sized for the
width at which the figure prints (Figure 1 at 86% of a 7 in text column, Figure
2 at 88% of a 3.33 in column, the three supplement panels at full text width),
which keeps labels between 7.5 and 9 pt in the final PDF.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "diagrams"
FIG = ROOT / "docs" / "figures"
DRAWIO = Path("/Applications/draw.io.app/Contents/MacOS/draw.io")

PALETTE = {
    "ink": "#1a1a1a",
    "soft": "#4a4a4a",
    "green": "#00a087",   # sound / typed
    "teal": "#4dbbd5",    # routing / data
    "amber": "#f39b7f",   # gates / thresholds
    "blue": "#3c5488",   # external systems
    "brick": "#e64b35",   # damage / refusal
}
TINT = {  # 16% accent into white, computed offline and pinned
    "green": "#e6f5f1",
    "teal": "#e8f6fb",
    "amber": "#fdefe9",
    "blue": "#e9edf5",
    "brick": "#fdeceb",
    "paper": "#eef1f4",
}
FONT = "Helvetica"


def tint_to_hex(tint: str) -> str:
    return TINT[tint if tint in TINT else "paper"]


class Diagram:
    """Collects mxCells and serializes one draw.io page."""

    def __init__(self, name: str, width: int, height: int):
        self.name = name
        self.width = width
        self.height = height
        self.cells: list[str] = []
        self._id = 100

    def _next(self) -> str:
        self._id += 1
        return str(self._id)

    def node(self, x, y, w, h, label="", *, accent="paper", fs=10, rounded=0.10,
             dashed=False, fill=None, font_color=None, bold=False, halign="center"):
        """Rounded panel: tint fill, accent edge, centred label."""
        cid = self._next()
        stroke = PALETTE.get(accent, PALETTE["soft"])
        style = (
            f"rounded=1;whiteSpace=wrap;html=1;arcSize={int(rounded * 100)};"
            f"fillColor={fill or tint_to_hex(accent)};strokeColor={stroke};"
            f"strokeWidth=1;fontFamily={FONT};fontSize={fs};fontColor={font_color or PALETTE['ink']};"
            f"align={halign};verticalAlign=middle;"
            + ("dashed=1;" if dashed else "")
            + ("fontWeight=bold;" if bold else "")
        )
        self.cells.append(
            f'<mxCell id="{cid}" value="{esc(label)}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )
        return cid

    def text(self, x, y, w, h, label, *, fs=8, color=None, align="left", bold=False):
        cid = self._next()
        style = (
            f"text;html=1;whiteSpace=wrap;fontFamily={FONT};fontSize={fs};"
            f"fontColor={color or PALETTE['soft']};align={align};verticalAlign=middle;"
            + ("fontWeight=bold;" if bold else "")
        )
        self.cells.append(
            f'<mxCell id="{cid}" value="{esc(label)}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )
        return cid

    def edge(self, src, dst, *, label="", color=None, exit_=(1, 0.5), entry=(0, 0.5),
             points=(), fs=8, label_color=None, dashed=False, entry_dx=0, exit_dx=0,
             curved=False, end_arrow=True, width=1):
        color = color or PALETTE["soft"]
        style = (
            ("edgeStyle=orthogonalEdgeStyle;rounded=1;curved=0;" if not curved
             else "edgeStyle=none;curved=1;rounded=1;")
            + f"html=1;strokeColor={color};strokeWidth={width};"
            + (f"endArrow=block;endFill=1;endSize=4;" if end_arrow else "endArrow=none;")
            + ("dashed=1;" if dashed else "")
            + f"exitX={exit_[0]};exitY={exit_[1]};exitDx={exit_dx};exitDy=0;"
              f"entryX={entry[0]};entryY={entry[1]};entryDx={entry_dx};entryDy=0;"
            f"fontFamily={FONT};fontSize={fs};fontColor={label_color or color};labelBackgroundColor=#ffffff;"
        )
        pts = "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in points)
        geo = (
            f'<mxGeometry relative="1" as="geometry">'
            + (f'<Array as="points">{pts}</Array>' if pts else "")
            + "</mxGeometry>"
        )
        edge_id = self._next()
        self.cells.append(
            f'<mxCell id="{edge_id}" value="{esc(label)}" style="{style}" edge="1" parent="1" '
            f'source="{src}" target="{dst}">{geo}</mxCell>'
        )
        return edge_id

    def save(self) -> Path:
        SRC.mkdir(parents=True, exist_ok=True)
        path = SRC / f"{self.name}.drawio"
        xml = (
            '<mxfile host="app">\n'
            f'  <diagram id="{self.name}" name="Page-1">\n'
            f'    <mxGraphModel dx="800" dy="600" grid="0" page="1" pageScale="1" '
            f'pageWidth="{self.width}" pageHeight="{self.height}">\n'
            '      <root>\n'
            '        <mxCell id="0"/>\n        <mxCell id="1" parent="0"/>\n'
            + "\n".join(f"        {c}" for c in self.cells)
            + "\n      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n"
        )
        path.write_text(xml, encoding="utf-8")
        return path


def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;").replace("\n", "&#10;"))


def raw_edge(d: Diagram, p0, p1, *, color=None, width=1, dashed=False):
    """A plain line with no endpoints on any cell (strike-throughs, ties)."""
    color = color or PALETTE["soft"]
    style = (
        f"endArrow=none;html=1;strokeColor={color};strokeWidth={width};"
        + ("dashed=1;" if dashed else "")
    )
    cid = d._next()
    d.cells.append(
        f'<mxCell id="{cid}" value="" style="{style}" edge="1" parent="1">'
        f'<mxGeometry relative="1" as="geometry">'
        f'<mxPoint x="{p0[0]}" y="{p0[1]}" as="sourcePoint"/>'
        f'<mxPoint x="{p1[0]}" y="{p1[1]}" as="targetPoint"/>'
        f"</mxGeometry></mxCell>"
    )
    return cid


# --------------------------------------------------------------------------- 1
def architecture() -> Diagram:
    """Figure 1: the runtime band, 780 x 184 px.

    The hit return path runs along the top, the re-optimization loop along the
    bottom, so no edge crosses any box. Box labels are sized to print at 8 pt
    when the 780 px canvas is set to 86% of a 7 in text column.
    """
    d = Diagram("wide_dia_1_architecture", 780, 184)
    stream = d.node(20, 36, 115, 60, "authored stream", accent="green", fs=14)
    gate = d.node(180, 40, 100, 52, "security gate", accent="green", fs=14)
    coord = d.node(320, 36, 110, 60, "coordinator (router)", accent="teal", fs=14)
    cache = d.node(470, 40, 105, 52, "typed result cache", accent="green", fs=14)
    gw = d.node(615, 36, 140, 60, "model gateway", accent="blue", fs=14)
    d.edge(stream, gate)
    d.edge(gate, coord)
    d.edge(coord, cache, label="miss", fs=11)
    d.edge(cache, gw, label="read", fs=11)
    d.edge(cache, stream, label="hit", color=PALETTE["green"],
           label_color=PALETTE["green"], fs=12,
           exit_=(0.5, 0), entry=(0.3, 0), points=[(522, 6), (55, 6)])
    comp = d.node(320, 132, 110, 44, "compiler (fuse, DCE)", accent="paper", fs=12)
    hot = d.node(470, 132, 105, 40, "hot-path detector", accent="amber", fs=12)
    d.edge(coord, comp, label="re-optimize", fs=11, exit_=(0.5, 1), entry=(0.5, 0))
    d.edge(hot, comp, fs=11, exit_=(0, 0.5), entry=(1, 0.5))
    d.edge(comp, stream, label="re-adopt", fs=11,
           exit_=(0, 0.5), entry=(0.5, 1), points=[(77, 154)])
    return d


# --------------------------------------------------------------------------- 2
def compile_pipeline() -> Diagram:
    """Figure 2: seven authored steps compile to six, 350 x 154 px.

    Annotations live in the inter-row gap and below row two, never on squares;
    the compile edge reuses the first square of each row (no stacked cells).
    Labels print at 7 pt at 88% of a 3.33 in column.
    """
    d = Diagram("dia_5_compile_pipeline", 350, 154)
    d.text(20, 4, 200, 14, "authored stream, 7 steps", fs=10, color=PALETTE["ink"])
    row1 = []
    for k in range(7):
        x = 20 + k * 44
        dead = k == 3
        row1.append(d.node(x, 22, 34, 34, str(k + 1),
                           accent="brick" if dead else "paper", fs=12,
                           fill="#fdeceb" if dead else "#eef1f4"))
        if dead:
            raw_edge(d, (x + 2, 24), (x + 32, 52), color=PALETTE["brick"])
            raw_edge(d, (x + 32, 24), (x + 2, 52), color=PALETTE["brick"])
    d.text(94, 62, 150, 16, "dead step: no consumer", fs=11,
           color=PALETTE["brick"], align="center")
    row2 = []
    for k in range(6):
        x = 20 + k * 44
        cached = k == 3
        row2.append(d.node(x, 94, 34, 34, str(k + 1),
                           accent="green" if cached else "paper", fs=12,
                           fill="#d2efe7" if cached else "#eef1f4"))
    d.text(252, 56, 90, 28, "compiled, 6 steps", fs=10, color=PALETTE["ink"])
    d.edge(row1[0], row2[0], label="compile", fs=10, exit_=(0.5, 1), entry=(0.5, 0))
    d.text(59, 134, 220, 16, "cache-served after first occurrence", fs=11,
           color=PALETTE["green"], align="center")
    return d


# --------------------------------------------------------------------------- 3
def cache_flow() -> Diagram:
    """Supplement I.12: Theorem 3.7 as a decision, 520 x 214 px."""
    d = Diagram("wide_dia_2_cache_flow", 520, 214)
    xs = [10, 114, 218, 322, 426]
    call = d.node(xs[0], 36, 88, 54, "call (instruction, params)", accent="teal", fs=11)
    pure = d.node(xs[1], 36, 88, 54, "declared pure?", accent="green", fs=11)
    tenant = d.node(xs[2], 36, 88, 54, "tenant in scope?", accent="green", fs=11)
    key = d.node(xs[3], 36, 88, 54, "key in cache?", accent="green", fs=11)
    serve = d.node(xs[4], 36, 88, 54, "serve stored value", accent="green",
                   fs=11, fill="#d2efe7")
    d.edge(call, pure)
    d.edge(pure, tenant, label="yes", fs=9, color=PALETTE["green"],
           label_color=PALETTE["green"])
    d.edge(tenant, key, label="yes", fs=9, color=PALETTE["green"],
           label_color=PALETTE["green"])
    d.edge(key, serve, label="hit", fs=9, color=PALETTE["green"],
           label_color=PALETTE["green"])
    n1 = d.node(xs[1], 118, 88, 50, "never stored: execute every time",
                accent="paper", fs=10)
    n2 = d.node(xs[2], 118, 88, 50, "refuse (do not share)", accent="paper", fs=10)
    n3 = d.node(xs[3], 118, 88, 50, "execute, store with footprint",
                accent="paper", fs=10)
    for src, dst in ((pure, n1), (tenant, n2), (key, n3)):
        d.edge(src, dst, label="no", fs=9, color=PALETTE["brick"],
               label_color=PALETTE["brick"], exit_=(0.5, 1), entry=(0.5, 0))
    d.node(10, 184, 350, 24, "any write overlapping a stored footprint evicts that entry",
           accent="teal", fs=10)
    d.text(370, 182, 144, 28, "capacity / TTL cost hits, never correctness",
           fs=9, color=PALETTE["soft"])
    return d


# --------------------------------------------------------------------------- 4
def learning_pipeline() -> Diagram:
    """Supplement I.9: recorded traces to dispatched compound, 520 x 194 px."""
    d = Diagram("wide_dia_3_learning_pipeline", 520, 194)
    xs = [10, 114, 218, 322, 426]
    tr = d.node(xs[0], 30, 90, 58, "recorded traces", accent="teal", fs=11)
    mine = d.node(xs[1], 30, 90, 58, "action-signature mining", accent="teal", fs=11)
    gate = d.node(xs[2], 30, 90, 58, "developer gate", accent="amber", fs=11)
    comp = d.node(xs[3], 30, 90, 58, "signature-consistent compound", accent="green", fs=11)
    ver = d.node(xs[4], 30, 90, 58, "replay verification", accent="teal", fs=11)
    d.edge(tr, mine)
    d.edge(mine, gate)
    d.edge(gate, comp, label="approve", fs=9, color=PALETTE["amber"],
           label_color=PALETTE["amber"])
    d.edge(comp, ver)
    o1 = d.node(xs[1], 120, 90, 50, "12 templates, 95.9% of steps",
                accent="paper", fs=10)
    o2 = d.node(xs[3], 120, 90, 50, "compounds, 10.07% holdout error",
                accent="paper", fs=10)
    o3 = d.node(xs[4], 120, 90, 50, "33 of 49 eligible transfer",
                accent="paper", fs=10)
    d.edge(mine, o1, exit_=(0.5, 1), entry=(0.5, 0), end_arrow=False)
    d.edge(comp, o2, exit_=(0.5, 1), entry=(0.5, 0), end_arrow=False)
    d.edge(ver, o3, exit_=(0.5, 1), entry=(0.5, 0), end_arrow=False)
    rej = d.node(xs[2], 120, 90, 50, "not promoted: fail closed",
                 accent="brick", fs=10)
    d.edge(gate, rej, label="unverified", fs=9, color=PALETTE["brick"],
           label_color=PALETTE["brick"], exit_=(0.5, 1), entry=(0.5, 0))
    return d


# --------------------------------------------------------------------------- 5
def policies() -> Diagram:
    """Supplement I.12: the four cache policies side by side, 520 x 256 px."""
    d = Diagram("wide_dia_4_policies", 520, 256)
    cols = {"cond": (200, 100), "hits": (310, 62), "dmg": (380, 100)}
    for key, (x, w) in cols.items():
        d.text(x, 8, w, 14, {"cond": "key condition", "hits": "hits",
                             "dmg": "damage"}[key], fs=9)
    rows = [
        ("P1  tool + args", "brick", "tenant-blind", "807", "240", PALETTE["brick"]),
        ("P2  + write invalidation", "amber", "tenant-blind", "655", "61", PALETTE["ink"]),
        ("P3  semantic shape", "brick", "identity-blind", "963", "491", PALETTE["brick"]),
        ("P4  + tenant, reads only", "green", "tenant-scoped, write-evicting",
         "466 of 1,728", "0", PALETTE["green"]),
    ]
    for i, (name, tag, cond, hits, dmg, dmg_color) in enumerate(rows):
        y = 30 + i * 52
        strong = tag == "green"
        d.node(10, y, 170, 40, "", accent=tag,
               fill="#d2efe7" if strong else None)
        d.node(18, y + 10, 20, 20, "", accent=tag, fill=PALETTE[tag])
        d.text(46, y + 4, 128, 32, name, fs=10, color=PALETTE["ink"])
        d.text(cols["cond"][0], y + 4, cols["cond"][1], 32, cond, fs=10)
        d.text(cols["hits"][0], y + 4, cols["hits"][1], 32, hits, fs=10)
        d.text(cols["dmg"][0], y + 4, cols["dmg"][1], 32, dmg, fs=10,
               color=dmg_color, bold=(tag == "green"))
    d.text(10, 242, 480, 12,
           "hits per 1,000-task replay of 1,728 admitted reads; damage = cross-tenant or stale hits",
           fs=8)
    return d


# --------------------------------------------------------------------------- export
def kill_gui_instances() -> None:
    """The desktop CLI waits forever behind an open GUI instance; close any."""
    subprocess.run(["pkill", "-f", "draw.io.app"], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)


def export(d: Diagram, fmt: str, out: Path, *, scale: int = 2) -> None:
    src = SRC / f"{d.name}.drawio"
    cmd = [str(DRAWIO), "--no-sandbox", "--disable-gpu", "--export",
           "--format", fmt, "--crop", "--border", "4"]
    if fmt == "png":
        cmd += ["--scale", str(scale)]
    cmd += ["-o", str(out), str(src)]
    for attempt in (1, 2):
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=240)
            if out.exists() and out.stat().st_size > 1000:
                return
            raise RuntimeError(f"{d.name}.{fmt}: no output written")
        except (subprocess.SubprocessError, RuntimeError) as exc:
            if attempt == 2:
                raise
            kill_gui_instances()


def main() -> int:
    if not DRAWIO.exists():
        print("draw.io desktop app not found at", DRAWIO, file=sys.stderr)
        return 1
    if not shutil.which("pdftotext"):
        print("pdftotext not found; needed to verify exports", file=sys.stderr)
        return 1
    SRC.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    kill_gui_instances()
    for build in (architecture, compile_pipeline, cache_flow,
                  learning_pipeline, policies):
        d = build()
        path = d.save()
        export(d, "pdf", FIG / f"{d.name}.pdf")
        export(d, "png", FIG / f"{d.name}.png")
        print(f"  wrote {path.name} -> figures/{d.name}.pdf/.png")
    # text must actually be in the PDFs, not silently dropped by the exporter
    import re as _re
    for f in sorted(FIG.glob("*dia*.pdf")):
        txt = subprocess.run(["pdftotext", str(f), "-"], capture_output=True,
                             text=True).stdout
        n = len(_re.sub(r"\s+", " ", txt).strip())
        print(f"  {f.name}: {n} chars of text")
    return 0


if __name__ == "__main__":
    sys.exit(main())
