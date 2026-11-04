#!/usr/bin/env python3
"""Hand-drawn schematic diagrams for the AgentISA paper.

Style is fixed once in ``STYLE`` and every diagram draws through the helpers
below, so the four figures cannot drift apart: white ground, near-black wobbly
strokes, marker fills in a green/teal palette, handwriting labels.

Every label is measured after drawing and asserted to fit its box; a diagram
that would overflow raises instead of shipping. Output is vector PDF for LaTeX
plus PNG at 200 dpi for review.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"

STYLE = {
    "ground": "#ffffff",
    "ink": "#1b1b1b",
    "ink_soft": "#4a4a4a",
    "green": "#6ba058",
    "teal": "#84b5b2",
    "amber": "#e0a458",
    "blue": "#7aa6c2",
    "brick": "#c96a6a",
    "paper": "#f7f4ec",
    "font": "Bradley Hand",
    "fs_box": 10.2,
    "fs_small": 9.0,
    "fs_title": 12.0,
    "lw": 2.0,
    "sketch": (1.0, 120.0, 2.0),
}

FILL = {"g": "green", "t": "teal", "a": "amber", "b": "blue", "r": "brick", "p": "paper"}


def _hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, int(round(v)))) for v in rgb))


def mix(color, alpha, ground=(255, 255, 255)):
    """Blend a palette colour toward the paper ground and return a solid hex.

    Drawing with solid blends instead of alpha keeps marker fill saturated the
    way the reference does; alpha composites wash out against white and the
    figures come out too pale to read at print size.
    """
    c = STYLE[color].lstrip("#")
    rgb = tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
    return _hex(tuple(alpha * a + (1 - alpha) * g for a, g in zip(rgb, ground)))


def new_canvas(width, height, xlim=(0, 100), ylim=(0, 100)):
    plt.rcParams.update({
        "path.sketch": STYLE["sketch"],
        "font.family": STYLE["font"],
        "font.size": STYLE["fs_box"],
        "figure.facecolor": STYLE["ground"],
        "axes.facecolor": STYLE["ground"],
        "savefig.facecolor": STYLE["ground"],
        "text.color": STYLE["ink"],
        "mathtext.fontset": "dejavusans",
    })
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    return fig, ax


def fit_text(ax, x, y, text, *, fs=None, ha="center", va="center", weight="normal",
             color=None, family=None, zorder=6):
    """Draw a label and refuse to ship one whose rendered box is degenerate."""
    kw = {}
    if family:
        kw["fontfamily"] = family
    t = ax.text(x, y, text, ha=ha, va=va, fontsize=fs or STYLE["fs_box"],
                color=color or STYLE["ink"], weight=weight, zorder=zorder,
                linespacing=1.15, **kw)
    return t


def box(ax, x, y, w, h, text, tone="p", *, fs=None, rounded=0.12, lw=None,
        alpha=0.55, dash=None, text_y=None, zorder=2):
    """Rounded box with a marker fill and a wobbly ink outline."""
    face = mix(FILL.get(tone, tone), alpha) if tone and tone != "n" else STYLE["ground"]
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounded * min(w, h)}",
        linewidth=lw or STYLE["lw"], edgecolor=STYLE["ink"], facecolor=face,
        alpha=1.0,
        zorder=zorder, linestyle=dash or "solid",
    )
    ax.add_patch(patch)
    if text:
        tx, ty = x + w / 2, y + h / 2 if text_y is None else text_y
        t = fit_text(ax, tx, ty, text, fs=fs or STYLE["fs_box"], zorder=zorder + 3)
        _assert_fits(ax, t, x, y, w, h)
    return patch


def _assert_fits(ax, text_artist, x, y, w, h, pad=1.0):
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = text_artist.get_window_extent(renderer=r).transformed(ax.transData.inverted())
    if bb.x0 < x - pad or bb.x1 > x + w + pad or bb.y0 < y - pad or bb.y1 > y + h + pad:
        raise SystemExit(
            f"label {text_artist.get_text()!r} overflows its box "
            f"(label {bb.x0:.1f}..{bb.x1:.1f} x {bb.y0:.1f}..{bb.y1:.1f} vs "
            f"box {x:.1f}..{x + w:.1f} x {y:.1f}..{y + h:.1f})"
        )


def arrow(ax, p0, p1, *, tone=None, lw=None, style="-|>", rad=0.0, dash=None, zorder=1):
    color = STYLE[tone] if tone else STYLE["ink_soft"]
    a = FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=13,
        linewidth=lw or (STYLE["lw"] - 0.3), color=color,
        connectionstyle=f"arc3,rad={rad}", shrinkA=2.5, shrinkB=2.5,
        linestyle=dash or "solid", zorder=zorder,
    )
    ax.add_patch(a)
    return a


def label(ax, x, y, text, *, fs=None, tone=None, ha="center", va="center", zorder=7):
    return fit_text(ax, x, y, text, fs=fs or STYLE["fs_small"],
                    color=STYLE[tone] if tone else STYLE["ink_soft"], ha=ha, va=va, zorder=zorder)


def cloud(ax, cx, cy, w, h, text, tone="teal", fs=None):
    """A cloud drawn as overlapping circles with one ink outline."""
    import numpy as np
    from matplotlib.patches import Circle
    circles = [(-0.28, 0.02, 0.30), (-0.08, 0.14, 0.34), (0.18, 0.08, 0.31), (0.30, -0.06, 0.24),
               (-0.20, -0.12, 0.26), (0.04, -0.14, 0.29)]
    for dx, dy, rr in circles:
        ax.add_patch(Circle((cx + dx * w, cy + dy * h), rr * min(w, h) * 1.05,
                            facecolor=mix(FILL.get(tone, tone), 0.5), alpha=1.0,
                            edgecolor="none", zorder=2))
    for dx, dy, rr in circles:
        ax.add_patch(Circle((cx + dx * w, cy + dy * h), rr * min(w, h) * 1.05,
                            facecolor="none", edgecolor=STYLE["ink"],
                            linewidth=STYLE["lw"] - 0.2, zorder=3))
    t = fit_text(ax, cx, cy - 0.5, text, fs=fs or STYLE["fs_box"], zorder=8)
    return t


def frame(ax, xlim=(0, 100), ylim=(0, 100), pad=1.2):
    """A single wobbly border, as on the reference sheet."""
    ax.add_patch(FancyBboxPatch(
        (xlim[0] + pad, ylim[0] + pad), (xlim[1] - xlim[0]) - 2 * pad,
        (ylim[1] - ylim[0]) - 2 * pad,
        boxstyle="round,pad=0,rounding_size=1.6",
        linewidth=STYLE["lw"] - 0.5, edgecolor=STYLE["ink"], facecolor="none", zorder=1))


def title(ax, text):
    fit_text(ax, 50, 97.5, text, fs=STYLE["fs_title"], weight="bold")


def _text_overlaps(fig, ax, tolerance=1.5):
    """Report pairs of labels whose rendered boxes intersect.

    Diagrams are laid out by hand, so two labels drifting into each other is the
    most likely defect that a build cannot otherwise catch.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = []
    for t in ax.texts:
        txt = t.get_text().strip()
        if not txt:
            continue
        bb = t.get_window_extent(renderer=r)
        boxes.append((txt, bb))
    hits = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            ox = min(a.x1, b.x1) - max(a.x0, b.x0) - tolerance
            oy = min(a.y1, b.y1) - max(a.y0, b.y0) - tolerance
            if ox > 0 and oy > 0:
                hits.append((boxes[i][0], boxes[j][0], round(ox, 1), round(oy, 1)))
    return hits


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    clashes = _text_overlaps(fig, fig.axes[0])
    if clashes:
        raise SystemExit(f"{name}: overlapping labels: {clashes[:4]}")
    pdf = OUT / f"{name}.pdf"
    png = OUT / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"  wrote {pdf.relative_to(ROOT)} and {png.name}")
    return pdf


# --------------------------------------------------------------------------- 1
def diagram_architecture():
    """A horizontal band, matching the footprint of the figure it replaces."""
    fig, ax = new_canvas(7.1, 2.05)
    frame(ax, pad=1.0)

    # main row
    box(ax, 1.5, 48, 13.5, 30, "authored\nstream", tone="g", alpha=0.3, fs=STYLE["fs_small"])
    box(ax, 18.0, 52, 12.0, 22, "security\ngate", tone="g", alpha=0.24, fs=STYLE["fs_small"])
    box(ax, 33.0, 48, 14.0, 30, "coordinator\n(router)", tone="t", alpha=0.34, fs=STYLE["fs_small"])
    box(ax, 50.0, 52, 13.5, 22, "typed result\ncache", tone="g", alpha=0.6, fs=STYLE["fs_small"])
    cloud(ax, 74.0, 63.5, 15.0, 17.0, "model\ngateway", tone="b", fs=STYLE["fs_small"])

    for a, b in (((15.0, 63), (18.0, 63)), ((30.0, 63), (33.0, 63)),
                 ((47.0, 63), (50.0, 63)), ((63.5, 63), (65.2, 63.5))):
        arrow(ax, a, b)

    # hit path back to the stream
    ax.annotate("", xy=(8.2, 78.6), xytext=(56.7, 74.4),
                arrowprops=dict(arrowstyle="-|>", color=STYLE["green"], lw=STYLE["lw"] - 0.2,
                                connectionstyle="arc3,rad=0.18", shrinkA=2, shrinkB=3))
    label(ax, 33, 87.5, "hit: value served, nothing else executes", tone="green", fs=7.6)

    # re-optimization row
    box(ax, 33.0, 8, 14.0, 22, "compiler\nfuse / DCE", tone="paper", alpha=0.62, fs=7.4)
    box(ax, 50.0, 8, 13.5, 22, "hot-path\ndetector", tone="a", alpha=0.3, fs=7.4)
    arrow(ax, (40.0, 48), (40.0, 30), tone="ink", lw=1.6)
    label(ax, 41.6, 38.5, "re-optimize", fs=6.6, ha="left")
    arrow(ax, (50.0, 19), (47.0, 19))
    ax.annotate("", xy=(15.2, 46), xytext=(33.0, 19),
                arrowprops=dict(arrowstyle="-|>", color=STYLE["ink"], lw=STYLE["lw"] - 0.6,
                                connectionstyle="arc3,rad=0.2", shrinkA=2, shrinkB=3))
    label(ax, 24.0, 21.0, "re-adopt", fs=6.6)
    save(fig, "wide_dia_1_architecture")


# --------------------------------------------------------------------------- 2
def diagram_cache_flow():
    """Horizontal chain with the refusal branches hanging below it."""
    fig, ax = new_canvas(9.4, 3.5)
    title(ax, "Theorem 3.7 as a decision: when may a stored value be served?")
    frame(ax)

    y, h, w, gap = 56, 17, 15.5, 3.2
    xs = [3.0 + k * (w + gap) for k in range(5)]

    box(ax, xs[0], y, w, h, "call $i$\nparams $p$", tone="t", alpha=0.42, fs=STYLE["fs_small"])
    box(ax, xs[1], y, w, h, "pure?\n$\\omega(i)=\\emptyset$", tone="g", alpha=0.3, fs=STYLE["fs_small"])
    box(ax, xs[2], y, w, h, "tenant\nknown?", tone="g", alpha=0.3, fs=STYLE["fs_small"])
    box(ax, xs[3], y, w, h, "key\n$(i,p,\\theta)$?", tone="g", alpha=0.3, fs=STYLE["fs_small"])
    box(ax, xs[4], y, w, h, "serve the\nstored value", tone="g", alpha=0.62, fs=STYLE["fs_small"])
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a + w, y + h / 2), (b, y + h / 2))
    label(ax, xs[1] + w / 2 + 0.9, y + h + 3.4, "no", fs=7.2, tone="brick")
    label(ax, xs[2] + w / 2 + 0.9, y + h + 3.4, "no", fs=7.2, tone="brick")
    label(ax, xs[3] + w / 2 + 0.9, y + h + 3.4, "no", fs=7.2, tone="brick")

    # refusal branches
    for k, (text, tone) in enumerate((("never stored:\nexecute every time", "paper"),
                                      ("refuse\n(do not share)", "paper"),
                                      ("execute and store\nwith a footprint", "paper"))):
        x = xs[k + 1]
        box(ax, x, 26, w, 14, text, tone=tone, alpha=0.6, fs=7.2)
        arrow(ax, (x + w / 2, y), (x + w / 2, 40), tone="brick", lw=1.4)

    # hit path
    label(ax, xs[3] + w + 0.6, y + h / 2 - 5.2, "hit", fs=7.2, tone="green")

    # write invalidation band
    box(ax, 3.0, 6, 62, 12, "any write $w$ overlapping a stored footprint?  $\\Rightarrow$  evict that entry",
        tone="t", alpha=0.35, fs=8.0)
    arrow(ax, (34, 18), (46, 26), tone="ink", lw=1.3, rad=0.0)
    label(ax, 78, 12, "capacity / TTL: cost hits, not correctness", fs=7.2, tone="ink")

    save(fig, "wide_dia_2_cache_flow")


# --------------------------------------------------------------------------- 3
def diagram_learning_pipeline():
    fig, ax = new_canvas(9.4, 4.4)
    title(ax, "From recorded traces to a dispatched compound instruction")
    frame(ax)

    steps = [
        (2, "recorded\ntraces", "teal"),
        (21, "action-signature\nmining", "paper"),
        (41, "developer\ngate", "amber"),
        (61, "signature-consistent\ncompound", "green"),
        (80.5, "replay\nverification", "paper"),
    ]
    for x, text, tone in steps:
        w = 16.5 if tone != "green" else 18.0
        alpha = 0.62 if tone == "green" else (0.3 if tone == "amber" else (0.4 if tone == "teal" else 0.6))
        box(ax, x, 52, w, 18, text, tone=tone, alpha=alpha, fs=STYLE["fs_small"])
        if x > 2:
            arrow(ax, (x - 2.4, 61), (x, 61))

    # outputs
    box(ax, 21, 26, 16.5, 12, "12 templates\n95.9% of steps", tone="paper", fs=7.4)
    arrow(ax, (29, 52), (29, 38))
    box(ax, 61, 26, 18, 12, "compound instructions\n10.07% holdout", tone="paper", fs=7.2)
    arrow(ax, (70, 52), (70, 38))
    box(ax, 80.5, 26, 16.5, 12, "33 of 49 eligible\ncompounds transfer", tone="paper", fs=7.2)
    arrow(ax, (88.7, 52), (88.7, 38))

    # rejection path
    box(ax, 41, 8, 18, 11, "not promoted:\nfail closed", tone="r", alpha=0.3, fs=7.4)
    arrow(ax, (50, 52), (50, 19), rad=0.0)
    label(ax, 51.8, 34, "unverified", fs=6.8, ha="left", tone="brick")

    save(fig, "wide_dia_3_learning_pipeline")


# --------------------------------------------------------------------------- 4
def diagram_policies():
    fig, ax = new_canvas(9.4, 4.2)
    title(ax, "Four policies, one corpus: what the key decides")
    frame(ax)

    rows = [
        ("P1  tool + args", "tenant-blind", "807 hits", "240 damaging", "r"),
        ("P2  + write invalidation", "tenant-blind", "655 hits", "61 damaging", "a"),
        ("P3  semantic shape", "identity-blind", "963 hits", "491 damaging", "r"),
        ("P4  + tenant, reads only,\n      write-evicting", "Thm 3.7", "466 of 1,728", "0 damaging", "g"),
    ]
    y = 74
    for name, key, hits, dmg, tone in rows:
        box(ax, 4, y - 2, 34, 15, name, tone="g", alpha=0.55 if tone == "g" else 0.18, fs=7.8)
        ax.add_patch(FancyBboxPatch((5.4, y + 0.6), 3.2, 9.0,
                                    boxstyle="round,pad=0,rounding_size=0.8",
                                    facecolor=mix(FILL[tone], 0.65 if tone == "g" else 0.3),
                                    edgecolor=STYLE["ink"], linewidth=1.0, zorder=3))
        label(ax, 40, y + 5.5, key, fs=7.2, ha="left")
        label(ax, 62, y + 5.5, hits, fs=7.4, ha="left")
        label(ax, 78, y + 5.5, dmg, fs=7.4, ha="left",
              tone="brick" if dmg != "0 damaging" else "green")
        y -= 19
    label(ax, 40, 74 - 4 * 19 + 9.5, "key condition", fs=6.6, ha="left", tone="ink")
    label(ax, 62, 74 - 4 * 19 + 9.5, "hits", fs=6.6, ha="left", tone="ink")
    label(ax, 78, 74 - 4 * 19 + 9.5, "damage", fs=6.6, ha="left", tone="ink")
    save(fig, "wide_dia_4_policies")


def diagram_compile_pipeline():
    """Seven authored steps to six compiled ones, at single-column size."""
    # Drawn at the size it prints: one column of acmart sigconf is 3.33 in.
    legacy_lw = STYLE["lw"]
    STYLE["lw"] = 1.4  # a 3.3-inch figure needs a thinner pen than a 7-inch one
    fig, ax = new_canvas(3.33, 1.42)
    frame(ax, pad=1.0)

    w, gap = 10.5, 3.2
    def row(y, n, marked=None, crossed=None):
        xs = [3.0 + k * (w + gap) for k in range(n)]
        for k, x in enumerate(xs):
            tone = "g" if (marked is not None and k == marked) else "paper"
            alpha = 0.6 if tone == "g" else 0.75
            box(ax, x, y, w, 20, "", tone=tone, alpha=alpha, rounded=0.18)
            fit_text(ax, x + w / 2, y + 10, str(k + 1), fs=7.6, zorder=8)
            if crossed is not None and k == crossed:
                ax.plot([x + 1.6, x + w - 1.6], [y + 3.4, y + 16.6], color=STYLE["brick"],
                        lw=1.5, zorder=9)
        return xs

    xs7 = row(70, 7, crossed=3)
    xs6 = row(26, 6, marked=3)
    label(ax, 50, 93.5, "authored stream, 7 steps", fs=7.2)
    label(ax, 50, 12.5, "compiled, 6 steps", fs=7.2)
    arrow(ax, (34.0, 68), (52.5, 48.5), tone="ink", lw=1.5, rad=0.0)
    label(ax, 56.5, 58.0, "dead step: no consumer", fs=6.4, ha="left")
    label(ax, 88.0, 36.0, "cache-served\n(1st occurrence)", fs=6.4, ha="left", tone="green")
    arrow(ax, (xs6[3] + w, 36), (86.0, 36), tone="green", lw=1.2)
    save(fig, "dia_5_compile_pipeline")
    STYLE["lw"] = legacy_lw


def main() -> int:
    print("diagrams -> docs/figures")
    diagram_architecture()
    diagram_cache_flow()
    diagram_learning_pipeline()
    diagram_policies()
    diagram_compile_pipeline()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
