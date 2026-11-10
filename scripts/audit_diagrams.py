#!/usr/bin/env python3
"""Geometric audit of the .drawio diagram sources.

The figures are authored blind (no image preview in this environment), so this
script is the eye: it parses each .drawio file and reports defects that make a
diagram look wrong even when every label is present and every export succeeds.

Checks, in order of how badly each one damages a figure:
  1. duplicate   -- two vertices at the same rectangle (stacked boxes)
  2. overlap     -- two vertices whose rectangles intersect, other than a tag
                    square inside its own row panel
  3. textflow    -- a label that cannot fit its box at Helvetica metrics
                    (estimated width wraps to more lines than the box is tall)
  4. crossing    -- an edge segment passing through a vertex it does not
                    connect (arrows must route around, never through)
  5. labelhit    -- an edge label rectangle landing on a non-incident vertex

Exit code 1 if anything is found; the messages name the cells and coordinates
so the fix is a one-line layout change in make_diagrams_drawio.py.
"""

from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "diagrams"

# Helvetica average advance widths, measured per font size, as a fraction of
# the em: mixed-case labels run ~0.55, digit-heavy ~0.5. Use 0.56 to be strict.
CHAR_W = 0.56
LINE_H = 1.30


def cells_of(path: Path):
    tree = ET.parse(path)
    model = tree.getroot().find(".//mxGraphModel")
    page_w, page_h = float(model.get("pageWidth")), float(model.get("pageHeight"))
    verts, edges = [], []
    for c in model.find("root"):
        if c.tag != "mxCell" or c.get("vertex") != "1" and c.get("edge") != "1":
            continue
        value = c.get("value") or ""
        style = c.get("style") or ""
        fs = 12.0
        for part in style.split(";"):
            if part.startswith("fontSize="):
                fs = float(part.split("=")[1])
        g = c.find("mxGeometry")
        if c.get("vertex") == "1" and g is not None:
            verts.append({
                "id": c.get("id"), "label": _plain(value),
                "x": float(g.get("x", 0)), "y": float(g.get("y", 0)),
                "w": float(g.get("width", 0)), "h": float(g.get("height", 0)),
                "fs": fs, "no_fill": "fillColor=none" in style,
            })
        elif c.get("edge") == "1":
            pts = []
            if g is not None:
                for arr in g.findall("Array"):
                    for p in arr.findall("mxPoint"):
                        pts.append((float(p.get("x")), float(p.get("y"))))
            edges.append({
                "id": c.get("id"), "label": _plain(value), "fs": fs,
                "source": c.get("source"), "target": c.get("target"),
                "points": pts,
            })
    return page_w, page_h, verts, edges


def _plain(v: str) -> str:
    return (v.replace("&#10;", " ").replace("&lt;", "<").replace("&gt;", ">")
             .replace("&amp;", "&").replace("&quot;", '"'))


def rect(v):
    return (v["x"], v["y"], v["x"] + v["w"], v["y"] + v["h"])


def inter(a, b):
    ox = min(a[2], b[2]) - max(a[0], b[0])
    oy = min(a[3], b[3]) - max(a[1], b[1])
    return ox, oy


def contained(inner, outer, tol=1.0):
    return (inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol
            and inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol)


def edge_polyline(e, verts_by_id):
    """Anchor-to-anchor polyline: source anchor, waypoints, target anchor."""
    anchors = []
    for ref in (e["source"], e["target"]):
        v = verts_by_id.get(ref)
        anchors.append((v["x"] + v["w"] / 2, v["y"] + v["h"] / 2) if v else None)
    if not all(anchors):
        return []
    return [anchors[0]] + list(e["points"]) + [anchors[1]]


def seg_rect_hit(p, q, r):
    """True if segment pq intersects axis-aligned rect r (inclusive edges)."""
    x0, x1 = sorted((p[0], q[0]))
    y0, y1 = sorted((p[1], q[1]))
    return not (x1 < r[0] or x0 > r[2] or y1 < r[1] or y0 > r[3])


def audit(path: Path) -> list[str]:
    page_w, page_h, verts, edges = cells_of(path)
    by_id = {v["id"]: v for v in verts}
    out = []

    # 1. duplicates (same rectangle twice, excluding zero-size)
    seen = {}
    for v in verts:
        if v["w"] < 2 or v["h"] < 2:
            continue
        key = (round(v["x"], 1), round(v["y"], 1), round(v["w"], 1), round(v["h"], 1))
        if key in seen:
            out.append(f"duplicate: {v['label']!r} at {key} (also cell {seen[key]})")
        seen[key] = v["id"]

    # 2. vertex overlaps (allow a tag square fully inside a panel)
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            a, b = verts[i], verts[j]
            ra, rb = rect(a), rect(b)
            ox, oy = inter(ra, rb)
            if ox > 1 and oy > 1:
                if contained(rb, ra) or contained(ra, rb):
                    continue  # tag inside its row panel
                out.append(f"overlap: {a['label']!r} x {b['label']!r} "
                           f"intersect {ox:.0f}x{oy:.0f}px")

    # 3. text does not fit its box
    for v in verts:
        if not v["label"] or v["w"] < 2:
            continue
        words = v["label"].split()
        usable = max(v["w"] - 10, 4)
        # greedy wrap at Helvetica metrics
        lines, cur = 1, 0.0
        space = v["fs"] * CHAR_W * 0.5
        for w_ in words:
            ww = len(w_) * v["fs"] * CHAR_W
            add = ww if cur == 0 else space + ww
            if cur + add > usable and cur > 0:
                lines += 1
                cur = ww
            else:
                cur += add
        need_h = lines * v["fs"] * LINE_H + 2
        if need_h > v["h"] + 1:
            out.append(f"textflow: {v['label']!r} needs ~{lines} lines "
                       f"({need_h:.0f}px) in {v['w']:.0f}x{v['h']:.0f}px box")

    # 4. edges through non-incident vertices
    for e in edges:
        poly = edge_polyline(e, by_id)
        if len(poly) < 2:
            continue
        for v in verts:
            if v["id"] in (e["source"], e["target"]) or v["w"] < 2:
                continue
            r = rect(v)
            for p, q in zip(poly, poly[1:]):
                if seg_rect_hit(p, q, r):
                    out.append(f"crossing: edge {e['label']!r} passes through "
                               f"{v['label']!r} near ({p[0]:.0f},{p[1]:.0f})")
                    break

    # 5. edge labels on non-incident vertices
    for e in edges:
        if not e["label"]:
            continue
        poly = edge_polyline(e, by_id)
        if len(poly) < 2:
            continue
        total = sum(math.dist(p, q) for p, q in zip(poly, poly[1:]))
        walked, mid = 0.0, None
        for p, q in zip(poly, poly[1:]):
            seg = math.dist(p, q)
            if walked + seg >= total / 2:
                t = (total / 2 - walked) / seg
                mid = (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)
                break
            walked += seg
        if mid is None:
            continue
        lw = len(e["label"]) * e["fs"] * CHAR_W + 8
        lh = e["fs"] * LINE_H + 4
        lr = (mid[0] - lw / 2, mid[1] - lh / 2, mid[0] + lw / 2, mid[1] + lh / 2)
        for v in verts:
            if v["id"] in (e["source"], e["target"]) or v["w"] < 2:
                continue
            ox, oy = inter(lr, rect(v))
            if ox > 2 and oy > 2:
                out.append(f"labelhit: label {e['label']!r} lands on "
                           f"{v['label']!r} ({ox:.0f}x{oy:.0f}px)")

    # 6. content outside the page
    for v in verts:
        r = rect(v)
        if r[0] < -1 or r[1] < -1 or r[2] > page_w + 1 or r[3] > page_h + 1:
            out.append(f"offpage: {v['label']!r} at {r} vs page {page_w:.0f}x{page_h:.0f}")
    return out


def main() -> int:
    bad = 0
    for f in sorted(SRC.glob("*.drawio")):
        findings = audit(f)
        status = "clean" if not findings else f"{len(findings)} defect(s)"
        print(f"{f.name}: {status}")
        for line in findings:
            print(f"  - {line}")
            bad += 1
    print("AUDIT CLEAN" if not bad else f"AUDIT FAILED: {bad} defect(s)")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
