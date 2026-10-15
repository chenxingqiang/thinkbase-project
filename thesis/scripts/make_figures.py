#!/usr/bin/env python3
"""Generate the thesis's data figures (iteration curves, drift curves)
as TikZ/pgfplots .tex from measured results.  No hand-drawn numbers."""

import json
from pathlib import Path

THESIS = Path(__file__).resolve().parents[1]
RESULTS = THESIS / "experiments" / "results"


def coords(series):
    return " ".join(f"({i}, {v:.4f})" for i, v in enumerate(series))


def rounds_figure():
    path = RESULTS / "main_results.json"
    if not path.exists():
        return None
    curves = json.loads(path.read_text())["curves"]
    plots = []
    style = {
        "naive": "red, mark=x, dashed",
        "no_rsi": "black!55, mark=o, densely dotted",
        "conf_only": "orange, mark=triangle",
        "no_adaptive": "blue!60, mark=square",
        "gated": "green!50!black, mark=*, thick",
    }
    for variant, spec in style.items():
        if variant in curves:
            plots.append(
                f"\\addplot[{spec}] {{{coords(curves[variant]['ndcg'])}}};"
                f"\\addlegendentry{{{variant.replace('_', ' ')}}}")
    body = "\n".join(plots)
    return (
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[width=0.9\\textwidth, height=6.2cm,\n"
        "  xlabel={RSI 迭代轮次 $r$}, ylabel={NDCG@10},\n"
        "  xmin=0, xmax=4, legend style={font=\\scriptsize, at={"
        "(0.02,0.98)}, anchor=north west, draw=none},\n"
        "  legend cell align=left, tick label style={font=\\small},\n"
        "  label style={font=\\small}]\n"
        f"{body}\n"
        "\\end{axis}\n\\end{tikzpicture}\n")


def drift_figure():
    path = RESULTS / "main_results.json"
    if not path.exists():
        return None
    curves = json.loads(path.read_text())["curves"]
    plots = []
    style = {
        "naive": "red, mark=x, dashed",
        "conf_only": "orange, mark=triangle",
        "gated": "green!50!black, mark=*, thick",
    }
    for variant, spec in style.items():
        if variant in curves:
            js = [max(0.0, v) for v in curves[variant]["js"]]
            plots.append(
                f"\\addplot[{spec}] {{{coords(js)}}};"
                f"\\addlegendentry{{{variant.replace('_', ' ')}}}")
    body = "\n".join(plots)
    return (
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[width=0.9\\textwidth, height=6.2cm,\n"
        "  xlabel={RSI 迭代轮次 $r$}, ylabel={分层 JS 距离},\n"
        "  xmin=0, xmax=4, legend style={font=\\scriptsize, at={"
        "(0.98,0.98)}, anchor=north east, draw=none},\n"
        "  legend cell align=left, tick label style={font=\\small},\n"
        "  label style={font=\\small}]\n"
        f"{body}\n"
        "\\addplot[black, dotted, domain=0:4] {0.08};"
        "\\addlegendentry{预算 $\\varepsilon$}\n"
        "\\end{axis}\n\\end{tikzpicture}\n")


def main():
    figs = THESIS / "figures"
    figs.mkdir(exist_ok=True)
    for name, maker in (("rounds", rounds_figure),
                        ("drift", drift_figure)):
        tex = maker()
        if tex:
            (figs / f"{name}.tex").write_text(tex)
            print(f"wrote figures/{name}.tex")
        else:
            print(f"skipped {name}: results/main_results.json not found")


if __name__ == "__main__":
    main()
