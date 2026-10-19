#!/usr/bin/env python3
"""Build the LaTeX + PDF version of the AgentISA paper.

Pipeline: docs/AgentISA.md -> (preprocess) -> pandoc -> docs/AgentISA.tex
-> latexmk/xelatex -> docs/AgentISA.pdf

Preprocessing makes the markdown losslessly compilable:
* strips manual section numbers (pandoc --number-sections renumbers
  identically, keeping "Section 5.2"-style cross-references valid);
* moves Abstract into pandoc metadata and marks front-matter sections
  (Keywords / Research Highlights / CCS Concepts / Funding Information)
  and Artifact Availability as unnumbered;
* converts the References section into a raw ``thebibliography`` block
  whose \\bibitem order matches the literal [n] citations in the text;
* wraps text-mode Unicode math symbols (± -> $\\pm$ etc.) so xelatex
  never needs a missing text glyph; code blocks are left untouched and
  are set in DejaVu Sans Mono, which covers them.

Usage:  python3 scripts/build_paper.py
Outputs: docs/AgentISA.tex, docs/AgentISA.pdf
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SRC = DOCS / "AgentISA.md"
BUILD = DOCS / ".paper_build"
TEX_OUT = DOCS / "AgentISA.tex"
PDF_OUT = DOCS / "AgentISA.pdf"

# TODO: replace with the real author list before submission.
AUTHOR = "Anonymous Author(s)"
TITLE = "AgentISA: Formal Instruction Set Design for LLM Agents"

FRONT_UNNUMBERED = {
    "Keywords", "Research Highlights", "CCS Concepts", "Funding Information",
}

# Text-mode Unicode -> inline math.  (Math regions are pure ASCII in the
# source; code blocks are excluded from replacement.)
UNICODE_TO_MATH = {
    "±": r"$\pm$", "×": r"$\times$",
    "→": r"$\rightarrow$", "←": r"$\leftarrow$",
    "∎": r"$\blacksquare$", "⊢": r"$\vdash$",
    "⟹": r"$\Longrightarrow$", "≠": r"$\neq$",
    "∈": r"$\in$", "∞": r"$\infty$", "≡": r"$\equiv$",
    "∅": r"$\emptyset$", "∀": r"$\forall$", "∃": r"$\exists$",
    "∧": r"$\wedge$", "∨": r"$\vee$",
    "δ": r"$\delta$", "θ": r"$\theta$", "σ": r"$\sigma$", "τ": r"$\tau$",
    "β": r"$\beta$", "α": r"$\alpha$", "γ": r"$\gamma$",
    "ε": r"$\varepsilon$", "λ": r"$\lambda$", "μ": r"$\mu$",
    "φ": r"$\varphi$", "ψ": r"$\psi$", "Σ": r"$\Sigma$", "Π": r"$\Pi$",
    "₀": r"$_{0}$", "₁": r"$_{1}$", "₂": r"$_{2}$",
    "ₙ": r"$_{n}$", "ᵢ": r"$_{i}$",
}


def space_math_edges(part: str) -> str:
    """Guard pandoc's tex_math_dollars rule: a closing $ must not be
    followed by a digit.  When a converted $\\pm$-style wrapper is glued
    to a digit (``59.5%$\\pm$0.3%``), pandoc refuses the pairing, demotes
    the dollars to literal ``\\$``, and swallows surrounding text into
    math spans where ``%`` turns into a line-eating LaTeX comment -- this
    silently corrupted the PDF abstract as ``59.5%$±0.3±0.2...`` with
    every headline number between the ±values deleted.  Insert protective
    spaces whenever a converted wrapper is glued to a digit on either
    edge (also normalises typography to ``59.5% ± 0.3%``)."""
    part = re.sub(r"(\$\\[A-Za-z]+\$)(?=[0-9])", r"\1 ", part)
    part = re.sub(r"(?<=[0-9%])(\$\\[A-Za-z]+\$)", r" \1", part)
    return part


def replace_text_unicode(md: str) -> str:
    """Apply UNICODE_TO_MATH outside fenced code blocks."""
    parts = re.split(r"(^```.*?$)", md, flags=re.M | re.S)
    out, in_code = [], False
    for part in parts:
        if part.startswith("```"):
            in_code = not in_code
            out.append(part)
            continue
        if in_code:
            out.append(part)
            continue
        # Inline code spans (`...`) are typeset verbatim, so Unicode math
        # inside them must stay literal (rendered by the mono font) rather
        # than being wrapped in $...$, which would display as literal
        # "$\neq$" text in the PDF.
        segs = re.split(r"(`[^`\n]*`)", part)
        for j, seg in enumerate(segs):
            if seg.startswith("`"):
                continue
            for ch, tex in UNICODE_TO_MATH.items():
                seg = seg.replace(ch, tex)
            segs[j] = space_math_edges(seg)
        out.append("".join(segs))
    return "".join(out)


def escape_bib(text: str) -> str:
    for ch in ("%", "&", "#", "_"):
        text = text.replace(ch, "\\" + ch)
    return text


def make_bibliography(body: str) -> str:
    entries = [e.strip() for e in re.split(r"\n\s*\n", body) if e.strip()]
    lines = ["```{=latex}", r"\begin{thebibliography}{99}"]
    for entry in entries:
        m = re.match(r"\[(\d+)\]\s*(.+)", entry, re.S)
        if not m:
            raise ValueError(f"unparsable reference entry: {entry[:60]!r}")
        content = " ".join(m.group(2).split())
        lines.append(rf"\bibitem{{ref{m.group(1)}}} {escape_bib(content)}")
    lines.append(r"\end{thebibliography}")
    lines.append("```")
    return "\n".join(lines)


def build_body_md(text: str) -> tuple[str, str]:
    """Return (metadata_yaml, body_markdown)."""
    h1s = list(re.finditer(r"^# (.+)$", text, re.M))
    sections = []
    for i, m in enumerate(h1s):
        end = h1s[i + 1].start() if i + 1 < len(h1s) else len(text)
        sections.append((m.group(1).strip(), text[m.end():end].strip("\n")))

    abstract = ""
    body_parts: list[str] = []
    appendix_inserted = False

    for title, body in sections:
        if title == TITLE:
            continue
        if title == "Abstract":
            abstract = body
            continue
        if title in FRONT_UNNUMBERED:
            body_parts.append(f"# {title} {{.unnumbered}}\n\n{body}")
            continue
        if title == "References":
            body_parts.append(make_bibliography(body))
            continue
        m = re.match(r"^(\d+)\. (.+)$", title)
        if m:  # numbered top-level section: drop the manual number
            body_parts.append(f"# {m.group(2)}\n\n{body}")
            continue
        m = re.match(r"^Appendix ([A-Z]): (.+)$", title)
        if m:
            if not appendix_inserted:
                body_parts.append("```{=latex}\n\\appendix\n```\n")
                appendix_inserted = True
            body_parts.append(f"# {m.group(2)}\n\n{body}")
            continue
        if title == "Artifact Availability":
            body_parts.append(f"# {title} {{.unnumbered}}\n\n{body}")
            continue
        raise ValueError(f"unhandled top-level section: {title!r}")

    if not abstract:
        raise ValueError("Abstract section not found")

    # Subheading numbers ("## 2.1 Foo", "### 3.1.1 Bar", "## A.1 Baz",
    # "## C.4 Qux") are stripped; --number-sections reproduces them.
    body_md = "\n\n".join(body_parts)
    body_md = re.sub(r"^(##+) (\d+(?:\.\d+)*) ", r"\1 ", body_md, flags=re.M)
    body_md = re.sub(r"^(##+) [A-Z]\.\d+ ", r"\1 ", body_md, flags=re.M)

    body_md = replace_text_unicode(body_md)
    abstract_tex_md = replace_text_unicode(abstract)

    # A closing math $ glued to a digit makes pandoc mispair the dollars
    # and silently ship a corrupted PDF (see space_math_edges).  Fail the
    # build loudly instead of discovering it in the rendered abstract.
    for _name, _text in (("body", body_md), ("abstract", abstract_tex_md)):
        _m = re.search(r"\$\\[A-Za-z]+\$[0-9]", _text)
        if _m:
            raise SystemExit(
                f"FATAL: {_name}: math dollar glued to digit would "
                f"mispair pandoc math: {_m.group(0)!r}"
            )

    yaml = (
        "---\n"
        f'title: "{TITLE}"\n'
        f'author: "{AUTHOR}"\n'
        'date: ""\n'
        "abstract: |\n"
        + "".join(f"  {line}\n" for line in abstract_tex_md.splitlines())
        + "---\n"
    )
    return yaml, body_md


HEADER_TEX = r"""\usepackage{etoolbox}
\usepackage[htt]{hyphenat}
\AtBeginEnvironment{longtable}{\small\setlength{\tabcolsep}{4pt}}
\emergencystretch=3em
\setmonofont{DejaVuSansMono.ttf}[Scale=0.86]
% Math macros used by the paper that standard LaTeX lacks:
\DeclareMathOperator*{\argmin}{arg\,min}
\newcommand{\llbracket}{[\![}
\newcommand{\rrbracket}{]\!]}
"""


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kw)


def main() -> int:
    if not SRC.exists():
        print(f"source not found: {SRC}", file=sys.stderr)
        return 1
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir()

    yaml, body_md = build_body_md(SRC.read_text(encoding="utf-8"))
    (BUILD / "meta.yaml").write_text(yaml, encoding="utf-8")
    (BUILD / "body.md").write_text(body_md, encoding="utf-8")
    (BUILD / "header.tex").write_text(HEADER_TEX, encoding="utf-8")

    tex = run(
        [
            "pandoc", str(BUILD / "meta.yaml"), str(BUILD / "body.md"),
            "-o", str(BUILD / "AgentISA.tex"),
            "--standalone", "--number-sections", "--top-level-division=section",
            "-V", "documentclass=article", "-V", "fontsize=11pt",
            "-V", "geometry:margin=1in",
            "-V", "colorlinks=true", "-V", "linkcolor=blue",
            "-V", "urlcolor=blue",
            "-H", str(BUILD / "header.tex"),
        ],
        capture_output=True, text=True,
    )
    if tex.returncode != 0:
        print(tex.stdout, tex.stderr, file=sys.stderr)
        return 1

    pdf = run(
        [
            "latexmk", "-xelatex", "-interaction=nonstopmode",
            "-halt-on-error", str(BUILD / "AgentISA.tex"),
        ],
        cwd=BUILD, capture_output=True, text=True,
    )
    log = BUILD / "AgentISA.log"
    if pdf.returncode != 0 or not (BUILD / "AgentISA.pdf").exists():
        print(pdf.stdout[-4000:], pdf.stderr[-2000:], file=sys.stderr)
        return 1

    # Report warnings that matter.
    if log.exists():
        problems = [
            ln for ln in log.read_text(errors="replace").splitlines()
            if ("Missing character" in ln or "Overfull \\hbox" in ln
                or "Missing $ inserted" in ln
                or "Undefined control sequence" in ln)
        ]
        for ln in problems[:15]:
            print("WARN:", ln.strip())

    shutil.copy(BUILD / "AgentISA.tex", TEX_OUT)
    shutil.copy(BUILD / "AgentISA.pdf", PDF_OUT)
    pages = pdf.stdout.count("Output written on")
    print(f"\nOK: {TEX_OUT}\nOK: {PDF_OUT} (latexmk output lines scanned: {pages>0})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
