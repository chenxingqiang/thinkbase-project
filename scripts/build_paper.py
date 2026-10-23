#!/usr/bin/env python3
"""Build the LaTeX + PDF versions of the AgentISA paper and its supplement.

Pipeline per target:
  docs/AgentISA.md            -> (preprocess) -> pandoc -> acmart (sigconf,
                                 anonymous, two-column) -> post-process
                                 (abstract/keywords before \maketitle;
                                 longtable -> table/table*) -> latexmk/xelatex
                                 -> docs/AgentISA.tex / AgentISA.pdf
  docs/AgentISA_supplementary.md -> same preprocessing -> pandoc -> article
                                 (single column) -> latexmk/xelatex
                                 -> docs/AgentISA_supplementary.tex / .pdf

Preprocessing (shared) makes the markdown losslessly compilable:
* strips manual section numbers (pandoc --number-sections renumbers
  identically, keeping "Section 5.2"-style cross-references valid);
* moves Abstract into pandoc metadata and marks front-matter sections
  as unnumbered;
* main target: extracts Keywords and CCS Concepts for acmart metadata
  (\keywords, \CCSDesc) and drops them from the body;
* converts the References section into a raw ``thebibliography`` block
  whose \\bibitem order matches the literal [n] citations in the text;
* wraps text-mode Unicode math symbols (± -> $\\pm$ etc.) so xelatex
  never needs a missing text glyph; code blocks are left untouched and
  are set in DejaVu Sans Mono, which covers them.

Two-column post-processing (main target only):
* pandoc emits pipe tables as longtable, which cannot live in a
  two-column layout.  Each longtable is converted to a proper float:
  column specs without \\columnwidth become single-column ``table``,
  specs with fractional \\columnwidth widths become full-width
  ``table*`` (with \\columnwidth promoted to \\textwidth);
* pandoc's default template places \\maketitle before the abstract;
  acmart requires abstract + \\keywords before \\maketitle.

Usage:  python3 scripts/build_paper.py [main|supp|all]
Outputs: docs/AgentISA.tex, docs/AgentISA.pdf,
         docs/AgentISA_supplementary.tex, docs/AgentISA_supplementary.pdf
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
BUILD = ROOT / "docs" / ".paper_build"
FIGURES = DOCS / "figures"
TEX_OUT = DOCS / "AgentISA.tex"
PDF_OUT = DOCS / "AgentISA.pdf"
SUPP_TEX_OUT = DOCS / "AgentISA_supplementary.tex"
SUPP_PDF_OUT = DOCS / "AgentISA_supplementary.pdf"

# TODO: replace with the real author list before submission.
AUTHOR = "Anonymous Author(s)"
TITLE = "AgentISA: Formal Instruction Set Design for LLM Agents"
SUPP_TITLE = ("Supplementary Material for AgentISA: "
              "Formal Instruction Set Design for LLM Agents")

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


def build_body_md(text: str, *, title: str, extract_acmart_meta: bool):
    """Return (metadata_yaml, body_markdown, keywords, ccs_descriptors)."""
    h1s = list(re.finditer(r"^# (.+)$", text, re.M))
    sections = []
    for i, m in enumerate(h1s):
        end = h1s[i + 1].start() if i + 1 < len(h1s) else len(text)
        sections.append((m.group(1).strip(), text[m.end():end].strip("\n")))

    abstract = ""
    keywords: list[str] = []
    ccs: list[str] = []
    body_parts: list[str] = []
    appendix_inserted = False

    for sec_title, body in sections:
        if sec_title == title:
            continue
        if sec_title == "Abstract":
            abstract = body
            continue
        if extract_acmart_meta and sec_title == "Keywords":
            # "- Term" bullet list -> acmart \keywords
            keywords = [re.sub(r"^-\s*", "", ln).strip()
                        for ln in body.splitlines()
                        if ln.strip().startswith("-")]
            continue
        if extract_acmart_meta and sec_title == "CCS Concepts":
            # "- **Parent → Child**: gloss" -> acmart \CCSDesc descriptors
            for ln in body.splitlines():
                m = re.match(r"-\s*\*\*(.+?)\s*→\s*(.+?)\*\*(.*)$", ln)
                if m:
                    parent, child = m.group(1).strip(), m.group(2).strip()
                    ccs.append(
                        parent.replace("~", r"\textasciitilde{}")
                        + "~\\textgreater{}" + child)
            continue
        if sec_title in FRONT_UNNUMBERED:
            body_parts.append(f"# {sec_title} {{.unnumbered}}\n\n{body}")
            continue
        if sec_title == "References":
            body_parts.append(make_bibliography(body))
            continue
        m = re.match(r"^(\d+)\. (.+)$", sec_title)
        if m:  # numbered top-level section: drop the manual number
            body_parts.append(f"# {m.group(2)}\n\n{body}")
            continue
        m = re.match(r"^Appendix ([A-Z]): (.+)$", sec_title)
        if m:
            if not appendix_inserted:
                body_parts.append("```{=latex}\n\\appendix\n```\n")
                appendix_inserted = True
            body_parts.append(f"# {m.group(2)}\n\n{body}")
            continue
        if sec_title == "Artifact Availability":
            body_parts.append(f"# {sec_title} {{.unnumbered}}\n\n{body}")
            continue
        raise ValueError(f"unhandled top-level section: {sec_title!r}")

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
        f'title: "{title}"\n'
        f'author: "{AUTHOR}"\n'
        'date: ""\n'
        "abstract: |\n"
        + "".join(f"  {line}\n" for line in abstract_tex_md.splitlines())
        + "---\n"
    )
    return yaml, body_md, keywords, ccs


HEADER_MAIN_TEX = r"""\setmonofont{DejaVuSansMono.ttf}[Scale=0.78]
\usepackage[htt]{hyphenat}
\usepackage{fvextra}
\RecustomVerbatimEnvironment{verbatim}{Verbatim}{fontsize=\scriptsize,breaklines,breakanywhere}
\emergencystretch=3em
% Float tuning: the paper concentrates figures/tables in Sections 3 and 5;
% permissive fractions stop LaTeX from breaking pages early around them.
\renewcommand{\topfraction}{0.92}
\renewcommand{\bottomfraction}{0.7}
\renewcommand{\textfraction}{0.06}
\renewcommand{\floatpagefraction}{0.86}
\setcounter{topnumber}{3}
\setcounter{totalnumber}{5}
% Math macros used by the paper that standard LaTeX lacks:
\DeclareMathOperator*{\argmin}{arg\,min}
\newcommand{\llbracket}{[\![}
\newcommand{\rrbracket}{]\!]}
"""

HEADER_SUPP_TEX = r"""\usepackage{etoolbox}
\usepackage[htt]{hyphenat}
\AtBeginEnvironment{longtable}{\small\setlength{\tabcolsep}{4pt}}
\emergencystretch=3em
\setmonofont{DejaVuSansMono.ttf}[Scale=0.86]
\DeclareMathOperator*{\argmin}{arg\,min}
\newcommand{\llbracket}{[\![}
\newcommand{\rrbracket}{]\!]}
"""


def ccsdesc_lines(ccs: list[str]) -> str:
    # acmart defines lowercase \ccsdesc, issued in the document body
    # before \maketitle (see reorder_front), not in the preamble.
    return "\n".join(rf"\ccsdesc[500]{{{d}}}" for d in ccs)


LONGTABLE_RE = re.compile(
    r"\\begin\{longtable\}\[\]\{@{}(.*?)@\{\}\}\s*\n"
    r"\\caption\{(.*?)\}(?:\\label\{(.*?)\})?\\tabularnewline\s*\n"
    r"\\toprule\\noalign\{\}\s*\n"
    r"(.*?)\s*\n"
    r"\\midrule\\noalign\{\}\s*\n"
    r"\\endfirsthead\s*\n"
    r".*?\\endhead\s*\n"
    r"\\bottomrule\\noalign\{\}\s*\n"
    r"\\endlastfoot\s*\n"
    r"(.*?)\s*\n"
    r"\\end\{longtable\}",
    re.S,
)


def convert_longtables(tex: str) -> tuple[str, int]:
    """longtable -> table (narrow) / table* (wide) for the two-column class.

    Wide = the column spec carries fractional \\columnwidth widths (pandoc
    emits plain alignment letters when every cell fits a narrow column).
    Wide tables are promoted to \\textwidth so the same fractions span the
    full page.
    """
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        count += 1
        spec, caption, label, header, body = (
            m.group(1), m.group(2), m.group(3), m.group(4), m.group(5))
        wide = "\\columnwidth" in spec
        env = "table*" if wide else "table"
        if wide:
            spec = spec.replace("\\columnwidth", "\\textwidth")
        cap_label = rf"\caption{{{caption}}}" + (rf"\label{{{label}}}" if label else "")
        return (
            rf"\begin{{{env}}}[t]"
            + "\n" + cap_label + "\n"
            + r"\small\setlength{\tabcolsep}{3pt}" + "\n"
            + rf"\begin{{tabular}}{{@{{}}{spec}@{{}}}}" + "\n"
            + r"\toprule\noalign{}" + "\n"
            + header + "\n"
            + r"\midrule\noalign{}" + "\n"
            + body + "\n"
            + r"\bottomrule\noalign{}" + "\n"
            + rf"\end{{tabular}}" + "\n"
            + rf"\end{{{env}}}"
        )

    return LONGTABLE_RE.sub(repl, tex), count


def promote_wide_figures(tex: str) -> tuple[str, int]:
    """figure -> figure* for assets named figures/wide_* (two-column class).

    Pandoc emits every image as a single-column ``figure``; assets whose
    filename carries the ``wide_`` prefix are promoted to full-width
    ``figure*`` floats so diagrams can span both columns."""
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        block = m.group(0)
        if "figures/wide_" in block:
            count += 1
            block = block.replace(r"\begin{figure}", r"\begin{figure*}")
            block = block.replace(r"\end{figure}", r"\end{figure*}")
        return block

    tex = re.sub(r"\\begin\{figure\}\n.*?\n\\end\{figure\}", repl, tex,
                 flags=re.S)
    return tex, count


def strip_class_clashes(tex: str) -> str:
    """Remove pandoc-template package loads that clash with acmart:
    unicode-math fights acmart's math setup under xelatex, and lmodern
    (loaded after the class) would switch the text font back from
    Libertine to Latin Modern."""
    tex = re.sub(r"^\\usepackage\{unicode-math\}.*?\n", "", tex, flags=re.M)
    tex = re.sub(r"^\\usepackage\{lmodern\}\n", "", tex, flags=re.M)
    # amssymb redefines symbols that acmart's math setup already provides
    # (\Bbbk et al. -> "already defined" errors).
    tex = re.sub(r"^\\usepackage\{amsmath,amssymb\}\n",
                 r"\\usepackage{amsmath}\n", tex, flags=re.M)
    return tex


def reorder_front(tex: str, keywords: list[str], ccs: list[str]) -> str:
    """pandoc: \\maketitle then abstract; acmart: abstract + \\keywords +
    \\ccsdesc before \\maketitle.  Rewrite accordingly."""
    m = re.search(
        r"\\maketitle\s*\n\\begin\{abstract\}\s*\n(.*?)\n\\end\{abstract\}",
        tex, re.S)
    if not m:
        raise SystemExit("FATAL: pandoc title/abstract block not found")
    front = ""
    for d in ccs:
        front += rf"\ccsdesc[500]{{{d}}}" + "\n"
    if keywords:
        front += r"\keywords{" + "; ".join(keywords) + "}\n"
    replacement = (
        "\\begin{abstract}\n" + m.group(1)
        + "\n\\end{abstract}\n" + front + "\\maketitle"
    )
    return tex[:m.start()] + replacement + tex[m.end():]


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kw)


def build_target(name: str, src: Path, tex_out: Path, pdf_out: Path,
                 title: str, flavor: str) -> int:
    """flavor: 'main' (acmart sigconf two-column) | 'supp' (article)."""
    build_dir = BUILD / name
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    if FIGURES.is_dir() and any(FIGURES.glob("*.pdf")):
        shutil.copytree(FIGURES, build_dir / "figures",
                        ignore=shutil.ignore_patterns("*.png"))

    yaml, body_md, keywords, ccs = build_body_md(
        src.read_text(encoding="utf-8"), title=title,
        extract_acmart_meta=(flavor == "main"))
    (build_dir / "meta.yaml").write_text(yaml, encoding="utf-8")
    (build_dir / "body.md").write_text(body_md, encoding="utf-8")

    header = HEADER_MAIN_TEX if flavor == "main" else HEADER_SUPP_TEX
    (build_dir / "header.tex").write_text(header, encoding="utf-8")

    pandoc_args = [
        "pandoc", str(build_dir / "meta.yaml"), str(build_dir / "body.md"),
        "-o", str(build_dir / "out.tex"),
        "--standalone", "--number-sections", "--top-level-division=section",
        "-H", str(build_dir / "header.tex"),
    ]
    if flavor == "main":
        pandoc_args += [
            "-V", "documentclass=acmart",
            "-V", "classoption=sigconf",
            "-V", "classoption=anonymous",
        ]
    else:
        pandoc_args += [
            "-V", "documentclass=article", "-V", "fontsize=11pt",
            "-V", "geometry:margin=1in",
            "-V", "colorlinks=true", "-V", "linkcolor=blue",
            "-V", "urlcolor=blue",
        ]
    tex = run(pandoc_args, capture_output=True, text=True)
    if tex.returncode != 0:
        print(tex.stdout, tex.stderr, file=sys.stderr)
        return 1

    out_tex_path = build_dir / "out.tex"
    out_tex = out_tex_path.read_text(encoding="utf-8")
    if flavor == "main":
        out_tex = strip_class_clashes(out_tex)
        out_tex, n_tables = convert_longtables(out_tex)
        print(f"  converted {n_tables} longtables to table floats")
        out_tex, n_wide = promote_wide_figures(out_tex)
        print(f"  promoted {n_wide} wide figures to figure* floats")
        out_tex = reorder_front(out_tex, keywords, ccs)
    out_tex_path.write_text(out_tex, encoding="utf-8")

    pdf = run(
        ["latexmk", "-xelatex", "-interaction=nonstopmode",
         "-halt-on-error", str(out_tex_path)],
        cwd=build_dir, capture_output=True, text=True,
    )
    log = build_dir / "out.log"
    if pdf.returncode != 0 or not (build_dir / "out.pdf").exists():
        print(pdf.stdout[-6000:], pdf.stderr[-2000:], file=sys.stderr)
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

    shutil.copy(build_dir / "out.tex", tex_out)
    shutil.copy(build_dir / "out.pdf", pdf_out)
    m = re.search(r"Output written on .*?\((\d+) pages?", pdf.stdout)
    pages = m.group(1) if m else "?"
    print(f"\nOK: {tex_out}\nOK: {pdf_out} ({pages} pages)")
    return 0


def main() -> int:
    targets = {
        "main": (DOCS / "AgentISA.md", TEX_OUT, PDF_OUT, TITLE, "main"),
        "supp": (DOCS / "AgentISA_supplementary.md", SUPP_TEX_OUT,
                 SUPP_PDF_OUT, SUPP_TITLE, "supp"),
    }
    want = sys.argv[1:] or ["all"]
    if want == ["all"]:
        want = list(targets)
    for key in want:
        if key not in targets:
            print(f"unknown target: {key}", file=sys.stderr)
            return 1
        src, tex_out, pdf_out, title, flavor = targets[key]
        rc = build_target(key, src, tex_out, pdf_out, title, flavor)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
