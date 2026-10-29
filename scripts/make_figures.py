#!/usr/bin/env python3
"""Generate the paper's figures from committed results JSONs.

Discipline (same as scripts/number_regression.py): every plotted data value is
read from a committed artifact and asserted against it; a manifest
(docs/figures/figures_manifest.json) records the plotted values so the
number regression can re-verify figures against their sources independently.

Output is deterministic: SOURCE_DATE_EPOCH is pinned before matplotlib is
imported, the Agg backend is forced, and no timestamps enter the PDFs.

Figure inventory
  wide_1_architecture.pdf   conceptual (no data)      main, Section 2
  2_compile_pipeline.pdf    conceptual (no data)      supplementary
  3_cache_exactness.pdf     bench JSONs, 2-3 families main, Section 5.7
  4_token_mix_sweep.pdf     promptcache_sweep JSON    main, Section 5.7.1
  5_quality_per_class.pdf   quality JSONs             supplementary I.3
  6_ceiling_census.pdf      compiler/identity JSONs   supplementary I.3
  7_instruction_memory.pdf  precipitation + pilot JSONs main, Section 5.12
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "946684800")  # 2000-01-01
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dsh-figures-mpl")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "examples" / "benchmarks"
OUT = ROOT / "docs" / "figures"

plt.rcParams.update({
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.4,
    "ytick.labelsize": 6.4,
    "legend.fontsize": 6.2,
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
})

MANIFEST: dict = {"figures": {}}


def jload(name: str):
    return json.loads((BENCH / name).read_text(encoding="utf-8"))


def chk(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"FIGURE DATA MISMATCH: {msg}")


def save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def box(ax, xy, wh, text, *, fc="#eef2f7", ec="#33475b", fs=6.2, lw=0.7):
    ax.add_patch(FancyBboxPatch(xy, wh[0], wh[1], boxstyle="round,pad=0.04",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, text, ha="center", va="center",
            fontsize=fs, zorder=3, linespacing=1.15)


def arrow(ax, p0, p1, *, style="-|>", lw=0.8, color="#33475b", ls="-",
          rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=7,
                                 lw=lw, color=color, linestyle=ls, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))


# ── F1: architecture (conceptual, no data) ─────────────────────────────────
def fig1() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 1.62))
    ax.set_xlim(0, 14.6)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    box(ax, (0.1, 3.1), (1.9, 1.5), "Agent\ndialogues")
    box(ax, (2.5, 3.1), (1.9, 1.5), "Parse /\nauthor")
    box(ax, (4.9, 3.1), (2.2, 1.5), "Instruction\nstream (ISA)")
    box(ax, (7.6, 3.1), (2.0, 1.5), "Security\nmonitor")
    box(ax, (10.1, 3.1), (2.0, 1.5), "Coordinator")
    box(ax, (10.1, 0.4), (2.0, 1.5), "Executor")
    box(ax, (7.0, 0.4), (2.4, 1.5), "Result cache\n(first occurrence)")
    box(ax, (4.0, 0.4), (2.2, 1.5), "Gateway\n(production models)")
    box(ax, (4.9, 3.1 - 1.0), (2.2, 0.0), "", fc="none", ec="none")  # spacer
    box(ax, (1.4, 0.4), (2.4, 1.5), "Hot-path EMA\n+ static passes")
    arrow(ax, (2.0, 3.85), (2.5, 3.85))
    arrow(ax, (4.4, 3.85), (4.9, 3.85))
    arrow(ax, (7.1, 3.85), (7.6, 3.85))
    arrow(ax, (9.6, 3.85), (10.1, 3.85))
    arrow(ax, (11.1, 3.1), (11.1, 1.9), rad=-0.25)
    arrow(ax, (10.1, 1.15), (9.4, 1.15))
    arrow(ax, (7.0, 1.15), (6.2, 1.15))
    arrow(ax, (10.6, 0.4), (2.6, 0.75), style="-|>", ls=(0, (3, 2)), rad=0.18,
          color="#8a5a00")
    ax.text(6.6, 0.06, "re-optimize and re-adopt (threshold-triggered)",
            fontsize=5.8, color="#8a5a00", ha="center")
    ax.text(5.1, 2.75, "deny / allow", fontsize=5.6, ha="center", color="#33475b")
    ax.text(9.25, 2.45, "LLM-type steps only", fontsize=5.6, ha="center",
            color="#33475b")
    save(fig, "wide_1_architecture.pdf")
    MANIFEST["figures"]["wide_1_architecture.pdf"] = {"kind": "conceptual"}


# ── F2: compile pipeline on the triage example (conceptual) ────────────────
def fig2() -> None:
    fig, ax = plt.subplots(figsize=(3.35, 1.45))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    auth = ["ingest\nlog", "check\nformat", "classify\nsignat.", "correlate",
            "enrich", "route", "notify"]
    comp = ["ingest\nlog", "classify\nsignat.", "correlate", "enrich",
            "route", "notify"]
    for i, s in enumerate(auth):
        dead = (i == 1)
        llm = (i == 2)
        box(ax, (0.15 + i * 1.40, 4.6), (1.16, 1.1), s,
            fc="#f3d6d6" if dead else ("#fdf0d5" if llm else "#eef2f7"),
            ec="#a33" if dead else ("#b8860b" if llm else "#33475b"), fs=5.4)
        if dead:
            ax.plot([0.22 + i * 1.40, 1.24 + i * 1.40], [5.15, 5.15],
                    color="#a33", lw=0.9)
    for i, s in enumerate(comp):
        llm = (i == 1)
        box(ax, (0.15 + i * 1.66, 1.2), (1.42, 1.1), s,
            fc="#fdf0d5" if llm else "#eef2f7",
            ec="#b8860b" if llm else "#33475b", fs=5.4)
    for i in range(6):
        arrow(ax, (1.57 + i * 1.66, 1.75), (1.81 + i * 1.66, 1.75), lw=0.6)
    for i in range(6):
        arrow(ax, (1.57 + i * 1.40, 5.15), (1.55 + i * 1.40, 5.15), lw=0.0)
    ax.text(0.15, 6.1, "Authored stream (7 steps)", fontsize=6.2,
            weight="bold")
    ax.text(0.15, 2.85, "Compiled stream (6 steps)", fontsize=6.2,
            weight="bold")
    arrow(ax, (2.3, 4.5), (2.0, 2.5), style="-|>", ls=(0, (2, 2)),
          color="#a33", rad=0.0)
    ax.text(2.62, 3.55, "dead step eliminated\n(dataflow-preserving)",
            fontsize=5.4, color="#a33")
    ax.text(4.3, 0.82, "LLM step served from the result cache after first occurrence",
            fontsize=5.2, color="#b8860b")
    ax.text(7.0, 6.1, "7 $\\rightarrow$ 6", fontsize=7.5, weight="bold",
            color="#a33")
    save(fig, "2_compile_pipeline.pdf")
    MANIFEST["figures"]["2_compile_pipeline.pdf"] = {
        "kind": "conceptual", "steps_authored": 7, "steps_compiled": 6}


# ── F3: cache exactness across model families (bench JSONs) ───────────────
def fig3() -> None:
    fams = [("GLM 5.3 flash", "isa_real_llm_results.json"),
            ("Doubao Seed 2.0", "isa_real_llm_results_doubao_seed_2_0_code.json")]
    kimi = BENCH / "isa_real_llm_results_kimi_k2_7_code.json"
    if kimi.exists():
        fams.append(("Kimi K2.7", "isa_real_llm_results_kimi_k2_7_code.json"))
    skews = [0.5, 1.0, 1.5]
    colors = ["#33475b", "#4f8a5b", "#8a5a00"]
    measured: dict = {}
    fig, ax = plt.subplots(figsize=(3.35, 1.75))
    for fi, (label, fn) in enumerate(fams):
        d = jload(fn)
        cells = {z["zipf_s"]: z["per_seed"][0] for z in d["data"]}
        for si, s in enumerate(skews):
            c = cells[s]
            chk(c["hit_rate"] == c["expected_hit_rate"],
                f"{fn} s={s} measured hit {c['hit_rate']} != expected {c['expected_hit_rate']}")
            chk(c["baseline"]["api_calls"] == 200,
                f"{fn} s={s} baseline calls {c['baseline']['api_calls']} != 200")
            x = si + (fi - (len(fams) - 1) / 2) * 0.26
            ax.bar(x, c["hit_rate"] * 100, width=0.24, color=colors[fi],
                   label=label if si == 0 else None, zorder=2)
            ax.plot([x - 0.12, x + 0.12],
                    [c["expected_hit_rate"] * 100] * 2, color="white", lw=0.9,
                    zorder=3)
            ax.text(x, c["hit_rate"] * 100 + 1.2, f"{c['hit_rate']*100:.1f}",
                    ha="center", fontsize=5.2, zorder=4)
            measured.setdefault(fn, []).append(
                {"zipf_s": s, "hit_rate": c["hit_rate"],
                 "expected": c["expected_hit_rate"],
                 "calls": f'{c["baseline"]["api_calls"]}\u2192{c["cached"]["api_calls"]}'})
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"s = {s}" for s in skews])
    ax.set_ylabel("Cache hit rate (%)")
    ax.set_xlabel("Zipf skew of the 200-task protocol")
    ax.set_ylim(0, 56)
    ax.legend(loc="upper left", frameon=False)
    ax.text(0.98, 0.965,
            "white tick = computed expectation;\nmeasured = expected in every cell",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.4)
    save(fig, "3_cache_exactness.pdf")
    MANIFEST["figures"]["3_cache_exactness.pdf"] = {
        "kind": "data", "sources": [fn for _, fn in fams],
        "families": [lbl for lbl, _ in fams], "cells": measured,
        "all_cells_exact": True}


# ── F4: token-mix sweep (MODELED pricing overlay) ─────────────────────────
def fig4() -> None:
    d = jload("promptcache_sweep_results.json")
    chk(d["provenance"]["scale"]["cells"] == 195
        and d["coverage"]["sweep_cells_completed"] == 195,
        "sweep coverage != 195 cells")
    fig, ax = plt.subplots(figsize=(3.35, 1.62))
    series = {}
    for pi, ps in enumerate(d["analysis"]["per_skew"]):
        s = ps["zipf_s"]
        xs, ys_a, ys_p = [], [], []
        for pt in ps["curve"]:
            if pt["x_completion_tokens_per_call"] is None:
                continue
            xs.append(pt["x_completion_tokens_per_call"])
            ys_a.append(pt["savings_agentisa_cache_pct"])
            ys_p.append(pt["savings_provider_prompt_cache_pct"])
            chk(ys_a[-1] >= ys_p[-1],
                f"sweep ordering violated at s={s} x={xs[-1]}")
        series[s] = {"x": xs, "agentisa": ys_a, "provider": ys_p}
        ax.plot(xs, ys_p, "-", color=colors3(pi), label=f"prompt cache, s = {s}",
                lw=1.0, zorder=3)
        ax.plot(xs, ys_a, "--", color=colors3(pi), lw=0.8, alpha=0.75,
                zorder=2)
        chk(abs(max(ys_a) - max(ys_p)) > 0, "degenerate series")
    ax.plot([], [], "k--", lw=0.8, alpha=0.75, label="result cache (all skews)")
    for x, lbl in ((64, "bench ceiling\n(max_tokens 64)"),
                   (512, "compiler harness\n(max_tokens 512)")):
        ax.axvline(x, color="#999", lw=0.5, ls=":", zorder=1)
        ax.text(x, 3.5, lbl, fontsize=5.0, ha="center", color="#666",
                rotation=90, va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 16, 64, 256, 1024])
    ax.set_xticklabels(["4", "16", "64", "256", "1024"])
    ax.set_xlim(3.6, 1200)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Modeled completion tokens per call")
    ax.set_ylabel("Modeled cost saving (%)")
    ax.legend(loc="lower left", frameon=False, ncol=1)
    ax.text(0.985, 0.03, "MODELED pricing overlay\n(promptcache_sweep, 195 cells)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.2,
            color="#666")
    save(fig, "4_token_mix_sweep.pdf")
    MANIFEST["figures"]["4_token_mix_sweep.pdf"] = {
        "kind": "data", "sources": ["promptcache_sweep_results.json"],
        "cells": 195, "series": {str(k): v for k, v in series.items()}}


def colors3(i: int) -> str:
    return ["#33475b", "#4f8a5b", "#8a5a00"][i % 3]


# ── F5: quality per-class agreement (two models vs fallback rule) ─────────
def fig5() -> None:
    glm = jload("gateway_quality_results.json")["summary_recomputed"]
    dsk = jload("gateway_quality_deepseek.json")
    dsm = dsk["summary"]
    classes = ["low", "medium", "high", "critical"]
    # Fallback per-class recomputed from the deepseek run's raw records.
    fb = {c: [0, 0] for c in classes}
    for b in dsk["data"]:
        for sig, gold in b["gold"].items():
            c = fb[gold]
            c[1] += 1
            if b["rule_fallback"][sig] == gold:
                c[0] += 1
    fb_overall = sum(c[0] for c in fb.values()) / sum(c[1] for c in fb.values())
    chk(round(fb_overall, 4) == round(dsm["rule_fallback_agreement_vs_gold"], 4),
        "fallback per-class recompute != overall")
    chk(round(dsm["final_agreement_vs_gold"], 3) == 0.521
        and round(dsm["rule_fallback_agreement_vs_gold"], 3) == 0.319,
        "deepseek overall agreement drifted")
    chk(round(glm["final_agreement_vs_gold"], 4) == 0.4545
        and round(glm["rule_fallback_agreement_vs_gold"], 4) == 0.3239,
        "glm overall agreement drifted")
    fig, ax = plt.subplots(figsize=(3.35, 1.65))
    xs = range(len(classes))
    w = 0.26
    for off, (vals, lbl, fc) in enumerate([
            ([glm["per_gold_class"][c]["agreement_rate"] for c in classes],
             "GLM 5.3 @3072", "#33475b"),
            ([dsm["per_gold_class"][c]["agreement_rate"] for c in classes],
             "DeepSeek v4 @1500", "#4f8a5b"),
            ([fb[c][0] / fb[c][1] for c in classes],
             "deterministic fallback rule", "#c9c9c9")]):
        ax.bar([x + (off - 1) * w for x in xs], [v * 100 for v in vals], w,
               color=fc, label=lbl, zorder=2)
    ax.axhline(dsm["final_agreement_vs_gold"] * 100, color="#4f8a5b", lw=0.5,
               ls=":", zorder=1)
    ax.axhline(fb_overall * 100, color="#888", lw=0.5, ls=":", zorder=1)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(classes)
    ax.set_ylabel("Agreement with gold (%)")
    ax.set_ylim(0, 78)
    ax.legend(loc="upper right", frameon=False, fontsize=5.6)
    ax.text(0.02, 0.955, "dotted: overall agreement", transform=ax.transAxes,
            fontsize=5.2, color="#666", va="top")
    save(fig, "5_quality_per_class.pdf")
    MANIFEST["figures"]["5_quality_per_class.pdf"] = {
        "kind": "data",
        "sources": ["gateway_quality_results.json",
                    "gateway_quality_deepseek.json"],
        "glm_per_class": {c: glm["per_gold_class"][c]["agreement_rate"] for c in classes},
        "deepseek_per_class": {c: dsm["per_gold_class"][c]["agreement_rate"] for c in classes},
        "fallback_per_class": {c: fb[c][0] / fb[c][1] for c in classes},
        "fallback_overall": fb_overall}


# ── F6: ceiling-binding census across model families and budgets ──────────
def _census_calls(fn: str, cap: int):
    d = json.loads((BENCH / fn).read_text(encoding="utf-8"))
    tot = bound = 0
    for cell in d["data"]:
        for r in cell["arm_a_unoptimized"]["per_realization"]:
            if r["calls"] > 0:
                tot += 1
                if r["completion_tokens"] >= cap:
                    bound += 1
    return bound, tot


def fig6() -> None:
    glm_run = json.loads(
        (ROOT / "examples" / "paper_real_llm_compiler_results_real.json")
        .read_text(encoding="utf-8"))
    tot = bound = 0
    for cell in glm_run["data"]:
        for r in cell["arm_a_unoptimized"]["per_realization"]:
            if r["calls"] > 0:
                tot += 1
                if r["completion_tokens"] >= 512:
                    bound += 1
    glm512 = (bound, tot)
    chk(glm512 == (22, 22), f"glm@512 census {glm512} != (22, 22)")
    db512 = _census_calls("paper_real_llm_compiler_results_doubao_1500.json", 1500)
    chk(db512 == (0, 104), f"doubao@1500 census {db512} != (0, 104)")
    mm512 = _census_calls("paper_real_llm_compiler_results_minimax.json", 512)
    mm1500 = _census_calls("paper_real_llm_compiler_results_minimax_1500.json", 1500)
    chk(mm512 == (101, 104) and mm1500 == (68, 104),
        f"minimax census {mm512}/{mm1500} != (101,104)/(68,104)")
    rows = [("GLM 5.3\n@512", *glm512), ("Doubao 2.0\n@1500", *db512),
            ("Minimax m3\n@512", *mm512), ("Minimax m3\n@1500", *mm1500)]
    fig, ax = plt.subplots(figsize=(3.35, 1.5))
    xs = range(len(rows))
    vals = [b / t * 100 for _, b, t in rows]
    ax.bar(xs, vals, width=0.55,
           color=["#33475b", "#4f8a5b", "#8a5a00", "#b8860b"], zorder=2)
    for x, (_, b, t) in zip(xs, rows):
        ax.text(x, b / t * 100 + 2.5, f"{b}/{t}", ha="center", fontsize=5.6)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([r[0] for r in rows], fontsize=5.8)
    ax.set_ylabel("Calls binding at ceiling (%)")
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.text(0.985, 0.9, "replies bound at max_tokens\n(reasoning traces cut)",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.2,
            color="#666")
    save(fig, "6_ceiling_census.pdf")
    MANIFEST["figures"]["6_ceiling_census.pdf"] = {
        "kind": "data",
        "sources": ["examples/paper_real_llm_compiler_results_real.json",
                    "paper_real_llm_compiler_results_doubao_1500.json",
                    "paper_real_llm_compiler_results_minimax.json",
                    "paper_real_llm_compiler_results_minimax_1500.json"],
        "glm_512": list(glm512), "doubao_1500": list(db512),
        "minimax_512": list(mm512), "minimax_1500": list(mm1500)}


def _smoothed(tasks, points=60):
    """Task-wise coverage -> a readable curve: x and y on a percentage scale."""
    if not tasks:
        return [], []
    window = max(3, len(tasks) // 40)
    values, xs = [], []
    for i in range(len(tasks)):
        lo = max(0, i - window + 1)
        chunk = tasks[lo:i + 1]
        values.append(100.0 * sum(t["served"] for t in chunk)
                      / max(1, sum(t["steps"] for t in chunk)))
        xs.append(100.0 * (i + 1) / len(tasks))
    step = max(1, len(xs) // points)
    return xs[::step], values[::step]


def fig7() -> None:
    """Instruction memory: what accumulates from experience, and what it does not buy."""
    tau = jload("instruction_precipitation_results.json")
    swe = jload("instruction_precipitation_swe_agent_results.json")
    agt = jload("instruction_precipitation_agentgym_results.json")
    pilot = jload("real_trace_pilot_tau_bench_results.json")
    micro = json.loads((BENCH / "isa_microbench_results.json").read_text(encoding="utf-8"))
    compiler = next(r for r in micro if r["experiment"] == "compiler")
    planted_row = next(d for d in compiler["data"]
                       if d["length"] == 200 and d["level"] == "BASIC")
    planted = 100.0 * planted_row["effective_instruction_count_reduction"]["mean"]

    series = [
        ("closed: tool use, retail", tau["per_source"]["gpt-4o-retail"], "#1f4e79", "-"),
        ("closed: household manipulation", agt["per_source"]["agentgym-alfworld"], "#2e7d32", "-"),
        ("closed: grid navigation", agt["per_source"]["agentgym-babyai"], "#8a5a00", "-"),
        ("open: shell coding", swe["per_source"]["swe-agent-swe-agent-llama-70b"], "#6a3d9a", "-"),
        ("open: SQL analytics", agt["per_source"]["agentgym-sqlgym"], "#b03030", "--"),
    ]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(7.0, 1.80),
        gridspec_kw={"width_ratios": [2.0, 1.0], "wspace": 0.32})

    ax.axhspan(80, 103, color="#eef4ee", zorder=0)
    ax.axhspan(-2, 70, color="#fbf2e8", zorder=0)
    for label, stream, color, ls in series:
        xs, ys = _smoothed(stream["per_task"])
        ax.plot(xs, ys, color=color, ls=ls, lw=1.15, solid_capstyle="round")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("experience: share of the stream already seen (%)", labelpad=1.5)
    ax.set_ylabel("steps served by memory (%)", labelpad=1.5)
    ax.set_title("A  instructions sediment; the workload decides how far", loc="left",
                 fontsize=7.0, pad=3)
    handles = [plt.Line2D([], [], color=c, ls=ls, lw=1.15) for _, _, c, ls in series]
    labels = [f"{name} ({100.0 * st['coverage']:.0f}%)" for name, st, _, _ in series]
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.005, 0.015),
              frameon=False, handlelength=1.4, borderaxespad=0.0, labelspacing=0.22,
              fontsize=5.7)

    real_static = 100.0 * pilot["aggregate"]["effective_reduction_vs_none"]["BASIC"]
    within = 100.0 * pilot["cache_scopes"]["first_trial"]["within_task_hit_rate"]
    across = 100.0 * pilot["cache_scopes"]["first_trial"]["within_run_hit_rate"]
    chk(abs(planted - 34.2) < 0.05, f"planted reduction {planted:.2f} != 34.2")
    chk(abs(real_static) < 1e-9, f"real-trace static reduction {real_static} != 0.0")
    chk(abs(within - 1.73) < 0.005 and abs(across - 41.70) < 0.005,
        f"cache rates {within:.2f}/{across:.2f} != 1.73/41.70")

    # two metric groups, each with a planted/synthetic arm and a real-traces arm
    groups = [
        ("static reduction", [("planted\nstream", planted, "#c3d2e2"),
                              ("real\ntraces", real_static, "#33475b")]),
        ("cache hit rate", [("within\ntask", within, "#c3d2e2"),
                            ("across\ntasks", across, "#33475b")]),
    ]
    tick_positions, tick_labels = [], []
    for g, (title, bars) in enumerate(groups):
        base = g * 3
        for k, (label, value, color) in enumerate(bars):
            x = base + k
            ax2.bar(x, value, width=0.70, color=color, edgecolor="#33475b",
                    linewidth=0.6)
            ax2.text(x, value + 1.4, f"{value:.1f}", ha="center", va="bottom",
                     fontsize=6.2, color="#1b2b3a")
            tick_positions.append(x)
            tick_labels.append(label)
        ax2.text(base + 0.5, 47.0, title, ha="center", fontsize=6.2, color="#33475b")
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=5.5, linespacing=1.05)
    ax2.set_ylim(0, 50)
    ax2.set_yticks([0, 20, 40])
    ax2.set_ylabel("share (%)", labelpad=1.5)
    ax2.set_title("B  what memory does not buy", loc="left", fontsize=7.0, pad=3)
    ax2.grid(axis="x", visible=False)

    save(fig, "7_instruction_memory.pdf")
    MANIFEST["figures"]["7_instruction_memory.pdf"] = {
        "kind": "data",
        "series_final_pct": {name: round(100.0 * st["coverage"], 2)
                             for name, st, _, _ in series},
        "bars_pct": {"compiler_planted": round(planted, 2),
                     "compiler_real": round(real_static, 2),
                     "cache_within_task": round(within, 2),
                     "cache_across_tasks": round(across, 2)},
    }


def main() -> int:
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures_manifest.json").write_text(
        json.dumps(MANIFEST, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"OK: {len(MANIFEST['figures'])} figures -> {OUT}")
    for name in sorted(MANIFEST["figures"]):
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
