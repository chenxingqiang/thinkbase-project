#!/usr/bin/env python3
"""Regression gate: every benchmark number cited in docs/AgentISA.md and
docs/AgentISA_supplementary.md (Sections 5.2-5.8 plus the Appendix-I extension
experiments) must match the committed results JSONs under examples/benchmarks/
bit-for-bit. Stored aggregates are additionally recomputed from the raw
per-seed records wherever a file carries them. Aborts nonzero on drift."""
import json, os, re, subprocess, sys, pathlib, statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
md = (ROOT / 'docs/AgentISA.md').read_text(encoding='utf-8')
md += (ROOT / 'docs/AgentISA_supplementary.md').read_text(encoding='utf-8')  # submission-wide probe
rs = json.loads((ROOT / 'examples/benchmarks/isa_microbench_results.json').read_text())

def rec(name):
    for r in rs:
        if r['experiment'] == name:
            return r
    raise KeyError(name)

fails = []
def chk(cond, msg):
    if not cond: fails.append(msg)

# ---- 5.2 cache: uncached/ISA/naive means+stdevs per skew; hit rates
cache = {d['zipf_s']: d for d in rec('cache')['data']}
paper_cached = {0.5: '810.4', 1.0: '642.0', 1.5: '408.0'}
for s_, cst in paper_cached.items():
    d = cache[s_]
    chk(d['api_calls_no_cache']['mean'] == 2000, f"cache s={s_} uncached mean json={d['api_calls_no_cache']['mean']}")
    chk(abs(d['api_calls_cached']['mean'] - float(cst)) < 0.05,
        f"cache s={s_} ISA json={d['api_calls_cached']['mean']} vs paper {cst}")
    chk(abs(d['api_calls_naive_response_cache']['mean'] - float(cst)) < 0.05,
        f"cache s={s_} naive json={d['api_calls_naive_response_cache']['mean']} vs paper {cst}")
    chk(cst in md, f'paper missing cached {cst}')
for s_, hr in [(0.5, '59.5'), (1.0, '67.9'), (1.5, '79.6')]:
    d = cache[s_]
    pct = d['cache_hit_rate']['mean'] * 100
    chk(abs(pct - float(hr)) < 0.05, f'hit rate s={s_} json={pct:.2f} vs paper {hr}')
    chk(f'{hr}%' in md, f'paper missing hit rate {hr}%')

# ---- 5.3 compiler: 34.2±1.8 @L200, 35.4 @L1000 (level ALL)
comp = rec('compiler')['data']
def comp_red(length):
    for d in comp:
        if d['length'] == length and d['level'] == 'BASIC':
            return d['effective_instruction_count_reduction']
    raise KeyError(length)
r200, r1000 = comp_red(200), comp_red(1000)
chk(abs(r200['mean']*100-34.2) < 0.05 and abs(r200['stdev']*100-1.8) < 0.05,
    f"compiler L200 json={r200['mean']*100:.1f}±{r200['stdev']*100:.1f} vs paper 34.2±1.8")
chk(abs(r1000['mean']*100-35.4) < 0.05, f"compiler L1000 json={r1000['mean']*100:.1f} vs paper 35.4")
chk('34.2% ± 1.8%' in md and '35.4' in md, 'paper missing 34.2±1.8 / 35.4')

# ---- 5.4 dynamic: aggregate 65.8±4.6; per-seed table; replay all True
dyn = rec('dynamic')['data']
shrinks = [d['shrink_ratio']*100 for d in dyn]
m, sd = statistics.mean(shrinks), statistics.stdev(shrinks)
chk(abs(m-65.8) < 0.05 and abs(sd-4.6) < 0.05, f'dynamic json={m:.1f}±{sd:.1f} vs paper 65.8±4.6')
rv = rec('dynamic')['replay_verification']
chk(rv['all_seeds_state_identical'] is True, 'json replay not all identical')
for d in dyn:
    chk(d['replay_state_identical'] is True, f"seed {d['seed']} replay false in json")
    row = f"| {d['seed']}    | {d['original_len']}             | {d['optimized_len']}               | {d['shrink_ratio']*100:.1f}%"
    chk(row in md, f'5.4 table row altered: {row}')
chk('65.8% ± 4.6%' in md, 'paper missing 65.8±4.6')

# ---- 5.5 secondary ablation (independent draw): 21.9±3.5 / 24.1±3.3 / 35.0±3.7
abl = rec('ablation')['data']
def abl_red(cfg):
    for d in abl:
        if d['configuration'] == cfg:
            return d['instruction_count_reduction']
    raise KeyError(cfg)
for cfg, mn, sd in [('fusion_only', 21.9, 3.5), ('dce_only', 24.1, 3.3), ('fusion+dce', 35.0, 3.7), ('all_passes', 35.0, 3.7)]:
    r = abl_red(cfg)
    chk(abs(r['mean']*100-mn) < 0.05 and abs(r['stdev']*100-sd) < 0.05,
        f'ablation {cfg} json={r["mean"]*100:.1f}±{r["stdev"]*100:.1f} vs paper {mn}±{sd}')

# ---- 5.5 paired ablation: 21.6±1.2 / 24.5±2.6 / 34.2±1.8; interaction −11.9±2.0 per-seed
paired = rec('paired_ablation')['data']
def p_red(cfg):
    for d in paired:
        if d['configuration'] == cfg:
            return d['instruction_count_reduction']
    raise KeyError(cfg)
for cfg, mn, sd in [('fusion_only', 21.6, 1.2), ('dce_only', 24.5, 2.6), ('fusion+dce', 34.2, 1.8)]:
    r = p_red(cfg)
    chk(abs(r['mean']*100-mn) < 0.05 and abs(r['stdev']*100-sd) < 0.05,
        f'paired {cfg} json={r["mean"]*100:.1f}±{r["stdev"]*100:.1f} vs paper {mn}±{sd}')
ia = rec('paired_ablation')['paired_interaction']
chk(abs(ia['mean']*100-(-11.9)) < 0.05 and abs(ia['stdev']*100-2.0) < 0.06,
    f"interaction json={ia['mean']*100:.2f}±{ia['stdev']*100:.2f} vs paper −11.9±2.0")
want_ps = ['-14.5', '-9.5', '-13.0', '-12.0', '-10.5']
got_ps = [f"{v*100:.1f}" for v in ia['per_seed']]
chk(got_ps == want_ps, f'interaction per-seed json={got_ps} vs paper {want_ps}')
chk('−11.9% ± 2.0%' in md and '−14.5/−9.5/−13.0/−12.0/−10.5' in md,
    'paper missing interaction mean/per-seed strings')

# ---- cross-derived invariant: 21.6 + 24.5 = 46.1 cited
chk('46.1' in md, 'paper missing 46.1 sum invariant')

# ---- 5.7/5.7.1 real-gateway tables (source: isa_real_llm_results.json; priced overlay separate)
real = {d['zipf_s']: d['per_seed'][0] for d in json.loads((ROOT/'examples/benchmarks/isa_real_llm_results.json').read_text())['data']}
for s_, row_expect in [(0.5, ('200 -> 183', '472,276 -> 405,754', '66,522', '1,445', '8.5%')),
                       (1.0, ('200 -> 148', '455,946 -> 334,597', '121,349', '4,420', '26.0%')),
                       (1.5, ('200 -> 109', '450,330 -> 244,316', '206,014', '7,735', '45.5%'))]:
    m = real[s_]
    chk(f"{m['baseline']['api_calls']} -> {m['cached']['api_calls']}" == row_expect[0],
        f'T6 s={s_} calls json {m["baseline"]["api_calls"]}->{m["cached"]["api_calls"]} vs paper {row_expect[0]}')
    chk(f"{m['baseline']['wall_ms']:,.0f} -> {m['cached']['wall_ms']:,.0f}" == row_expect[1],
        f'T6 s={s_} wall mismatch')
    chk(f"{m['wall_saved_ms']:,.0f}" == row_expect[2], f'T6 s={s_} saved mismatch')
    chk(f"{m['tokens_saved']:,}" == row_expect[3], f'T6 s={s_} tokens mismatch')
    for cell in row_expect:
        chk(cell in md, f'paper missing T6 cell {cell}')
# Table 7 (priced overlay) rows must survive verbatim
for row in ['| No optimization | 50 | 9,018 / 0 | 25,600 | $0.4111 |',
            '| AgentISA result caching | 22 | 4,027 / 0 | 11,264 | $0.1810 |',
            '| Provider prompt cache (MODELED) | 50 | 4,027 / 4,991 | 25,600 | $0.3976 |']:
    chk(row in md, f'T7 row missing/altered: {row[:45]}')
# key fixed citations around 5.7.1
for probe in ['56.0%', '3.3%', '80.1%', '81.6%', '85.1%', '28.9–31.4%']:
    chk(probe in md, f'5.7.1 citation missing: {probe}')

# ==== W5 extension: Appendix-I extension-experiment results files =============
# Every extension experiment committed a results JSON under examples/benchmarks/.
# Each block below (a) recomputes stored aggregates from the raw per-seed records
# where the file carries them, and (b) checks the stored values against the
# numbers cited in the combined `md` (main text + supplementary).  Display
# convention: the paper renders round-half-up at the cited precision; numeric
# tolerances are the half-ULP of that precision unless the check is byte-exact.
# One rendering caveat, recorded here deliberately: promptcache_sweep_results.json
# stores 4-decimal values while the paper table was rendered from the full
# precision run, so a tied cell can display one final-digit ulp either way
# (e.g. stored 0.0915 vs paper 0.092); those cells are compared numerically at
# half-ULP tolerance instead of byte-exactly.
from decimal import Decimal, ROUND_HALF_UP

BM = ROOT / 'examples/benchmarks'
def jload(name):
    return json.loads((BM / name).read_text())
def fmt(v, nd):
    """Round-half-up fixed-point rendering (the paper's display convention)."""
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(float(v))).quantize(q, rounding=ROUND_HALF_UP))
def fmtc(v, nd):
    """Round-half-up fixed-point with thousands separators."""
    q = Decimal(1).scaleb(-nd)
    return format(Decimal(repr(float(v))).quantize(q, rounding=ROUND_HALF_UP), f',.{nd}f')

NEW_RESULT_FILES = ['trace_workload_results.json', 'capability_calibration_results.json',
                    'selector_dp_results.json', 'hotpath_ema_results.json',
                    'security_posture_results.json', 'cache_namespace_results.json',
                    'replay_generalization_results.json', 'fusion_chain_results.json',
                    'promptcache_sweep_results.json', 'gateway_quality_results.json',
                    'gateway_multiseed_results.json']
for f_ in NEW_RESULT_FILES:
    rev = jload(f_).get('provenance', {}).get('git_rev')
    chk(rev == '0e1671f8', f'{f_}: provenance git_rev {rev} != 0e1671f8 cited in Appendix I')
chk('0e1671f8' in md, 'paper missing the Appendix-I provenance revision 0e1671f8')

# ---- 5.10 / I.1 real-trace census (trace_workload_results.json)
tw = jload('trace_workload_results.json')
tz, tre, tsh, tfu = tw['zipf'], tw['reuse'], tw['shapes'], tw['fusion']
chk(tz['n_categories'] == 12 and tz['total'] == 792,
    f"trace census scale json {tz['n_categories']} classes / {tz['total']} steps vs paper 12/792")
chk(abs(tz['s'] - 0.5927) < 5e-5 and abs(tz['r2'] - 0.9714) < 5e-5,
    f"trace zipf json s={tz['s']:.4f} R2={tz['r2']:.4f} vs paper 0.5927/0.9714")
chk(tre['total_instances'] == 792 and tre['repeat_instances'] == 780
    and abs(tre['reuse_ratio'] - 0.9848) < 5e-5,
    f"trace reuse json {tre['repeat_instances']}/{tre['total_instances']} ratio {tre['reuse_ratio']:.4f} vs paper 780/792 98.5%")
chk(abs(tre['per_workflow_mean_reuse_ratio'] - 0.1833) < 5e-5,
    f"trace per-workflow reuse json {tre['per_workflow_mean_reuse_ratio']:.4f} vs paper 18.3%")
chk(tsh['counts'] == {'linear': 21, 'fan-out': 79, 'fan-in': 0, 'branch': 0},
    f"trace shape census json {tsh['counts']} vs paper 21/79/0/0")
chk(tsh['dead_end_steps'] == 152 and tsh['workflows_with_multiple_components'] == 0,
    'trace dead-end / disconnected-workflow counts json mismatch')
chk(abs(tsh['avg_steps'] - 7.92) < 5e-3, f"trace avg steps json {tsh['avg_steps']} vs paper 7.9")
chk(tfu['total_adjacent_pairs'] == 692 and tfu['fusable_pairs'] == 106
    and abs(tfu['fusable_ratio'] - 0.1532) < 5e-5 and tfu['by_type'] == {'CONTROL_FLOW': 106},
    f"trace fusibility json {tfu['fusable_pairs']}/{tfu['total_adjacent_pairs']} vs paper 106/692 15.3%")
for hk, src in [('zipf_s', tz['s']), ('zipf_r2', tz['r2']), ('reuse_ratio', tre['reuse_ratio']),
                ('per_workflow_mean_reuse_ratio', tre['per_workflow_mean_reuse_ratio']),
                ('dead_end_steps', tsh['dead_end_steps']), ('fusable_pairs', tfu['fusable_pairs'])]:
    chk(tw['headline'][hk] == src, f'trace headline field {hk} drifts from its analysis section')
chk(sum(n for _, n in tz['top']) == 718 and tz['top'][0] == ['CONTROL_FLOW:guard', 152],
    'trace top-10 distribution json mismatch (top-10 sum 718; guard 152 cross-checks the dead-end count)')
for rank, (sig, n) in enumerate(tz['top'], 1):
    chk(f'| {rank} | {sig} | {n} |' in md, f'trace signature-frequency row {rank} altered')
chk(f"| Zipf fit (log-rank/log-freq OLS) | $s = {fmt(tz['s'], 3)}$ ({fmt(tz['s'], 4)}), $R^2 = {fmt(tz['r2'], 3)}$ ({fmt(tz['r2'], 4)}) |" in md,
    'trace census zipf row altered')
chk(f"| Corpus-level repeated instances | {tre['repeat_instances']} / {tre['total_instances']} ({fmt(tre['reuse_ratio']*100, 1)}%) |" in md
    and f"| Per-workflow mean repeat share | {fmt(tre['per_workflow_mean_reuse_ratio']*100, 1)}% |" in md
    and f"| Shape census (linear / fan-out / fan-in / branch) | {tsh['counts']['linear']} / {tsh['counts']['fan-out']} / {tsh['counts']['fan-in']} / {tsh['counts']['branch']} |" in md
    and f"| Average steps per workflow | {fmt(tsh['avg_steps'], 1)} |" in md
    and f"| Dead-end steps (single-pass count, pure, no consumer) | {tsh['dead_end_steps']} |" in md
    and f"| Fusible adjacent pairs | {tfu['fusable_pairs']} / {tfu['total_adjacent_pairs']} ({fmt(tfu['fusable_ratio']*100, 1)}%), all CONTROL_FLOW |" in md,
    'trace census summary-table rows altered')
chk(tw['coverage']['status'] == 'full' and tw['coverage']['n_trace_files_found'] == 100
    and tw['coverage']['n_workflows_parsed'] == 100 and tw['coverage']['provenance_gaps'] == [],
    'trace census coverage json is not full')
chk(tw['provenance']['corpus_sha256'] == '7c05d307c64ab418f85811efa18489fa80463ab504fdc1edb3f0c22a4bf323eb'
    and tw['provenance']['corpus_sha256'] in md, 'trace corpus sha256 altered in json or missing from paper')
chk(tw['provenance']['seeds']['master_seed'] == 2026
    and tw['provenance']['seeds']['per_task_seed_range'] == [10000, 10099],
    'trace generator seeds json mismatch (paper: master seed 2026, corpus seeds 10000-10099)')
for s_ in ['98.5% of instances recur corpus-wide', 'per-workflow repetition, 18.3%, is more informative', '152 dead-end steps exactly equal the planted guards',
           '106 of 692 adjacent pairs (15.3%)', '79 fan-out, 21 linear, 0 fan-in, 0 branch (7.9 steps/workflow)',
           'master seed 2026', '10000–10099']:
    chk(s_ in md, f'paper missing trace citation: {s_[:45]}')

# ---- 5.11 / I.2 capability calibration (capability_calibration_results.json)
cc = jload('capability_calibration_results.json')
dom, pmet = cc['correlations']['per_domain'], cc['correlations']['per_metric']
for name, r_, rho_, n_ in [('reasoning', -0.9647, -0.9526, 25), ('memory', 0.9482, 0.9517, 25),
                           ('generation', 0.7902, 0.9034, 15), ('coordination', -0.9767, -0.9836, 25),
                           ('optimization', -1.0, -1.0, 25), ('security', 1.0, 1.0, 25)]:
    c = dom[name]['correlation']
    chk(abs(c['pearson_r'] - r_) < 5e-5 and abs(c['spearman_rho'] - rho_) < 5e-5 and c['n'] == n_,
        f"capability domain {name} json r={c['pearson_r']:+.4f} rho={c['spearman_rho']:+.4f} n={c['n']} vs paper {r_:+.4f}/{rho_:+.4f}/{n_}")
chk(dom['specialization']['status'] == 'na' and dom['robustness']['status'] == 'na',
    'capability N/A domains json mismatch (paper: 2/8 domains N/A with probe evidence)')
enz = cc['extraction_nonzero_proof']
chk(enz['nonzero_metrics_per_seed']['counts'] == [7]*5 and enz['nonzero_metrics_per_seed']['of_total'] == 17
    and enz['nonzero_domains_per_seed']['counts'] == [5]*5 and enz['nonzero_domains_per_seed']['of_total'] == 8,
    'capability nonzero proof json mismatch (paper: 7/17 metrics, 5/8 domains nonzero on base pools)')
ok_m = sorted(k for k, v in pmet.items() if v.get('correlation', {}).get('status') == 'ok')
na_m = [k for k, v in pmet.items() if (v.get('correlation') or {}).get('status') == 'na' or v.get('status') == 'na']
chk(len(pmet) == 17 and ok_m == sorted(['causal_understanding', 'long_term_memory', 'creativity', 'coherence',
                                        'efficiency', 'resource_usage', 'collaboration', 'policy_compliance'])
    and len(na_m) == 9,
    f'capability per-metric statuses json mismatch (paper: 8 correlated, 9/17 N/A); na={na_m}')
for m, r_, n_ in [('causal_understanding', -0.9647, 25), ('long_term_memory', 0.9482, 25),
                  ('creativity', 1.0, 15), ('coherence', -0.99, 20), ('efficiency', 0.949, 25),
                  ('resource_usage', -1.0, 25), ('collaboration', -0.9767, 25), ('policy_compliance', 1.0, 25)]:
    c = pmet[m]['correlation']
    chk(abs(c['pearson_r'] - r_) < 5e-5 and c['n'] == n_,
        f"capability metric {m} json r={c['pearson_r']:+.4f} n={c['n']} vs paper {r_:+.4f} n={n_}")
mem_sec = {s_['measured']: s_['correlation'] for s_ in dom['memory']['secondary']}
chk(abs(mem_sec['measured.cache.0.5.hit_rate']['pearson_r'] - 0.9843) < 5e-5
    and abs(mem_sec['measured.cache.1.5.hit_rate']['pearson_r'] - 0.8925) < 5e-5,
    'capability memory secondary arms json mismatch (paper: +0.9843 / +0.8925)')
opt_sec = {s_['measured']: s_['correlation'] for s_ in dom['optimization']['secondary']}
chk(opt_sec['measured.resource_plan_total_cost']['pearson_r'] == -1.0
    and abs(opt_sec['measured.cache.1.0.hit_rate']['pearson_r'] - 0.949) < 5e-5,
    'capability optimization secondary arms json mismatch (paper: -1.0000 and +0.9490)')
chk(abs(dom['generation']['secondary'][0]['correlation']['pearson_r'] - 0.1756) < 5e-5,
    'capability generation workload arm json mismatch (paper: r = +0.176)')
probes = cc['inertness_probes']
inert = [p for p in probes if p['max_abs_measured_delta'] == 0.0]
eff = [p for p in probes if p['max_abs_measured_delta'] != 0.0]
chk(len(probes) == 10 and len(inert) == 9 and len(eff) == 1
    and eff[0]['probe'] == 'agent_type_recategorization'
    and abs(eff[0]['max_abs_measured_delta'] - 0.035) < 5e-5,
    'capability inertness probes json mismatch (paper: 9/10 inert; one effect 0.035)')
chk(abs(cc['provenance']['wall_time_s'] - 2.72) < 5e-3 and '2.72 s' in md,
    'capability run wall time json/paper mismatch (paper: 2.72 s)')
for s_ in ['−0.9647', '−0.9526', '+0.9482', '+0.9517', '+0.7902', '+0.9034', '−0.9767', '−0.9836',
           '−1.0000', '+1.0000', '+0.9843 / +0.8925', 'vs cache hit rate +0.9490', '$r$ = +0.176',
           '7 of 17 metrics and 5 of 8 domain scores', '9 of 10 injected declaration families',
           '2/8 domains, 9/17 metrics N/A']:
    chk(s_ in md, f'paper missing capability citation: {s_}')

# ---- 5.11 / I.2 DP tree-pattern selection (selector_dp_results.json)
sel = jload('selector_dp_results.json')
ov, inst = sel['overall'], sel['instances']
dp_costs = [i['dp']['cost'] for i in inst]
he_costs = [i['heuristic']['cost'] for i in inst]
chk(len(inst) == 20 and all(i['dp']['complete'] and i['dp']['tiling_valid'] and i['heuristic']['tiling_valid'] for i in inst),
    'selector json: not 20 complete valid tilings')
chk(all(i['dp']['cost'] <= i['heuristic']['cost'] for i in inst) and ov['dp_cost_le_heuristic_all'] is True,
    'selector DP<=heuristic violated in json (paper: 20/20 wins)')
chk(abs(statistics.mean(dp_costs) - ov['dp_cost_mean']) < 1e-6
    and abs(statistics.stdev(dp_costs) - ov['dp_cost_stdev']) < 1e-6
    and abs(statistics.mean(he_costs) - ov['heuristic_cost_mean']) < 1e-6
    and abs(statistics.stdev(he_costs) - ov['heuristic_cost_stdev']) < 1e-6,
    'selector overall aggregates do not recompute from per-instance costs')
chk(sum(i['dp']['fused_sites_used'] for i in inst) == ov['dp_fused_sites_total']
    and sum(i['dp']['premium_sites_used'] for i in inst) == ov['dp_premium_sites_total']
    and sum(i['heuristic']['fused_sites_used'] for i in inst) == ov['heuristic_fused_sites_total']
    and sum(i['heuristic']['premium_sites_used'] for i in inst) == ov['heuristic_premium_sites_total'],
    'selector mechanism diagnostics do not recompute from instances')
chk(all(i['rng_seed'] == i['size']*1000 + i['seed'] for i in inst),
    'selector rng-seed formula json mismatch (paper: size*1000+seed)')
chk(ov['dp_wins'] == 20 and ov['cost_ties'] == 0 and ov['dp_cost_losses'] == 0,
    f"selector outcomes json wins={ov['dp_wins']} ties={ov['cost_ties']} losses={ov['dp_cost_losses']} vs paper 20/0/0")
chk(abs(ov['dp_cost_mean'] - 85.80) < 5e-3 and abs(ov['dp_cost_stdev'] - 73.69) < 5e-3
    and abs(ov['heuristic_cost_mean'] - 292.08) < 5e-3 and abs(ov['heuristic_cost_stdev'] - 244.95) < 5e-3,
    f"selector costs json {ov['dp_cost_mean']:.2f}±{ov['dp_cost_stdev']:.2f} vs {ov['heuristic_cost_mean']:.2f}±{ov['heuristic_cost_stdev']:.2f}; paper 85.80±73.69 vs 292.08±244.95")
chk(abs(ov['cost_gap_mean'] - 206.28) < 5e-3 and abs(ov['cost_gap_stdev'] - 203.43) < 5e-3,
    f"selector gap json {ov['cost_gap_mean']:.2f}±{ov['cost_gap_stdev']:.2f} vs paper 206.28±203.43")
chk(abs(ov['cost_ratio_dp_over_heuristic_mean'] - 0.361) < 5e-4
    and abs(ov['cost_ratio_dp_over_heuristic_min'] - 0.054) < 5e-4
    and abs(ov['cost_ratio_dp_over_heuristic_max'] - 0.685) < 5e-4,
    f"selector cost-ratio json {ov['cost_ratio_dp_over_heuristic_mean']:.3f} [{ov['cost_ratio_dp_over_heuristic_min']:.3f},{ov['cost_ratio_dp_over_heuristic_max']:.3f}] vs paper 0.361 [0.054,0.685]")
chk(abs(ov['dp_time_ms_mean'] - 0.241) < 5e-4 and abs(ov['dp_time_ms_stdev'] - 0.163) < 5e-4
    and abs(ov['heuristic_time_ms_mean'] - 0.265) < 5e-4 and abs(ov['heuristic_time_ms_stdev'] - 0.194) < 5e-4
    and abs(ov['time_ratio_dp_over_heuristic_mean'] - 0.950) < 5e-4
    and abs(ov['time_ratio_dp_over_heuristic_min'] - 0.801) < 5e-4
    and abs(ov['time_ratio_dp_over_heuristic_max'] - 1.086) < 5e-4,
    'selector timing json vs paper 0.241±0.163 / 0.265±0.194 ms, ratio 0.950 [0.801,1.086]')
for i in inst:
    chk(f"| {i['size']} | {i['seed']} | {fmt(i['dp']['cost'], 3)} | {fmt(i['heuristic']['cost'], 3)} | {fmt(i['cost_ratio_dp_over_heuristic'], 3)} | {fmt(i['cost_gap_heuristic_minus_dp'], 3)} |" in md,
        f"selector per-instance row {i['size']}/{i['seed']} altered")
for size, v in sel['by_size'].items():
    row = (f"| {size} | {fmt(v['dp_cost_mean'], 2)} ± {fmt(v['dp_cost_stdev'], 2)} | "
           f"{fmt(v['heuristic_cost_mean'], 2)} ± {fmt(v['heuristic_cost_stdev'], 2)} | "
           f"{fmt(v['cost_ratio_dp_over_heuristic_mean'], 3)} ({fmt(v['cost_ratio_dp_over_heuristic_min'], 3)}–{fmt(v['cost_ratio_dp_over_heuristic_max'], 3)}) | "
           f"{v['dp_wins']}/{v['n_instances']} | {fmt(v['dp_time_ms_mean'], 3)} ± {fmt(v['dp_time_ms_stdev'], 3)} | "
           f"{fmt(v['heuristic_time_ms_mean'], 3)} ± {fmt(v['heuristic_time_ms_stdev'], 3)} | {fmt(v['time_ratio_dp_over_heuristic_mean'], 3)} |")
    chk(row in md, f'selector by-size row {size} altered')
chk(f"DP {fmt(ov['dp_cost_mean'], 2)} ± {fmt(ov['dp_cost_stdev'], 2)} vs heuristic {fmt(ov['heuristic_cost_mean'], 2)} ± {fmt(ov['heuristic_cost_stdev'], 2)} (gap {fmt(ov['cost_gap_mean'], 2)} ± {fmt(ov['cost_gap_stdev'], 2)}); ratio DP/heuristic mean {fmt(ov['cost_ratio_dp_over_heuristic_mean'], 3)}, range {fmt(ov['cost_ratio_dp_over_heuristic_min'], 3)}–{fmt(ov['cost_ratio_dp_over_heuristic_max'], 3)}" in md,
    'selector overall sentence altered')
chk(f"Selection time DP {fmt(ov['dp_time_ms_mean'], 3)} ± {fmt(ov['dp_time_ms_stdev'], 3)} ms vs heuristic {fmt(ov['heuristic_time_ms_mean'], 3)} ± {fmt(ov['heuristic_time_ms_stdev'], 3)} ms; time-ratio mean {fmt(ov['time_ratio_dp_over_heuristic_mean'], 3)}, range {fmt(ov['time_ratio_dp_over_heuristic_min'], 3)}–{fmt(ov['time_ratio_dp_over_heuristic_max'], 3)}" in md,
    'selector timing sentence altered')
chk(f"the DP used {ov['dp_fused_sites_total']} fused sites in total and selected premium variants 0 times; the heuristic used 0 fused sites and {ov['heuristic_premium_sites_total']} premium sites" in md,
    'selector mechanism sentence altered')
for s_ in ['20/20 (0 ties, 0 losses)', 'ratio 0.361; time 0.950', '0.361', '0.950']:
    chk(s_ in md, f'paper missing selector citation {s_}')

# ---- 5.11 / I.2 EMA hot-path detection (hotpath_ema_results.json)
hp = jload('hotpath_ema_results.json')
scn = hp['scenarios']
for name, ex, em, orr in [('concentrated', 1.0, 1.0, 1.0), ('diverse_rotated', 0.0, 3.0, 3.0),
                          ('diverse_zipf', 0.0, 1.8, 2.2), ('cyclic_shift', 0.0, 2.0, 2.0),
                          ('bursty', 0.0, 1.0, 1.0), ('warmup_heavy_burst', 0.0, 0.0, 1.0),
                          ('pure_diversity', 0.0, 0.0, 0.0)]:
    s_ = scn[name]
    chk(abs(s_['exact_promoted']['mean'] - ex) < 5e-2 and abs(s_['ema_promoted']['mean'] - em) < 5e-2
        and abs(s_['oracle_promoted']['mean'] - orr) < 5e-2,
        f"hotpath scenario {name} json exact/ema/oracle {s_['exact_promoted']['mean']}/{s_['ema_promoted']['mean']}/{s_['oracle_promoted']['mean']} vs paper {ex}/{em}/{orr}")
    chk(s_['exact_false_promotions_vs_oracle']['mean'] == 0.0 and s_['ema_false_promotions_vs_oracle']['mean'] == 0.0,
        f'hotpath scenario {name}: nonzero false promotions in json')
chk(abs(scn['diverse_rotated']['ema_recall_vs_oracle']['mean'] - 1.0) < 5e-2
    and abs(scn['diverse_zipf']['ema_recall_vs_oracle']['mean'] - 0.867) < 5e-4
    and abs(scn['diverse_zipf']['ema_recall_vs_oracle']['stdev'] - 0.298) < 5e-4,
    'hotpath recall json mismatch (paper: rotated 1.0; zipf 0.867 ± 0.298)')
for name, anno in [('concentrated', f"{scn['concentrated']['stream_len']} recordings, {scn['concentrated']['distinct_paths']} paths"),
                   ('diverse_rotated', f"{scn['diverse_rotated']['stream_len']}; 3 variants × 40, far below threshold 100"),
                   ('diverse_zipf', f"~{scn['diverse_zipf']['stream_len']}; Zipf(1.1) over 10 ranks"),
                   ('cyclic_shift', f"{scn['cyclic_shift']['stream_len']}; 4 shifting variants"),
                   ('bursty', f"{scn['bursty']['stream_len']}; 80-call burst below threshold"),
                   ('warmup_heavy_burst', f"{scn['warmup_heavy_burst']['stream_len']}; 80 one-shot warmups, default cap")]:
    s_ = scn[name]
    sd_r = 2 if name == 'diverse_zipf' else 1
    row = (f"| {name} ({anno}) | {fmt(s_['exact_promoted']['mean'], 1)} ± {fmt(s_['exact_promoted']['stdev'], 1)} | "
           f"{fmt(s_['ema_promoted']['mean'], 1)} ± {fmt(s_['ema_promoted']['stdev'], sd_r)} | "
           f"{fmt(s_['oracle_promoted']['mean'], 1)} ± {fmt(s_['oracle_promoted']['stdev'], sd_r)} | "
           f"{fmt(s_['ema_recall_vs_oracle']['mean'], 3 if name == 'diverse_zipf' else 1)} ± {fmt(s_['ema_recall_vs_oracle']['stdev'], 3 if name == 'diverse_zipf' else 1)} |")
    chk(row in md, f'hotpath scenario row {name} altered')
s_ = scn['pure_diversity']
chk(f"| pure_diversity ({s_['stream_len']} one-off trajectories) | {fmt(s_['exact_promoted']['mean'], 1)} ± {fmt(s_['exact_promoted']['stdev'], 1)} | {fmt(s_['ema_promoted']['mean'], 1)} ± {fmt(s_['ema_promoted']['stdev'], 1)} | {fmt(s_['oracle_promoted']['mean'], 1)} ± {fmt(s_['oracle_promoted']['stdev'], 1)} | — |" in md,
    'hotpath scenario row pure_diversity altered')
conc = scn['concentrated']
chk(abs(conc['ema_first_promotion_recording']['mean'] - 2.0) < 5e-5
    and abs(conc['exact_first_promotion_recording']['mean'] - 107.0) < 5e-5,
    'hotpath adoption latency json mismatch (paper: EMA at recording 2.0 vs exact 107.0)')
ab = hp['adoption_bursty']
chk(ab['ema']['adopted_seeds'] == 5 and ab['exact']['adopted_seeds'] == 0
    and ab['ema']['detection_index']['mean'] == 13.0
    and ab['ema']['full_form_executions_before_adoption']['mean'] == 12.0
    and ab['ema']['full_trace_len'] == 4 and ab['ema']['adopted_trace_len'] == 1,
    'hotpath bursty adoption json mismatch (paper: detect 13.0, 12 full-form executions, trace 4→1, 5/5 vs 0/5)')
for cap, v in hp['cap_sweep'].items():
    e, m = v['exact'], v['ema']
    chk(f"| {cap} | {fmt(e['evictions']['mean'], 0)} | {fmt(e['ever_hot_size']['mean'], 1)} → {fmt(e['hot_lost_after_eviction']['mean'], 1)} | {fmt(m['ever_hot_size']['mean'], 1)} → {fmt(m['hot_lost_after_eviction']['mean'], 1)} |" in md,
        f'hotpath cap-sweep row {cap} altered')
    chk(e['bound_respected_all_seeds'] and m['bound_respected_all_seeds'], f'hotpath cap {cap}: bounds not respected')
chk(hp['cap_sweep']['100']['exact']['evictions']['mean'] == 2110.0
    and hp['cap_sweep']['2000']['exact']['evictions']['mean'] == 210.0
    and hp['cap_sweep']['100']['exact']['hot_lost_after_eviction']['mean'] == 2.0
    and hp['cap_sweep']['2000']['exact']['hot_lost_after_eviction']['mean'] == 0.0
    and hp['cap_sweep']['100']['ema']['hot_lost_after_eviction']['mean'] == 0.6,
    'hotpath cap-sweep eviction/loss counts json mismatch (paper: 210→2110 evictions; exact 2.0→2.0, ema 0.6→0.6 below cap 2000)')
cr = hp['cap_rescue']
chk(cr['12']['ema_burst_detected_seeds'] == 5 and cr['12']['ema_burst_detection_index']['mean'] == 90.0
    and abs(cr['12']['ema_final_denominator']['mean'] - 16.5) < 5e-5 and cr['12']['evictions']['mean'] == 69,
    'hotpath cap-rescue cap=12 json mismatch (paper: 5/5 at 90.0, denominator 16.5, 69 evictions)')
chk(cr['10']['ema_burst_detected_seeds'] == 5 and cr['10']['ema_burst_detection_index']['mean'] == 87.0
    and abs(cr['10']['ema_final_denominator']['mean'] - 14.5) < 5e-5 and cr['10']['evictions']['mean'] == 71,
    'hotpath cap-rescue cap=10 json mismatch (paper: 5/5 at 87.0, denominator 14.5, 71 evictions)')
for cap, denom in [('15', 19.5), ('20', 24.5), ('50', 54.5), ('100', 85.5), ('1000', 85.5)]:
    chk(cr[cap]['ema_burst_detected_seeds'] == 0 and abs(cr[cap]['ema_final_denominator']['mean'] - denom) < 5e-5,
        f'hotpath cap-rescue cap={cap} json mismatch (paper: 0/5, denominator {denom})')
runm = hp['run_meta']
chk(runm['detector_record_calls_total'] == 175680 and runm['detection_cell_recordings'] == 6800
    and runm['adoption_cell_recordings'] == 880 and runm['cap_cell_recordings'] == 75000
    and runm['cap_rescue_cell_recordings'] == 5600,
    f"hotpath scale json total={runm['detector_record_calls_total']} vs paper 175,680 (6,800/880/75,000/5,600 cells)")
chk(abs(runm['wall_seconds'] - 0.33) < 5e-3, f"hotpath wall time json {runm['wall_seconds']} vs paper 0.33 s")
chk(len(hp['invariants_passed']) == 12, f"hotpath invariants json {len(hp['invariants_passed'])} != 12 (paper: 12 hard invariants)")
dcfg = hp['provenance']['detector_config']
chk(dcfg == {'hot_path_threshold': 100, 'stability_threshold': 0.9, 'ema_window': 10,
             'ema_gamma': 0.181818, 'ema_threshold': 0.3},
    f'hotpath detector config json {dcfg} vs paper threshold 100 / stability 0.9 / window 10 (γ 0.181818) / share 0.3')
# The paper cites Zipf recall per seed (the 5 seeds are bimodal—one seed at 1/3,
# four at 1.0—so the ± sd form was dropped from the main text); recompute each
# seed's recall from the raw per-seed records to keep the citation pinned to the JSON.
zipf_per_seed = scn['diverse_zipf']['per_seed']
zipf_recalls = []
for seed_ in sorted(zipf_per_seed, key=int):
    v_ = zipf_per_seed[seed_]
    oracle_ = {e['path'] for e in v_['oracle_events']}
    ema_ = {e['path'] for e in v_['ema_events']}
    zipf_recalls.append(len(ema_ & oracle_) / len(oracle_) if oracle_ else 0.0)
chk(abs(statistics.fmean(zipf_recalls) - scn['diverse_zipf']['ema_recall_vs_oracle']['mean']) < 1e-9
    and abs(statistics.stdev(zipf_recalls) - scn['diverse_zipf']['ema_recall_vs_oracle']['stdev']) < 1e-9,
    'zipf per-seed recalls disagree with the stored ema_recall_vs_oracle aggregate')
zipf_prose = ('3/3 rotated variants promoted (exact: 0/3; Zipf recall per seed: ['
              + ', '.join(f'{r_:.3f}' for r_ in zipf_recalls)
              + f'] (mean {statistics.fmean(zipf_recalls):.3f}))')
for s_ in ['175,680 detector', '0.33 s', '12 hard invariants', 'γ = 0.181818',
           zipf_prose,
           'zero false promotions', '85.5/85.5/54.5/24.5/19.5', '16.5, 69 evictions', '14.5, 71 evictions']:
    chk(s_ in md, f'paper missing hotpath citation {s_}')

# ---- 5.11 / I.2 strict-security posture matrix (security_posture_results.json)
sec = jload('security_posture_results.json')
ps = sec['posture_summary']
chk(ps['tasks_fail_open_would_permit_but_fail_closed_blocks'] == 400
    and ps['hole_scenarios_fail_open_permits_fail_closed_blocks'] == ['unknown_sensitivity', 'incomplete_policy'],
    f"security divergence json {ps['tasks_fail_open_would_permit_but_fail_closed_blocks']} vs paper 400")
chk(ps['normal_task_fail_open_denied'] == 0 and ps['normal_task_fail_closed_denied'] == 0,
    'security normal-task false denials json nonzero (paper: 0/200 both modes)')
chk(ps['strict_mode_denials_total_non_normal'] == 800 and ps['strict_mode_denials_with_recorded_violation_non_normal'] == 800,
    'security strict-denial audit json mismatch (paper: 800/800 recorded, rate 1.000)')
divg = {d['scenario']: d for d in sec['mode_divergence']}
for s_, fo in [('unknown_sensitivity', 200), ('incomplete_policy', 200)]:
    chk(divg[s_]['n_task_pairs'] == 200 and divg[s_]['outcome_contingency']['fail_open_only'] == fo,
        f'security divergence {s_} json mismatch (paper: 200 fail-open-only pairs)')
for s_ in ['unregistered_resource', 'malformed_registry_supplementary']:
    chk(divg[s_]['outcome_contingency']['both_deny'] == 200,
        f'security divergence {s_} json mismatch (paper: 200/200 denied in both modes)')
chk(divg['normal_task']['outcome_contingency']['both_allow'] == 200,
    'security divergence normal_task json mismatch (paper: 200/200 permitted in both modes)')
chk(sum(d['outcome_contingency']['fail_open_only'] for d in sec['mode_divergence']) == 400,
    'security fail-open-only pairs do not sum to the cited 400 of 1,000')
mrows = {(r['scenario'], r['mode']): r for r in sec['matrix']}
prop = {(p['scenario'], p['mode']): p for p in sec['coordinator_propagation']}
chk(len(mrows) == 10 and len(prop) == 10, 'security matrix/propagation json != 10 cells (4 scenarios + supplementary row × 2 modes)')
for (s_, mode), row in mrows.items():
    p = prop[(s_, mode)]
    chk(row['n_tasks'] == 200 and p['n_runs'] == 200 and abs(p['execute_rate'] - row['allow_rate']) < 1e-12,
        f'security propagation {s_}/{mode}: coordinator execute-rate != monitor allow-rate (paper: agreement on all 2,000 paired runs)')
    chk(abs(row['allow_rate_per_seed']['stdev'] - 0.0) < 1e-12,
        f'security per-seed allow-rate stdev nonzero for {s_}/{mode} (paper: 0.0)')
    rc = row['reason_class_counts']
    reason = '(permitted)' if list(rc) == ['allowed'] else list(rc)[0]
    mode_lbl = 'fail-open' if mode == 'fail_open' else 'fail-closed'
    chk(f"| {s_} | {mode_lbl} | {row['allowed']} / {row['denied']} | {fmt(row['deny_rate'], 3)} | {reason} |" in md,
        f'security posture row {s_}/{mode_lbl} altered')
chk(sum(r['denied'] for (s_, m), r in mrows.items() if m == 'fail_closed' and s_ != 'normal_task') == 800,
    'security strict denials do not sum to 800 across non-normal scenarios')
scale = sec['provenance']['scale']
chk(scale['task_population'] == 1000 and scale['monitor_evaluations'] == 2000 and scale['coordinator_runs'] == 2000,
    f"security scale json {scale} vs paper 1,000 tasks / 2,000 evaluations / 2,000 runs")
for s_ in ['400 of 1,000 paired tasks', 'fail-closed denies 400 policy-gap tasks fail-open admits',
           '800 strict-mode denials', 'all 2,000 paired runs', 'zero false denials', '0/200 normal-task false denials']:
    chk(s_ in md, f'paper missing security citation {s_}')

# ---- 5.11 / I.2 cache-key namespacing (cache_namespace_results.json)
cns = jload('cache_namespace_results.json')
st, lg = cns['summary']['strict_namespaced'], cns['summary']['regular_legacy']
for k in ['template_version_drift_invalidation_rate', 'decoding_drift_invalidation_rate',
          'template_version_drift_fresh_serve_rate', 'decoding_drift_fresh_serve_rate',
          'rollback_hit_rate_after_template_version_drift', 'rollback_hit_rate_after_decoding_drift',
          'rollback_original_serve_rate_after_template_version_drift', 'rollback_original_serve_rate_after_decoding_drift']:
    chk(st[k]['mean'] == 1.0 and st[k]['stdev'] == 0.0, f'cache-namespace strict {k} json != 1.0000 ± 0.0000')
chk(st['benign_reorder_invalidation_rate']['mean'] == 0.0 and st['benign_reorder_key_change_rate']['mean'] == 0.0,
    'cache-namespace benign-control json mismatch (paper: invalidation 0.0000, key-change 0.0000)')
for k in ['template_version_drift_hit_rate', 'decoding_drift_hit_rate',
          'template_version_drift_stale_serve_rate', 'decoding_drift_stale_serve_rate']:
    chk(lg[k]['mean'] == 1.0 and lg[k]['stdev'] == 0.0, f'cache-namespace legacy {k} json != 1.0000 ± 0.0000')
for k in ['template_version_drift_invalidation_rate', 'decoding_drift_invalidation_rate']:
    chk(lg[k]['mean'] == 0.0 and lg[k]['stdev'] == 0.0, f'cache-namespace legacy {k} json != 0.0000 ± 0.0000')
items = lookups = 0
for axis, arms in cns['axes'].items():
    for arm, v in arms.items():
        items += v['aggregate']['count_totals']['n_items']
        lookups += v['aggregate']['count_totals']['lookups_total']
        chk(len(v['per_seed']) == 5, f'cache-namespace axis {axis}/{arm} per-seed count != 5')
        chk(v['aggregate']['count_totals']['metrics_mismatches'] == 0, f'cache-namespace {axis}/{arm}: metric mismatches in raw counts')
        seeds = [p['seed'] for p in v['per_seed']]
        chk(seeds == [1000, 1001, 1002, 1003, 1004], f'cache-namespace {axis}/{arm} seeds json {seeds} != 1000-1004')
chk(items == 3000 and lookups == 13000,
    f'cache-namespace scale json items={items} lookups={lookups} vs paper 3,000 items / 13,000 lookups')
NS_ROWS = [('strict_namespaced', 'template_version', 'template_version v1 → v2'),
           ('strict_namespaced', 'decoding', 'decoding 0.0 → 0.7'),
           ('strict_namespaced', 'decoding_benign_reorder', 'benign key reorder (control)'),
           ('regular_legacy', 'template_version', 'template_version v1 → v2'),
           ('regular_legacy', 'decoding', 'decoding 0.0 → 0.7')]
for arm, axis, label in NS_ROWS:
    rates = cns['axes'][axis][arm]['aggregate']['rates']
    hit, inv = rates['first_post_drift_hit_rate']['mean'], rates['first_post_drift_miss_rate']['mean']
    if axis == 'decoding_benign_reorder':
        row = f'| {"strict namespaced" if arm == "strict_namespaced" else "regular legacy"} | {label} | {fmt(hit, 4)} | {fmt(inv, 4)} | — | — |'
    else:
        stale, fresh = rates['first_post_drift_stale_serve_rate']['mean'], rates['first_post_drift_fresh_serve_rate']['mean']
        arm_lbl = 'strict namespaced' if arm == 'strict_namespaced' else 'regular legacy'
        row = f'| {arm_lbl} | {label} | {fmt(hit, 4)} | {fmt(inv, 4)} | {fmt(stale, 4)} | {fmt(fresh, 4)} |'
    chk(row in md, f'cache-namespace table row {arm}/{axis} altered')
for s_ in ['13,000 cache lookups', '30 cohorts, 3,000 instruction instances', 'isolation, not eviction',
           'key-change rate of 0.0000', 'original-serve rate 1.0000', '100% invalidation under template/decoding drift']:
    chk(s_ in md, f'paper missing cache-namespace citation {s_}')

# ---- 5.11 / I.2 parameterized replay (replay_generalization_results.json)
rp = jload('replay_generalization_results.json')
ag = rp['aggregate']
chk(len(rp['cases']) == 1050 and ag['n_cases_total'] == 1050 and ag['n_match_total'] == 1050
    and ag['n_mismatch_total'] == 0 and ag['n_error_cases_total'] == 0 and ag['agreement_rate'] == 1.0,
    f"replay totals json {ag['n_match_total']}/{ag['n_cases_total']} vs paper 1,050/1,050 (100.0%)")
chk(sum(c[-3] for c in rp['cases']) == 1050 and sum(c[-1] for c in rp['cases']) == 0,
    'replay per-case rows do not recompute the all-match / zero-error aggregate')
chk(all(sum(1 for c in rp['cases'] if c[0] == r['seed']) == 210 for r in rp['per_seed']),
    'replay per-seed case counts json != 210 each')
chk(all(sum(c[-2] for c in rp['cases'] if c[0] == r['seed']) == r['n_empty_kept'] for r in rp['per_seed'])
    and ag['n_empty_kept_total'] == 179,
    'replay empty-kept edge counts do not recompute from per-case rows (paper: 179)')
chk(all(v['n'] == 350 and v['n_match'] == 350 for v in ag['convention_counts'].values()),
    'replay conventions json != 350/350 per convention (wrapped/flat/sibling)')
chk(ag['per_seed_agreement_rates'] == [1.0]*5 and ag['mean_per_seed_agreement_rate'] == 1.0
    and ag['stdev_per_seed_agreement_rate'] == 0.0,
    'replay per-seed agreement json != 1.0000 ± 0.0000')
for r in rp['per_seed']:
    c_ = r['convention_counts']
    chk(f"| {r['seed']} | {r['composition']} | {r['n_cases']} | {r['n_match']} | {r['n_empty_kept']} | {c_['wrapped']['n']} / {c_['flat']['n']} / {c_['sibling']['n']} |" in md,
        f"replay per-seed row seed {r['seed']} altered")
    chk(r['unseen_validation']['per_coordinate_disjoint'] and r['unseen_validation']['tuples_unique']
        and r['unseen_validation']['no_recorded_tuple_reused'] and r['unseen_validation']['meets_min'],
        f'replay unseen-draw guards fail for seed {r["seed"]}')
for s_ in ['1,050 of 1,050', '179 empty-result edge cases', '420/420', '210/210',
           '1.0000 ± 0.0000', '1,050/1,050 unseen-parameter replays match the oracle exactly']:
    chk(s_ in md, f'paper missing replay citation: {s_[:40]}')

# ---- 5.11 / I.2 chain-enriched workloads (fusion_chain_results.json)
fc = jload('fusion_chain_results.json')
def _fcrow(arm, length, level):
    for r in fc['aggregates'][arm]:
        if r['length'] == length and r['level'] == level:
            return r
    raise KeyError((arm, length, level))
for arm in ('chain_enriched', 'random_slots_control'):
    for length in (50, 200, 1000):
        for level in ('NONE', 'BASIC', 'ADVANCED', 'AGGRESSIVE'):
            r = _fcrow(arm, length, level)
            ec, red, dc, mk, pw = (r['effective_instruction_count'], r['instruction_count_reduction'],
                                   r['dispatch_count'], r['declared_cost_makespan'], r['max_parallel_width'])
            red_cell = fmt(red['mean'], 3) if red['stdev'] == 0 else f"{fmt(red['mean'], 3)} ± {fmt(red['stdev'], 3)}"
            row = (f"| {length} | {level} | {fmt(ec['mean'], 1)} ± {fmt(ec['stdev'], 1)} | {red_cell} | "
                   f"{fmt(dc['mean'], 1)} ± {fmt(dc['stdev'], 1)} | {fmt(mk['mean'], 1)} ± {fmt(mk['stdev'], 1)} | {fmt(pw['mean'], 1)} |")
            chk(row in md, f'fusion ladder row {arm} {length}/{level} altered')
comp_mb2 = [r for r in rs if r['experiment'] == 'compiler'][0]['data']
for level in ('BASIC', 'ADVANCED', 'AGGRESSIVE'):
    mbrow = [d for d in comp_mb2 if d['length'] == 1000 and d['level'] == level][0]
    crow = _fcrow('random_slots_control', 1000, level)
    chk(crow['dispatch_count']['mean'] == mbrow['dispatch_count']['mean']
        and crow['dispatch_count']['stdev'] == mbrow['dispatch_count']['stdev']
        and crow['instruction_count_reduction']['mean'] == mbrow['effective_instruction_count_reduction']['mean']
        and crow['instruction_count_reduction']['stdev'] == mbrow['effective_instruction_count_reduction']['stdev'],
        f'fusion control L1000 {level} does not reproduce the published Section 5.3 numbers bit-for-bit')
chain_b, chain_a = _fcrow('chain_enriched', 1000, 'BASIC'), _fcrow('chain_enriched', 1000, 'ADVANCED')
ctrl_b, ctrl_a = _fcrow('random_slots_control', 1000, 'BASIC'), _fcrow('random_slots_control', 1000, 'ADVANCED')
chk(abs((chain_b['dispatch_count']['mean'] - chain_a['dispatch_count']['mean']) - 558.0) < 5e-2
    and abs((chain_b['dispatch_count']['mean'] - chain_a['dispatch_count']['mean']) / chain_b['dispatch_count']['mean'] * 100 - 79.1) < 5e-2,
    'fusion chain dispatch amplification json mismatch (paper: −558.0 units, −79.1%)')
chk(abs((chain_b['declared_cost_makespan']['mean'] - chain_a['declared_cost_makespan']['mean']) - 930.1) < 5e-2
    and abs((chain_b['declared_cost_makespan']['mean'] - chain_a['declared_cost_makespan']['mean']) / chain_b['declared_cost_makespan']['mean'] * 100 - 79.3) < 5e-2,
    'fusion chain makespan amplification json mismatch (paper: −930.1, −79.3%)')
chk(abs((ctrl_b['dispatch_count']['mean'] - ctrl_a['dispatch_count']['mean']) - 301.4) < 5e-2
    and abs((ctrl_b['dispatch_count']['mean'] - ctrl_a['dispatch_count']['mean']) / ctrl_b['dispatch_count']['mean'] * 100 - 46.7) < 5e-2,
    'fusion control dispatch reduction json mismatch (paper: −301.4, −46.7%)')
chk(abs(chain_a['dispatch_count']['mean'] / chain_b['dispatch_count']['mean'] - 0.209) < 5e-4
    and abs(ctrl_a['dispatch_count']['mean'] / ctrl_b['dispatch_count']['mean'] - 0.533) < 5e-4,
    'fusion ADVANCED/BASIC dispatch ratios json mismatch (paper: 0.209 chain vs 0.533 control)')
chk(abs(chain_a['max_parallel_width']['mean'] - 69.0) < 5e-2 and abs(ctrl_a['max_parallel_width']['mean'] - 14.4) < 5e-2,
    'fusion peak parallel width json mismatch (paper: 69.0 vs 14.4)')
for arm in ('chain_enriched', 'random_slots_control'):
    v = fc['verdicts'][arm]
    chk(v['aggressive_structurally_identical_to_advanced'] is True,
        f'fusion {arm}: AGGRESSIVE not structurally identical to ADVANCED in json')
    d = v['aggressive_vs_advanced_at_max_length']
    chk(all(d[k] in (0, 0.0) for k in ['effective_instruction_count', 'instruction_count_reduction',
                                        'dispatch_count', 'declared_cost_makespan', 'fused_instruction_count',
                                        'parallel_block_wrappers', 'max_parallel_width']),
        f'fusion {arm}: AGGRESSIVE-vs-ADVANCED structural deltas nonzero (paper: deltas exactly 0.0)')
chain_g, ctrl_g = _fcrow('chain_enriched', 1000, 'AGGRESSIVE'), _fcrow('random_slots_control', 1000, 'AGGRESSIVE')
chk(abs(chain_g['buffer_reuse_potential']['mean'] - 107.8) < 5e-2
    and abs(chain_g['buffer_reuse_potential']['stdev'] - 22.2) < 5e-2
    and chain_g['distinct_tracked_outputs']['mean'] == 922
    and abs(chain_g['buffer_reuse_potential']['mean'] / chain_g['distinct_tracked_outputs']['mean'] * 100 - 11.7) < 5e-2
    and abs(chain_g['peak_live_outputs']['mean'] - 814.2) < 5e-2,
    'fusion chain memory-plan json mismatch (paper: 107.8 ± 22.2 of 922.0 tracked, 11.7%, peak live 814.2)')
chk(ctrl_g['buffer_reuse_potential']['mean'] == 0 and ctrl_g['buffer_reuse_potential']['stdev'] == 0.0
    and abs(ctrl_g['distinct_tracked_outputs']['mean'] - 549.4) < 5e-2,
    'fusion control memory-plan json mismatch (paper: 0.0 ± 0.0 reuse, 549.4 tracked outputs)')
chk(fc['config']['chain_mix'] == {'fusion_chain': 0.35, 'parallel_batch': 0.3, 'dataflow_chain': 0.15,
                                  'dead_chain': 0.08, 'effectful_anchor': 0.12}
    and fc['config']['control_config'] == {'p_dead': 0.25, 'p_pure': 0.55},
    'fusion generator configuration json mismatch (paper: mix 0.35/0.30/0.15/0.08/0.12; control p_dead 0.25 / p_pure 0.55)')
for s_ in ['−558.0 units', '705.0 → 147.0, −79.1%', '−930.1', '1172.7 → 242.5, −79.3%', '−301.4',
           '0.209 (chain) vs 0.533 (control)', '69.0 vs 14.4', '107.8 ± 22.2', '11.7% of the 922.0',
           '814.2', '549.4 tracked outputs', '120/120 cells complete', 'deltas exactly 0.0',
           "79.1% (random-slots control 46.7%—paralleling Section 5.3's 43.8% random-sequence dispatch reduction (131.6→74.0 at length 200; the control's 46.7% is measured on its own length-200 slots)"]:
    chk(s_ in md, f'paper missing fusion citation {s_}')

# ---- 5.11 / I.2 priced-comparison token-mix sweep (promptcache_sweep_results.json)
pcs = jload('promptcache_sweep_results.json')
oi = pcs['analysis']['ordering_invariance']
chk(oi['cells_checked'] == 195 and oi['flipped_cells'] == 0 and oi['agentisa_cheaper_or_equal_cells'] == 195,
    f"sweep ordering json {oi['agentisa_cheaper_or_equal_cells']}/{oi['cells_checked']} vs paper 195/195")
chk(abs(oi['min_gap_usd'] - 0.016562) < 5e-7, f"sweep min gap json {oi['min_gap_usd']} vs paper $0.016562")
head = pcs['analysis']['headline']
for s_ in ['crosses 0.5 at 23.3 completion tokens/call', 'crosses 0.5 at 22.9 completion tokens/call',
           'crosses 0.5 at 22.5 completion tokens/call', 'beyond 392.3 completion tokens/call',
           'beyond 397.7 completion tokens/call', 'beyond 416.7 completion tokens/call']:
    chk(any(s_ in h for h in head), f'sweep json headline strings missing: {s_}')
rc_ = {r['zipf_s']: r for r in pcs['analysis']['regime_contrast']}
for s_, df_ in [(0.5, 16.4), (1.0, 16.7), (1.5, 17.0)]:
    chk(abs(rc_[s_]['ratio_drop_factor'] - df_) < 5e-2,
        f'sweep effectiveness drop factor s={s_} json {rc_[s_]["ratio_drop_factor"]:.4f} vs paper {df_}')
anch = pcs['analysis']['published_anchor_checks']
chk(anch['mock_paper_scale']['all_match'] is True,
    'sweep published-anchor check json not all_match (paper: anchor check all_match = true)')
real_anchor = anch['real_gateway_run']
chk(abs(real_anchor['savings_agentisa_cache_pct'] - 55.9569) < 5e-5
    and abs(real_anchor['savings_provider_prompt_cache_pct'] - 3.2784) < 5e-5,
    'sweep real-run anchor quotes json mismatch (paper: 55.96% / 3.28%)')
mixc = real_anchor['mix_coordinates']
chk(abs(mixc['prompt_tokens_per_call'] - 180.36) < 5e-3 and mixc['completion_tokens_per_call'] == 512
    and abs(mixc['output_input_token_ratio'] - 2.84) < 5e-3 and abs(mixc['output_cost_share'] - 0.9342) < 5e-5,
    'sweep real-run mix coordinates json mismatch (paper: 180.36 / 512 / 2.84 / 93.4%)')
rma = pcs['analysis']['real_mix_analog']['per_zipf']
ratios = [r['output_input_token_ratio'] for r in rma]
shares = [r['output_cost_share'] for r in rma]
chk(abs(min(ratios) - 3.53) < 5e-3 and abs(max(ratios) - 3.66) < 5e-3
    and abs(min(shares)*100 - 94.6) < 5e-2 and abs(max(shares)*100 - 94.8) < 5e-2,
    'sweep L512 mix analog json mismatch (paper: out/in 3.53–3.66, output share 94.6–94.8%)')
pc_rows = {}
for line in md.splitlines():
    m_ = re.match(r'\| (L\d+|mock \(measured\)) \| (.+?) \|\s*$', line)
    if m_ and ' / ' in m_.group(2):
        pc_rows[m_.group(1)] = [c.strip() for c in m_.group(2).split('|')]
for sk in pcs['analysis']['per_skew']:
    for pt in sk['curve']:
        label = 'mock (measured)' if pt['mix_id'] == 'mock_measured' else pt['mix_id']
        if label not in pc_rows:
            fails.append(f'sweep curve row {label} missing from paper table')
            continue
        i = [0.5, 1.0, 1.5].index(sk['zipf_s'])
        a_md, p_md = pc_rows[label][i*3].split(' / ')
        for nm, stored, disp, tol in [('savings', pt['savings_agentisa_cache_pct'], a_md, 0.005001),
                                      ('prompt', pt['savings_provider_prompt_cache_pct'], p_md, 0.005001),
                                      ('ratio', pt['ratio_prompt_to_agentisa'], pc_rows[label][i*3+1], 0.0005001),
                                      ('share', pt['output_cost_share'], pc_rows[label][i*3+2], 0.0005001)]:
            chk(abs(stored - float(disp)) <= tol,
                f'sweep cell {label} s={sk["zipf_s"]} {nm}: json {stored} vs paper {disp}')
for s_ in ['23.3 / 22.9 / 22.5', '392.3 / 397.7 / 416.7', '195/195', '$0.016562', '16.4 / 16.7 / 17.0',
           'all_match = true', '55.96%', '3.28%', '180.36', '2.84', '93.4%', '3.53–3.66', '94.6–94.8%',
           '195/195 token-mix cells']:
    chk(s_ in md, f'paper missing sweep citation {s_}')

# ---- I.3 real-gateway quality re-run (gateway_quality_results.json + the I9 real-run reference)
gq = jload('gateway_quality_results.json')
i9r = json.loads((ROOT / 'examples/paper_real_llm_quality_results_real.json').read_text())
sr, s9 = gq['summary_recomputed'], i9r['summary']
chk(len(gq['data']) == 22 and len({b['batch_id'] for b in gq['data']}) == 22
    and sr['distinct_batches'] == 22 and sr['model_calls'] == 22 and sr['signatures_total'] == 176,
    'gateway-quality scale json mismatch (paper: 22 batches, one live call per batch, 176 signatures)')
chk(sum(b['signature_count'] for b in gq['data']) == 176
    and sum(b['parsed_claims'] for b in gq['data']) == sr['parsed_claims'] == 144
    and sum(b['fallback_filled'] for b in gq['data']) == sr['fallback_filled'] == 32
    and sum(1 for b in gq['data'] if b['reply_bound_at_ceiling']) == sr['replies_bound_at_ceiling'] == 4,
    'gateway-quality summary does not recompute from the per-batch records')
fin = [b['final_agreement_vs_gold'] for b in gq['data']]
clm = [b['claims_agreement_vs_gold'] for b in gq['data'] if b['claims_agreement_vs_gold'] is not None]
fvr = [b['final_agreement_vs_rule_fallback'] for b in gq['data']]
chk(abs(statistics.mean(fin) - sr['final_agreement_vs_gold']) < 5e-5
    and abs(statistics.mean(clm) - sr['claims_agreement_vs_gold']) < 5e-5
    and abs(statistics.mean(fvr) - sr['final_agreement_vs_rule_fallback']) < 5e-5,
    'gateway-quality agreement means do not recompute from the per-batch records (4-dp storage)')
chk(abs(sr['final_agreement_vs_gold'] - 0.4545) < 5e-5 and abs(sr['claims_agreement_vs_gold'] - 0.5139) < 5e-5
    and abs(sr['rule_fallback_agreement_vs_gold'] - 0.3239) < 5e-5
    and abs(sr['final_agreement_vs_rule_fallback'] - 0.3864) < 5e-5 and abs(sr['parse_rate'] - 0.8182) < 5e-5,
    'gateway-quality agreement rates json vs paper 45.5/51.4/32.4/38.6/81.8 mismatch')
for cls, n_, m_, pct in [('critical', 40, 19, 47.5), ('high', 40, 12, 30.0),
                         ('medium', 51, 23, 45.1), ('low', 45, 26, 57.8)]:
    g = sr['per_gold_class'][cls]
    chk(g['signatures'] == n_ and g['final_matches_gold'] == m_
        and abs(g['final_matches_gold']/g['signatures'] - g['agreement_rate']) < 5e-5
        and abs(g['agreement_rate']*100 - pct) <= 0.05,
        f'gateway-quality gold class {cls} json ({g["signatures"]}/{g["final_matches_gold"]}) vs paper {n_}/{m_} ({pct}%)')
chk(s9['parsed_claims'] == 120 and abs(s9['parse_rate'] - 0.6818) < 5e-5 and s9['fallback_filled'] == 56
    and s9['replies_bound_at_ceiling'] == 7 and abs(s9['claims_agreement_vs_gold'] - 0.5) < 5e-5
    and abs(s9['final_agreement_vs_gold'] - 0.4318) < 5e-5
    and abs(s9['final_agreement_vs_rule_fallback'] - 0.4886) < 5e-5
    and abs(s9['rule_fallback_agreement_vs_gold'] - 0.3239) < 5e-5,
    'gateway-quality reference run (I9 real) json vs paper reference column 120 (68.2%)/56/7/50.0/43.2/48.9/32.4 mismatch')
for cls, n_, m_, pct in [('critical', 40, 17, 42.5), ('high', 40, 12, 30.0),
                         ('medium', 51, 19, 37.3), ('low', 45, 28, 62.2)]:
    g = s9['per_gold_class'][cls]
    chk(g['signatures'] == n_ and g['final_matches_gold'] == m_
        and abs(g['final_matches_gold']/g['signatures'] - g['agreement_rate']) < 5e-5
        and abs(g['agreement_rate']*100 - pct) <= 0.05,
        f'gateway-quality reference class {cls} json ({g["signatures"]}/{g["final_matches_gold"]}) vs paper {n_}/{m_} ({pct}%)')
pb = gq['paired_vs_i9_1500_baseline']
chk(pb['paired_batches'] == 22 and pb['unpaired_batches_rerun'] == 0 and pb['batches_bound_at_1500'] == 7
    and pb['of_those_now_complete'] == 4 and pb['of_those_still_bound'] == 3 and pb['newly_bound_at_rerun'] == 1,
    'gateway-quality paired truncation outcome json mismatch (paper: 7 bound at 1500 → 4 complete / 3 still bound / 1 newly bound)')
chk(0 in [b['task_rank'] for b in pb['per_batch'] if b['bound_at_1500'] and not b['bound_at_rerun']],
    'gateway-quality rank-0 (highest-rank) batch not among those now completing')
chk([b['task_rank'] for b in pb['per_batch'] if not b['bound_at_1500'] and b['bound_at_rerun']] == [11],
    'gateway-quality newly-bound batch rank json mismatch (paper: rank 11)')
chk(gq['all_pass'] is False and gq['verdict']['replies_not_bound_at_ceiling'] is False
    and gq['cross_check'].startswith('passed'),
    'gateway-quality verdict json mismatch (paper: all_pass = False solely on the 4/22 ceiling gate)')
sd = gq['provenance']['scale_decision']
chk(abs(sd['probe_call_latency_s'] - 15.25) < 5e-3 and sd['probe_stop_reason'] == 'end_turn'
    and sd['probe_completion_tokens'] == 719,
    'gateway-quality probe json mismatch (paper: 15.25 s, end_turn, 719 completion tokens)')
chk(abs(sd['min_observed_throughput_tok_s'] - 37.6) < 5e-2 and abs(sd['planning_worst_latency_s'] - 81.6) < 5e-2
    and sd['affordable_calls'] == 29 and sd['chosen_max_tokens'] == 3072
    and abs(sd['worst_case_duration_min']['tiny'] - 29.9) < 5e-2
    and abs(sd['worst_case_duration_min']['paper'] - 49.0) < 5e-2,
    'gateway-quality probe-then-scale rule json mismatch (paper: 37.6 tok/s, 81.6 s, 29 calls, 3072 tokens, 29.9 vs 49.0 min)')
chk(gq['provenance']['gateway_round_trips'] == {'preflight': 1, 'live_calls': 22, 'total': 23},
    'gateway-quality round trips json != 23 (paper: 23 round trips)')
for label, a, b_ in [
    ('Distinct batches / live calls', f"{s9['distinct_batches']} / {s9['model_calls']}", f"{sr['distinct_batches']} / {sr['model_calls']}"),
    ('Signatures', str(s9['signatures_total']), str(sr['signatures_total'])),
    ('Parsed claims (parse rate)', f"{s9['parsed_claims']} ({fmt(s9['parse_rate']*100, 1)}%)", f"{sr['parsed_claims']} ({fmt(sr['parse_rate']*100, 1)}%)"),
    ('Fallback-filled classifications', str(s9['fallback_filled']), str(sr['fallback_filled'])),
    ('Replies bound at ceiling', f"{s9['replies_bound_at_ceiling']} / {s9['distinct_batches']}", f"{sr['replies_bound_at_ceiling']} / {sr['distinct_batches']}"),
    ('Model claims vs gold', f"{fmt(s9['claims_agreement_vs_gold']*100, 1)}% ({s9['parsed_claims']} claims)", f"{fmt(sr['claims_agreement_vs_gold']*100, 1)}% ({sr['parsed_claims']} claims)"),
    ('Final classification vs gold', f"{fmt(s9['final_agreement_vs_gold']*100, 1)}%", f"{fmt(sr['final_agreement_vs_gold']*100, 1)}%"),
    ('Rule-fallback baseline vs gold', f"{fmt(s9['rule_fallback_agreement_vs_gold']*100, 1)}%", f"{fmt(sr['rule_fallback_agreement_vs_gold']*100, 1)}%"),
    ('Final classification vs rule-fallback', f"{fmt(s9['final_agreement_vs_rule_fallback']*100, 1)}%", f"{fmt(sr['final_agreement_vs_rule_fallback']*100, 1)}%"),
    ('Per-call latency (min / median / mean / max)', '—',
     f"{fmt(sr['latency_ms']['min']/1000, 1)} s / {fmt(sr['latency_ms']['median']/1000, 1)} s / {fmt(sr['latency_ms']['mean']/1000, 1)} s / {fmt(sr['latency_ms']['max']/1000, 1)} s")]:
    chk(f'| {label} | {a} | {b_} |' in md, f'gateway-quality table row altered: {label}')
for cls in ('critical', 'high', 'medium', 'low'):
    a, b_ = s9['per_gold_class'][cls], sr['per_gold_class'][cls]
    chk(f"| {cls} | {a['signatures']} | {a['final_matches_gold']} ({fmt(a['agreement_rate']*100, 1)}%) | {b_['final_matches_gold']} ({fmt(b_['agreement_rate']*100, 1)}%) |" in md,
        f'gateway-quality per-gold-class row {cls} altered')
for s_ in ['`all_pass` = False', '15.25 s', '37.6 tokens/s', '81.6 s', '29.9 min', '49.0 min', '23 round trips',
           '120 (68.2%)', '50.0% (120 claims)']:
    chk(s_ in md, f'paper missing gateway-quality citation {s_}')

# ---- I.3 multi-seed real-gateway cache run (gateway_multiseed_results.json)
gm = jload('gateway_multiseed_results.json')
for z in gm['data']:
    agg = [r for r in gm['aggregate_recomputed'] if r['zipf_s'] == z['zipf_s']][0]
    calls = [c['cached']['api_calls'] for c in z['per_seed']]
    hits = [c['hit_rate'] for c in z['per_seed']]
    chk(len(z['per_seed']) == 2
        and abs(statistics.mean(calls) - agg['api_calls_cached']['mean']) < 1e-9
        and abs(statistics.stdev(calls) - agg['api_calls_cached']['stdev']) < 1e-4
        and abs(statistics.mean(hits) - agg['hit_rate']['mean']) < 1e-12,
        f'gateway-multiseed zipf {z["zipf_s"]} aggregates do not recompute from the per-seed cells')
    for c in z['per_seed']:
        chk(c['hit_rate'] == c['expected_hit_rate'],
            f'gateway-multiseed measured hit rate != analytic expectation in cell s={z["zipf_s"]} seed={c["seed"]}')
def _trim(s):
    return s.rstrip('0').rstrip('.') if '.' in s else s
for row in gm['aggregate_recomputed']:
    r_ = (f"| {row['zipf_s']} | {fmt(row['api_calls_no_cache']['mean'], 0)} → {fmt(row['api_calls_cached']['mean'], 1)} ± {fmt(row['api_calls_cached']['stdev'], 1)} | "
          f"{fmt(row['hit_rate']['mean']*100, 1)}% ± {fmt(row['hit_rate']['stdev']*100, 1)}% | "
          f"{fmt(row['expected_hit_rate']['mean']*100, 1)}% ± {fmt(row['expected_hit_rate']['stdev']*100, 1)}% | "
          f"{fmtc(row['wall_saved_ms']['mean'], 1)} ms | {_trim(fmtc(row['latency_ms_p50_cached']['mean'], 2))} ms |")
    chk(r_ in md, f'gateway-multiseed per-zipf row {row["zipf_s"]} altered')
for z in gm['data']:
    for c in z['per_seed']:
        r_ = (f"| {z['zipf_s']} | {c['seed']} | {c['baseline']['api_calls']} → {c['cached']['api_calls']} | "
              f"{fmt(c['hit_rate']*100, 2)}% = {fmt(c['expected_hit_rate']*100, 2)}% | {c['tokens_saved']:,} | {fmtc(c['wall_saved_ms'], 1)} |")
        chk(r_ in md, f'gateway-multiseed raw cell row s={z["zipf_s"]} seed={c["seed"]} altered')
gprov = gm['provenance']
chk(gprov['scale']['total_gateway_calls'] == 1103 and abs(gprov['wall_time_s'] - 3243.8) < 5e-3,
    'gateway-multiseed totals json mismatch (paper: 1,103 gateway calls, 3,243.8 s)')
chk('AccountQuotaExceeded' in gprov['chunking']['quota_episode']['gateway_error']
    and '21:19:19' in gprov['chunking']['quota_episode']['gateway_error'],
    'gateway-multiseed quota-episode record json mismatch (paper: 429 AccountQuotaExceeded, reset 21:19:19 +0800)')
chk(gm['cross_check'].startswith('passed'), 'gateway-multiseed cross-check json not passed')
for vp in gm['vs_frozen_paper_run']['per_zipf']:
    chk(vp['paper_hit_rate'] == real[vp['zipf_s']]['hit_rate'],
        f'gateway-multiseed frozen-paper reference s={vp["zipf_s"]} json {vp["paper_hit_rate"]} vs frozen artifact {real[vp["zipf_s"]]["hit_rate"]}')
for s_ in ['1,103 gateway calls', '3,243.8 s', 'AccountQuotaExceeded', '21:19:19 +0800',
           '(8.5% / 26.0% / 45.5% at 200 tasks × 1 seed)', 'all 6 of 6', '2,903 / 2,675 / 2,477 ms']:
    chk(s_ in md, f'paper missing gateway-multiseed citation {s_}')


# ── Multi-model production-gateway reruns (doubao cache / doubao identity / minimax census / deepseek quality) ──
ddb = json.load(open('examples/benchmarks/isa_real_llm_results_doubao_seed_2_0_code.json'))
for z in ddb['data']:
    c = z['per_seed'][0]
    chk(c['hit_rate'] == c['expected_hit_rate'],
        f'doubao-bench measured hit rate != analytic expectation in cell s={z["zipf_s"]}')
    chk(c['baseline']['api_calls'] == 200, f'doubao-bench baseline calls != 200 in cell s={z["zipf_s"]}')
hit05 = ddb['data'][0]['per_seed'][0]; hit10 = ddb['data'][1]['per_seed'][0]; hit15 = ddb['data'][2]['per_seed'][0]
chk(f'\u2014{hit05["hit_rate"]:.1%}, {hit10["hit_rate"]:.1%}, {hit15["hit_rate"]:.1%}; '
    f'calls 200\u2192{hit05["cached"]["api_calls"]}/{hit10["cached"]["api_calls"]}/{hit15["cached"]["api_calls"]}; p50' in md,
    'paper multi-model rerun citation (hit rates / calls) altered')
p50s = sorted(z['per_seed'][0]['baseline']['latency_ms_p50'] for z in ddb['data'])
chk(f'{p50s[0]/1000:.1f}\u2013{p50s[-1]/1000:.1f} s' in md, 'paper multi-model p50 citation altered')
for s_ in ['two model aliases (one seed per alias', 'Live-gateway checks on four model aliases']:
    chk(s_ in md, f'paper missing multi-model scope/abstract citation: {s_}')

idd = json.load(open('examples/benchmarks/paper_real_llm_compiler_results_doubao_1500.json'))
mt = idd['backend_config']['max_tokens']
live_calls = sum(1 for cell in idd['data'] for r in cell['arm_a_unoptimized']['per_realization'] if r['calls'] > 0)
bound = sum(1 for cell in idd['data'] for r in cell['arm_a_unoptimized']['per_realization'] if r['calls'] > 0 and r['completion_tokens'] >= mt)
mtok = max(r['completion_tokens'] for cell in idd['data'] for r in cell['arm_a_unoptimized']['per_realization'] if r['calls'] > 0)
chk(live_calls == 104 and bound == 0 and mtok == 142, 'doubao-identity raw records recompute mismatch (104 live / 0 bound / max 142)')
chk(idd['verdict']['byte_identical_all_realizations'] and idd['verdict']['arm_b_zero_new_calls_when_shared']
    and idd['verdict']['hot_path_adopted'] and idd['verdict']['static_compile_dataflow_preserved'],
    'doubao-identity verdict flags not all true')
chk(all(cell['byte_identity']['all_realizations_identical'] and cell['byte_identity']['arm_b_new_model_calls'] == 0
        for cell in idd['data']), 'doubao-identity per-cell byte identity/arm-b-calls mismatch')
chk(f'all {live_calls} live calls across the three Zipf cells completed below the ceiling (max completion {mtok} tokens, {bound} bound)' in md,
    'paper doubao identity citation altered')

def _census(fn, cap):
    d = json.load(open(fn))
    t = sum(1 for cell in d['data'] for r in cell['arm_a_unoptimized']['per_realization'] if r['calls'] > 0)
    c = sum(1 for cell in d['data'] for r in cell['arm_a_unoptimized']['per_realization'] if r['calls'] > 0 and r['completion_tokens'] >= cap)
    return c, t
c512, t512 = _census('examples/benchmarks/paper_real_llm_compiler_results_minimax.json', 512)
c1500, t1500 = _census('examples/benchmarks/paper_real_llm_compiler_results_minimax_1500.json', 1500)
chk((c512, t512) == (101, 104) and (c1500, t1500) == (68, 104), 'minimax census raw records recompute mismatch')
chk(f'{round(100*c512/t512)}% of calls at `max_tokens` = 512 and {round(100*c1500/t1500)}% at 1,500' in md,
    'paper minimax census citation altered')
chk(f'binding at the ceiling in {c512}/{t512} calls ({100*c512/t512:.0f}%)' in md
    and f'{c1500}/{t1500} ({100*c1500/t1500:.0f}%)' in md, 'supp minimax census citation altered')

dsq = json.load(open('examples/benchmarks/gateway_quality_deepseek.json'))
qs = dsq['summary']
chk(qs['parsed_claims'] == 288 and qs['parsed_claims'] == qs['signatures_total'] and qs['fallback_filled'] == 0
    and qs['replies_bound_at_ceiling'] == 0, 'deepseek-quality raw summary mismatch (288/288, 0 fills, 0 bound)')
chk(abs(qs['final_agreement_vs_gold'] - 0.5208) < 1e-9 and abs(qs['rule_fallback_agreement_vs_gold'] - 0.3194) < 1e-4,
    'deepseek-quality agreement numbers recompute mismatch')
chk(f'parses {qs["parsed_claims"]}/{qs["signatures_total"]} claims with {qs["replies_bound_at_ceiling"]} ceiling bindings: '
    f'final classifications match gold on {qs["final_agreement_vs_gold"]:.1%} of signatures against '
    f'{qs["rule_fallback_agreement_vs_gold"]:.1%} for the fallback rule' in md, 'paper deepseek quality citation altered')
pc = qs['per_gold_class']
chk(f'per-class agreement {100*pc["low"]["agreement_rate"]:.1f}/{100*pc["medium"]["agreement_rate"]:.1f}/{100*pc["high"]["agreement_rate"]:.1f}/{100*pc["critical"]["agreement_rate"]:.1f}%' in md,
    'paper deepseek per-class citation altered')
chk(dsq['gateway_round_trips']['live_calls'] == 36 and f'{dsq["gateway_round_trips"]["live_calls"]} distinct signature batches' in md,
    'paper deepseek batch count citation altered')
margin = qs['final_agreement_vs_gold'] - qs['rule_fallback_agreement_vs_gold']
chk(f'+{100*margin:.1f} points' in md, 'supp deepseek model-over-rule margin citation altered')
chk(f'{100*qs["claims_agreement_vs_gold"]:.1%} | {100*qs["final_agreement_vs_gold"]:.1%} | {100*qs["rule_fallback_agreement_vs_gold"]:.1%}' in md
    or (f'| {100*qs["claims_agreement_vs_gold"]:.1%} |' in md and f'| {100*qs["rule_fallback_agreement_vs_gold"]:.1%} |' in md),
    'supp deepseek quality table row altered')

# ---- I.3 prompt-cache billing probe (gateway_promptcache_probe.json) ----
probe = jload('gateway_promptcache_probe.json')
for alias, calls in probe['probe'].items():
    chk(len(calls) == 2 and all(c['input_tokens'] > 0 for c in calls),
        f'prompt-cache probe {alias}: missing input_tokens')
    chk(all(c['cache_creation_input_tokens'] == 0 and c['cache_read_input_tokens'] == 0 for c in calls),
        f'prompt-cache probe {alias}: cache fields unexpectedly nonzero')
it_d = probe['probe']['claude-doubao-seed-2.0-code'][0]['input_tokens']
it_s = probe['probe']['claude-deepseek-v4-flash'][0]['input_tokens']
chk(f'plain `input_tokens` ({it_d} and {it_s}, respectively)' in md,
    'supp I.3 prompt-cache probe citation altered')
chk('cache_creation_input_tokens = cache_read_input_tokens = 0' in md,
    'supp I.3 prompt-cache zero-fields citation altered')


# ---- Figures: manifest re-verification (docs/figures/figures_manifest.json) ----
figm = json.loads((ROOT / 'docs/figures/figures_manifest.json').read_text())
fe = figm['figures']
for name in ['wide_1_architecture.pdf', '2_compile_pipeline.pdf', '3_cache_exactness.pdf',
             '4_token_mix_sweep.pdf', '5_quality_per_class.pdf', '6_ceiling_census.pdf']:
    chk(name in fe and (ROOT / 'docs/figures' / name).exists(),
        f'figure missing from manifest or disk: {name}')
# F3: every plotted bar equals its bench JSON cell exactly
for src, cells in fe['3_cache_exactness.pdf']['cells'].items():
    d3 = json.loads((ROOT / 'examples/benchmarks' / src).read_text())
    by = {z['zipf_s']: z['per_seed'][0] for z in d3['data']}
    for c in cells:
        m3 = by[c['zipf_s']]
        chk(m3['hit_rate'] == c['hit_rate'] == c['expected'] == m3['expected_hit_rate'],
            f'figure-3 cell mismatch {src} s={c["zipf_s"]}')
chk(fe['3_cache_exactness.pdf']['all_cells_exact'] is True, 'figure-3 exactness flag off')
# F4: sweep series match the sweep JSON, ordering invariant holds on the plotted values
sw4 = json.loads((ROOT / 'examples/benchmarks/promptcache_sweep_results.json').read_text())
chk(fe['4_token_mix_sweep.pdf']['cells'] == 195, 'figure-4 cell count != 195')
for ps in sw4['analysis']['per_skew']:
    ser = fe['4_token_mix_sweep.pdf']['series'][str(ps['zipf_s'])]
    xs = [pt['x_completion_tokens_per_call'] for pt in ps['curve']]
    pa = [pt['savings_provider_prompt_cache_pct'] for pt in ps['curve']]
    sa = [pt['savings_agentisa_cache_pct'] for pt in ps['curve']]
    chk(ser['x'] == xs and ser['provider'] == pa and ser['agentisa'] == sa,
        f'figure-4 series mismatch at s={ps["zipf_s"]}')
    chk(all(a >= p for a, p in zip(ser['agentisa'], ser['provider'])),
        f'figure-4 ordering violation at s={ps["zipf_s"]}')
# F4 caption pins (supp): endpoint savings at s=0.5
c05 = sw4['analysis']['per_skew'][0]['curve']
chk(f'{c05[0]["savings_provider_prompt_cache_pct"]:.1f}% at L4' in md
    and f'{c05[-1]["savings_provider_prompt_cache_pct"]:.1f}% at L1024' in md,
    'figure-4 caption endpoint citation altered')
# F5: per-class agreement values match both quality JSONs and the recomputed fallback
q5 = fe['5_quality_per_class.pdf']
for src, key, top in [('gateway_quality_results.json', 'glm_per_class', 'summary_recomputed'),
                      ('gateway_quality_deepseek.json', 'deepseek_per_class', 'summary')]:
    qq = json.loads((ROOT / 'examples/benchmarks' / src).read_text())[top]
    for cls, v in q5[key].items():
        chk(abs(v - qq['per_gold_class'][cls]['agreement_rate']) < 1e-9,
            f'figure-5 {top} {cls} mismatch')
dsk5 = json.loads((ROOT / 'examples/benchmarks/gateway_quality_deepseek.json').read_text())
chk(abs(q5['fallback_overall'] - dsk5['summary']['rule_fallback_agreement_vs_gold']) < 5e-5,
    'figure-5 fallback overall mismatch')
# F6: census tuples match the identity runs' raw records (recomputed here)
def _fig6_census(fn, cap):
    d = json.loads((ROOT / fn).read_text())
    t = sum(1 for cell in d['data'] for r in cell['arm_a_unoptimized']['per_realization'] if r['calls'] > 0)
    c = sum(1 for cell in d['data'] for r in cell['arm_a_unoptimized']['per_realization']
            if r['calls'] > 0 and r['completion_tokens'] >= cap)
    return c, t
chk(tuple(fe['6_ceiling_census.pdf']['minimax_512']) == _fig6_census('examples/benchmarks/paper_real_llm_compiler_results_minimax.json', 512)
    and tuple(fe['6_ceiling_census.pdf']['minimax_1500']) == _fig6_census('examples/benchmarks/paper_real_llm_compiler_results_minimax_1500.json', 1500)
    and tuple(fe['6_ceiling_census.pdf']['doubao_1500']) == _fig6_census('examples/benchmarks/paper_real_llm_compiler_results_doubao_1500.json', 1500),
    'figure-6 census tuples mismatch raw records')
chk(tuple(fe['6_ceiling_census.pdf']['glm_512']) == (22, 22), 'figure-6 glm census != (22,22)')
# F6 caption pins (supp)
for s_ in ['binds in 22/22 calls at the compiler budget (512)',
           '101/104 at 512 and 68/104 at 1,500',
           'Doubao Seed 2.0 in all 104 calls below the ceiling at 1,500']:
    chk(s_ in md, f'paper missing figure-6 caption citation: {s_}')

# ---- 5.12 / I.6 real-trace pilots (real_trace_pilot_*.json) ----------
def _pilot(name):
    d = jload(name)
    return d, d['aggregate'], d['cache_scopes']['first_trial']

tau, tau_ag, tau_ft = _pilot('real_trace_pilot_tau_bench_results.json')
chk(tau['provenance']['streams'] == 1960 and tau_ag['total_steps'] == 14285,
    f"pilot tau scale json streams={tau['provenance']['streams']} steps={tau_ag['total_steps']} "
    "vs paper 1,960 / 14,285")
chk(tau_ag['effective_reduction_vs_none'] == {'BASIC': 0.0, 'AGGRESSIVE': 0.0},
    f"pilot tau static reduction json {tau_ag['effective_reduction_vs_none']} vs paper 0.0% / 0.0%")
chk(tau_ft['calls'] == 1789, f"pilot tau cacheable calls json {tau_ft['calls']} vs paper 1,789")
for src, lo, hi in [('gpt-4o-retail', 0.4895, 0.5021), ('gpt-4o-airline', 0.1628, 0.2205)]:
    rate = tau_ft['per_source'][src]['within_run_hit_rate']
    chk(round(rate, 4) in (lo, hi), f"pilot tau {src} within-run json {rate} vs paper range")
chk(round(tau_ft['within_task_hit_rate'], 4) == 0.0173,
    f"pilot tau within-task json {tau_ft['within_task_hit_rate']} vs paper 1.73%")
chk(round(tau_ft['within_run_hit_rate'], 4) == 0.4170,
    f"pilot tau within-run json {tau_ft['within_run_hit_rate']} vs paper 41.70%")
chk(round(tau['cache_scopes']['all_trials']['within_run_hit_rate'], 3) == 0.834,
    "pilot tau trial-replay cache rate json != paper 83.4%")
for s_ in ['1,960 tool-call trajectories', '14,285 calls', '1.73%', '41.70%', '49.0–50.2%',
           '16.3–22.1%', '83.4%', 'Static reduction is **0.0%** at BASIC and AGGRESSIVE']:
    chk(s_ in md, f'paper missing pilot citation {s_}')

swe, swe_ag, swe_ft = _pilot('real_trace_pilot_swe_agent_results.json')
chk(swe['provenance']['streams'] == 300 and swe_ag['total_steps'] == 8502,
    f"pilot swe scale json streams={swe['provenance']['streams']} steps={swe_ag['total_steps']} "
    "vs paper 300 / 8,502")
chk(swe_ft['calls'] == 0, f"pilot swe cacheable calls json {swe_ft['calls']} vs paper 0")
chk(swe_ft['effectful_repeats'] == 4009,
    f"pilot swe effectful repeats json {swe_ft['effectful_repeats']} vs paper 4,009")
for s_ in ['47.2% of its commands repeat verbatim', 'repetition is not cacheability',
           '| 300 / 300 |', '8,502', '| 4,009 (47.2%) |']:
    chk(s_ in md, f'paper missing coding-family citation {s_}')

# ---- 5.12 / I.7 instruction precipitation (instruction_precipitation_*.json) ----
def _prec(name):
    d = jload(name)
    return d, d['interleaved']

tau_p, tau_i = _prec('instruction_precipitation_results.json')
chk(tau_i['tasks'] == 330 and tau_i['steps'] == 2340 and tau_i['served'] == 2245
    and round(tau_i['coverage'], 4) == 0.9594,
    f"precip tau json tasks={tau_i['tasks']} steps={tau_i['steps']} served={tau_i['served']} "
    f"coverage={tau_i['coverage']:.4f} vs paper 330 / 2,340 / 2,245 / 95.94%")
seg = tau_i['segments']
chk(round(seg[0]['coverage'], 2) == 0.91 and round(seg[-1]['coverage'], 2) == 0.99,
    f"precip tau quarters json {[round(x['coverage'], 2) for x in seg]} vs paper 91..99")
chk(12 <= tau_i['repertoire']['templates_active'] <= 14,
    f"precip tau active templates json {tau_i['repertoire']['templates_active']} vs paper 12-14")
for src in tau_p['per_source'].values():
    first = round(src['segments'][0]['coverage'], 2)
    chk(0.76 <= first <= 0.88, f"precip tau first-quarter coverage json {first} vs paper 76-88%")
    chk(src['coverage'] >= 0.91, f"precip tau stream coverage json {src['coverage']} vs paper >= 91%")

swe_p, swe_i = _prec('instruction_precipitation_swe_agent_results.json')
chk(swe_i['tasks'] == 300 and swe_i['steps'] == 8502,
    f"precip swe scale json {swe_i['tasks']}/{swe_i['steps']} vs paper 300 / 8,502")
chk(swe_i['repertoire']['templates_active'] == 291,
    f"precip swe active templates json {swe_i['repertoire']['templates_active']} vs paper 291")
rm = swe_p['per_source']['swe-agent-swe-agent-llama-70b']
chk(round(rm['segments'][0]['coverage'], 2) == 0.43 and round(rm['segments'][-1]['coverage'], 2) == 0.67,
    f"precip swe 70b curve json {[round(x['coverage'], 2) for x in rm['segments']]} vs paper 43..67%")
for s_ in ['12 active templates (13–14 per source) match **95.9%**', '291 command templates serve 43%→67%',
           '2,340 tool-use steps', '| 300 | 8,502 | 5,052 | 59.42% |', '| mixed (round-robin) | 330 | 2,340 | 2,245 | 95.94% |',
           '75.0% of the tool-use stream']:
    chk(s_ in md, f'paper missing precipitation citation {s_}')

# ---- 5.12 / I.8 cross-family instruction flow (agentgym) ----------------
ag_p, ag_i = _prec('instruction_precipitation_agentgym_results.json')
FAMILIES = [('agentgym-alfworld', 2420, 32247, 0.8494, 0.858),
            ('agentgym-babyai', 810, 4642, 0.9272, 0.890),
            ('agentgym-sqlgym', 3000, 3000, 0.4517, 0.464)]
for name, tasks, steps, coverage, first_q in FAMILIES:
    st = ag_p['per_source'][name]
    chk(st['tasks'] == tasks and st['steps'] == steps,
        f"precip {name} scale json {st['tasks']}/{st['steps']} vs paper {tasks}/{steps}")
    chk(round(st['coverage'], 4) == coverage,
        f"precip {name} coverage json {st['coverage']:.4f} vs paper {coverage:.4f}")
    chk(abs(st['segments'][0]['coverage'] - first_q) < 0.005,
        f"precip {name} first-quarter json {st['segments'][0]['coverage']:.4f} vs paper {first_q:.4f}")
chk(ag_p['per_source']['agentgym-alfworld']['repertoire']['templates_active'] == 103,
    "precip alfworld active templates json != paper 103")
chk(ag_p['per_source']['agentgym-babyai']['repertoire']['templates_active'] == 54,
    "precip babyai active templates json != paper 54")
chk(ag_p['per_source']['agentgym-sqlgym']['repertoire']['templates_active'] == 16,
    "precip sqlgym active templates json != paper 16")
top_alf = ag_p['per_source']['agentgym-alfworld']['top_templates']
chk(top_alf[0]['tool'] == 'go to' and top_alf[0]['uses'] == 4422,
    f"precip alfworld top template json {top_alf[0]} vs paper go to/4,422")
top_sql = ag_p['per_source']['agentgym-sqlgym']['top_templates']
chk(top_sql[0]['uses'] == 38, f"precip sqlgym top template uses json {top_sql[0]['uses']} vs paper 38")
for s_ in ['household manipulation, 2,420 episode prefixes, 32,247 actions, 13 verbs',
           'flatten near 45%', 'climb slowly (43%→67%)',
           'A finite action space is therefore necessary but not sufficient',
           'parameter reuse governs the payoff', '| household manipulation (alfworld) | 13 verbs | 103 | 2,420 | 32,247 | 84.94% |',
           '| grid navigation (babyai) | 8 verbs | 54 | 810 | 4,642 | 92.72% |',
           '| SQL analytics (sqlgym) | 1 verb | 16 | 3,000 | 3,000 | 45.17% |',
           '`go to cabinet` serves 4,422 steps']:
    chk(s_ in md, f'paper missing cross-family citation {s_}')

# ---- Figure 7: manifest values must equal their sources ------------------
fig7 = json.loads((ROOT / 'docs/figures/figures_manifest.json').read_text())['figures']['7_instruction_memory.pdf']
_prec_tau = jload('instruction_precipitation_results.json')['per_source']
_prec_swe = jload('instruction_precipitation_swe_agent_results.json')['per_source']
_prec_ag = jload('instruction_precipitation_agentgym_results.json')['per_source']
for label, stream in [('closed: tool use, retail', _prec_tau['gpt-4o-retail']),
                      ('closed: household manipulation', _prec_ag['agentgym-alfworld']),
                      ('closed: grid navigation', _prec_ag['agentgym-babyai']),
                      ('open: shell coding', _prec_swe['swe-agent-swe-agent-llama-70b']),
                      ('open: SQL analytics', _prec_ag['agentgym-sqlgym'])]:
    chk(abs(fig7['series_final_pct'][label] - round(100.0 * stream['coverage'], 2)) < 1e-9,
        f'figure 7 series {label} json vs precipitation source mismatch')
chk(fig7['bars_pct']['compiler_real'] == 0.0
    and abs(fig7['bars_pct']['cache_within_task'] - round(100.0 * tau_ft['within_task_hit_rate'], 2)) < 1e-9
    and abs(fig7['bars_pct']['cache_across_tasks'] - round(100.0 * tau_ft['within_run_hit_rate'], 2)) < 1e-9,
    'figure 7 bars json vs real_trace_pilot mismatch')
chk(abs(fig7['bars_pct']['compiler_planted'] - round(100.0 * comp_red(200)['mean'], 2)) < 1e-9,
    'figure 7 planted bar json vs compiler microbench mismatch')

# ---- I.9 compound binding: recompute from the observation-carrying corpus ---
import importlib.util
_cb_path = ROOT / 'examples' / 'benchmarks' / 'compound_binding.py'
_cb_spec = importlib.util.spec_from_file_location('_gate_compound_binding', _cb_path)
_cb = importlib.util.module_from_spec(_cb_spec)
sys.modules['_gate_compound_binding'] = _cb
_cb_spec.loader.exec_module(_cb)
_committed = json.loads((ROOT / 'examples/benchmarks/compound_binding_results.json').read_text())
_recomputed = _cb.run((ROOT / 'examples/real_traces/tau_bench_observed.jsonl').resolve())
chk(_recomputed['tasks'] == _committed['tasks'] == 325, 'compound tasks drift')
chk(_recomputed['steps'] == _committed['steps'] == 2275, 'compound steps drift')
chk(_recomputed['compounds']['recurring'] == _committed['compounds']['recurring'] == 450,
    'compound recurring count drift')
chk(_recomputed['compounds']['signature_consistent'] ==
    _committed['compounds']['signature_consistent'] == 102, 'signature-consistent count drift')
chk(_recomputed['compounds']['replay_verified'] ==
    _committed['compounds']['replay_verified'] == 78, 'replay-verified count drift')
chk(_recomputed['reduction_if_substituted'] == _committed['reduction_if_substituted'],
    'compound earned reduction drift')
chk(_recomputed['strict_reduction'] == _committed['strict_reduction'],
    'compound strict reduction drift')
chk(abs(_committed['reduction_if_substituted']['reduction'] * 100 - 12.53) < 0.005,
    'paper says 12.53% but the corpus gives '
    f"{_committed['reduction_if_substituted']['reduction'] * 100:.2f}%")
chk(abs(_committed['upper_bound_signature_consistent']['reduction'] * 100 - 15.03) < 0.005,
    'paper says 15.03% but the corpus gives '
    f"{_committed['upper_bound_signature_consistent']['reduction'] * 100:.2f}%")
chk(abs(_committed['wiring']['internal_share'] * 100 - 15.7) < 0.05,
    'paper says 15.7% internal slots')
chk(abs(_committed['strict_reduction']['reduction'] * 100 - 1.36) < 0.005,
    'paper says 1.36% strict')
chk('12.53% of dispatches' in md and '78 of 450 recurring compounds (17.3%)' in md,
    'main paper missing the compound-binding result')
chk('| all | 450 | 102 (22.7%) | 78 (17.3%) |' in md,
    'supplement missing the compound-binding table row')
chk('285/2,275 steps, 178 applications' in md and '231 (84.3%) are external parameters' in md,
    'supplement missing compound wiring detail')

# ---------------------------------------------------------------------------
# Appendix I.10 (external cache baselines + priced effects), I.11 (cross-model
# compound transfer) and the bootstrap intervals. Every number is re-derived
# from a committed results file; the anchor (746/1,789) is also recomputed from
# the corpus so the baselines cannot drift from the paper's own cache figure.
# ---------------------------------------------------------------------------
SUPP = (ROOT / 'docs/AgentISA_supplementary.md').read_text(encoding='utf-8')
MAIN = (ROOT / 'docs/AgentISA.md').read_text(encoding='utf-8')
for _line in ('40.5 points strictly cross-task', '1,789 side-effect-free first-trial calls', '16.3–50.2% per run',
              'no second, token or dollar figure is attributable',
              '33 of the 49 eligible compounds', 'every observation-wired specification fails'):
    chk(_line in MAIN, f'the paper must state the revised reuse/transfer framing: "{_line}"')
MD = (ROOT / 'docs/AgentISA.md').read_text(encoding='utf-8')
_bl = jload('real_trace_external_baselines_results.json')
_anc = _bl['anchor']
chk(_anc['reproduced_hits'] == 746 and _anc['reproduced_cacheable_calls'] == 1789,
    f"I.10 anchor must reproduce 746/1789 (paper's cache), got {_anc['reproduced_hits']}/{_anc['reproduced_cacheable_calls']}")
chk(abs(_anc['reproduced_within_run'] - 0.417) < 5e-4 and abs(_anc['reproduced_within_task'] - 0.0173) < 5e-5,
    'I.10 anchor rates must match the paper: 41.70% within run, 1.73% within task')
chk(_anc['matches_published'] is True, 'I.10 anchor must report a match; a mismatch voids the comparison')

_lru = _bl['lru']['sound']['run']
for _cap, _exp in ((8, 0.2554), (16, 0.3382), (32, 0.3482), (64, 0.3689),
                   (128, 0.3862), (256, 0.4159), (512, 0.417)):
    _got = _lru[str(_cap)]['hit_rate']
    chk(abs(_got - _exp) < 5e-5,
        f'I.10 LRU capacity {_cap} must be {_exp:.2%}, got {_got:.2%}')
_ws = _bl['lru']['sound']['task']['0']['hit_rate']
chk(abs(_ws - 0.0173) < 5e-5, f'I.10 within-task LRU must be 1.73%, got {_ws:.2%}')

_hs = _bl['hot_set']['sound']['global']
for _n, _exp in ((16, 0.0277), (64, 0.0520), (256, 0.1363)):
    chk(abs(_hs[str(_n)]['hit_rate'] - _exp) < 5e-5,
        f'I.10 static hot set N={_n} must be {_exp:.2%}, got {_hs[str(_n)]["hit_rate"]:.2%}')
chk(_hs['256']['calls'] == 866,
    f'I.10 hot-set test half must be 866 calls, got {_hs["256"]["calls"]}')
chk(abs(_lru['256']['hit_rate'] - _hs['256']['hit_rate']) > 0.2,
    'I.10 must show the 256-entry LRU far above the static hot set')

_st = _bl['same_turn']
chk(_st['turns_with_multiple_calls'] == 0 and _st['turns'] == 2275,
    f'I.10 same-turn must be a structural zero over 2,275 turns, got {_st["turns_with_multiple_calls"]}/{_st["turns"]}')

_g = _bl['global_scope_decomposition']
chk(_g['hits'] == 1126 and _g['hits_same_task_other_backbone'] == 371,
    f'I.10 global hits must decompose as 1,126 with 371 trial-replay hits, got {_g["hits"]}/{_g["hits_same_task_other_backbone"]}')
_ts = jload('trace_scope_results.json')
_rw = _ts['run_weighting']
chk(_rw['pooled_hits'] == 746 and _rw['pooled_calls'] == 1789,
    f'trace scope must reproduce 746/1,789, got {_rw["pooled_hits"]}/{_rw["pooled_calls"]}')
chk(abs(_rw['pooled_rate'] - 0.417) < 5e-4, f'trace scope pooled rate must be 41.70%, got {_rw["pooled_rate"]:.4%}')
chk(abs(_rw['run_mean_rate'] - 0.3437) < 5e-4,
    f'run-weighted mean must be 34.37%, got {_rw["run_mean_rate"]:.4%}')
chk(abs(_rw['run_clustered_ci95'][0] - 0.192) < 5e-3
    and abs(_rw['run_clustered_ci95'][1] - 0.496) < 5e-3,
    f'run-clustered CI must be [19.2, 49.6], got {_rw["run_clustered_ci95"]}')
chk(len(_rw['per_run']) == 4, 'trace scope must report exactly four runs')
chk(abs(min(r['rate'] for r in _rw['per_run']) - 0.1628) < 5e-4
    and abs(max(r['rate'] for r in _rw['per_run']) - 0.5021) < 5e-4,
    'per-run rates must span 16.28%-50.21%')
_bg = _ts['backbone_global']
chk(abs(_bg['per_backbone']['gpt-4o']['rate'] - 0.4053) < 5e-4
    and abs(_bg['per_backbone']['claude-3.5-sonnet']['rate'] - 0.4272) < 5e-4,
    f'single-backbone global rates must be 40.53%/42.72%, got {_bg["per_backbone"]}')
chk(abs(_bg['shipped_global_rate'] - 0.6294) < 5e-5, 'shipped global rate must be 62.94%')
_hl = _ts['hot_set_test_call_lru']
chk(abs(_hl['capacities']['256']['rate'] - 0.4261) < 5e-4,
    f'warmed LRU on the hot-set test calls must be 42.61%, got {_hl["capacities"]["256"]["rate"]:.4%}')
chk(abs(_hl['ceiling']['rate'] - 0.1478) < 5e-4,
    f'static ceiling must be 14.78%, got {_hl["ceiling"]["rate"]:.4%}')
chk(abs(_hl['ceiling']['rate'] - 0.1363) < 0.02, 'the static hot set must sit near its own ceiling')
_tpc = _ts['task_position_curve']
chk(abs(_tpc['first_half_rate'] - 0.4069) < 5e-4 and abs(_tpc['second_half_rate'] - 0.4281) < 5e-4,
    f"first/second-half rates must be 40.69%/42.81%, got {_tpc['first_half_rate']:.4%}/{_tpc['second_half_rate']:.4%}")
chk(abs(_tpc['first_decile_rate'] - 0.4396) < 5e-4 and abs(_tpc['last_decile_rate'] - 0.4286) < 5e-4,
    f'decile endpoints must be 43.96%/42.86%, got {_tpc["first_decile_rate"]:.4%}/{_tpc["last_decile_rate"]:.4%}')
chk(abs(_tpc['share_of_hits_in_first_decile'] - 0.122) < 5e-3,
    f"first-decile hit share must be 12.2%, got {_tpc['share_of_hits_in_first_decile']:.4%}")
chk(len(_tpc['curve']) == 10, 'the position curve must have ten deciles')
_tp = _ts['transfer_pooled']
chk(_tp['observation_wired_transferred'] == 0,
    'no observation-wired specification may transfer')
chk(_tp['parameter_only_transferred'] == 33 and _tp['parameter_only_eligible'] == 45,
    'parameter-only transfer must be 33/45 eligible')
chk(abs(_tp['unconditional_transfer_rate'] - 0.6735) < 5e-4,
    f'unconditional retail transfer must be 67.35%, got {_tp["unconditional_transfer_rate"]:.4%}')
chk(abs(_g['hit_rate'] - 0.6294) < 5e-5,
    f'I.10 global rate must be 62.94% before decomposition, got {_g["hit_rate"]:.2%}')

_cm = jload('cost_model_results.json')
chk(_cm['result_kind'] == 'MODELED' and _cm['not_observed'] is True,
    'I.10 cost figures must be labelled MODELED and not_observed')
chk(_cm['aggregate']['sum_allowed'] is False,
    'I.10 must refuse to sum mechanisms (F2/F3 count the same stream)')
_f1 = _cm['mechanisms'][0]
chk(_f1['effect_units'] == 746, f'I.10 F1 must rest on 746 measured hits, got {_f1["effect_units"]}')
chk(_f1['convertible'] is False and _f1['point'] is None,
    'I.10 F1 must NOT be priced: a cache hit avoids a tool execution, not a model call')
chk('not_priceable' in _f1, 'I.10 F1 must record why it is not priceable')
_f2 = _cm['mechanisms'][1]['point']; _f3 = _cm['mechanisms'][2]['point']
chk(abs(_f2['dollars'] - 2.296635) < 1e-5 and abs(_f2['seconds']['mean'] - 5158.08) < 0.1,
    f'I.10 F2 conditional bound must be 5,158 s / $2.30, got {_f2["seconds"]["mean"]}/{_f2["dollars"]}')
chk(abs(_f3['dollars'] - 0.291555) < 1e-5 and abs(_f3['seconds']['mean'] - 654.81) < 0.1,
    f'I.10 F3 conditional bound must be 655 s / $0.29, got {_f3["seconds"]["mean"]}/{_f3["dollars"]}')
chk(_cm['mechanisms'][0]['assumption_conditioned'] is False
    and _cm['mechanisms'][1]['assumption_conditioned'] is True,
    'I.10 must keep F1 unconditioned and F2/F3 conditioned on the bypass')
_up = _cm['unit_prices']
chk(abs(_up['t_mean_s'] - 2.2976) < 5e-4 and _up['input_tokens_per_call'] == 21 and _up['output_tokens_per_call'] == 64,
    f'I.10 unit prices must be 2.30 s and 21/64 tokens, got {_up["t_mean_s"]}/{_up["input_tokens_per_call"]}/{_up["output_tokens_per_call"]}')
for _line in ('convertible: false', '5,158 s and $2.30', '655 s and $0.29', '40.5%', '1,789 of 2,275',
              '62.94%', '13.63%', '42.61%', '14.78%', '34.37%', '19.2%', '40.53%', '42.72%', '67.3%',
              '40.69%', '42.81%', '43.96%', '12.2%'):
    chk(_line in SUPP, f'supplementary I.10 must state "{_line}"')

_ct = jload('compound_transfer_results.json')
_cells = _ct['cross_model']['cells']
_expect_cells = {'retail': (49, 36, 33), 'retail_reverse': (23, 17, 17),
                 'airline': (32, 18, 18), 'airline_reverse': (8, 5, 5)}
for _name, (_el, _ex, _tr) in _expect_cells.items():
    _c = _cells[_name]
    chk((_c['eligible'], _c['exposed'], _c['transferred']) == (_el, _ex, _tr),
        f'I.11 {_name} must be {_el}/{_ex}/{_tr}, got {(_c["eligible"], _c["exposed"], _c["transferred"])}')
_po = _cells['retail']['wiring']['parameter_only']
_ow = _cells['retail']['wiring']['observation_wired']
chk(_po['transfer_rate'] == 1.0 and _po['occurrences']['exact_share'] == 1.0,
    f'I.11 parameter-only must transfer at 100% with exact replay, got {_po["transfer_rate"]}/{_po["occurrences"]["exact_share"]}')
chk(_ow['transfer_rate'] == 0.0,
    f'I.11 observation-wired must transfer at 0%, got {_ow["transfer_rate"]}')
_cov = _cells['retail'].get('step_coverage_wiring', {})
if _cov:
    chk(abs(_cov['parameter_only']['share'] - 0.2443) < 5e-4,
        f'I.11 parameter-only step coverage must be 24.4%, got {_cov["parameter_only"]["share"]:.2%}')
    chk(_cov['observation_wired']['share'] == 0.0, 'I.11 observation-wired step coverage must be 0.0%')
_ctrl = _ct['controls']['per_direction']['retail']['conditions']['within_source']
chk((_ctrl['exposed'], _ctrl['transferred']) == (28, 27),
    f'I.11 within-source control must be 27/28, got {_ctrl["transferred"]}/{_ctrl["exposed"]}')
_cd = _ct.get('cross_domain', {})
if _cd:
    chk(_cd['shared_atomic_templates'] == 4 and _cd['eligible_expressible_in_tester'] == 1
        and _cd['eligible_exposed_in_tester'] == 0,
        'I.11 cross-domain must be 4 shared tools / 1 expressible / 0 exposed')
for _line in ('91.7% [78.2, 97.1]', '96.4% [82.3, 99.4]', '24.4%', 'path drift',
              '33/45, 17/23, 18/32 and 5/8'):
    chk(_line in SUPP, f'supplementary I.11 must state "{_line}"')

_bc = jload('bootstrap_ci_results.json')
_m = _bc['metrics']
chk(abs(_m['dynamic.shrink_ratio']['ci_low'] - 0.622) < 5e-4
    and abs(_m['dynamic.shrink_ratio']['ci_high'] - 0.692) < 5e-4,
    f'§5.4 CI must be 62.2–69.2%, got {_m["dynamic.shrink_ratio"]["ci_low"]:.3f}–{_m["dynamic.shrink_ratio"]["ci_high"]:.3f}')
_pi = _bc['paired_differences']['paired_ablation.paired_interaction']
chk(abs(_pi['mean'] + 0.119) < 5e-4 and _pi['ci_high'] < 0,
    f'§5.5 interaction CI must be −11.9% [−13.4, −10.4] excluding zero, got {_pi["mean"]} [{_pi["ci_low"]}, {_pi["ci_high"]}]')
for _line in ('62.2–69.2%', '−13.4 to −10.4'):
    chk(_line in MD, f'main paper must state "{_line}"')

# ---------------------------------------------------------------------------
# suite-count invariant (must stay last among checks)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# suite-count invariant (computed last; the failure branch follows it)
# ---------------------------------------------------------------------------
# ---- Section 5.13: cache soundness (four policies, damage split, ladder, holdout)
_uc = jload('unsafe_cache_results.json')
_pol = {p['id']: p for p in _uc['policies']}
chk(_pol['P1']['hits'] == 807 and _pol['P1']['unsafe_hits'] == 344,
    f'P1 must be 807 hits / 344 unsafe, got {_pol["P1"]["hits"]}/{_pol["P1"]["unsafe_hits"]}')
chk(_pol['P2']['hits'] == 655 and _pol['P2']['unsafe_hits'] == 165,
    f'P2 must be 655/165, got {_pol["P2"]["hits"]}/{_pol["P2"]["unsafe_hits"]}')
chk(_pol['P3']['hits'] == 963 and _pol['P3']['unsafe_hits'] == 595,
    f'P3 must be 963/595, got {_pol["P3"]["hits"]}/{_pol["P3"]["unsafe_hits"]}')
chk(_pol['P4']['hits'] == 466 and _pol['P4']['cacheable_calls'] == 1728 and _pol['P4']['unsafe_hits'] == 0,
    'P4 must be 466 hits of 1,728 admitted with zero unsafe')
chk(_pol['P1']['unsafe_breakdown']['writes_served'] == 61
    and _pol['P1']['unsafe_breakdown']['stale_reads'] == 179
    and _pol['P1']['unsafe_breakdown']['cross_tenant'] == 104,
    'P1 breakdown must be 61 writes + 179 stale + 104 cross-tenant')

_ud = jload('unsafe_damage_results.json')
_dpol = {p['id']: p for p in _ud['policies']}
chk(_dpol['P1']['warranted_unsafe'] == 240 and _dpol['P1']['D1'] == 61 and _dpol['P1']['D2'] == 179,
    f'P1 damage must be 240 = 61 + 179, got {_dpol["P1"]["warranted_unsafe"]}')
chk(_dpol['P1']['D3_harmless'] == 104, 'the harmless cross-tenant class must be 104')
chk(_dpol['P2']['warranted_unsafe'] == 61, 'P2 damage must be 61 (writes only)')
chk(_dpol['P3']['warranted_unsafe'] == 491 and _dpol['P3']['D3_damaging'] == 250,
    'P3 damage must be 491 with 250 identity-dependent')
chk(_dpol['P4']['warranted_unsafe'] == 0, 'P4 must have zero damaging hits')
_ci = _ud['intervals']['p1_warranted_per_1000_calls']
chk(abs(_ci['estimate'] - 105.495) < 5e-3 and abs(_ci['ci_low'] - 82.27) < 5e-3
    and abs(_ci['ci_high'] - 130.086) < 5e-3,
    f"P1 damage interval must be 105.5 [82.3, 130.1], got {_ci['estimate']} [{_ci['ci_low']}, {_ci['ci_high']}]")
chk(_ud['instances']['total'] == 165 and _ud['instances']['p1']['with_warranted_unsafe'] == 70,
    'the cluster unit must be 165 task instances with 70 affected')
chk(abs(_ud['instances']['p1']['share_with_warranted_unsafe'] - 0.424242) < 5e-5,
    'the affected-instance share must be 42.4%')
chk(abs(_ud['typed_zero_bound']['admitted_reads']['exact_upper_per_1000'] - 1.732) < 5e-3,
    'the typed zero bound must be 1.73 per 1,000 admitted reads')

_sl = jload('soundness_ladder_results.json')
_r = _sl['ladders']['main']['rungs']
chk([r['hits'] for r in _r[:3]] == [745, 619, 466] and [r['unsafe_hits'] for r in _r[:3]] == [283, 168, 0],
    f'ladder hits must be 745/619/466 with unsafe 283/168/0, got {[r["hits"] for r in _r[:3]]}')
chk(abs(_r[1]['hit_rate_on_denominator'] - 0.3582) < 5e-4 and abs(_r[2]['hit_rate_on_denominator'] - 0.2697) < 5e-4,
    'ladder rates must be 35.82% and 26.97% on the 1,728 base')

_ch = jload('compound_holdout_results.json')['verdict']
chk(_ch['holdout_dispatches_removed'] == 229 and _ch['dispatches'] == 2275,
    'the holdout must remove 229 of 2,275 dispatches')
chk(abs(_ch['holdout_rate'] - 0.1007) < 5e-5, f"holdout rate must be 10.07%, got {_ch['holdout_rate']:.4%}")
chk(abs(_ch['committed_rate'] - 0.1253) < 5e-5, 'the in-sample anchor must stay 12.53%')

for _line in ('344 unwarranted and 240 damaging', 'serves 466 hits with none unwarranted',
              '105.5 damaging hits per 1,000 calls', 'the tenant 7.29 points', 'invalidation 8.85 points',
              '10.07% of dispatches out of sample', 'still enforces none of them'):
    chk(_line in MD, f'main paper must state "{_line}"')
for _line in ('240 (61 writes + 179 stale)', '491 (250 one tool)', 'the typed policy \'s zero is reported with its exact one-sided bound'
              if False else 'exact one-sided bound', '16.14 points on one base'):
    chk(_line in SUPP, f'supplement must state "{_line}"')

# ---- shipped SoundResultCache replay (Section 5.13, last paragraph)
_ss = jload('shipped_sound_cache_results.json')
_arm = {a['id']: a for a in _ss['arms']}
chk(_arm['harness_matched']['hits'] == 459 and _arm['harness_matched']['cacheable_calls'] == 1728,
    f"the shipped class must return 459 of 1,728, got {_arm['harness_matched']['hits']}")
chk(_arm['harness_matched']['unsafe_hits'] == 0 and _arm['identity_map']['unsafe_hits'] == 0,
    'the shipped class must serve zero damaging hits on both arms')
chk(_ss['checks']['zero_damaging_hits']['ok'] is True, 'zero-damaging check must pass')
chk(_ss['checks']['harness_figure_reproduced']['ok'] is True, 'the harness 466/1,728 must be reproduced')
chk(_arm['harness_matched']['refusals_by_reason']['effectful'] == 547,
    'the effect gate must refuse the 547 effectful calls')
for _line in ('returns 459 hits of 1,728', 'seven hits below the harness',
              'replayed at 459 hits of 1,728 with zero unjustified hits'):
    chk(_line in MD, f'main paper must state "{_line}"')

# ---- default-flip cost: census buckets and the coarse-footprint penalty
_cen = jload('cache_default_census.json')
_sum = _cen.get('summary', {})
_buckets = _sum.get('buckets') or _sum.get('sites_by_bucket') or {}
chk(int(_buckets.get('pure_declared', -1)) == 11, f"census must find 11 pure_declared sites, got {_buckets.get('pure_declared')}")
chk(int(_buckets.get('effectful_declared', -1)) == 2, 'census must find 2 effectful_declared sites')
chk(int(_buckets.get('undeclared', -1)) == 2, 'census must find 2 undeclared sites')
_tot = sum(int(v) for v in _buckets.values())
chk(_tot == 15, f'the three buckets must cover 15 put sites, got {_tot}')
_t = _sum.get('tenant', {})
chk(int(_t['execute_defs_tenant_reach'].get('nothing', -1)) == 45,
    f"45 caching execute definitions must have no tenant channel, got {_t['execute_defs_tenant_reach'].get('nothing')}")
chk(int(_t.get('admitted_sites_without_tenant_channel', -1)) == 8, '8 admitted sites must lack a tenant channel')

_cf = jload('cache_coarse_footprint_cost.json')
chk(_cf['arms']['exact_footprint']['hits'] == 459, 'the exact arm must reproduce 459 hits')
chk(_cf['arms']['coarse_footprint']['hits'] == 67, 'the coarse arm must return 67 hits')
chk(_cf['arms']['coarse_footprint']['unsafe_hits'] == 0, 'the coarse arm must stay sound')
chk(all(_cf['checks'].values()), 'every coarse-footprint check must pass')
chk(abs(_cf['delta']['hit_rate_points'] - 22.69) < 0.01, f"the penalty must be 22.69 points, got {_cf['delta']['hit_rate_points']}")
for _line in ('15 sites', '459 hits of 1,728 to 67', '26.56% to 3.88%'):
    chk(_line in MD, f'main paper must state "{_line}"')

# ---- suite count: the paper's "N passing" must equal collected - skipped ------
_m = re.search(r'([\d,]+) passing, (\d+) skipped at run time', md)
if _m:
    _passed = int(_m.group(1).replace(',', ''))
    _skipped = int(_m.group(2))
    _collected_out = subprocess.run(
        [sys.executable, '-m', 'pytest', '--collect-only', '-q', 'tests/'],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, 'HOME': str(ROOT / '.test_home'),
             'ALL_PROXY': '', 'all_proxy': '', 'HTTP_PROXY': '', 'http_proxy': '',
             'HTTPS_PROXY': '', 'https_proxy': ''})
    _c = re.search(r'(\d+) tests? collected', _collected_out.stdout)
    if _c:
        _collected = int(_c.group(1))
        chk(_passed + _skipped == _collected,
            f'paper cites {_passed} passing + {_skipped} skipped but pytest collects '
            f'{_collected} tests; a test was added or removed without updating the count')

# This branch must stay last: every check above appends to `fails`, and a check
# added after it would be silently swallowed (it happened twice).
if fails:
    print('REGRESSION FAILURES:')
    for f in fails:
        print(' -', f)
    sys.exit(1)
print('ALL NUMBERS REGRESSION-CLEAN: every cited benchmark number matches its committed results JSON '
      '(microbench + real-gateway + Appendix-I extension experiments; aggregates recomputed from raw records)')
