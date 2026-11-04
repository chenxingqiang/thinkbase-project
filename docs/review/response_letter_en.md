# Response to reviewers

We thank all four reviewers. The revision changes the paper's contribution rather than
adding evidence to the old one: it now states and proves the condition under which result
reuse is safe, measures what an untyped cache gets wrong on the recorded corpus, prices
enforcing the condition, ships an implementation of it, and corrects one of our own
numbers downward. Section and table numbers below refer to the revised manuscript.

## Summary of what changed

1. **New Theorem 3.7 and Corollary 3.7.1** (§3.2.4, proof in Appendix C.8): no unsafe
   reuse under four hypotheses, with three counterexample families showing each hypothesis
   is necessary.
2. **New §5.13 and Appendix I.12**: four cache policies over the same 2,275 first-trial
   calls — a blind key serves 807 hits, 344 unwarranted, **240 damaging** (61 write results
   replayed, 179 reads from before a landed write); a typed key serves 466 of 1,728 admitted
   calls with **zero** unjustified hits; enforcing soundness costs **16.14 points** of hit
   rate on a fixed base, not the 8.5 a cross-denominator comparison would suggest.
3. **The typed cache now ships** (`agentflow/core/isa/sound_cache.py`) and is replayed
   through the class itself: 459 of 1,728 calls, zero damaging hits, seven below the
   harness figure, with the difference explained (121 refusals without a tenant in scope).
4. **Correction**: the compound-absorption figure does not survive a task-level holdout.
   Mining on one half, measuring on the other gives **10.07%** (229/2,275) against 12.53%
   in sample. The paper now quotes the holdout figure as the earned rate.

## R1 — the `unsafe` label was defined by our own key

You are right that the original definition made the typed policy's zero analytic. Damage is
now re-derived independently of the key: a hit is damaging if serving it changes the
observable outcome of the trajectory, in three classes — D1 a write response replayed
(61), D2 a value read before a landed write (179), D3 a value whose dependence on identity
we measured rather than assumed. D3 is what separates the two readings of the same
104 cross-tenant hits: catalog tools (`get_product_details`, `list_all_product_types`)
return the same value for every caller, so 104 such hits are counted as harmless;
`get_user_details` does not, and 250 such hits are counted as damaging. The identity
dependence is measured per tool (`tool_effect_map.json`), not asserted.

What remains unaddressed: the per-tool map is ours. A reviewer may reasonably hold that a
cross-tenant *read* is a leak even when the value is public. We do not claim otherwise; we
report the two classes separately so the reader can price them differently.

## R2 — numbers may not match the artifacts

Every number cited in Section 5 is recomputed from the committed results JSON by
`scripts/number_regression.py`, which asserts the paper's exact strings and fails closed.
It runs as the last step of the build and caught two errors during this revision (a stale
suite count and a phrase the text no longer contained). Suite counts are checked against
`pytest --collect-only`, so a cited "N passed, M skipped" cannot drift from the suite.

Provenance: each artifact records the corpus hash, the module hashes, seeds and the exact
command; `REPRODUCING.md` maps each claim to its module, test and JSON. One artifact
carries a pre-revision `git_rev` and now says so in a note rather than being silently
regenerated.

## R3 — the intervals were too narrow and used the wrong unit

Accepted and fixed. The earlier figures resampled 325 source-labelled cells; calls within
a task share a trajectory and a cache, so the cluster unit is the 165 task instances. With
that unit, the blind key damages **105.5 hits per 1,000 calls [82.3, 130.1]** and **42.4%**
of instances contain at least one damaging hit. The typed policy's zero is reported with an
**exact one-sided** Clopper–Pearson bound (≤1.73 per 1,000 admitted reads) instead of a
bootstrap interval, because a percentile interval on a zero count is degenerate. The 325-cell
numbers appear nowhere in the revision.

We also state in §5.9 that real-gateway runs are single-seed by design, with the two-seed
rerun cited where it exists.

## R4 — no agent in the loop, 0.0% static on real traces, our cache violates the theorem, length

- **0.0% static reduction** is a negative result and we lead with it: recorded traces chain
  their observations, so no step is provably dead and no pure pair provably independent.
  The 34.2% figure belongs to the planted family and the paper now says so in the same
  sentence as the 0.0%.
- **Our cache violated the theorem.** It did. `SoundResultCache` now enforces all three
  hypotheses as refusal rules, and the paper reports the shipped class's replay (459/1,728,
  zero damaging) rather than a harness number. The legacy store remains the default, and
  §5.9 item 11 states why with counts: 0 of 45 caching `execute()` definitions can name a
  tenant, 0 of 15 cache sites implement `footprint()`, and flipping the default today costs
  22.7 points of hit rate (459 → 67) because every write would be coarse. Compatibility is
  not the blocker — under `AGENTFLOW_CACHE_POLICY=typed` the full suite passes.
- **No agent in the loop.** We do not rebut this; we scope it. Every claim is offline, the
  abstract says so, and `docs/review/end_to_end_protocol.md` is the pre-registered design of
  the experiment that would settle it, including its own power analysis (≈6 points at 200
  tasks) and the rule that a smaller effect will be reported as inconclusive.
- **Length.** Main paper 20 pages with page 21 empty; proofs, protocols and the full
  Appendix-I tables are in a 60-page supplement.

## What we would still ask the reviewers to weigh

The paper's value now rests on a claim that is falsifiable and, we think, interesting
independent of the system's hit rate: *reuse that is not conditioned on tenant and
invalidation is not a performance parameter, it is a correctness bug, and its cost is
measurable — 240 of 807 hits in a 2,275-call corpus.* If that claim is right, the exact
hit-rate numbers matter less than the audit method, and the method is what we would like
reviewed.
