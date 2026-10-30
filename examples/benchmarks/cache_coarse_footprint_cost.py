#!/usr/bin/env python3
"""What a typed default costs while no instruction implements ``footprint(params)``.

The census (``cache_default_census.py``) finds 0 of 15 cache call sites overriding
``footprint``, so a typed default today would have to treat every read as
"footprint unknown" and every write as coarse: any landed write flushes the whole
run-scope cache. That is sound (over-eviction never resurrects a stale value) but
it can only lose hits. This script measures the loss by driving the *same* replay
as ``shipped_sound_cache_replay.py`` — same corpus, same key policy, same class —
while forcing every admission to the coarse token. The exact-footprint arm is the
committed 459/1,728; monkey-patching only the footprint keeps every other variable
fixed, so the difference is attributable to footprints alone.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPLAY = ROOT / "examples" / "benchmarks" / "shipped_sound_cache_replay.py"
OUT = ROOT / "examples" / "benchmarks" / "cache_coarse_footprint_cost.json"
COARSE = "__coarse__"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    replay = _load("shipped_sound_cache_replay", REPLAY)
    ci, _ = replay.load_identity()
    ucb, _ = replay.load_baseline()
    sound = replay.load_sound_class()

    document = json.loads(replay.EFFECT_MAP_PATH.read_text(encoding="utf-8"))
    policies = replay.build_policies(replay.build_identity_map(document))
    policy = next(p for p in policies if p.pid == "harness_matched")

    records = ucb.load_records(replay.DEFAULT_CORPUS)
    streams = ucb.build_streams(records)

    exact = replay.replay_arm(streams, ci, policy, sound)

    original_put = sound.SoundResultCache.put

    def coarse_put(self, instruction_id, params, value, footprint=frozenset(), **kwargs):
        return original_put(self, instruction_id, params, value,
                            footprint=frozenset({COARSE}), **kwargs)

    original_write = sound.SoundResultCache.record_write

    def coarse_write(self, entities=None, coarse=False):
        # A site without footprint() cannot name what it touched, so its writes are
        # coarse as well; leaving writes named would make the coarse read token
        # un-evictable, which is unsound rather than merely expensive.
        return original_write(self, entities=None, coarse=True)

    sound.SoundResultCache.put = coarse_put
    sound.SoundResultCache.record_write = coarse_write
    try:
        coarse = replay.replay_arm(streams, ci, policy, sound)
    finally:
        sound.SoundResultCache.put = original_put
        sound.SoundResultCache.record_write = original_write

    exact_hits, coarse_hits = exact["hits"], coarse["hits"]
    cacheable = exact["cacheable_calls"]
    report = {
        "schema": "cache-coarse-footprint-cost/v1",
        "task": "cost of a typed default before any instruction implements footprint(params)",
        "result_kind": "measured",
        "scope": (
            "the shipped_sound_cache_replay corpus and key policy, re-run twice: once with the "
            "replay's exact footprints, once with every admission forced to the coarse token so "
            "that any landed write flushes the run-scope cache. Only the footprint differs."
        ),
        "corpus_sha256": replay.sha256_file(replay.DEFAULT_CORPUS),
        "sound_cache_sha256": replay.sha256_file(replay.SOUND_CACHE_PATH),
        "arms": {
            "exact_footprint": {
                "hits": exact_hits,
                "cacheable_calls": cacheable,
                "hit_rate": round(exact_hits / cacheable, 6),
                "unsafe_hits": exact["unsafe_hits"],
                "coarse_invalidations": exact.get("coarse_invalidations", 0),
            },
            "coarse_footprint": {
                "hits": coarse_hits,
                "cacheable_calls": cacheable,
                "hit_rate": round(coarse_hits / cacheable, 6),
                "unsafe_hits": coarse["unsafe_hits"],
                "coarse_invalidations": coarse.get("coarse_invalidations", 0),
            },
        },
        "delta": {
            "hits": exact_hits - coarse_hits,
            "hit_rate_points": round((exact_hits - coarse_hits) / cacheable * 100, 2),
        },
        "checks": {
            "exact_arm_reproduces_committed_459": exact_hits == 459,
            "coarse_arm_is_sound": coarse["unsafe_hits"] == 0,
            "coarse_costs_hits": coarse_hits <= exact_hits,
            "same_denominator": exact["cacheable_calls"] == coarse["cacheable_calls"],
        },
        "reading": (
            "A typed default without footprint() implementations is sound but pays the coarse "
            "eviction on every landed write. This is the cost of flipping the default *today*; "
            "it is a hit-rate cost, not a correctness cost, and it disappears site by site as "
            "footprint() is implemented."
        ),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ok = all(report["checks"].values())
    print(f"exact {exact_hits}/{cacheable} -> coarse {coarse_hits}/{cacheable} "
          f"(−{report['delta']['hit_rate_points']} points), checks {'ok' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
