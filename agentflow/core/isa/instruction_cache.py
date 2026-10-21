"""Cache-key contract for the ISA instruction result cache.

The AgentISA cache-key contract (paper Section 5.2) keys a cached
entry on the (instruction, params) pair; soundness of a *hit* assumes
the model alias, prompt-template version, and decoding configuration
are held fixed.  A deployment in which any of those factors can
change must invalidate or namespace the cache -- historically the
production cache could not detect such semantic-configuration drift.

This module owns both halves of that contract:

- :func:`build_cache_key` with ``namespace=None`` (the default, and
  the only mode used in-tree before this module existed) produces the
  legacy key: the sha256 of the canonical JSON of ``input_params``
  alone, byte-identical to the historical format so every existing
  hit/miss decision is preserved.
- :func:`build_cache_key` with a :class:`CacheKeyNamespace` attached
  opts into the *strict namespace*: the key is extended to
  (instruction, params, model_alias, template_version, decoding_hash),
  so drifting any component misses the old key and starts hitting
  under the new one instead of serving a stale entry.

The LRU-TTL store itself (:class:`~agentflow.core.isa.instruction.
InstructionCache`) lives in ``instruction.py``; only the key contract
lives here.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _canonical_hash(payload: Any) -> str:
    """Sha256 of the canonical JSON serialization of ``payload``."""
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


@dataclass
class CacheKeyNamespace:
    """Semantic configuration a strict cache key is namespaced by.

    Attaching an instance to an instruction's ``cache_namespace``
    opts that instruction into the strict cache key
    (instruction, params, model_alias, template_version,
    decoding_hash).  The dataclass is deliberately mutable so a
    deployment can drift one component in place (or replace the whole
    namespace object); either way the next cache lookup computes a
    different key and cannot be served an entry cached under the old
    semantic configuration.

    Attributes:
        model_alias: model identity the instruction targets (e.g.
            ``"claude-glm-5.3-flash"``).
        template_version: version tag of the prompt template the
            instruction renders.
        decoding: decoding configuration (e.g. temperature,
            max_tokens); folded into the key via its canonical
            ``decoding_hash``.
    """

    model_alias: Optional[str] = None
    template_version: Optional[str] = None
    decoding: Optional[Dict[str, Any]] = None

    @property
    def decoding_hash(self) -> str:
        """Canonical sha256 of the decoding configuration.

        Canonical JSON (sorted keys) makes the hash insensitive to
        dict insertion order while remaining sensitive to any value
        change.
        """
        return _canonical_hash(self.decoding)


def build_cache_key(instruction_id: str,
                    input_params: Dict[str, Any],
                    namespace: Optional[CacheKeyNamespace] = None) -> str:
    """Build a cache key under the AgentISA cache-key contract.

    With ``namespace=None`` this returns the legacy key -- the sha256
    of the canonical JSON of ``input_params`` alone -- so the default
    behavior is unchanged.  With a namespace attached it returns the
    strict key: the sha256 of the canonical JSON of the full
    five-tuple (instruction, params, model_alias, template_version,
    decoding_hash).
    """
    if namespace is None:
        return _canonical_hash(input_params)
    return _canonical_hash({
        "instruction": instruction_id,
        "params": input_params,
        "model_alias": namespace.model_alias,
        "template_version": namespace.template_version,
        "decoding_hash": namespace.decoding_hash,
    })
