"""Cache identity for tool calls: what a key must bind, and what a write must evict.

A cache in front of a tool-using agent has one job it cannot fail at: never answer
with a value the world has moved past, and never answer a mutation at all. That
gives the three identities this module implements.

  effect    which calls may be stored. Only reads. A cached write is a mutation
            that never ran while the agent believes it did -- the second
            ``cancel_pending_order`` is "answered" from cache and the order is
            still pending. ``EFFECT_CLASSES`` carries the map from
            ``tool_effect_map.json`` (11 read / 13 write / 2 control) and
            ``is_cacheable`` is the single gate.

  key       which calls are the same call. ``key_blind`` is the unsound baseline
            behind the published 41.70% reuse figure: sha256 of the tool and its
            arguments, with no tenant and no effect filter. It is sound only as an
            *equality* test -- it will happily memoise a write and will answer
            tenant A with tenant B's row. ``key_typed`` is the version this module
            is built on: tenant-bound, canonically serialised, domain-separated
            from the baseline, and defined only for reads.

  eviction  which stored answers a write makes obsolete. A key proves two calls
            asked the same question; it says nothing about whether the answer is
            still true. ``remember`` files a stored key under every entity id its
            arguments name; ``invalidate`` takes the ids a write names and returns
            every key filed under them.

The soundness argument, in one sentence: a key that survives ``invalidate`` names
no entity the write named, is bound to the tenant that asked, and was only ever
created by a read. Everything below is a case analysis of that sentence.

What this module does not claim
-------------------------------
Freshness is relative to the writes the cache observes. Four gaps are known,
argued in the docstrings, and each is a reason to flush more, never less:

  1. a write whose mutation target no argument names (the map's
     ``target_inferred`` rows, see ``UNNAMED_TARGET_WRITES``). The target is
     server-assigned or read out of a record, so no *earlier* read can have named
     it -- the read did not know the id either -- and the named parent is evicted
     normally. A write that names nothing at all flushes the whole index.
  2. a read whose arguments carry no mutable entity id (``find_user_id_by_email``,
     ``find_user_id_by_name_zip``). It cannot be joined to a write by id, so it is
     filed under ``LOOKUP_SCOPE`` and flushed by the writes that can falsify it
     (``LOOKUP_INVALIDATING_WRITES``).
  3. catalog reads (``list_all_*``, ``search_*``, ``calculate``). No tool in the
     map writes a catalog, so nothing flushes them. If a future write does, add it
     to ``LOOKUP_INVALIDATING_WRITES``.
  4. a failed write changed nothing, so it need not invalidate; calling
     ``invalidate`` anyway is merely extra eviction, and over-eviction is sound.

Conventions
-----------
Pure standard library; no clock, environment, network or randomness is read. The
only side effect is one optional file read at import (the committed map), and the
tables are built in sorted order so import is deterministic for given bytes. Keys
are lowercase hex sha256. ``key_blind`` is byte-compatible with the published
recipe ``sha256("tool|" + json.dumps(args, sort_keys=True, default=str))`` used by
``stale_window_audit.py`` and ``real_trace_external_baselines.py``, so digests and
reuse counts are comparable across the suite; ``key_typed`` is domain-separated
from it and cannot collide.

Usage::

    import cache_identity as ci

    index: dict[str, set[str]] = {}
    key = ci.remember(index, "get_order_details", {"order_id": "#W2378156"},
                      tenant="shop-a")
    cache[key] = output                      # store a read
    ...
    for key in ci.invalidate(index, "cancel_pending_order",
                             {"order_id": "#W2378156"}):
        cache.pop(key, None)                 # a write landed: evict what it touched

The embedded table mirrors :file:`tool_effect_map.json` field for field (the
fields this module consumes). If the file is absent or unreadable the embedded
copy is authoritative, so the module keeps working; the self-check fails if the
two disagree, which is the prompt to update the copy in the same commit as the
map.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from pathlib import Path
from typing import Any, Final

__all__ = [
    "CONTROL",
    "EFFECT_CLASSES",
    "EMBEDDED_TOOL_TABLE",
    "ENTITY_ARG_KEYS",
    "ENTITY_TYPE_OF_ARG",
    "LOOKUP_INVALIDATING_WRITES",
    "LOOKUP_READS",
    "LOOKUP_SCOPE",
    "MAP_ERROR",
    "MAP_PATH",
    "MAP_SOURCE",
    "READ",
    "UNNAMED_TARGET_WRITES",
    "WRITE",
    "effect_class",
    "entity_keys",
    "invalidate",
    "is_cacheable",
    "key_blind",
    "key_typed",
    "remember",
]

READ: Final[str] = "read"
WRITE: Final[str] = "write"
CONTROL: Final[str] = "control"
_CLASSES: Final[frozenset[str]] = frozenset({READ, WRITE, CONTROL})

# ---------------------------------------------------------------------------
# The effect map: (class, entity-identifying argument keys, target_inferred)
#
# Copied field for field from examples/benchmarks/tool_effect_map.json, which is
# itself derived from the tau-bench recordings. Kept as a literal so a checkout
# that loses the JSON (or imports this module from a wheel) still gets the exact
# same policy, and checked against the JSON at import so the two cannot drift
# silently.
# ---------------------------------------------------------------------------
EMBEDDED_TOOL_TABLE: Final[dict[str, tuple[str, tuple[str, ...], bool]]] = {
    "book_reservation": ("write", ("user_id", "payment_methods[].payment_id"), True),
    "calculate": ("read", ("expression",), False),
    "cancel_pending_order": ("write", ("order_id",), True),
    "cancel_reservation": ("write", ("reservation_id",), True),
    "exchange_delivered_order_items": (
        "write",
        ("order_id", "item_ids", "new_item_ids", "payment_method_id"),
        False,
    ),
    "find_user_id_by_email": ("read", ("email",), False),
    "find_user_id_by_name_zip": ("read", ("first_name", "last_name", "zip"), False),
    "get_order_details": ("read", ("order_id",), False),
    "get_product_details": ("read", ("product_id",), False),
    "get_reservation_details": ("read", ("reservation_id",), False),
    "get_user_details": ("read", ("user_id",), False),
    "list_all_airports": ("read", (), False),
    "list_all_product_types": ("read", (), False),
    "modify_pending_order_address": ("write", ("order_id",), False),
    "modify_pending_order_items": (
        "write",
        ("order_id", "item_ids", "new_item_ids", "payment_method_id"),
        False,
    ),
    "modify_pending_order_payment": ("write", ("order_id", "payment_method_id"), False),
    "modify_user_address": ("write", ("user_id",), False),
    "return_delivered_order_items": (
        "write",
        ("order_id", "item_ids", "payment_method_id"),
        False,
    ),
    "search_direct_flight": ("read", ("origin", "destination", "date"), False),
    "search_onestop_flight": ("read", ("origin", "destination", "date"), False),
    "send_certificate": ("write", ("user_id", "amount"), True),
    "think": ("control", ("thought",), False),
    "transfer_to_human_agents": ("control", ("summary",), False),
    "update_reservation_baggages": ("write", ("reservation_id", "payment_id"), False),
    "update_reservation_flights": ("write", ("reservation_id", "payment_id"), False),
    "update_reservation_passengers": ("write", ("reservation_id",), False),
}

# Which namespace each entity-identifying argument belongs to. An id is only
# unique inside its namespace (a user id and a product id can both be "123"), so
# the index token is "type:id" rather than the bare id. Keys not listed here fall
# back to their own name as the type, so an argument the table gains later is
# scoped by construction rather than silently dropped.
ENTITY_TYPE_OF_ARG: Final[dict[str, str]] = {
    "order_id": "order",
    "user_id": "user",
    "reservation_id": "reservation",
    "product_id": "product",
    "email": "email",
    "first_name": "name_zip",
    "last_name": "name_zip",
    "zip": "name_zip",
    "item_ids": "item",
    "new_item_ids": "item",
    "payment_method_id": "payment_method",
    "payment_id": "payment_method",
    "payment_methods[].payment_id": "payment_method",
    "origin": "flight",
    "destination": "flight",
    "date": "flight",
    "expression": "expression",
    "amount": "amount",
    "thought": "thought",
    "summary": "summary",
}

# Reads that name no id: they resolve a lookup key to a record. Their key is filed
# under LOOKUP_SCOPE as well as their own lookup token, because no write can name
# them by id.
LOOKUP_SCOPE: Final[str] = "*lookup*"
LOOKUP_READS: Final[frozenset[str]] = frozenset(
    {"find_user_id_by_email", "find_user_id_by_name_zip"}
)

# Writes that can falsify a lookup read. ``modify_user_address`` rewrites the zip
# that ``find_user_id_by_name_zip`` matches on, so every cached name+zip answer may
# be wrong afterwards -- for the old zip (the user no longer lives there) and for
# the new one (they now do). ``find_user_id_by_email`` is flushed with it because
# both lookups resolve to a user and an extra eviction is sound where a missed one
# is not. No other write in the map changes a lookup field or a catalog.
LOOKUP_INVALIDATING_WRITES: Final[frozenset[str]] = frozenset({"modify_user_address"})

_MapRow = tuple[str, tuple[str, ...], bool]
_MAP_FILENAME: Final[str] = "tool_effect_map.json"


def _map_candidates() -> tuple[Path, ...]:
    """Where the committed map may live, in precedence order.

    The real location is the module's own directory; the other two cover the
    ``tools/tool_effect_map.json`` spelling and a copy at the repository root, so
    moving the file does not silently drop the table on the floor.
    """
    here = Path(__file__).resolve().parent
    return (
        here / _MAP_FILENAME,
        here / "tools" / _MAP_FILENAME,
        here.parent.parent / "tools" / _MAP_FILENAME,
    )


def _rows_from_document(document: Any, source: str) -> dict[str, _MapRow]:
    """Extract the fields this module consumes from a ``tool_effect_map/v1`` doc."""
    if not isinstance(document, dict):
        raise TypeError(f"{source}: top level is not a JSON object")
    tools = document.get("tools")
    if not isinstance(tools, list):
        raise TypeError(f"{source}: no 'tools' list")
    rows: dict[str, _MapRow] = {}
    for entry in tools:
        if not isinstance(entry, dict):
            raise TypeError(f"{source}: a tool entry is not an object")
        name = entry.get("name")
        cls = entry.get("class")
        keys = entry.get("entity_keys") or []
        if not isinstance(name, str) or not name:
            raise ValueError(f"{source}: a tool entry has no name")
        if cls not in _CLASSES:
            raise ValueError(f"{source}: {name}: class {cls!r} is not read/write/control")
        if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
            raise TypeError(f"{source}: {name}: entity_keys is not a list of strings")
        rows[name] = (str(cls), tuple(keys), bool(entry.get("target_inferred")))
    if not rows:
        raise ValueError(f"{source}: 'tools' is empty")
    return rows


def _load_tool_table() -> tuple[dict[str, _MapRow], str, Path | None, str | None]:
    """Load the map file over the embedded copy; return ``(rows, source, path, error)``.

    Soundness. The file is the single source of truth for the policy, so it wins
    per tool; the embedded rows fill any tool the file omits, so a truncated file
    cannot quietly narrow the map. An unusable file is reported, not raised: a
    half-written JSON must not make every importer of this module fail, and the
    embedded copy is sufficient to keep working. The self-check turns a reported
    error into a failed check, so the defect is still seen.
    """
    for path in _map_candidates():
        if not path.is_file():
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            rows = _rows_from_document(document, path.name)
        except (OSError, ValueError, TypeError) as exc:
            return dict(EMBEDDED_TOOL_TABLE), "embedded", path, f"{path.name}: {exc}"
        merged = dict(EMBEDDED_TOOL_TABLE)
        merged.update(rows)
        return merged, "file", path, None
    return dict(EMBEDDED_TOOL_TABLE), "embedded", None, None


_TOOL_TABLE, MAP_SOURCE, MAP_PATH, MAP_ERROR = _load_tool_table()


def _sorted_tools() -> tuple[str, ...]:
    return tuple(sorted(_TOOL_TABLE))


#: tool -> ``"read"`` | ``"write"`` | ``"control"``. The mapping required by the
#: module's contract, derived from the map file when it is present and from
#: :data:`EMBEDDED_TOOL_TABLE` otherwise. Always consult it through
#: :func:`effect_class`, which defaults an unknown tool to ``"write"``.
EFFECT_CLASSES: Final[dict[str, str]] = {
    name: _TOOL_TABLE[name][0] for name in _sorted_tools()
}

#: tool -> the argument keys that identify the entity it touches (payload
#: arguments excluded; ``payment_methods[].payment_id`` is dotted because it is
#: nested in a list argument).
ENTITY_ARG_KEYS: Final[dict[str, tuple[str, ...]]] = {
    name: _TOOL_TABLE[name][1] for name in _sorted_tools()
}

#: Writes whose effect reaches an entity that no argument names -- a server
#: assigned id, or a refund target read out of a record. Precise eviction remains
#: sound for them because no earlier read could name that entity either; the flag
#: is here so the assumption is visible rather than implicit.
UNNAMED_TARGET_WRITES: Final[frozenset[str]] = frozenset(
    name
    for name in _sorted_tools()
    if _TOOL_TABLE[name][2] and _TOOL_TABLE[name][0] == WRITE
)


# ---------------------------------------------------------------------------
# Effects: may this call be stored at all?
# ---------------------------------------------------------------------------
def effect_class(tool: str) -> str:
    """``"read"``, ``"write"`` or ``"control"`` for ``tool``; unknown means write.

    Soundness. The cache's policy is only as sound as this function: one tool
    wrongly called a read turns a mutation into a hit, and the agent's belief that
    it acted becomes false. Two rules keep that from happening silently. Known
    tools are looked up in the committed map, whose class is derived from the verb
    of the name and corroborated against the recorded arguments, so the assignment
    is auditable rather than invented here. Unknown tools are classed ``"write"``
    -- the map's own rule 4, "unknown -> effectful", and the same conservative
    default the pilot uses -- so a tool this module has never seen cannot reach the
    cache by being presumed harmless.

    ``"write"`` is therefore a statement of risk, not of fact: for a tool we
    cannot classify the cost is a lost hit, never a wrong answer.
    """
    return EFFECT_CLASSES.get(tool, WRITE)


def is_cacheable(tool: str) -> bool:
    """True iff ``tool`` is a read, i.e. replaying it cannot change server state.

    Soundness. A hit returns a value captured earlier *without running the tool*,
    so it is sound exactly when running the tool would have been unobservable.
    Reads have that property by the map's definition -- they create, change and
    remove nothing persisted. Writes do not, and memoising one is the classic
    false-success bug: the second ``cancel_pending_order`` is answered from cache
    and never lands, while the agent reports a cancellation. Control tools are
    excluded too: ``think`` returns an episode-local reasoning string rather than a
    server record, and ``transfer_to_human_agents`` ends the episode, so a hit
    would silently fail to transfer. Unknown tools fall out as writes.

    Scope of the claim: cacheable is about *replay*, not about *freshness*. A
    cacheable read can still return a value that a later write made stale; that is
    what :func:`invalidate` is for. This is also why the cacheable set here is the
    11 reads and not the pilot's ``read + think`` set: the published 1,789 / 11,240
    "side-effect-free" counts include ``think``, which is pure but is not a data
    answer, so a typed value cache excludes it.
    """
    return effect_class(tool) == READ


# ---------------------------------------------------------------------------
# Identity: which entity ids does a call name?
# ---------------------------------------------------------------------------
def _flatten(value: Any) -> Iterator[Any]:
    """Yield the scalars of a possibly list-valued argument; skip nulls and maps."""
    if value is None or isinstance(value, Mapping):
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten(item)
        return
    yield value


def _values(args: Mapping[str, Any], arg_key: str) -> Iterator[Any]:
    """Yield the raw values named by ``arg_key``, following one level of ``[]``."""
    if "[]" in arg_key:
        head, _, tail = arg_key.partition("[].")
        raw = args.get(head)
        if not isinstance(raw, (list, tuple)):
            return
        for item in raw:
            if isinstance(item, Mapping):
                yield from _flatten(item.get(tail))
        return
    yield from _flatten(args.get(arg_key))


def _id_text(value: Any) -> str:
    """Stable text for an entity id.

    Strings are used verbatim -- an id's spelling is part of its identity, so
    nothing is stripped, lowercased or otherwise normalised. Other JSON values are
    serialised canonically, which keeps ``1`` and ``"1"`` the same entity while
    still failing loudly (``TypeError``) on values JSON cannot represent, rather
    than falling back on ``repr()`` whose ordering can depend on the process hash
    seed.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _refs(tool: str, args: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """``(entity type, id)`` pairs the call names, deduplicated, map order.

    Soundness. Only the map's identity subset is read, never the whole argument
    dict: ``modify_pending_order_address`` carries seven arguments and one id, and
    a payload argument promoted to an entity would evict a record the call never
    touched. Composite lookups contribute every part, so two different name+zip
    lookups cannot collapse onto one token. Nested ids are read out of list
    arguments. A malformed argument (a JSON string where a list of dicts belongs)
    contributes no id at all, which can only fail to evict; it can never evict the
    wrong record.
    """
    refs: list[tuple[str, str]] = []
    for arg_key in ENTITY_ARG_KEYS.get(tool, ()):
        entity = ENTITY_TYPE_OF_ARG.get(arg_key, arg_key)
        for value in _values(args, arg_key):
            text = _id_text(value)
            if not text:
                continue
            ref = (entity, text)
            if ref not in refs:
                refs.append(ref)
    return tuple(refs)


def entity_keys(tool: str, args: Mapping[str, Any]) -> tuple[str, ...]:
    """The entity ids a call names, in the map's order, without duplicates.

    This is the join between a read and a write: the same tuple on both sides
    means the write can have falsified the read's answer. Order and deduplication
    are for readability only -- ``invalidate`` works on set membership.

    Soundness. The tuple is the *identity* subset of the arguments, taken from the
    committed map (``ENTITY_ARG_KEYS``), not the argument list: ``modify_user_address``
    names ``user_id`` and carries a whole new address, and admitting the payload
    would evict a user that the call did not touch. Composite lookups list every
    part, so ``find_user_id_by_name_zip`` cannot collide with itself across
    different names or zips. Ids are returned as text, unnormalised, because the
    spelling is part of the identity; ``#W2378156`` and ``W2378156`` are different
    strings and the corpus uses one spelling per namespace.

    Two honest wrinkles, both from the map rather than this function. A lookup
    read names no *record*: ``find_user_id_by_name_zip`` returns a user id but its
    ids are ``first_name``, ``last_name``, ``zip``, so its tuple cannot join a
    write on ``user_id`` (see ``LOOKUP_SCOPE``). And tools with no server entity
    return whatever the map lists -- ``calculate`` returns its expression,
    ``send_certificate`` returns ``user_id`` and ``amount`` -- because the map's
    identity subset is the contract; neither token is ever evicted by a write.

    An unknown tool returns ``()``: it is classed as a write, so it is never
    cached, and there is no map row to read ids from.
    """
    keys: list[str] = []
    for _entity, ident in _refs(tool, args):
        if ident not in keys:
            keys.append(ident)
    return tuple(keys)


def _scope_tokens(tool: str, args: Mapping[str, Any]) -> tuple[str, ...]:
    """Index tokens (``"order:#W1"``) for the entities a call names."""
    tokens: list[str] = []
    for entity, ident in _refs(tool, args):
        token = f"{entity}:{ident}"
        if token not in tokens:
            tokens.append(token)
    return tuple(tokens)


# ---------------------------------------------------------------------------
# Keys: when are two calls the same call?
# ---------------------------------------------------------------------------
def _args_text(args: Mapping[str, Any]) -> str:
    """The published canonical form: sorted keys, ``default=str`` fallback."""
    try:
        return json.dumps(dict(args), sort_keys=True, default=str)
    except Exception:  # pragma: no cover - mirrors the published baseline
        return str(args)


def key_blind(tool: str, args: Mapping[str, Any]) -> str:
    """sha256 over the tool and its canonical arguments -- the unsound baseline.

    This is the key the suite's published reuse figures are measured under, and it
    is reproduced here byte for byte (``sha256("tool|" + json.dumps(args,
    sort_keys=True, default=str))``) so digests stay comparable with
    ``stale_window_audit.py`` and ``real_trace_external_baselines.py``. It is
    reproduced, not endorsed. Its soundness argument is only this: the digest is
    stable for one call's inputs and separates different inputs, so a hit is
    always the same *question*.

    It says nothing about the other two identities, and each omission is a defect
    the rest of this module exists to fix. No effect filter, so a write can be
    stored and served -- the mutation does not run, and the agent believes it did.
    No tenant, so if two tenants share a cache, one is answered with the other's
    row. No entity notion, so ``invalidate`` has nothing to join on and a cached
    read survives the write that falsified it. Determinism inheritance: the
    ``default=str`` fallback is kept for digest compatibility, so arguments holding
    objects whose ``str`` depends on insertion or hash order would poison the key;
    the corpus is JSON, and :func:`key_typed` refuses such arguments outright.
    """
    return hashlib.sha256(f"{tool}|{_args_text(args)}".encode("utf-8")).hexdigest()


_TYPED_DOMAIN: Final[str] = "cache_identity/typed/v1"


def _json_native(value: Any, path: str) -> None:
    """Reject anything JSON cannot represent injectively, naming where.

    Soundness. ``key_typed`` is injective only if its encoder is: ``json.dumps``
    silently coerces ``{1: "a"}`` to ``{"1": "a"}`` (and folds a dict with both
    keys into one), and accepts ``NaN`` as a non-standard token. Either would let
    two different argument mappings share a digest, so the argument tree is
    validated first and anything JSON-native does not cover -- a set, an object,
    a non-string key, a non-finite float -- is refused with a ``TypeError`` at the
    path that holds it. What remains serialises without surprises.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise TypeError(f"{path}: non-finite float {value!r} is not JSON")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _json_native(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path}: mapping key {key!r} is not a string")
            _json_native(item, f"{path}.{key}")
        return
    raise TypeError(f"{path}: {type(value).__name__} is not JSON-native")


def key_typed(tool: str, args: Mapping[str, Any], tenant: str | None = None) -> str:
    """Tenant-scoped, effect-typed key. Defined only for reads.

    Returns the lowercase hex sha256 of a domain tag, the tenant, the tool and the
    canonical arguments.

    Soundness. Four properties, each closing one hole in :func:`key_blind`.
    (a) Effect-typed: anything that is not a read raises, so a mutation can never
    be memoised -- a typed cache refuses to store a write rather than trusting
    callers to check first. (b) Tenant-bound: the tenant is inside the digest, and
    is encoded as JSON, so ``None`` and the string ``"None"`` are different scopes
    and no two tenants can share a key. (c) Argument-bound: the full canonical
    arguments are in the digest, so a hit is only ever for an identical request;
    no normalisation of ids happens on the way, because two spellings of an id are
    not the same request unless the server says so. (d) Domain-separated: the
    ``cache_identity/typed/v1`` tag keeps this key space disjoint from the blind
    baseline's, so the two can never accidentally share an entry.

    Entity ids are deliberately *not* folded in separately. They are a function of
    the arguments, so they cannot separate two calls the arguments do not already
    separate; the key answers "which question", and the index built by
    :func:`remember` answers "which record". What the key cannot prove is
    freshness: a hit means the same question, not the same answer. That is
    ``invalidate``'s job, and the key alone is never a licence to serve a value
    once a write to a named entity has landed.

    Raises:
        ValueError: if ``tool`` is not a read (writes and control tools are
            refused). This is the refusal the cache depends on: it fires before
            any argument is looked at.
        TypeError: if ``tenant`` is neither ``str`` nor ``None``, or if ``args``
            holds a value JSON cannot represent injectively (a set, a custom
            object, a non-string mapping key, a non-finite float) -- see
            :func:`_json_native`. Refusing is deliberate: coercing would let two
            different requests share a key.
    """
    cls = effect_class(tool)
    if cls != READ:
        raise ValueError(
            f"key_typed() is defined only for read tools; {tool!r} is {cls!r}. "
            "A typed cache must refuse to memoise a write (the mutation would "
            "never run) and a control tool (its value is an episode act or a "
            "reasoning step, not a server record)."
        )
    if tenant is not None and not isinstance(tenant, str):
        raise TypeError(f"tenant must be str or None, not {type(tenant).__name__}")
    payload = {
        "domain": _TYPED_DOMAIN,
        "tool": tool,
        "tenant": tenant,
        "args": dict(args),
    }
    _json_native(payload, "key")
    text = json.dumps(payload, sort_keys=True, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Eviction: which stored answers does a write obsolete?
# ---------------------------------------------------------------------------
def remember(
    index: MutableMapping[str, set[str]],
    tool: str,
    args: Mapping[str, Any],
    tenant: str | None = None,
) -> str:
    """File a read's key in ``index`` under every entity it names; return the key.

    ``index`` maps a scope token to the set of keys filed under it. Tokens are
    ``"<entity type>:<id>"`` (``"order:#W2378156"``, ``"user:u1"``,
    ``"payment_method:credit_card_1"``) so that ids from different namespaces
    cannot be confused, plus :data:`LOOKUP_SCOPE` for the id-less lookup reads.

    Soundness. Filing is what makes :func:`invalidate` total: a cached answer that
    names entity ``e`` becomes reachable from ``e``, so a write that names ``e`` can
    find it. Filing under *every* id the arguments carry, and unioning at eviction,
    means one reachable path is enough -- the failure mode that matters is a key
    filed nowhere, because that is a stale hit; a key filed twice is only an extra
    eviction. The key itself comes from :func:`key_typed`, so ``remember`` inherits
    the refusal to file writes and control tools and cannot be used to smuggle one
    into the cache.

    A caller that builds an index by hand must use the same tokens; if it files
    bare ids instead, :func:`invalidate` still finds them (it probes both), but the
    lookup scope is lost.

    An index shared by several tenants is safe and over-evicts: the key itself is
    tenant-bound, so no hit can cross tenants, but a write in one tenant evicts the
    same entity's keys in the others. Partition the index by tenant when that cost
    matters.
    """
    key = key_typed(tool, args, tenant)
    for token in _scope_tokens(tool, args):
        index.setdefault(token, set()).add(key)
    if tool in LOOKUP_READS:
        index.setdefault(LOOKUP_SCOPE, set()).add(key)
    return key


def _filed(index: Mapping[str, Iterable[str]], token: str) -> set[str]:
    """Keys filed under ``token``; a bare string counts as a one-key entry."""
    value = index.get(token)
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    return set(value)


def _all_keys(index: Mapping[str, Iterable[str]]) -> set[str]:
    """Every key filed anywhere in ``index``."""
    keys: set[str] = set()
    for value in index.values():
        if isinstance(value, str):
            keys.add(value)
        else:
            keys |= set(value)
    return keys


def invalidate(
    index: Mapping[str, Iterable[str]],
    tool: str,
    args: Mapping[str, Any],
) -> set[str]:
    """Keys a successful ``tool`` call must evict, given the ids it names.

    Returns the union of the keys filed under every entity the write names. The
    index is left untouched: eviction is the caller's move, and a key left in the
    index cannot be served stale because the cache no longer holds it -- if the
    read is repeated, :func:`remember` files the same key again, idempotently.

    Call this *after* the write returned success. A failed write changed nothing,
    so invalidating on failure is unnecessary (and merely extra eviction). A read
    or control tool returns the empty set: it cannot falsify anything.

    The premise is the index: every cached read must have been filed by
    :func:`remember` (or by an equivalent hand-built index). A read the caller
    stored without filing it cannot be found by any eviction policy, so filing is
    part of the cache's contract, not an optimisation.

    Soundness. Suppose a cached read ``r`` named entity ``e`` and the write ``w``
    named ``e``. Then ``r``'s key was filed under ``e``, and the write's id set
    contains ``e``, so the key is in the returned set. Contrapositive: every
    surviving key names no entity the write named. Three refinements keep the
    premise true rather than assumed.

    * A write whose ``target_inferred`` flag is set mutates an entity no argument
      names (a server-assigned reservation or certificate id, a refund target read
      out of a record). No *earlier* read can have named that entity either -- the
      id did not exist, or was not in the read's arguments -- so there is nothing
      to join and nothing missed; the named parent (``order_id``, ``user_id``,
      ``reservation_id``) is evicted normally. See :data:`UNNAMED_TARGET_WRITES`.
    * A read that names no id at all (``find_user_id_by_name_zip``) cannot be
      joined, so it is filed under :data:`LOOKUP_SCOPE`, and the writes that can
      falsify it (:data:`LOOKUP_INVALIDATING_WRITES`) flush that scope outright.
      Without this, changing a user's zip would leave a cached name+zip answer
      pointing at a user who no longer lives there.
    * A write with no usable identity -- unknown tool, or a call whose id
      arguments are missing -- returns every key in the index. Being unable to
      locate a mutation is not evidence that it touched nothing, so the sound
      fallback is a full flush, not an empty set.

    Over-eviction is possible and harmless: tokens are namespaced, but a
    hand-built index keyed by bare ids may collide across types, and an extra
    eviction costs a recomputation, never a wrong answer. Returning a ``set``
    makes that asymmetry explicit -- the caller copies out keys, and duplicate
    evictions collapse.
    """
    if effect_class(tool) != WRITE:
        return set()
    tokens = _scope_tokens(tool, args)
    if not tokens:
        return _all_keys(index)
    evicted: set[str] = set()
    for token in tokens:
        evicted |= _filed(index, token)
    for ident in entity_keys(tool, args):
        # Tolerance for an index filed by bare id instead of by scope token.
        evicted |= _filed(index, ident)
    if tool in LOOKUP_INVALIDATING_WRITES:
        evicted |= _filed(index, LOOKUP_SCOPE)
    return evicted


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def _class_counts() -> dict[str, int]:
    """Number of tools per effect class."""
    return {
        cls: sum(1 for value in EFFECT_CLASSES.values() if value == cls)
        for cls in (READ, WRITE, CONTROL)
    }


def _self_check() -> list[str]:
    """Return every failed invariant; an empty list means all of them held."""
    problems: list[str] = []

    def check(held: bool, message: str) -> None:
        if not held:
            problems.append(message)

    try:
        counts = _class_counts()
        check(len(EFFECT_CLASSES) == 26, f"map has {len(EFFECT_CLASSES)} tools, expected 26")
        check(counts == {READ: 11, WRITE: 13, CONTROL: 2}, f"classes are {counts}")
        check(MAP_ERROR is None, f"map file unusable: {MAP_ERROR}")
        if MAP_SOURCE == "file":
            drift = [
                name
                for name in sorted(EMBEDDED_TOOL_TABLE)
                if _TOOL_TABLE.get(name) != EMBEDDED_TOOL_TABLE[name]
            ]
            check(not drift, f"embedded table differs from the map file for {drift[:3]}")

        # Identity: the id subset only, composite lookups, nested payment ids.
        check(
            entity_keys("get_order_details", {"order_id": "#W1", "note": "x"}) == ("#W1",),
            "entity_keys(get_order_details) is not the order id alone",
        )
        check(
            entity_keys(
                "find_user_id_by_name_zip",
                {"first_name": "Mei", "last_name": "Kovacs", "zip": "8020"},
            )
            == ("Mei", "Kovacs", "8020"),
            "entity_keys(name+zip) is not every part",
        )
        check(
            entity_keys(
                "book_reservation",
                {
                    "user_id": "u1",
                    "payment_methods": [
                        {"payment_id": "credit_card_1"},
                        {"payment_id": "gift_card_2"},
                    ],
                },
            )
            == ("u1", "credit_card_1", "gift_card_2"),
            "entity_keys(book_reservation) misses a nested payment id",
        )
        check(entity_keys("list_all_airports", {}) == (), "a catalog read names an entity")

        # Keys: canonical, tool-bound, tenant-bound, disjoint from the baseline.
        args = {"order_id": "#W1", "reason": "no longer needed"}
        reordered = {"reason": "no longer needed", "order_id": "#W1"}
        check(
            key_blind("get_order_details", args) == key_blind("get_order_details", reordered),
            "key_blind is not canonical in argument order",
        )
        check(
            key_blind("get_order_details", args) != key_blind("get_user_details", args),
            "key_blind ignores the tool",
        )
        typed = key_typed("get_order_details", args, tenant="shop-a")
        check(
            typed == key_typed("get_order_details", reordered, tenant="shop-a"),
            "key_typed is not canonical in argument order",
        )
        check(
            typed != key_typed("get_order_details", args, tenant="shop-b"),
            "key_typed does not bind the tenant",
        )
        check(
            typed != key_blind("get_order_details", args),
            "key_typed collides with key_blind",
        )
        check(
            key_typed("calculate", {"expression": "1 + 1"})
            != key_typed("calculate", {"expression": "1 + 1"}, tenant="None"),
            "key_typed conflates tenant None with 'None'",
        )
        for tool in (
            "cancel_pending_order",
            "send_certificate",
            "think",
            "transfer_to_human_agents",
            "not_a_tool",
        ):
            try:
                key_typed(tool, {"order_id": "#W1", "user_id": "u1"})
            except ValueError:
                continue
            problems.append(f"key_typed accepted {tool!r}")
        for tool, want in (
            ("get_order_details", True),
            ("list_all_airports", True),
            ("modify_user_address", False),
            ("think", False),
            ("transfer_to_human_agents", False),
            ("not_a_tool", False),
        ):
            check(is_cacheable(tool) == want, f"is_cacheable({tool!r}) is not {want}")

        # Eviction: exact for a named write, lookup-flushed for a zip change,
        # full flush for a write whose identity cannot be read.
        index: dict[str, set[str]] = {}
        order1 = remember(index, "get_order_details", {"order_id": "#W1"}, tenant="t")
        order2 = remember(index, "get_order_details", {"order_id": "#W2"}, tenant="t")
        user = remember(index, "get_user_details", {"user_id": "u1"}, tenant="t")
        lookup = remember(
            index,
            "find_user_id_by_name_zip",
            {"first_name": "Mei", "last_name": "Kovacs", "zip": "8020"},
            tenant="t",
        )
        check(len({order1, order2, user, lookup}) == 4, "distinct reads share a key")
        try:
            remember(index, "cancel_pending_order", {"order_id": "#W1"}, tenant="t")
        except ValueError:
            pass
        else:
            problems.append("remember filed a write")

        evicted = invalidate(
            index,
            "modify_pending_order_address",
            {"order_id": "#W1", "address1": "1 Main St"},
        )
        check(evicted == {order1}, f"an order write evicted {len(evicted)} keys, expected 1")
        evicted = invalidate(index, "modify_user_address", {"user_id": "u1", "zip": "8021"})
        check(evicted == {user, lookup}, "a user write did not evict the name+zip lookup")
        check(not invalidate(index, "get_order_details", {"order_id": "#W1"}),
              "a read invalidated something")
        check(
            not invalidate(index, "cancel_pending_order", {"order_id": "#W9"}),
            "a write to an unrelated order evicted something",
        )
        check(
            invalidate(index, "who_knows", {"whatever": 1}) == {order1, order2, user, lookup},
            "a write naming no known entity did not flush the index",
        )
        check(
            invalidate(
                {"#W1": {order1}}, "modify_pending_order_address", {"order_id": "#W1"}
            )
            == {order1},
            "invalidate does not read a bare-id index",
        )
    except Exception as exc:  # a self-check reports; it does not raise
        problems.append(f"raised {type(exc).__name__}: {exc}")
    return problems


def _main() -> int:
    """Print the one-line self-check and return the process exit status."""
    problems = _self_check()
    if problems:
        print("cache_identity self-check: FAIL | " + " | ".join(problems))
        return 1
    counts = _class_counts()
    embedded = "identical" if MAP_SOURCE == "file" else "used"
    print(
        "cache_identity self-check: OK"
        f" | {len(EFFECT_CLASSES)} tools"
        f" ({counts[READ]} read / {counts[WRITE]} write / {counts[CONTROL]} control)"
        f" | map={MAP_SOURCE} embedded={embedded}"
        " | blind=canonical,no-tenant"
        " | typed=tenant-bound,refuses-write"
        " | evict=exact,lookup-flushed,unlocatable-flushes-all"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
