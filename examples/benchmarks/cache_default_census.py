#!/usr/bin/env python3
"""Census: every place the repo creates or fills a result cache.

This is a static AST pass over ``agentflow/**``, ``examples/**`` and
``scripts/**``.  It answers three questions:

(a) every constructor of ``InstructionCache`` or ``SoundResultCache`` --
    file, line, enclosing class/function, and whether a tenant/identity is
    available at that site;
(b) every ``.put(...)`` call on one of those stores -- the enclosing
    instruction class, its ``side_effects`` declaration (``False`` /
    ``True`` / undeclared), whether ``footprint(params)`` is overridden, and
    the bucket the typed default would place it in;
(c) every ``execute`` definition whose class also holds a result cache --
    whether a tenant reaches it (class attribute, argument, ContextVar, or
    nothing).

The classification buckets are fixed:

* ``pure_declared``      -- ``side_effects is False``; the typed default admits.
* ``effectful_declared`` -- ``side_effects is True``; the typed default refuses.
* ``undeclared``         -- no declaration; the legacy default is ``True``,
                            so the typed default refuses (fail closed).

Only ``pure_declared`` admits.  The JSON therefore separates the sites the
instruction-default flip actually governs (legacy ``InstructionCache`` put
sites inside ``Instruction`` subclasses) from sites that are already typed
(``SoundResultCache``) and from legacy stores outside the instruction
hierarchy (``PatternRecognizer``).

Every number in the output is computed from the tree; nothing is copied from
the gap report.  In particular the report's "~96" figure is reconciled
explicitly: this pass finds 96 ``execute`` definitions under
``agentflow/core``, 104 under ``agentflow/`` and 144 repo-wide.

Deterministic: files are walked in sorted order, sites are sorted by
``(file, line)``, no timestamps or wall-clock values enter the JSON.

Usage::

    ./venv/bin/python examples/benchmarks/cache_default_census.py
    ./venv/bin/python examples/benchmarks/cache_default_census.py --check
    ./venv/bin/python examples/benchmarks/cache_default_census.py --json /tmp/c.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import warnings
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

SCHEMA_VERSION = 1
ROOTS = ("agentflow", "examples", "scripts")
SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache", "venv", "node_modules"}

#: The two stores whose constructor/put sites are in scope for question (a)/(b).
CACHE_TYPES = ("InstructionCache", "SoundResultCache")
#: A second, independent result store the gap report calls out.  Put sites on it
#: are recorded but excluded from the flip totals because changing the
#: instruction default does not touch this class.
OTHER_STORE_TYPES = ("DecompositionCache",)
ALL_STORE_TYPES = CACHE_TYPES + OTHER_STORE_TYPES

BUCKETS = ("pure_declared", "effectful_declared", "undeclared")
BUCKET_ORDER = {name: i for i, name in enumerate(BUCKETS)}

#: Where ``InstructionCache`` and ``SoundResultCache`` are defined.  Used only
#: to distinguish the two ``Instruction`` hierarchies (``isa.instruction`` vs
#: ``instructions.base``), never to count anything.
INSTRUCTION_REF = ("agentflow.core.isa.instruction", "Instruction")
BASE_INSTRUCTION_REF = ("agentflow.core.instructions.base", "BaseInstruction")

#: Key-contract facts asserted by the source, with the lines they come from.
#: These are qualitative, not counted figures.
KEY_CONTRACT = {
    "InstructionCache": {
        "tenant_in_key": False,
        "evidence": "agentflow/core/isa/instruction_cache.py:78-98 "
                    "(build_cache_key has no tenant parameter; namespace is "
                    "model/template/decoding, not identity)",
    },
    "SoundResultCache": {
        "tenant_in_key": True,
        "evidence": "agentflow/core/isa/sound_cache.py:89-97 "
                    "(_cache_key payload scope is {kind: tenant, tenant} when "
                    "identity_dependent, 'shared' otherwise)",
    },
    "DecompositionCache": {
        "tenant_in_key": False,
        "evidence": "agentflow/core/instructions/objective_decomposition.py:42-84 "
                    "(key is the caller-supplied string; no identity component)",
    },
}


# ---------------------------------------------------------------------------
# small AST helpers
# ---------------------------------------------------------------------------
def literal_bool(node: Optional[ast.AST]) -> Optional[bool]:
    """Return the value of a boolean literal, else ``None``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return None


def expr_text(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - ast.unparse is total in 3.14
        return None


def call_name(node: ast.AST) -> Optional[str]:
    """Last component of a call target (``X`` or ``a.b.X`` -> ``X``)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def dotted_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = dotted_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return None


def walk_own(node: ast.AST) -> Iterator[ast.AST]:
    """Walk ``node`` without descending into nested classes or lambdas."""
    stack: List[ast.AST] = [node]
    first = True
    while stack:
        cur = stack.pop()
        if not first and isinstance(cur, (ast.ClassDef, ast.Lambda)):
            continue
        first = False
        yield cur
        stack.extend(reversed(list(ast.iter_child_nodes(cur))))


def enclosing_class(tree: ast.Module, lineno: int) -> Optional[ast.ClassDef]:
    best: Optional[ast.ClassDef] = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            end = node.end_lineno or node.lineno
            if node.lineno <= lineno <= end:
                if best is None or node.lineno >= best.lineno:
                    best = node
    return best


def enclosing_function(tree: ast.Module, lineno: int) -> Optional[ast.AST]:
    best: Optional[ast.AST] = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = node.end_lineno or node.lineno
            if node.lineno <= lineno <= end:
                if best is None or node.lineno >= best.lineno:
                    best = node
    return best


def function_chain(tree: ast.Module, lineno: int) -> List[ast.AST]:
    """Enclosing functions, innermost first."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = node.end_lineno or node.lineno
            if node.lineno <= lineno <= end:
                out.append(node)
    out.sort(key=lambda n: n.lineno, reverse=True)
    return out


def arg_names(fn: ast.AST) -> List[str]:
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    return [a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs]


def param_default(fn: ast.AST, name: str) -> Optional[ast.AST]:
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    args = fn.args
    positional = list(args.posonlyargs) + list(args.args)
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for arg, default in zip(positional, defaults):
        if arg.arg == name:
            return default
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        if arg.arg == name:
            return default
    return None


def annotation_name(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.split(".")[-1]
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return annotation_name(node.value)
    return None


# ---------------------------------------------------------------------------
# repo index: modules, imports, classes
# ---------------------------------------------------------------------------
class Repo:
    def __init__(self, root: Path):
        self.root = root
        self.files: List[Path] = []
        self.trees: Dict[Path, ast.Module] = {}
        self.modules: Dict[str, Path] = {}
        self.imports: Dict[Path, Dict[str, Tuple[str, str]]] = {}
        self.classes: Dict[Tuple[str, str], List[Tuple[Path, ast.ClassDef]]] = {}
        self._cache_types: Dict[Tuple[str, str], set] = {}
        self._declarations: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._scan()

    # -- scanning ----------------------------------------------------------
    def _scan(self) -> None:
        for root_name in ROOTS:
            base = self.root / root_name
            if not base.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
                for filename in sorted(filenames):
                    if filename.endswith(".py"):
                        self.files.append(Path(dirpath) / filename)
        self.files.sort()
        for path in self.files:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            self.trees[path] = tree
            module = self.module_name(path)
            self.modules[module] = path
            if path.name == "__init__.py":
                self.modules[module.rsplit(".", 1)[0] if "." in module else ""] = path
        for path in self.files:
            self.imports[path] = self._imports_of(path, self.trees[path])
        for path in self.files:
            module = self.module_name(path)
            for node in ast.walk(self.trees[path]):
                if isinstance(node, ast.ClassDef):
                    self.classes.setdefault((module, node.name), []).append((path, node))

    def module_name(self, path: Path) -> str:
        rel = path.relative_to(self.root).with_suffix("")
        return ".".join(rel.parts)

    def _imports_of(self, path: Path, tree: ast.Module) -> Dict[str, Tuple[str, str]]:
        result: Dict[str, Tuple[str, str]] = {}
        module = self.module_name(path)
        package = module.split(".")[:-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    trim = node.level - 1
                    base = package[: len(package) - trim] if trim else package
                    source = ".".join(base + ([node.module] if node.module else []))
                else:
                    source = node.module or ""
                for alias in node.names:
                    result[alias.asname or alias.name] = (source, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    local = alias.asname or alias.name.split(".")[0]
                    result[local] = (alias.name, "")
        return result

    # -- resolution --------------------------------------------------------
    def class_def(self, ref: Optional[Tuple[str, str]]) -> Optional[Tuple[Path, ast.ClassDef]]:
        if ref is None:
            return None
        defs = self.classes.get(ref, [])
        if not defs:
            return None
        # Prefer a definition in the module's own file, then the first.
        module_file = self.modules.get(ref[0])
        for path, node in defs:
            if path == module_file:
                return path, node
        return defs[0]

    def resolve_class_ref(self, path: Path, expr: ast.AST) -> Optional[Tuple[str, str]]:
        imports = self.imports.get(path, {})
        if isinstance(expr, ast.Name):
            if expr.id in imports:
                source, original = imports[expr.id]
                if original and (source, original) in self.classes:
                    return source, original
            module = self.module_name(path)
            if (module, expr.id) in self.classes:
                return module, expr.id
            for (mod, name) in self.classes:
                if name == expr.id and any(p == path for p, _ in self.classes[(mod, name)]):
                    return mod, name
            return None
        if isinstance(expr, ast.Attribute):
            dotted = dotted_name(expr)
            if not dotted:
                return None
            head, _, tail = dotted.rpartition(".")
            if (head, tail) in self.classes:
                return head, tail
            if dotted in self.modules:
                return None
        return None

    def base_refs(self, path: Path, cls: ast.ClassDef) -> List[Optional[Tuple[str, str]]]:
        return [self.resolve_class_ref(path, base) for base in cls.bases]

    def is_subclass_of(self, ref: Tuple[str, str], target: Tuple[str, str],
                       seen: Optional[set] = None) -> bool:
        if ref == target:
            return True
        seen = set() if seen is None else seen
        if ref in seen:
            return False
        seen.add(ref)
        found = self.class_def(ref)
        if found is None:
            return False
        path, node = found
        for base in self.base_refs(path, node):
            if base is not None and self.is_subclass_of(base, target, seen):
                return True
        return False

    # -- cache detection ---------------------------------------------------
    def _cache_assignments(self, path: Path, cls: ast.ClassDef) -> List[Dict[str, Any]]:
        """``self.<attr> = <store>(...)`` or a class-level ``<attr> = <store>``.

        Local variables inside a method (``cache = SoundResultCache(...)``) are
        *not* attributes and are deliberately ignored here; they are reported
        separately by :meth:`constructed_store_types`.
        """
        out: List[Dict[str, Any]] = []

        def add(node: ast.AST, value: Optional[ast.AST],
                targets: List[ast.AST], class_level: bool) -> None:
            if value is None:
                return
            cache_type = self.cache_type_of_value(path, value, cls)
            if cache_type is None:
                return
            for target in targets:
                if (isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"):
                    out.append({"attr": target.attr, "lineno": node.lineno,
                                "cache_type": cache_type,
                                "class_level": class_level})
                elif class_level and isinstance(target, ast.Name):
                    out.append({"attr": target.id, "lineno": node.lineno,
                                "cache_type": cache_type, "class_level": True})

        for node in cls.body:
            if isinstance(node, ast.Assign):
                add(node, node.value, list(node.targets), True)
            elif isinstance(node, ast.AnnAssign):
                add(node, node.value, [node.target], True)
        for stmt in cls.body:
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in walk_own(stmt):
                if isinstance(node, ast.Assign):
                    add(node, node.value, list(node.targets), False)
                elif isinstance(node, ast.AnnAssign):
                    add(node, node.value, [node.target], False)
        return out

    def dict_cache_assignments(self, path: Path, cls: ast.ClassDef) -> List[Dict[str, Any]]:
        """``self.<attr> = {}`` / ``OrderedDict()`` assignments in this class."""
        out: List[Dict[str, Any]] = []
        for stmt in cls.body:
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in walk_own(stmt):
                if not isinstance(node, ast.Assign):
                    continue
                value = node.value
                is_dict = isinstance(value, ast.Dict) or (
                    isinstance(value, ast.Call) and call_name(value.func) in
                    ("dict", "OrderedDict"))
                if not is_dict:
                    continue
                for target in node.targets:
                    if (isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                            and target.attr in ("cache", "_cache")):
                        out.append({"attribute": target.attr,
                                    "line": node.lineno,
                                    "method": stmt.name})
        return out

    def constructed_store_types(self, path: Path, cls: ast.ClassDef) -> List[Dict[str, Any]]:
        """Store constructors appearing anywhere in this class's own methods."""
        out: List[Dict[str, Any]] = []
        for stmt in cls.body:
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in walk_own(stmt):
                if not isinstance(node, ast.Call):
                    continue
                name = call_name(node.func)
                if name in ALL_STORE_TYPES:
                    out.append({"cache_type": name, "line": node.lineno,
                                "method": stmt.name})
        return out

    def cache_type_of_value(self, path: Path, value: ast.AST,
                            cls: Optional[ast.ClassDef] = None) -> Optional[str]:
        """Infer a store type from an assignment right-hand side."""
        if isinstance(value, ast.Call):
            name = call_name(value.func)
            if name in ALL_STORE_TYPES:
                return name
            if name and name.startswith("_get_") and cls is not None:
                found = self._method_return_type(path, cls, name)
                if found in ALL_STORE_TYPES:
                    return found
            return None
        if isinstance(value, ast.Attribute):
            if value.attr in ALL_STORE_TYPES:
                return value.attr
            return None
        if isinstance(value, ast.IfExp):
            left = self.cache_type_of_value(path, value.body, cls)
            right = self.cache_type_of_value(path, value.orelse, cls)
            if left and right and left != right:
                return None
            return left or right
        return None

    def _method_return_type(self, path: Path, cls: ast.ClassDef,
                            method_name: str) -> Optional[str]:
        found = self._find_method(path, cls, method_name, set())
        if found is None:
            return None
        _, fn = found
        return annotation_name(fn.returns)

    def _find_method(self, path: Path, cls: ast.ClassDef, name: str,
                     seen: set) -> Optional[Tuple[Path, ast.AST]]:
        for node in cls.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return path, node
        module = self.module_name(path)
        ref = (module, cls.name)
        if ref in seen:
            return None
        seen.add(ref)
        for base in self.base_refs(path, cls):
            found = self.class_def(base)
            if found is not None:
                result = self._find_method(found[0], found[1], name, seen)
                if result is not None:
                    return result
        return None

    def cache_types_for_class(self, ref: Tuple[str, str],
                              seen: Optional[set] = None) -> set:
        if ref in self._cache_types:
            return set(self._cache_types[ref])
        seen = set() if seen is None else seen
        if ref in seen:
            return set()
        seen.add(ref)
        found = self.class_def(ref)
        types: set = set()
        if found is not None:
            path, cls = found
            for entry in self._cache_assignments(path, cls):
                types.add(entry["cache_type"])
            for entry in self.constructed_store_types(path, cls):
                types.add(entry["cache_type"])
            for base in self.base_refs(path, cls):
                if base is not None:
                    types |= self.cache_types_for_class(base, seen)
        self._cache_types[ref] = set(types)
        return types

    def cache_attr_type(self, path: Path, cls: ast.ClassDef, attr: str,
                        seen: Optional[set] = None) -> Optional[str]:
        """Type of ``self.<attr>`` in ``cls`` or an ancestor."""
        seen = set() if seen is None else seen
        ref = (self.module_name(path), cls.name)
        if ref in seen:
            return None
        seen.add(ref)
        for entry in self._cache_assignments(path, cls):
            if entry["attr"] == attr:
                return entry["cache_type"]
        for base in self.base_refs(path, cls):
            found = self.class_def(base)
            if found is not None:
                result = self.cache_attr_type(found[0], found[1], attr, seen)
                if result is not None:
                    return result
        return None

    # -- side_effects declarations ----------------------------------------
    def side_effects_declaration(self, ref: Tuple[str, str],
                                 seen: Optional[set] = None) -> Dict[str, Any]:
        if ref in self._declarations:
            return dict(self._declarations[ref])
        seen = set() if seen is None else seen
        if ref in seen:
            return {"kind": "undeclared", "legacy_default": True, "source": "cycle"}
        seen.add(ref)
        found = self.class_def(ref)
        if found is None:
            return {"kind": "undeclared", "legacy_default": True, "source": "no_class"}
        path, cls = found
        own = self._own_declaration(path, cls)
        if own is not None:
            self._declarations[ref] = dict(own)
            return dict(own)
        # An explicit declaration on an ancestor is inherited.
        for base in self.base_refs(path, cls):
            if base is None:
                continue
            inherited = self.side_effects_declaration(base, seen)
            if inherited["kind"] != "undeclared":
                result = dict(inherited)
                result["source"] = "inherited"
                result["inherited_from"] = f"{base[0]}.{base[1]}"
                self._declarations[ref] = dict(result)
                return dict(result)
        result = {"kind": "undeclared", "legacy_default": True, "source": "none"}
        self._declarations[ref] = dict(result)
        return dict(result)

    def declaration_for_node(self, path: Path, cls: ast.ClassDef,
                             seen: Optional[set] = None) -> Dict[str, Any]:
        """Declaration for a specific ClassDef node (not a name lookup)."""
        own = self._own_declaration(path, cls)
        if own is not None:
            return dict(own)
        seen = set() if seen is None else seen
        ref = (self.module_name(path), cls.name)
        if ref in seen:
            return {"kind": "undeclared", "legacy_default": True, "source": "cycle"}
        seen.add(ref)
        for base in self.base_refs(path, cls):
            if base is None:
                continue
            inherited = self.side_effects_declaration(base, seen)
            if inherited["kind"] != "undeclared":
                result = dict(inherited)
                result["source"] = "inherited"
                result["inherited_from"] = f"{base[0]}.{base[1]}"
                return result
        return {"kind": "undeclared", "legacy_default": True, "source": "none"}

    def _own_declaration(self, path: Path, cls: ast.ClassDef) -> Optional[Dict[str, Any]]:
        # 1. class body: side_effects = False / side_effects: bool = True
        for node in cls.body:
            value = None
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "side_effects":
                        value = node.value
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.target.id == "side_effects":
                    value = node.value
            if value is not None:
                literal = literal_bool(value)
                if literal is not None:
                    return {"kind": "literal", "value": literal,
                            "source": "class_attribute",
                            "lineno": node.lineno}
                return {"kind": "unknown", "value": None,
                        "expression": expr_text(value),
                        "source": "class_attribute", "lineno": node.lineno}
        init = next((n for n in cls.body
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.name == "__init__"), None)
        if init is None:
            return None
        # 2. explicit self.side_effects = <literal>
        for node in walk_own(init):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and target.attr == "side_effects"):
                    literal = literal_bool(node.value)
                    if literal is not None:
                        return {"kind": "literal", "value": literal,
                                "source": "instance_attribute",
                                "lineno": node.lineno}
                    if isinstance(node.value, ast.Name):
                        # ``self.side_effects = side_effects`` only forwards the
                        # constructor parameter.  For the base Instruction that
                        # parameter's default (True) *is* the legacy default, not
                        # a declaration by the subclass, so this is recorded as
                        # undeclared rather than inherited.
                        return None
                    return {"kind": "unknown", "value": None,
                            "expression": expr_text(node.value),
                            "source": "instance_attribute", "lineno": node.lineno}
        # 3. super().__init__(..., side_effects=<expr>)
        for node in walk_own(init):
            if not isinstance(node, ast.Call):
                continue
            if not (isinstance(node.func, ast.Attribute) and node.func.attr == "__init__"):
                continue
            for kw in node.keywords:
                if kw.arg != "side_effects":
                    continue
                parsed = self._declaration_from_expr(init, kw.value)
                if parsed is not None:
                    parsed["lineno"] = node.lineno
                    return parsed
        return None

    def _declaration_from_expr(self, init: ast.AST,
                               expr: ast.AST) -> Optional[Dict[str, Any]]:
        literal = literal_bool(expr)
        if literal is not None:
            return {"kind": "literal", "value": literal,
                    "source": "super_init_keyword"}
        if isinstance(expr, ast.IfExp):
            # ``True if side_effects is None else side_effects``
            default = literal_bool(expr.body)
            passthrough = isinstance(expr.orelse, ast.Name)
            if default is not None and passthrough:
                # ``True if side_effects is None else side_effects``: the
                # parameter being None means "use the literal default", and a
                # caller-supplied boolean overrides it.  The parameter's own
                # default (None) is therefore not the effective default.
                return {"kind": "conditional", "value": default,
                        "default_value": default,
                        "overridable": True,
                        "expression": expr_text(expr),
                        "source": "super_init_keyword"}
            return {"kind": "conditional", "value": None,
                    "default_value": None, "overridable": True,
                    "expression": expr_text(expr),
                    "source": "super_init_keyword"}
        if isinstance(expr, ast.Name):
            default = literal_bool(param_default(init, expr.id))
            return {"kind": "conditional", "value": default,
                    "default_value": default, "overridable": True,
                    "expression": expr_text(expr),
                    "source": "super_init_keyword"}
        return {"kind": "unknown", "value": None,
                "expression": expr_text(expr), "source": "super_init_keyword"}

    # -- receiver type inference ------------------------------------------
    def infer_receiver_type(self, path: Path, tree: ast.Module,
                            call: ast.Call, cls: Optional[ast.ClassDef]) -> Optional[str]:
        """Best-effort type of the object a ``.put`` is called on."""
        func = call.func
        if not isinstance(func, ast.Attribute):
            return None
        receiver = func.value
        if isinstance(receiver, ast.Call):
            return self.cache_type_of_value(path, receiver, cls)
        if isinstance(receiver, ast.Attribute):
            # self.cache / instr.cache / sound.X
            owner = receiver.value
            attr = receiver.attr
            if attr in ALL_STORE_TYPES:
                return attr
            if isinstance(owner, ast.Name) and owner.id == "self":
                if cls is None:
                    return None
                return self.cache_attr_type(path, cls, attr)
            if isinstance(owner, ast.Name):
                owner_cls = self._param_class(path, tree, call, owner.id)
                if owner_cls is not None and attr == "cache":
                    return self.cache_attr_type(owner_cls[0], owner_cls[1], "cache")
            return None
        if isinstance(receiver, ast.Name):
            expr = self._find_assignment(path, tree, call.lineno, receiver.id)
            if expr is None:
                return None
            return self.cache_type_of_value(path, expr, cls)
        return None

    def _find_assignment(self, path: Path, tree: ast.Module, lineno: int,
                         name: str) -> Optional[ast.AST]:
        candidates: List[Tuple[int, ast.AST]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == name:
                        candidates.append((node.lineno, node.value))
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.target.id == name:
                    candidates.append((node.lineno, node.value))
        before = [c for c in candidates if c[0] <= lineno and c[1] is not None]
        if not before:
            return None
        before.sort(key=lambda c: c[0])
        return before[-1][1]

    def _param_class(self, path: Path, tree: ast.Module, call: ast.Call,
                     param: str) -> Optional[Tuple[Path, ast.ClassDef]]:
        """Resolve a parameter name to the class of its annotated/unpacked value."""
        candidates: List[ast.AST] = function_chain(tree, call.lineno)
        annotation_candidates: List[str] = []
        for node in candidates:
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:  # type: ignore[union-attr]
                if arg.arg != param:
                    continue
                name = annotation_name(arg.annotation)
                if name:
                    annotation_candidates.append(name)
        for name in annotation_candidates:
            ref = self.resolve_class_ref(path, ast.Name(id=name, ctx=ast.Load()))
            if ref is not None:
                return self.class_def(ref)
            # The annotation may be a post-poned string with no import in this
            # file (``from __future__ import annotations``).  Match by class
            # name when exactly one such class holds a result cache.
            matches = [r for r in self.classes
                       if r[1] == name
                       and (self.cache_types_for_class(r) & set(ALL_STORE_TYPES))]
            if len(matches) == 1:
                return self.class_def(matches[0])
        # No (resolvable) annotation: infer from a method call on the same
        # object inside the enclosing function (``instr._get_cache_key(...)``).
        ref = self._class_from_method_usage(tree, call.lineno, param)
        if ref is not None:
            return self.class_def(ref)
        # Last resort, recorded in the site's evidence: a parameter literally
        # named ``instr`` on a helper that calls ``_get_cache_key`` on it is an
        # Instruction by the factory convention used across the examples.
        if param == "instr":
            matches = [r for r in self.classes
                       if r[1] == "Instruction"
                       and (self.cache_types_for_class(r) & set(CACHE_TYPES))]
            if len(matches) == 1:
                return self.class_def(matches[0])
        return None

    def _class_from_method_usage(self, tree: ast.Module, lineno: int,
                                 param: str) -> Optional[Tuple[str, str]]:
        chain = function_chain(tree, lineno)
        if not chain:
            return None
        method_names = set()
        for fn in chain:
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == param):
                    method_names.add(node.func.attr)
        if "_get_cache_key" not in method_names:
            return None
        hits: List[Tuple[str, str]] = []
        for ref in self.classes:
            found = self.class_def(ref)
            if found is None:
                continue
            names = {n.name for n in found[1].body
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            if "_get_cache_key" in names and (method_names & names):
                hits.append(ref)
        uniq = sorted(set(hits))
        return uniq[0] if len(uniq) == 1 else None


# ---------------------------------------------------------------------------
# census
# ---------------------------------------------------------------------------
class Census:
    def __init__(self, root: Path):
        self.repo = Repo(root)
        self.constructors: List[Dict[str, Any]] = []
        self.out_of_scope_constructors: List[Dict[str, Any]] = []
        self.put_sites: List[Dict[str, Any]] = []
        self.out_of_scope_put_sites: List[Dict[str, Any]] = []
        self.execute_defs: List[Dict[str, Any]] = []
        self.excluded: List[Dict[str, Any]] = []

    # -- (a) constructors --------------------------------------------------
    def collect_constructors(self) -> None:
        for path in self.repo.files:
            tree = self.repo.trees[path]
            for call in ast.walk(tree):
                if not isinstance(call, ast.Call):
                    continue
                name = call_name(call.func)
                if name not in ALL_STORE_TYPES:
                    continue
                cls = enclosing_class(tree, call.lineno)
                cls_ref = None
                if cls is not None:
                    cls_ref = (self.repo.module_name(path), cls.name)
                fn = enclosing_function(tree, call.lineno)
                record = {
                    "file": self._rel(path),
                    "line": call.lineno,
                    "cache_type": name,
                    "enclosing_class": cls.name if cls else None,
                    "enclosing_function": fn.name if fn else "<module>",
                    "constructor_accepts_tenant": self._constructor_accepts_tenant(name),
                    "tenant_available_at_construction": False,
                    "tenant_at_construction": {
                        "available": False,
                        "reason": self._tenant_construction_reason(name),
                    },
                    "tenant_available_in_enclosing_scope": self._tenant_in_scope(
                        tree, call.lineno, cls, fn),
                    "tenant_channel_at_use": self._tenant_channel_at_use(name, tree, call.lineno),
                    "key_contract": KEY_CONTRACT.get(name, {}),
                    "already_typed": name == "SoundResultCache",
                }
                if name in CACHE_TYPES:
                    self.constructors.append(record)
                else:
                    record["exclusion_reason"] = (
                        "DecompositionCache is a second, independent TTL/LRU store; "
                        "changing the Instruction cache default does not touch it"
                    )
                    self.out_of_scope_constructors.append(record)
        self.constructors.sort(key=lambda r: (r["file"], r["line"]))
        self.out_of_scope_constructors.sort(key=lambda r: (r["file"], r["line"]))

    def _constructor_accepts_tenant(self, cache_type: str) -> bool:
        ref = self._store_class_ref(cache_type)
        found = self.repo.class_def(ref)
        if found is None:
            return False
        path, cls = found
        init = next((n for n in cls.body
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.name == "__init__"), None)
        return init is not None and any("tenant" in a for a in arg_names(init))

    def _store_class_ref(self, name: str) -> Optional[Tuple[str, str]]:
        for ref in self.repo.classes:
            if ref[1] == name:
                found = self.repo.class_def(ref)
                if found is not None and self._class_defines_store(found[1]):
                    return ref
        return None

    @staticmethod
    def _class_defines_store(cls: ast.ClassDef) -> bool:
        methods = {n.name for n in cls.body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        return {"get", "put"} <= methods

    def _tenant_in_scope(self, tree: ast.Module, lineno: int,
                         cls: Optional[ast.ClassDef],
                         fn: Optional[ast.AST]) -> bool:
        if fn is not None and any("tenant" in a for a in arg_names(fn)):
            return True
        if cls is not None:
            for node in ast.walk(cls):
                if isinstance(node, ast.arg) and "tenant" in node.arg:
                    return True
                if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                        and node.value.id == "self" and "tenant" in node.attr):
                    return True
        if fn is not None:
            for node in ast.walk(fn):
                if isinstance(node, ast.Name) and "tenant" in node.id:
                    return True
                if isinstance(node, ast.arg) and "tenant" in node.arg:
                    return True
                if (isinstance(node, ast.Attribute) and "tenant" in node.attr):
                    return True
        return False

    @staticmethod
    def _tenant_construction_reason(cache_type: str) -> str:
        if cache_type == "SoundResultCache":
            return ("__init__(max_size, ttl) has no tenant parameter; get/put "
                    "resolve tenant= or the current_tenant ContextVar per call")
        if cache_type == "InstructionCache":
            return ("__init__(max_size, ttl) has no tenant parameter and the key "
                    "contract has no identity component")
        return "caller supplies the string key; the store has no identity field"

    def _tenant_channel_at_use(self, cache_type: str, tree: ast.Module,
                               lineno: int) -> str:
        if cache_type == "InstructionCache":
            return "none"
        if cache_type == "DecompositionCache":
            return "caller_supplied_key_only"
        fn = enclosing_function(tree, lineno)
        if fn is not None:
            for node in ast.walk(fn):
                if isinstance(node, ast.keyword) and node.arg == "tenant":
                    return "explicit_put_argument"
        return "contextvar_in_sound_cache"

    # -- (b) put sites -----------------------------------------------------
    def collect_put_sites(self) -> None:
        for path in self.repo.files:
            tree = self.repo.trees[path]
            for call in ast.walk(tree):
                if not isinstance(call, ast.Call):
                    continue
                if not (isinstance(call.func, ast.Attribute) and call.func.attr == "put"):
                    continue
                cls = enclosing_class(tree, call.lineno)
                cls_ref = (self.repo.module_name(path), cls.name) if cls else None
                cache_type = self.repo.infer_receiver_type(path, tree, call, cls)
                if cache_type not in ALL_STORE_TYPES:
                    continue
                site = self._build_put_site(path, tree, call, cls, cls_ref, cache_type)
                if cache_type in CACHE_TYPES:
                    self.put_sites.append(site)
                else:
                    site["exclusion_reason"] = (
                        "put on DecompositionCache, not InstructionCache/SoundResultCache"
                    )
                    self.out_of_scope_put_sites.append(site)
        self.put_sites.sort(key=lambda r: (r["file"], r["line"]))
        self.out_of_scope_put_sites.sort(key=lambda r: (r["file"], r["line"]))

    def _build_put_site(self, path: Path, tree: ast.Module, call: ast.Call,
                        cls: Optional[ast.ClassDef],
                        cls_ref: Optional[Tuple[str, str]],
                        cache_type: str) -> Dict[str, Any]:
        fn = enclosing_function(tree, call.lineno)
        keyword_side_effects = None
        for kw in call.keywords:
            if kw.arg == "side_effects":
                keyword_side_effects = literal_bool(kw.value)
        class_decl = None
        if cls is not None:
            class_decl = self.repo.declaration_for_node(path, cls)

        resolution: Dict[str, Any]
        if keyword_side_effects is not None:
            declaration = {"kind": "literal", "value": keyword_side_effects,
                           "source": "put_keyword"}
            resolution = {"method": "put_keyword_literal",
                          "evidence": [{"file": self._rel(path),
                                        "line": call.lineno,
                                        "kind": "put_keyword",
                                        "side_effects": keyword_side_effects}]}
        elif cls_ref is not None:
            declaration = class_decl or {"kind": "undeclared",
                                         "legacy_default": True, "source": "none"}
            resolution = {"method": "enclosing_class",
                          "evidence": [{"file": self._rel(path),
                                        "line": call.lineno,
                                        "kind": "enclosing_class",
                                        "class": cls.name if cls else None}]}
        else:
            declaration, resolution = self._resolve_helper(path, tree, call, fn)

        bucket = self._bucket(declaration)
        footprint = self._footprint(path, tree, call, cls_ref, cls)
        tenant = self._site_tenant(path, tree, call, cls, fn, cache_type)
        owner_ref = cls_ref or self._receiver_owner_ref(path, tree, call, cls)
        class_is_instruction = (
            owner_ref is not None
            and self.repo.is_subclass_of(owner_ref, INSTRUCTION_REF)
        )
        flip = cache_type == "InstructionCache" and class_is_instruction
        admitted = bucket == "pure_declared"
        return {
            "file": self._rel(path),
            "line": call.lineno,
            "cache_type": cache_type,
            "already_typed": cache_type == "SoundResultCache",
            "receiver": expr_text(call.func.value),
            "enclosing_class": cls.name if cls else None,
            "enclosing_function": fn.name if fn else "<module>",
            "enclosing_function_chain": [f.name for f in function_chain(tree, call.lineno)],
            "class_ref": f"{cls_ref[0]}.{cls_ref[1]}" if cls_ref else None,
            "class_is_instruction_subclass": class_is_instruction,
            "flip_applies": flip,
            "bucket": bucket,
            "bucket_basis": declaration.get("source"),
            "bucket_basis_detail": (
                "explicit side_effects keyword on this put call"
                if declaration.get("source") == "put_keyword"
                else "enclosing class declaration"
                if declaration.get("source") in ("class_attribute",
                                                 "instance_attribute",
                                                 "super_init_keyword")
                else "inherited from " + str(declaration.get("inherited_from"))
                if declaration.get("source") == "inherited"
                else "caller factory declaration"
                if declaration.get("source") == "caller_declaration"
                else "no declaration (legacy default True)"
            ),
            "admitted_under_typed_default": admitted,
            "legacy_memoised": cache_type == "InstructionCache",
            "side_effects_declaration": declaration,
            "enclosing_class_declaration": class_decl,
            "resolution": resolution,
            "footprint": footprint,
            "tenant": tenant,
            "put_arguments": {
                "positional": len(call.args),
                "keywords": [kw.arg for kw in call.keywords],
            },
            "expression": expr_text(call),
        }

    def _receiver_owner_ref(self, path: Path, tree: ast.Module, call: ast.Call,
                            cls: Optional[ast.ClassDef]) -> Optional[Tuple[str, str]]:
        """Class that owns the receiver of ``<owner>.cache.put(...)``."""
        receiver = call.func.value if isinstance(call.func, ast.Attribute) else None
        if not isinstance(receiver, ast.Attribute) or receiver.attr != "cache":
            return None
        owner = receiver.value
        if isinstance(owner, ast.Name) and owner.id == "self" and cls is not None:
            return (self.repo.module_name(path), cls.name)
        if isinstance(owner, ast.Name):
            found = self.repo._param_class(path, tree, call, owner.id)
            if found is not None:
                return (self.repo.module_name(found[0]), found[1].name)
        return None

    def _resolve_helper(self, path: Path, tree: ast.Module, call: ast.Call,
                        fn: Optional[ast.AST]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Resolve a put in a module-level helper to its callers' declaration."""
        if fn is None:
            return ({"kind": "undeclared", "legacy_default": True,
                     "source": "no_enclosing_class_or_function"},
                    {"method": "unresolved", "evidence": []})
        values: List[bool] = []
        evidence: List[Dict[str, Any]] = []
        unresolved: List[Dict[str, Any]] = []
        seen_funcs: set = set()

        def visit(func: ast.AST, depth: int) -> None:
            if depth <= 0 or func.name in seen_funcs:
                return
            seen_funcs.add(func.name)
            same_file = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
            # (1) func passed as an argument to a factory call carrying
            #     side_effects=<literal>
            for candidate in same_file:
                for arg in candidate.args:
                    if isinstance(arg, ast.Name) and arg.id == func.name:
                        for kw in candidate.keywords:
                            if kw.arg == "side_effects":
                                literal = literal_bool(kw.value)
                                if literal is not None:
                                    values.append(literal)
                                    evidence.append({
                                        "file": self._rel(path),
                                        "line": candidate.lineno,
                                        "kind": "factory_keyword",
                                        "function": func.name,
                                        "side_effects": literal,
                                    })
                for kw in candidate.keywords:
                    if kw.arg == "side_effects" and isinstance(kw.value, ast.Name) \
                            and kw.value.id == func.name:
                        literal = literal_bool(kw.value)
                        if literal is not None:
                            values.append(literal)
                            evidence.append({
                                "file": self._rel(path), "line": candidate.lineno,
                                "kind": "factory_keyword", "function": func.name,
                                "side_effects": literal})
            # (2) func is called by another function; recurse upward.
            for candidate in same_file:
                if isinstance(candidate.func, ast.Name) and candidate.func.id == func.name:
                    outer = enclosing_function(tree, candidate.lineno)
                    if outer is None or outer is func:
                        continue
                    outer_cls = enclosing_class(tree, outer.lineno)
                    if outer_cls is not None:
                        ref = (self.repo.module_name(path), outer_cls.name)
                        decl = self.repo.side_effects_declaration(ref)
                        literal = decl.get("value")
                        if isinstance(literal, bool):
                            values.append(literal)
                            evidence.append({
                                "file": self._rel(path),
                                "line": outer.lineno,
                                "kind": "caller_class",
                                "class": outer_cls.name,
                                "side_effects": literal,
                            })
                        else:
                            unresolved.append({"file": self._rel(path),
                                               "line": outer.lineno,
                                               "reason": "caller class undeclared"})
                    else:
                        visit(outer, depth - 1)

        visit(fn, 4)
        unique = sorted(set(values))
        metadata = {"method": "caller_resolution" if values else "unresolved",
                    "evidence": evidence, "unresolved": unresolved}
        if len(unique) == 1:
            return ({"kind": "literal", "value": unique[0],
                     "source": "caller_declaration"},
                    metadata)
        if len(unique) > 1:
            metadata["conflict"] = True
            return ({"kind": "undeclared", "legacy_default": True,
                     "source": "conflicting_callers"}, metadata)
        return ({"kind": "undeclared", "legacy_default": True,
                 "source": "unresolved_helper"}, metadata)

    @staticmethod
    def _bucket(declaration: Dict[str, Any]) -> str:
        value = declaration.get("value")
        if declaration.get("kind") in ("literal", "conditional") and value is False:
            return "pure_declared"
        if declaration.get("kind") in ("literal", "conditional") and value is True:
            return "effectful_declared"
        return "undeclared"

    def _footprint(self, path: Path, tree: ast.Module, call: ast.Call,
                   cls_ref: Optional[Tuple[str, str]],
                   cls: Optional[ast.ClassDef]) -> Dict[str, Any]:
        keyword_footprint = any(kw.arg == "footprint" for kw in call.keywords)
        override = False
        override_ref = None
        if cls_ref is not None:
            override, override_ref = self._overrides_footprint(cls_ref)
        if keyword_footprint:
            uses_sound_footprint = any(
                kw.arg == "footprint"
                and "_sound_footprint" in (expr_text(kw.value) or "")
                for kw in call.keywords
            )
            if uses_sound_footprint:
                exactness = "exact_method" if override else "coarse_method_default"
            else:
                exactness = "exact_argument_and_method" if override else "exact_argument"
        elif override:
            exactness = "exact_method"
        elif cls_ref is not None:
            exactness = "coarse_method_default"
        else:
            exactness = "coarse_no_footprint_hook"
        return {
            "overridden": override,
            "override_definition": override_ref,
            "explicit_put_argument": keyword_footprint,
            "eviction_exactness": exactness,
        }

    def _overrides_footprint(self, ref: Tuple[str, str]) -> Tuple[bool, Optional[str]]:
        """Whether a class defines ``footprint`` other than the base default.

        ``Instruction.footprint`` (the empty frozenset) *is* the default the
        typed cache falls back to; inheriting it is not an override.
        """
        seen: set = set()

        def walk(current: Tuple[str, str]) -> Optional[str]:
            if current in seen:
                return None
            seen.add(current)
            found = self.repo.class_def(current)
            if found is None:
                return None
            path, cls = found
            for node in cls.body:
                if isinstance(node, ast.FunctionDef) and node.name == "footprint":
                    if current == INSTRUCTION_REF:
                        return None  # the default, not an override
                    return f"{self._rel(path)}:{node.lineno}"
            for base in self.repo.base_refs(path, cls):
                if base is not None:
                    hit = walk(base)
                    if hit is not None:
                        return hit
            return None

        hit = walk(ref)
        return (hit is not None, hit)

    def _site_tenant(self, path: Path, tree: ast.Module, call: ast.Call,
                     cls: Optional[ast.ClassDef], fn: Optional[ast.AST],
                     cache_type: str) -> Dict[str, Any]:
        explicit = any(kw.arg == "tenant" for kw in call.keywords)
        lexical = False
        if fn is not None:
            for node in ast.walk(fn):
                if isinstance(node, ast.Name) and "tenant" in node.id:
                    lexical = True
                if isinstance(node, ast.Attribute) and "tenant" in node.attr:
                    lexical = True
                if isinstance(node, ast.arg) and "tenant" in node.arg:
                    lexical = True
        if cls is not None:
            for node in ast.walk(cls):
                if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                        and node.value.id == "self" and "tenant" in node.attr):
                    lexical = True
        if explicit:
            channel = "explicit_put_keyword"
        elif lexical:
            channel = "lexical_tenant_value"
        elif "current_tenant" in (expr_text(call) or "") or \
                "tenant_scope" in (expr_text(call) or ""):
            channel = "contextvar_call"
        elif cache_type == "SoundResultCache":
            # SoundResultCache.get/put resolve tenant= or current_tenant, so a
            # ContextVar can supply identity even when this call does not.
            channel = "contextvar_in_sound_cache"
        else:
            channel = "none"
        return {
            "explicit_put_keyword": explicit,
            "tenant_value_in_lexical_scope": lexical,
            "channel": channel,
            "identity_gate_applies": cache_type == "SoundResultCache",
            "identity_dependent_key_requires_tenant": cache_type == "SoundResultCache",
        }

    # -- (c) execute definitions ------------------------------------------
    def collect_execute_defs(self) -> None:
        for path in self.repo.files:
            tree = self.repo.trees[path]
            for node in ast.walk(tree):
                if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and node.name == "execute"):
                    continue
                cls = enclosing_class(tree, node.lineno)
                if cls is None:
                    continue
                ref = (self.repo.module_name(path), cls.name)
                cache_types = self.repo.cache_types_for_class(ref)
                if not cache_types:
                    continue
                direct_attrs = [
                    {"attribute": entry["attr"],
                     "cache_type": entry["cache_type"],
                     "line": entry["lineno"]}
                    for entry in self.repo._cache_assignments(path, cls)
                ]
                record = {
                    "file": self._rel(path),
                    "line": node.lineno,
                    "class": cls.name,
                    "class_ref": f"{ref[0]}.{ref[1]}",
                    "signature": self._signature(node),
                    "cache_types": sorted(cache_types),
                    "direct_cache_attributes": direct_attrs,
                    "stores_constructed_in_class": self.repo.constructed_store_types(path, cls),
                    "in_scope_cache": bool(cache_types & set(CACHE_TYPES)),
                    "is_instruction_subclass": self.repo.is_subclass_of(ref, INSTRUCTION_REF),
                    "tenant_reach": self._tenant_reach(node, cls, path),
                }
                self.execute_defs.append(record)
        self.execute_defs.sort(key=lambda r: (r["file"], r["line"]))

    @staticmethod
    def _signature(node: ast.AST) -> str:
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        parts = [a.arg for a in node.args.posonlyargs + node.args.args]
        parts += [f"*{a.arg}" for a in node.args.kwonlyargs if a.arg != "self"]
        return f"execute({', '.join(parts)})"

    def _tenant_reach(self, fn: ast.AST, cls: ast.ClassDef, path: Path) -> Dict[str, Any]:
        arguments = sorted({a for a in arg_names(fn) if "tenant" in a})
        class_attrs: List[str] = []
        for node in cls.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and "tenant" in target.id:
                        class_attrs.append(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                    and "tenant" in node.target.id:
                class_attrs.append(node.target.id)
        for node in ast.walk(cls):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == "self" and "tenant" in node.attr):
                class_attrs.append(f"self.{node.attr}")
        contextvar = False
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and node.id in ("current_tenant", "cache_tenant"):
                contextvar = True
            if (isinstance(node, ast.Attribute)
                    and node.attr in ("current_tenant", "cache_tenant")):
                contextvar = True
            if (isinstance(node, ast.Name) and node.id == "tenant_scope"):
                contextvar = True
        if arguments:
            reach = "argument"
        elif contextvar:
            reach = "contextvar"
        elif class_attrs:
            reach = "class_attribute"
        else:
            reach = "nothing"
        return {
            "reach": reach,
            "arguments": arguments,
            "class_attributes": sorted(set(class_attrs)),
            "contextvar_use": contextvar,
        }

    # -- utilities ---------------------------------------------------------
    def _rel(self, path: Path) -> str:
        return path.relative_to(self.repo.root).as_posix()

    # -- report ------------------------------------------------------------
    def report(self) -> Dict[str, Any]:
        self.collect_constructors()
        self.collect_put_sites()
        self.collect_execute_defs()

        bucket_counts = Counter(site["bucket"] for site in self.put_sites)
        legacy_sites = [s for s in self.put_sites if s["cache_type"] == "InstructionCache"]
        typed_sites = [s for s in self.put_sites if s["cache_type"] == "SoundResultCache"]
        flip_sites = [s for s in self.put_sites if s["flip_applies"]]
        legacy_buckets = Counter(s["bucket"] for s in legacy_sites)
        typed_buckets = Counter(s["bucket"] for s in typed_sites)
        flip_buckets = Counter(s["bucket"] for s in flip_sites)

        pure = [s for s in self.put_sites if s["bucket"] == "pure_declared"]
        effectful = [s for s in self.put_sites if s["bucket"] == "effectful_declared"]
        undeclared = [s for s in self.put_sites if s["bucket"] == "undeclared"]
        bug_sites = [s for s in effectful if s["legacy_memoised"]]

        exact = [s for s in self.put_sites if s["footprint"]["explicit_put_argument"]
                 and s["footprint"]["eviction_exactness"] == "exact_argument"]
        admitted_needing_footprint = [
            s for s in pure
            if s["footprint"]["eviction_exactness"].startswith("coarse")
        ]
        legacy_after_declarations = [
            s for s in legacy_sites if s["bucket"] in ("pure_declared", "undeclared")
        ]

        tenant_reach = Counter(r["tenant_reach"]["reach"] for r in self.execute_defs)
        all_tenant_reach = self._all_execute_tenant_counts()
        execute_by_root = Counter(r["file"].split("/", 1)[0] for r in self.execute_defs)
        all_execute = self._all_execute_counts()

        all_put_calls = self._count_all_put_calls()
        all_constructors = self._count_all_store_constructors()
        dict_cache_inventory = self._dict_cache_inventory()
        counts = OrderedDict([
            ("python_files_scanned", len(self.repo.files)),
            ("put_calls_all_receivers", all_put_calls["total"]),
            ("put_calls_unattributed", len(all_put_calls["unattributed"])),
            ("store_constructor_calls_all", all_constructors["total"]),
            ("execute_defs_in_dict_cache_classes",
             dict_cache_inventory["execute_def_count"]),
            ("constructors_in_scope", len(self.constructors)),
            ("constructors_out_of_scope", len(self.out_of_scope_constructors)),
            ("put_sites_in_scope", len(self.put_sites)),
            ("put_sites_out_of_scope", len(self.out_of_scope_put_sites)),
            ("put_sites_on_legacy_instruction_cache", len(legacy_sites)),
            ("put_sites_on_sound_result_cache", len(typed_sites)),
            ("put_sites_governed_by_instruction_default_flip", len(flip_sites)),
            ("execute_defs_total", all_execute["repo_total"]),
            ("execute_defs_agentflow", all_execute["agentflow_total"]),
            ("execute_defs_agentflow_core", all_execute["agentflow_core"]),
            ("execute_defs_agentflow_agents", all_execute["agentflow_agents"]),
            ("execute_defs_agentflow_services", all_execute["agentflow_services"]),
            ("execute_defs_examples", all_execute["examples_total"]),
            ("execute_defs_scripts", all_execute["scripts_total"]),
            ("execute_defs_with_input_params_signature",
             all_execute["input_params_signature"]),
            ("execute_defs_in_caching_classes", len(self.execute_defs)),
            ("execute_defs_in_caching_classes_in_scope",
             sum(1 for r in self.execute_defs if r["in_scope_cache"])),
            ("execute_defs_in_caching_classes_other_store",
             sum(1 for r in self.execute_defs if not r["in_scope_cache"])),
        ])

        summary = OrderedDict([
            ("buckets", OrderedDict((b, bucket_counts.get(b, 0)) for b in BUCKETS)),
            ("admitted_under_typed_default", len(pure)),
            ("refused_under_typed_default", len(self.put_sites) - len(pure)),
            ("legacy_store_buckets",
             OrderedDict((b, legacy_buckets.get(b, 0)) for b in BUCKETS)),
            ("sound_store_buckets",
             OrderedDict((b, typed_buckets.get(b, 0)) for b in BUCKETS)),
            ("flip_cohort_buckets",
             OrderedDict((b, flip_buckets.get(b, 0)) for b in BUCKETS)),
            ("flip_impact", OrderedDict([
                ("legacy_sites", len(legacy_sites)),
                ("sites_governed_by_instruction_flip", len(flip_sites)),
                ("flip_cohort_buckets", OrderedDict([
                    ("survive_unchanged", flip_buckets.get("pure_declared", 0)),
                    ("need_side_effects_false", flip_buckets.get("undeclared", 0)),
                    ("effectful_memoised_bug", flip_buckets.get("effectful_declared", 0)),
                ])),
                ("all_legacy_buckets", OrderedDict([
                    ("survive_unchanged", legacy_buckets.get("pure_declared", 0)),
                    ("need_side_effects_false", legacy_buckets.get("undeclared", 0)),
                    ("effectful_memoised_bug", legacy_buckets.get("effectful_declared", 0)),
                ])),
                ("effectful_bug_sites",
                 [f"{s['file']}:{s['line']}" for s in bug_sites
                  if s["flip_applies"]]),
                ("legacy_sites_outside_instruction_flip",
                 [f"{s['file']}:{s['line']}" for s in legacy_sites
                  if not s["flip_applies"]]),
                ("undeclared_sites_all_legacy",
                 [f"{s['file']}:{s['line']}" for s in legacy_sites
                  if s["bucket"] == "undeclared"]),
            ])),
            ("footprint", OrderedDict([
                ("overrides_found", sum(1 for s in self.put_sites
                                        if s["footprint"]["overridden"])),
                ("sites_with_exact_put_argument", len(exact)),
                ("sites_with_exact_put_argument_list",
                 [f"{s['file']}:{s['line']}" for s in exact]),
                ("admitted_sites_needing_footprint", len(admitted_needing_footprint)),
                ("admitted_sites_needing_footprint_list",
                 [f"{s['file']}:{s['line']}" for s in admitted_needing_footprint]),
                ("legacy_sites_needing_footprint_after_declaration_fix",
                 len(legacy_after_declarations)),
            ])),
            ("tenant", OrderedDict([
                ("constructors_with_tenant_at_construction",
                 sum(1 for c in self.constructors
                     if c["tenant_available_at_construction"])),
                ("put_sites_with_tenant_in_lexical_scope",
                 sum(1 for s in self.put_sites
                     if s["tenant"]["tenant_value_in_lexical_scope"])),
                ("put_sites_with_explicit_tenant_keyword",
                 sum(1 for s in self.put_sites
                     if s["tenant"]["explicit_put_keyword"])),
                ("admitted_sites_without_tenant_channel",
                 sum(1 for s in pure if s["tenant"]["channel"] == "none")),
                ("admitted_sites_without_tenant_channel_list",
                 [f"{s['file']}:{s['line']}" for s in pure
                  if s["tenant"]["channel"] == "none"]),
                ("execute_defs_tenant_reach",
                 OrderedDict((k, tenant_reach.get(k, 0))
                             for k in ("argument", "class_attribute",
                                       "contextvar", "nothing"))),
                ("all_execute_defs_tenant_reach",
                 OrderedDict((k, all_tenant_reach["counts"].get(k, 0))
                             for k in ("argument", "class_attribute",
                                       "contextvar", "nothing"))),
                ("all_execute_defs_with_tenant_channel",
                 sum(all_tenant_reach["counts"].get(k, 0)
                     for k in ("argument", "class_attribute", "contextvar"))),
            ])),
            ("execute_defs_by_root",
             OrderedDict((k, execute_by_root.get(k, 0)) for k in
                         ("agentflow", "examples", "scripts"))),
        ])

        checks = self._checks(counts, summary, bucket_counts, legacy_sites, flip_sites)

        return OrderedDict([
            ("schema_version", SCHEMA_VERSION),
            ("generated_by", "examples/benchmarks/cache_default_census.py"),
            ("scope", OrderedDict([
                ("roots", list(ROOTS)),
                ("excluded_dirs", sorted(SKIP_DIRS)),
                ("in_scope_cache_types", list(CACHE_TYPES)),
                ("other_store_types_recorded_separately", list(OTHER_STORE_TYPES)),
                ("classification_rule",
                 "bucket is the effective side_effects declaration governing the "
                 "put: literal put keyword > enclosing class declaration > "
                 "caller-resolved declaration for module-level helpers. Only "
                 "pure_declared (side_effects is False) is admitted by the typed "
                 "default; True and undeclared are refused (undeclared fails "
                 "closed)."),
            ])),
            ("method", OrderedDict([
                ("parser", "python ast (ast.parse + ast.walk); no grep"),
                ("file_discovery", "os.walk over the three roots, sorted"),
                ("constructors", "Call whose callee resolves to a store class name"),
                ("put_sites",
                 "Call with Attribute(attr='put') whose receiver type is inferred "
                 "from (i) a constructor call, (ii) self.<attr> = <store>(...), "
                 "(iii) an annotated parameter whose class assigns .cache, "
                 "(iv) a local assignment whose RHS is a store constructor or a "
                 "method returning one, or (v) method-usage inference for "
                 "unannotated parameters"),
                ("side_effects",
                 "class-body attribute, self.side_effects assignment, or "
                 "super().__init__(side_effects=...) in the class's own __init__; "
                 "an explicit declaration on a local ancestor is inherited; a bare "
                 "constructor default is recorded as undeclared (legacy default "
                 "True)"),
                ("helper_puts",
                 "a put inside a module-level helper is resolved to the factory "
                 "calls / callers that pass side_effects=<literal>; unresolved or "
                 "conflicting callers stay undeclared"),
                ("tenant_reach",
                 "execute signature argument containing 'tenant', a class-level "
                 "tenant attribute, a current_tenant/tenant_scope reference, else "
                 "nothing; precedence argument > contextvar > class_attribute"),
                ("determinism",
                 "sorted files and sites, no timestamps or wall-clock values"),
            ])),
            ("counts", counts),
            ("buckets", OrderedDict(
                (bucket, OrderedDict([
                    ("count", bucket_counts.get(bucket, 0)),
                    ("admitted_by_typed_default", bucket == "pure_declared"),
                    ("meaning", {
                        "pure_declared":
                            "side_effects is False; the typed default admits",
                        "effectful_declared":
                            "side_effects is True; the typed default refuses "
                            "(legacy put stored it unconditionally)",
                        "undeclared":
                            "no declaration; the legacy default is True, so the "
                            "typed default refuses (fail closed)",
                    }[bucket]),
                    ("sites", [
                        {"file": site["file"], "line": site["line"],
                         "cache_type": site["cache_type"],
                         "flip_applies": site["flip_applies"],
                         "legacy_memoised": site["legacy_memoised"]}
                        for site in self.put_sites if site["bucket"] == bucket
                    ]),
                ]))
                for bucket in BUCKETS
            )),
            ("summary", summary),
            ("constructors", self.constructors),
            ("put_sites", self.put_sites),
            ("execute_defs", self.execute_defs),
            ("out_of_scope", OrderedDict([
                ("constructors", self.out_of_scope_constructors),
                ("put_sites", self.out_of_scope_put_sites),
                ("dict_based_caches", dict_cache_inventory["classes"]),
                ("dict_based_caches_note",
                 "plain-dict stores are not instances of InstructionCache or "
                 "SoundResultCache and are outside this census; their execute "
                 f"definitions are counted separately "
                 f"({dict_cache_inventory['execute_def_count']} definitions)"),
                ("dict_based_cache_execute_defs",
                 dict_cache_inventory["execute_defs"]),
            ])),
            ("notes", [
                "Reconciliation with gap report section 1: it states 96 "
                "execute(input_params) implementations in agentflow/. This AST "
                f"pass finds {counts['execute_defs_agentflow_core']} execute "
                "definitions under agentflow/core, "
                f"{counts['execute_defs_agentflow']} under agentflow/ and "
                f"{counts['execute_defs_total']} repo-wide; "
                f"{counts['execute_defs_with_input_params_signature']} have the "
                "(self, input_params) signature.",
                "The (c) cohort is every execute definition whose class "
                "transitively holds a result store: "
                f"{counts['execute_defs_in_caching_classes']} definitions "
                f"({counts['execute_defs_in_caching_classes_in_scope']} with an "
                "InstructionCache/SoundResultCache and "
                f"{counts['execute_defs_in_caching_classes_other_store']} with "
                "ObjectiveDecompositionInstruction's DecompositionCache).",
                "Every .put call in the three roots is attributed: "
                f"{counts['put_calls_all_receivers']} total = "
                f"{counts['put_sites_in_scope']} on the two in-scope stores + "
                f"{counts['put_sites_out_of_scope']} on DecompositionCache.",
                "A put inside a module-level helper is resolved through the "
                "factory calls that receive the helper; each such site carries "
                "the resolving file:line in resolution.evidence.",
                "All numbers are recomputed from the tree on every run; no value "
                "is copied from a report or artifact. The file contains no "
                "timestamps, absolute paths or wall-clock values.",
            ]),
            ("caveats", [
                "admitted_under_typed_default applies the effect gate exactly as "
                "specified (only side_effects is False admits). SoundResultCache "
                "also refuses an identity-dependent put with no tenant and a put "
                "with footprint=None; the tenant and footprint blocks report those "
                "readiness gaps separately instead of folding them into the "
                "buckets.",
                f"{summary['tenant']['admitted_sites_without_tenant_channel']} of "
                "the admitted sites have no tenant channel in lexical scope. They "
                "pass the effect gate but would need tenant_scope(...) or "
                "identity_dependent=False to be admitted by the typed store.",
                "Effectful_declared contains one intentional already-typed probe "
                "(shipped_sound_cache_replay.py:411) that the class refuses; only "
                "the flip-cohort effectful sites are legacy memoisation bugs.",
                "CompositeInstruction's effectful declaration is conditional "
                "(True when the caller passes no side_effects argument, "
                "overridable per instance).  Its unconditional legacy put at "
                "instruction.py:537 is the memoisation bug only under that "
                "default; a caller that passes side_effects=False is pure.",
                "benchmark-local policy helpers (cache_identity.remember / "
                "invalidate) operate on caller-owned dictionaries and are not "
                "store constructors; they are not part of this census.",
                "The pass is static: it counts lexical construction/put sites and "
                "cannot see stores created dynamically (getattr, exec, factories "
                "that return a cache object).",
            ]),
            ("checks", checks),
        ])

    def _all_execute_tenant_counts(self) -> Dict[str, Any]:
        counts: Counter = Counter()
        for path in self.repo.files:
            tree = self.repo.trees[path]
            for node in ast.walk(tree):
                if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and node.name == "execute"):
                    continue
                cls = enclosing_class(tree, node.lineno)
                if cls is None:
                    counts["no_class"] += 1
                    continue
                reach = self._tenant_reach(node, cls, path)["reach"]
                counts[reach] += 1
        return {"counts": counts, "total": sum(counts.values())}

    def _dict_cache_inventory(self) -> Dict[str, Any]:
        classes: List[Dict[str, Any]] = []
        refs = set()
        for ref in sorted(self.repo.classes):
            if ref[1].endswith("Cache"):
                continue  # the store implementation itself, not a user class
            found = self.repo.class_def(ref)
            if found is None:
                continue
            path, cls = found
            assignments = self.repo.dict_cache_assignments(path, cls)
            if not assignments:
                continue
            refs.add(ref)
            classes.append({"file": self._rel(path), "class": cls.name,
                            "assignments": assignments})

        def has_dict_cache(ref: Tuple[str, str], seen: Optional[set] = None) -> bool:
            if ref in refs:
                return True
            seen = set() if seen is None else seen
            if ref in seen:
                return False
            seen.add(ref)
            found = self.repo.class_def(ref)
            if found is None:
                return False
            path, cls = found
            for base in self.repo.base_refs(path, cls):
                if base is not None and has_dict_cache(base, seen):
                    return True
            return False

        defs: List[str] = []
        for path in self.repo.files:
            for node in ast.walk(self.repo.trees[path]):
                if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and node.name == "execute"):
                    continue
                cls = enclosing_class(self.repo.trees[path], node.lineno)
                if cls is None:
                    continue
                ref = (self.repo.module_name(path), cls.name)
                if has_dict_cache(ref):
                    defs.append(f"{self._rel(path)}:{node.lineno} {cls.name}.execute")
        return {"classes": classes, "execute_defs": defs,
                "execute_def_count": len(defs)}

    def _count_all_put_calls(self) -> Dict[str, Any]:
        total = 0
        unattributed: List[str] = []
        for path in self.repo.files:
            tree = self.repo.trees[path]
            for call in ast.walk(tree):
                if not (isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and call.func.attr == "put"):
                    continue
                total += 1
                cls = enclosing_class(tree, call.lineno)
                cache_type = self.repo.infer_receiver_type(path, tree, call, cls)
                if cache_type not in ALL_STORE_TYPES:
                    unattributed.append(f"{self._rel(path)}:{call.lineno}")
        return {"total": total, "unattributed": unattributed}

    def _count_all_store_constructors(self) -> Dict[str, int]:
        total = 0
        for path in self.repo.files:
            for call in ast.walk(self.repo.trees[path]):
                if isinstance(call, ast.Call) and call_name(call.func) in ALL_STORE_TYPES:
                    total += 1
        return {"total": total}

    def _all_execute_counts(self) -> Dict[str, int]:
        counts = Counter()
        signature = 0
        for path in self.repo.files:
            rel = self._rel(path)
            root = rel.split("/", 1)[0]
            counts[f"{root}_total"] = counts.get(f"{root}_total", 0)
            for node in ast.walk(self.repo.trees[path]):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and node.name == "execute":
                    counts[f"{root}_total"] += 1
                    if root == "agentflow":
                        parts = rel.split("/")
                        if len(parts) > 1:
                            counts[f"agentflow_{parts[1]}"] += 1
                    if [a.arg for a in node.args.posonlyargs + node.args.args] == \
                            ["self", "input_params"]:
                        signature += 1
        return {
            "repo_total": sum(counts[f"{r}_total"] for r in ROOTS),
            "agentflow_total": counts["agentflow_total"],
            "agentflow_core": counts["agentflow_core"],
            "agentflow_agents": counts["agentflow_agents"],
            "agentflow_services": counts["agentflow_services"],
            "examples_total": counts["examples_total"],
            "scripts_total": counts["scripts_total"],
            "input_params_signature": signature,
        }

    def _checks(self, counts, summary, bucket_counts,
                legacy_sites, flip_sites) -> List[Dict[str, Any]]:
        checks: List[Dict[str, Any]] = []

        def add(name: str, ok: bool, detail: str) -> None:
            checks.append({"name": name, "ok": bool(ok), "detail": detail})

        total = counts["put_sites_in_scope"]
        add("buckets_sum_to_put_sites",
            sum(bucket_counts.get(b, 0) for b in BUCKETS) == total,
            f"{dict((b, bucket_counts.get(b, 0)) for b in BUCKETS)} vs {total}")
        add("legacy_buckets_sum_to_legacy_sites",
            sum(summary["legacy_store_buckets"].values()) == len(legacy_sites),
            f"{summary['legacy_store_buckets']} vs {len(legacy_sites)}")
        add("flip_buckets_sum_to_flip_sites",
            sum(summary["flip_cohort_buckets"].values()) == len(flip_sites),
            f"{summary['flip_cohort_buckets']} vs {len(flip_sites)}")
        add("admitted_equals_pure_declared",
            summary["admitted_under_typed_default"] == bucket_counts.get("pure_declared", 0),
            f"{summary['admitted_under_typed_default']} vs "
            f"{bucket_counts.get('pure_declared', 0)}")
        add("refused_equals_non_pure",
            summary["refused_under_typed_default"] ==
            total - bucket_counts.get("pure_declared", 0),
            f"{summary['refused_under_typed_default']} vs "
            f"{total - bucket_counts.get('pure_declared', 0)}")
        add("flip_cohort_sums_to_flip_sites",
            sum(summary["flip_impact"]["flip_cohort_buckets"].values())
            == len(flip_sites),
            f"{summary['flip_impact']['flip_cohort_buckets']} vs {len(flip_sites)}")
        add("all_legacy_sums_to_legacy_sites",
            sum(summary["flip_impact"]["all_legacy_buckets"].values())
            == len(legacy_sites),
            f"{summary['flip_impact']['all_legacy_buckets']} vs {len(legacy_sites)}")
        add("flip_sites_are_legacy_instructioncache",
            all(s["cache_type"] == "InstructionCache" and
                s["class_is_instruction_subclass"] for s in flip_sites),
            "every flip site is a legacy store in an Instruction subclass")
        add("every_declared_bucket_has_matching_value",
            all((s["bucket"] == "pure_declared") ==
                (s["side_effects_declaration"].get("value") is False)
                for s in self.put_sites
                if s["side_effects_declaration"].get("kind") in ("literal", "conditional")),
            "literal/conditional declarations agree with their bucket")
        add("undeclared_sites_have_no_boolean_value",
            all(s["side_effects_declaration"].get("value") is None
                for s in self.put_sites if s["bucket"] == "undeclared"),
            "all undeclared sites report value=None")
        add("effectful_bug_list_matches_legacy_memoised",
            all(s["legacy_memoised"] for s in self.put_sites
                if s["bucket"] == "effectful_declared" and s["flip_applies"]),
            "every effectful site governed by the legacy store is memoised")
        add("put_site_keys_unique",
            len({(s["file"], s["line"], s["cache_type"]) for s in self.put_sites})
            == total,
            "no duplicate (file, line, cache_type)")
        add("constructor_types_known",
            all(c["cache_type"] in ALL_STORE_TYPES for c in
                self.constructors + self.out_of_scope_constructors),
            "every constructor resolves to a known store class")
        add("execute_defs_breakdown_sums",
            counts["execute_defs_agentflow"] + counts["execute_defs_examples"]
            + counts["execute_defs_scripts"] == counts["execute_defs_total"],
            "agentflow + examples + scripts == repo total")
        add("execute_defs_agentflow_breakdown_sums",
            counts["execute_defs_agentflow_core"]
            + counts["execute_defs_agentflow_agents"]
            + counts["execute_defs_agentflow_services"]
            == counts["execute_defs_agentflow"],
            "core + agents + services == agentflow total")
        add("caching_execute_defs_subset_of_total",
            counts["execute_defs_in_caching_classes"] <= counts["execute_defs_total"],
            f"{counts['execute_defs_in_caching_classes']} <= "
            f"{counts['execute_defs_total']}")
        add("caching_execute_defs_split_sums",
            counts["execute_defs_in_caching_classes_in_scope"]
            + counts["execute_defs_in_caching_classes_other_store"]
            == counts["execute_defs_in_caching_classes"],
            "in-scope + other-store == caching total")
        add("tenant_reach_counts_sum",
            sum(summary["tenant"]["execute_defs_tenant_reach"].values())
            == counts["execute_defs_in_caching_classes"],
            f"{summary['tenant']['execute_defs_tenant_reach']} vs "
            f"{counts['execute_defs_in_caching_classes']}")
        add("footprint_need_is_subset_of_admitted",
            summary["footprint"]["admitted_sites_needing_footprint"]
            <= summary["admitted_under_typed_default"],
            "sites needing a footprint cannot exceed admitted sites")
        add("no_footprint_overrides_implies_coarse_default",
            summary["footprint"]["overrides_found"] == 0 or
            any(s["footprint"]["overridden"] for s in self.put_sites),
            "footprint override count agrees with per-site flags")
        add("all_put_calls_attributed",
            counts["put_calls_all_receivers"]
            == counts["put_sites_in_scope"] + counts["put_sites_out_of_scope"],
            f"{counts['put_calls_all_receivers']} .put calls == "
            f"{counts['put_sites_in_scope']} in scope + "
            f"{counts['put_sites_out_of_scope']} out of scope")
        add("no_unattributed_put_calls",
            counts["put_calls_unattributed"] == 0,
            f"{counts['put_calls_unattributed']} .put calls with no inferred receiver type")
        add("all_store_constructors_attributed",
            counts["store_constructor_calls_all"]
            == counts["constructors_in_scope"] + counts["constructors_out_of_scope"],
            f"{counts['store_constructor_calls_all']} constructor calls == "
            f"{counts['constructors_in_scope']} in scope + "
            f"{counts['constructors_out_of_scope']} out of scope")
        add("all_execute_tenant_counts_sum",
            sum(summary["tenant"]["all_execute_defs_tenant_reach"].values())
            == counts["execute_defs_total"],
            f"{summary['tenant']['all_execute_defs_tenant_reach']} vs "
            f"{counts['execute_defs_total']}")
        add("counts_nonnegative", all(v >= 0 for v in counts.values()),
            "all counts are non-negative")
        return checks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None,
                        help="output path (default: cache_default_census.json beside this file)")
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if any invariant fails")
    parser.add_argument("--quiet", action="store_true", help="do not print the summary")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    census = Census(root)
    report = census.report()
    out = args.json or (Path(__file__).resolve().parent / "cache_default_census.json")
    out.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    failed = [c for c in report["checks"] if not c["ok"]]
    if not args.quiet:
        summary = report["summary"]
        counts = report["counts"]
        print(f"wrote {out}")
        print(f"put sites (in scope): {counts['put_sites_in_scope']}  "
              f"buckets: {dict(summary['buckets'])}")
        print(f"legacy InstructionCache sites: {counts['put_sites_on_legacy_instruction_cache']}  "
              f"flip cohort: {counts['put_sites_governed_by_instruction_default_flip']}")
        print(f"flip impact: {dict(summary['flip_impact'])}")
        print(f"footprint: {dict(summary['footprint'])}")
        print(f"execute defs in caching classes: "
              f"{counts['execute_defs_in_caching_classes']}  "
              f"tenant reach: {dict(summary['tenant']['execute_defs_tenant_reach'])}")
        print(f"checks: {len(report['checks']) - len(failed)}/{len(report['checks'])} ok")
        if failed:
            for check in failed:
                print(f"FAILED: {check['name']}: {check['detail']}", file=sys.stderr)
    return 1 if (args.check and failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
