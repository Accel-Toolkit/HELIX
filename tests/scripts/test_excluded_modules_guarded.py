"""A test that needs a module the public release leaves out skips without it.

The public tree is this tree minus the paths ``scripts/cut_public_release.sh``
removes in its exclusion step.  A test that imports one of the removed modules
with no ``pytest.importorskip`` first fails there with ModuleNotFoundError, on
every OS the public CI runs, while the dev suite (which has the modules) passes.

Rule: every function, class body or module body under ``tests/`` that imports
one of the ``linac_gen`` modules the exclusion step removes (an ``import``, a
relative import, ``importlib.import_module`` or ``__import__``) calls
``pytest.importorskip("<that module>")`` earlier in the same body.  Code that
ships (``linac_gen/``, ``gui/``) never imports them.  The release script ships
with the public tree, so this test runs in both.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RELEASE_SCRIPT = REPO / "scripts" / "cut_public_release.sh"
_PY_PATH = re.compile(r"linac_gen/[\w/]+\.py")


def _excluded() -> tuple[set[str], set[str]]:
    """(modules, test files) that the release script's exclusion step removes.  A removal
    under ``linac_gen/`` that is not a plain file path (a glob, a brace list, a directory)
    fails here rather than slipping past the rule."""
    text = RELEASE_SCRIPT.read_text(encoding="utf-8")
    block = text.split("2/5 exclusions", 1)[1].split("3/5 heal the manual", 1)[0]
    tokens = re.findall(r"\S+", block)
    odd = [t for t in tokens if t.startswith("linac_gen/") and not _PY_PATH.fullmatch(t)]
    assert not odd, f"exclusion entries under linac_gen/ this test cannot read: {odd}"
    mods = {t[:-3].replace("/", ".") for t in tokens if _PY_PATH.fullmatch(t)}
    tests = {m.group(0) for m in re.finditer(r"\btests/[\w/]+\.py\b", block)}
    return mods, tests


def _package(rel: str) -> str:
    """The package a source file belongs to, for its relative imports."""
    parts = rel[:-3].split("/")
    return ".".join(parts[:-1])


def _imported(node: ast.AST, package: str) -> list[str]:
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    if isinstance(node, ast.ImportFrom):
        if node.level:
            base = package.split(".")
            base = base[:len(base) - (node.level - 1)] if node.level > 1 else base
            mod = ".".join(base + ([node.module] if node.module else []))
        else:
            mod = node.module or ""
        # "from linac_gen.analysis import mode_growth" imports the submodule too
        return [mod] + [f"{mod}.{a.name}" for a in node.names]
    if (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)):
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
        if name in ("import_module", "__import__"):
            return [node.args[0].value]
    return []


def _skip_target(node: ast.AST) -> str | None:
    """The module of a ``pytest.importorskip("module", ...)`` call, else None."""
    if not isinstance(node, ast.Call):
        return None
    f = node.func
    if not ((isinstance(f, ast.Attribute) and f.attr == "importorskip")
            or (isinstance(f, ast.Name) and f.id == "importorskip")):
        return None
    arg = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg == "modname"), None)
    return arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else None


def _scopes(tree: ast.Module):
    yield tree
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield n


def _own_nodes(scope):
    """The nodes of this scope, not those of the functions and classes inside it."""
    stack = list(scope.body)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield n
        stack.extend(ast.iter_child_nodes(n))


def _hits(names, mods):
    return {m for name in names for m in mods if name == m or name.startswith(m + ".")}


def _unguarded(tree: ast.Module, rel: str, mods) -> list[str]:
    bad = []
    package = _package(rel)
    for scope in _scopes(tree):
        nodes = list(_own_nodes(scope))
        first_skip = {}
        for n in nodes:
            t = _skip_target(n)
            if t is not None:
                first_skip[t] = min(first_skip.get(t, n.lineno), n.lineno)
        for n in nodes:
            for hit in sorted(_hits(_imported(n, package), mods)):
                if not (hit in first_skip and first_skip[hit] < n.lineno):
                    bad.append(f"{rel}:{n.lineno} {getattr(scope, 'name', '<module>')} imports {hit}")
    return bad


def _any_import(tree: ast.Module, rel: str, mods) -> list[str]:
    package = _package(rel)
    return [f"{rel}:{n.lineno} imports {hit}" for n in ast.walk(tree)
            for hit in sorted(_hits(_imported(n, package), mods))]


def test_the_exclusion_step_is_read():
    """The rule below is not vacuous: the exclusion step names modules and tests."""
    mods, tests = _excluded()
    assert mods, "no linac_gen module found in the exclusion step"
    assert all(m.startswith("linac_gen.") for m in mods)
    assert tests


def test_the_rule_flags_what_it_is_for():
    """Canaries: each way of importing a removed module is seen, and each guard is honoured."""
    mods = {"linac_gen.analysis.gone"}

    def flagged(src, rel="tests/x/test_c.py"):
        return len(_unguarded(ast.parse(src), rel, mods))

    assert flagged("def t():\n    from linac_gen.analysis import gone\n") == 1
    assert flagged("def t():\n    import importlib\n    importlib.import_module('linac_gen.analysis.gone')\n") == 1
    assert flagged("class T:\n    from linac_gen.analysis.gone import f\n") == 1
    assert flagged("def t():\n    import linac_gen.analysis.gone as g\n    pytest.importorskip("
                   "'linac_gen.analysis.gone')\n") == 1                     # guard after the import
    assert flagged("def t():\n    pytest.importorskip('linac_gen.analysis.gone')\n"
                   "    from linac_gen.analysis.gone import f\n") == 0
    assert flagged("def t():\n    pytest.importorskip(modname='linac_gen.analysis.gone')\n"
                   "    import linac_gen.analysis.gone\n") == 0
    shipped = ast.parse("from .gone import f\nfrom ..analysis import gone\n")
    assert len(_any_import(shipped, "linac_gen/analysis/user.py", mods)) == 2


def test_tests_skip_without_the_modules_the_release_leaves_out():
    mods, removed_tests = _excluded()
    bad = []
    for p in sorted((REPO / "tests").rglob("*.py")):
        rel = p.relative_to(REPO).as_posix()
        if rel in removed_tests:
            continue
        bad += _unguarded(ast.parse(p.read_text(encoding="utf-8"), filename=rel), rel, mods)
    assert not bad, ("imports of modules the public release removes, with no "
                     "pytest.importorskip before them:\n" + "\n".join(bad))


def test_shipped_code_never_imports_them():
    mods, _ = _excluded()
    removed = {m.replace(".", "/") + ".py" for m in mods}
    bad = []
    for top in ("linac_gen", "gui"):
        for p in sorted((REPO / top).rglob("*.py")):
            rel = p.relative_to(REPO).as_posix()
            if rel not in removed:
                bad += _any_import(ast.parse(p.read_text(encoding="utf-8"), filename=rel), rel, mods)
    assert not bad, "shipped code imports modules the public release removes:\n" + "\n".join(bad)
