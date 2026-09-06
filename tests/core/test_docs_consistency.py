"""Docs/code consistency gate (2026-09 docs-drift round).

Every test here pins a public-facing documentation claim to the code
artefact it describes, so the next drift fails CI instead of shipping:

* element-catalog count and rows (introspection over ``linac_gen.elements``);
* CLI subcommand tables vs the real argparse registry;
* ``;@LG`` PIC value tokens vs ``SpaceChargeConfig`` validation (both
  directions: documented-but-refused AND valid-but-undocumented fail);
* Hofmann S^4 sign statement vs the pinned code sign;
* bunch-train neighbour-image documentation vs ``train_images``;
* the test-count figure quoted across README/CONTRIBUTING/CI/examples;
* Elegant ``.lte`` import documented everywhere the other importers are;
* no phantom "hybrid RK4-residual" surrogate path in docs or docstrings.

NOTE (intended behaviour): the catalog test enforces the definition
"concrete ``Element`` subclass defined outside ``base.py`` /
``lattice_commands.py``, not abstract, not a ``LatticeCommand``".  If a
future helper subclasses ``Element`` for internal use it WILL demand a
catalog row — that is the point: either document it or don't make it an
``Element``.

House rule honoured: no in-process pytest spawn — the test-count check
parses the prose figures and asserts cross-file agreement only.
"""
from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MANUAL = REPO / "docs" / "manual"


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Element catalog
# ---------------------------------------------------------------------------
def _concrete_element_classes() -> dict[str, str]:
    """Name -> module for every concrete Element subclass in the catalog."""
    import linac_gen.elements as pkg
    from linac_gen.elements.base import Element
    from linac_gen.elements.lattice_commands import LatticeCommand

    found: dict[str, str] = {}
    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name in ("base", "lattice_commands"):
            continue
        m = importlib.import_module(f"linac_gen.elements.{mod.name}")
        for nm, obj in vars(m).items():
            if (inspect.isclass(obj)
                    and issubclass(obj, Element)
                    and not issubclass(obj, LatticeCommand)
                    and not inspect.isabstract(obj)
                    and obj.__module__ == m.__name__):
                found[nm] = obj.__module__
    return found


def test_element_catalog_count_and_rows():
    classes = _concrete_element_classes()
    n = len(classes)

    index_md = _read("docs/manual/index.md")
    m = re.search(r"(\d+) element types", index_md)
    assert m, "index.md no longer states an element-type count"
    assert m.group(1) == str(n), (
        f"index.md says {m.group(1)} element types but introspection "
        f"finds {n}")

    overview = _read("docs/manual/03_elements/00_overview.md")
    m = re.search(r"HELIX has (\d+) concrete element classes", overview)
    assert m, "00_overview.md no longer states the class count"
    assert m.group(1) == str(n), (
        f"00_overview.md says {m.group(1)} concrete element classes but "
        f"introspection finds {n}")

    missing = [nm for nm in sorted(classes) if f"[{nm}](" not in overview]
    assert not missing, (
        "element classes without a TL;DR catalog row in "
        f"00_overview.md: {missing}")


# ---------------------------------------------------------------------------
# 2. CLI subcommand tables
# ---------------------------------------------------------------------------
def _cli_subcommands(capsys) -> list[str]:
    """Subcommand names parsed from `python -m linac_gen --help` output."""
    from linac_gen.__main__ import main

    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    m = re.search(r"\{([a-z,]+)\}", out)
    assert m, f"could not find the subcommand set in --help output:\n{out}"
    return m.group(1).split(",")


def test_cli_subcommand_table_complete(capsys):
    names = _cli_subcommands(capsys)
    assert "study" in names and "assist" in names  # sanity on the parse

    batch_cli = _read("docs/manual/06_running/06_batch_cli.md")
    for name in names:
        assert f"| `{name}` |" in batch_cli, (
            f"06_batch_cli.md subcommand table has no `{name}` row")

    readme = REPO.joinpath("README.md").read_text(encoding="utf-8")
    feature_row = next((ln for ln in readme.splitlines()
                        if "**Batch CLI**" in ln), None)
    assert feature_row is not None, "README lost its Batch CLI feature row"
    layout_row = next((ln for ln in readme.splitlines()
                       if ln.strip().startswith("cli/")), None)
    assert layout_row is not None, "README lost its cli/ layout entry"
    layout_tokens = {tok.strip() for tok in
                     layout_row.split("—", 1)[-1].split("/")}
    for name in names:
        assert f"`{name}`" in feature_row, (
            f"README Batch CLI feature row omits `{name}`")
        assert name in layout_tokens, (
            f"README cli/ layout entry omits {name} "
            f"(tokens: {sorted(layout_tokens)})")


# ---------------------------------------------------------------------------
# 3. ;@LG PIC value tokens
# ---------------------------------------------------------------------------
def _row_note_tokens(page_text: str, key: str) -> list[str]:
    """Backticked value tokens from the Notes column of a `| key | ...` row."""
    row = next((ln for ln in page_text.splitlines()
                if ln.strip().startswith(f"| `{key}` |")), None)
    assert row is not None, f"no `{key}` row in the page"
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    assert len(cells) >= 4, f"malformed `{key}` row: {row!r}"
    return re.findall(r"`([^`]+)`", cells[3])


def test_lg_extensions_pic_values_validate():
    from linac_gen.core.config import SpaceChargeConfig
    from linac_gen.pic.gpu_backend import _VALID_MODES

    page = _read("docs/manual/06_running/03_lg_extensions.md")

    # Regime A: every documented boundary token must validate (a
    # documented-but-refused token such as "periodic" fails here).
    for tok in _row_note_tokens(page, "boundary"):
        SpaceChargeConfig(boundary=tok)  # must not raise

    # Regime B: the documented use_gpu token set must equal the code's
    # whitelist exactly (a valid-but-undocumented token fails here).
    documented = set(_row_note_tokens(page, "use_gpu"))
    assert documented == set(_VALID_MODES), (
        f"03_lg_extensions.md use_gpu tokens {sorted(documented)} != "
        f"gpu_backend._VALID_MODES {sorted(_VALID_MODES)}")


# ---------------------------------------------------------------------------
# 4. Hofmann S^4 sign
# ---------------------------------------------------------------------------
def test_hofmann_footprint_sign_matches_code():
    page = _read("docs/manual/09_diagnostics/07_hofmann_footprint.md")
    assert "+S⁴" not in page.replace("`", ""), (
        "07_hofmann_footprint.md still claims the S⁴ block enters with "
        "+S⁴ — the code (hofmann_dispersion.py) uses −S⁴")
    assert ("−S⁴" in page) or ("-S⁴" in page.replace("`", "")), (
        "07_hofmann_footprint.md no longer states the −S⁴ sign")
    # The named arbiter test must exist so the doc claim stays pinned.
    pin = _read("tests/analysis/test_hofmann_dispersion.py")
    assert "def test_eq43_roots_fix_the_l4_even_sp4_sign(" in pin


# ---------------------------------------------------------------------------
# 5. Bunch-train neighbour images
# ---------------------------------------------------------------------------
def test_models_page_documents_train_images():
    from linac_gen.core.config import SpaceChargeConfig

    field_names = {f.name for f in dataclasses.fields(SpaceChargeConfig)}
    assert "train_images" in field_names

    models = _read("docs/manual/05_space_charge/01_models.md")
    assert "train_images" in models
    assert "bunch_train" in models
    assert "are not implemented" not in models, (
        "01_models.md still claims neighbour images are not implemented")

    rfqcell = _read("docs/manual/03_elements/09_rfqcell.md")
    assert "are deferred" not in rfqcell, (
        "09_rfqcell.md still calls neighbour-bunch images deferred")
    assert "≥ 35°" in rfqcell, (
        "09_rfqcell.md does not quote the real 35° engage threshold "
        "(pic_solver.py hysteresis: engage ≥ 35°, release ≤ 25°)")


# ---------------------------------------------------------------------------
# 6. Test-count figure
# ---------------------------------------------------------------------------
def test_test_count_claims_agree():
    files = ["README.md", "CONTRIBUTING.md", ".github/workflows/tests.yml",
             "examples/FIELD_MAPS.md"]
    figures = {}
    for rel in files:
        found = re.findall(r"(\d,\d{3})\+?\s+tests", _read(rel))
        assert found, f"{rel} quotes no test-count figure"
        figures[rel] = set(found)
    union = set().union(*figures.values())
    assert len(union) == 1, (
        f"test-count figures disagree across files: {figures}")


# ---------------------------------------------------------------------------
# 7. Elegant .lte import
# ---------------------------------------------------------------------------
def test_elegant_import_documented():
    readme = _read("README.md")
    assert "Elegant" in readme, "README never mentions the Elegant importer"

    batch_cli = _read("docs/manual/06_running/06_batch_cli.md")
    lines = batch_cli.splitlines()
    start = next((i for i, ln in enumerate(lines) if "lattice file" in ln),
                 None)
    assert start is not None, "06_batch_cli.md lost its input-model bullet"
    bullet_lines = [lines[start]]
    for ln in lines[start + 1:]:  # continuation lines of the same bullet
        if not ln.startswith("  ") or not ln.strip():
            break
        bullet_lines.append(ln)
    bullet = " ".join(bullet_lines)
    for suf in (".dat", ".madx", ".seq", ".lat", ".lte", ".bmad", ".jl", ".pals.yaml"):
        assert f"`{suf}`" in bullet, (
            f"06_batch_cli.md input-model bullet omits {suf} "
            "(cli/common.py:load_lattice accepts it)")

    cli_run = _read("docs/manual/06_running/07_cli_run.md")
    assert ".lte" in cli_run, "07_cli_run.md omits .lte from <input>"

    tw = _read("docs/manual/06_running/02_tracewin_dat.md")
    assert "## Importing Elegant lattices" in tw, (
        "02_tracewin_dat.md has no Elegant import section")
    assert "## Importing Bmad, SciBmad and PALS lattices" in tw, (
        "02_tracewin_dat.md has no lattix import section")
    assert ".bmad" in cli_run and ".jl" in cli_run, "07_cli_run.md omits the lattix suffixes"


# ---------------------------------------------------------------------------
# 8. No phantom hybrid RK4-residual surrogate path
# ---------------------------------------------------------------------------
def _squash(text: str) -> str:
    return " ".join(text.split())


def test_no_phantom_hybrid_residual_path():
    from linac_gen.surrogates.compare import compare_mp

    doc = compare_mp.__doc__ or ""
    assert "RK4-residual path engaged" not in _squash(doc), (
        "compare_mp docstring still describes the non-existent hybrid "
        "RK4-residual path")
    assert "reserved" in doc.lower(), (
        "compare_mp docstring should state residual_n_steps is reserved")

    gui_md = _read("docs/manual/13_surrogates/02_gui.md")
    assert "cheap RK4 residual" not in gui_md, (
        "02_gui.md still promises a cheap-RK4-residual surrogate call")
    assert "reserved" in gui_md.lower(), (
        "02_gui.md should mark the substeps control as reserved")

    api_md = _read("docs/manual/13_surrogates/04_python_api.md")
    assert ("hybrid linear-anchor + RK4-residual path with the surrogates "
            "registered") not in _squash(api_md), (
        "04_python_api.md still describes the phantom hybrid path")
    assert "reserved" in api_md.lower(), (
        "04_python_api.md should state residual_n_steps is reserved")
