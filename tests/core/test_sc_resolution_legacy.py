"""Projects saved before named space-charge methods must resolve to exactly
the configuration they always produced.

``_legacy_make_sc_config`` and ``_legacy_scan_sc`` transcribe the
space-charge parts of ``make_sc_config`` and ``build_scan_point`` (plus the
scan worker's construction) as they were in HELIX 1.14.0 (04d5e0a; unchanged
through e46b330), returning field dicts with the 1.14.0 dataclass defaults
instead of building the old class.  Left out on purpose: the unknown-key
checks (an error then and now), and ``--sc`` overrides of nx/ny/nz,
grid_extent or use_gpu on the scan path, which 1.14.0's worker rejected with
a duplicate-keyword TypeError (they now work; see
tests/parallelization/test_scan_point_sc.py).  The ``--sc`` values used here
avoid the fields whose text conversion changed (train_images, halo; see
tests/core/test_sc_methods.py).  The shipped projects are found by glob at
test time; a project that names a space-charge method other than "custom"
was saved after named methods existed and resolves through that method
(tests/core/test_sc_methods.py), so it is left out here.
"""
from __future__ import annotations

import glob
import json
import os

import pytest

from linac_gen.cli import common

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Field values SpaceChargeConfig() had in 1.14.0.
_D = dict(nx=96, ny=96, nz=96, boundary="open", solver="fft", grid_extent=5.0,
          shape_order=1, grid_mode="fixed", use_gpu="auto", kernel="cic",
          green_kind="igf", dc_kernel="uniform", train_images=None,
          csr_enabled=False, csr_bins=200, csr_model="1d_steady",
          sc_backend="numpy", halo=None)


def _first(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def _coerce_like(old, raw):
    if isinstance(old, bool):
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(old, int):
        return int(float(raw))
    if isinstance(old, float):
        return float(raw)
    return raw


def _legacy_make_sc_config(conv, cli):
    """make_sc_config of 1.14.0 (current > 0 branch) -> dict of fields."""
    conv = conv or {}
    nx = int(_first(cli.get("nx"), conv.get("grid_nx"), _D["nx"]))
    kw = dict(
        nx=nx, ny=nx, nz=nx,
        grid_extent=float(_first(cli.get("grid_extent"),
                                 conv.get("grid_extent_sigma"),
                                 _D["grid_extent"])),
        kernel=str(_first(cli.get("kernel"), conv.get("kernel"), _D["kernel"])),
        green_kind=str(conv.get("green_kind", _D["green_kind"])),
        grid_mode=str(conv.get("grid_mode", _D["grid_mode"])),
        dc_kernel=str(conv.get("dc_kernel", _D["dc_kernel"])),
        use_gpu=str(_first(cli.get("backend"), conv.get("backend"),
                           _D["use_gpu"])),
        csr_enabled=bool(conv.get("csr_enabled", _D["csr_enabled"])),
    )
    for key, raw in (cli.get("sc") or {}).items():
        kw[key] = _coerce_like(_D[key], str(raw))
    return {**_D, **kw}


def _legacy_scan_sc(conv, cli, sc_overrides):
    """build_scan_point (SC part) + scan worker construction of 1.14.0."""
    nx = int(_first(cli.get("nx"), conv.get("grid_nx"), _D["nx"]))
    grid_extent = float(_first(cli.get("grid_extent"),
                               conv.get("grid_extent_sigma"), _D["grid_extent"]))
    backend = str(_first(cli.get("backend"), conv.get("backend"), _D["use_gpu"]))
    sc_kw = {}
    for key in ("kernel", "green_kind", "grid_mode", "dc_kernel", "csr_enabled"):
        if conv and key in conv:
            sc_kw[key] = conv[key]
    for key, raw in (sc_overrides or {}).items():
        sc_kw[key] = _coerce_like(_D[key], str(raw))
    return {**_D, "nx": nx, "ny": nx, "nz": nx, "grid_extent": grid_extent,
            "use_gpu": backend, **sc_kw}


def _fields(cfg):
    """The 1.14.0 fields; a project without sc_method also keeps applying
    HELIX_SC_GRID cards (grid_card, added later)."""
    assert cfg.grid_card == "apply"
    return {k: getattr(cfg, k) for k in _D}


def _projects():
    out = []
    for p in sorted(glob.glob(os.path.join(_REPO, "examples", "**", "*.lgproj"),
                              recursive=True)):
        try:
            conv = json.load(open(p)).get("convergence") or {}
        except (OSError, ValueError):
            continue
        if conv.get("sc_method") not in (None, "custom"):
            continue                    # (a named method: not a legacy project)
        out.append(pytest.param(conv, id=os.path.relpath(p, _REPO)))
    return out


_SYNTHETIC = [
    {}, {"grid_nx": 32}, {"grid_extent_sigma": 7.0}, {"kernel": "tsc"},
    {"grid_mode": "adaptive"}, {"green_kind": "point"}, {"dc_kernel": "pic2d"},
    {"backend": "cpu"}, {"csr_enabled": True}, {"sc_backend": "torch"},
    {"grid_nx": 64, "grid_extent_sigma": 20.0, "kernel": "tsc",
     "grid_mode": "fixed", "green_kind": "igf", "backend": "auto",
     "dc_kernel": "uniform", "csr_enabled": False},
]
_CLI = [
    {}, {"nx": 40}, {"grid_extent": 6.5}, {"kernel": "tsc"}, {"backend": "cpu"},
    {"sc": {"green_kind": "point", "csr_bins": "300"}},
    {"nx": 24, "grid_extent": 4.0, "kernel": "cic", "backend": "auto",
     "sc": {"grid_mode": "adaptive"}},
]


@pytest.mark.parametrize("conv", _projects() + _SYNTHETIC)
@pytest.mark.parametrize("cli", _CLI)
def test_make_sc_config_unchanged_for_projects_without_method(conv, cli):
    class _Beam:
        current = 5.0
    got = _fields(common.make_sc_config(_Beam(), conv, cli))
    assert got == _legacy_make_sc_config(conv, cli)


@pytest.mark.parametrize("conv", _projects() + _SYNTHETIC)
@pytest.mark.parametrize("cli,sc", [({}, {}), ({"nx": 40, "backend": "cpu"}, {}),
                                    ({}, {"green_kind": "point"}),
                                    ({"grid_extent": 9.0}, {"grid_mode": "adaptive",
                                                            "kernel": "tsc"})])
def test_scan_point_config_unchanged_for_projects_without_method(conv, cli, sc, tmp_path):
    from linac_gen.parallel.scan_pool import worker_sc_config
    proj = tmp_path / "p.lgproj"
    deck = tmp_path / "d.dat"
    deck.write_text("DRIFT 100 20 0\nEND\n")
    proj.write_text(json.dumps({"__kind__": "linac_gen_project", "__version__": 1,
                                "lattice_path": "d.dat",
                                "beam": {"current": 5.0}, "convergence": conv}))
    point = common.build_scan_point(str(proj), sc_overrides=sc, cli=cli)
    got = _fields(worker_sc_config(point))
    assert got == _legacy_scan_sc(conv, cli, sc)


@pytest.mark.parametrize("raw,new", [("false", False), ("0", False),
                                     ("auto", None), ("true", True),
                                     (False, False), (None, None)])
def test_train_images_now_means_what_it_says(raw, new):
    """The one deliberate change.  1.14.0 kept a --sc (or job-file) value of
    train_images as text -- the field has no typed default -- and any text
    is true, so "false", "0", "auto" and a JSON false or null all switched
    the bunch-train images ON."""
    class _Beam:
        current = 5.0
    old = _legacy_make_sc_config({}, {"sc": {"train_images": raw}})
    assert old["train_images"] == str(raw) and bool(old["train_images"])
    got = common.make_sc_config(_Beam(), {}, {"sc": {"train_images": raw}})
    assert got.train_images is new
