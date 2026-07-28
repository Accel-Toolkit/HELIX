"""Parser dispatch for FIELD_MAP: every realistic geom lands on the right element.

After the `decode_geom` rewrite (plan §Task 8), the parser no longer has a
hard-coded `fm_type ∈ {70, 71, 74}` special case.  Instead it decodes the
5-digit geom, opens the component files named by the manual's channel/digit
table, and a factory selects ``FieldMap`` (1-D / 2-D cyl / 1-D quad gradient)
or ``FieldMap3D`` (3-D Cartesian).  These tests exercise realistic geoms:

* 70 — 3-D static magnetic solenoid/quadrupole → FieldMap3D
* 100 — 1-D RF electric cavity → FieldMap
* 400 — 2-D cyl RF electric (TM mode with Bθ) → FieldMap
* 7700 — 3-D RF electric + RF magnetic (6 files) → FieldMap3D
* 90 — 1-D quad gradient G(z) → FieldMap
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from linac_gen.io.tracewin_parser import parse_tracewin

# Reuse the canonical-format writers from the Task 4 test file.
from tests.io.test_tracewin_fieldmap_reader import _w_1d, _w_2d_cyl, _w_3d_cart


def _dat(body: str) -> str:
    return f"TITLE t\nFREQ 352.21\n{body}\nEND\n"


# ---------------------------------------------------------------------
# Minimal fixture writers — one function per geom, each writes every
# file that `component_files` expects for that geom.
# ---------------------------------------------------------------------

def _write_geom_70(prefix: str) -> None:
    """geom=70 → 3-D Cart static magnetic: .bsx .bsy .bsz."""
    Nz, Nx, Ny = 3, 2, 2
    zeros = np.zeros((Nz + 1, Ny + 1, Nx + 1))
    for suf in (".bsx", ".bsy", ".bsz"):
        _w_3d_cart(prefix + suf,
                   Nz=Nz, Zmax_m=0.01,
                   Nx=Nx, Xmin_m=-0.005, Xmax_m=0.005,
                   Ny=Ny, Ymin_m=-0.005, Ymax_m=0.005,
                   norm=1.0, values=zeros)


def _write_geom_100(prefix: str) -> None:
    """geom=100 → 1-D RF electric: .edz."""
    _w_1d(prefix + ".edz",
          Nz=3, Zmax_m=0.01, norm=1.0,
          values=np.array([0.0, 0.5, 1.0, 0.5]))


def _write_geom_400(prefix: str) -> None:
    """geom=400 → 2-D cyl RF electric (TM): .edr .edz .bdq."""
    Nz, Nr = 3, 2
    zeros = np.zeros((Nz + 1, Nr + 1))
    for suf in (".edr", ".edz", ".bdq"):
        _w_2d_cyl(prefix + suf,
                  Nz=Nz, Zmax_m=0.01,
                  Nr=Nr, Rmax_m=0.005,
                  norm=1.0, values=zeros)


def _write_geom_7700(prefix: str) -> None:
    """geom=7700 → 3-D Cart RF electric + RF magnetic: 6 files."""
    Nz, Nx, Ny = 3, 2, 2
    zeros = np.zeros((Nz + 1, Ny + 1, Nx + 1))
    for suf in (".edx", ".edy", ".edz", ".bdx", ".bdy", ".bdz"):
        _w_3d_cart(prefix + suf,
                   Nz=Nz, Zmax_m=0.01,
                   Nx=Nx, Xmin_m=-0.005, Xmax_m=0.005,
                   Ny=Ny, Ymin_m=-0.005, Ymax_m=0.005,
                   norm=1.0, values=zeros)


def _write_geom_90(prefix: str) -> None:
    """geom=90 → 1-D G(z) quad gradient (STAT_B only): .bsz."""
    _w_1d(prefix + ".bsz",
          Nz=3, Zmax_m=0.01, norm=1.0,
          values=np.array([1.0, 1.0, 1.0, 1.0]))   # uniform gradient


# ---------------------------------------------------------------------
# Parametrised dispatch test
# ---------------------------------------------------------------------

@pytest.mark.parametrize("geom, element_type, writer", [
    (70,    "FieldMap3D",  _write_geom_70),
    (100,   "FieldMap",    _write_geom_100),
    (400,   "FieldMap",    _write_geom_400),
    (7700,  "FieldMap3D",  _write_geom_7700),
    (90,    "FieldMap",    _write_geom_90),
])
def test_dispatch_lands_on_correct_element(
    tmp_path, geom, element_type, writer,
):
    prefix = str(tmp_path / "x")
    writer(prefix)
    dat = tmp_path / "lattice.dat"
    # Note: Ka=0 (no aperture map file) because no .ouv fixture is written.
    dat.write_text(
        _dat(f"FIELD_MAP {geom} 10 0 20 1 1 0 0 x 0")
    )
    lat, meta = parse_tracewin(str(dat))
    types = [type(e).__name__ for e in lat.elements]
    assert element_type in types, (
        f"geom={geom}: expected {element_type}, got {types}; "
        f"warnings: {meta.get('warnings', [])}"
    )


def test_geom_7700_element_has_both_e_and_b_channels(tmp_path):
    """A 7700 FIELD_MAP should yield a FieldMap3D whose field_data has
    both RF_E and RF_B channels populated."""
    from linac_gen.io.tracewin_geom import Channel

    prefix = str(tmp_path / "cav")
    _write_geom_7700(prefix)
    dat = tmp_path / "lattice.dat"
    dat.write_text(_dat("FIELD_MAP 7700 10 0 20 1 1 0 0 cav 0"))
    lat, meta = parse_tracewin(str(dat))
    fmap3d = next((e for e in lat.elements
                   if type(e).__name__ == "FieldMap3D"), None)
    assert fmap3d is not None, meta.get("warnings", [])
    assert Channel.RF_E in fmap3d.field_data.channels
    assert Channel.RF_B in fmap3d.field_data.channels


def test_missing_3d_file_records_warning(tmp_path):
    """Missing files produce a warning and skip the element, not a crash."""
    dat = tmp_path / "lattice.dat"
    dat.write_text(_dat("FIELD_MAP 70 10 0 20 1 1 0 0 nope 0"))
    lat, meta = parse_tracewin(str(dat))
    assert all(type(e).__name__ != "FieldMap3D" for e in lat.elements)
    assert any(
        "missing" in w.lower() or "not found" in w.lower()
        for w in meta.get("warnings", [])
    ), meta.get("warnings", [])


def test_kb_ke_propagated_to_element(tmp_path):
    """Scale factors kb and ke from the FIELD_MAP card reach the element."""
    prefix = str(tmp_path / "sol")
    _write_geom_70(prefix)
    dat = tmp_path / "lattice.dat"
    dat.write_text(_dat("FIELD_MAP 70 10 0 20 9.174 0 0 0 sol 0"))
    lat, meta = parse_tracewin(str(dat))
    fmap3d = next((e for e in lat.elements
                   if type(e).__name__ == "FieldMap3D"), None)
    assert fmap3d is not None, meta.get("warnings", [])
    assert fmap3d.kb == pytest.approx(9.174)
    assert fmap3d.ke == pytest.approx(0.0)
