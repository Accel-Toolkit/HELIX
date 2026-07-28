# tests/analysis/test_superpose_realdeck.py
"""Real-deck SUPERPOSE_MAP validation against the PIP-II FDR TraceWin deck.

Gated on the user's (never-committed) PIP-II paper repository: the FDR
SC-linac deck ``final_lattice.dat`` carries 111 genuine ``superpose_map``
cards (every HWR period stacks hwrcx + hwrcy steerer maps on the
HWR-SOL-ANLMAP solenoid; SSR periods stack b1 + b2 + SSR_SOL_cut), plus
TraceWin's own outputs as external ground truth.

Anchors pinned here (2026-07 phase F):
* cluster census — 37 containers, 8 of shape (1-D solenoid + two 3-D
  correctors) and 29 all-3-D;
* geometry — every TraceWin element-end z in ``tracewin.out`` coincides
  with a HELIX element boundary (<0.5 mm) over the full 162.9 m export;
* physics (slow) — σx/σy/σφ at the exit of the FIRST cluster (upstream
  of any cavity, so no cavity-model differences pollute it) match the
  TraceWin envelope export to 2 %.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pytest

FDR = Path.home() / (
    "Desktop/Projects/PIP_II/Paper/Lattice_Design_and_Beam_dynamics/"
    "TraceWin/SC_Linac/FDR_Design"
)
DECK = FDR / "calculations" / "final_lattice.dat"
TW_OUT = FDR / "calculations" / "tracewin.out"
DST = FDR / "input.dst"

_needs_fdr = pytest.mark.skipif(
    not (DECK.is_file() and TW_OUT.is_file() and (FDR / "Fields").is_dir()),
    reason="PIP-II FDR TraceWin deck + reference outputs not present",
)


def _patched_deck(tmp_path) -> str:
    """Copy the deck with FIELD_MAP_PATH pointed at the real field root
    (the shipped relative path resolves one level above the deck)."""
    text = DECK.read_text(encoding="latin-1")
    text = re.sub(r"(?im)^FIELD_MAP_PATH .*$",
                  f"FIELD_MAP_PATH {FDR / 'Fields'}", text)
    p = tmp_path / "fdr_final_lattice.dat"
    p.write_text(text, encoding="latin-1")
    return str(p)


def _parse(tmp_path):
    from linac_gen.io.tracewin_parser import parse_tracewin
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parse_tracewin(_patched_deck(tmp_path))


def _tw_rows(path=None):
    rows = []
    with open(path or TW_OUT) as fh:
        for line in fh:
            t = line.split()
            if len(t) < 30:
                continue
            try:
                int(t[0])
                rows.append([float(v) for v in t[:30]])
            except ValueError:
                continue
    return np.asarray(rows)


@_needs_fdr
def test_fdr_cluster_census(tmp_path):
    lat, meta = _parse(tmp_path)
    from linac_gen.elements.superposed_field_map import SuperposedFieldMap
    sups = [e for e in lat.elements if isinstance(e, SuperposedFieldMap)]
    assert len(sups) == 37
    shapes = sorted(
        tuple(sorted(type(c).__name__ for _z, c in e.children))
        for e in sups
    )
    assert shapes.count(("FieldMap", "FieldMap3D", "FieldMap3D")) == 8
    assert shapes.count(("FieldMap3D", "FieldMap3D", "FieldMap3D")) == 29
    # All at z0 = 0 (full-window stacks); the HWR packages span 300 mm.
    assert all(abs(z0) < 1e-9 for e in sups for z0, _c in e.children)
    hwr = [e for e in sups
           if tuple(sorted(type(c).__name__ for _z, c in e.children))
           == ("FieldMap", "FieldMap3D", "FieldMap3D")]
    assert all(e.length == pytest.approx(300.0) for e in hwr)
    # No superpose/shift-related parse degradation.
    assert not any("SUPERPOSE" in w or "SHIFT" in w
                   for w in meta["warnings"]), meta["warnings"]


@_needs_fdr
def test_fdr_geometry_matches_tracewin(tmp_path):
    """Every TW element-end z coincides with a HELIX element boundary —
    the cluster span rule (s advances once by max(z0+L)) reproduces
    TraceWin's s-accounting over the whole 162.9 m export."""
    lat, _ = _parse(tmp_path)
    bounds = np.cumsum([float(getattr(e, "length", 0.0) or 0.0)
                        for e in lat.elements])
    # An element's END is its LAST partran row (both exports also carry
    # entry/sub-element rows); pin each element-end z to a boundary.
    tw = _tw_rows(FDR / "calculations" / "partran1.out")
    ends = {}
    for row in tw:
        ends[int(row[0])] = max(ends.get(int(row[0]), -1e9),
                                row[1] * 1000.0)
    tw_z = np.unique(np.round(sorted(ends.values()), 3))
    tw_z = tw_z[tw_z > 1.0]
    for z in tw_z:
        j = np.searchsorted(bounds, z)
        near = min(abs(bounds[max(j - 1, 0)] - z),
                   abs(bounds[min(j, len(bounds) - 1)] - z))
        assert near < 0.5, f"TW boundary z={z:.3f} mm has no HELIX match"


@_needs_fdr
@pytest.mark.slow
def test_fdr_first_cluster_matches_tracewin_envelope(tmp_path):
    """σ at the exit of the first hwrcx+hwrcy+solenoid cluster (before
    any cavity) vs the TraceWin envelope export, same input beam
    (TraceWin's own input.dst moments), same current."""
    if not DST.is_file():
        pytest.skip("input.dst not present")
    from linac_gen.io.tracewin_dst import load_dst
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.tracking.envelope import EnvelopeSolver

    parts, hdr = load_dst(str(DST))
    cov = np.cov(parts.T)

    def _twiss(c):
        e = float(np.sqrt(max(np.linalg.det(c), 1e-30)))
        return -c[0, 1] / e, c[0, 0] / e, e

    ax, bx, ex = _twiss(cov[np.ix_([0, 1], [0, 1])])
    ay, by, ey = _twiss(cov[np.ix_([2, 3], [2, 3])])
    az, bz, ez = _twiss(cov[np.ix_([4, 5], [4, 5])])
    init = dict(alpha_x=ax, beta_x=bx, emit_x=ex,
                alpha_y=ay, beta_y=by, emit_y=ey,
                alpha_z=az, beta_z=bz, emit_z=ez)
    ref = ReferenceParticle(species=H_MINUS, w_kin=hdr["w_kin_ref"],
                            frequency=hdr["frequency_MHz"])
    lat, _ = _parse(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = EnvelopeSolver(lat, ref, init, current=hdr["current_mA"],
                             initial_sigma=cov).run()
    names = list(res.element_names)
    i = names.index("SUPERP_001")
    s1 = res.s[i]
    tw = _tw_rows()
    tz = tw[:, 1] * 1000.0
    tw_sx = np.interp(s1, tz, tw[:, 9])
    tw_sy = np.interp(s1, tz, tw[:, 10])
    tw_sp = np.interp(s1, tz, tw[:, 11])
    # 2026-07 measured: 1.500/1.522, 1.493/1.514, 4.785/4.784 (≤1.5 %).
    assert res.sigma_x[i] == pytest.approx(tw_sx, rel=0.02)
    assert res.sigma_y[i] == pytest.approx(tw_sy, rel=0.02)
    assert res.sigma_phi[i] == pytest.approx(tw_sp, rel=0.02)
