"""External oracle: HELIX's MAD-X export checked by MAD-X itself (cpymad).

Skips cleanly when ``cpymad`` is not installed (same pattern as the
Cheetah cross-checks).  Three statements:

1. every public deck's export LOADS in MAD-X (parses, ``USE`` succeeds);
2. for magnet-only decks the end-to-end transverse transfer matrix MAD-X
   computes for the exported line equals HELIX's own composed matrix
   (the 4×4 transverse block is invariant under the mm/mrad ↔ m/rad
   rescale, so it compares directly);
3. a MAD-X original (``examples/madx/*.madx``) imported into HELIX and
   re-exported gives MAD-X the same optics again: end-of-line Twiss,
   cumulative R-matrix and length agree.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

cpymad = pytest.importorskip("cpymad.madx")

from linac_gen.cli.common import build_ref
from linac_gen.core.config import BeamConfig
from linac_gen.io.madx_parser import parse_madx
from linac_gen.io.madx_writer import write_madx
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.tracking.matrix_tracking import compute_transfer_matrix

REPO = Path(__file__).resolve().parents[2]

MAGNET_DECKS = [
    "examples/fodo_cell.dat",
    "examples/impactx_features/coupled_fodo.dat",
    "examples/solenoid_channel.dat",
    "examples/csr_chicane.dat",
    "examples/correction_demo/correction_demo.dat",
    "examples/lebt_scc_demo.dat",
    "examples/matching_demo.dat",
]
RF_DECKS = ["examples/dtl_section.dat",
            "examples/hofmann_stability/hofmann_demo_linac.dat"]


def _ref(species="proton", energy=3.0):
    return build_ref(BeamConfig(species=species, energy=energy, frequency=352.21))


def _madx_end_rmatrix(path: Path):
    """Cumulative 6×6 R-matrix at the end of the exported line, from MAD-X."""
    m = cpymad.Madx(stdout=False)
    m.call(str(path))
    m.input("twiss, betx=1, bety=1, rmatrix;")
    t = m.table.twiss
    R = np.array([[t[f"re{i}{j}"][-1] for j in range(1, 7)] for i in range(1, 7)])
    s_end = float(t.s[-1])
    m.quit()
    return R, s_end


def _madx_end_twiss(path: Path):
    m = cpymad.Madx(stdout=False)
    m.call(str(path))
    m.input("twiss, betx=2.5, alfx=-0.3, bety=1.7, alfy=0.4, dx=0, dpx=0, rmatrix;")
    t = m.table.twiss
    out = {k: float(t[k][-1]) for k in ("s", "betx", "alfx", "bety", "alfy", "dx", "dpx")}
    R = np.array([[t[f"re{i}{j}"][-1] for j in range(1, 7)] for i in range(1, 7)])
    m.quit()
    return out, R


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("deck", MAGNET_DECKS + RF_DECKS)
def test_export_loads_in_real_madx(tmp_path, deck):
    lat = parse_tracewin(str(REPO / deck))[0]
    out = tmp_path / (Path(deck).stem + ".madx")
    write_madx(lat, out, _ref())
    m = cpymad.Madx(stdout=False)
    m.call(str(out))                 # raises on any syntax / semantic error
    seqs = list(m.sequence)
    assert len(seqs) == 1
    m.input("twiss, betx=1, bety=1;")
    assert len(m.table.twiss.s) > 1
    m.quit()


@pytest.mark.parametrize("species", ["proton", "H-"])
@pytest.mark.parametrize("deck", MAGNET_DECKS)
def test_transverse_matrix_matches_real_madx(tmp_path, deck, species):
    """HELIX composed 4x4 transverse matrix == MAD-X's cumulative R-matrix
    for the exported line (drifts, quads incl. skew, sector bends with
    edges and field index, solenoids, kickers, apertures)."""
    lat = parse_tracewin(str(REPO / deck))[0]
    ref = _ref(species)
    out = tmp_path / f"{Path(deck).stem}_{species}.madx"
    write_madx(lat, out, ref)
    R, s_end = _madx_end_rmatrix(out)
    M = np.asarray(compute_transfer_matrix(lat, copy.deepcopy(ref)))
    L = sum(float(getattr(e, "length", 0.0) or 0.0) for e in lat.elements)
    assert s_end == pytest.approx(L * 1e-3, rel=1e-12)
    np.testing.assert_allclose(M[:4, :4], R[:4, :4], rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("madx_file", ["examples/madx/fodo.madx",
                                       "examples/madx/transport.madx"])
def test_madx_original_survives_helix_round_trip(tmp_path, madx_file):
    """MAD-X original -> HELIX -> MAD-X export: MAD-X sees the same line."""
    src = REPO / madx_file
    lat, meta = parse_madx(str(src))
    out = tmp_path / "reexport.madx"
    warnings = write_madx(lat, out, meta["reference"])
    assert warnings == [], warnings
    tw_a, R_a = _madx_end_twiss(src)
    tw_b, R_b = _madx_end_twiss(out)
    assert tw_a["s"] == pytest.approx(tw_b["s"], rel=1e-12)
    for k in ("betx", "alfx", "bety", "alfy", "dx", "dpx"):
        assert tw_a[k] == pytest.approx(tw_b[k], rel=1e-8, abs=1e-10), k
    np.testing.assert_allclose(R_a[:4, :4], R_b[:4, :4], rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------
# linearised (MATRIX) export — MAD-X canonical basis pinned by MAD-X itself
# ---------------------------------------------------------------------------
from linac_gen.core.particle import PROTON, H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.matrix_element import MatrixElement
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.tracking.longitudinal_coords import matrix_to_madx, vector_from_madx
from tests.io.test_madx_linearize import (_rf_map, _solenoid_map,
                                          _symplectic_4x4)


def _madx_track_columns(path, amplitudes=1e-4):
    """Track one particle per transverse coordinate through the exported
    line; return the 4x4 map MAD-X applied (columns, in m/rad × 1e4) and
    the pt every particle ends with."""
    m = cpymad.Madx(stdout=False)
    m.call(str(path))
    starts = "".join(
        f"start, x={x}, px={px}, y={y}, py={py};"
        for x, px, y, py in [(amplitudes, 0, 0, 0), (0, amplitudes, 0, 0),
                             (0, 0, amplitudes, 0), (0, 0, 0, amplitudes),
                             (0, 0, 0, 0)])                 # 5th: the origin
    m.input("track, onepass, onetable;" + starts + "run, turns=1; endtrack;")
    t = m.table.trackone
    cols = {c: np.array(t[c]) for c in ("number", "s", "x", "px", "y", "py", "pt")}
    m.quit()
    end = cols["s"] > 0
    R = np.zeros((4, 4))
    for k in range(4):
        row = [j for j in np.where(end)[0] if int(cols["number"][j]) == k + 1][0]
        R[:, k] = [cols[c][row] / amplitudes for c in ("x", "px", "y", "py")]
    origin = [j for j in np.where(end)[0] if int(cols["number"][j]) == 5][0]
    return R, float(cols["pt"][origin])


@pytest.mark.parametrize("species", [PROTON, H_MINUS])
def test_drift_six_by_six_matches_madx(tmp_path, species):
    """The whole 6x6 (t, pt conventions included): HELIX drift → MAD-X
    canonical == MAD-X's own drift R-matrix, incl. R56 = L/β²γ²."""
    ref = ReferenceParticle(species=species, w_kin=5.0, frequency=352.21)
    lat = Lattice()
    lat.add(Drift("d", length=1000.0))
    out = tmp_path / "drift.madx"
    write_madx(lat, out, ref)
    R, _ = _madx_end_rmatrix(out)
    expected = matrix_to_madx(compute_transfer_matrix(lat, ref.copy()), ref)
    np.testing.assert_allclose(R, expected, rtol=1e-10, atol=1e-12)


def test_symplectic_matrix_element_survives_madx_twiss(tmp_path):
    """A symplectic explicit map (with dispersion-like x–pt and t–px
    coupling) exported as MATRIX: MAD-X TWISS reproduces the converted
    HELIX line matrix (so MAD-X's symplectification leaves it alone)."""
    ref = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    M = _symplectic_4x4()
    lat = Lattice()
    lat.add(Drift("d1", length=100.0))
    lat.add(MatrixElement("mx", M, length=250.0))
    lat.add(Quadrupole("q", length=100.0, gradient=2.0))
    lat.add(Drift("d2", length=100.0))
    out = tmp_path / "mx.madx"
    write_madx(lat, out, ref, linearize=True)
    R, s_end = _madx_end_rmatrix(out)
    assert s_end == pytest.approx(0.55, rel=1e-12)
    expected = matrix_to_madx(compute_transfer_matrix(lat, ref.copy()), ref)
    np.testing.assert_allclose(R, expected, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("regime", ["static", "rf"])
def test_linearised_field_map_matches_madx_track(tmp_path, regime):
    """MAD-X TRACK (which applies MATRIX verbatim) through the exported
    line == HELIX's composed line matrix on the transverse coordinates,
    in the ΔW = 0 and the ΔW ≠ 0 regime.  In the RF regime every particle
    leaves with pt = ΔW_ref/(p0 c): MAD-X sees the gain as a pt shift."""
    ref = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    lat = Lattice()
    lat.add(Drift("d1", length=100.0))
    fm = _solenoid_map() if regime == "static" else _rf_map()
    lat.add(fm)
    lat.add(Drift("d2", length=300.0))
    out = tmp_path / f"{regime}.madx"
    warnings = write_madx(lat, out, ref, linearize=True)
    assert len([w for w in warnings if "linearised as MAD-X MATRIX" in w]) == 1
    M_line = compute_transfer_matrix(lat, ref.copy())
    # local exit reference (for the angle rescale of MAD-X's px)
    r = ref.copy()
    r.s += 100.0
    r.phi_s += 360.0 * 100.0 / (r.beta * r.wavelength)
    fm.reset_run_state()
    fm.fitted_matrix(r.copy())
    fm.advance_ref(r)
    dW = r.w_kin - ref.w_kin
    p0c = ref.bg * PROTON.mass
    R_track, pt_origin = _madx_track_columns(out)
    # the reference-like particle (origin) leaves with exactly the kick6
    assert pt_origin == pytest.approx(dW / p0c, abs=1e-15)
    # MAD-X (x, px) at the exit → HELIX (mm, mrad) at the exit momentum
    T_out = np.diag(vector_from_madx(np.ones(6), ref, r))[:4, :4]
    T_in = np.diag([1000.0] * 4)
    helix_from_madx = T_out @ R_track @ np.linalg.inv(T_in)
    np.testing.assert_allclose(helix_from_madx, M_line[:4, :4], rtol=1e-7, atol=1e-9)
    if regime == "static":
        assert dW == 0.0
    else:
        assert dW != 0.0
