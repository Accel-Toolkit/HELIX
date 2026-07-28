"""Tests for transfer-line input matching.

Covers ``find_fodo_cells`` and ``find_matched_input_twiss`` — the
FODO-cell periodic Twiss back-propagated to the lattice entrance — plus
the ``twiss`` CLI subcommand.
"""

from tests.dataguard import needs, require  # noqa: E402
from pathlib import Path

import numpy as np
import pytest

from linac_gen.core.particle import PROTON, H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.tracking.matrix_tracking import (
    compute_transfer_matrix, compute_twiss,
)
from linac_gen.matching.periodic import (
    find_fodo_cells, find_matched_input_twiss, find_periodic_twiss,
    find_sc_matched_input_twiss,
)

BTL_DAT = Path(__file__).parents[2] / "examples" / "pipii" / "btl" / "btl.dat"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _fodo_cell():
    """Elements of one FODO period: QF, D, QD, D."""
    return [
        Quadrupole("QF", 50.0, gradient=5.0, aperture=20.0, n_steps=5),
        Drift("D1", 200.0, aperture=20.0),
        Quadrupole("QD", 50.0, gradient=-5.0, aperture=20.0, n_steps=5),
        Drift("D2", 200.0, aperture=20.0),
    ]


def _multi_cell_lattice(n_cells=4, front_drift=None):
    """A lattice of n identical FODO cells, optionally preceded by a drift."""
    lat = Lattice()
    if front_drift is not None:
        lat.add(Drift("FRONT", front_drift, aperture=20.0))
    for c in range(n_cells):
        for e in _fodo_cell():
            e.name = f"{e.name}_{c}"
            lat.add(e)
    return lat


def _ref():
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)


def _btl_ref():
    return ReferenceParticle(species=H_MINUS, w_kin=800.0, frequency=162.5)


# ---------------------------------------------------------------------------
# find_fodo_cells
# ---------------------------------------------------------------------------
def test_find_fodo_cells_basic():
    lat = _multi_cell_lattice(n_cells=4)
    cells = find_fodo_cells(lat)
    assert len(cells) > 0
    for cs, ce in cells:
        assert 0 <= cs <= ce < len(lat.elements)


def test_find_fodo_cells_too_few_quads():
    lat = Lattice()
    lat.add(Drift("D", 100.0, aperture=20.0))
    lat.add(Quadrupole("Q", 50.0, gradient=5.0, aperture=20.0, n_steps=5))
    assert find_fodo_cells(lat) == []


# ---------------------------------------------------------------------------
# find_matched_input_twiss
# ---------------------------------------------------------------------------
def test_single_cell_equals_find_periodic_twiss():
    """cell = whole lattice, no front section => same as find_periodic_twiss."""
    lat = _multi_cell_lattice(n_cells=1)
    ref = _ref()
    periodic = find_periodic_twiss(lat, ref)
    matched = find_matched_input_twiss(lat, ref, 0, len(lat.elements) - 1)
    for key in ("alpha_x", "beta_x", "alpha_y", "beta_y", "mu_x", "mu_y"):
        assert matched[key] == pytest.approx(periodic[key], rel=1e-9)


def test_back_propagation_round_trip():
    """Forward-transport the matched input through the front section (pure
    numpy, independent of propagate_twiss) -> must reproduce the cell's
    periodic Twiss."""
    lat = _multi_cell_lattice(n_cells=4, front_drift=137.0)
    ref = _ref()
    cs, ce = find_fodo_cells(lat)[1]               # an interior cell
    matched = find_matched_input_twiss(lat, ref, cs, ce)
    m_cell = compute_transfer_matrix(lat, ref, start=cs, end=ce)
    m_front = np.asarray(compute_transfer_matrix(lat, ref, start=0,
                                                 end=cs - 1))
    for plane, lo in (("x", 0), ("y", 2)):
        cell_tw = compute_twiss(m_cell, plane)
        a, b = matched[f"alpha_{plane}"], matched[f"beta_{plane}"]
        sigma = np.array([[b, -a], [-a, (1.0 + a * a) / b]])
        mf = m_front[lo:lo + 2, lo:lo + 2]
        sig_cell = mf @ sigma @ mf.T
        assert sig_cell[0, 0] == pytest.approx(cell_tw["beta"], rel=1e-9)
        assert -sig_cell[0, 1] == pytest.approx(cell_tw["alpha"],
                                                rel=1e-9, abs=1e-9)


def test_out_of_bounds_raises():
    lat = _multi_cell_lattice(n_cells=2)
    ref = _ref()
    n = len(lat.elements)
    with pytest.raises(ValueError, match="out of bounds"):
        find_matched_input_twiss(lat, ref, 0, n)        # end too large
    with pytest.raises(ValueError, match="out of bounds"):
        find_matched_input_twiss(lat, ref, 5, 2)        # start > end


# ---------------------------------------------------------------------------
# BTL integration — the known-correct input matched Twiss
# ---------------------------------------------------------------------------
@needs("examples/pipii/btl/btl.dat")
def test_btl_matched_input():
    """The BTL input matched Twiss must reproduce the validated value
    (compute_twiss.py / btl.lgproj / TraceWin)."""
    lat, _ = parse_tracewin(str(BTL_DAT))
    ref = _btl_ref()
    cells = find_fodo_cells(lat)
    assert cells, "BTL should have FODO cells"
    tw = find_matched_input_twiss(lat, ref, *cells[0])
    assert tw["beta_x"] == pytest.approx(13.814, abs=0.01)
    assert tw["alpha_x"] == pytest.approx(1.919, abs=0.01)
    assert tw["beta_y"] == pytest.approx(5.384, abs=0.01)
    assert tw["alpha_y"] == pytest.approx(-0.916, abs=0.01)


@needs("examples/pipii/btl/btl.dat")
def test_btl_cell_choice_invariance():
    """Different FODO cells of the same section yield the same entrance
    Twiss — the matched input is anchor-independent."""
    lat, _ = parse_tracewin(str(BTL_DAT))
    ref = _btl_ref()
    cells = find_fodo_cells(lat)
    t0 = find_matched_input_twiss(lat, ref, *cells[0])
    t1 = find_matched_input_twiss(lat, ref, *cells[1])
    for key in ("alpha_x", "beta_x", "alpha_y", "beta_y"):
        assert t0[key] == pytest.approx(t1[key], rel=1e-6)


@needs("examples/pipii/btl/btl.dat")
def test_btl_whole_vs_cell_differ():
    """The whole-line periodic Twiss is NOT the input match — they must
    differ (the whole point of the fix)."""
    lat, _ = parse_tracewin(str(BTL_DAT))
    ref = _btl_ref()
    whole = find_periodic_twiss(lat, ref)
    cell = find_matched_input_twiss(lat, ref, *find_fodo_cells(lat)[0])
    assert abs(whole["beta_x"] - cell["beta_x"]) > 1.0


# ---------------------------------------------------------------------------
# twiss CLI subcommand
# ---------------------------------------------------------------------------
@needs("examples/pipii/btl/btl.dat")
def test_twiss_cli_cell_mode(capsys):
    from linac_gen.__main__ import main
    rc = main(["twiss", str(BTL_DAT), "--mode", "cell", "-q",
               "--energy", "800", "--species", "H-", "--freq", "162.5"])
    assert rc == 0
    nums = capsys.readouterr().out.split()
    assert len(nums) == 4
    # alpha_x beta_x alpha_y beta_y
    assert float(nums[1]) == pytest.approx(13.814, abs=0.01)
    assert float(nums[3]) == pytest.approx(5.384, abs=0.01)


@needs("examples/pipii/btl/btl.dat")
def test_twiss_cli_whole_mode(capsys):
    from linac_gen.__main__ import main
    rc = main(["twiss", str(BTL_DAT), "--mode", "whole", "-q",
               "--energy", "800", "--species", "H-", "--freq", "162.5"])
    assert rc == 0
    nums = capsys.readouterr().out.split()
    # Re-pinned for the H⁻ ion-mass fix (938.272 → 939.294 MeV): the 0.05%
    # Bρ shift moves the matched β by ~0.5%.
    assert float(nums[1]) == pytest.approx(6.135, abs=0.01)


@needs("examples/pipii/btl/btl.dat")
def test_twiss_cli_list_cells(capsys):
    from linac_gen.__main__ import main
    rc = main(["twiss", str(BTL_DAT), "--list-cells",
               "--energy", "800", "--species", "H-"])
    assert rc == 0
    assert "FODO-cell candidate" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# find_sc_matched_input_twiss — space-charge cell matching
# ---------------------------------------------------------------------------
_SC_BASE = {"alpha_z": 0.0, "beta_z": 1.0,
            "emit_x": 0.16, "emit_y": 0.16, "emit_z": 0.3}


@needs("examples/pipii/btl/btl.dat")
def test_sc_matched_requires_current():
    lat, _ = parse_tracewin(str(BTL_DAT))
    ref = _btl_ref()
    cs, ce = find_fodo_cells(lat)[0]
    with pytest.raises(ValueError, match="current"):
        find_sc_matched_input_twiss(lat, ref, cs, ce, 0.0, _SC_BASE)


@needs("examples/pipii/btl/btl.dat")
def test_sc_matched_zero_current_limit():
    """At I → 0 the SC matcher must recover the zero-current betatron match
    (the dispersion-handling regression test)."""
    lat, _ = parse_tracewin(str(BTL_DAT))
    ref = _btl_ref()
    cs, ce = find_fodo_cells(lat)[0]
    zc = find_matched_input_twiss(lat, ref, cs, ce)
    sc = find_sc_matched_input_twiss(lat, ref, cs, ce, 1e-3, _SC_BASE)
    assert sc["converged"]
    for key in ("alpha_x", "beta_x", "alpha_y", "beta_y"):
        assert sc[key] == pytest.approx(zc[key], rel=5e-3)


@needs("examples/pipii/btl/btl.dat")
def test_sc_matched_shifts_with_current():
    """A real current keeps the match converged and raises beta (SC)."""
    lat, _ = parse_tracewin(str(BTL_DAT))
    ref = _btl_ref()
    cs, ce = find_fodo_cells(lat)[0]
    zc = find_matched_input_twiss(lat, ref, cs, ce)
    sc = find_sc_matched_input_twiss(lat, ref, cs, ce, 5.0, _SC_BASE)
    assert sc["converged"]
    assert sc["beta_x"] > zc["beta_x"]
    assert sc["beta_y"] > zc["beta_y"]
