"""The matched beam of a periodic cell where nothing changes a particle's energy with its phase.

In a cell with no RF (a transport line's FODO or arc) a bunched beam has no matched bunch: it
lengthens on every pass.  Iterating the longitudinal block too, the matched-beam search handed on
wherever it stopped: on the FODO below it did not converge, Σ_xx sat 29 % under the matched beam
with the bunch held, and a 1e-15 change of the input moved the transverse block by 8.7e-4 and
Σ_φφ by 0.26 deg²; with a long bunch its bunch-length entry dominated the tolerance and let a
false convergence through (measured 2026-11-07).  The search now holds the incoming beam's bunch
length and energy spread there, solves for the transverse beam and its dispersion, and says so;
a DC beam and a cell with RF iterate as before, and a search that does not converge warns.
"""
import copy
import warnings

import numpy as np
import pytest

from linac_gen.analysis.period_detect import PeriodicStructure
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.step_config import StepConfig
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.matrix_element import MatrixElement
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.elements.solenoid import Solenoid
from linac_gen.matching import periodic
from linac_gen.matching.periodic import find_matched_period_sigma, make_period_sigma_map
from linac_gen.tracking.matrix_tracking import compute_transfer_matrix

INITIAL = {"emit_x": 0.2, "emit_y": 0.2, "emit_z": 0.2, "alpha_z": 0.0, "beta_z": 1.0,
           "alpha_x": 0.0, "beta_x": 1.0, "alpha_y": 0.0, "beta_y": 1.0}
# a bunch with a chirp, so that its phase correlates with the energy and, through the
# dispersion, with x
ARC_INITIAL = dict(INITIAL, alpha_z=0.5)


def _ref(w=120.0):
    return ReferenceParticle(species=PROTON, w_kin=w, frequency=352.21)


def _fodo(i):
    return [Quadrupole(f"QF{i}", length=100.0, gradient=8.0, aperture=30.0),
            Drift(f"Da{i}", 300.0),
            Quadrupole(f"QD{i}", length=100.0, gradient=-8.0, aperture=30.0),
            Drift(f"Db{i}", 300.0)]


def _arc(i):
    """A FODO cell with two 5° bends in it: it makes dispersion."""
    return [Quadrupole(f"QF{i}", length=100.0, gradient=8.0, aperture=30.0), Drift(f"Da{i}", 20.0),
            Dipole(f"B1{i}", angle=5.0, rho=3000.0), Drift(f"Db{i}", 20.0),
            Quadrupole(f"QD{i}", length=100.0, gradient=-8.0, aperture=30.0), Drift(f"Dc{i}", 20.0),
            Dipole(f"B2{i}", angle=5.0, rho=3000.0), Drift(f"Dd{i}", 20.0)]


def _rf_gap(phase=-30.0):
    return RFGap("G", voltage=5.0, phase=phase, frequency=352.21, ttf=1.0)


def _periodic(cell=_fodo, extra=(), n_cells=4, prefix=()):
    """n_cells repeats of cell(i) followed by extra, after the elements of prefix; the period
    is the repeats."""
    lat = Lattice()
    for e in prefix:
        lat.add(e)
    n = 0
    for i in range(n_cells):
        els = [*cell(i), *[e_() for e_ in extra]]
        n = len(els)
        for e in els:
            lat.add(e)
    lat.step_config = StepConfig(100.0, 50.0)
    a = len(prefix)
    return lat, PeriodicStructure(start=a, end=len(lat.elements), inner_period_length=n,
                                  inner_slice_end=a + n, n_repeats=n_cells, label="p",
                                  source="fallback")


def _solve(lat, period, initial=INITIAL, current=5.0, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return find_matched_period_sigma(lat, _ref(), period, current, dict(initial), **kw)


def _cell_matrix(lat, period):
    a, b = period.spans()[0]
    sub = Lattice()
    for e in copy.deepcopy(lat.elements[a:b]):
        sub.add(e)
    sub.step_config = lat.step_config
    return compute_transfer_matrix(sub, _ref())


def _matched_at_zero_current(M, eps_x, eps_y, s55):
    """The closed-form matched transverse Σ of an uncoupled cell at zero current: the betatron
    beam of emittances eps_x and eps_y, plus the periodic dispersion's share D·Dᵀ·Σ55."""
    S = np.zeros((4, 4))
    for (i, j), eps in (((0, 1), eps_x), ((2, 3), eps_y)):
        m = M[i:j + 1, i:j + 1]
        c = 0.5 * (m[0, 0] + m[1, 1])
        s = np.sign(m[0, 1]) * np.sqrt(1.0 - c * c)
        beta, alpha = m[0, 1] / s, (m[0, 0] - m[1, 1]) / (2.0 * s)
        S[i:j + 1, i:j + 1] = eps * np.array([[beta, -alpha], [-alpha, (1.0 + alpha ** 2) / beta]])
    D = np.linalg.solve(np.eye(4) - M[0:4, 0:4], M[0:4, 5])
    return S + np.outer(D, D) * s55, D


def test_a_bunched_beam_in_a_cell_without_rf_is_matched_with_its_bunch_held():
    lat, period = _periodic()
    with pytest.warns(UserWarning, match="no matched bunch") as rec:
        got = find_matched_period_sigma(lat, _ref(), period, 5.0, dict(INITIAL))
    assert not [w for w in rec if issubclass(w.category, RuntimeWarning)]   # it converged
    assert got["converged"] and got["longitudinal"] == "held"
    assert got["sigma_model_approx"] == "held-longitudinal" and "no RF" in got["longitudinal_held"]
    S = got["sigma_entry"]
    seed = make_period_sigma_map(lat, _ref(), period, 5.0, dict(INITIAL))[1]["sigma_seed"]
    np.testing.assert_array_equal(S[4:6, 4:6], seed[4:6, 4:6])           # the bunch, held exactly
    frozen = _solve(lat, period, longitudinal="frozen")["sigma_entry"]   # no bends: the same beam
    assert np.abs(S - frozen).max() < 1e-6 * np.abs(frozen).max()


def test_the_matched_beam_does_not_follow_round_off():
    """A 1e-15 change of the input moves the matched beam by no more than its solver tolerance
    allows (it moved Σ_φφ by 0.26 deg² and the transverse block by 8.7e-4 before)."""
    lat, period = _periodic()
    a = _solve(lat, period)["sigma_entry"]
    for key in ("emit_x", "beta_x"):
        pert = dict(INITIAL)
        pert[key] *= 1.0 + 1e-15
        b = _solve(lat, period, pert)["sigma_entry"]
        assert np.abs(b - a).max() < 1e-6 * np.abs(a).max()


@pytest.mark.parametrize("incoming_dispersion", [0.0, 0.5])
def test_a_cell_with_bends_gets_its_periodic_dispersion_and_the_incoming_betatron_emittances(
        incoming_dispersion):
    """At zero current the held beam is the cell's closed-form matched beam, whatever dispersion
    the beam came in with (a fraction of the periodic one here), and a covariance matrix.  The
    phase correlates with x only through the energy."""
    lat, period = _periodic(cell=_arc)
    M = _cell_matrix(lat, period)
    D = np.linalg.solve(np.eye(4) - M[0:4, 0:4], M[0:4, 5])
    initial = dict(ARC_INITIAL, disp_x=incoming_dispersion * D[0],
                   disp_xp=incoming_dispersion * D[1])
    got = _solve(lat, period, initial, current=0.0)
    S = got["sigma_entry"]
    assert got["converged"] and got["longitudinal"] == "held"
    assert np.linalg.eigvalsh(S).min() > 0.0
    want, D = _matched_at_zero_current(M, INITIAL["emit_x"], INITIAL["emit_y"], S[5, 5])
    assert np.abs(S[0:4, 0:4] - want).max() < 1e-6 * np.abs(want).max()
    scale = np.abs(D).max() * S[5, 5]
    np.testing.assert_allclose(S[0:4, 5], D * S[5, 5], rtol=0.0, atol=1e-6 * scale)
    np.testing.assert_allclose(S[0:4, 4], S[0:4, 5] * S[4, 5] / S[5, 5], rtol=0.0,
                               atol=1e-12 * scale)


def test_with_space_charge_a_bend_cell_s_held_beam_does_not_depend_on_the_incoming_dispersion():
    lat, period = _periodic(cell=_arc)
    M = _cell_matrix(lat, period)
    D = np.linalg.solve(np.eye(4) - M[0:4, 0:4], M[0:4, 5])
    a = _solve(lat, period, ARC_INITIAL)
    b = _solve(lat, period, dict(ARC_INITIAL, disp_x=D[0], disp_xp=D[1]))
    for got in (a, b):
        assert got["converged"] and np.linalg.eigvalsh(got["sigma_entry"]).min() > 0.0
    A = a["sigma_entry"]
    assert np.abs(b["sigma_entry"] - A).max() < 1e-6 * np.abs(A).max()


def _matched_twiss(M, scale=1.0):
    """The Twiss of an uncoupled cell's matched beam, with β scaled (a mismatch)."""
    out = {}
    for name, (i, j) in (("x", (0, 1)), ("y", (2, 3))):
        m = M[i:j + 1, i:j + 1]
        c = 0.5 * (m[0, 0] + m[1, 1])
        s = np.sign(m[0, 1]) * np.sqrt(1.0 - c * c)
        out[f"beta_{name}"] = m[0, 1] / s * scale
        out[f"alpha_{name}"] = (m[0, 0] - m[1, 1]) / (2.0 * s)
    return out


@pytest.mark.parametrize("sigma_phi", [25.0, 60.0])
def test_a_long_bunch_does_not_loosen_the_tolerance_of_the_beam_solved_for(sigma_phi):
    """The bunch-length entry (deg²) is held, so it does not count in the scale of the residual:
    a long bunch and an incoming beam 1 % off the matched one converge to the matched beam, not to
    the incoming one (with the held block in the scale, the search stopped after one pass, 1e-3 to
    1e-2 off)."""
    lat, period = _periodic(cell=_arc)
    M = _cell_matrix(lat, period)
    spread = 2e-3                                           # MeV
    initial = dict(INITIAL, **_matched_twiss(M, 1.01), alpha_z=0.0,
                   emit_z=sigma_phi * spread, beta_z=sigma_phi / spread)
    got = _solve(lat, period, initial, current=0.0)
    S = got["sigma_entry"]
    assert got["converged"]
    want, _ = _matched_at_zero_current(M, INITIAL["emit_x"], INITIAL["emit_y"], S[5, 5])
    assert np.abs(S[0:4, 0:4] - want).max() < 1e-6 * np.abs(want).max()


def test_a_plane_of_zero_emittance_stays_zero():
    """No mode holds its size: the plane is rebuilt as zero, not left to wherever the search
    goes."""
    lat, period = _periodic(cell=_arc)
    M = _cell_matrix(lat, period)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = find_matched_period_sigma(lat, _ref(), period, 0.0, dict(ARC_INITIAL, emit_y=0.0))
    S = got["sigma_entry"]
    assert got["converged"] and not [w for w in rec if issubclass(w.category, RuntimeWarning)]
    assert np.abs(S[2:4, :]).max() < 1e-12 * S[0, 0]
    want, _ = _matched_at_zero_current(M, INITIAL["emit_x"], 0.0, S[5, 5])
    assert np.abs(S[0:4, 0:4] - want).max() < 1e-6 * np.abs(want).max()


def test_a_bunch_without_energy_spread_is_a_beam():
    """Its held block is singular, which is not a negative eigenvalue."""
    lat, period = _periodic()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = find_matched_period_sigma(lat, _ref(), period, 0.0, dict(INITIAL, emit_z=0.0))
    assert got["converged"] and got["longitudinal"] == "held"
    assert not [w for w in rec if issubclass(w.category, RuntimeWarning)]


def test_a_coupled_incoming_beam_gets_the_matched_beam_of_an_uncoupled_cell():
    """A solenoid ahead of the arc couples x and y: the held beam is the cell's uncoupled matched
    beam with the incoming beam's two normal-mode emittances.  Paired with the modes by overlap,
    the coupled beam's targets overlapped both of the cell's modes nearly equally and the search
    stalled 3e-4 off."""
    lat, period = _periodic(cell=_arc, prefix=[Solenoid("S", length=500.0, field=5.13)])
    initial = dict(ARC_INITIAL, emit_x=0.3, emit_y=0.1)
    got = _solve(lat, period, initial, current=0.0)
    S = got["sigma_entry"]
    assert got["converged"]
    seed = make_period_sigma_map(lat, _ref(), period, 0.0, dict(initial))[1]["sigma_seed"]
    assert np.abs(seed[0:2, 2:4]).max() > 0.1 * np.sqrt(seed[0, 0] * seed[2, 2])   # coupled
    M = _cell_matrix(lat, period)
    off = min(np.abs(S[0:4, 0:4] - _matched_at_zero_current(M, a, b, S[5, 5])[0]).max()
              for a, b in ((0.3, 0.1), (0.1, 0.3)))
    assert off < 1e-6 * np.abs(S[0:4, 0:4]).max()


def test_a_cell_with_rf_iterates_its_bunch():
    """A gap that bunches the beam: the search iterates the bunch, as before (on this cell it ends
    a little above its tolerance, as it did before)."""
    lat, period = _periodic(extra=(lambda: RFGap("G", voltage=0.5, phase=-30.0,
                                                 frequency=352.21, ttf=1.0),))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = find_matched_period_sigma(lat, _ref(), period, 5.0, dict(INITIAL))
    assert got["longitudinal"] == "iterate" and got["residual"] < 1e-5
    assert got["longitudinal_held"] is None and got["sigma_model_approx"] is None
    assert not [w for w in rec if "no matched bunch" in str(w.message)]
    seed = make_period_sigma_map(lat, _ref(), period, 5.0, dict(INITIAL))[1]["sigma_seed"]
    S = got["sigma_entry"]
    assert abs(S[4, 4] - seed[4, 4]) > 1e-3 * seed[4, 4]          # the bunch was solved for


def test_a_dc_beam_iterates_as_before():
    lat, period = _periodic()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = find_matched_period_sigma(lat, _ref(), period, 5.0, dict(INITIAL, continuous=True))
    assert got["longitudinal"] == "iterate" and got["longitudinal_held"] is None
    assert not [w for w in rec if "no matched bunch" in str(w.message)]


def test_a_matrix_element_whose_energy_depends_on_the_phase_is_not_held():
    """Not an RF element, but its map makes the energy depend on the phase."""
    M = np.eye(6)
    M[5, 4] = 1e-4                   # the sign of a bunching gap (+4.4e-2 for the -30° gap here)
    lat, period = _periodic(extra=(lambda: MatrixElement("M", M),))
    assert not periodic._no_longitudinal_focusing(lat, period, _ref())


def test_an_rf_gap_on_crest_is_not_held_although_its_map_has_no_phase_dependence():
    lat, period = _periodic(extra=(lambda: _rf_gap(0.0),))
    assert _cell_matrix(lat, period)[5, 4] == 0.0       # the map alone would say "no RF"
    assert not periodic._no_longitudinal_focusing(lat, period, _ref())


def test_a_cell_whose_map_cannot_be_computed_is_not_held(monkeypatch):
    import linac_gen.tracking.matrix_tracking as matrix_tracking
    lat, period = _periodic()
    assert periodic._no_longitudinal_focusing(lat, period, _ref())

    def _no_map(*args, **kwargs):
        raise RuntimeError("no map")

    monkeypatch.setattr(matrix_tracking, "compute_transfer_matrix", _no_map)
    assert not periodic._no_longitudinal_focusing(lat, period, _ref())


def test_the_cell_examined_is_the_period_s_own():
    """A period that starts after an RF cell is held; one whose own cell holds RF is not."""
    lat, period = _periodic(prefix=[_rf_gap(), *_fodo("p")])
    assert periodic._no_longitudinal_focusing(lat, period, _ref())
    lat, period = _periodic(extra=(_rf_gap,), prefix=_fodo("p"))
    assert not periodic._no_longitudinal_focusing(lat, period, _ref())


def test_a_search_that_does_not_converge_warns_and_counts_its_passes(monkeypatch):
    real = periodic.make_period_sigma_map
    passes = [0]

    def _counted(*args, **kwargs):
        sigma_map, state = real(*args, **kwargs)

        def _map(S):
            passes[0] += 1
            return sigma_map(S)
        return _map, state

    monkeypatch.setattr(periodic, "make_period_sigma_map", _counted)
    lat, period = _periodic()
    with pytest.warns(RuntimeWarning, match="did not converge") as rec:
        got = find_matched_period_sigma(lat, _ref(), period, 5.0, dict(INITIAL, continuous=True),
                                        tol=1e-30)
    assert not got["converged"] and passes[0] > got["n_iter"]
    message = next(str(w.message) for w in rec if "did not converge" in str(w.message))
    assert f"after {passes[0]} passes of the cell" in message


def test_the_warning_marks_exactly_the_searches_that_did_not_converge(monkeypatch):
    """With its residual just above the tolerance the search warns; just below, it does not.  The
    polish is stood in for by one that ends where it starts with a residual of 1.5e-6, so the
    residual does not depend on where the real one stops (its last bits vary from run to run)."""
    import types

    import scipy.optimize

    def _polish(fun, x0, **kwargs):
        return types.SimpleNamespace(x=np.asarray(x0, float), fun=np.array([1.5e-6, -1e-7]), nfev=1)

    monkeypatch.setattr(scipy.optimize, "least_squares", _polish)
    lat, period = _periodic()
    dc = dict(INITIAL, continuous=True)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        above = find_matched_period_sigma(lat, _ref(), period, 5.0, dict(dc), tol=1e-6)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        below = find_matched_period_sigma(lat, _ref(), period, 5.0, dict(dc), tol=2e-6)
    assert not [w for w in rec if "did not converge" in str(w.message)]
    assert not above["converged"] and below["converged"]
    assert above["residual"] == below["residual"] == 1.5e-6


def test_a_held_beam_with_a_negative_eigenvalue_is_reported_as_not_converged(monkeypatch):
    """A safety net: should the renormalized betatron block lose a mode, the held beam is not a
    beam, and the search says so rather than report it converged.  In a cell with bends the
    dispersion's share can keep the transverse block positive definite while the beam is not,
    so the check is on the betatron block."""
    lat, period = _periodic(cell=_arc)
    good = _solve(lat, period, ARC_INITIAL, current=0.0)["sigma_entry"]
    c, s55 = good[0:4, 5], good[5, 5]
    B = good[0:4, 0:4] - np.outer(c, c) / s55
    u = c / np.linalg.norm(c)
    bad = B - (u @ B @ u + 1e-2 * np.diag(B).max()) * np.outer(u, u)
    assert np.linalg.eigvalsh(bad).min() < 0.0
    assert np.linalg.eigvalsh(bad + np.outer(c, c) / s55).min() > 0.0   # Σ_T alone looks fine
    real = periodic._renormalize_eigenemittances
    monkeypatch.setattr(periodic, "_renormalize_eigenemittances",
                        lambda S, t, **k: bad.copy() if np.shape(S) == (4, 4) else real(S, t, **k))
    with pytest.warns(RuntimeWarning, match="negative eigenvalue"):
        got = find_matched_period_sigma(lat, _ref(), period, 0.0, dict(ARC_INITIAL))
    assert not got["converged"] and got["residual"] < 1e-6


def test_the_matched_channel_phase_advance_says_what_was_held():
    from linac_gen.analysis.phase_advance import channel_phase_advance_matched
    lat, period = _periodic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = channel_phase_advance_matched(lat, _ref(), period, current=5.0,
                                            base_initial=dict(INITIAL))
    ms = out["matched_state"]
    assert ms["longitudinal"] == "held" and "no RF" in ms["longitudinal_held"]
    assert ms["sigma_model_approx"] == "held-longitudinal"


def test_the_footprint_says_what_its_matched_beam_held(monkeypatch):
    """The footprint's model line, shown under the GUI plot, says when the matched beam held the
    bunch length and energy spread, and when the search did not converge; it says neither when
    neither happened."""
    from linac_gen.analysis.footprint import tune_footprint

    lat, period = _periodic(n_cells=2)
    with pytest.warns(UserWarning, match="no matched bunch"):
        out = tune_footprint(lat, _ref(), period, dict(INITIAL), 5.0, n_turns=32, n_particles=12)
    ms = out["matched_state"]
    assert ms["converged"] and ms["longitudinal_held"]
    assert ms["sigma_model_approx"] == "held-longitudinal"
    assert "bunch length and energy spread held" in out["model"]
    assert "NOT converged" not in out["model"]

    real = periodic.find_matched_period_sigma
    for change, note, shown in (({"converged": False}, "matched Σ NOT converged", True),
                                ({"longitudinal_held": None}, "energy spread held", False)):
        monkeypatch.setattr(periodic, "find_matched_period_sigma",
                            lambda *a, _c=change, **k: dict(real(*a, **k), **_c))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = tune_footprint(lat, _ref(), period, dict(INITIAL), 5.0, n_turns=32,
                                 n_particles=12)
        assert (note in out["model"]) is shown, (change, out["model"])
