"""StepConfig.drift_single_push: analytic losses inside a field-free drift.

With the option on, a drift is pushed once and every particle that leaves the
pipe inside it is located on its straight line — exact s, coordinates on the
wall — instead of at the end of the sub-step bundle that first saw it outside.
The sub-stepped walk (option off) is the reference: same lost particles, loss
positions within one bundle, survivors identical.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.step_config import StepConfig
from linac_gen.elements.drift import Drift
from linac_gen.tracking.tracker import Tracker

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _ref():
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)


def _one(x=0.0, xp=0.0, y=0.0, yp=0.0, dphi=0.0, dw=0.0):
    beam = Beam(ref=_ref(), n_particles=1, current=0.0)
    beam.continuous = False
    beam.particles[0] = [x, xp, y, yp, dphi, dw]
    return beam


def _lat(single_push, aperture=1.0, aperture_y=None, L=200.0, dx=0.0, tilt=0.0, second=True):
    lat = Lattice()
    lat.step_config = StepConfig(drift_single_push=single_push)
    lat.add(Drift("D1", length=L, aperture=aperture, aperture_y=aperture_y, dx=dx, tilt_deg=tilt))
    if second:
        lat.add(Drift("D2", length=100.0, aperture=aperture, aperture_y=aperture_y))
    return lat


def _run(beam, lat):
    Tracker(lat, beam).run()
    lt = beam.loss_table
    return (None if len(lt) == 0 else lt[0]), beam.particles[0].copy()


def _m45(ref, s):
    return -360.0 * s / (ref.beta ** 3 * ref.gamma ** 3 * ref.species.mass * ref.wavelength)


# ---------------------------------------------------------------- single particles
def test_circular_crossing_is_exact():
    """x' = 10 mrad from the axis in a 1 mm pipe: the wall is reached at s = 100 mm."""
    loss, p = _run(_one(xp=10.0, dw=0.01), _lat(True))
    assert loss is not None and loss["element_name"] == "D1"
    assert loss["s"] == pytest.approx(100.0, abs=1e-9)
    assert loss["x"] == pytest.approx(1.0, abs=1e-12) and loss["y"] == 0.0
    assert loss["energy"] == pytest.approx(3.0 + 0.01)
    # frozen on the wall, angles and energy untouched, phase advanced over 100 mm
    assert p[0] == pytest.approx(1.0, abs=1e-12) and p[1] == 10.0 and p[5] == 0.01
    assert p[4] == pytest.approx(_m45(_ref(), 100.0) * 0.01, rel=1e-12)
    # the sub-stepped reference records the bundle end (20 mm bundles at 100/50)
    loss_off, _ = _run(_one(xp=10.0, dw=0.01), _lat(False))
    assert loss_off["s"] == pytest.approx(120.0) and 0.0 <= loss_off["s"] - loss["s"] <= 20.0


def test_offset_and_angle_in_the_other_plane():
    loss, p = _run(_one(x=0.6, yp=8.0), _lat(True))
    assert loss["s"] == pytest.approx(100.0, rel=1e-12)
    assert loss["x"] == pytest.approx(0.6) and loss["y"] == pytest.approx(0.8, rel=1e-12)
    assert loss["x"] ** 2 + loss["y"] ** 2 == pytest.approx(1.0, rel=1e-12)


def test_rectangular_pipe_takes_the_earlier_plane():
    loss, _ = _run(_one(xp=10.0, yp=10.0), _lat(True, aperture=1.0, aperture_y=0.5))
    assert loss["s"] == pytest.approx(50.0, rel=1e-12)
    assert loss["x"] == pytest.approx(0.5) and loss["y"] == pytest.approx(0.5)


@pytest.mark.parametrize("kw", [dict(x=1.5), dict(x=1.0, xp=5.0), dict(x=1.0, yp=5.0)])
def test_outside_or_on_the_wall_and_leaving_is_lost_at_entry(kw):
    """Outside at entry, or on the wall and leaving or tangential: lost at s_entry
    with the entry coordinates (the sub-stepped walk reported the first bundle end)."""
    beam = _one(**kw)
    x_entry = beam.particles[0, 0]
    loss, p = _run(beam, _lat(True))
    assert loss["s"] == 0.0 and loss["x"] == x_entry and p[0] == x_entry
    loss_off, _ = _run(_one(**kw), _lat(False))
    assert loss_off["s"] == pytest.approx(20.0)


@pytest.mark.parametrize("kw", [dict(x=1.0, xp=-5.0), dict(xp=0.0), dict(xp=1.0), dict(x=0.999, xp=0.0, yp=0.0)])
def test_entering_parallel_or_never_reaching_the_wall_survives(kw):
    loss, p_on = _run(_one(**kw), _lat(True))
    assert loss is None
    _, p_off = _run(_one(**kw), _lat(False))
    np.testing.assert_allclose(p_on, p_off, rtol=0, atol=1e-12)


def test_survivor_coordinates_equal_the_plain_drift_map():
    beam = _one(x=0.2, xp=1.0, y=-0.1, yp=0.5, dphi=2.0, dw=0.01)
    expected = beam.particles[0].copy()
    d = Drift("D1", length=200.0, aperture=1.0)
    ref = _ref()
    expected = d.transfer_matrix(ref, ds=200.0) @ expected
    loss, p = _run(beam, _lat(True, second=False))
    assert loss is None
    assert np.array_equal(p, expected)


def test_downstream_drift_does_not_record_again():
    beam = _one(xp=10.0)
    _run(beam, _lat(True))
    assert len(beam.loss_table) == 1 and beam.n_alive == 0


def test_misaligned_drift_records_wall_in_the_element_frame():
    """dx = 2 mm shifts the pipe: a particle at lab x = 2.5 moving outward hits the
    wall x = +1 (element frame) at 50 mm; the sub-stepped walk agrees to one bundle."""
    loss_on, _ = _run(_one(x=2.5, xp=10.0), _lat(True, dx=2.0))
    loss_off, _ = _run(_one(x=2.5, xp=10.0), _lat(False, dx=2.0))
    assert loss_on["s"] == pytest.approx(50.0, rel=1e-12) and loss_on["x"] == pytest.approx(1.0, rel=1e-12)
    # sub-stepped: first bundle end past the wall (60 mm, x = 1.1 in the element frame)
    assert loss_off["s"] == pytest.approx(60.0) and loss_off["x"] == pytest.approx(1.1)
    assert 0.0 <= loss_off["s"] - loss_on["s"] <= 20.0


def test_tilted_rectangular_pipe_uses_the_rotated_frame():
    """A 90° tilted rectangle (a = 1 along x, b = 0.5 along y) seen by a particle
    moving in lab y: in the element frame that motion is along x, so the wall
    is the a = 1 half-width, reached at 100 mm."""
    loss_on, _ = _run(_one(yp=10.0), _lat(True, aperture=1.0, aperture_y=0.5, tilt=90.0))
    loss_off, _ = _run(_one(yp=10.0), _lat(False, aperture=1.0, aperture_y=0.5, tilt=90.0))
    assert loss_on["s"] == pytest.approx(100.0, rel=1e-9)
    assert 0.0 <= loss_off["s"] - loss_on["s"] <= 20.0


# ---------------------------------------------------------------- whole decks
def _bundle_len_mm(cfg, length_mm):
    n_int = cfg.integration_steps_for_length_mm(length_mm)
    n_sc = cfg.sc_steps_for_length_mm(length_mm)
    ds = length_mm / n_int
    sc_every = max(1, n_int // n_sc)
    n_bundles = n_int // sc_every
    return max(sc_every * ds, length_mm - n_bundles * sc_every * ds)


def _big_beam(lat_cfg, seed=42, n=4000):
    from linac_gen.distributions.factory import create_beam
    cfg = BeamConfig(species="H-", energy=2.1226695, frequency=162.5, current=0.0, n_particles=n,
                     distribution="gaussian", emit_nx=0.25 * 25, emit_ny=0.25 * 25, emit_z=0.3,
                     alpha_x=0.0, beta_x=0.5, alpha_y=0.0, beta_y=0.5, alpha_z=0.0, beta_z=1.0)
    return create_beam(cfg, seed=seed)


@pytest.mark.parametrize("deck, narrow_drifts", [
    ("fodo_cell.dat", False), ("halo_fodo.dat", False), ("bend_line.dat", False),
    ("fodo_cell.dat", True), ("halo_fodo.dat", True),      # drift pipes narrower than the magnets
])
def test_deck_losses_match_the_substepped_walk(deck, narrow_drifts):
    """Same particles lost; each loss located within one sub-step bundle of the
    sub-stepped record, on the wall, inside its element; survivors and the rms
    record identical to 1e-12.  The example decks lose particles in their
    magnets (smaller apertures than the drifts); the narrow-drift variants put
    the losses inside the drifts, the case this option changes."""
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.core.simulation import Simulation
    results = {}
    for flag in (True, False):
        lat, _ = parse_tracewin(str(EXAMPLES / deck))
        if narrow_drifts:
            for e in lat.elements:
                if isinstance(e, Drift) and e.aperture > 0:
                    e.aperture = 0.4 * e.aperture
        lat.step_config = dataclasses.replace(lat.step_config, drift_single_push=flag)
        beam = _big_beam(lat)
        sim = Simulation(lat, beam, space_charge="off")
        rec = sim.run()
        results[flag] = (lat, beam, rec)
    lat_on, b_on, r_on = results[True]
    lat_off, b_off, r_off = results[False]
    assert b_on.loss_table.size > 0, "precondition: the oversized beam must lose particles"
    t_on = {int(r["particle_id"]): r for r in b_on.loss_table}
    t_off = {int(r["particle_id"]): r for r in b_off.loss_table}
    # element entry positions and apertures
    s_entry, elems = {}, {}
    s = 0.0
    for e in lat_on.elements:
        s_entry.setdefault(e.name, []).append(s)
        elems[e.name] = e
        s += e.length
    # Every particle the sub-stepped walk lost is lost analytically too (a straight
    # line through a convex pipe cannot be outside between two inside points).  The
    # analytic walk may lose MORE: a particle already outside the pipe when it
    # enters the drift (let through by a wider magnet upstream) is recorded at the
    # drift entrance, where the sub-stepped walk only looked at the first bundle
    # end and kept it if it had wandered back inside by then.
    assert set(t_off) <= set(t_on)
    entry_only = set(t_on) - set(t_off)
    for pid in entry_only:
        r = t_on[pid]
        e = elems[r["element_name"]]
        assert isinstance(e, Drift)
        assert any(abs(r["s"] - se) < 1e-9 for se in s_entry[e.name]), (pid, r["s"])
        assert r["x"] ** 2 + r["y"] ** 2 > e.aperture ** 2      # outside at entry, in the wall
    checked_drift = 0
    for pid, r in t_on.items():
        if pid in entry_only:
            continue
        ro = t_off[pid]
        e = elems[r["element_name"]]
        eo = elems[ro["element_name"]]
        if r["element_name"] != ro["element_name"]:
            assert isinstance(e, Drift)
            at_entry = any(abs(r["s"] - se) < 1e-9 for se in s_entry[e.name])
            if at_entry:
                # outside the pipe when it entered this drift (in the wall): recorded
                # there; the sub-stepped walk kept it while it was back inside at the
                # bundle ends and lost it further downstream
                assert r["x"] ** 2 + r["y"] ** 2 > e.aperture ** 2 or (e.aperture_y and e.aperture_y > 0)
                assert ro["s"] >= r["s"] - 1e-9
            else:
                # a crossing within rounding of the drift's end: recorded at the end of
                # this drift; the sub-stepped walk (composed pushes round differently)
                # in the first bundle of the next element
                assert any(abs(r["s"] - (se + e.length)) < 1e-6 for se in s_entry[e.name]), (pid, r["s"])
                assert any(abs(ro["s"] - se) <= _bundle_len_mm(lat_on.step_config, eo.length) + 1e-6
                           for se in s_entry[eo.name]), (pid, ro["s"])
            continue
        if isinstance(e, Drift):
            if any(abs(r["s"] - se) < 1e-9 for se in s_entry[e.name]) and \
                    (r["x"] ** 2 + r["y"] ** 2 > e.aperture ** 2 or (e.aperture_y and e.aperture_y > 0)):
                # outside at entry: the sub-stepped walk may have kept it through several
                # bundles (a line can re-enter a convex pipe once) and lost it later
                assert ro["s"] >= r["s"] - 1e-9
                continue
            tol = _bundle_len_mm(lat_on.step_config, e.length) + 1e-9
            assert -1e-9 <= ro["s"] - r["s"] <= tol, (pid, r["s"], ro["s"], tol)
            assert any(se - 1e-9 <= r["s"] <= se + e.length + 1e-9 for se in s_entry[e.name])
            if not (e.aperture_y and e.aperture_y > 0) and r["x"] ** 2 + r["y"] ** 2 <= e.aperture ** 2 * (1 + 1e-9):
                # inside at entry: the analytic loss sits on the wall
                assert r["x"] ** 2 + r["y"] ** 2 == pytest.approx(e.aperture ** 2, rel=1e-9)
                checked_drift += 1
        else:
            assert r["s"] == ro["s"]
        # angles and energy of a lost particle are what they were at the loss
        assert r["energy"] == pytest.approx(ro["energy"], rel=1e-12)
    if narrow_drifts:
        assert checked_drift > 0, "precondition: some losses must happen inside drifts"
    alive = ~b_on.lost & ~b_off.lost
    np.testing.assert_allclose(b_on.particles[alive], b_off.particles[alive], rtol=1e-12, atol=1e-12)
    if entry_only:
        # the extra entry losses are few and the rms record only moves by their removal
        assert len(entry_only) < 0.02 * b_on.particles.shape[0]
        return
    for k in ("sigma_x", "sigma_y", "emit_x", "emit_y"):
        np.testing.assert_allclose(np.asarray(getattr(r_on, k)), np.asarray(getattr(r_off, k)), rtol=1e-12, atol=1e-15)
    # x', y', dW of every lost particle are unchanged by the crossing itself; the two
    # runs differ only by the composition rounding of the upstream drift pushes
    lost = b_on.lost & b_off.lost
    np.testing.assert_allclose(b_on.particles[lost][:, [1, 3, 5]], b_off.particles[lost][:, [1, 3, 5]],
                               rtol=1e-12, atol=1e-12)


def test_non_finite_angles_do_not_warn_and_are_lost_at_the_exit_check():
    """A diverged particle (inf angle) takes no part in the crossing algebra: the
    crossing pass itself raises no RuntimeWarning and leaves the particle to the
    exit check, which loses it as the sub-stepped walk did."""
    import warnings
    for kw in (dict(xp=np.inf), dict(yp=-np.inf), dict(xp=1e200), dict(x=np.nan)):
        beam = _one(**kw)
        lat = _lat(True, second=False)
        tr = Tracker(lat, beam)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tr._freeze_analytic_drift_losses(lat.elements[0], 0.0)      # no warning, no raise
        assert beam.n_alive == 1                                         # left to the exit check
    # the element walk itself (crossing pass, push, exit check) gives the diverged
    # particle the same fate as the sub-stepped walk: a finite huge angle is lost at
    # the exit, an infinite one turns to NaN in the push and is kept by both (the
    # recorder is not involved: its eigen-emittances reject non-finite coordinates
    # in either mode)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for kw, expect_lost in ((dict(xp=1e200), True), (dict(xp=np.inf), False), (dict(yp=-np.inf), False)):
            fates = []
            for flag in (True, False):
                beam = _one(**kw)
                lat = _lat(flag, second=False)
                Tracker(lat, beam)._track_element(lat.elements[0])
                fates.append(beam.n_alive == 0)
            assert fates == [expect_lost, expect_lost], (kw, fates)


def test_loss_table_s_is_non_decreasing_with_analytic_losses():
    """Analytic losses inside one drift are recorded in order of s, so the loss
    table keeps the ordering the sub-stepped walk produced (readers that
    searchsorted on s keep working)."""
    beam = Beam(ref=_ref(), n_particles=200, current=0.0)
    beam.continuous = False
    rng = np.random.default_rng(5)
    beam.particles[:, 0] = rng.normal(0, 0.3, 200)
    beam.particles[:, 1] = rng.normal(0, 8.0, 200)          # large angles: many crossings
    beam.particles[:, 2] = rng.normal(0, 0.3, 200)
    beam.particles[:, 3] = rng.normal(0, 8.0, 200)
    lat = _lat(True)
    Tracker(lat, beam).run()
    s = beam.loss_table["s"]
    assert s.size > 20
    assert np.all(np.diff(s) >= 0)
