"""Halo action scan through real multi-particle ``Simulation`` runs.

* Opt-in and inert: ``record_action_scan=True`` changes NOTHING else —
  every recorder attribute and the final particles are bit-identical to
  the same run without it (apertures, space charge, sub-step records).
* Physics invariants: n = W/eps_rms is a Courant–Snyder invariant, so the
  counts are exactly constant through linear drifts; a FREQ x2 card
  doubles eps_z (deg*MeV) while the counts and the normalized emittance
  are unchanged; a bend with energy spread has a betatron emittance
  below the raw one.
* Both regimes of the DC flag and an all-lost beam stay aligned.
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.simulation import Simulation
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import Freq
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap


def _beam(n=2000, seed=11, w_kin=3.0, current=0.0, sig_w=0.004):
    ref = ReferenceParticle(species=PROTON, w_kin=w_kin, frequency=162.5)
    beam = Beam(ref=ref, n_particles=n, current=current)
    rng = np.random.default_rng(seed)
    p = rng.standard_normal((n, 6)) * np.array([1.5, 1.2, 1.5, 1.2, 8.0,
                                                 sig_w])
    p[:, 1] += 0.4 * p[:, 0]
    p[:, 3] -= 0.3 * p[:, 2]
    p[:, 5] += 0.0002 * p[:, 4]
    beam.particles[:] = p
    return beam


def _differences(a, b, path="", out=None):
    """Paths where two recorder values differ (bitwise, NaN == NaN)."""
    out = [] if out is None else out
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a2, b2 = np.asarray(a), np.asarray(b)
        if (a2.dtype != b2.dtype or a2.shape != b2.shape
                or not np.array_equal(a2, b2,
                                      equal_nan=a2.dtype.kind in "fc")):
            out.append(path)
    elif isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.append(f"{path}.{k}")
            else:
                _differences(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(path)
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _differences(x, y, f"{path}[{i}]", out)
    elif isinstance(a, float) and isinstance(b, float):
        if not (a == b or (a != a and b != b)):
            out.append(path)
    elif hasattr(a, "__dict__") and type(a) is type(b):
        _differences(vars(a), vars(b), path, out)
    elif a != b:
        out.append(path)
    return out


def test_option_changes_nothing_else_bitwise():
    def lattice():
        lat = Lattice()
        lat.add(Drift("D0", length=150.0, aperture=4.0))       # scrapes
        lat.add(Quadrupole("Q1", length=80.0, gradient=12.0, aperture=10.0))
        lat.add(Drift("D1", length=200.0, aperture=10.0))
        lat.add(Quadrupole("Q2", length=80.0, gradient=-12.0,
                           aperture=10.0))
        lat.add(RFGap("G1", voltage=0.3, phase=-30.0, frequency=162.5))
        lat.add(Drift("D2", length=150.0, aperture=10.0))
        return lat

    runs = []
    for on in (False, True):
        beam = _beam(current=5.0)
        rec = Simulation(lattice(), beam,
                         space_charge=SpaceChargeConfig(nx=12, ny=12, nz=12),
                         record_substeps=True,
                         record_action_scan=on).run()
        runs.append((rec, beam))
    (off, b_off), (on, b_on) = runs
    assert set(vars(on)) - set(vars(off)) == {"action_scan"}
    assert set(vars(off)) <= set(vars(on))
    diffs = []
    for key in vars(off):
        _differences(vars(off)[key], vars(on)[key], key, diffs)
    assert diffs == []
    np.testing.assert_array_equal(b_off.particles, b_on.particles)
    np.testing.assert_array_equal(b_off.lost, b_on.lost)
    # the scan itself: one row per record, losses seen
    n_rec = len(on.s)
    assert n_rec > 20                                # sub-step records
    assert all(len(v) == n_rec for k, v in on.action_scan.items() if k != "n")
    assert on.action_scan["n_alive"][0] == 2000
    assert on.action_scan["n_alive"][-1] == int((~b_on.lost).sum()) < 2000


def test_counts_exactly_constant_through_drifts():
    """Linear transport keeps n = W/eps_rms of every particle: the counts
    of the transverse planes are identical at every record."""
    lat = Lattice()
    for i in range(4):
        lat.add(Drift(f"D{i}", length=250.0))
    rec = Simulation(lat, _beam(seed=3), space_charge="off",
                     record_substeps=True, record_action_scan=True).run()
    sc = rec.action_scan
    assert len(rec.s) > 5
    for plane in ("x", "y", "x_raw", "y_raw"):
        for row in sc[f"count_{plane}"]:
            np.testing.assert_array_equal(row, sc[f"count_{plane}"][0])
        np.testing.assert_allclose(sc[f"eps_{plane}"], sc[f"eps_{plane}"][0],
                                   rtol=1e-12)
        np.testing.assert_allclose(sc[f"n_max_{plane}"],
                                   sc[f"n_max_{plane}"][0], rtol=1e-9)


def test_freq_jump_scales_eps_z_not_the_counts():
    lat = Lattice()
    lat.add(Drift("D0", length=100.0))
    lat.add(Freq("F2", frequency_mhz=325.0))
    lat.add(Drift("D1", length=100.0))
    rec = Simulation(lat, _beam(n=5000, seed=1), space_charge="off",
                     record_action_scan=True).run()
    sc = rec.action_scan
    i_before, i_card = rec.element_names.index("D0"), \
        rec.element_names.index("F2")
    np.testing.assert_array_equal(sc["count_z"][i_before],
                                  sc["count_z"][i_card])
    assert sc["eps_z"][i_card] == pytest.approx(2.0 * sc["eps_z"][i_before],
                                                rel=1e-12)
    assert sc["eps_n_z"][i_card] == pytest.approx(sc["eps_n_z"][i_before],
                                                  rel=1e-12)
    np.testing.assert_allclose(sc["eps_n_z"], rec.emit_nz, rtol=1e-12)


def test_dc_then_bunched_records_both_regimes():
    lat = Lattice()
    for i in range(3):
        lat.add(Drift(f"DA{i}", length=100.0, aperture=50.0))
    lat.add(RFGap("GAP", voltage=0.01, phase=-90.0, frequency=162.5))
    for i in range(3):
        lat.add(Drift(f"DB{i}", length=100.0, aperture=50.0))
    ref = ReferenceParticle(species=PROTON, w_kin=0.03, frequency=162.5)
    beam = Beam(ref=ref, n_particles=1500, current=5.0)
    beam.continuous = True
    rng = np.random.default_rng(7)
    beam.particles[:, 0] = rng.normal(0, 2.0, 1500)
    beam.particles[:, 1] = rng.normal(0, 5.0, 1500)
    beam.particles[:, 2] = rng.normal(0, 2.0, 1500)
    beam.particles[:, 3] = rng.normal(0, 5.0, 1500)
    beam.particles[:, 4] = rng.uniform(-180.0, 180.0, 1500)
    beam.particles[:, 5] = rng.normal(0, 1e-5, 1500)
    rec = Simulation(lat, beam,          # DC kernel, then PIC once bunched
                     space_charge=SpaceChargeConfig(nx=12, ny=12, nz=12),
                     record_action_scan=True).run()
    sc = rec.action_scan
    assert True in rec.continuous_at and False in rec.continuous_at
    assert all(len(v) == len(rec.s) for k, v in sc.items() if k != "n")
    for i, dc in enumerate(rec.continuous_at):
        if sc["n_alive"][i] >= 3:
            assert sc["eps_z"][i] > 0.0                 # recorded either way
            assert sc["count_x"][i][0] == sc["n_alive"][i]


def test_all_lost_run_stays_aligned():
    lat = Lattice()
    lat.add(Drift("D0", length=100.0, aperture=10.0))
    lat.add(Drift("KILL", length=100.0, aperture=0.01))
    lat.add(Drift("D2", length=100.0, aperture=10.0))
    beam = _beam(n=500, seed=4)
    rec = Simulation(lat, beam, space_charge="off",
                     record_action_scan=True).run()
    sc = rec.action_scan
    assert bool(beam.lost.all())
    assert all(len(v) == len(rec.s) for k, v in sc.items() if k != "n")
    assert sc["n_alive"][0] == 500 and sc["n_alive"][-1] == 0
    assert not sc["count_x"][-1].any() and sc["eps_x"][-1] == 0.0


def test_bend_betatron_emittance_below_raw():
    """With energy spread a bend adds D*delta to x: the betatron (Schur)
    emittance stays below the projected one downstream of it."""
    lat = Lattice()
    lat.add(Drift("D0", length=100.0))
    lat.add(Dipole("B1", angle=20.0, rho=1500.0))
    lat.add(Drift("D1", length=500.0))
    rec = Simulation(lat, _beam(n=4000, seed=5, sig_w=0.03),
                     space_charge="off", record_action_scan=True).run()
    sc = rec.action_scan
    end = len(rec.s) - 1
    assert sc["eps_x"][end] < 0.9 * sc["eps_x_raw"][end]
    np.testing.assert_allclose(sc["eps_x_raw"], rec.emit_x, rtol=1e-10)
