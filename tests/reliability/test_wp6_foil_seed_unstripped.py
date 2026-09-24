"""WP6 — the foil for leg D: seeded and reproducible, unchanged kicks at
defaults, two-step stripping on the two anchors, a separate unstripped
record (not a loss), offset + extent → missed fraction, the Bethe mean
loss against an independent formula, HDF5 persistence, deck-line round
trip with defaults byte-identical."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from linac_gen.cli import common
from linac_gen.core.beam import UNSTRIPPED_DTYPE, Beam
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import H_MINUS, PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.foil import _MATERIALS, Foil
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.io.tracewin_writer import write_tracewin

REPO = Path(__file__).resolve().parents[2]


def _beam(species=H_MINUS, w=800.0, n=20000, sigma_mm=1.0, seed=0):
    ref = ReferenceParticle(species=species, w_kin=w, frequency=162.5)
    beam = Beam(ref=ref, n_particles=n, current=0.0)
    rng = np.random.default_rng(seed)
    beam.particles[:, 0] = rng.normal(0.0, sigma_mm, n)
    beam.particles[:, 2] = rng.normal(0.0, sigma_mm, n)
    return beam


def test_seed_property_reproducible_and_new_draws_come_after_the_old_ones():
    a = _beam(); b = _beam(); c = _beam(); d = _beam()
    Foil("F", "C", 600.0, seed=42).apply_kick(a)
    Foil("F", "C", 600.0, seed=42, strip_model="two_step").apply_kick(b)
    Foil("F", "C", 600.0, seed=43).apply_kick(c)
    f = Foil("F", "C", 600.0); f.seed = 42; f.apply_kick(d)
    assert np.array_equal(a.particles, b.particles)       # stripping draws AFTER the kicks
    assert np.array_equal(a.particles, d.particles)       # the property rebuilds the generator
    assert not np.array_equal(a.particles, c.particles)
    assert a.unstripped_table.size == 0 and b.unstripped_table.size > 0
    assert Foil("F").seed is None and f.seed == 42 and isinstance(f.seed, int)
    lat = Lattice(); lat.add(Foil("FOIL", "C", 600.0))
    common.apply_element_override(lat, "FOIL.seed", "7")   # None slot -> float -> int
    assert lat.elements[0].seed == 7
    for bad in (dict(strip_model="x"), dict(dedx_model="y"), dict(extent_mm=-1.0)):
        with pytest.raises(ValueError):
            Foil("F", "C", 600.0, **bad)


def test_two_step_stripping_reproduces_both_anchors_and_is_monotonic():
    f = Foil("F", "C", 600.0)
    ref = ReferenceParticle(species=H_MINUS, w_kin=800.0, frequency=162.5)
    assert f.stripping_fractions(ref, 600.0)[0] == pytest.approx(0.99956, rel=1e-9)
    assert f.stripping_fractions(ref, 380.0)[0] == pytest.approx(0.991, rel=1e-9)
    fr = [f.stripping_fractions(ref, x) for x in (0.0, 100.0, 200.0, 380.0, 600.0, 800.0)]
    assert fr[0] == (0.0, 0.0, 1.0)
    assert all(abs(sum(t) - 1.0) < 1e-12 for t in fr)
    assert all(fr[i][0] < fr[i + 1][0] for i in range(len(fr) - 1))
    assert f.stripping_fractions(ReferenceParticle(species=PROTON, w_kin=800.0, frequency=162.5)) == (1.0, 0.0, 0.0)
    slow = ReferenceParticle(species=H_MINUS, w_kin=400.0, frequency=162.5)
    assert f.stripping_fractions(slow, 380.0)[0] > 0.991              # 1/beta^2: thicker at low energy
    assert Foil("A", "Al", 600.0).strip_cross_sections_cm2(ref)[0] > f.strip_cross_sections_cm2(ref)[0]


def test_two_step_records_the_unstripped_populations_without_losing_them():
    beam = _beam(n=50000)
    f = Foil("F", "C", 380.0, seed=3, strip_model="two_step")
    f.apply_kick(beam)
    ut = beam.unstripped_table
    assert ut.dtype == UNSTRIPPED_DTYPE
    n = beam.n_particles
    f_p, f_h0, f_hm = f.stripping_fractions(beam.ref)
    n_h0 = int(np.count_nonzero(ut["state"] == "H0"))
    n_hm = int(np.count_nonzero(ut["state"] == "H-"))
    assert abs(n_h0 - f_h0 * n) < 4.0 * math.sqrt(f_h0 * n) + 1
    assert n_hm <= 3                                                    # 1.9e-6 x 50000
    assert set(ut["element_name"]) == {"F"} and np.all(ut["energy"] < 800.0)
    assert beam.n_alive == n and beam.loss_table.size == 0               # not a loss
    prot = _beam(species=PROTON)
    Foil("F", "C", 380.0, seed=3, strip_model="two_step").apply_kick(prot)
    assert prot.unstripped_table.size == 0                              # inert for protons


def test_offset_and_extent_give_the_missed_fraction_and_no_kick_on_misses():
    from scipy.special import erf
    n, sig = 60000, 1.0
    ex, ey, dx = 2.0, 3.0, 1.0
    beam = _beam(n=n, sigma_mm=sig, seed=11)
    # the tracker translates the beam into the foil frame; do it by hand here
    beam.particles[:, 0] -= dx
    before = beam.particles.copy()
    f = Foil("F", "C", 600.0, seed=5, dx=dx, extent_mm=(ex, ey))
    f.apply_kick(beam)
    ut = beam.unstripped_table
    missed = ut[ut["state"] == "missed"]
    p_in_x = 0.5 * (erf((ex + dx) / (sig * math.sqrt(2))) + erf((ex - dx) / (sig * math.sqrt(2))))
    p_in_y = erf(ey / (sig * math.sqrt(2)))
    p_miss = 1.0 - p_in_x * p_in_y
    assert abs(len(missed) / n - p_miss) < 4.0 * math.sqrt(p_miss * (1 - p_miss) / n)
    ids = missed["particle_id"]
    assert np.array_equal(beam.particles[ids], before[ids])             # untouched
    hit = np.setdiff1d(np.arange(n), ids)
    assert not np.array_equal(beam.particles[hit, 1], before[hit, 1])
    assert f.extent_mm == (2.0, 3.0) and Foil("G", extent_mm=1.5).extent_mm == (1.5, 1.5)


def test_bethe_mean_loss_matches_an_independent_formula_and_mip_is_unchanged():
    K, m_e = 0.307075, 0.51099895
    mat = _MATERIALS["C"]
    for species, w in ((PROTON, 800.0), (H_MINUS, 800.0), (PROTON, 23.7)):
        ref = ReferenceParticle(species=species, w_kin=w, frequency=162.5)
        g = 1.0 + w / species.mass
        b2 = 1.0 - 1.0 / (g * g)
        bg2 = b2 * g * g
        mr = m_e / species.mass
        t_max = 2.0 * m_e * bg2 / (1.0 + 2.0 * g * mr + mr * mr)
        i_mev = mat["I_eV"] * 1e-6
        dedx = K * (mat["Z"] / mat["A"]) / b2 * (0.5 * math.log(2.0 * m_e * bg2 * t_max / i_mev ** 2) - b2)
        want = dedx * 600e-6
        got = Foil("F", "C", 600.0, dedx_model="bethe")._mean_energy_loss_MeV(ref)
        assert got == pytest.approx(want, rel=1e-6), (species.name, w)
        mip = Foil("F", "C", 600.0)._mean_energy_loss_MeV(ref)
        assert mip == pytest.approx(mat["dEdx_min_MeVcm2_g"] * 600e-6, rel=1e-12)
    ref = ReferenceParticle(species=PROTON, w_kin=800.0, frequency=162.5)
    assert Foil("F", "C", 600.0, dedx_model="bethe")._mean_energy_loss_MeV(ref) > Foil("F", "C", 600.0)._mean_energy_loss_MeV(ref)
    # the envelope reference drop follows the model
    from linac_gen.core.config import BeamConfig
    for model in ("mip", "bethe"):
        lat = Lattice(); lat.add(Drift("d", 10.0)); lat.add(Foil("F", "C", 600.0, dedx_model=model)); lat.add(Drift("e", 10.0))
        cfg = BeamConfig(species="H-", energy=800.0, frequency=162.5, current=0.0, n_particles=100)
        res = common.run_envelope_sim(lat, cfg)
        drop = res.ref_w_kin[0] - res.ref_w_kin[-1]
        assert drop == pytest.approx(lat.elements[1]._mean_energy_loss_MeV(
            ReferenceParticle(species=H_MINUS, w_kin=800.0, frequency=162.5)), rel=1e-9)


def test_hdf5_round_trip_and_default_files_carry_no_group(tmp_path):
    import h5py
    from linac_gen.core.config import BeamConfig
    from linac_gen.io.hdf5_output import load_results_hdf5
    def run(foil):
        lat = Lattice(); lat.add(Drift("d", 100.0)); lat.add(foil); lat.add(Drift("e", 100.0))
        cfg = BeamConfig(species="H-", energy=800.0, frequency=162.5, current=0.0,
                         n_particles=4000, beta_x=5.0, beta_y=5.0, emit_nx=2.0, emit_ny=2.0)
        sc = common.make_sc_config(cfg, {}, {}); st = common.make_step_config({}, {})
        rec, beam = common.run_mp_sim(lat, cfg, sc, st, seed=1)
        return rec, beam, cfg, lat
    rec, beam, cfg, lat = run(Foil("F", "C", 380.0, seed=2, strip_model="two_step", extent_mm=(3.0, 3.0)))
    assert rec.unstripped_table.size == beam.unstripped_table.size > 0
    out = tmp_path / "strip.h5"
    common.write_results(rec, str(out), "hdf5", cfg, lat)
    back = load_results_hdf5(str(out))
    ut = back["unstripped_table"]
    assert ut.dtype == UNSTRIPPED_DTYPE and len(ut) == len(rec.unstripped_table)
    assert set(ut["state"]) <= {"H0", "H-", "missed"} and "missed" in set(ut["state"])
    assert np.array_equal(ut["particle_id"], rec.unstripped_table["particle_id"])
    assert rec.transmission[-1] == 100.0
    rec0, _b, cfg0, lat0 = run(Foil("F", "C", 380.0, seed=2))
    out0 = tmp_path / "plain.h5"
    common.write_results(rec0, str(out0), "hdf5", cfg0, lat0)
    with h5py.File(out0, "r") as f:
        assert "unstripped" not in f
    assert "unstripped_table" not in load_results_hdf5(str(out0))


def test_deck_line_round_trip_and_defaults_byte_identical(tmp_path):
    lat = Lattice(); lat.add(Drift("d", 100.0))
    lat.add(Foil("F1", "C", 600.0))
    lat.add(Foil("F2", "C", 435.0, straggling="gaussian", strip_model="two_step",
                 dedx_model="bethe", dx=1.0, dy=-0.5, extent_mm=(2.5, 4.0), seed=7))
    lat.add(Foil("F3", "Be", 300.0, extent_mm=2.0))
    out = tmp_path / "foil.dat"
    write_tracewin(lat, str(out), frequency=162.5)
    lines = [ln.rstrip("\n") for ln in out.read_text(encoding="utf-8").splitlines()]
    assert "; HELIX_FOIL F1 C 600.0" in lines
    assert ("; HELIX_FOIL F2 C 435.0 gaussian strip_model=two_step dedx_model=bethe "
            "dx=1.0 dy=-0.5 extent_mm=2.5,4.0 seed=7") in lines
    assert "; HELIX_FOIL F3 Be 300.0 extent_mm=2.0,2.0" in lines
    lat2, meta = parse_tracewin(str(out))
    assert meta.get("warnings") == []
    f1, f2, f3 = [e for e in lat2.elements if isinstance(e, Foil)]
    assert (f1.strip_model, f1.dedx_model, f1.dx, f1.dy, f1.extent_mm, f1.seed) == ("off", "mip", 0.0, 0.0, None, None)
    assert (f2.straggling, f2.strip_model, f2.dedx_model, f2.dx, f2.dy, f2.extent_mm, f2.seed) == (
        "gaussian", "two_step", "bethe", 1.0, -0.5, (2.5, 4.0), 7)
    assert f3.extent_mm == (2.0, 2.0) and f3.material == "Be"
    bad = tmp_path / "bad.dat"
    bad.write_text("FREQ 162.5\nDRIFT 10 20\n; HELIX_FOIL F C 600 strip_model=nope\nEND\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strip_model"):
        parse_tracewin(str(bad))
    demo = REPO / "examples" / "reliability_demo" / "reliability_demo.dat"
    lat3, _ = parse_tracewin(str(demo))
    assert [e for e in lat3.elements if isinstance(e, Foil)][0].straggling == "gaussian"


def test_extent_override_string_and_property():
    """The campaign transports the extent as "x,y" through --set: the
    property setter normalises strings, numbers and pairs."""
    from linac_gen.core.lattice import Lattice
    lat = Lattice(); lat.add(Drift("d", 10.0)); lat.add(Foil("F", "C", 600.0))
    common.apply_element_override(lat, "F.extent_mm", "2.5,4")
    assert lat.elements[1].extent_mm == (2.5, 4.0)
    common.apply_element_override(lat, "F.extent_mm", "3")
    assert lat.elements[1].extent_mm == (3.0, 3.0)
    lat.elements[1].extent_mm = None
    assert lat.elements[1].extent_mm is None
    beam = _beam(n=200)
    lat.elements[1].extent_mm = "1,1"
    lat.elements[1].apply_kick(beam)                      # no unpack error
    assert (beam.unstripped_table["state"] == "missed").any()
    with pytest.raises(ValueError):
        Foil("G", extent_mm="0,1")
