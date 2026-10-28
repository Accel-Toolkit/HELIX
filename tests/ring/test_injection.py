"""Ring mode, milestone M5: injection through a stripping foil.

The settings and their refusals; the waveforms; the linac bunches the chopper
keeps; the injected beam in ring coordinates (an H⁻ stripped to a proton of
the same velocity, the centroid placed exactly); the foil's statistics
against its formulas; the turn-by-turn injection, its ledger and its
restart; the bump magnets as exact dipoles (the bump's height at the foil
against its exact value); painting with a square-root waveform against the
uniform distribution in action it gives; the emittance a foil adds per
crossing.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tests.ring import injection_rings as R

SRC = {"kind": "gaussian", "species": "H-", "kinetic_mev": 800.0, "frequency_mhz": 162.5,
       "emit_nx_um": 0.3, "emit_ny_um": 0.3, "beta_x_m": 30.0, "beta_y_m": 5.0,
       "sigma_dp": 1e-4, "sigma_z_m": 0.002}
FOIL = {"at": "foil", "thickness_ug_cm2": 600.0, "x_min_m": -0.01, "x_max_m": 0.01,
        "y_min_m": -0.01, "y_max_m": 0.01}


def _inj(**kw):
    d = {"foil": dict(FOIL), "source": dict(SRC), "turns": 4, "particles_per_turn": 50,
         "current_ma": 2.0}
    d.update(kw)
    return d


@pytest.fixture(scope="module")
def foil_ring():
    return R.with_foil_marker(R.load_booster())


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
def test_the_settings_round_trip_and_name_what_they_refuse():
    from linac_gen.ring.injection.config import InjectionConfig, InjectionError
    d = _inj(magnets=[{"name": "bump", "elements": ["kb1", "kb2"], "model": "dipole",
                       "fint": 0.5, "hgap_m": 0.02,
                       "waveform": {"kind": "table", "times_s": [0, 1e-4], "values": [1, 0]}}])
    c = InjectionConfig.from_dict(d)
    assert InjectionConfig.from_dict(c.to_dict()).to_dict() == c.to_dict()
    assert json.loads(json.dumps(c.to_dict())) == c.to_dict()
    bad = [
        ({"current_ma": None}, "current_ma"),
        ({"turns": 0}, "turns"),
        ({"chopper_window_deg": 0.0}, "chopper_window_deg"),
        ({"chopper_window_deg": 200.0}, "chopper_window_deg"),
        ({"notch_buckets": -1}, "notch_buckets"),
        ({"nonsense": 1}, "unknown key"),
        ({"source": {"kind": "file"}}, "path"),
        ({"source": {"kind": "gaussian"}}, "kinetic_mev"),
        ({"foil": {"x_min_m": 0.01, "x_max_m": 0.0}}, "x_max_m"),
        ({"foil": {"map_bins": [0, 10]}}, "map_bins"),
        ({"foil": {"material": "Xx"}}, "material"),
        ({"turns": 20_000, "particles_per_turn": 10_000}, "in all"),
        ({"source": {"kind": "gaussian", "kinetic_mev": 800.0, "bunch_frequency_mhz": 0.0}},
         "bunch_frequency_mhz"),
        ({"source": dict(SRC, cutoff=0.99)}, "cutoff must be at least 1"),
        ({"source": dict(SRC, cutoff=1e-4)}, "cutoff must be at least 1"),
        ({"magnets": [{"name": "a", "elements": ["k1"]}, {"name": "b", "elements": ["K1"]}]},
         "one group per element"),
        ({"magnets": [{"name": "a", "elements": ["k1", "k2"], "kick_rad": [1e-3]}]},
         "one kick_rad per element"),
        ({"magnets": [{"name": "a", "elements": ["k1"], "model": "scale", "kick_rad": 1e-3}]},
         "no kick_rad"),
        ({"magnets": [{"name": "a", "elements": ["k1"], "fint": 0.5}]}, "'dipole' model"),
        ({"magnets": [{"name": "a", "elements": ["k1"],
                       "waveform": {"kind": "sqrt", "t_start_s": 1.0, "t_end_s": 1.0}}]},
         "t_end_s"),
    ]
    for change, word in bad:
        with pytest.raises(InjectionError, match=word):
            InjectionConfig.from_dict(dict(d, **change))


def test_the_ring_settings_hold_the_injection_and_its_empty_start():
    from linac_gen.ring import RingConfig
    cfg = RingConfig(source="empty", injection=_inj())
    assert RingConfig.from_dict(cfg.to_dict()).injection == cfg.injection
    with pytest.raises(ValueError, match="needs injection"):
        RingConfig(source="empty")
    with pytest.raises(ValueError, match="capacity"):
        RingConfig(source="empty", injection=_inj(), capacity=150)
    RingConfig(source="empty", injection=_inj(), capacity=200)
    with pytest.raises(ValueError, match="intensity"):
        RingConfig(source="empty", injection=_inj(), space_charge={"intensity": 1e10})


def test_settings_from_a_file_with_a_waveform_table(tmp_path):
    from linac_gen.ring.injection.config import injection_from_file, write_waveform_csv
    write_waveform_csv(tmp_path / "w.csv", [0.0, 1e-4, 2e-4], [1.0, 0.5, 0.0])
    d = _inj(magnets=[{"name": "bump", "elements": ["kb1"], "waveform": {"file": "w.csv"}}])
    d["source"] = {"kind": "file", "path": "beam.dst"}
    (tmp_path / "inj.json").write_text(json.dumps(d))
    c = injection_from_file(tmp_path / "inj.json")
    assert c.magnets[0].waveform.kind == "table"
    assert c.magnets[0].waveform.values == (1.0, 0.5, 0.0)
    # (kept absolute: the settings do not depend on how the file's path was spelled)
    assert c.source.path == str((tmp_path / "beam.dst").resolve())


# ---------------------------------------------------------------------------
# waveforms, chopper
# ---------------------------------------------------------------------------
def test_the_waveforms():
    from linac_gen.ring.injection.config import Waveform
    assert Waveform(kind="constant", value=0.25)(7.0) == 0.25
    t = Waveform(kind="table", times_s=(0.0, 1.0, 3.0), values=(1.0, 3.0, 0.0))
    assert [t(-1.0), t(0.5), t(2.0), t(9.0)] == [1.0, 2.0, 1.5, 0.0]
    lin = Waveform(kind="linear", t_start_s=1.0, t_end_s=3.0, start=2.0, end=0.0)
    assert [lin(0.0), lin(2.0), lin(4.0)] == [2.0, 1.0, 0.0]
    sq = Waveform(kind="sqrt", t_start_s=0.0, t_end_s=4.0, start=1.0, end=0.0)
    assert [sq(-1.0), sq(1.0), sq(4.0), sq(5.0)] == [1.0, 0.5, 0.0, 0.0]
    se = Waveform(kind="sqrt_end", t_start_s=0.0, t_end_s=4.0, start=1.0, end=0.0)
    assert [se(0.0), se(3.0), se(4.0)] == [1.0, 0.5, 0.0]


def test_the_linac_bunches_of_a_turn_and_the_chopper():
    from linac_gen.ring.injection import chopper as CH
    T = R.revolution_time()
    f_l = 162.5e6
    period = R.LENGTH_M / 84
    z0 = CH.bunch_zeta(0.0, T, f_linac_hz=f_l, beta0=R.BETA0, period_m=period)
    z1 = CH.bunch_zeta(T, 2 * T, f_linac_hz=f_l, beta0=R.BETA0, period_m=period)
    # (every bunch counted once: the turns' bunches add up to those of two turns)
    z01 = CH.bunch_zeta(0.0, 2 * T, f_linac_hz=f_l, beta0=R.BETA0, period_m=period)
    assert len(z0) + len(z1) == len(z01) == math.ceil(2 * T * f_l - 1e-9)
    assert z0[0] == 0.0                             # (the first bunch on the reference)
    # bunch j: zeta = -beta0 c j/f_L, wrapped into the RF period
    j = 7
    want = -R.BETA0 * R.C_LIGHT * j / f_l
    want -= period * math.floor(want / period + 0.5)
    assert z0[j] == pytest.approx(want, abs=1e-12)
    assert np.all(np.abs(z01) <= period / 2)
    # the chopper: the bunches within +-60 deg of the bucket's centre, about a third
    k = CH.kept(z01, period_m=period, window_deg=60.0, rf_on=True)
    assert np.all(np.abs(k) <= period / 6)
    assert len(k) / len(z01) == pytest.approx(1 / 3, abs=0.01)
    assert len(CH.kept(z01, period_m=period, window_deg=60.0, rf_on=False)) == len(z01)


# ---------------------------------------------------------------------------
# the injected beam
# ---------------------------------------------------------------------------
def test_an_hminus_is_stripped_to_a_proton_of_the_same_velocity():
    from linac_gen.ring.injection import source as S
    from linac_gen.ring.injection.config import SourceConfig
    from linac_gen.ring.reference import RingReference
    ring = RingReference.from_kinetic("proton", 800.0)
    cfg = SourceConfig(kind="gaussian", species="H-", kinetic_mev=800.0)
    pool = S.load_source(cfg, n_needed=10, seed=1, constants=ring.constants)
    pool.X[:] = 0.0                                  # (particles on the H- reference)
    X = S.to_ring(pool, np.arange(3), ring_ref=ring, centroid=(0.01, 2e-3, -0.02, -1e-3))
    m_h, m_p = pool.reference.mass, ring.mass
    gamma = 1.0 + 800.0 / m_h
    w_p = (gamma - 1.0) * m_p                        # 799.1296 MeV
    assert w_p == pytest.approx(799.1296, abs=5e-5)
    p_p = math.sqrt(w_p * (w_p + 2.0 * m_p))
    assert X[5, 0] == pytest.approx(p_p / ring.p0c - 1.0, rel=1e-13, abs=1e-15)
    # (its momentum m_p/m_H- of the ion's, 1.0880e-3 below; 7.067e-4 below an
    # 800 MeV proton's)
    assert (1.0 + X[5, 0]) * ring.p0c / pool.reference.p0c - 1.0 == pytest.approx(
        m_p / m_h - 1.0, rel=1e-12)
    assert m_p / m_h - 1.0 == pytest.approx(-1.0880e-3, abs=1e-7)
    assert X[5, 0] == pytest.approx(-7.067e-4, abs=1e-7)
    # the centroid placed exactly
    assert np.array_equal(X[:4, 0], [0.01, 2e-3, -0.02, -1e-3])
    assert X[4, 0] == 0.0


def test_the_pool_order_and_the_draw_by_turn():
    from linac_gen.ring.injection import source as S
    from linac_gen.ring.injection.config import SourceConfig
    from linac_gen.ring.reference import RingReference
    ring = RingReference.from_kinetic("proton", 800.0)
    pool = S.load_source(SourceConfig(kind="gaussian", kinetic_mev=800.0), n_needed=12, seed=3,
                         constants=ring.constants)
    perm = pool.order(3, "shuffled")
    assert sorted(perm.tolist()) == list(range(12))
    got = np.concatenate([pool.indices(k, 4, perm) for k in range(3)])
    assert np.array_equal(got, perm)                  # (each particle once)
    assert np.array_equal(pool.indices(3, 4, perm), perm[:4])      # (then again)
    assert pool.order(3, "in_order") is None
    assert np.array_equal(pool.indices(1, 4, None), [4, 5, 6, 7])


def test_a_gaussian_sources_cut_draws_as_before_and_a_cut_too_small_is_refused():
    """A pair inside the cut is kept (with probability 1 - exp(-c^2/2)), one
    outside drawn again: a cut below 1 sigma is refused — at 1e-4 the draw
    never ended — also when set past the settings' check; a cut of 1 sigma
    or more draws the numbers it drew before."""
    from linac_gen.ring.injection import source as S
    from linac_gen.ring.injection.config import InjectionError, SourceConfig
    n, seed = 3000, 11
    for cut in (1.0, 3.0):
        cfg = SourceConfig.from_dict(dict(SRC, cutoff=cut))
        pool = S.load_source(cfg, n_needed=n, seed=seed, constants="helix")
        rng = np.random.default_rng([seed, S.PURPOSE_GAUSSIAN])      # (the draw, written out)
        U = np.empty((6, n))
        for k in range(3):
            got = 0
            while got < n:
                u = rng.standard_normal((2, max(2 * (n - got), 64)))
                u = u[:, (u[0] ** 2 + u[1] ** 2) < cut * cut]
                m = min(u.shape[1], n - got)
                U[2 * k:2 * k + 2, got:got + m] = u[:, :m]
                got += m
        bg = pool.reference.beta0gamma0
        assert np.array_equal(pool.X[0], math.sqrt(cfg.emit_nx_um * 1e-6 / bg * cfg.beta_x_m)
                              * U[0])
        assert np.array_equal(pool.X[4], cfg.sigma_z_m * U[4])
        assert np.array_equal(pool.X[5], cfg.sigma_dp * U[5])
    for bad in (0.999, 1e-4):
        with pytest.raises(InjectionError, match="cutoff must be at least 1"):
            SourceConfig.from_dict(dict(SRC, cutoff=bad))
    cfg = SourceConfig.from_dict(dict(SRC, cutoff=3.0))
    cfg.cutoff = 1e-4
    with pytest.raises(InjectionError, match="cutoff must be at least 1"):
        S.load_source(cfg, n_needed=10, seed=1, constants="helix")


def test_tracewin_particle_files_are_read(tmp_path):
    from linac_gen.ring.injection.source import read_tracewin_particles
    p = tmp_path / "beam.dat"
    p.write_text("Number_of_particle Mass(MeV) Beam_energy(MeV) Beam_Frequency(MHz) "
                 "Beam_courant(A) Beam_Charge\n"
                 "3 939.294 160.0 352.21 0.025 -1\n"
                 "x(mm) x'(mrad) y(mm) y'(mrad) z(mm) z'(mrad) Phase(deg) Time(s) Energy(MeV) "
                 "Loss\n"
                 "1.0 0.1 -2.0 0.2 10.0 1.0 -5.0 -3.9e-11 160.5 0\n"
                 "2.0 0.3 -1.0 0.4 -10.0 1.0 5.0 3.9e-11 159.5 0\n"
                 "9.0 9.0 9.0 9.0 0.0 0.0 0.0 0.0 160.0 1\n")
    P, head = read_tracewin_particles(p)
    assert head["n_lost"] == 1 and head["charge"] == -1 and head["kinetic_mev"] == 160.0
    assert np.array_equal(P, [[1.0, 0.1, -2.0, 0.2, -5.0, 0.5], [2.0, 0.3, -1.0, 0.4, 5.0, -0.5]])


# ---------------------------------------------------------------------------
# the foil
# ---------------------------------------------------------------------------
def _foil(**kw):
    from linac_gen.ring.injection.foil import FoilSettings, RingFoil
    d = dict(name="f", node=0, material="C", thickness_ug_cm2=600.0, straggling="auto",
             dedx_model="mip", strip_model="two_step", x_min_m=-0.01, x_max_m=0.01,
             y_min_m=None, y_max_m=0.005, map_bins=(10, 10))
    d.update(kw)
    return RingFoil(FoilSettings(**d), mass_mev=938.27208816)


def test_the_foils_rectangle_with_open_sides():
    f = _foil()
    x = np.array([-0.01, 0.01, 0.0100001, 0.0, 0.0])
    y = np.array([0.0, 0.0, 0.0, -9.0, 0.0050001])
    assert f.inside(x, y).tolist() == [True, True, False, True, False]


def test_the_foils_scattering_loss_and_stripping_against_their_formulas():
    """10^6 crossings: the rms angle kick is Highland's theta0, the mean energy
    loss the tabulated rate (0.5 %); the stripping fractions those of the
    two-step model (3 sigma)."""
    from linac_gen.ring.reference import RingReference
    ref = RingReference.from_kinetic("proton", 800.0)
    f = _foil(thickness_ug_cm2=6000.0)
    n = 1_000_000
    X = np.zeros((6, n))
    rng = np.random.default_rng(11)
    loss = f.cross(X, rng, p0c=ref.p0c, beta0=ref.beta0, kinetic_mev=ref.kinetic0)
    ph = f.physics(ref.kinetic0)
    assert np.std(X[1] / (1.0 + X[5])) == pytest.approx(ph["theta0"], rel=5e-3)
    assert np.std(X[3] / (1.0 + X[5])) == pytest.approx(ph["theta0"], rel=5e-3)
    assert np.mean(loss) == pytest.approx(ph["mean_loss_mev"], rel=5e-3)
    # the energy lost, in delta: E' = E - loss
    e = ref.energy0 + ref.p0c * np.asarray(
        __import__("linac_gen.ring.coords", fromlist=["x"]).ptau_from_delta(X[5], ref.beta0))
    assert np.allclose(e, ref.energy0 - loss, rtol=0.0, atol=1e-9)
    codes = f.strip(n, rng, kinetic_mev=800.0, species="H-")
    f_p, f_h0, f_hm = f.stripping_fractions(800.0)
    for code, frac in ((0, f_p), (1, f_h0)):
        k = int(np.sum(codes == code))
        assert abs(k - n * frac) <= 3.0 * math.sqrt(n * frac * (1 - frac)) + 1.0
    assert f.strip(5, rng, kinetic_mev=800.0, species="proton").tolist() == [0] * 5


def test_repeated_crossings_add_up_to_highland_at_the_total_thickness():
    """A particle's k-th crossing gets the growth of Highland's theta0^2 from
    k - 1 to k thicknesses: 1, 10 and 100 crossings of the same 2e5 particles
    (no transport between) spread their angles as Highland at 1, 10 and 100
    times the foil (1 %) — 1.15 and 1.30 times wider than the sum of
    one-crossing kicks (Highland's logarithm of x z^2 / (X0 beta^2)); a first
    crossing is the one-crossing kick bit for bit."""
    from linac_gen.ring.reference import RingReference
    ref = RingReference.from_kinetic("proton", 800.0)
    f = _foil(thickness_ug_cm2=600.0)
    ph = f.physics(ref.kinetic0)
    a = f.elem._thickness_over_X0() / ref.beta0 ** 2     # (x z^2 / (X0 beta^2), z = 1)
    th1 = ph["theta0"]

    def highland(m):            # Highland's theta0 at m thicknesses, from one
        return th1 * math.sqrt(m) * (1.0 + 0.038 * math.log(m * a)) / (1.0 + 0.038 * math.log(a))

    n = 200_000
    kw = dict(p0c=ref.p0c, beta0=ref.beta0, kinetic_mev=ref.kinetic0)
    X1, X2 = np.zeros((6, n)), np.zeros((6, n))
    l1 = f.cross(X1, np.random.default_rng(5), **kw)
    l2 = f.cross(X2, np.random.default_rng(5), crossings=np.zeros(n, dtype=np.int32), **kw)
    assert np.array_equal(X1, X2) and np.array_equal(l1, l2)
    for N, wider in ((1, 1.0), (10, 1.149), (100, 1.297)):
        X = np.zeros((6, n))
        rng = np.random.default_rng(7)
        for k in range(N):
            f.cross(X, rng, crossings=np.full(n, k, dtype=np.int32), **kw)
        for i in (1, 3):
            assert np.std(X[i] / (1.0 + X[5])) == pytest.approx(highland(N), rel=1e-2)
        assert highland(N) / (math.sqrt(N) * th1) == pytest.approx(wider, abs=1e-3)
    # (the widths themselves: their squares sum to Highland's theta0^2 at N thicknesses)
    w = np.concatenate([f.widths(ref.kinetic0, np.full(1, k), 1) for k in range(100)])
    assert np.sum(w ** 2) == pytest.approx(highland(100) ** 2, rel=1e-12)


def test_each_crossing_gets_the_particles_own_count_of_crossings_so_far(monkeypatch, tmp_path):
    """The width of a particle's next crossing follows the crossings it has
    made: an injection crossing passes none (a first crossing); a circulating
    one the particle's own count from an independent tally of who was inside
    the foil when (its injection crossing included); a stored beam crossing a
    foil element every turn passes 0, 1, 2, ...; a run restarted from a
    checkpoint passes the counts the uninterrupted run passed."""
    from linac_gen.elements.foil import Foil
    from linac_gen.ring import RingConfig, RingSimulation
    from linac_gen.ring import particles as P
    from linac_gen.ring.injection import foil as F
    from linac_gen.ring.injection import injector as I
    seen = []
    orig_cross = F.RingFoil.cross

    def cross(self, X, rng, *, crossings=None, **kw):
        seen.append(None if crossings is None else np.asarray(crossings, dtype=np.int64).copy())
        return orig_cross(self, X, rng, crossings=crossings, **kw)

    monkeypatch.setattr(F.RingFoil, "cross", cross)
    # a stored beam on a foil element: every particle, every turn
    lat = R.load_booster()
    lat.elements.insert(1, Foil(name="foil", material="C", thickness_ug_cm2=600.0,
                                extent_mm=(50.0, 50.0)))
    beam = R.booster_beam(emit_nx=0.05, emit_ny=0.05, emit_z=1.0, beta_z=100.0, cutoff=4.0)
    res = RingSimulation(lat, beam, RingConfig(n_turns=4, n_particles=300, source="matched",
                                               tune_probes=0, seed=8)).run()
    assert res.n_alive[-1] == 300 and len(seen) == 4
    assert all(np.array_equal(k, np.full(300, t)) for t, k in enumerate(seen))
    # injection, then circulating crossings against an independent tally
    tally, bad, checked = {}, [], [0]
    orig_pass, orig_inject = I.Injection.foil_pass, I.Injection.inject

    def foil_pass(self, arrays, weight, n_used, turn, **kw):
        coords, _pt, _rp, _rv, state, _hits = arrays
        circ = np.flatnonzero(state[:int(n_used)] == P.CIRCULATING)
        idx = circ[self.foil.inside(coords[0, circ], coords[2, circ])]
        n0 = len(seen)
        orig_pass(self, arrays, weight, n_used, turn, **kw)
        if idx.size:
            want = np.array([tally[int(i)] for i in idx])
            bad.append(len(seen) != n0 + 1 or not np.array_equal(seen[-1], want))
            checked[0] += idx.size
            for i in idx:
                tally[int(i)] += 1

    def inject(self, parts, turn, **kw):
        n0 = len(seen)
        sl = orig_inject(self, parts, turn, **kw)
        bad.extend(k is not None for k in seen[n0:])
        if sl is not None:
            for i in range(sl.start, sl.stop):
                if not (parts.state[i] == P.WASTE and parts.lost_cause[i] == I.CAUSE_MISSED):
                    tally[i] = 1
        return sl

    monkeypatch.setattr(I.Injection, "foil_pass", foil_pass)
    monkeypatch.setattr(I.Injection, "inject", inject)
    base = dict(n_turns=8, source="empty", injection=_bump_injection(n_turns=4, per_turn=60),
                tune_probes=0, seed=4)
    ck = tmp_path / "ck.npz"
    seen.clear()
    RingSimulation(R.bumped(0.02), R.booster_beam(),
                   RingConfig(checkpoint_every=4, **base)).run(checkpoint_path=ck)
    assert checked[0] > 100 and not any(bad)
    assert max(tally.values()) >= 3                 # (particles crossing again and again)
    full = [k for k in seen if k is not None]
    seen.clear()
    monkeypatch.setattr(I.Injection, "foil_pass", orig_pass)
    monkeypatch.setattr(I.Injection, "inject", orig_inject)
    RingSimulation(R.bumped(0.02), R.booster_beam(),
                   RingConfig(checkpoint_every=4, **base)).run(restart_from=ck)
    tail = [k for k in seen if k is not None]
    assert tail and all(np.array_equal(a, b) for a, b in zip(full[-len(tail):], tail,
                                                             strict=True))


# ---------------------------------------------------------------------------
# a run with injection
# ---------------------------------------------------------------------------
def _run(lat, inj, **kw):
    from linac_gen.ring import RingConfig, RingSimulation
    d = dict(n_turns=6, source="empty", injection=inj, tune_probes=0, seed=3)
    d.update(kw)
    return RingSimulation(lat, R.booster_beam(), RingConfig(**d)).run()


def test_the_ring_fills_turn_by_turn_and_the_ledger_holds(foil_ring):
    res = _run(foil_ring, _inj(), snapshot_turns=["last"])
    rec = res.injection
    assert res.n_alive.tolist() == [0, 50, 100, 150, 200, 200, 200]
    assert rec["injected"].tolist() == [50, 50, 50, 50, 0, 0]
    led = res.ledger
    assert led["injected"] == led["circulating"] + led["lost"] + led["waste"] == 200
    assert led["injected_charge"] == pytest.approx(
        led["circulating_charge"] + led["lost_charge"] + led["waste_charge"], rel=1e-15)
    # each macroparticle: I T / e over the 84 buckets and the 50 of a turn; the
    # record counts the ring's real particles (every bucket), the ledger the
    # tracked bucket's
    w = 2e-3 * R.revolution_time() / 1.602176634e-19 / 84 / 50
    assert rec["buckets"] == 84
    assert rec["injected_charge"][0] == pytest.approx(84 * 50 * w, rel=1e-9)
    assert rec["injected_charge"][0] == pytest.approx(
        2e-3 * R.revolution_time() / 1.602176634e-19, rel=1e-9)
    assert led["injected_charge"] == pytest.approx(np.sum(rec["injected_charge"]) / 84,
                                                   rel=1e-12)
    # the stripped ions circulate; the injected beam's first crossing is counted
    assert np.sum(rec["stripped"]) + np.sum(rec["h0"]) + np.sum(rec["hminus"]) == 200
    assert rec["foil_hits_injected"].tolist() == [50, 50, 50, 50, 0, 0]
    assert rec["foil"]["theta0_rad"] == pytest.approx(2.4357e-5, rel=1e-4)
    # an injected particle in the snapshot: in the RF period, relative to the ring's reference
    X = res.snapshots[6]["coords"]
    assert np.all(np.abs(X[4]) <= R.LENGTH_M / 84 / 2)
    assert np.mean(X[5]) == pytest.approx(-7.067e-4, abs=3e-5)


def test_a_beam_beside_the_foil_misses_it(foil_ring):
    inj = _inj()
    inj["source"] = dict(SRC, x_m=0.05)                  # (5 cm off, the foil +-1 cm)
    res = _run(foil_ring, inj)
    assert res.ledger["circulating"] == 0 and res.ledger["waste"] == 200
    assert res.injection["waste"]["missed"] == 200
    # (the ring empty once the injection is over: the run ends there)
    assert res.injection["missed"].tolist() == [50, 50, 50, 50]
    assert res.n_alive.tolist() == [0] * 5
    # (nothing ever circulated: nothing lost — a run that ended, not one cut short)
    assert "all_lost_at_turn" not in res.meta and not res.truncated
    assert any("nothing circulated" in n for n in res.meta["notes"])


def test_two_step_stripping_leaves_waste(foil_ring):
    inj = _inj(turns=2, particles_per_turn=2000)
    inj["foil"] = dict(FOIL, thickness_ug_cm2=150.0)     # (a thin foil: about 1/4 not stripped)
    res = _run(foil_ring, inj, n_turns=2)
    rec = res.injection
    n_waste = int(np.sum(rec["h0"]) + np.sum(rec["hminus"]))
    assert n_waste == res.injection["waste"]["unstripped"] == res.ledger["waste"]
    f_p = rec["foil"]["stripped_fraction"]
    n = 4000
    assert abs(np.sum(rec["stripped"]) - n * f_p) <= 3 * math.sqrt(n * f_p * (1 - f_p)) + 1
    assert res.ledger["circulating"] == np.sum(rec["stripped"])


def test_a_restart_continues_the_injection_bit_for_bit(foil_ring, tmp_path):
    from linac_gen.ring import RingConfig, RingSimulation
    inj = _inj(turns=6, particles_per_turn=20)
    kw = dict(n_turns=8, source="empty", injection=inj, tune_probes=1, seed=4,
              snapshot_turns=["last"])
    full = RingSimulation(foil_ring, R.booster_beam(), RingConfig(**kw)).run()
    ck = tmp_path / "ck.npz"
    # (a checkpoint at turn 3 of a run of 8 turns, stopped there)
    part = RingSimulation(foil_ring, R.booster_beam(), RingConfig(**dict(kw, checkpoint_every=3)))
    part_res = part.run(checkpoint_path=ck, should_abort=_abort_after(3))
    assert part_res.turns_done == 3
    rest = RingSimulation(foil_ring, R.booster_beam(), RingConfig(**kw)).run(restart_from=ck)
    assert np.array_equal(rest.snapshots[8]["coords"], full.snapshots[8]["coords"])
    assert np.array_equal(rest.cov, full.cov, equal_nan=True)
    for k in ("injected", "foil_hits", "deposit_mev"):
        assert np.array_equal(rest.injection[k], full.injection[k])
    assert rest.ledger == full.ledger


def _abort_after(n):
    calls = {"k": 0}

    def stop():
        calls["k"] += 1
        return calls["k"] > n
    return stop


def test_the_threads_change_no_bit(foil_ring):
    a = _run(foil_ring, _inj(turns=3, particles_per_turn=300), threads=1, snapshot_turns=["last"])
    b = _run(foil_ring, _inj(turns=3, particles_per_turn=300), threads=3, snapshot_turns=["last"])
    assert np.array_equal(a.snapshots[6]["coords"], b.snapshots[6]["coords"])
    assert np.array_equal(a.injection["foil_hits"], b.injection["foil_hits"])


# ---------------------------------------------------------------------------
# bump magnets and painting
# ---------------------------------------------------------------------------
def test_a_bump_of_exact_dipoles_has_its_exact_height_at_the_foil():
    """Four thick kickers compiled as exact straight-frame dipoles make a closed
    bump: its height at the foil is the exact one (the field's trajectory),
    in the optics and in the run's closed-orbit record, turn by turn as the
    waveform takes it down."""
    from linac_gen.ring import RingConfig
    from linac_gen.ring.compile import compile_from_config
    from linac_gen.ring.optics import twiss
    th, L, g12 = 0.02, 0.3, 1.0
    lat = R.bumped(th, L, g12)
    inj = _inj(turns=4, particles_per_turn=5, magnets=[{
        "name": "bump", "elements": ["kb1", "kb2", "kb3", "kb4"], "model": "dipole",
        "waveform": {"kind": "linear", "t_start_s": 0.0, "t_end_s": 1.0, "start": 1.0,
                     "end": 0.0}}])
    inj["foil"] = dict(FOIL, y_min_m=-0.05, y_max_m=0.05)
    cfg = RingConfig(n_turns=5, source="empty", injection=inj, tune_probes=0)
    rl = compile_from_config(lat, cfg, beam=R.booster_beam())
    # (down to zero over four turns of the ring's own revolution time)
    inj["magnets"][0]["waveform"]["t_end_s"] = 4 * rl.revolution_time
    assert [b["plane"] for b in rl.bumps] == ["y"] * 4
    assert rl.bumps[0]["k0"] == pytest.approx(-th / L, rel=1e-15)
    opt = twiss(rl.with_rf(False))
    k = rl.element_names.index("foil")
    h = R.bump_height(th, L, g12)
    assert opt.table["y"][k] == pytest.approx(h, abs=1e-15)
    assert abs(opt.table["y"][rl.element_names.index("bump_c")]) < 1e-15   # (closed)
    res = _run(lat, inj, n_turns=5)
    co = res.injection["closed_orbit_foil"]
    w = res.injection["waveforms"][:, 0]
    assert w[:5].tolist() == pytest.approx([1.0, 0.75, 0.5, 0.25, 0.0], abs=1e-15)
    assert w[4] == 0.0
    for turn in range(5):
        want = R.bump_height(th * w[turn], L, g12)
        assert co[turn, 2] == pytest.approx(want, abs=1e-14), turn
    assert np.all(np.abs(co[:5, 0]) < 1e-9)


def test_painting_with_a_square_root_waveform_fills_the_actions_uniformly():
    """The bump collapses as 1 - sqrt(t/T) under a beam injected at its full
    height: the turn-k particles start at the closed orbit's distance
    dy_k = h(theta) - h(theta (1 - sqrt(k/N))) from it with no angle, so their
    action is gamma_y dy_k^2 / 2 at the foil — the actions of a uniform
    distribution, a sqrt(t) closed orbit giving actions linear in t.  The
    particles' actions at the end (the bump gone; Twiss at the start of the
    ring) against these (1 %)."""
    from linac_gen.ring import RingConfig
    from linac_gen.ring.compile import compile_from_config
    from linac_gen.ring.optics import twiss
    th, L, g12, N = 0.02, 0.3, 1.0, 20
    lat = R.bumped(th, L, g12)
    T = R.revolution_time()
    h = R.bump_height(th, L, g12)
    src = dict(SRC, species="proton", emit_nx_um=1e-7, emit_ny_um=1e-7, sigma_dp=0.0,
               sigma_z_m=0.0, y_m=h)
    inj = _inj(source=src, turns=N, particles_per_turn=10, magnets=[{
        "name": "bump", "elements": ["kb1", "kb2", "kb3", "kb4"], "model": "dipole",
        "waveform": {"kind": "sqrt", "t_start_s": 0.0, "t_end_s": N * T, "start": 1.0,
                     "end": 0.0}}])
    inj["foil"] = dict(FOIL, thickness_ug_cm2=1e-9, y_min_m=-0.05, y_max_m=0.05)
    res = _run(lat, inj, n_turns=N + 1, snapshot_turns=["last"])
    cfg = RingConfig(n_turns=1, source="empty", injection=inj, tune_probes=0)
    rl = compile_from_config(lat, cfg, beam=R.booster_beam())
    opt = twiss(rl.with_rf(False))          # (the compiled ring: its bumps at full strength)
    kf = rl.element_names.index("foil")
    bf, af = opt.table["bety"][kf], opt.table["alfy"][kf]
    gf = (1.0 + af * af) / bf
    want = np.mean([0.5 * gf * (h - R.bump_height(th * (1.0 - math.sqrt(k / N)), L, g12)) ** 2
                    for k in range(N)])
    X = res.snapshots[N + 1]["coords"]
    b0, a0 = opt.start["bety"], opt.start["alfy"]
    g0 = (1.0 + a0 * a0) / b0
    yp = X[3] / (1.0 + X[5])
    J = 0.5 * (g0 * X[2] ** 2 + 2.0 * a0 * X[2] * yp + b0 * yp ** 2)
    assert np.mean(J) == pytest.approx(want, rel=1e-2)
    # (the emittance of a uniform distribution in action from 0 to J_max is J_max/2)
    assert np.mean(J) == pytest.approx(0.5 * gf * h * h * (N - 1) / (2 * N), rel=1e-2)


# ---------------------------------------------------------------------------
# a foil in the ring
# ---------------------------------------------------------------------------
def test_a_foil_element_crossed_every_turn_scatters_as_highland_at_the_total_thickness():
    """A ring with a foil element (no injection settings: the foil acts on the
    circulating beam) and a matched beam inside it: every particle crosses it
    every turn, and after N crossings its vertical action has grown on average
    by beta_y theta0(N x)^2 / 2 — Highland's width at the total thickness
    (random angle kicks, whatever alpha is; no vertical dispersion for the
    energy loss to act through): for 20 crossings of 6000 ug/cm^2, 1.37 times
    20 one-crossing kicks; against that (5 %)."""
    from linac_gen.elements.foil import Foil
    from linac_gen.ring import RingConfig, RingSimulation
    from linac_gen.ring.compile import compile_from_config
    from linac_gen.ring.optics import twiss
    lat = R.load_booster()
    lat.elements.insert(1, Foil(name="foil", material="C", thickness_ug_cm2=6000.0,
                                extent_mm=(50.0, 50.0)))
    beam = R.booster_beam(emit_nx=0.05, emit_ny=0.05, emit_z=1.0, beta_z=100.0, cutoff=4.0)
    cfg = RingConfig(n_turns=20, n_particles=3000, source="matched", tune_probes=0, seed=8,
                     snapshot_turns=[0, 20])
    res = RingSimulation(lat, beam, cfg).run()
    rec = res.injection
    assert rec["foil"]["name"] == "foil" and rec["foil"]["thickness_ug_cm2"] == 6000.0
    assert rec["foil"]["strip_model"] == "off"            # (the element's own)
    assert res.n_alive[-1] == 3000                       # (none lost)
    assert rec["foil_hits"].tolist() == [3000] * 20        # (every one, every turn)
    assert res.ledger["injected"] == 3000                # (no injection: the start's beam)
    rl = compile_from_config(lat, cfg, beam=beam)
    opt = twiss(rl.with_rf(False))
    kf = rl.element_names.index("foil")
    b0, a0 = opt.start["bety"], opt.start["alfy"]
    g0 = (1.0 + a0 * a0) / b0

    def mean_action(X):
        yp = X[3] / (1.0 + X[5])
        return np.mean(0.5 * (g0 * X[2] ** 2 + 2.0 * a0 * X[2] * yp + b0 * yp ** 2))

    grew = mean_action(res.snapshots[20]["coords"]) - mean_action(res.snapshots[0]["coords"])
    from linac_gen.ring.reference import RingReference
    a = (Foil(name="f", material="C", thickness_ug_cm2=6000.0)._thickness_over_X0()
         / RingReference.from_kinetic("proton", 800.0).beta0 ** 2)   # (x z^2 / (X0 beta^2))
    th20 = (rec["foil"]["theta0_rad"] * math.sqrt(20.0) * (1.0 + 0.038 * math.log(20.0 * a))
            / (1.0 + 0.038 * math.log(a)))
    assert (th20 / rec["foil"]["theta0_rad"]) ** 2 / 20.0 == pytest.approx(1.365, abs=1e-3)
    want = 0.5 * opt.table["bety"][kf] * th20 ** 2
    assert grew == pytest.approx(want, rel=5e-2)
    # the energy left in the foil: the crossings' losses, about the mean loss each
    w = res.ledger["injected_charge"] / 3000
    assert np.sum(rec["deposit_mev"]) == pytest.approx(
        20 * 3000 * w * rec["foil"]["mean_loss_mev"], rel=3e-2)
    assert np.sum(rec["hit_map"]) == pytest.approx(20 * 3000 * w, rel=1e-12)


def test_a_ring_with_two_foils_asks_which(foil_ring):
    from linac_gen.elements.foil import Foil
    from linac_gen.ring import RingConfig, RingSimulation
    from linac_gen.ring.injection.config import InjectionError
    lat = R.load_booster()
    lat.elements.insert(1, Foil(name="f1", thickness_ug_cm2=600.0))
    lat.elements.insert(3, Foil(name="f2", thickness_ug_cm2=600.0))
    cfg = RingConfig(n_turns=1, n_particles=10, tune_probes=0)
    with pytest.raises(InjectionError, match="several foil elements"):
        RingSimulation(lat, R.booster_beam(emit_nx=1.0, emit_ny=1.0), cfg).run()
    inj = _inj()
    inj["foil"] = {"at": "nowhere", "thickness_ug_cm2": 600.0}
    with pytest.raises(InjectionError, match="not in the ring"):
        _run(foil_ring, inj)
    inj["foil"] = {"at": "foil"}
    with pytest.raises(InjectionError, match="thickness"):
        _run(foil_ring, inj)
    # (one foil per ring: the other foil element would not act)
    inj["foil"] = {"at": "f1"}
    with pytest.raises(InjectionError, match="'f2' would not act"):
        _run(lat, inj)
    lat2 = R.with_foil_marker(R.load_booster())
    lat2.elements.insert(5, Foil(name="f3", thickness_ug_cm2=600.0))
    inj["foil"] = {"at": "foil", "thickness_ug_cm2": 600.0}
    with pytest.raises(InjectionError, match="'f3' would not act"):
        _run(lat2, inj)


# ---------------------------------------------------------------------------
# magnets: kickers, multipoles, refusals
# ---------------------------------------------------------------------------
def test_kicker_and_scale_groups_write_their_strengths():
    from linac_gen.elements.multipole import Multipole
    from linac_gen.ring import RingConfig
    from linac_gen.ring.compile import compile_from_config
    from linac_gen.ring.injection.config import InjectionConfig
    from linac_gen.ring.injection.programs import MagnetPrograms
    lat = R.bumped(0.01, 0.3, 1.0, plane="x")
    lat.elements.insert(2, Multipole(name="mb", knl=[0.0, 0.0, 0.4], ksl=[0.0, 1e-3]))
    inj = _inj(magnets=[
        {"name": "paint", "elements": ["kb1"], "model": "kick", "plane": "y", "kick_rad": 2e-3,
         "waveform": {"kind": "constant", "value": 0.5}},
        {"name": "err", "elements": ["mb", "kb2"], "model": "scale",
         "waveform": {"kind": "constant", "value": 0.25}}])
    cfg = RingConfig(n_turns=1, source="empty", injection=inj)
    rl = compile_from_config(lat, cfg, beam=R.booster_beam())
    ic = InjectionConfig.from_dict(inj)
    pr = MagnetPrograms(ic.magnets, rl)
    p = rl.params.copy()
    pr.apply(0.0, p)
    from linac_gen.ring.kernels import nodes as nd

    def offset(kind, name):
        return rl.offsets[[j for j in range(rl.n_nodes) if rl.kinds[j] == kind
                           and rl.element_names[rl.node_element[j]] == name][0]]

    o = offset(nd.N_KICK, "kb1")
    # (the y plane, 2 mrad x 0.5; x keeps the kicker's own 10 mrad)
    assert (p[o], p[o + 1]) == (rl.params[o], 1e-3) and rl.params[o] == pytest.approx(0.01)
    o2 = offset(nd.N_KICK, "kb2")
    assert (p[o2], p[o2 + 1]) == (0.25 * rl.params[o2], 0.0)
    om = offset(nd.N_MULTIPOLE, "mb")
    order = int(rl.params[om])
    blk = slice(om + 2, om + 4 + 2 * order)
    assert np.array_equal(p[blk], 0.25 * rl.params[blk])


def test_the_magnet_models_refuse_what_they_cannot_drive(foil_ring):
    from linac_gen.ring import RingConfig, RingSimulation
    from linac_gen.ring.compile import RingCompileError
    from linac_gen.ring.injection.config import InjectionError
    lat = R.bumped(0.01)
    for group, err, word in (
            ({"elements": ["kb1"], "model": "dipole"}, None, None),
            ({"elements": ["foil"], "model": "dipole"}, RingCompileError, "dipole"),
            ({"elements": ["bump_a"], "model": "kick"}, InjectionError, "not a kicker"),
            ({"elements": ["bump_a"], "model": "scale"}, InjectionError, "no kick or multipole"),
            ({"elements": ["nothere"], "model": "kick"}, InjectionError, "not in the ring")):
        inj = _inj(turns=1, particles_per_turn=2, magnets=[dict(group, name="g")])
        inj["foil"] = dict(FOIL, y_min_m=-0.05, y_max_m=0.05)
        cfg = RingConfig(n_turns=1, source="empty", injection=inj, tune_probes=0)
        if err is None:
            RingSimulation(lat, R.booster_beam(), cfg).run()
            continue
        with pytest.raises(err, match=word):
            RingSimulation(lat, R.booster_beam(), cfg).run()


# ---------------------------------------------------------------------------
# sources through a run
# ---------------------------------------------------------------------------
def test_a_dst_file_source_and_the_chopper_window(foil_ring, tmp_path):
    from linac_gen.io.tracewin_dst import write_dst
    rng = np.random.default_rng(2)
    P = np.zeros((500, 6))
    P[:, 0] = rng.normal(0.0, 0.5, 500)              # mm
    P[:, 1] = rng.normal(0.0, 0.05, 500)             # mrad
    P[:, 4] = rng.normal(0.0, 2.0, 500)              # deg at 162.5 MHz
    path = tmp_path / "linac.dst"
    write_dst(str(path), P, current_mA=2.0, frequency_MHz=162.5, mass_MeV=939.294,
              w_kin_ref=800.0)
    inj = _inj(turns=3, particles_per_turn=400, chopper_window_deg=45.0)
    inj["source"] = {"kind": "file", "path": str(path)}
    res = _run(foil_ring, inj, n_turns=3)
    assert res.injection["source"]["species"] == "H-"          # (from the file's mass)
    assert res.injection["source"]["frequency_mhz"] == 162.5
    assert res.injection["source"]["pool"] == 500
    # the same file as protons at the ring's energy (no slip from an energy offset):
    # the bunches kept within +-45 deg of the bucket's centre, each of rms length 8.6 mm
    inj["source"] = {"kind": "file", "path": str(path), "species": "proton"}
    res = _run(foil_ring, inj, n_turns=1, snapshot_turns=["last"])
    X = res.snapshots[1]["coords"]
    period = R.LENGTH_M / 84
    assert np.max(np.abs(X[4])) < period * (45.0 / 360.0) + 0.04
    assert np.max(np.abs(X[4])) > period * (45.0 / 360.0) - 0.04


# ---------------------------------------------------------------------------
# results file, devices, with space charge and a ramp
# ---------------------------------------------------------------------------
def _bump_injection(n_turns=3, per_turn=10):
    inj = _inj(turns=n_turns, particles_per_turn=per_turn, magnets=[{
        "name": "bump", "elements": ["kb1", "kb2", "kb3", "kb4"], "model": "dipole",
        "waveform": {"kind": "linear", "t_start_s": 0.0, "t_end_s": 1e-5, "start": 1.0,
                     "end": 0.2}}])
    inj["source"] = dict(SRC, y_m=0.02)
    inj["foil"] = dict(FOIL, y_min_m=0.0, y_max_m=0.03)
    return inj


def test_the_results_file_keeps_the_injection(tmp_path):
    from linac_gen.ring.results import load_ring_hdf5, save_ring_hdf5
    res = _run(R.bumped(0.02), _bump_injection(), n_turns=4)
    back = load_ring_hdf5(save_ring_hdf5(res, tmp_path / "r.h5"))
    a, b = res.injection, back.injection
    for k in ("injected", "stripped", "h0", "hminus", "missed", "injected_charge", "foil_hits",
              "foil_hits_injected", "deposit_mev", "electron_mev", "closed_orbit_foil",
              "waveforms", "hit_map", "deposit_map_mev", "hits_histogram"):
        assert np.array_equal(a[k], b[k], equal_nan=True), k
    assert b["waveform_names"] == ["bump"] and b["map_extent_m"] == a["map_extent_m"]
    for k in ("foil", "settings", "source", "waste", "mean_hits_circulating", "buckets"):
        assert b[k] == a[k], k
    assert back.ledger == res.ledger


def test_space_charge_takes_the_injected_intensity(foil_ring):
    sc = {"kicks_per_turn": 12, "grid": [16, 16], "z_bins": 32}
    for model in ("pic", "frozen"):
        res = _run(foil_ring, _inj(turns=3, particles_per_turn=200), n_turns=4,
                   space_charge=dict(sc, model=model))
        assert res.n_alive[-1] == 600
        m = res.meta["space_charge"]
        # (reported: the bunch circulating at the end, from the injected current,
        # and its analytic tune shift then)
        assert m["intensity_used"] == pytest.approx(res.ledger["circulating_charge"],
                                                    rel=1e-15)
        assert m["intensity"] is None and m["intensity_from"] == "injection"
        assert m["laslett_gaussian"] is not None and m["laslett_gaussian"][0] < 0.0


def test_auto_line_density_bins_wait_for_the_injected_bunch(foil_ring):
    """z_bins "auto" with a ring that starts empty: the start's bins stay
    until a turn's first kick finds particles; then the bins follow them
    (few particles: few bins)."""
    from linac_gen.ring.sc.solver import AUTO_NZ_START
    res = _run(foil_ring, _inj(turns=3, particles_per_turn=200), n_turns=4,
               space_charge={"kicks_per_turn": 12, "grid": [16, 16]})
    m = res.meta["space_charge"]
    assert m["z_bins"] == "auto" and m["z_bins_auto"] is True
    hist = m["z_bins_used"]
    assert hist[0] == [0, AUTO_NZ_START] and len(hist) >= 2
    assert all(t >= 1 for t, _b in hist[1:]) and hist[1][1] < AUTO_NZ_START
    assert res.n_alive[-1] == 600


def test_a_ramp_with_injection_and_magnets_runs():
    ramp = {"energy": {"kind": "table", "times_s": [0.0, 1e-4], "values": [800.0, 830.0]},
            "voltage": {"kind": "table", "times_s": [0.0, 1.0], "values_mv": [1.2, 1.2]},
            "gamma_t": 5.449167323}
    res = _run(R.bumped(0.02), _bump_injection(), n_turns=4, ramp=ramp)
    assert res.ledger["injected"] == 30
    assert res.ramp["kinetic_end_mev"] > 800.0
    # (no closed orbit in a ramp: the reference changes within the turn)
    assert np.all(np.isnan(res.injection["closed_orbit_foil"]))
    assert np.all(np.isfinite(res.injection["waveforms"][:4]))


def test_a_cuda_run_in_the_simulator_injects_as_the_cpu():
    """The CUDA path (numba's simulator, a fresh process): the beam synced to
    the host at the foil and back, the injected slots uploaded — the same
    particles and foil record as the CPU's to rounding."""
    import os
    import subprocess
    import sys
    code = f"""
import numpy as np
from tests.ring import injection_rings as R
from linac_gen.ring import RingConfig, RingSimulation
inj = {_bump_injection(n_turns=2, per_turn=3)!r}
out = {{}}
for dev in ("cpu", "cuda"):
    cfg = RingConfig(n_turns=3, source="empty", injection=inj, tune_probes=0, device=dev,
                     snapshot_turns=[3], seed=4)
    r = RingSimulation(R.bumped(0.02), R.booster_beam(), cfg).run()
    out[dev] = (r.snapshots[3]["coords"], r.injection, r.meta["device"])
a, b = out["cpu"], out["cuda"]
assert b[2] == "cuda", b[2]
d = float(np.abs(a[0] - b[0]).max() / np.abs(a[0]).max())
same = all(np.array_equal(a[1][k], b[1][k]) for k in ("injected", "foil_hits", "stripped"))
print("CUDASIM_INJ", d, int(same), flush=True)
"""
    env = dict(os.environ)
    env.update({"NUMBA_ENABLE_CUDASIM": "1", "PYTHONPATH": str(R.ROOT), "OMP_NUM_THREADS": "1"})
    r = subprocess.run([sys.executable, "-c", code], cwd=R.ROOT, env=env, capture_output=True,
                       text=True, timeout=1800)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("CUDASIM_INJ")]
    assert line, (r.returncode, r.stderr[-3000:])
    d, same = float(line[-1].split()[1]), int(line[-1].split()[2])
    assert same == 1 and d <= 1e-14


def test_a_cuda_runs_foil_step_moves_only_the_particles_at_the_foil():
    """The CUDA path (numba's simulator, a fresh process) with circulating
    particles crossing the foil turn after turn and statistics every third
    turn: the GPU finds the particles inside the foil, only they and the
    injected slots cross to the host and back — the foil step never copies
    the beam whole — and the run is the CPU's: the foil's record exactly, the
    particles, the statistics (with slots to spare: the host gets the used
    ones) and the energy left in the foil to rounding.  With frozen space
    charge too, and a GPU run restarted in the middle of the injection ends
    bit for bit as the one not interrupted."""
    import os
    import subprocess
    import sys
    code = f"""
import numpy as np
from tests.ring import injection_rings as R
from linac_gen.ring import RingConfig, RingSimulation
from linac_gen.ring import engine as E
calls = {{"push_host": 0, "gathered": 0, "sync_in_foil": 0, "partial": 0}}
in_foil = [False]
def push_host(self):
    calls["push_host"] += 1
E._CudaTracker.push_host = push_host
f0 = E._FoilStep.__call__
def foil_step(self):
    in_foil[0] = True
    try:
        return f0(self)
    finally:
        in_foil[0] = False
E._FoilStep.__call__ = foil_step
s0 = E._CudaTracker.sync_host
def sync_host(self):
    calls["sync_in_foil"] += int(in_foil[0])
    calls["partial"] += int(self.full and int(self._parts.n_used) < self.n)
    return s0(self)
E._CudaTracker.sync_host = sync_host
g0 = E._CudaTracker.gather
def gather(self, idx):
    calls["gathered"] += int(idx.size)
    return g0(self, idx)
E._CudaTracker.gather = gather
inj = {_bump_injection(n_turns=3, per_turn=40)!r}
out = {{}}
for dev in ("cpu", "cuda"):
    cfg = RingConfig(n_turns=6, source="empty", injection=inj, tune_probes=0, device=dev,
                     snapshot_turns=[6], stats_every=3, seed=4, capacity=200)
    out[dev] = RingSimulation(R.bumped(0.02), R.booster_beam(), cfg).run()
a, b = out["cpu"], out["cuda"]
assert b.meta["device"] == "cuda", b.meta["device"]
Xa, Xb = a.snapshots[6]["coords"], b.snapshots[6]["coords"]
d = float(np.abs(Xa - Xb).max() / np.abs(Xa).max())
ints = all(np.array_equal(a.injection[k], b.injection[k]) for k in (
    "injected", "foil_hits", "foil_hits_injected", "stripped", "missed"))
da, db = a.injection["deposit_mev"], b.injection["deposit_mev"]
dep = float(np.max(np.abs(da - db)) / np.max(np.abs(da)))
circ = int(np.sum(a.injection["foil_hits"]))
rows = np.asarray(a.n_alive) > 0                  # (the ring starts empty)
ma, mb = np.asarray(a.mean)[rows], np.asarray(b.mean)[rows]
ca, cb = np.asarray(a.cov)[rows], np.asarray(b.cov)[rows]
sd = np.sqrt(np.maximum(np.einsum("kii->ki", ca), 1e-300))
st = max(float(np.max(np.abs(ma - mb) / sd)),
         float(np.max(np.abs(ca - cb) / (sd[:, :, None] * sd[:, None, :]))))
st = st if np.array_equal(a.n_alive, b.n_alive) and rows.sum() == 2 else 1.0
print("CUDASIM_FOIL", d, int(ints), dep, circ, calls["push_host"], calls["gathered"],
      int(a.ledger == b.ledger), st, calls["sync_in_foil"], calls["partial"], flush=True)
import tempfile
from pathlib import Path
ck = Path(tempfile.mkdtemp()) / "ck.npz"
inj = {_bump_injection(n_turns=3, per_turn=20)!r}
sc = {{"model": "frozen", "kicks_per_turn": 8, "grid": [16, 16], "refresh_turns": 2}}
base = dict(n_turns=4, source="empty", injection=inj, tune_probes=0, snapshot_turns=[4],
            stats_every=2, seed=5, space_charge=sc)
c = RingSimulation(R.bumped(0.02), R.booster_beam(), RingConfig(device="cpu", **base)).run()
g = RingSimulation(R.bumped(0.02), R.booster_beam(),
                   RingConfig(device="cuda", checkpoint_every=2, **base)).run(checkpoint_path=ck)
r = RingSimulation(R.bumped(0.02), R.booster_beam(),
                   RingConfig(device="cuda", checkpoint_every=2, **base)).run(restart_from=ck)
keys = ("injected", "foil_hits", "foil_hits_injected", "stripped", "missed")
same = all(np.array_equal(c.injection[k], g.injection[k]) for k in keys)
dsc = float(np.abs(c.snapshots[4]["coords"] - g.snapshots[4]["coords"]).max()
            / np.abs(c.snapshots[4]["coords"]).max())
rest = bool(np.array_equal(g.snapshots[4]["coords"], r.snapshots[4]["coords"])
            and all(np.array_equal(g.injection[k], r.injection[k]) for k in keys))
print("CUDASIM_SC", dsc, int(same), int(rest), int(np.sum(c.injection["foil_hits"])), flush=True)
"""
    env = dict(os.environ)
    env.update({"NUMBA_ENABLE_CUDASIM": "1", "PYTHONPATH": str(R.ROOT), "OMP_NUM_THREADS": "1"})
    r = subprocess.run([sys.executable, "-c", code], cwd=R.ROOT, env=env, capture_output=True,
                       text=True, timeout=1800)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("CUDASIM_FOIL")]
    assert line, (r.returncode, r.stderr[-3000:])
    f = line[-1].split()[1:]
    d, ints, dep, circ, pushes, gathered, ledger = (float(f[0]), int(f[1]), float(f[2]),
                                                    int(f[3]), int(f[4]), int(f[5]), int(f[6]))
    stats, sync_in_foil, partial = float(f[7]), int(f[8]), int(f[9])
    assert circ > 0                       # (circulating particles crossed the foil)
    assert pushes == 0 and sync_in_foil == 0 and gathered == circ
    assert partial > 0 and stats <= 1e-12          # (the used slots alone to the host)
    assert ints == 1 and ledger == 1
    assert d <= 1e-14 and dep <= 1e-12
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("CUDASIM_SC")]
    assert line, (r.returncode, r.stderr[-3000:])
    f = line[-1].split()[1:]
    dsc, same, rest, circ_sc = float(f[0]), int(f[1]), int(f[2]), int(f[3])
    assert circ_sc > 0 and same == 1 and rest == 1 and dsc <= 1e-14


def test_the_cuda_foil_mask_takes_the_particles_the_cpu_takes():
    """The GPU's choice of the particles inside the foil (numba's simulator,
    a fresh process) against RingFoil.inside: coordinates on the edges and
    next to them, NaN, infinities, -0.0, every particle state, open sides,
    fewer slots used than there are — the same slots in the same order; a
    slot out of range is refused (a GPU would read past its arrays)."""
    import os
    import subprocess
    import sys
    code = """
import numpy as np
from linac_gen.ring.kernels import cuda as kc
from linac_gen.ring.injection.foil import FoilSettings, RingFoil
kc.kernel()
cuda = kc.cuda
rng = np.random.default_rng(1)
cap = 300
edge = [-0.01, 0.0, 0.01]
special = np.array(edge + [-0.0, np.nan, np.inf, -np.inf, 5e-324, -5e-324]
                   + [np.nextafter(e, t) for e in edge for t in (-1.0, 1.0)])
lows, highs = np.array([None, -0.01, 0.0], dtype=object), np.array([None, 0.0, 0.01], dtype=object)
bad = found = on_edge = 0
for trial in range(40):
    coords = rng.normal(0.0, 0.01, (6, cap))
    k = rng.integers(0, cap, 120)
    coords[0, k[:60]] = rng.choice(special, 60)
    coords[2, k[60:]] = rng.choice(special, 60)
    state = rng.choice(np.array([-1, 0, 1, 2], dtype=np.int8), cap, p=[0.1, 0.1, 0.7, 0.1])
    n = int(rng.integers(0, cap + 1))
    b = [None if v is None else float(v)
         for v in (rng.choice(lows), rng.choice(highs), rng.choice(lows), rng.choice(highs))]
    foil = RingFoil(FoilSettings(name="f", node=0, material="C", thickness_ug_cm2=600.0,
                                 straggling="auto", dedx_model="mip", strip_model="off",
                                 x_min_m=b[0], x_max_m=b[1], y_min_m=b[2], y_max_m=b[3],
                                 map_bins=(10, 10)), mass_mev=938.272)
    dev = [cuda.to_device(np.ascontiguousarray(a))
           for a in (coords, np.zeros(cap), np.ones(cap), np.ones(cap), state)]
    got = kc.foil_slots(dev, n, foil.bounds, cuda.device_array(cap, dtype=np.uint8))
    circ = np.flatnonzero(state[:n] == 1)
    want = circ[foil.inside(coords[0, circ], coords[2, circ])]
    bad += int(not np.array_equal(got, want))
    found += int(want.size)
    on_edge += int(np.isin(coords[0, want], special).sum() + np.isin(coords[2, want], special).sum())
refused = 0
for call in (lambda: kc.foil_slots(dev, cap + 1, (None,) * 4, cuda.device_array(cap, np.uint8)),
             lambda: kc.gather(dev, np.array([0, cap])),
             lambda: kc.put_slots(dev, (coords, np.zeros(cap), np.ones(cap), np.ones(cap), state),
                                  cap - 1, cap + 1)):
    try:
        call()
    except ValueError:
        refused += 1
print("CUDASIM_MASK", bad, found, on_edge, refused, flush=True)
"""
    env = dict(os.environ)
    env.update({"NUMBA_ENABLE_CUDASIM": "1", "PYTHONPATH": str(R.ROOT), "OMP_NUM_THREADS": "1"})
    r = subprocess.run([sys.executable, "-c", code], cwd=R.ROOT, env=env, capture_output=True,
                       text=True, timeout=1800)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("CUDASIM_MASK")]
    assert line, (r.returncode, r.stderr[-3000:])
    bad, found, on_edge, refused = (int(v) for v in line[-1].split()[1:])
    assert bad == 0 and refused == 3
    assert found > 1000 and on_edge > 300          # (measured 1515 and 553)


# ---------------------------------------------------------------------------
# the review round: buckets, the foil's record, frozen space charge, restarts,
# sources, the chopper's bunch rate and boundaries, notes and refusals
# ---------------------------------------------------------------------------
def test_the_filled_buckets_follow_the_rf_period_with_mixed_harmonics():
    """A cavity at twice the others' harmonic listed first: the RF period is
    C/84 (the common harmonic of the cavities with a voltage), so 84 buckets
    less the notch share each turn's particles."""
    lat = R.with_foil_marker(R.load_booster())
    cav = [e for e in lat.elements if type(e).__name__ == "RFGap"]
    cav[0].harmonic = 168
    res = _run(lat, _inj(turns=1, notch_buckets=4), n_turns=1)
    assert res.injection["buckets"] == 80
    w = 2e-3 * R.revolution_time() / 1.602176634e-19 / 80 / 50
    assert res.ledger["injected_charge"] == pytest.approx(50 * w, rel=1e-6)


def test_the_foils_record_counts_the_rings_particles_and_the_electrons_apart(foil_ring):
    from linac_gen.core.constants import M_ELECTRON
    from linac_gen.core.particle import H_MINUS
    res = _run(foil_ring, _inj())
    rec = res.injection
    # one macroparticle stands for I T / (e B n) of the tracked bucket's: the
    # ring's B buckets hold B times that
    w_ring = 2e-3 * R.revolution_time() / 1.602176634e-19 / 50
    crossings = np.sum(rec["foil_hits"]) + np.sum(rec["foil_hits_injected"])
    assert np.sum(rec["foil_hits"]) > 0
    assert rec["hit_map"].sum() == pytest.approx(crossings * w_ring, rel=1e-6)
    # the deposit map: the ionization energy, as the record's
    assert rec["deposit_map_mev"].sum() == pytest.approx(np.sum(rec["deposit_mev"]), rel=1e-12)
    # the stripped electrons apart: (gamma - 1) m_e c^2, gamma the ion's
    e_el = 800.0 / H_MINUS.mass * M_ELECTRON
    assert e_el == pytest.approx(0.4352, abs=1e-4)
    assert rec["foil"]["electron_mev_each"] == pytest.approx(e_el, rel=1e-12)
    n_el = 2 * np.sum(rec["stripped"]) + np.sum(rec["h0"])
    assert np.sum(rec["electron_mev"]) == pytest.approx(n_el * w_ring * e_el, rel=1e-6)
    # (the ionization a crossing: keV, against the two electrons' 0.87 MeV)
    assert np.sum(rec["deposit_mev"]) < 0.01 * np.sum(rec["electron_mev"])


def test_a_frozen_space_charge_model_follows_the_ring_as_it_fills(foil_ring):
    """Measured only at the start (refresh_turns 0), a frozen model would see
    an empty ring: with injection it is measured again every turn that
    injects and on the turn after the last — here, every turn of the run, as
    a model refreshed every turn."""
    sc = {"model": "frozen", "refresh_turns": 0, "kicks_per_turn": 12, "grid": [16, 16],
          "z_bins": 32}
    inj = _inj(turns=2, start_turn=1, particles_per_turn=200)
    kw = {"n_turns": 4, "snapshot_turns": ["last"]}
    a = _run(foil_ring, inj, space_charge=sc, **kw)
    b = _run(foil_ring, inj, space_charge=dict(sc, refresh_turns=1), **kw)
    none = _run(foil_ring, inj, **kw)
    assert np.array_equal(a.snapshots[4]["coords"], b.snapshots[4]["coords"])
    assert not np.array_equal(a.snapshots[4]["coords"], none.snapshots[4]["coords"])


def test_a_restart_while_the_magnets_change_keeps_every_bit(tmp_path):
    from linac_gen.ring import RingConfig, RingSimulation
    inj = _bump_injection(n_turns=6, per_turn=10)
    inj["magnets"][0]["waveform"] = {"kind": "sqrt_end", "t_start_s": 0.0,
                                     "t_end_s": 7.5 * R.revolution_time()}
    kw = dict(n_turns=8, source="empty", injection=inj, tune_probes=1, seed=4,
              snapshot_turns=["last"])
    full = RingSimulation(R.bumped(0.02), R.booster_beam(), RingConfig(**kw)).run()
    ck = tmp_path / "ck.npz"
    part = RingSimulation(R.bumped(0.02), R.booster_beam(),
                          RingConfig(**dict(kw, checkpoint_every=3)))
    assert part.run(checkpoint_path=ck, should_abort=_abort_after(3)).turns_done == 3
    rest = RingSimulation(R.bumped(0.02), R.booster_beam(), RingConfig(**kw)).run(
        restart_from=ck)
    a, b = full.injection, rest.injection
    assert np.array_equal(rest.snapshots[8]["coords"], full.snapshots[8]["coords"])
    # (the closed orbit's warm start in the checkpoint: its record bit for bit)
    for k in ("closed_orbit_foil", "waveforms", "deposit_mev", "electron_mev"):
        assert np.array_equal(a[k], b[k]), k
    assert a["notes"] == b["notes"]
    # the last row: the magnets after the last turn (the waveform at its end) and
    # the closed orbit then, on the axis
    assert np.all(np.isfinite(a["closed_orbit_foil"])) and np.all(np.isfinite(a["waveforms"]))
    assert a["waveforms"][-1, 0] == 0.0
    assert np.max(np.abs(a["closed_orbit_foil"][-1])) < 1e-10      # (the Newton search's 1e-12)
    assert a["closed_orbit_foil"][0, 2] > 0.01


def test_a_foil_elements_strip_model_and_its_note():
    from linac_gen.elements.foil import Foil
    lat = R.load_booster()
    lat.elements.insert(1, Foil(name="f1", thickness_ug_cm2=150.0))
    inj = _inj(turns=1, particles_per_turn=400)
    inj["foil"] = {"at": "f1"}
    res = _run(lat, inj, n_turns=1)
    # (the foil element's own strip model: off — every ion stripped, said so)
    assert res.injection["foil"]["strip_model"] == "off"
    assert res.ledger["waste"] == 0
    assert any("strips every H- ion" in n for n in res.meta["notes"])
    inj["foil"] = {"at": "f1", "strip_model": "two_step"}
    res = _run(lat, inj, n_turns=1)
    assert res.ledger["waste"] > 0
    assert not any("strips every" in n for n in res.meta["notes"])


def _injected_zeta(lat, src):
    """zeta of one injection turn's particles at the foil (no tracking)."""
    from linac_gen.ring import RingConfig
    from linac_gen.ring import particles as P
    from linac_gen.ring.compile import compile_from_config
    from linac_gen.ring.injection.injector import Injection, bucket_length
    inj = _inj(turns=1, particles_per_turn=200)
    inj["source"] = src
    cfg = RingConfig(n_turns=1, source="empty", injection=inj)
    rl = compile_from_config(lat, cfg, beam=R.booster_beam())
    injection = Injection(cfg.injection, rl, lattice=lat, beam_config=R.booster_beam(), seed=3,
                          rf_on=True, period_m=bucket_length(rl, True), n_turns=1)
    parts = P.RingParticles.allocate(200)
    injection.inject(parts, 0, t_turn=0.0, turn_time_s=rl.revolution_time, ref=rl.reference)
    return parts.coords[4, :parts.n_used]


def test_the_chopper_sees_the_linacs_bunch_rate(foil_ring):
    """The source's phases at one frequency, the linac's bunches at another:
    a bunch rate of the revolution frequency brings one linac bunch a turn —
    every particle of the turn on it (the source's own length zero)."""
    src = dict(SRC, sigma_z_m=0.0, frequency_mhz=650.0)
    assert np.unique(_injected_zeta(foil_ring, src)).size > 10        # (650 MHz bunches)
    one = dict(src, bunch_frequency_mhz=1e-6 / R.revolution_time())
    assert np.unique(_injected_zeta(foil_ring, one)).size == 1


def test_a_bunch_on_a_turns_boundary_is_counted_once():
    """A linac bunch rate at a multiple of the revolution frequency puts a
    bunch on every turn's boundary: it goes to the turn that starts there,
    whether the boundary is reached as one turn's end or the next one's start,
    however many turns have run."""
    from linac_gen.ring.injection import chopper as CH
    T = 2.6597e-6
    f = 1234 / T
    # (an absolute tolerance of 1e-9 bunch fails here from turn 23107 on)
    for k in [*range(23100, 23120), *range(23495, 23520), *range(66000, 66020), 10 ** 6,
              10 ** 6 + 1]:
        z = CH.bunch_zeta(k * T, k * T + T, f_linac_hz=f, beta0=0.84, period_m=5.6)
        assert z.size == 1234, (k, z.size)
        assert CH.first_bunch(k * T + T, f) == CH.first_bunch((k + 1) * T, f), k


def test_a_bump_dipole_with_no_field_needs_its_kick():
    from linac_gen.elements.dipole import Dipole
    from linac_gen.ring import RingConfig, RingSimulation
    from linac_gen.ring.compile import RingCompileError

    def ring():
        lat = R.with_foil_marker(R.load_booster())
        b = Dipole(name="bd", angle=0.0, rho=1e9)
        b.length = 300.0                                    # mm: a magnet with no field
        lat.elements.insert(3, b)
        return lat

    inj = _inj(turns=1, particles_per_turn=2,
               magnets=[{"name": "b", "elements": ["bd"], "model": "dipole"}])
    cfg = RingConfig(n_turns=1, source="empty", injection=inj, tune_probes=0)
    with pytest.raises(RingCompileError, match="no field of its own"):
        RingSimulation(ring(), R.booster_beam(), cfg).run()
    inj["magnets"][0].update(plane="x", kick_rad=1e-4)
    cfg = RingConfig(n_turns=1, source="empty", injection=inj, tune_probes=0)
    assert RingSimulation(ring(), R.booster_beam(), cfg).run().turns_done == 1


def test_helix_ascii_files_say_their_species_and_frequency(tmp_path):
    from linac_gen.ring.injection import source as S
    from linac_gen.ring.injection.config import InjectionError, SourceConfig
    from linac_gen.ring.reference import RingReference
    ring = RingReference.from_kinetic("proton", 800.0)
    rng = np.random.default_rng(1)
    head = ["# Linac_Gen distribution file", "# species: proton", "# w_kin_ref: 800.000000 MeV",
            "# phi_ref: 0.000000 deg", "# frequency: 162.500000 MHz",
            "# columns: x(mm) xp(mrad) y(mm) yp(mrad) phi_abs(deg) W_abs(MeV)"]
    rows = [f"{rng.normal(0, 1):.6f} {rng.normal(0, 0.1):.6f} 0.0 0.0 {rng.normal(0, 2):.6f} "
            f"{800.0 + rng.normal(0, 0.05):.6f}" for _ in range(200)]
    path = tmp_path / "beam.txt"
    path.write_text("\n".join(head + rows) + "\n")

    def pool(**kw):
        return S.load_source(SourceConfig(kind="file", path=str(path), **kw), n_needed=10,
                             seed=1, constants=ring.constants)

    p = pool()
    assert p.species == "proton" and p.frequency_mhz == 162.5
    assert abs(S.mean_delta(p, ring)) < 1e-4                  # (a proton at the ring's energy)
    # kinetic_mev moves the beam there (its offsets from the file's energy kept)
    p810 = pool(kinetic_mev=810.0)
    ref810 = RingReference.from_kinetic("proton", 810.0)
    assert p810.reference.kinetic0 == pytest.approx(810.0, rel=1e-14)
    assert S.mean_delta(p810, ring) == pytest.approx(ref810.p0c / ring.p0c - 1.0, rel=2e-3)
    # a file that does not say which particles it holds
    path.write_text("\n".join([h for h in head if "species" not in h] + rows) + "\n")
    with pytest.raises(InjectionError, match="species"):
        pool()
    assert pool(species="proton").species == "proton"


def test_a_chopper_without_a_bunch_rate_is_refused_before_the_run(foil_ring):
    from linac_gen.ring import RingConfig, RingSimulation
    from linac_gen.ring.injection.config import InjectionError
    inj = _inj(start_turn=3, chopper_window_deg=120.0)
    inj["source"] = {k: v for k, v in SRC.items() if k != "frequency_mhz"}
    turns = []
    cfg = RingConfig(n_turns=6, source="empty", injection=inj, tune_probes=0)
    with pytest.raises(InjectionError, match="bunch rate"):
        RingSimulation(foil_ring, R.booster_beam(), cfg).run(
            progress_callback=lambda t, n: turns.append(t))
    assert turns == []


def test_injections_that_bring_little_say_so(foil_ring):
    res = _run(foil_ring, _inj(start_turn=8), n_turns=4)
    assert res.ledger["injected"] == 0 and "all_lost_at_turn" not in res.meta
    assert any("after the run's 4 turn(s)" in n for n in res.meta["notes"])
    res = _run(foil_ring, _inj(turns=10), n_turns=4)
    assert res.ledger["injected"] == 200
    assert any("4 of its 10 turns run" in n for n in res.meta["notes"])
    # a chopper that keeps no linac bunch (bunches every 90 deg of the RF, from
    # 45 deg; a window of 30 deg): one note with the count
    inj = _inj(turns=5, chopper_window_deg=30.0, chopper_phase_deg=45.0)
    inj["source"] = dict(SRC, bunch_frequency_mhz=4 * 84 * 1e-6 / R.revolution_time())
    res = _run(foil_ring, inj, n_turns=5)
    assert res.ledger["injected"] == 0
    notes = [n for n in res.meta["notes"] if "kept no linac bunch" in n]
    assert len(notes) == 1 and "in 5 injection turn(s)" in notes[0]


def test_a_coasting_beam_at_the_start_keeps_its_charge_with_injection():
    """Space charge, no RF, a beam at the start and injection: the starting
    beam fills the ring (the Beam tab's current times the revolution time),
    as without injection."""
    from linac_gen.ring import RingConfig, RingSimulation
    lat = R.with_foil_marker(R.load_booster())
    for e in lat.elements:
        if type(e).__name__ == "RFGap":
            e.voltage = 0.0
    sc = {"model": "frozen", "kicks_per_turn": 12, "grid": [16, 16], "z_bins": 32}
    beam = R.booster_beam(current=5.0, emit_nx=1.0, emit_ny=1.0)
    cfg = RingConfig(n_turns=1, n_particles=100, source="matched", tune_probes=0,
                     space_charge=sc, injection=_inj(turns=1, particles_per_turn=10))
    res = RingSimulation(lat, beam, cfg).run()
    assert res.injection["buckets"] == 1
    start = res.ledger["injected_charge"] - np.sum(res.injection["injected_charge"])
    assert start == pytest.approx(5e-3 * R.revolution_time() / 1.602176634e-19, rel=1e-6)


def test_space_charge_sees_the_charge_of_a_starting_and_an_injected_beam(monkeypatch):
    """A beam at the start (5 mA) and an injected one (2 mA) carry different
    weights: the frozen model's Gaussian, measured on every turn that
    injects and the turn after, has the circulating charge's rms size (the
    particles', counted alike, is 9 % smaller at the end)."""
    import linac_gen.ring.sc.solver as S
    from linac_gen.ring import RingConfig, RingSimulation
    seen = []
    orig = S.SpaceCharge._kick_frozen

    def spy(self, k, beam, probes, *, n_b, refresh):
        orig(self, k, beam, probes, n_b=n_b, refresh=refresh)
        if refresh and k == 0:
            coords, _pt, _rp, _rv, state = beam.arrays
            alive = state == 1
            w, x = beam.weight[alive], coords[0, alive]
            xw = np.sum(w * x) / np.sum(w)
            seen.append((self._frozen_state["gauss"][0][2],
                         math.sqrt(np.sum(w * (x - xw) ** 2) / np.sum(w)), float(np.std(x)),
                         np.unique(w).size))

    monkeypatch.setattr(S.SpaceCharge, "_kick_frozen", spy)
    lat = R.with_foil_marker(R.load_booster())
    beam = R.booster_beam(current=5.0, emit_nx=4.0, emit_ny=4.0, emit_z=5.0, beta_z=100.0)
    cfg = RingConfig(n_turns=7, n_particles=500, source="matched", tune_probes=0,
                     space_charge={"model": "frozen", "kicks_per_turn": 12, "grid": [32, 32]},
                     injection=_inj(turns=6, particles_per_turn=300))
    RingSimulation(lat, beam, cfg).run()
    assert len(seen) == 7 and seen[0][3] == 1 and all(s[3] == 2 for s in seen[1:])
    for s_frozen, s_charge, _s_count, _nw in seen:
        assert s_frozen == pytest.approx(s_charge, rel=1e-9)
    assert seen[-1][2] < 0.95 * seen[-1][1]


def test_a_second_run_does_not_repeat_the_notes(foil_ring):
    from linac_gen.ring import RingConfig, RingSimulation
    sim = RingSimulation(foil_ring, R.booster_beam(),
                         RingConfig(n_turns=2, source="empty", injection=_inj(turns=1),
                                    tune_probes=0))
    a, b = sim.run(), sim.run()
    assert a.meta["notes"] == b.meta["notes"]
    assert sum("foil at foil" in n for n in b.meta["notes"]) == 1
