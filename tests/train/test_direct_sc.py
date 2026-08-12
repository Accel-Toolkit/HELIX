"""Multibunch M5 — direct bunch-to-bunch space charge anchors.

Solver level: pattern image factors on the ±1 bunch-train images,
engagement-gate force-on, distinct-neighbour snapshots.  Driver level:
TrainRunner wiring (pattern → per-bunch factors), loud refusals,
teardown, and the zero-coupling protection.

Comparison convention (tests/rfq/test_rfq_train_and_boundary.py): the
OpenMP C++ deposit/gather kernels and alignment-dependent FFT paths are
not bit-reproducible across differing allocation sequences, so equality
asserts use rtol=1e-12; the code-path SPLIT is what these tests pin.
True HEAD-vs-branch bit-identity is probed once per milestone with the
git-stash protocol (OMP_NUM_THREADS=1, identical allocation order).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.config import BeamConfig, SpaceChargeConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.simulation import Simulation
from linac_gen.distributions.factory import create_beam
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.pic.pic_solver import PicSolver
from linac_gen.train import PulsePattern, TrainConfig, TrainRunner
from linac_gen.train.direct_sc import NeighborSnapshotBuffer

CFG = dict(nx=32, ny=32, nz=32, grid_extent=5.0, grid_mode="adaptive")


@pytest.fixture(autouse=True)
def _deterministic_fft(monkeypatch):
    monkeypatch.setenv("LINAC_GEN_FFT_WORKERS", "1")


def _same(a, b):
    # rtol per the rfq fixture convention; atol=1e-15 (not 1e-18)
    # because under FULL-SUITE load the multi-threaded OpenMP deposit
    # reorders atomics run-to-run and near-zero kick components pick up
    # ulp-level absolute noise (observed 2026-08-10: (0,0)-vs-isolated
    # equal in isolation, > atol=1e-18 apart under load).  1e-15 is
    # ~9 orders below the smallest physical kick here.
    return np.allclose(a, b, rtol=1e-12, atol=1e-15)


def _bunch(n=400, current=5.0, sig_phi_deg=100.0, seed=1, train=True):
    rng = np.random.default_rng(seed)
    b = Beam(ref=ReferenceParticle(species=H_MINUS, w_kin=0.06,
                                   frequency=162.5),
             n_particles=n, current=current)
    b.particles[:, 0] = rng.normal(0, 1.0, n)
    b.particles[:, 2] = rng.normal(0, 1.0, n)
    b.particles[:, 4] = rng.normal(0, sig_phi_deg, n)
    b.particles[:, [1, 3, 5]] = 0.0
    b.continuous = False
    b.bunch_train = train
    return b


def _kick(beam, factors=None, force=False, provider=None, recorder=None,
          **cfg_over):
    solver = PicSolver(SpaceChargeConfig(**{**CFG, **cfg_over}))
    solver.train_image_factors = factors
    solver.train_force_engage = force
    solver.train_neighbor_provider = provider
    solver.train_snapshot_recorder = recorder
    solver.kick(beam, 1.0)
    return beam.particles[:, [1, 3, 5]].copy(), solver


def _snapshot_of(beam):
    """A full-resolution NeighborSnapshotBuffer-format snapshot."""
    al = beam.alive_mask
    return {"xyphi": beam.particles[al][:, (0, 2, 4)].copy(),
            "n_alive": float(np.count_nonzero(al)),
            "ref_phi_s": float(beam.ref.phi_s),
            "ref_w_kin": float(beam.ref.w_kin)}


# ======================================================================
# Anchor 1+2 — the load-bearing protections
# ======================================================================
class TestFactorProtections:
    def test_factors_one_one_identical_to_default(self):
        """(1, 1) must reproduce the default (None) train path — the
        machinery validated against TW/Toutatis (PXIE 5 mA benchmark)."""
        k_def, _ = _kick(_bunch())
        k_11, _ = _kick(_bunch(), factors=(1.0, 1.0))
        assert _same(k_def, k_11)

    def test_factors_zero_zero_identical_to_isolated(self):
        """(0, 0): both neighbours chopped — must route to the isolated
        solve on the finer single-bunch grid, bit-identically (the
        companion train solver is never even built)."""
        k_00, s00 = _kick(_bunch(), factors=(0.0, 0.0))
        k_iso, _ = _kick(_bunch(), train_images=False)
        assert _same(k_00, k_iso)
        assert s00._train_solver is None

    def test_factors_identity_holds_in_fixed_grid_mode(self):
        """Dual-regime rule: the (1,1) ≡ None protection must hold in
        BOTH grid modes — fixed mode freezes the companion grid on the
        first kick, a separate code branch."""
        k_def, _ = _kick(_bunch(), grid_mode="fixed")
        k_11, _ = _kick(_bunch(), factors=(1.0, 1.0), grid_mode="fixed")
        assert _same(k_def, k_11)

    def test_fractional_factors_are_exactly_linear(self):
        """Deposit → Poisson → gather is linear in charge, and factors
        in (0, 1] keep the particle stack (hence the adaptive grid)
        identical — so the kick must be exactly linear in the factors:
        k(1,1) − k(.75,.75) == k(.75,.75) − k(.5,.5) to the FFT noise
        floor.  Pins the per-particle weight arithmetic end to end."""
        k_11, _ = _kick(_bunch(), factors=(1.0, 1.0))
        k_75, _ = _kick(_bunch(), factors=(0.75, 0.75))
        k_50, _ = _kick(_bunch(), factors=(0.5, 0.5))
        d_hi = k_11 - k_75
        d_lo = k_75 - k_50
        assert np.abs(d_hi).max() > 0.0            # weights are live
        assert np.allclose(d_hi, d_lo, rtol=1e-9, atol=1e-13)


# ======================================================================
# Anchor 3 — chopped-edge asymmetry, sign hand-checked
# ======================================================================
class TestChoppedEdgeAsymmetry:
    """Sign chain (measured before implementation, 2026-08-10):
    z = −dphi·βλ/360, so the LEADING image (neighbour injected one slot
    earlier) sits at dphi −360h → z > 0 (ahead) and the TRAILING image
    at +360h → z < 0 (behind).  Like charges repel (unsigned deposit +
    |Z| kick — pic_solver sign-convention block), so a bunch missing
    its leading neighbour ((0, 1): first bunch after a chopped gap)
    feels the unbalanced push of the surviving behind-neighbour —
    FORWARD, mean dW kick > 0; the last bunch before a gap ((1, 0))
    is pushed backward, mean dW kick < 0.  Mid-train (1, 1) cancels by
    symmetry."""

    def test_first_bunch_after_gap_pushed_forward(self):
        k_01, _ = _kick(_bunch(), factors=(0.0, 1.0))
        k_11, _ = _kick(_bunch(), factors=(1.0, 1.0))
        assert k_11[:, 2].mean() == pytest.approx(0.0, abs=1e-9)
        assert k_01[:, 2].mean() - k_11[:, 2].mean() > 1e-7

    def test_last_bunch_before_gap_pushed_backward(self):
        k_10, _ = _kick(_bunch(), factors=(1.0, 0.0))
        k_11, _ = _kick(_bunch(), factors=(1.0, 1.0))
        assert k_10[:, 2].mean() - k_11[:, 2].mean() < -1e-7

    def test_edge_cases_mirror_each_other(self):
        """One-sided pushes are equal and opposite for a symmetric
        bunch — pins that the two image blocks carry equal charge."""
        k_01, _ = _kick(_bunch(), factors=(0.0, 1.0))
        k_10, _ = _kick(_bunch(), factors=(1.0, 0.0))
        m_fwd = k_01[:, 2].mean()
        m_bwd = k_10[:, 2].mean()
        assert m_fwd == pytest.approx(-m_bwd, rel=1e-6)


# ======================================================================
# Anchor 4 — engagement-gate force-on
# ======================================================================
class TestForceEngage:
    def test_short_bunch_engages_only_when_forced(self):
        """σφ = 5° « the 35° gate: today's auto path silently runs the
        isolated solve; force-on must actually engage the images."""
        k_def, s_def = _kick(_bunch(sig_phi_deg=5.0))
        k_iso, _ = _kick(_bunch(sig_phi_deg=5.0), train_images=False)
        assert _same(k_def, k_iso)          # documented auto behaviour
        assert not s_def._train_ever_engaged
        k_f, s_f = _kick(_bunch(sig_phi_deg=5.0), force=True)
        assert s_f._train_ever_engaged
        assert s_f._train_solver is not None
        assert not _same(k_f, k_iso)

    def test_born_bunched_beam_engages_under_force(self):
        """bunch_train=False (born bunched) never enters the train path
        on auto; force-on declares it part of a real train."""
        _, s_auto = _kick(_bunch(train=False))
        assert not s_auto._train_ever_engaged
        k_f, s_f = _kick(_bunch(train=False), force=True)
        assert s_f._train_ever_engaged
        k_iso, _ = _kick(_bunch(train=False), train_images=False)
        assert not _same(k_f, k_iso)

    def test_explicit_train_images_false_beats_force(self):
        """config.train_images=False is an explicit user decision — the
        force flag must not override it (the driver refuses the
        combination instead)."""
        k_f, s_f = _kick(_bunch(), force=True, train_images=False)
        k_iso, _ = _kick(_bunch(), train_images=False)
        assert _same(k_f, k_iso)
        assert not s_f._train_ever_engaged


# ======================================================================
# Anchor 5 — distinct neighbours (M5b)
# ======================================================================
class TestDistinctNeighbors:
    def test_identical_snapshot_reproduces_self_image(self):
        """A full-resolution snapshot of the SAME bunch must reproduce
        the exact-copy leading image.  The companion solver's inputs
        are bit-identical (verified while building the milestone); the
        residual is the alignment-dependent FFT floor (~5e-16), far
        below any physics."""
        snap = _snapshot_of(_bunch())
        k_dist, _ = _kick(_bunch(), factors=(1.0, 1.0),
                          provider=lambda o: snap)
        k_11, _ = _kick(_bunch(), factors=(1.0, 1.0))
        assert np.allclose(k_dist, k_11, rtol=1e-9, atol=1e-13)

    def test_subsampled_snapshot_within_deposition_noise(self):
        """A 4× subsampled neighbour (charge-conserving weights) stays
        within deposition noise of the exact copy: the leading image is
        a full train period away, so its field at the bunch is smooth."""
        full = _snapshot_of(_bunch())
        sel = np.linspace(0.0, full["xyphi"].shape[0] - 1, 100)
        sub = {"xyphi": full["xyphi"][sel.astype(np.int64)],
               "n_alive": full["n_alive"],
               "ref_phi_s": full["ref_phi_s"],
               "ref_w_kin": full["ref_w_kin"]}
        k_sub, _ = _kick(_bunch(), factors=(1.0, 1.0),
                         provider=lambda o: sub)
        k_11, _ = _kick(_bunch(), factors=(1.0, 1.0))
        # kick-field deviation bounded well below the image's own effect
        k_iso, _ = _kick(_bunch(), train_images=False)
        image_effect = np.abs(k_11 - k_iso).max()
        assert np.abs(k_sub - k_11).max() < 0.5 * image_effect
        # net longitudinal asymmetry induced by subsampling stays a
        # small fraction of a SINGLE image's push (the physical scale —
        # the symmetric (1,1) mean is ~0 and no basis for a rel-compare)
        k_one, _ = _kick(_bunch(), factors=(1.0, 0.0))
        one_sided_push = abs(k_one[:, 2].mean())
        assert abs(k_sub[:, 2].mean() - k_11[:, 2].mean()) \
            < 0.2 * one_sided_push

    def test_neighbor_energy_cannot_enter_documents_shared_boost(self):
        """SHARED-BOOST APPROXIMATION, documented: the image deposit
        reads only (x, y, dphi), so even a huge neighbour ref-energy
        offset changes NOTHING — first order in ΔW/W by construction;
        a neighbour's true γ never enters its image field."""
        snap = _snapshot_of(_bunch())
        shifted = dict(snap)
        shifted["ref_w_kin"] = snap["ref_w_kin"] + 0.05   # ~83 % of W
        k_a, _ = _kick(_bunch(), factors=(1.0, 1.0),
                       provider=lambda o: snap)
        k_b, _ = _kick(_bunch(), factors=(1.0, 1.0),
                       provider=lambda o: shifted)
        assert np.allclose(k_a, k_b, rtol=1e-9, atol=1e-13)

    def test_ref_phase_difference_shifts_the_image(self):
        """The phase-frame offset is 360h + (ref-phase difference): a
        snapshot recorded under a shifted ref clock must move the image
        by exactly that shift — pin by equivalence with shifting the
        particle phases themselves."""
        snap = _snapshot_of(_bunch())
        by_ref = dict(snap)
        by_ref["ref_phi_s"] = snap["ref_phi_s"] + 30.0
        by_parts = dict(snap)
        by_parts["xyphi"] = snap["xyphi"].copy()
        by_parts["xyphi"][:, 2] += 30.0
        k_a, _ = _kick(_bunch(), factors=(1.0, 1.0),
                       provider=lambda o: by_ref)
        k_b, _ = _kick(_bunch(), factors=(1.0, 1.0),
                       provider=lambda o: by_parts)
        assert np.allclose(k_a, k_b, rtol=1e-9, atol=1e-13)

    def test_lossy_neighbor_deposits_less(self):
        """Loss-scaled neighbour charge: n_alive halved (same subsample)
        halves the leading push — the physics gain over self-copies."""
        snap = _snapshot_of(_bunch())
        half = dict(snap)
        half["n_alive"] = 0.5 * snap["n_alive"]
        k_full, _ = _kick(_bunch(), factors=(1.0, 0.0),
                          provider=lambda o: snap)
        k_half, _ = _kick(_bunch(), factors=(1.0, 0.0),
                          provider=lambda o: half)
        assert k_half[:, 2].mean() == pytest.approx(
            0.5 * k_full[:, 2].mean(), rel=0.02)

    def test_recorder_called_at_engaged_kicks_only(self):
        seen = []

        def rec(ordinal, beam):
            seen.append((ordinal, int(beam.n_alive)))

        _kick(_bunch(sig_phi_deg=5.0), recorder=rec)      # not engaged
        assert seen == []
        _kick(_bunch(), factors=(1.0, 1.0), recorder=rec)  # engaged
        assert seen == [(0, 400)]

    def test_snapshot_buffer_roundtrip_and_fallback_warning(self):
        buf = NeighborSnapshotBuffer(n_sub=64)
        b = _bunch(n=200)
        buf.record(3, b)
        assert buf.n_recorded == 1
        snap = buf.snapshot(3)
        assert snap["xyphi"].shape == (64, 3)
        assert snap["n_alive"] == 200.0
        # deterministic subsample: same call, same rows
        buf2 = NeighborSnapshotBuffer(n_sub=64)
        buf2.record(3, _bunch(n=200))
        assert np.array_equal(snap["xyphi"], buf2.snapshot(3)["xyphi"])
        with pytest.warns(UserWarning, match="falling back"):
            assert buf.snapshot(99) is None


# ======================================================================
# Torch / halo backends — explicit refusal (no silent inertness)
# ======================================================================
def test_torch_backend_raises_on_train_controls():
    pytest.importorskip("torch")
    from linac_gen.pic.torch.solver import TorchPicSolver
    cfg = SpaceChargeConfig(**{**CFG, "sc_backend": "torch"})
    s = TorchPicSolver(cfg)
    s.train_image_factors = (1.0, 0.0)
    with pytest.raises(NotImplementedError, match="torch"):
        s.kick(_bunch(), 1.0)
    s2 = TorchPicSolver(cfg)
    s2.train_force_engage = True
    with pytest.raises(NotImplementedError, match="torch"):
        s2.kick(_bunch(), 1.0)


def test_halo_backend_raises_on_train_controls():
    """HaloPicSolver re-implements kick() without the train machinery —
    inherited-but-ignored controls must refuse, not silently run
    isolated physics."""
    from linac_gen.pic.ml.solver import HaloPicSolver
    s = HaloPicSolver(SpaceChargeConfig(**CFG))
    s.train_image_factors = (1.0, 1.0)
    with pytest.raises(NotImplementedError, match="halo"):
        s.kick(_bunch(), 1.0)


# ======================================================================
# Driver wiring
# ======================================================================
def _lattice():
    lat = Lattice()
    lat.add(Quadrupole("QF", 40.0, gradient=18.0, aperture=20.0))
    lat.add(Drift("D1", 30.0, aperture=20.0))
    lat.add(RFGap(name="G", voltage=0.05, phase=-30.0, frequency=162.5))
    lat.add(Drift("D2", 30.0, aperture=20.0))
    lat.add(Quadrupole("QD", 40.0, gradient=-18.0, aperture=20.0))
    lat.add(Drift("D3", 30.0, aperture=20.0))
    return lat


def _cfg(**kw):
    base = dict(species="proton", energy=3.0, frequency=162.5,
                current=5.0, n_particles=300, distribution="waterbag",
                emit_nx=0.25, alpha_x=0.0, beta_x=0.4,
                emit_ny=0.25, alpha_y=0.0, beta_y=0.4,
                emit_z=0.15, alpha_z=0.0, beta_z=1.2)
    base.update(kw)
    return BeamConfig(**base)


def _sc(**kw):
    return SpaceChargeConfig(**{**dict(nx=16, ny=16, nz=16,
                                       grid_extent=4.0,
                                       grid_mode="adaptive"), **kw})


def _tc(pattern="1*3", **kw):
    base = dict(bunch_frequency_MHz=162.5,
                pattern=PulsePattern.from_rle(pattern))
    base.update(kw)
    return TrainConfig(**base)


class TestDriverRefusals:
    def test_direct_sc_needs_mp_mode(self):
        from linac_gen.train.config import TrainPhysics
        for mode in ("envelope", "fast"):
            with pytest.raises(ValueError, match="mode='mp'"):
                _tc(mode=mode, physics=TrainPhysics(direct_sc=True))

    def test_direct_sc_needs_sc_config(self):
        from linac_gen.train.config import TrainPhysics
        tc = _tc(physics=TrainPhysics(direct_sc=True))
        with pytest.raises(ValueError, match="3-D PIC"):
            TrainRunner(_lattice(), _cfg(), tc, sc_config=None)
        with pytest.raises(ValueError, match="3-D PIC"):
            TrainRunner(_lattice(), _cfg(), tc, sc_config="off")

    def test_direct_sc_refuses_torch_backend(self):
        from linac_gen.train.config import TrainPhysics
        tc = _tc(physics=TrainPhysics(direct_sc=True))
        with pytest.raises(NotImplementedError, match="torch"):
            TrainRunner(_lattice(), _cfg(), tc,
                        sc_config=_sc(sc_backend="torch"))
        with pytest.raises(ValueError, match="numpy"):
            TrainRunner(_lattice(), _cfg(), tc,
                        sc_config=_sc(sc_backend="halo"))

    def test_direct_sc_refuses_explicit_images_off(self):
        from linac_gen.train.config import TrainPhysics
        tc = _tc(physics=TrainPhysics(direct_sc=True))
        with pytest.raises(ValueError, match="train_images"):
            TrainRunner(_lattice(), _cfg(), tc,
                        sc_config=_sc(train_images=False))

    def test_neighbors_and_subsample_validated(self):
        with pytest.raises(ValueError, match="direct_sc_neighbors"):
            _tc(direct_sc_neighbors="exotic")
        with pytest.raises(ValueError, match="direct_sc_subsample"):
            _tc(direct_sc_subsample=2)


class TestDriverWiring:
    def test_pattern_maps_to_per_bunch_factors(self):
        """1101 → slot 0 (0,1); slot 1 (1,0); slot 3 (0,0) — leading =
        earlier slot, trailing = later slot, pulse edges lose the
        corresponding image.  Runs through the PUBLIC run_train entry
        point (bugs live in the seams)."""
        from linac_gen.train import run_train
        from linac_gen.train.config import TrainPhysics
        tc = _tc(pattern="1*2 0*1 1*1",
                 physics=TrainPhysics(direct_sc=True),
                 direct_sc_force_engage=True)
        res = run_train(_lattice(), _cfg(), tc, sc_config=_sc())
        assert res.direct_sc == {0: (0.0, 1.0), 1: (1.0, 0.0),
                                 3: (0.0, 0.0)}

    def test_forced_train_differs_from_singles_and_edges_differ(self):
        """Force-engaged direct SC must actually change the physics —
        and the chopped edge (0,1)/(1,0) bunches must differ from the
        mid-train (1,1) bunch."""
        from linac_gen.train.config import TrainPhysics
        cfg = _cfg()
        sc = _sc()
        tc = _tc(pattern="1*3", physics=TrainPhysics(direct_sc=True),
                 direct_sc_force_engage=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = TrainRunner(_lattice(), cfg, tc, sc_config=sc).run()
        assert not any("never engaged" in str(w.message) for w in caught)
        assert res.direct_sc == {0: (0.0, 1.0), 1: (1.0, 1.0),
                                 2: (1.0, 0.0)}
        single = Simulation(_lattice(), create_beam(cfg, seed=tc.seed),
                            space_charge=sc).run()
        mid = res.bunch_results[1]
        assert not np.array_equal(np.asarray(mid.sigma_phi, float),
                                  np.asarray(single.sigma_phi, float))
        first = res.bunch_results[0]
        assert not np.array_equal(np.asarray(first.sigma_phi, float),
                                  np.asarray(mid.sigma_phi, float))

    def test_never_engaged_warns_and_results_match_singles(self):
        """direct_sc on, no force, born-bunched short beam: the gate
        never engages — the run must SAY so, and (factors inert) the
        bunches must equal independent singles (anchor 6 corollary)."""
        from linac_gen.train.config import TrainPhysics
        cfg = _cfg()
        sc = _sc()
        tc = _tc(pattern="1*2", physics=TrainPhysics(direct_sc=True))
        with pytest.warns(UserWarning, match="never engaged"):
            res = TrainRunner(_lattice(), cfg, tc, sc_config=sc).run()
        single = Simulation(_lattice(), create_beam(cfg, seed=tc.seed),
                            space_charge=sc).run()
        for r in res.bunch_results:
            assert np.array_equal(np.asarray(r.sigma_x, float),
                                  np.asarray(single.sigma_x, float))

    def test_zero_coupling_direct_sc_off(self):
        """Anchor 6: direct_sc OFF + PIC on = M4 bit-identity to
        independent singles, and no factor provenance is recorded."""
        cfg = _cfg()
        sc = _sc()
        tc = _tc(pattern="1*2")
        res = TrainRunner(_lattice(), cfg, tc, sc_config=sc).run()
        assert res.direct_sc == {}
        single = Simulation(_lattice(), create_beam(cfg, seed=tc.seed),
                            space_charge=sc).run()
        for r in res.bunch_results:
            for k in ("s", "sigma_x", "sigma_y", "sigma_phi", "sigma_w",
                      "ref_w_kin"):
                assert np.array_equal(np.asarray(getattr(r, k), float),
                                      np.asarray(getattr(single, k),
                                                 float)), k

    def test_teardown_complete_and_config_untouched(self):
        """The caller's sc_config must never grow train state (factors
        ride the per-bunch solver via pic_setup_hook), and the runner
        must drop its buffers in the finally teardown."""
        from linac_gen.train.config import TrainPhysics
        sc = _sc()
        tc = _tc(pattern="1*2", physics=TrainPhysics(direct_sc=True),
                 direct_sc_force_engage=True,
                 direct_sc_neighbors="distinct", direct_sc_subsample=64)
        runner = TrainRunner(_lattice(), _cfg(), tc, sc_config=sc)
        runner.run()
        assert not hasattr(sc, "train_image_factors")
        assert not hasattr(sc, "train_force_engage")
        assert runner._dsc_factors is None
        assert runner._dsc_recorder is None
        assert runner._dsc_provider is None
        assert runner._last_pic is None

    def test_distinct_mode_close_to_images_for_identical_bunches(self):
        """Driver-level anchor-5 corollary: for same-config bunches,
        distinct neighbours stay CLOSE to the images mode (measured
        ~0.3 % end-σ here).  The residual is NOT pure noise: it holds
        the real distinct-neighbour signal (bunch 0 evolved under its
        own (0, 1) edge factors, and ITS snapshot feeds bunch 1's
        leading image) plus subsample noise — the quantitative basis of
        the M5b severability call (marginal gain for identical-bunch
        trains).  The strict identical-snapshot ≡ self-image anchor is
        the solver-level test above."""
        from linac_gen.train.config import TrainPhysics
        cfg = _cfg()
        sc = _sc()
        kw = dict(physics=TrainPhysics(direct_sc=True),
                  direct_sc_force_engage=True)
        res_img = TrainRunner(_lattice(), cfg, _tc(pattern="1*2", **kw),
                              sc_config=sc).run()
        res_dst = TrainRunner(
            _lattice(), cfg,
            _tc(pattern="1*2", direct_sc_neighbors="distinct",
                direct_sc_subsample=256, **kw),
            sc_config=sc).run()
        for r_i, r_d in zip(res_img.bunch_results, res_dst.bunch_results):
            si = np.asarray(r_i.sigma_x, float)
            sd = np.asarray(r_d.sigma_x, float)
            assert np.allclose(si, sd, rtol=1e-2), \
                np.abs(sd / si - 1.0).max()
        # and it genuinely differs (the distinct path is live)
        s1_img = np.asarray(res_img.bunch_results[1].sigma_x, float)
        s1_dst = np.asarray(res_dst.bunch_results[1].sigma_x, float)
        assert not np.array_equal(s1_img, s1_dst)
