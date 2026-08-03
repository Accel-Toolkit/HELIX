"""Bunch-train PIC images and the real-boundary RFQ loss model."""
import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.rfq_cell import RfqCell
from linac_gen.pic.pic_solver import PicSolver


def _bunch(n=400, current=5.0, sig_phi_deg=5.0, seed=1, train=False):
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


class TestTrainImages:
    CFG = dict(nx=32, ny=32, nz=32, grid_extent=5.0, grid_mode="adaptive")

    @pytest.fixture(autouse=True)
    def _deterministic_fft(self, monkeypatch):
        # Reduce (not eliminate) run-to-run noise: single-worker FFT.
        # The OpenMP C++ deposit/gather kernels are still not
        # bit-reproducible under full-suite load -- the same lesson
        # that turned the old PIC hash tests into rtol=1e-12 baseline
        # comparisons.  Equality asserts below use rtol=1e-12
        # accordingly; the code-path SPLIT is what these tests pin.
        monkeypatch.setenv("LINAC_GEN_FFT_WORKERS", "1")

    @staticmethod
    def _same(a, b):
        return np.allclose(a, b, rtol=1e-12, atol=1e-18)

    def _kick(self, beam, **cfg_over):
        cfg = SpaceChargeConfig(**{**self.CFG, **cfg_over})
        PicSolver(cfg).kick(beam, 1.0)
        return beam.particles[:, [1, 3, 5]].copy()

    def test_born_bunched_beam_unchanged_by_default(self):
        """bunch_train=False + train_images=None == isolated (bit-id)."""
        k_auto = self._kick(_bunch(train=False))
        k_off = self._kick(_bunch(train=False), train_images=False)
        assert self._same(k_auto, k_off)

    def test_train_beam_gets_images_automatically(self):
        k_auto = self._kick(_bunch(sig_phi_deg=100.0, train=True))
        k_off = self._kick(_bunch(sig_phi_deg=100.0, train=True),
                           train_images=False)
        k_forced = self._kick(_bunch(sig_phi_deg=100.0, train=True),
                              train_images=True)
        assert not self._same(k_auto, k_off)
        assert self._same(k_auto, k_forced)

    def test_short_bunch_uses_isolated_solve(self):
        """σφ < 30°: neighbours negligible, isolated grid finer —
        the train path must fall back bit-identically."""
        k_trn = self._kick(_bunch(sig_phi_deg=5.0, train=True))
        k_iso = self._kick(_bunch(sig_phi_deg=5.0, train=True),
                           train_images=False)
        assert self._same(k_trn, k_iso)

    def test_long_bunch_engages_images(self):
        k_trn = self._kick(_bunch(sig_phi_deg=100.0, train=True))
        k_iso = self._kick(_bunch(sig_phi_deg=100.0, train=True),
                           train_images=False)
        assert not self._same(k_trn, k_iso)

    def test_train_solver_grid_is_tripled(self):
        cfg = SpaceChargeConfig(**self.CFG)
        s = PicSolver(cfg)
        s.kick(_bunch(sig_phi_deg=100.0, train=True), 1.0)
        assert s._train_solver is not None
        assert s._train_solver.config.nz == 3 * cfg.nz
        assert s._train_solver.config.train_images is False


class TestBoundaryLosses:
    """Loss model on a symmetric cell: r0=5, A10=0 (x_lim=y_lim=5),
    Tc=3.75, wall 6 mm.  Voltage 0 isolates the loss logic."""

    def _cell(self, armed):
        c = RfqCell(name="C", voltage_V=0.0, r0_mm=5.0, A10=0.0,
                    modulation=1.0, length_mm=10.0, phi_s_deg=-90.0,
                    cell_type=2, Tc_mm=3.75, n_steps=20)
        if armed:
            c._geom_boundary = True
            c._wall_mm = 6.0
        return c

    def _track_points(self, pts, armed):
        b = Beam(ref=ReferenceParticle(species=H_MINUS, w_kin=0.06,
                                       frequency=162.5),
                 n_particles=len(pts), current=0.0)
        b.particles[:] = 0.0
        b.particles[:, 0] = [p[0] for p in pts]
        b.particles[:, 2] = [p[1] for p in pts]
        cell = self._cell(armed)
        cell.track_rk4(b, 0.5)
        return b.lost.copy()

    def test_open_corner_survives_only_when_armed(self):
        # (5.5, 2.0): outside the two-term box, but in the open corner
        # (outside both tip arcs, bodies, and inside the wall)
        lost_box = self._track_points([(5.5, 2.0)], armed=False)
        lost_geo = self._track_points([(5.5, 2.0)], armed=True)
        assert lost_box[0] and not lost_geo[0]

    def test_tip_arc_kills_in_both_models(self):
        # (5.2, 0.5): inside the x-vane tip arc — dead either way
        assert self._track_points([(5.2, 0.5)], armed=False)[0]
        assert self._track_points([(5.2, 0.5)], armed=True)[0]

    def test_wall_kills_only_when_armed(self):
        # (4.5, 4.2): r = 6.16 mm — inside the box (survives it) but
        # beyond the 6 mm wall
        assert not self._track_points([(4.5, 4.2)], armed=False)[0]
        assert self._track_points([(4.5, 4.2)], armed=True)[0]

    def test_axis_particle_survives_both(self):
        assert not self._track_points([(0.5, 0.3)], armed=False)[0]
        assert not self._track_points([(0.5, 0.3)], armed=True)[0]

    def test_electrode_body_kills_when_armed(self):
        # (9.5, 0.2): behind the x-vane tip arc centre, |y| < Tc
        assert self._track_points([(9.5, 0.2)], armed=True)[0]

    def test_default_cell_not_armed(self):
        c = RfqCell(name="C", voltage_V=0.0, r0_mm=5.0, A10=0.0,
                    modulation=1.0, length_mm=10.0, phi_s_deg=-90.0,
                    cell_type=2)
        assert c._geom_boundary is False and c._wall_mm == 0.0
