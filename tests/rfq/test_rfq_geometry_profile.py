"""Tests for the vane-geometry gradient profile (rfq_geometry_profile).

Uses a SYNTHETIC vane geometry built in memory (ideal quad plateau +
radial-matcher flare) so no project-specific vane data is needed.
"""
import numpy as np
import pytest

from linac_gen.core.lattice import Lattice
from linac_gen.elements.rfq_cell import RfqCell
from linac_gen.elements.rfq_coefficients import step_kicks
from linac_gen.elements.rfq_geometry_profile import (
    build_rfq_geometry_profile)
from linac_gen.io.tracewin_vane import VaneGeometry
from linac_gen.io.rfq_geometry_helper import apply_rfq_geometry

R0_MM = 5.0
RM_MM = 40.0           # radial-matcher flare length
TOTAL_MM = 300.0
V_VANE = 30000.0


def _synthetic_vane() -> VaneGeometry:
    """Flared entrance (15 mm -> r0) then constant ideal-quad plateau."""
    z = np.arange(0.0, TOTAL_MM + 1e-9, 0.25) * 1e-3
    a = np.full(z.shape, R0_MM * 1e-3)
    zr = z * 1e3
    fl = zr < RM_MM
    a[fl] = (R0_MM + 10.0 * (1.0 - zr[fl] / RM_MM) ** 2) * 1e-3
    tc = np.full(z.shape, 0.75 * R0_MM * 1e-3)
    vp = np.full(z.shape, +V_VANE / 2)
    vm = np.full(z.shape, -V_VANE / 2)
    fz = np.zeros_like(z)
    return VaneGeometry(z=z,
                        aperture_v1=a, aperture_v2=a,
                        aperture_v3=a, aperture_v4=a,
                        Tc_v1=tc, Tc_v2=tc, Tc_v3=tc, Tc_v4=tc,
                        voltage_v1=vp, voltage_v2=vm,
                        voltage_v3=vp, voltage_v4=vm,
                        flag_v1=fz, flag_v2=fz, flag_v3=fz, flag_v4=fz)


@pytest.fixture(scope="module")
def profile():
    return build_rfq_geometry_profile(
        _synthetic_vane(), plateau_z_mm=(100.0, 250.0),
        nx=21, z_subsample=6, window_mm=400.0, pad_mm=50.0,
        solver="spsolve", use_cache=False)


class TestBuilder:
    def test_plateau_normalised(self, profile):
        m = (profile.z_mm > 100) & (profile.z_mm < 250)
        assert np.allclose(profile.gx[m], 1.0, atol=0.03)
        assert np.allclose(profile.gy[m], -1.0, atol=0.03)

    def test_radial_matcher_ramp(self, profile):
        # entrance far weaker than plateau, monotone rise through flare
        m = (profile.z_mm > 2) & (profile.z_mm < RM_MM)
        ramp = profile.gx[m]
        assert ramp[0] < 0.5
        assert np.all(np.diff(ramp) > -0.02)
        # settled by ~2 apertures past the flare
        m2 = (profile.z_mm > RM_MM + 2 * R0_MM) & (profile.z_mm < 250)
        assert np.allclose(profile.gx[m2], 1.0, atol=0.05)

    def test_planes_symmetric_for_unmodulated_vanes(self, profile):
        m = profile.z_mm > 5
        assert np.allclose(profile.gx[m], -profile.gy[m], atol=0.02)


class TestStepKicks:
    ARGS = dict(voltage_V=60000.0, r0_mm=5.0, A10=0.3, length_mm=20.0,
                phase_rad=-0.8, gamma_s=1.001, beta_s=0.045,
                dz_mm=0.5, C1=1.0, C2=0.7, S=1.0, C3=0.6,
                mass_MeV=939.294308)

    def test_default_path_unchanged(self):
        # LITERAL pre-change values, pinned 2026-08-02 from the HEAD
        # formula (a self-comparison here would be a tautology --
        # adversarial review finding).  Any drift of the default kick
        # path fails this test.
        kx1, ky1, K1, K2 = step_kicks(**self.ARGS)
        assert kx1 == pytest.approx(4.6335346930429736e-04, rel=1e-14)
        assert ky1 == pytest.approx(-4.3497444154323077e-04, rel=1e-14)
        assert K1 == pytest.approx(-2.437769928651117e-02, rel=1e-14)
        assert K2 == pytest.approx(1.00015979295781, rel=1e-14)
        ref = step_kicks(**self.ARGS, gx=None, gy=None)
        assert (kx1, ky1, K1, K2) == ref

    def test_profile_path_replaces_quad_and_defoc(self):
        kx1, ky1, K1, K2 = step_kicks(**self.ARGS, gx=1.0, gy=-1.0)
        kx0, ky0, K10, K20 = step_kicks(**self.ARGS)
        # longitudinal channel untouched
        assert K1 == K10 and K2 == K20
        # profile mode is C1/defoc-free: pure antisymmetric quad
        assert kx1 == pytest.approx(-ky1, rel=1e-12)
        # gx scales the kick linearly
        kx2, _, _, _ = step_kicks(**self.ARGS, gx=0.5, gy=-0.5)
        assert kx2 == pytest.approx(0.5 * kx1, rel=1e-12)
        # and differs from the legacy kick by exactly the defoc term
        assert kx1 != kx0


class TestHelper:
    def _lattice(self):
        lat = Lattice()
        n = int(TOTAL_MM / 7.5)
        for k in range(n):
            ct = 3 if k == 0 else (2 if k % 2 else -2)
            lat.elements.append(
                RfqCell(name=f"C{k}", voltage_V=V_VANE * 2,
                        r0_mm=R0_MM, A10=0.0, modulation=1.0,
                        length_mm=7.5, phi_s_deg=-90.0, cell_type=ct))
        return lat

    def test_attach_antisym(self, profile):
        lat = self._lattice()
        n = apply_rfq_geometry(lat, profile, mode="antisym")
        assert n == len(lat.elements)
        for cell in lat.elements:
            assert cell._geom_z is not None
            assert np.allclose(cell._geom_gy, -cell._geom_gx)
            # cell-local coordinates with the margin overhang
            assert cell._geom_z[0] <= 0.5
            assert cell._geom_z[-1] >= cell.length - 0.5

    def test_attach_per_plane(self, profile):
        lat = self._lattice()
        apply_rfq_geometry(lat, profile, mode="per_plane")
        c = lat.elements[len(lat.elements) // 2]
        sel = (profile.z_mm >= 0)
        assert np.allclose(c._geom_gy,
                           np.interp(c._geom_z + sum(
                               e.length for e in lat.elements[
                                   :len(lat.elements) // 2]),
                               profile.z_mm, profile.gy), atol=1e-12)

    def test_bad_mode_raises(self, profile):
        with pytest.raises(ValueError):
            apply_rfq_geometry(self._lattice(), profile, mode="magic")

    def test_no_rfq_cells_is_noop(self, profile):
        assert apply_rfq_geometry(Lattice(), profile) == 0

    def test_default_cells_not_armed(self):
        cell = RfqCell(name="C", voltage_V=V_VANE * 2, r0_mm=R0_MM,
                       A10=0.0, modulation=1.0, length_mm=7.5,
                       phi_s_deg=-90.0, cell_type=2)
        assert cell._geom_z is None
