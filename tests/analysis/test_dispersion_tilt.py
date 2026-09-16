"""Per-element analyses honour ``tilt_deg`` like the composed matrix (2026-09-06):
a 90°-tilted horizontal bend disperses in y, and ``dispersion_along_s`` must say so."""
import numpy as np

from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift


def _ref():
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)


def _line(tilt):
    lat = Lattice()
    lat.add(Drift("d1", length=100.0))
    b = Dipole("b", angle=20.0, rho=800.0)
    b.tilt_deg = tilt
    lat.add(b)
    lat.add(Drift("d2", length=100.0))
    return lat


def test_dispersion_along_s_matches_the_composed_matrix_for_a_tilted_bend():
    from linac_gen.analysis.dispersion import dispersion_along_s
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix
    for tilt in (0.0, 90.0, 30.0):
        lat = _line(tilt)
        res = dispersion_along_s(lat, _ref())
        M = compute_transfer_matrix(lat, _ref())
        # the dispersion ray propagated element by element == column 5 of the
        # composed matrix (both in HELIX units; compare shapes not scale)
        eta_end = np.array([res["disp_x_m"][-1], res["disp_y_m"][-1]]) if "disp_x_m" in res else None
        assert eta_end is not None
        col = np.array([M[0, 5], M[2, 5]])
        ratio = eta_end / np.where(np.abs(col) > 1e-15, col, np.nan)
        finite = np.isfinite(ratio)
        assert finite.any()
        np.testing.assert_allclose(ratio[finite], ratio[finite][0], rtol=1e-9)
    res90 = dispersion_along_s(_line(90.0), _ref())
    res0 = dispersion_along_s(_line(0.0), _ref())
    assert abs(res90["disp_y_m"][-1]) > 1e-9 and abs(res90["disp_x_m"][-1]) < 1e-12
    assert abs(res0["disp_x_m"][-1]) > 1e-9 and abs(res0["disp_y_m"][-1]) < 1e-12
