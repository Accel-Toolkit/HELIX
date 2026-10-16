"""A particle that leaves the stored field region (beyond the vane tips,
between the vanes) is lost there.  Before, it flew on in zero field: one did,
to 1 m off axis, and its offset dominated the rms rows along an RFQ."""
from __future__ import annotations

import math

import numpy as np
import pytest

from linac_gen.rfq_td.fieldmap import RfqFieldMap
from linac_gen.rfq_td.geometry import RfqElectrodes
from linac_gen.rfq_td.tracker import (REASON_OUTSIDE, TimeDomainTracker)

MC2 = 938.272       # proton, MeV


def _zero_map(h=0.1, nx=81, z0=-30.0, z1=130.0, hz=0.5):
    zs = np.arange(z0 - 2 * hz, z1 + 3 * hz - 1e-9, hz)
    phi = np.zeros((nx, nx, zs.size), np.float32)
    return RfqFieldMap.from_blocks(h, [(z0, z1)], [(float(zs[0]), hz, phi)])


def _far_electrodes():
    z = np.linspace(0.0, 100.0, 11)
    return RfqElectrodes(z=z, R_x=np.full(11, 50.0), R_y=np.full(11, 50.0),
                         rho_x=np.full(11, 4.0), rho_y=np.full(11, 4.0),
                         V_x=-1.0, V_y=1.0)


def test_outside_is_the_box_within_the_z_span():
    m = _zero_map()                           # box (81 - 2) x 0.1 = 7.9 mm
    assert m.box_half_width == pytest.approx(7.9)
    x = np.array([0.0, 7.85, 7.95, 0.0, -8.0, 7.95, 7.95])
    y = np.array([0.0, -7.85, 0.0, -7.95, 0.0, 0.0, 0.0])
    z = np.array([50.0, 50.0, 50.0, 50.0, 50.0, -40.0, 140.0])
    np.testing.assert_array_equal(
        m.outside(x, y, z), [False, False, True, True, True, False, False])
    # inside the z span, exactly the points field() gives no field and counts
    inz = (z > -30.0) & (z < 130.0)
    _ex, _ey, _ez, n_out = m.field(x[inz], y[inz], z[inz], use_kernel=False)
    assert n_out == int(m.outside(x, y, z)[inz].sum())


def _start(xp, yp, W=0.05, z=-10.0):
    xp, yp = np.asarray(xp, float), np.asarray(yp, float)
    g = 1 + W / MC2
    bg = math.sqrt(g * g - 1)
    uz = bg / np.sqrt(1 + xp ** 2 + yp ** 2)
    n = xp.size
    return (np.zeros(n), np.zeros(n), np.full(n, z), uz * xp, uz * yp, uz)


def test_a_particle_leaving_the_field_region_is_lost_there():
    m = _zero_map()
    # on axis; out through x at z = -10 + 7.9 / 0.1 = 69 mm; out diagonally
    # (both |x| and |y| reach 7.9 mm at z = 88.75 mm)
    X, Y, Z, ux, uy, uz = _start([0.0, 0.1, 0.08], [0.0, 0.0, 0.08])
    trk = TimeDomainTracker(m, _far_electrodes(), freq_mhz=162.5,
                            mass_mev=MC2, charge=1.0, steps_per_period=24)
    res = trk.run(X, Y, Z, ux, uy, uz, 0.0, np.array([20.0, 60.0, 120.0]))
    assert list(res.lost) == [False, True, True]
    assert list(res.loss_reason[1:]) == [REASON_OUTSIDE, REASON_OUTSIDE]
    # lost on the step that took it past the box edge
    r = np.maximum(np.abs(res.loss_xy[1:, 0]), np.abs(res.loss_xy[1:, 1]))
    assert np.all((r >= 7.9) & (r < 8.0)), r
    # ... within one step (v dt = 0.79 mm at 50 keV, 24 steps) of the edge
    dz = 299.792458 * math.sqrt(1 - (MC2 / (MC2 + 0.05)) ** 2) / 162.5e-3 / 24
    assert np.all(res.loss_z[1:] >= [69.0, 88.75])
    assert np.all(res.loss_z[1:] < np.array([69.0, 88.75]) + dz), res.loss_z
    # recorded up to the loss, not after it
    assert np.isfinite(res.rec[1, 1, 0]) and np.isnan(res.rec[2, 1, 0])
    assert np.isfinite(res.rec[2, 0, 0])
    assert res.outside_map_steps == 2          # one step each, then lost


def test_a_map_without_the_region_test_loses_nothing_there():
    """Duck-typed field maps keep the old behaviour."""
    m = _zero_map()

    class NoTest:
        def __init__(self, inner):
            self._m = inner

        def field(self, *a, **k):
            return self._m.field(*a, **k)

        def push(self, *a, **k):
            return self._m.push(*a, **k)

    X, Y, Z, ux, uy, uz = _start([0.1], [0.0])
    trk = TimeDomainTracker(NoTest(m), _far_electrodes(), freq_mhz=162.5,
                            mass_mev=MC2, charge=1.0, steps_per_period=24)
    res = trk.run(X, Y, Z, ux, uy, uz, 0.0, np.array([120.0]))
    assert not res.lost.any() and res.outside_map_steps > 2
