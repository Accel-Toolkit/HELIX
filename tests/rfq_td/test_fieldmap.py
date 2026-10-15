"""Field-map storage and evaluation (tricubic, E = -grad phi)."""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.rfq_td import fieldmap as fm
from linac_gen.rfq_td.fieldmap import RfqFieldMap, cache_dir


def _quadratic_map(a=1500.0, b=-20.0, c=3.0, h=0.1, nx=81):
    """Two blocks of phi = a(x^2 - y^2) + b z + c z^2 (V, mm): tricubic
    Catmull-Rom reproduces any quadratic exactly."""
    xs = h * np.arange(nx)
    blocks, cores = [], []
    for (z0, z1, hz) in ((0.0, 10.0, 0.25), (10.0, 30.0, 0.5)):
        zs = np.arange(z0 - 2 * hz, z1 + 3 * hz - 1e-9, hz)
        X, Y, Z = np.meshgrid(xs, xs, zs, indexing="ij")
        phi = a * (X * X - Y * Y) + b * Z + c * Z * Z
        blocks.append((float(zs[0]), hz, phi.astype(np.float32)))
        cores.append((z0, z1))
    return RfqFieldMap.from_blocks(h, cores, blocks, key="test"), (a, b, c)


@pytest.mark.parametrize("use_kernel", [False, True])
def test_quadratic_potential_exact(use_kernel):
    if use_kernel and not fm._kernel():
        pytest.skip("compiled _rfq_td_kernels not built")
    m, (a, b, c) = _quadratic_map()
    rng = np.random.default_rng(3)
    n = 2000
    x = rng.uniform(-7.5, 7.5, n)
    y = rng.uniform(-7.5, 7.5, n)
    z = rng.uniform(0.1, 29.0, n)
    Ex, Ey, Ez, n_out = m.field(x, y, z, use_kernel=use_kernel)
    assert n_out == 0
    # float32 storage: relative accuracy ~1e-6 of the largest phi
    np.testing.assert_allclose(Ex, -2 * a * x, atol=0.1, rtol=1e-5)
    np.testing.assert_allclose(Ey, 2 * a * y, atol=0.1, rtol=1e-5)
    np.testing.assert_allclose(Ez, -(b + 2 * c * z), atol=0.05, rtol=1e-5)


def test_kernel_matches_numpy_reference():
    if not fm._kernel():
        pytest.skip("compiled _rfq_td_kernels not built")
    m, _ = _quadratic_map()
    # a non-polynomial map, so the two paths really have to agree term by term
    rng = np.random.default_rng(5)
    m.phi = (m.phi + rng.normal(0.0, 50.0, m.phi.size)).astype(np.float32)
    x = rng.uniform(-8.5, 8.5, 5000)
    y = rng.uniform(-8.5, 8.5, 5000)
    z = rng.uniform(-1.0, 31.0, 5000)
    act = rng.random(5000) < 0.9
    a = m.field(x, y, z, act, use_kernel=False)
    b = m.field(x, y, z, act, use_kernel=True)
    assert a[3] == b[3]
    for u, v in zip(a[:3], b[:3]):
        np.testing.assert_allclose(v, u, rtol=1e-13, atol=1e-12)


@pytest.mark.parametrize("use_kernel", [False, True])
def test_outside_region_is_zero_and_counted(use_kernel):
    """Beyond the stored box or after the z span: zero field, counted.
    Before the z span (upstream of the entrance plate, where particles
    start): zero field, not counted.  Non-finite positions: zero field,
    counted (the compiled gather once read them as an undefined index)."""
    if use_kernel and not fm._kernel():
        pytest.skip("compiled kernel not built")
    m, _ = _quadratic_map()
    nan = float("nan")
    x = np.array([0.5, 9.0, 0.5, 0.5, 0.5, nan])
    y = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    z = np.array([5.0, 5.0, -5.0, 40.0, nan, 5.0])
    Ex, Ey, Ez, n_out = m.field(x, y, z, use_kernel=use_kernel)
    assert n_out == 4                      # box, after the span, two NaN
    assert Ex[0] != 0.0
    assert np.all(Ex[1:] == 0.0) and np.all(Ez[1:] == 0.0)


def test_mirror_symmetry_signs():
    m, _ = _quadratic_map()
    x = np.array([1.3, -1.3, 1.3, -1.3])
    y = np.array([0.7, 0.7, -0.7, -0.7])
    z = np.full(4, 12.0)
    Ex, Ey, Ez, _ = m.field(x, y, z)
    np.testing.assert_allclose(Ex, Ex[0] * np.sign(x / x[0]), rtol=1e-12)
    np.testing.assert_allclose(Ey, Ey[0] * np.sign(y / y[0]), rtol=1e-12)
    np.testing.assert_allclose(Ez, Ez[0], rtol=1e-12)


def test_save_load_roundtrip_and_cache_dir(tmp_path, monkeypatch):
    m, _ = _quadratic_map()
    monkeypatch.setenv("HELIX_CACHE_DIR", str(tmp_path / "c"))
    assert cache_dir() == tmp_path / "c" / "rfq_td"
    p = cache_dir() / "map-x.npz"
    m.save(p)
    m2 = RfqFieldMap.load(p)
    assert m2.key == "test"
    np.testing.assert_array_equal(m2.phi, m.phi)
    np.testing.assert_array_equal(m2.core0, m.core0)
    x = np.array([0.4]); y = np.array([0.2]); z = np.array([3.0])
    assert m.field(x, y, z)[0][0] == m2.field(x, y, z)[0][0]


def test_scaled_map():
    m, _ = _quadratic_map()
    s = m.scaled(0.5)
    x = np.array([1.0]); y = np.array([0.5]); z = np.array([4.0])
    np.testing.assert_allclose(s.field(x, y, z)[0], 0.5 * m.field(x, y, z)[0],
                               rtol=1e-6)
