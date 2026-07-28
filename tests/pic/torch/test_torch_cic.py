"""M2 — parity + autograd tests for the torch CIC deposit/gather kernels.

The torch kernels must reproduce the numpy CIC kernels to round-off,
conserve charge, be adjoint as a deposit/gather pair, and have gradients
that match finite differences.
"""
import numpy as np
import torch

from linac_gen.pic.charge_deposition import deposit_cic
from linac_gen.pic.field_interpolation import interpolate_cic
from linac_gen.pic.torch.deposition import deposit_cic_torch
from linac_gen.pic.torch.interpolation import interpolate_cic_torch

GMIN = np.array([-3.0, -2.0, -5.0])
GMAX = np.array([3.0, 2.0, 5.0])


def _random_setup(n=16, npart=200, seed=0):
    """Random positions anywhere inside the grid (for parity tests)."""
    rng = np.random.default_rng(seed)
    ng = np.array([n, n, n], dtype=np.int64)
    pos = rng.uniform(GMIN * 0.7, GMAX * 0.7, size=(npart, 3))
    q = rng.uniform(0.5, 1.5, size=npart)
    return ng, pos, q


def _safe_setup(n=14, npart=30, seed=0):
    """Particles parked at safe fractional offsets [0.25, 0.75] inside
    interior cells — a finite-difference step never crosses a cell
    boundary, so autograd and central FD agree to round-off."""
    rng = np.random.default_rng(seed)
    ng = np.array([n, n, n], dtype=np.int64)
    dx = (GMAX - GMIN) / (ng - 1)
    cells = rng.integers(2, n - 4, size=(npart, 3))
    frac = rng.uniform(0.25, 0.75, size=(npart, 3))
    pos = GMIN + (cells + frac) * dx
    q = rng.uniform(0.5, 1.5, size=npart)
    return ng, pos, q


def test_deposit_parity():
    ng, pos, q = _random_setup(seed=0)
    rho_n = deposit_cic(pos, q, GMIN, GMAX, ng)
    rho_t = deposit_cic_torch(pos, q, GMIN, GMAX, ng).numpy()
    np.testing.assert_allclose(rho_t, rho_n, rtol=1e-12,
                               atol=1e-12 * np.abs(rho_n).max())


def test_interpolate_parity():
    ng, pos, q = _random_setup(seed=1)
    n = int(ng[0])
    rng = np.random.default_rng(2)
    fx, fy, fz = (rng.random((n, n, n)) for _ in range(3))
    e_n = interpolate_cic(fx, fy, fz, pos, GMIN, GMAX, ng)
    e_t = interpolate_cic_torch(fx, fy, fz, pos, GMIN, GMAX, ng).numpy()
    np.testing.assert_allclose(e_t, e_n, rtol=1e-12,
                               atol=1e-12 * np.abs(e_n).max())


def test_charge_conservation():
    ng, pos, q = _random_setup(seed=3)
    rho = deposit_cic_torch(pos, q, GMIN, GMAX, ng)
    dx = (GMAX - GMIN) / (ng - 1)
    cell_vol = float(dx[0] * dx[1] * dx[2])
    np.testing.assert_allclose(float(rho.sum()) * cell_vol,
                               float(q.sum()), rtol=1e-12)


def test_deposit_gather_adjoint():
    """<deposit(pos,q), F> == <q, gather(F,pos)> — the deposit and gather
    share weights, so they are exact adjoints."""
    ng, pos, q = _random_setup(n=14, npart=80, seed=7)
    n = int(ng[0])
    fld = torch.as_tensor(np.random.default_rng(8).random((n, n, n)),
                          dtype=torch.float64)
    rho = deposit_cic_torch(pos, q, GMIN, GMAX, ng)
    dx = (GMAX - GMIN) / (ng - 1)
    cell_vol = float(dx[0] * dx[1] * dx[2])
    lhs = float((rho * cell_vol * fld).sum())          # charge-on-grid . F
    gathered = interpolate_cic_torch(fld, fld, fld, pos, GMIN, GMAX, ng)
    rhs = float((torch.as_tensor(q, dtype=torch.float64)
                 * gathered[:, 0]).sum())
    np.testing.assert_allclose(lhs, rhs, rtol=1e-11)


def test_autograd_deposit_vs_fd():
    """d(sum rho^2)/d(positions): autograd vs central finite differences."""
    ng, pos, q = _safe_setup(n=14, npart=24, seed=4)
    qt = torch.as_tensor(q, dtype=torch.float64)

    def loss_fn(p):
        rho = deposit_cic_torch(p, qt, GMIN, GMAX, ng)
        return (rho ** 2).sum()

    post = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
    loss_fn(post).backward()
    g_auto = post.grad.detach().clone()

    base = post.detach()
    eps = 1e-6
    g_fd = torch.zeros_like(base)
    for i in range(base.shape[0]):
        for d in range(3):
            pp = base.clone(); pp[i, d] += eps
            pm = base.clone(); pm[i, d] -= eps
            g_fd[i, d] = (loss_fn(pp) - loss_fn(pm)) / (2 * eps)
    np.testing.assert_allclose(g_auto.numpy(), g_fd.numpy(), rtol=1e-6,
                               atol=1e-7 * float(g_fd.abs().max()))


def test_autograd_interpolate_vs_fd():
    """d(sum E^2)/d(field): autograd vs central FD; positions grad finite."""
    ng, pos, q = _safe_setup(n=14, npart=20, seed=5)
    n = int(ng[0])
    fld0 = np.random.default_rng(6).random((n, n, n))

    def loss_fn(field, p):
        e = interpolate_cic_torch(field, field, field, p, GMIN, GMAX, ng)
        return (e ** 2).sum()

    fld = torch.tensor(fld0, dtype=torch.float64, requires_grad=True)
    post = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
    loss_fn(fld, post).backward()

    base_f = fld.detach()
    eps = 1e-6
    for (i, j, k) in [(5, 5, 5), (3, 8, 4), (9, 2, 7)]:
        fp = base_f.clone(); fp[i, j, k] += eps
        fm = base_f.clone(); fm[i, j, k] -= eps
        fd = (loss_fn(fp, post.detach())
              - loss_fn(fm, post.detach())) / (2 * eps)
        np.testing.assert_allclose(float(fld.grad[i, j, k]), float(fd),
                                   rtol=1e-6, atol=1e-9)
    assert torch.isfinite(post.grad).all()
    assert post.grad.abs().sum() > 0
