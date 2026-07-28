"""HALO-PIC M3 core tests: bit-identity, basis conservation, anchors,
K controller, config validation."""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle


def _beam(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    ref = ReferenceParticle(species=H_MINUS, w_kin=2.12, frequency=162.5)
    b = Beam(ref=ref, n_particles=n, current=5.0)
    b.particles[:, 0] = rng.normal(0, 1.5, n)
    b.particles[:, 2] = rng.normal(0, 1.2, n)
    b.particles[:, 4] = rng.normal(0, 15.0, n)
    b.particles[:, 1] = rng.normal(0, 0.5, n)
    b.particles[:, 3] = rng.normal(0, 0.5, n)
    b.particles[:, 5] = rng.normal(0, 0.005, n)
    return b


def _sc(backend="halo", **halo):
    return SpaceChargeConfig(nx=24, ny=24, nz=24, sc_backend=backend,
                             halo=halo or None)


def test_alpha_zero_bit_identical_to_picsolver():
    """No net, anchors off => byte-identical to the production PicSolver
    over a multi-kick sequence (the house-rule baseline test)."""
    from linac_gen.pic.pic_solver import PicSolver
    from linac_gen.pic.ml.solver import HaloPicSolver
    b1, b2 = _beam(seed=3), _beam(seed=3)
    ref_solver = PicSolver(_sc("numpy"))
    halo_solver = HaloPicSolver(_sc("halo", anchors=False))
    for _ in range(5):
        ref_solver.kick(b1, 10.0)
        halo_solver.kick(b2, 10.0)
    np.testing.assert_array_equal(b1.particles, b2.particles)


def test_basis_conservation():
    """Every basis density has zero total charge and zero dipole moment
    on the grid (Gauss + zero-net-self-force by construction)."""
    from linac_gen.pic.ml.basis import BasisFieldCache
    cache = BasisFieldCache()
    gmin, gmax = np.full(3, -5.0), np.full(3, 5.0)
    rho = cache._build_rho(np.array([24, 24, 24]), gmin, gmax,
                           sigma=np.ones(3))
    ax = np.linspace(-1, 1, 24) * 5.0
    ux, uy, uz = np.meshgrid(ax, ax, ax, indexing="ij")
    for k in range(rho.shape[0]):
        tot = abs(rho[k].sum())
        dip = max(abs((rho[k] * ux).sum()), abs((rho[k] * uy).sum()),
                  abs((rho[k] * uz).sum()))
        assert tot < 1e-10, f"mode {k} monopole {tot}"
        assert dip < 1e-9, f"mode {k} dipole {dip}"
        assert abs(np.sqrt((rho[k] ** 2).sum()) - 1.0) < 1e-12


def test_basis_projection_roundtrip():
    """A defect that IS a basis combination projects back exactly."""
    from linac_gen.pic.ml.basis import BasisFieldCache
    from linac_gen.pic.poisson_solver import PoissonSolverFFT
    n = np.array([16, 16, 16])
    gmin, gmax = np.array([-5.0, -5.0, -5.0]), np.array([5.0, 5.0, 5.0])
    solver = PoissonSolverFFT(gmin, gmax, n)
    cache = BasisFieldCache()
    entry = cache.get(solver, gmin, gmax, n, sigma=np.ones(3))
    rng = np.random.default_rng(0)
    c_true = rng.normal(0, 1, cache.n_basis())
    ex, ey, ez = cache.correction_field(entry, c_true)
    c_hat = cache.project_defect(entry, ex, ey, ez)
    np.testing.assert_allclose(c_hat, c_true, rtol=1e-6, atol=1e-8)


def test_weighted_projection_both_regimes():
    """Weighted LS is exact on in-span defects (any positive weights),
    and on an out-of-span defect it fits the core better than the
    unweighted fit does (that is its purpose)."""
    from linac_gen.pic.ml.basis import BasisFieldCache
    from linac_gen.pic.poisson_solver import PoissonSolverFFT
    n = np.array([16, 16, 16])
    gmin, gmax = np.array([-5.0, -5.0, -5.0]), np.array([5.0, 5.0, 5.0])
    solver = PoissonSolverFFT(gmin, gmax, n)
    cache = BasisFieldCache()
    entry = cache.get(solver, gmin, gmax, n, sigma=np.ones(3))
    rng = np.random.default_rng(1)
    # regime 1: in-span defect, gaussian-core weights -> exact recovery
    c_true = rng.normal(0, 1, cache.n_basis())
    ex, ey, ez = cache.correction_field(entry, c_true)
    ax = np.linspace(-5, 5, 16)
    ux, uy, uz = np.meshgrid(ax, ax, ax, indexing="ij")
    rho = np.exp(-(ux**2 + uy**2 + uz**2) / 2)
    w = cache.cell_weights(rho)
    c_hat = cache.project_defect_weighted(entry, ex, ey, ez, w)
    np.testing.assert_allclose(c_hat, c_true, rtol=1e-5, atol=1e-7)
    # regime 2: out-of-span defect concentrated in the core -> the
    # weighted fit leaves a smaller core-weighted residual
    core = np.exp(-(ux**2 + uy**2 + uz**2) / 0.5)
    dEx, dEy, dEz = core * ux, core * uy, 0.3 * core
    cw = cache.project_defect_weighted(entry, dEx, dEy, dEz, w)
    cu = cache.project_defect(entry, dEx, dEy, dEz)
    def core_res(c):
        px, py, pz = cache.correction_field(entry, c)
        return cache.weighted_norm(w, dEx - px, dEy - py, dEz - pz)
    assert core_res(cw) < core_res(cu)
    # degenerate weights fall back gracefully
    w0 = cache.cell_weights(np.zeros_like(rho))
    assert np.all(w0 == 1.0)


def test_anchor_logs_and_fine_field():
    """Anchors run at the configured cadence, log training pairs, and
    the defect magnitude is sane (fine vs coarse differ but not wildly)."""
    from linac_gen.pic.ml.solver import HaloPicSolver
    s = HaloPicSolver(_sc("halo", k_init=2, k_min=2, k_max=2,
                          fine_factor=2))
    b = _beam(seed=5)
    for _ in range(6):
        s.kick(b, 10.0)
    assert len(s.log["e"]) == 3            # kicks 0, 2, 4
    assert s.log["kick_of_anchor"] == [0, 2, 4]
    assert len(s.log["features"]) == 3
    assert s.log["features"][0].shape[0] > 10
    assert all(0.0 < e < 1.0 for e in s.log["e"])
    assert len(s.log["coeffs"][0]) == s.basis.n_basis()


def test_net_scored_on_anchors_without_collect():
    """Eval mode (collect=False) must still score the net on every
    anchor: finite r_after, e_ctl != e_raw, and the alpha gate able to
    re-open after a bad first anchor.  Regression: feats were only built
    when collecting, so the gate closed at anchor 0 and never reopened."""
    import torch
    from linac_gen.pic.ml.solver import HaloPicSolver
    s = HaloPicSolver(_sc("halo", collect=False, k_init=2, k_min=2,
                          k_max=2, fine_factor=2))
    torch.manual_seed(0)
    from linac_gen.surrogates.base import MlpHead
    from linac_gen.pic.ml.features import FEATURE_DIM
    s.attach_net(MlpHead(FEATURE_DIM, s.basis.n_basis(), (8,)).eval())
    b = _beam(seed=7)
    for _ in range(8):
        s.kick(b, 10.0)
    r_after = np.asarray(s.log["r_after"], float)
    assert len(r_after) == 4
    assert np.isfinite(r_after).all()
    e = np.asarray(s.log["e"]); er = np.asarray(s.log["e_raw"])
    assert not np.allclose(e, er)          # e_ctl is net-vs-anchor, not raw


def test_k_controller_both_branches():
    from linac_gen.pic.ml.solver import KController
    kc = KController(k_init=8, k_min=2, k_max=32,
                     tau_lo=0.02, tau_hi=0.10, tau_hard=0.30)
    kc.update(0.01)
    assert kc.k == 16 and not kc.alpha_killed       # grow
    kc.update(0.15)
    assert kc.k == 8 and not kc.alpha_killed        # shrink
    kc.update(0.5)
    assert kc.alpha_killed and kc.k == 4            # kill switch


def test_config_validation():
    with pytest.raises(ValueError):
        SpaceChargeConfig(sc_backend="bogus")
    cfg = _sc("halo", k_init=4)
    assert cfg.sc_backend == "halo"
    assert cfg.halo["k_init"] == 4


def test_fine_anchor_grid_guard():
    """Default 96^3 grid x fine_factor 4 = 384^3 anchors would OOM on the
    first kick — must fail fast at construction with guidance."""
    from linac_gen.pic.ml.solver import HaloPicSolver
    cfg = SpaceChargeConfig(nx=96, ny=96, nz=96, sc_backend="halo",
                            halo={"anchors": True})
    with pytest.raises(ValueError, match="fine anchor grid"):
        HaloPicSolver(cfg)
    # anchors off -> no fine solver -> no guard needed
    cfg2 = SpaceChargeConfig(nx=96, ny=96, nz=96, sc_backend="halo",
                             halo={"anchors": False})
    HaloPicSolver(cfg2)


def test_basis_cache_bounded_and_degenerate_gram():
    from linac_gen.pic.ml.basis import BasisFieldCache
    from linac_gen.pic.poisson_solver import PoissonSolverFFT
    n = np.array([8, 8, 8])
    gmin, gmax = np.full(3, -5.0), np.full(3, 5.0)
    solver = PoissonSolverFFT(gmin, gmax, n)
    cache = BasisFieldCache()
    cache.max_entries = 3
    # distinct sigma bins force distinct entries; FIFO keeps <= 3
    for s in (0.5, 1.0, 2.0, 4.0, 8.0):
        cache.get(solver, gmin, gmax, n, sigma=np.full(3, s))
    assert len(cache._entries) == 3
    # degenerate: sigma collapsed 12 orders below the box -> densities
    # underflow -> Gram trace 0 -> correction disabled, no LinAlgError
    entry = cache.get(solver, gmin, gmax, n, sigma=np.full(3, 1e-12))
    assert np.all(entry["gram_inv"] == 0.0)
    c = cache.project_defect(entry, *[np.ones(tuple(n))] * 3)
    assert np.all(c == 0.0)


def test_trained_net_roundtrip(tmp_path):
    """save via train_offline -> load into solver -> forward runs and the
    folded standardization reproduces the training-time prediction."""
    import torch
    from linac_gen.pic.ml.train_offline import train_corrector
    from linac_gen.pic.ml.solver import HaloPicSolver
    from linac_gen.pic.ml.features import FEATURE_DIM

    rng = np.random.default_rng(0)
    nb = 31
    X = rng.normal(0, 1, (200, FEATURE_DIM))
    W = rng.normal(0, 0.3, (FEATURE_DIM, nb))
    Y = X @ W + 0.01 * rng.normal(size=(200, nb))
    log = tmp_path / "log.npz"
    np.savez(log, features=X, coeffs=Y, e=np.zeros(200),
             k=np.zeros(200), kick_of_anchor=np.zeros(200))
    r = train_corrector([str(log)], tmp_path / "net", hidden=(32,),
                        epochs=800, verbose=False)
    assert r["val_mse"] < 0.1               # a linear map is learnable
    s = HaloPicSolver(_sc("halo"))
    s.load(tmp_path / "net")
    assert not s.net_scale_normalized       # log had no scale array
    with torch.no_grad():
        out = s.net(torch.from_numpy(X[0])).numpy()
    # folded scalers: prediction approximates the linear target
    assert np.mean((out - Y[0]) ** 2) / np.mean(Y[0] ** 2) < 0.2


def test_trained_net_scale_normalized(tmp_path):
    """Second regime: logs WITH a per-anchor scale array train on
    coeffs/scale and the solver rescales predictions by the local
    field norm."""
    import torch
    from linac_gen.pic.ml.train_offline import train_corrector
    from linac_gen.pic.ml.solver import HaloPicSolver
    from linac_gen.pic.ml.features import FEATURE_DIM

    rng = np.random.default_rng(2)
    nb = 31
    X = rng.normal(0, 1, (200, FEATURE_DIM))
    W = rng.normal(0, 0.3, (FEATURE_DIM, nb))
    S = np.exp(rng.normal(0, 1, 200))       # wildly varying local scale
    Y = (X @ W) * S[:, None]                # raw targets scale with S
    log = tmp_path / "log.npz"
    np.savez(log, features=X, coeffs=Y, scale=S, e=np.zeros(200),
             k=np.zeros(200), kick_of_anchor=np.zeros(200))
    r = train_corrector([str(log)], tmp_path / "net", hidden=(32,),
                        epochs=800, scale_normalize=True, verbose=False)
    assert r["val_mse"] < 0.1
    s = HaloPicSolver(_sc("halo"))
    s.load(tmp_path / "net")
    assert s.net_scale_normalized
    out = s._net_coeffs(X[0], float(S[0]))
    assert np.mean((out - Y[0]) ** 2) / np.mean(Y[0] ** 2) < 0.2
