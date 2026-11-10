"""An envelope run of a bunched beam starts from the beam a multiparticle run generates, input
dispersion included.  The generator shears each particle, q = q_betatron + D_q·ΔW, so the
starting sigma is S·Σβ·Sᵀ with S the shear.  The CLI and everything built on it, the matching
engine, the gradient matcher and grad_sensitivities left the dispersion out (on the three
tracked projects that set one, the exit σ_x of the envelope was up to 8.4 % off a multiparticle
run of the same beam, and 0.34 % with it, 2026-11-08), and the sigma builder left out the phase
correlation of a sheared bunch whose α_z is not zero.  A continuous beam's envelope holds no
energy spread, so the dispersion cannot enter it: it says so."""
from types import SimpleNamespace

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.edge import Edge
from linac_gen.elements.lattice_commands import SetSize
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.tracking.envelope import EnvelopeSolver, _build_sigma_matrix

DISP = {"disp_x": 120.0, "disp_xp": -35.0, "disp_y": -60.0, "disp_yp": 15.0}


def _cfg(**kw):
    base = dict(species="proton", energy=3.0, frequency=162.5, n_particles=1000,
                distribution="gaussian", emit_nx=0.25, emit_ny=0.3, emit_z=0.5,
                alpha_x=-0.8, beta_x=1.7, alpha_y=0.6, beta_y=2.3, alpha_z=1.3, beta_z=0.9,
                current=0.0, **DISP)
    base.update(kw)
    return BeamConfig(**base)


def _bg(cfg):
    from linac_gen.cli.common import build_ref
    return build_ref(cfg).bg


def _sheared(cfg):
    """S·Σβ·Sᵀ: the config's betatron beam sheared by its dispersion."""
    from linac_gen.distributions.factory import geometric_emittances
    ex, ey, ez = geometric_emittances(cfg, _bg(cfg))
    sb = _build_sigma_matrix(cfg.alpha_x, cfg.beta_x, ex, cfg.alpha_y, cfg.beta_y, ey,
                             cfg.alpha_z, cfg.beta_z, ez)
    s = np.eye(6)
    s[0:4, 5] = [cfg.disp_x, cfg.disp_xp, cfg.disp_y, cfg.disp_yp]
    return s @ sb @ s.T


def _close(got, want, rtol=1e-12):
    got, want = np.asarray(got, float), np.asarray(want, float)
    np.testing.assert_allclose(got, want, rtol=rtol, atol=rtol * float(np.max(np.abs(want))))


def _bend_line():
    lat = Lattice()
    for e in (Drift("D1", length=200.0), Quadrupole("QF", length=100.0, gradient=6.0),
              Drift("D2", length=150.0), Edge("E1", pole_rotation=4.0, rho=1000.0, gap=30.0),
              Dipole("B", angle=12.0, rho=1000.0, e1=4.0, e2=4.0),
              Edge("E2", pole_rotation=4.0, rho=1000.0, gap=30.0), Drift("D3", length=250.0),
              Quadrupole("QD", length=100.0, gradient=-6.0), Drift("D4", length=300.0)):
        lat.add(e)
    return lat


def _exit_sizes_from(lat, cfg, sigma0):
    """σ_x, σ_y, σ_φ at the exit of the numpy envelope at zero current started at sigma0."""
    from linac_gen.cli.common import _envelope_initial, build_ref
    ref = build_ref(cfg)
    res = EnvelopeSolver(lat, ref, _envelope_initial(cfg, ref), current=0.0,
                         initial_sigma=np.array(sigma0)).run()
    s = np.asarray(res.sigma_matrix[-1])
    return np.sqrt([s[0, 0], s[2, 2], s[4, 4]])


@pytest.mark.parametrize("alpha_z", [0.0, 1.3], ids=["upright bunch", "tilted bunch"])
def test_the_starting_sigma_is_the_betatron_beam_sheared_by_the_dispersion(alpha_z):
    cfg = _cfg(alpha_z=alpha_z)
    from linac_gen.distributions.factory import geometric_emittances
    ex, ey, ez = geometric_emittances(cfg, _bg(cfg))
    got = _build_sigma_matrix(cfg.alpha_x, cfg.beta_x, ex, cfg.alpha_y, cfg.beta_y, ey,
                              cfg.alpha_z, cfg.beta_z, ez, **DISP)
    _close(got, _sheared(cfg), rtol=1e-14)
    assert np.array_equal(got, got.T)


def test_it_is_the_covariance_of_the_beam_the_generator_makes():
    """A fixed-seed 200k-particle beam: every entry within six standard errors (before the
    fix the eight φ–transverse entries were up to 354 standard errors off)."""
    from linac_gen.distributions.factory import create_beam, geometric_emittances
    cfg = _cfg(n_particles=200_000)
    beam = create_beam(cfg, seed=11)
    cov = np.cov(beam.particles.T)
    ex, ey, ez = geometric_emittances(cfg, beam.ref.bg)
    want = _build_sigma_matrix(cfg.alpha_x, cfg.beta_x, ex, cfg.alpha_y, cfg.beta_y, ey,
                               cfg.alpha_z, cfg.beta_z, ez, **DISP)
    d = np.diag(want)
    stderr = np.sqrt((np.outer(d, d) + want ** 2) / len(beam.particles))
    assert np.max(np.abs(cov - want) / stderr) < 6.0


def _old_builder(alpha_x, beta_x, emit_x, alpha_y, beta_y, emit_y, alpha_z, beta_z, emit_z,
                 disp_x=0.0, disp_xp=0.0, disp_y=0.0, disp_yp=0.0):
    """The builder before the φ–transverse term, line for line."""
    sigma = np.zeros((6, 6))
    for (i, j), (a, b, e) in (((0, 1), (alpha_x, beta_x, emit_x)),
                              ((2, 3), (alpha_y, beta_y, emit_y)),
                              ((4, 5), (alpha_z, beta_z, emit_z))):
        g = (1.0 + a * a) / b
        sigma[i, i] = b * e
        sigma[i, j] = -a * e
        sigma[j, i] = -a * e
        sigma[j, j] = g * e
    if disp_x or disp_xp or disp_y or disp_yp:
        s_ww = sigma[5, 5]
        eta = {0: disp_x, 1: disp_xp, 2: disp_y, 3: disp_yp}
        for i, di in eta.items():
            if di == 0.0:
                continue
            sigma[i, 5] += di * s_ww
            sigma[5, i] += di * s_ww
            for j, dj in eta.items():
                sigma[i, j] += di * dj * s_ww
    return sigma


@pytest.mark.parametrize("disp", [{}, DISP, {"disp_x": -2.5}],
                         ids=["no dispersion", "all four", "one negative"])
def test_an_upright_bunch_or_no_dispersion_gives_the_same_bits_as_before(disp):
    twiss = (-0.8, 1.7, 0.31, 0.6, 2.3, 0.27, 0.0, 0.9, 0.5)
    got = _build_sigma_matrix(*twiss, **disp)
    want = _old_builder(*twiss, **disp)
    assert got.tobytes() == want.tobytes()


def test_a_beam_read_from_a_file_takes_no_dispersion_from_the_config(tmp_path):
    """The generator shears a generated beam only; a file's particles are taken as they are,
    so the envelope seed takes none of the config's values either."""
    from linac_gen.cli.common import _envelope_initial, build_ref
    from linac_gen.core.particle import PROTON
    from linac_gen.distributions.factory import input_dispersion
    from linac_gen.io.tracewin_dst import write_dst
    p = np.random.default_rng(5).normal(size=(2000, 6)) * [1.0, 1.5, 0.8, 1.2, 3.0, 0.004]
    path = tmp_path / "beam.dst"
    write_dst(str(path), p, current_mA=0.0, frequency_MHz=162.5, mass_MeV=PROTON.mass,
              w_kin_ref=3.0)
    cfg = _cfg(source="file", distribution_file=str(path))
    assert input_dispersion(cfg) == dict.fromkeys(DISP, 0.0)
    initial = _envelope_initial(cfg, build_ref(cfg))
    assert all(initial[k] == 0.0 for k in DISP)
    assert input_dispersion(_cfg()) == DISP


def test_the_cli_envelope_starts_from_the_sheared_beam():
    from linac_gen.cli.common import run_envelope_sim
    cfg = _cfg()
    res = run_envelope_sim(_bend_line(), cfg)
    assert float(res.s[0]) == 0.0
    _close(res.sigma_matrix[0], _sheared(cfg))


def test_the_matching_engines_envelope_starts_from_the_sheared_beam():
    from linac_gen.matching.engine import _run_envelope
    cfg = _cfg()
    res = _run_envelope(_bend_line(), cfg, space_charge=False)
    assert float(res.s[0]) == 0.0
    _close(res.sigma_matrix[0], _sheared(cfg))


def test_the_gradient_matcher_sees_the_sizes_of_the_sheared_beam():
    """No knobs and a SET_SIZE of (almost) zero: the torch residual is the exit σ_x, σ_y,
    σ_φ, which must be the numpy envelope's from the sheared beam — the input dispersion
    dominates σ_x and σ_y, and the bend's R51 and R52 carry the phase correlation into σ_φ."""
    torch = pytest.importorskip("torch")
    from linac_gen.distributions.factory import create_beam
    from linac_gen.matching.torch_objective import build_torch_residual
    cfg, lat = _cfg(), _bend_line()
    ref = create_beam(cfg, seed=42).ref
    tiny = 1e-300
    con = SimpleNamespace(source=SetSize("S", x_mm=tiny, y_mm=tiny, phi_or_z=tiny),
                          weight=1.0, label="SET_SIZE")
    res = build_torch_residual(lat, cfg, ref, [], [con], [], 0)
    got = res(torch.zeros(0, dtype=torch.float64)).detach().numpy()
    _close(got, _exit_sizes_from(lat, cfg, _sheared(cfg)), rtol=1e-9)


def test_the_gradient_matcher_with_space_charge_samples_the_sheared_beam(monkeypatch):
    pytest.importorskip("torch")
    import linac_gen.matching.torch_objective as tobj
    from linac_gen.distributions.factory import create_beam

    class _Seen(Exception):
        pass

    seen = {}

    def _spy(sigma_in, n, seed):
        seen["sigma"] = np.array(sigma_in)
        raise _Seen

    monkeypatch.setattr(tobj, "_sample_bunch", _spy)
    cfg = _cfg(current=5.0)
    with pytest.raises(_Seen):
        tobj.build_torch_residual_sc(_bend_line(), cfg, create_beam(cfg, seed=42).ref, [], [],
                                     [], 0, sc_cfg=None)
    _close(seen["sigma"], _sheared(cfg))


def test_grad_sensitivities_reports_the_sizes_of_the_sheared_beam():
    pytest.importorskip("torch")
    from linac_gen.assist.tools import TOOLS, WorkContext
    cfg, lat = _cfg(), _bend_line()
    want = dict(zip(("sigma_x", "sigma_y", "sigma_phi"),
                    _exit_sizes_from(lat, cfg, _sheared(cfg)), strict=True))
    for kpi, size in want.items():
        ctx = WorkContext()
        ctx.lattice, ctx.beam_config = lat, cfg
        got = TOOLS["grad_sensitivities"].fn(ctx, kpi=kpi)
        assert got["status"] == "ok", got
        assert got["data"]["kpi_value"] == pytest.approx(size, rel=5e-7)


def _dst_cfg(tmp_path, **kw):
    """A beam read from a .dst file, with a config that still holds a dispersion."""
    from linac_gen.core.particle import PROTON
    from linac_gen.io.tracewin_dst import write_dst
    p = np.random.default_rng(5).normal(size=(2000, 6)) * [1.0, 1.5, 0.8, 1.2, 3.0, 0.004]
    path = tmp_path / "beam.dst"
    write_dst(str(path), p, current_mA=0.0, frequency_MHz=162.5, mass_MeV=PROTON.mass,
              w_kin_ref=3.0)
    return _cfg(source="file", distribution_file=str(path), **kw)


class _Seen(Exception):
    pass


def _start_cli(cfg, monkeypatch):
    from linac_gen.cli.common import run_envelope_sim
    return run_envelope_sim(_bend_line(), cfg).sigma_matrix[0]


def _start_engine(cfg, monkeypatch):
    from linac_gen.matching.engine import _run_envelope
    return _run_envelope(_bend_line(), cfg, space_charge=False).sigma_matrix[0]


def _start_torch(cfg, monkeypatch):
    pytest.importorskip("torch")
    import linac_gen.matching.torch_objective as tobj
    from linac_gen.distributions.factory import create_beam
    seen, build = {}, tobj._build_sigma_matrix

    def spy(*a, **k):
        seen["sigma"] = build(*a, **k)
        return seen["sigma"]
    monkeypatch.setattr(tobj, "_build_sigma_matrix", spy)
    tobj.build_torch_residual(_bend_line(), cfg, create_beam(cfg, seed=42).ref, [], [], [], 0)
    return seen["sigma"]


def _start_torch_sc(cfg, monkeypatch):
    pytest.importorskip("torch")
    import linac_gen.matching.torch_objective as tobj
    from linac_gen.distributions.factory import create_beam
    seen = {}

    def spy(sigma_in, n, seed):
        seen["sigma"] = np.array(sigma_in)
        raise _Seen
    monkeypatch.setattr(tobj, "_sample_bunch", spy)
    with pytest.raises(_Seen):
        tobj.build_torch_residual_sc(_bend_line(), cfg, create_beam(cfg, seed=42).ref, [], [],
                                     [], 0, sc_cfg=None)
    return seen["sigma"]


def _start_grad(cfg, monkeypatch):
    pytest.importorskip("torch")
    import linac_gen.tracking.envelope as envelope
    from linac_gen.assist.tools import TOOLS, WorkContext
    seen, build = {}, envelope._build_sigma_matrix

    def spy(*a, **k):
        seen["sigma"] = build(*a, **k)
        return seen["sigma"]
    monkeypatch.setattr(envelope, "_build_sigma_matrix", spy)
    ctx = WorkContext()
    ctx.lattice, ctx.beam_config = _bend_line(), cfg
    assert TOOLS["grad_sensitivities"].fn(ctx, kpi="sigma_x")["status"] == "ok"
    return seen["sigma"]


@pytest.mark.parametrize("start", [_start_cli, _start_engine, _start_torch, _start_torch_sc,
                                   _start_grad],
                         ids=["cli", "matching engine", "gradient matcher",
                              "gradient matcher with space charge", "grad_sensitivities"])
def test_a_beam_read_from_a_file_takes_no_dispersion_at_any_entry_point(start, tmp_path,
                                                                        monkeypatch):
    """create_beam takes a file's particles as they are, so no entry point may add the
    config's dispersion to the file's beam."""
    sigma = np.asarray(start(_dst_cfg(tmp_path), monkeypatch))
    assert np.all(sigma[0:4, 4:6] == 0.0) and np.all(sigma[4:6, 0:4] == 0.0), sigma


def _straight_line():
    lat = Lattice()
    for e in (Drift("D1", length=200.0), Quadrupole("QF", length=100.0, gradient=6.0),
              Drift("D2", length=300.0)):
        lat.add(e)
    return lat


def test_a_continuous_beam_says_the_dispersion_does_not_enter_its_envelope():
    """The other branch: a continuous beam's envelope holds no energy spread before the first RF
    element, so it starts from the betatron beam, as before, and the seed says why (the
    multiparticle run shears the beam by its DC energy spread)."""
    from linac_gen.assist.tools import TOOLS, WorkContext
    from linac_gen.cli.common import _envelope_initial, build_ref
    from linac_gen.distributions.factory import geometric_emittances
    cfg = _cfg(continuous=True, dc_energy_spread_keV=2.0)
    ref = build_ref(cfg)
    with pytest.warns(UserWarning, match="continuous beam.*dispersion"):
        initial = _envelope_initial(cfg, ref)
    assert all(initial[k] == 0.0 for k in DISP)
    res = EnvelopeSolver(_bend_line(), ref, initial, current=0.0).run()
    ex, ey, _ = geometric_emittances(cfg, ref.bg)
    want = _build_sigma_matrix(cfg.alpha_x, cfg.beta_x, ex, cfg.alpha_y, cfg.beta_y, ey,
                               0.0, 1.0, 0.0)
    assert np.array_equal(np.asarray(res.sigma_matrix[0]), want)
    pytest.importorskip("torch")
    ctx = WorkContext()
    ctx.lattice, ctx.beam_config = _straight_line(), cfg
    got = TOOLS["grad_sensitivities"].fn(ctx, kpi="sigma_x")
    assert got["status"] == "ok" and any("dispersion" in w for w in got["warnings"]), got


def test_a_solve_that_seeds_a_continuous_beam_with_a_dispersion_itself_says_nothing():
    """The solver does not warn: internal solves seed what they solve for (the RFQ space-charge
    match perturbs its state by 1.49e-8 in disp_x, and a warning there fired 84 to 808 times per
    match about a dispersion nobody set)."""
    import warnings

    from linac_gen.cli.common import build_ref
    cfg = _cfg(continuous=True)
    initial = dict(alpha_x=0.0, beta_x=1.0, emit_x=0.2, alpha_y=0.0, beta_y=1.0, emit_y=0.2,
                   emit_z=0.0, alpha_z=0.0, beta_z=1.0, continuous=True, disp_x=1.4901161e-8)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        EnvelopeSolver(_straight_line(), build_ref(cfg), initial, current=0.0).run()


def test_the_sacherer_solver_says_it_does_not_model_the_dispersion():
    from linac_gen.cli.common import _envelope_initial, build_ref
    from linac_gen.tracking.sacherer import SachererSolver
    cfg = _cfg()
    ref = build_ref(cfg)
    with pytest.warns(UserWarning, match="Sacherer.*dispersion"):
        SachererSolver(_straight_line(), ref, _envelope_initial(cfg, ref), current=0.0)
