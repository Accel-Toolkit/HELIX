"""Re-shaping a beam read from a particle file onto given transverse Twiss
(``BeamConfig.reshape_to_twiss``, ``distributions.factory.reshape_to_twiss``).

Each transverse plane is mapped linearly onto (α, β), keeping the file's rms
emittance, centroid and normalised phase-space shape; off by default, where
a file is used exactly as it is.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.distributions.factory import create_beam, reshape_to_twiss
from linac_gen.io.tracewin_dst import load_dst, write_dst

MC2_P = 938.272088


def _rms(u, up):
    c = np.cov(u, up)
    eps = float(np.sqrt(np.linalg.det(c)))
    return -c[0, 1] / eps, c[0, 0] / eps, eps


def _particles(n=4000, seed=7):
    """A non-Gaussian, x–y-correlated beam with offset centroids."""
    rng = np.random.default_rng(seed)
    p = np.zeros((n, 6))
    r = rng.standard_t(5, size=(n, 4))
    p[:, 0] = 1.2 * r[:, 0] + 0.15
    p[:, 1] = 8.0 * r[:, 1] + 3.0 * r[:, 0] - 0.4
    p[:, 2] = 1.0 * r[:, 2] + 0.3 * r[:, 0]
    p[:, 3] = 9.0 * r[:, 3] - 4.0 * r[:, 2] + 0.2
    p[:, 4] = rng.uniform(-180, 180, n)
    p[:, 5] = rng.normal(0, 1e-3, n)
    return p


def test_reshape_hits_target_keeps_emittance_and_centroid():
    p = _particles()
    before = p.copy()
    q = reshape_to_twiss(p, 1.62, 0.069, 1.75, 0.074)
    assert np.array_equal(p, before)                     # input untouched
    assert np.array_equal(q[:, 4:], p[:, 4:])             # z untouched
    for (i, j), (a_t, b_t) in (((0, 1), (1.62, 0.069)), ((2, 3), (1.75, 0.074))):
        a, b, eps = _rms(q[:, i], q[:, j])
        assert a == pytest.approx(a_t, abs=1e-10)
        assert b == pytest.approx(b_t, rel=1e-10)
        assert eps == pytest.approx(_rms(p[:, i], p[:, j])[2], rel=1e-10)
        assert q[:, i].mean() == pytest.approx(p[:, i].mean(), abs=1e-12)
        assert q[:, j].mean() == pytest.approx(p[:, j].mean(), abs=1e-12)
    # the normalised shape is kept: amplitudes in each plane's own rms
    # ellipse are unchanged particle by particle
    for (i, j) in ((0, 1), (2, 3)):
        def amp(arr):
            u = arr[:, i] - arr[:, i].mean()
            v = arr[:, j] - arr[:, j].mean()
            a, b, eps = _rms(arr[:, i], arr[:, j])
            g = (1 + a * a) / b
            return (g * u * u + 2 * a * u * v + b * v * v) / eps
        assert np.allclose(amp(q), amp(p), rtol=1e-9, atol=1e-12)


def test_reshape_refuses_a_degenerate_plane():
    p = _particles()
    p[:, 1] = 0.5 * p[:, 0]                  # zero emittance in x
    with pytest.raises(ValueError, match="re-shape plane x"):
        reshape_to_twiss(p, 1.0, 0.1, 1.0, 0.1)


@pytest.fixture
def dst_file(tmp_path):
    path = tmp_path / "beam.dst"
    write_dst(str(path), _particles(), current_mA=5.0, frequency_MHz=352.21,
              mass_MeV=MC2_P, w_kin_ref=0.075)
    return path


def _file_cfg(path, **kw):
    base = dict(species="proton", energy=0.075, frequency=352.21,
                current=5.0, source="file", distribution_file=str(path),
                alpha_x=1.62, beta_x=0.069, alpha_y=1.75, beta_y=0.074)
    base.update(kw)
    return BeamConfig(**base)


def test_dst_reshaped_onto_config_twiss(dst_file):
    loaded, header = load_dst(str(dst_file))
    cfg = _file_cfg(dst_file, reshape_to_twiss=True)
    beam = create_beam(cfg, seed=1)
    p = beam.particles
    assert _rms(p[:, 0], p[:, 1])[:2] == pytest.approx((1.62, 0.069), rel=1e-9)
    assert _rms(p[:, 2], p[:, 3])[:2] == pytest.approx((1.75, 0.074), rel=1e-9)
    assert _rms(p[:, 0], p[:, 1])[2] == pytest.approx(
        _rms(loaded[:, 0], loaded[:, 1])[2], rel=1e-9)
    # the targets are kept on the config; the emittances are the file's
    assert (cfg.alpha_x, cfg.beta_x, cfg.alpha_y, cfg.beta_y) == (
        1.62, 0.069, 1.75, 0.074)
    assert cfg.emit_nx == pytest.approx(header["emit_nx"])


def test_dst_used_as_is_by_default(dst_file):
    loaded, header = load_dst(str(dst_file))
    cfg = _file_cfg(dst_file)
    assert cfg.reshape_to_twiss is False
    beam = create_beam(cfg, seed=1)
    assert np.array_equal(beam.particles, loaded)
    # as before: the header's Twiss overwrite the config's
    assert cfg.alpha_x == pytest.approx(header["alpha_x"])
    assert cfg.beta_y == pytest.approx(header["beta_y"])


def test_ascii_file_reshaped(tmp_path):
    path = tmp_path / "beam.txt"
    p = _particles()
    np.savetxt(path, p, header="w_kin_ref: 0.075")
    cfg = _file_cfg(path, reshape_to_twiss=True)
    q = create_beam(cfg, seed=1).particles
    assert _rms(q[:, 0], q[:, 1])[:2] == pytest.approx((1.62, 0.069), rel=1e-9)
    plain = create_beam(_file_cfg(path), seed=1).particles
    assert not np.allclose(plain[:, 0], q[:, 0])


def test_backtracking_reads_its_exit_file_as_is(dst_file, monkeypatch):
    """The backtrack paths copy the user's beam into a file source for the
    lattice exit: the user's re-shape option must not apply to it."""
    import linac_gen.distributions.factory as factory
    from linac_gen.cli.common import run_backtrack_sim
    from linac_gen.core.lattice import Lattice
    from linac_gen.core.step_config import StepConfig
    from linac_gen.elements.drift import Drift

    seen = {}

    class _Stop(Exception):
        pass

    def spy(cfg, seed=None):
        seen["reshape"] = cfg.reshape_to_twiss
        raise _Stop

    monkeypatch.setattr(factory, "create_beam", spy)
    lat = Lattice()
    lat.add(Drift(name="D", length=100.0, aperture=20.0))
    cfg = _file_cfg(dst_file, reshape_to_twiss=True)
    with pytest.raises(_Stop):
        run_backtrack_sim(lat, cfg, None, StepConfig(), dst_path=dst_file)
    assert seen["reshape"] is False
    assert cfg.reshape_to_twiss is True             # the user's config kept


def test_project_round_trip(tmp_path, dst_file):
    from linac_gen.io.project import _beam_from_dict, load_project, write_project
    lat = tmp_path / "deck.dat"
    lat.write_text("DRIFT 100 20\nEND\n")
    proj = write_project(tmp_path / "p.lgproj", lattice_path=lat,
                         beam=_file_cfg(dst_file, reshape_to_twiss=True))
    assert load_project(proj).beam.reshape_to_twiss is True
    legacy = {k: v for k, v in vars(_file_cfg(dst_file)).items()
              if k != "reshape_to_twiss"}
    assert _beam_from_dict(legacy).reshape_to_twiss is False
    assert replace(_file_cfg(dst_file)).reshape_to_twiss is False
