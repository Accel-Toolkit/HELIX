"""The GUI's envelope run and the Results tab's envelope companion start from the beam a
multiparticle run generates, input dispersion included (both left it out, 2026-11-07)."""
import os

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from linac_gen.core.config import BeamConfig  # noqa: E402
from linac_gen.tracking.envelope import EnvelopeSolver, _build_sigma_matrix  # noqa: E402

FODO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "examples",
                                    "fodo_cell.dat"))
DISP = {"disp_x": 120.0, "disp_xp": -35.0, "disp_y": -60.0, "disp_yp": 15.0}
TWISS = dict(species="proton", energy=3.0, frequency=162.5, emit_nx=0.25, emit_ny=0.3,
             emit_z=0.5, alpha_x=-0.8, beta_x=1.7, alpha_y=0.6, beta_y=2.3, alpha_z=1.3,
             beta_z=0.9, current=0.0)


def _sheared(cfg, bg):
    from linac_gen.distributions.factory import geometric_emittances
    ex, ey, ez = geometric_emittances(cfg, bg)
    sb = _build_sigma_matrix(cfg.alpha_x, cfg.beta_x, ex, cfg.alpha_y, cfg.beta_y, ey,
                             cfg.alpha_z, cfg.beta_z, ez)
    s = np.eye(6)
    s[0:4, 5] = [cfg.disp_x, cfg.disp_xp, cfg.disp_y, cfg.disp_yp]
    return s @ sb @ s.T


def _close(got, want):
    want = np.asarray(want, float)
    np.testing.assert_allclose(np.asarray(got, float), want, rtol=1e-12,
                               atol=1e-12 * float(np.max(np.abs(want))))


def test_the_gui_envelope_run_starts_from_the_sheared_beam(qapp, win):
    win.open_lattice(FODO)
    cfg = win.state.beam_config
    for k, v in {**TWISS, **DISP}.items():
        setattr(cfg, k, v)
    cfg.source = "generate"
    win.state.set_beam_config(cfg)
    win._run_envelope()
    win.wait_worker(win._envelope_worker)
    res = win.state.results
    assert float(res.s[0]) == 0.0
    from linac_gen.cli.common import build_ref
    _close(res.sigma_matrix[0], _sheared(cfg, build_ref(cfg).bg))


def test_the_results_tabs_envelope_companion_starts_from_the_sheared_beam(qapp):
    from linac_gen_gui.interphase.tabs.results_tab import _beam_inputs_from_config, _probe_sig

    from linac_gen.io.tracewin_parser import parse_tracewin
    cfg = BeamConfig(**TWISS, **DISP)
    ref, initial = _beam_inputs_from_config(cfg)
    lat, _ = parse_tracewin(FODO)
    res = EnvelopeSolver(lat, ref, initial, current=0.0).run()
    _close(res.sigma_matrix[0], _sheared(cfg, ref.bg))
    # the cached companion run is redone when the dispersion or the source changes
    other = BeamConfig(**TWISS, **{**DISP, "disp_xp": -34.0})
    assert _probe_sig(lat, other) != _probe_sig(lat, cfg)
    filed = BeamConfig(**TWISS, **DISP, source="file")
    assert _probe_sig(lat, filed) != _probe_sig(lat, cfg)


def _dst_config(tmp_path):
    from linac_gen.core.particle import PROTON
    from linac_gen.io.tracewin_dst import write_dst
    p = np.random.default_rng(5).normal(size=(2000, 6)) * [1.0, 1.5, 0.8, 1.2, 3.0, 0.004]
    path = tmp_path / "beam.dst"
    write_dst(str(path), p, current_mA=0.0, frequency_MHz=162.5, mass_MeV=PROTON.mass,
              w_kin_ref=3.0)
    return path


def test_a_beam_read_from_a_file_takes_no_dispersion_in_the_gui(qapp, win, tmp_path):
    from linac_gen_gui.interphase.tabs.results_tab import _beam_inputs_from_config
    path = _dst_config(tmp_path)
    win.open_lattice(FODO)
    cfg = win.state.beam_config
    for k, v in {**TWISS, **DISP}.items():
        setattr(cfg, k, v)
    cfg.source, cfg.distribution_file = "file", str(path)
    win.state.set_beam_config(cfg)
    win._run_envelope()
    win.wait_worker(win._envelope_worker)
    s0 = np.asarray(win.state.results.sigma_matrix[0])
    assert np.all(s0[0:4, 4:6] == 0.0), s0
    _ref, initial = _beam_inputs_from_config(BeamConfig(**TWISS, **DISP, source="file",
                                                        distribution_file=str(path)))
    assert all(initial[k] == 0.0 for k in DISP), initial


def test_the_surrogates_compare_starts_from_the_project_beam(qapp, tmp_path):
    """Its baseline-against-surrogate pair ran from a seed built by hand, without the
    dispersion (or a continuous beam's flag): now the one every envelope run takes."""
    from linac_gen_gui.interphase.tabs.surrogates_tab import _compare_seed

    from linac_gen.cli.common import _envelope_initial, build_ref
    cfg = BeamConfig(**TWISS, **DISP)
    ref = build_ref(cfg)
    seed = _compare_seed(cfg, ref)
    assert seed == _envelope_initial(cfg, ref)
    assert {k: seed[k] for k in DISP} == DISP
