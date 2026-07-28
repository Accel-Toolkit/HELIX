"""Phase 3: compare_runs + parameter_scan."""
from __future__ import annotations

from tests.dataguard import needs, require  # noqa: E402

import os

import pytest

from linac_gen.assist.tools import TOOLS, WorkContext

BTL = "examples/pipii/btl/btl.lgproj"


@pytest.fixture(scope="module")
def btl_ctx(tmp_path_factory):
    ctx = WorkContext(calc_dir=str(tmp_path_factory.mktemp("agency")))
    TOOLS["load_lattice"].fn(ctx, path=BTL)
    TOOLS["run_envelope"].fn(ctx)
    return ctx


@needs("examples/pipii/btl/btl.lgproj", "examples/pipii/btl/btl.dat")
def test_compare_current_vs_written_is_zero_delta(btl_ctx):
    p = os.path.join(btl_ctx.calc_dir, "same.h5")
    TOOLS["write_results"].fn(btl_ctx, path=p)
    r = TOOLS["compare_runs"].fn(btl_ctx, run_a="current", run_b=p)
    assert r["status"] == "ok"
    kp = r["data"]["exit_kpis"]
    assert kp["sigma_x"]["delta"] == 0.0
    assert kp["ref_w_kin"]["a"] == kp["ref_w_kin"]["b"] == 800.0
    # array-level max|Δ| present and zero on identical grids
    assert kp["max_abs_dsigma_x"]["value_mm"] == 0.0


@needs("examples/pipii/btl/btl.lgproj", "examples/pipii/btl/btl.dat")
def test_compare_refuses_missing(btl_ctx):
    r = TOOLS["compare_runs"].fn(btl_ctx, run_a="current",
                                 run_b="/nope/missing.h5")
    assert r["status"] == "error"
    ctx2 = WorkContext(calc_dir=".")
    r2 = TOOLS["compare_runs"].fn(ctx2, run_a="current", run_b="x.h5")
    assert r2["status"] == "refused"          # no in-session results


@needs("examples/pipii/btl/btl.lgproj", "examples/pipii/btl/btl.dat")
def test_scan_validates_element_and_attribute(btl_ctx):
    r = TOOLS["parameter_scan"].fn(btl_ctx, element="NOSUCH",
                                   attribute="gradient", start=1, stop=2)
    assert r["status"] == "refused"
    quad = next(e.name for e in btl_ctx.lattice.elements
                if type(e).__name__ == "Quadrupole")
    r2 = TOOLS["parameter_scan"].fn(btl_ctx, element=quad,
                                    attribute="frobnicate", start=1, stop=2)
    assert r2["status"] == "refused" and "frobnicate" in \
        r2["data"]["message"]


@needs("examples/pipii/btl/btl.lgproj", "examples/pipii/btl/btl.dat")
def test_scan_two_points_and_abort(btl_ctx):
    quads = [(i, e) for i, e in enumerate(btl_ctx.lattice.elements)
             if type(e).__name__ == "Quadrupole"]
    qi, qe = quads[0]
    g0 = qe.gradient
    r = TOOLS["parameter_scan"].fn(
        btl_ctx, element=qe.name, attribute="gradient",
        start=g0, stop=g0 * 1.02, n_points=2,
        should_abort=lambda: False)
    assert r["status"] == "ok" and not r["data"]["aborted"]
    pts = r["data"]["points"]
    assert len(pts) == 2 and all("sigma_x" in p for p in pts)
    assert pts[0]["value"] == pytest.approx(g0)
    # immediate abort -> no points, aborted flag set
    r2 = TOOLS["parameter_scan"].fn(
        btl_ctx, element=qe.name, attribute="gradient",
        start=g0, stop=g0 * 1.02, n_points=5,
        should_abort=lambda: True)
    assert r2["status"] == "ok" and r2["data"]["aborted"]
    assert r2["data"]["points"] == []


def test_scan_is_long_running():
    from linac_gen.assist.agent import LONG_RUNNING
    assert "parameter_scan" in LONG_RUNNING


@needs("examples/pipii/btl/btl.lgproj", "examples/pipii/btl/btl.dat")
def test_scan_restores_fft_workers_env(btl_ctx, monkeypatch):
    """Adversarial H1: the in-process scan must not permanently set
    LINAC_GEN_FFT_WORKERS=1 in the host process."""
    import os
    monkeypatch.delenv("LINAC_GEN_FFT_WORKERS", raising=False)
    quad = next(e for e in btl_ctx.lattice.elements
                if type(e).__name__ == "Quadrupole")
    g0 = quad.gradient
    r = TOOLS["parameter_scan"].fn(
        btl_ctx, element=quad.name, attribute="gradient",
        start=g0, stop=g0 * 1.01, n_points=2,
        should_abort=lambda: False)
    assert r["status"] == "ok"
    assert "LINAC_GEN_FFT_WORKERS" not in os.environ    # restored


@needs("examples/pipii/btl/btl.lgproj", "examples/pipii/btl/btl.dat")
def test_scan_refuses_fractional_sweep_of_int_attribute(btl_ctx):
    """Adversarial L2: int-typed attributes must not silently truncate."""
    quad = next(e for e in btl_ctx.lattice.elements
                if type(e).__name__ == "Quadrupole")
    quad.fake_int_knob = 3                    # inject an int-typed attr
    try:
        r = TOOLS["parameter_scan"].fn(
            btl_ctx, element=quad.name, attribute="fake_int_knob",
            start=3, stop=4, n_points=3)      # midpoint 3.5 -> truncation
        assert r["status"] == "refused"
        assert "integer" in r["data"]["message"]
    finally:
        del quad.fake_int_knob


def test_report_numbering_never_overwrites(tmp_path):
    """Adversarial M2: deleting an old report must not cause reuse."""
    import numpy as np
    from linac_gen.assist.tools import WorkContext

    class _Ctx(WorkContext):
        def __init__(self, d):
            super().__init__(calc_dir=str(d))
            class _R: sigma_x = np.array([1.0, 0.9])
            self.results = _R()

    ctx = _Ctx(tmp_path)
    p0 = TOOLS["generate_report"].fn(ctx)["data"]["report"]
    p1 = TOOLS["generate_report"].fn(ctx)["data"]["report"]
    p2 = TOOLS["generate_report"].fn(ctx)["data"]["report"]
    import os
    os.remove(p0)                              # delete the oldest
    p3 = TOOLS["generate_report"].fn(ctx)["data"]["report"]
    assert p3 not in (p1, p2)                  # no silent overwrite
    assert os.path.exists(p1) and os.path.exists(p2)


def test_report_on_file_embeds_no_live_figures(tmp_path):
    """Adversarial M1: reporting on a FILE must not embed the live
    session's figures (they may be a different run)."""
    import numpy as np
    from linac_gen.assist.tools import WorkContext
    ctx = WorkContext(calc_dir=str(tmp_path))
    class _R: sigma_x = np.array([1.0, 0.9])
    ctx.results = _R()
    # write the current results out, then report on the FILE while the
    # GUI hook would happily provide figures
    ctx.grab_plot = lambda name: {"img_b64": "QUJDRA==", "w": 2, "h": 2,
                                  "label": name, "mime": "image/png"}
    p = str(tmp_path / "r.h5")
    TOOLS["write_results"].fn(ctx, path=p)
    r = TOOLS["generate_report"].fn(ctx, path=p)
    assert r["status"] == "ok"
    assert r["data"]["figures"] == 0           # no cross-run figures
