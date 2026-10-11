"""Scan/batch/study points carry the full space-charge configuration, and the
worker rebuilds exactly the configuration the point was resolved to."""
from __future__ import annotations

import json

import pytest

from linac_gen.cli import common
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.parallel.scan_pool import ScanPoint, worker_sc_config
from linac_gen.pic import methods


def _project(tmp_path, conv):
    (tmp_path / "d.dat").write_text("DRIFT 100 20 0\nEND\n")
    p = tmp_path / "p.lgproj"
    p.write_text(json.dumps({"__kind__": "linac_gen_project", "__version__": 1,
                             "lattice_path": "d.dat", "beam": {"current": 5.0},
                             "convergence": conv}))
    return str(p)


@pytest.mark.parametrize("how", ["sc_overrides", "cli"])
@pytest.mark.parametrize("name", methods.names())
def test_worker_rebuilds_every_method_exactly(name, how, tmp_path):
    # `scan --sc-method NAME` passes it as an override; direct callers may
    # also pass it with the grid size in ``cli``
    kw = ({"sc_overrides": {"method": name}} if how == "sc_overrides"
          else {"cli": {"sc_method": name}})
    point = common.build_scan_point(_project(tmp_path, {}), **kw)
    got = worker_sc_config(point)
    want = methods.resolve_sc({}, {"sc_method": name})
    assert got == want and got.method == want.method
    if name != "custom":
        assert got.method == name


def test_sc_override_of_grid_size_no_longer_crashes(tmp_path):
    """--sc nx=... used to raise a duplicate-keyword TypeError in the
    worker; the override now wins."""
    point = common.build_scan_point(_project(tmp_path, {}),
                                    sc_overrides={"nx": "40", "ny": "40",
                                                  "nz": "40"})
    cfg = worker_sc_config(point)
    assert (cfg.nx, cfg.ny, cfg.nz) == (40, 40, 40)


def test_points_without_overrides_still_give_the_classic_config():
    """The GUI convergence tab builds points with only nx/extent/backend."""
    point = ScanPoint(lattice_path="x.dat", beam_config={}, nx=32,
                      grid_extent=6.0, step1=100.0, step2=50.0, seed=1,
                      use_gpu="cpu")
    cfg = worker_sc_config(point)
    assert cfg == SpaceChargeConfig(nx=32, ny=32, nz=32, grid_extent=6.0,
                                    use_gpu="cpu")


def test_build_scan_point_reports_the_resolution_notes(tmp_path):
    notes = []
    common.build_scan_point(_project(tmp_path, {"grid_nx": 64}),
                            sc_overrides={"method": "tsc_adaptive"},
                            notes=notes)
    assert notes and "tsc_adaptive" in notes[0]


def _bad_project(tmp_path, current):
    (tmp_path / "d.dat").write_text("DRIFT 100 20 0\nEND\n")
    p = tmp_path / "p.lgproj"
    p.write_text(json.dumps({"__kind__": "linac_gen_project", "__version__": 1,
                             "lattice_path": "d.dat",
                             "beam": {"current": current},
                             "convergence": {"kernel": "TSC"}}))
    return str(p)


@pytest.mark.parametrize("mode,current", [("envelope", 5.0), ("mp", 0.0)])
def test_points_without_space_charge_never_check_it(mode, current, tmp_path):
    """As before: an invalid space-charge setting does not stop an envelope
    point, or a point with no beam current, from being built and run."""
    point = common.build_scan_point(_bad_project(tmp_path, current), mode=mode)
    assert dict(point.sc_overrides)["kernel"] == "TSC"


def test_mp_point_checks_space_charge_in_its_worker(tmp_path):
    """An mp point with current is still built; its worker rejects the
    setting (one failed row, as before), not the whole campaign."""
    point = common.build_scan_point(_bad_project(tmp_path, 5.0), mode="mp")
    with pytest.raises(ValueError, match="kernel"):
        worker_sc_config(point)


def test_build_scan_point_warns_when_notes_are_not_collected(tmp_path):
    """Study and reliability campaigns do not collect the notes; they
    get them as warnings."""
    with pytest.warns(UserWarning, match="tsc_adaptive"):
        common.build_scan_point(_project(tmp_path, {"grid_nx": 64}),
                                sc_overrides={"method": "tsc_adaptive"})


@pytest.mark.parametrize("over", [{"method": "tsc_adaptive", "grid_mode": "fixed"},
                                  {"method": "cic_fixed", "grid_mode": "adaptive"}])
def test_scan_point_and_run_agree_on_a_contradicted_method(over, tmp_path):
    """The point re-bases on the method the settings match, as the config
    does (it used to keep the requested method's defaults)."""
    point = common.build_scan_point(_project(tmp_path, {}), sc_overrides=over)
    run_cfg = methods.resolve_sc({}, {"sc": dict(over)})
    assert worker_sc_config(point) == run_cfg
