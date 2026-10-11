"""The assistant's compute tools build their space charge from the GUI
session's Numerics tab (playing the part a project plays on the command
line), a named method, or the historical 32^3 +-5 sigma fallback, and say
which (linac_gen.assist.tools._tool_sc_config)."""
from __future__ import annotations

import pytest

from linac_gen.assist.tools import (TOOLS, WorkContext, _scan_point_sc,
                                    _tool_sc_config)
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.parallel.scan_pool import ScanPoint, worker_sc_config
from linac_gen.pic import methods


class _Session(WorkContext):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg

    def session_sc_config(self, current, continuous=False):
        return self._cfg


class _Broken(WorkContext):
    def session_sc_config(self, current, continuous=False):
        raise RuntimeError("the GUI did not return the Numerics tab's "
                           "settings")


def test_headless_fallback_is_the_historical_grid():
    cfg, info, warns = _tool_sc_config({}, WorkContext())
    assert (cfg.nx, cfg.grid_extent, cfg.use_gpu) == (32, 5.0, "cpu")
    assert cfg.method == "cic_fixed" and warns == []
    assert info["source"] == "fallback" and info["grid"] == 32
    assert _tool_sc_config({"space_charge": False}) == (
        None, {"space_charge": "off"}, [])


def test_a_named_method_without_a_session():
    cfg, info, _ = _tool_sc_config({"sc_method": "tsc_adaptive"},
                                   WorkContext())
    want = methods.resolve_sc({}, {"sc_method": "tsc_adaptive",
                                   "backend": "cpu"})
    assert cfg == want and cfg.method == "tsc_adaptive"
    assert info == {"source": "sc_method", "method": "tsc_adaptive",
                    "grid": 48, "grid_extent_sigma": 6.0,
                    "dc_kernel": "uniform", "csr": False}


def test_a_named_method_keeps_the_sessions_other_settings():
    """As --sc-method on a project the GUI saved with these settings
    (review: the DC kernel and CSR were dropped)."""
    tab = SpaceChargeConfig(nx=64, grid_extent=5.0, dc_kernel="pic2d",
                            csr_enabled=True, use_gpu="auto")
    cfg, info, warns = _tool_sc_config({"sc_method": "tsc_adaptive"},
                                       _Session(tab))
    saved = {"sc_method": "cic_fixed", "grid_nx": 64,
             "grid_extent_sigma": 5.0, "backend": "auto",
             "sc_backend": "numpy", "green_kind": "igf",
             "green_build": "exact", "kernel": "cic", "grid_mode": "fixed",
             "grid_card": "apply", "reuse_every": 1, "dc_kernel": "pic2d",
             "csr_enabled": True}
    assert cfg == methods.resolve_sc(saved, {"sc_method": "tsc_adaptive"})
    assert (cfg.nx, cfg.dc_kernel, cfg.csr_enabled, cfg.use_gpu) == (
        48, "pic2d", True, "auto")
    assert info["source"] == "sc_method with the Numerics tab"
    assert warns == []


@pytest.mark.parametrize("name", ["tsc_adaptive", "custom"])
def test_naming_the_sessions_own_method_keeps_its_settings(name):
    tab = SpaceChargeConfig(method="tsc_adaptive", nx=64, ny=64, nz=64,
                            grid_extent=4.5, dc_kernel="gaussian")
    cfg, _, _ = _tool_sc_config({"sc_method": name}, _Session(tab))
    assert cfg == tab


def test_the_gui_session_settings_run_and_grid_overrides_apply():
    tab = SpaceChargeConfig(method="tsc_adaptive_reuse2", nx=48)
    cfg, info, _ = _tool_sc_config({}, _Session(tab))
    assert cfg is tab and info["source"] == "Numerics tab"
    cfg, info, _ = _tool_sc_config({"grid": 40, "grid_extent": 7.5},
                                   _Session(tab))
    assert (cfg.nx, cfg.ny, cfg.nz, cfg.grid_extent) == (40, 40, 40, 7.5)
    assert cfg.method == "tsc_adaptive_reuse2" and cfg.reuse_every == 2
    assert (info["grid"], info["grid_extent_sigma"]) == (40, 7.5)


def test_a_session_that_does_not_answer_falls_back_with_a_warning():
    cfg, info, warns = _tool_sc_config({}, _Broken())
    assert cfg.nx == 32 and info["source"] == "fallback"
    assert len(warns) == 1 and "fallback" in warns[0]
    assert "did not return" in warns[0]
    cfg, info, warns = _tool_sc_config({"sc_method": "cic_adaptive"},
                                       _Broken())
    assert cfg.method == "cic_adaptive" and info["source"] == "sc_method"
    assert len(warns) == 1 and "'cic_adaptive'" in warns[0]
    assert "DC kernel" in warns[0]


def test_an_unknown_method_is_refused_before_the_gui_is_asked():
    class _Counting(WorkContext):
        calls = 0

        def session_sc_config(self, current, continuous=False):
            _Counting.calls += 1
    with pytest.raises(ValueError, match="valid names"):
        _tool_sc_config({"sc_method": "cic_fast"}, _Counting())
    assert _Counting.calls == 0


@pytest.mark.parametrize("cfg", [
    SpaceChargeConfig(nx=32, ny=32, nz=32, grid_extent=5.0, use_gpu="auto"),
    SpaceChargeConfig(method="tsc_adaptive_reuse4", nx=40, dc_kernel="pic2d",
                      csr_enabled=True),
    SpaceChargeConfig(method="custom", nx=56, kernel="tsc",
                      green_build="symmetric", grid_card="ignore"),
])
def test_a_scan_point_carries_the_configuration_exactly(cfg):
    point = ScanPoint(lattice_path="x.dat", beam_config={}, step1=1.0,
                      step2=1.0, **_scan_point_sc(cfg))
    assert worker_sc_config(point) == cfg


def test_parameter_scan_fallback_point_equals_the_old_one():
    """Headless, an mp parameter scan builds the configuration its fixed
    32^3 +-5 sigma point always built (FFT device "auto")."""
    cfg, _, _ = _tool_sc_config({}, WorkContext(), device="auto")
    new = ScanPoint(lattice_path="x.dat", beam_config={}, step1=1.0,
                    step2=1.0, **_scan_point_sc(cfg))
    old = ScanPoint(lattice_path="x.dat", beam_config={}, nx=32,
                    grid_extent=5.0, step1=1.0, step2=1.0)
    assert worker_sc_config(new) == worker_sc_config(old)


def test_schemas_offer_the_method_and_no_hidden_grid_default():
    for tool in ("run_mp", "run_train", "parameter_scan"):
        props = TOOLS[tool].schema["properties"]
        assert props["sc_method"]["type"] == "string"
        assert "default" not in props["grid"]
        assert "default" not in props["grid_extent"]
