"""Model ORM: analytic anchor, map vs tracking (both current regimes), restore and cancellation."""
from __future__ import annotations
import numpy as np
import pytest

from linac_gen.core.cancelled import OperationCancelled
from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.marker import Marker
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.steerer import Steerer
from linac_gen.orm import DeviceMap, resolve_devices
from linac_gen.orm.measured import MeasuredOrm
from linac_gen.orm.model import OrmModel
from tests.orm.conftest import fodo_with_devices


def _cfg(species="proton", energy=100.0, current=0.0):
    cfg = BeamConfig(); cfg.species = species; cfg.energy = energy; cfg.frequency = 352.2; cfg.current = current
    cfg.emit_nx = cfg.emit_ny = 0.2; cfg.emit_z = 0.3; cfg.beta_x = cfg.beta_y = 2.0; cfg.alpha_x = cfg.alpha_y = 0.0
    return cfg


def _selection(lat, trims, bpms):
    meas = {"x": MeasuredOrm(kick_plane="x", trims=list(trims), bpms={"x": list(bpms), "y": list(bpms)},
                             values={p: np.zeros((len(bpms), len(trims))) for p in "xy"}, errors={p: np.zeros((len(bpms), len(trims))) for p in "xy"}),
            "y": MeasuredOrm(kick_plane="y", trims=list(trims), bpms={"x": list(bpms), "y": list(bpms)},
                             values={p: np.zeros((len(bpms), len(trims))) for p in "xy"}, errors={p: np.zeros((len(bpms), len(trims))) for p in "xy"})}
    dm = DeviceMap(bpms={b: b for b in bpms}, trims={t: t for t in trims})
    return meas, resolve_devices(lat, meas, dm)


@pytest.mark.parametrize("species,sign", [("proton", 1.0), ("H-", -1.0)])
def test_drift_anchor(species, sign):
    """Steerer → drift L → BPM: R = sign(q)·1e3·L[m]/Bρ mm per T·m, and L mm/mrad."""
    lat = Lattice(); lat.add(Steerer("ST")); lat.add(Drift("D", 750.0, 20.0)); lat.add(Marker("BPM_A", is_bpm=True))
    cfg = _cfg(species=species); meas, sel = _selection(lat, ["ST"], ["BPM_A"])
    model = OrmModel(lat, cfg, sel, quads="all"); R = model.response()
    brho = ReferenceParticle(model.ref.species, 100.0, 352.2).brho
    assert model.kick_sign == sign
    np.testing.assert_allclose(R.per_mrad[("x", "x")], [[0.75]], rtol=1e-12)
    np.testing.assert_allclose(R.per_mrad[("y", "y")], [[0.75]], rtol=1e-12)
    np.testing.assert_allclose(R.per_Tm[("x", "x")], [[sign * 1e3 * 0.75 / brho]], rtol=1e-12)
    assert R.per_Tm[("x", "y")][0, 0] == 0.0 and R.per_Tm[("y", "x")][0, 0] == 0.0


@pytest.mark.parametrize("current", [0.0, 10.0])
def test_maps_agree_with_tracking(current):
    lat = fodo_with_devices(3)
    trims = [f"D{k:02d}T" for k in (1, 2, 3)]; bpms = [f"D{k:02d}{m}BPM" for k in (1, 2, 3) for m in (1, 2)]
    meas, sel = _selection(lat, trims, bpms)
    cfg = _cfg(current=current); model = OrmModel(lat, cfg, sel)
    assert len(model.quads) == 6                                   # every quad between the first trim and the last BPM
    R = model.response(); T = model.tracked_response(current_mA=current)
    assert model.selfcheck(R, T) < 1e-7
    assert (R.per_Tm[("x", "y")] == 0).all() and (R.per_Tm[("y", "x")] == 0).all()
    assert np.array_equal(R.per_Tm[("x", "x")][~sel.down], np.zeros(int((~sel.down).sum())))   # upstream = 0


def test_skew_quad_couples_planes():
    lat = fodo_with_devices(2)
    q = [e for e in lat.elements if isinstance(e, Quadrupole)][0]; q.skew_angle = 15.0
    meas, sel = _selection(lat, ["D01T"], ["D021BPM", "D022BPM"])
    R = OrmModel(lat, _cfg(), sel).response()
    assert np.abs(R.per_Tm[("x", "y")]).max() > 0.0


def test_scale_changes_response_and_is_restored():
    lat = fodo_with_devices(2)
    meas, sel = _selection(lat, ["D01T", "D02T"], ["D011BPM", "D012BPM", "D021BPM", "D022BPM"])
    quads = [e for e in lat.elements if isinstance(e, Quadrupole)]; quads[1].gradient_rel = 0.02      # pre-existing error slot
    model = OrmModel(lat, _cfg(), sel); g0 = [q.gradient for q in quads]
    R1 = model.response(); R2 = model.response(np.array([1.05, 0.95, 1.0, 1.0]))
    assert model.selfcheck(R1, R2) > 1e-3
    assert [q.gradient for q in quads] == g0 and quads[1].gradient_rel == 0.02 and quads[0].gradient_rel == 0.0
    with pytest.raises(ValueError):
        model.response(np.ones(3))


def test_cancellation_raises_and_restores():
    lat = fodo_with_devices(2)
    meas, sel = _selection(lat, ["D01T"], ["D021BPM"])
    model = OrmModel(lat, _cfg(), sel)
    with pytest.raises(OperationCancelled):
        model.response(np.ones(len(model.quads)) * 1.1, should_stop=lambda: True)
    assert all(q.gradient_rel == 0.0 for q in model.quads)
    with pytest.raises(OperationCancelled):
        model.tracked_response(should_stop=lambda: True)


@pytest.mark.slow
def test_fnalscl_maps_vs_tracking_and_prototype_class():
    import contextlib, io
    from pathlib import Path
    from linac_gen.cli import common
    repo = Path(__file__).resolve().parents[2]
    with contextlib.redirect_stdout(io.StringIO()):
        lat, cfg, _ = common.load_input(str(repo / "examples/piplattice/fnalscl.lgproj"))
    trims = [f"D{k}T" for k in ("01", "02", "03", "04", "11", "21", "22", "31", "32", "41", "42", "51", "52", "61", "62", "71", "72", "74")]
    bpms = ["D01BPM"] + [f"D{k}BPM" for k in ("02", "03", "11", "12", "13", "21", "22", "23", "31", "32", "33", "34", "41", "42", "43", "44", "51", "52", "53", "54", "61", "62", "63", "64", "71", "72", "73", "74")]
    meas, sel = _selection(lat, trims, bpms)
    model = OrmModel(lat, cfg, sel); R = model.response(); T = model.tracked_response()
    assert model.selfcheck(R, T) < 1e-6 and model.kick_sign == -1.0
    assert 4000 < np.abs(R.per_Tm[("x", "x")]).max() < 6000 and len(model.quads) == 31
