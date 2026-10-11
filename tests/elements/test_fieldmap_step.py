"""The fused KD step (``_fieldmap_kernels.kd_step``) reproduces FieldMap3D's
numpy kick-then-drift step for every non-NaN value, bit for bit (a NaN
stays NaN; its sign bit may differ): every channel kind (RF electric,
RF magnetic, static electric and magnetic), absent field components, lost
particles, particles outside the map box and a NaN coordinate; independent
of the OpenMP thread count.  A call spy proves which path ran."""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements import field_map_3d as fm
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.io.field_map_data import FieldChannel, FieldMapData
from linac_gen.io.tracewin_geom import Channel

needs = pytest.mark.skipif(
    not (fm.kernel_available() and fm.fused_step_available()),
    reason="compiled fused KD step not built")


def _map(seed=0, kinds=("RF_E", "RF_B", "STAT_B")):
    rng = np.random.default_rng(seed)
    x = np.linspace(-15.0, 15.0, 9)
    y = np.linspace(-12.0, 12.0, 7)
    z = np.linspace(0.0, 120.0, 25)
    shape = (x.size, y.size, z.size)
    fd = FieldMapData(z=z, frequency=162.5)
    comps = {
        "RF_E": dict(Fx=rng.normal(size=shape), Fy=rng.normal(size=shape),
                     Fz=5.0 + rng.normal(size=shape)),
        # Fz absent: its zeros still go through the scaling
        "RF_B": dict(Fx=0.01 * rng.normal(size=shape),
                     Fy=0.01 * rng.normal(size=shape), Fz=None),
        "STAT_B": dict(Fx=None, Fy=None,
                       Fz=0.2 + 0.01 * rng.normal(size=shape)),
        "STAT_E": dict(Fx=0.1 * rng.normal(size=shape), Fy=None,
                       Fz=0.3 * rng.normal(size=shape)),
    }
    for k in kinds:
        fd.channels[getattr(Channel, k)] = FieldChannel(
            geometry=7, x=x, y=y, z=z, **comps[k])
    return fd


def _beam(n=3000, seed=1):
    ref = ReferenceParticle(species=PROTON, w_kin=2.1, frequency=162.5)
    b = Beam(ref=ref, n_particles=n, current=0.0)
    rng = np.random.default_rng(seed)
    b.particles[:] = rng.normal(size=(n, 6)) * [4.0, 2.0, 4.0, 2.0, 10.0,
                                                0.02]
    b.particles[:5, 0] = [30.0, -30.0, 0.0, 14.999999, 15.0]  # box edges
    b.particles[5, 2] = np.nan
    b.lost[10:400:7] = True
    return b


def _run(fd, fused, steps=60, phase=30.0, spy=None):
    prev = fm._USE_STEP
    prev_env = os.environ.get("LINAC_GEN_FIELDMAP_STEP")
    fm.use_fused_step(fused)
    try:
        el = FieldMap3D(name="FM", length=120.0, field_data=fd, scale=1.0,
                        phase=phase, frequency=162.5, n_steps=steps)
        b = _beam()
        ds = el.length / el.n_steps
        el._step_idx = 0
        for _ in range(el.n_steps):
            el.track_rk4(b, ds)
        return b
    finally:
        fm._USE_STEP = prev
        if prev_env is None:
            os.environ.pop("LINAC_GEN_FIELDMAP_STEP", None)
        else:
            os.environ["LINAC_GEN_FIELDMAP_STEP"] = prev_env


def _spied(monkeypatch):
    calls = []
    real = fm._kd_step

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)
    monkeypatch.setattr(fm, "_kd_step", spy)
    return calls


@needs
@pytest.mark.parametrize("kinds", [("RF_E", "RF_B", "STAT_B"), ("RF_E",),
                                   ("STAT_B",), ("STAT_E", "RF_B")])
def test_fused_step_is_bit_identical(monkeypatch, kinds):
    fd = _map(kinds=kinds)
    calls = _spied(monkeypatch)
    a = _run(fd, fused=True)
    assert len(calls) == 60                       # the fused path ran
    b = _run(fd, fused=False)
    assert len(calls) == 60                       # ... and here it did not
    assert np.array_equal(a.particles, b.particles, equal_nan=True)
    assert np.array_equal(a.lost, b.lost)
    assert (a.ref.w_kin, a.ref.phi_s, a.ref.s) == (b.ref.w_kin, b.ref.phi_s,
                                                     b.ref.s)
    alive = ~a.lost & ~np.isnan(a.particles).any(axis=1)
    assert np.any(a.particles[alive] != _beam().particles[alive])   # moved


@needs
def test_scipy_sampling_keeps_the_numpy_path(monkeypatch):
    fd = _map()
    calls = _spied(monkeypatch)
    prev = fm.fused_kernel_enabled()
    fm.use_fused_kernel(False)
    try:
        a = _run(fd, fused=True)
    finally:
        fm.use_fused_kernel(prev)
    assert calls == []
    b = _run(fd, fused=False)
    assert np.array_equal(a.particles, b.particles, equal_nan=True)


@needs
def test_a_channel_without_a_fused_pack_keeps_the_numpy_path(monkeypatch):
    calls = _spied(monkeypatch)
    prev = FieldMap3D.interp_kind
    FieldMap3D.interp_kind = "cubic"              # no fused packs
    try:
        _run(_map(), fused=True, steps=5)
    finally:
        FieldMap3D.interp_kind = prev
    assert calls == []


_THREAD_SCRIPT = r"""
import sys
import numpy as np
sys.path.insert(0, {root!r})
from tests.elements.test_fieldmap_step import _map, _run
b = _run(_map(), fused=True)
np.save(sys.argv[1], b.particles)
"""


@needs
def test_independent_of_thread_count(tmp_path):
    from linac_gen import _fieldmap_kernels as K
    if not K._openmp_enabled:
        pytest.skip("single-threaded build: the thread count cannot change "
                    "anything")
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    outs = []
    for t in ("1", "3"):
        f = tmp_path / f"t{t}.npy"
        env = dict(os.environ, OMP_NUM_THREADS=t, PYTHONPATH=root)
        r = subprocess.run([sys.executable, "-c",
                            _THREAD_SCRIPT.format(root=root), str(f)],
                           env=env, capture_output=True, text=True,
                           timeout=300)
        assert r.returncode == 0, r.stderr[-2000:]
        outs.append(np.load(f))
    assert np.array_equal(outs[0], outs[1], equal_nan=True)


@needs
def test_kernel_rejects_an_unusable_particle_array():
    fd = _map()
    el = FieldMap3D(name="FM", length=120.0, field_data=fd, scale=1.0,
                    phase=0.0, frequency=162.5, n_steps=10)
    gx, gy, gz, stacked, names = next(iter(el._fused_packs.values()))
    chan = (gx, gy, gz, stacked, np.array([0, 1, 2]), 1.0, True, None)
    P = np.zeros((4, 6), order="F")
    with pytest.raises(ValueError, match="C-contiguous"):
        fm._kd_step(P, np.arange(4), 1.0, [chan], 1.0, 1e-3, 0.0,
                    False, 1.0, False, 0.0, False, 0.0, 0.0)
    with pytest.raises(ValueError, match="index out of range"):
        fm._kd_step(np.zeros((4, 6)), np.array([5]), 1.0, [chan], 1.0, 1e-3,
                    0.0, False, 1.0, False, 0.0, False, 0.0, 0.0)

    def call(idx=np.arange(4), ch=chan):
        fm._kd_step(np.zeros((4, 6)), idx, 1.0, [ch], 1.0, 1e-3, 0.0,
                    False, 1.0, False, 0.0, False, 0.0, 0.0)
    # review: duplicates would race between threads; a mask or floats
    # would be converted silently
    with pytest.raises(ValueError, match="strictly increasing"):
        call(np.array([0, 2, 2]))
    with pytest.raises(ValueError, match="int64"):
        call(np.array([True, False, True, True]))
    with pytest.raises(ValueError, match="int64"):
        call(np.array([0.0, 1.0]))
    # review: a non-1-D axis used to pass and read out of bounds
    with pytest.raises(ValueError, match="malformed"):
        call(ch=(gx.reshape(-1, 1), gy, gz) + chan[3:])
    with pytest.raises(ValueError, match="ascending"):
        call(ch=(gx[::-1].copy(), gy, gz) + chan[3:])
    with pytest.raises(ValueError, match="slot"):
        call(ch=chan[:4] + (np.array([0, 1, -2]),) + chan[5:])


@needs
def test_the_switch_reaches_spawned_workers(monkeypatch):
    monkeypatch.setenv("LINAC_GEN_FIELDMAP_STEP", "1")
    prev = fm._USE_STEP
    try:
        fm.use_fused_step(False)
        assert os.environ["LINAC_GEN_FIELDMAP_STEP"] == "0"
        fm.use_fused_step(True)
        assert os.environ["LINAC_GEN_FIELDMAP_STEP"] == "1"
    finally:
        fm._USE_STEP = prev
