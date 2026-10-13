"""The particle part of a 1-D on-axis field-map slice (FieldMap.track_rk4,
geometry 1) runs in the compiled kernel (linac_gen._fieldmap_kernels
.fm1d_step) and must give the numpy code's particles BIT FOR BIT (raw bits),
at any thread count: electric channels (where the adiabatic damping and the
phase slip are not trivial), static and RF magnetic channels, several
channels at once, particles at r = 0 and at r = 1e-12 exactly, lost and
non-finite particles, a frequency jump, p_flag = 1.  Maps with any channel
that is not geometry 1 keep the numpy code."""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.particle import H_MINUS, PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements import field_map as fm_mod
from linac_gen.elements.field_map import FieldMap
from linac_gen.io.field_map_data import FieldChannel, FieldMapData
from linac_gen.io.tracewin_geom import Channel

try:  # noqa: SIM105  -- on macOS the kernel needs torch's OpenMP runtime
    import torch as _torch_runtime  # type: ignore  # noqa: F401
except Exception:                                     # pragma: no cover
    pass
KERNEL = fm_mod._fm1d_lookup()
needs_kernel = pytest.mark.skipif(KERNEL is None,
                                  reason="_fieldmap_kernels without fm1d_step")
THREADS = (1, 2, 7, 14)


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def _profile(L, n=241, kind="cos"):
    z = np.linspace(0.0, L, n)
    if kind == "cos":
        return z, np.cos(np.pi * (z / L - 0.5)) ** 2
    return z, np.exp(-((z - L / 2) / (0.2 * L)) ** 2)


def _map(channels, L=240.0, freq=162.5):
    fd = FieldMapData(z=np.linspace(0.0, L, 241), frequency=freq)
    for kind, peak, shape in channels:
        z, f = _profile(L, kind=shape)
        fd.channels[kind] = FieldChannel(geometry=1, z=z, Fz=peak * f)
    return fd


MAPS = {
    "rf_electric": ([(Channel.RF_E, 6.0, "cos")], 162.5),
    "static_magnetic": ([(Channel.STAT_B, 0.6, "gauss")], 0.0),
    "rf_magnetic": ([(Channel.RF_B, 0.05, "cos")], 162.5),
    "electric_and_solenoid": ([(Channel.RF_E, 4.0, "cos"),
                               (Channel.STAT_B, 0.4, "gauss")], 162.5),
}


def _beam(n=4000, seed=5, species=PROTON, w=2.1, freq=162.5):
    rng = np.random.default_rng(seed)
    ref = ReferenceParticle(species=species, w_kin=w, frequency=freq)
    b = Beam(ref=ref, n_particles=n, current=0.0)
    b.particles[:] = rng.normal(size=(n, 6)) * [1.5, 2.0, 1.5, 2.0, 8.0, 0.02]
    b.particles[0, [0, 2]] = 0.0                 # r = 0 exactly
    b.particles[1, [0, 2]] = [1e-12, 0.0]        # r = 1e-12: not zeroed
    b.particles[2, [0, 2]] = [0.6e-12, 0.0]      # below 1e-12: zeroed
    b.particles[3, [0, 2]] = [-0.0, -0.0]        # signed zeros
    b.lost[10:40] = True                         # lost particles stay put
    return b


def _run(elem_kwargs, fd, beam, kernel, monkeypatch, threads=None):
    if threads is not None:
        from linac_gen.parallel import threads as thr
        thr.set_threads(threads)
    monkeypatch.setenv("LINAC_GEN_FIELDMAP1D_KERNEL", "1" if kernel else "0")
    elem = FieldMap(name="M", length=fd.z[-1], field_data=fd, **elem_kwargs)
    ds = elem.length / elem.n_steps
    elem._step_idx = 0
    for _ in range(elem.n_steps):
        elem.track_rk4(beam, ds)
    return beam.particles.copy(), beam.ref.w_kin


@needs_kernel
@pytest.mark.parametrize("name", list(MAPS))
@pytest.mark.parametrize("p_flag", [0, 1])
@pytest.mark.parametrize("n", [12, 1500, 4000])
@pytest.mark.parametrize("team", ["default", "whole"])
def test_kernel_slices_equal_numpy(name, p_flag, n, team, monkeypatch):
    """12: the envelope's probes.  "default": the kernel's own team, about
    sqrt(n / 1500) threads (one for 12 and 1500 particles, two for 4000);
    "whole": every thread the setting allows (team_particles=0)."""
    import functools
    from tests.helpers import kept_thread_setting
    if team == "whole":
        monkeypatch.setattr(fm_mod, "_fm1d_kernel",
                            functools.partial(KERNEL, team_particles=0))
    channels, freq = MAPS[name]
    kw = dict(scale=1.0, phase=-30.0, frequency=freq, n_steps=60, p_flag=p_flag)
    with kept_thread_setting():
        ref_parts, ref_w = _run(kw, _map(channels, freq=freq), _beam(n=n),
                                False, monkeypatch)
        for t in THREADS:
            parts, w = _run(kw, _map(channels, freq=freq), _beam(n=n), True,
                            monkeypatch, threads=t)
            assert np.array_equal(_bits(parts), _bits(ref_parts)), (name, t)
            assert w == ref_w


@needs_kernel
@pytest.mark.parametrize("name", list(MAPS))
def test_envelope_matrices_equal_numpy(name, monkeypatch):
    """The envelope's transfer matrices track twelve probes through
    track_rk4 (fitted_matrix over the element, fitted_matrix_slice per
    slice; a magnetic-only map is refined to 0.2 mm steps)."""
    channels, freq = MAPS[name]
    out = {}
    for kernel in (False, True):
        monkeypatch.setenv("LINAC_GEN_FIELDMAP1D_KERNEL", "1" if kernel else "0")
        fd = _map(channels, freq=freq)
        elem = FieldMap(name="M", length=fd.z[-1], field_data=fd, scale=1.0,
                        phase=-20.0, frequency=freq, n_steps=40)
        ref = ReferenceParticle(species=PROTON, w_kin=2.1, frequency=162.5)
        whole = elem.fitted_matrix(ref.copy())
        elem._step_idx = 0
        slices = [elem.fitted_matrix_slice(ref.copy(), 12.0) for _ in range(5)]
        out[kernel] = (whole, slices)
    assert np.array_equal(_bits(out[True][0]), _bits(out[False][0]))
    for a, b in zip(out[True][1], out[False][1]):
        assert np.array_equal(_bits(a), _bits(b))


@needs_kernel
def test_damping_and_slip_are_exercised(monkeypatch):
    """With an electric channel the damping factor and the phase slip are
    not 1 and 0, so their order matters -- and still the bits agree."""
    channels, freq = MAPS["rf_electric"]
    kw = dict(scale=1.0, phase=-25.0, frequency=freq, n_steps=40)
    before = _beam().particles.copy()
    parts, _ = _run(kw, _map(channels, freq=freq), _beam(), True, monkeypatch)
    alive = ~_beam().lost
    assert not np.array_equal(parts[alive, 5], before[alive, 5])   # energy kicks
    assert not np.array_equal(parts[alive, 1], before[alive, 1])
    assert np.array_equal(parts[~alive], before[~alive])           # lost: untouched


@needs_kernel
def test_frequency_jump_and_hminus(monkeypatch):
    """A cavity at twice the beam frequency (the per-particle phase rescale
    on the first slice) with H- (charge -1)."""
    from tests.helpers import kept_thread_setting
    channels, _ = MAPS["electric_and_solenoid"]
    kw = dict(scale=0.8, phase=10.0, frequency=325.0, n_steps=50)
    with kept_thread_setting():
        a, _ = _run(kw, _map(channels, freq=325.0),
                    _beam(species=H_MINUS, w=10.0, freq=162.5), False, monkeypatch)
        b, _ = _run(kw, _map(channels, freq=325.0),
                    _beam(species=H_MINUS, w=10.0, freq=162.5), True, monkeypatch,
                    threads=4)
    assert np.array_equal(_bits(a), _bits(b))


@needs_kernel
def test_non_finite_particles_match(monkeypatch):
    channels, freq = MAPS["electric_and_solenoid"]
    kw = dict(scale=1.0, phase=0.0, frequency=freq, n_steps=20)

    def beam():
        b = _beam(n=500)
        b.particles[50, 0] = np.nan
        b.particles[51, 2] = np.inf
        b.particles[52, 1] = -np.inf
        return b
    a, _ = _run(kw, _map(channels, freq=freq), beam(), False, monkeypatch)
    b, _ = _run(kw, _map(channels, freq=freq), beam(), True, monkeypatch)
    assert np.array_equal(_bits(a), _bits(b))


@needs_kernel
def test_kernel_is_used_only_for_all_geometry_1_maps(monkeypatch):
    calls = []
    real = KERNEL
    monkeypatch.setattr(fm_mod, "_fm1d_kernel",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    channels, freq = MAPS["rf_electric"]
    kw = dict(scale=1.0, phase=0.0, frequency=freq, n_steps=10)
    _run(kw, _map(channels, freq=freq), _beam(n=200), True, monkeypatch)
    assert len(calls) == 10
    calls.clear()
    fd = _map(channels, freq=freq)
    z = fd.z
    fd.channels[Channel.STAT_B] = FieldChannel(geometry=9, z=z,
                                               Fz=np.full_like(z, 2.0))
    _run(kw, fd, _beam(n=200), True, monkeypatch)       # a quad-gradient channel
    assert not calls
    _run(kw, _map(channels, freq=freq), _beam(n=200), False, monkeypatch)
    assert not calls                                    # switched off


@needs_kernel
def test_direct_call_validates_its_inputs():
    P = np.zeros((5, 6))
    idx = np.arange(5, dtype=np.int64)
    rs = np.zeros(5)
    args = (0.0, 1e-3, 0.0, False, 1.0, False, 0.0, False, 0.0, 0.0)
    with pytest.raises(ValueError):
        KERNEL(P[:, :5], idx, rs, [], *args)
    with pytest.raises(ValueError):
        KERNEL(P, idx[::-1].copy(), rs, [], *args)
    with pytest.raises(ValueError):
        KERNEL(P, idx, rs[:3], [], *args)
    with pytest.raises(ValueError):
        KERNEL(P, idx, rs, [(1.0, True, np.ones(2), 0.0, 0.0)], *args)
    with pytest.raises(ValueError):
        KERNEL(P, idx, rs, [(1.0, True)], *args)


def test_importing_the_module_does_not_import_torch():
    """Every deck parse imports linac_gen.elements.field_map; the kernel (and
    torch, which it needs first on macOS) is looked up at the first 1-D
    slice instead."""
    import subprocess
    import sys
    code = ("import sys; import linac_gen.elements.field_map as f; "
            "import linac_gen.io.tracewin_parser; "
            "print('torch' in sys.modules, f._fm1d_kernel); "
            "k = f._fm1d_lookup(); print('torch' in sys.modules, "
            "k is None or callable(k))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout.split()
    # the lookup itself never imports torch either (on macOS it then finds
    # no kernel and the numpy code runs until something imports torch)
    assert out == ["False", "None", "False", "True"]


@needs_kernel
@pytest.mark.parametrize("bad", ["short", "long", "int64", "uint8"])
def test_a_malformed_lost_mask_raises_as_before(bad, monkeypatch):
    """The numpy code raises for a lost-mask that is not one bool per
    particle; the kernel path leaves such a beam to it."""
    channels, freq = MAPS["electric_and_solenoid"]
    errors = []
    for kernel in (False, True):
        monkeypatch.setenv("LINAC_GEN_FIELDMAP1D_KERNEL", "1" if kernel else "0")
        b = _beam(n=200)
        b.lost = {"short": np.zeros(150, bool), "long": np.zeros(250, bool),
                  "int64": np.zeros(200, np.int64),
                  "uint8": np.zeros(200, np.uint8)}[bad]
        elem = FieldMap(name="M", length=240.0, field_data=_map(channels, freq=freq),
                        scale=1.0, phase=0.0, frequency=freq, n_steps=4)
        elem._step_idx = 0
        with pytest.raises((IndexError, ValueError)) as info:
            elem.track_rk4(b, 60.0)
        errors.append(type(info.value))
    assert errors[0] is errors[1]


@needs_kernel
def test_flush_to_zero_reaches_every_thread(monkeypatch):
    """The kernel's threads compute in the calling thread's floating-point
    environment, as numpy does: with flush-to-zero on, values that would be
    subnormal are zero in both."""
    torch = pytest.importorskip("torch")
    import functools
    from tests.helpers import kept_thread_setting
    # the whole team the setting allows, however few the particles
    monkeypatch.setattr(fm_mod, "_fm1d_kernel",
                        functools.partial(KERNEL, team_particles=0))
    channels, freq = MAPS["electric_and_solenoid"]
    kw = dict(scale=1.0, phase=-30.0, frequency=freq, n_steps=20)

    def beam():
        b = _beam(n=3000)
        b.particles[:, [0, 2]] *= 1e-305          # subnormal fields and kicks
        b.particles[:, [1, 3]] *= 1e-305
        return b
    if not torch.set_flush_denormal(True):
        pytest.skip("no flush-to-zero control on this CPU")
    try:
        with kept_thread_setting():
            want, _ = _run(kw, _map(channels, freq=freq), beam(), False,
                           monkeypatch)
            got, _ = _run(kw, _map(channels, freq=freq), beam(), True,
                          monkeypatch, threads=4)
    finally:
        torch.set_flush_denormal(False)
    assert np.array_equal(_bits(got), _bits(want))
