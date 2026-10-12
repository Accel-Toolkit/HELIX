"""The split FFT (linac_gen.pic.fft_split) gives scipy's one-thread
rfftn/irfftn bit for bit at any thread count -- its slabs start on multiples
of 8 lines, so pocketfft batches every line as in the one-thread call (some
builds round a batched and a lone line differently: scipy 1.18.1 on macOS
arm64).  The CPU backend uses it on every thread and never scipy's own pool.
Subprocesses started during transforms break nothing."""
from __future__ import annotations

import os
import subprocess
import sys
import threading

import numpy as np
import pytest

sfft = pytest.importorskip("scipy.fft")

from linac_gen.pic import fft_split  # noqa: E402
from linac_gen.pic.gpu_backend import _CpuBackend  # noqa: E402
from tests.helpers import kept_thread_setting  # noqa: E402

SHAPES = [(64, 64, 64), (66, 66, 130), (65, 63, 127), (97, 89, 101),
          (128, 64, 32)]


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("workers", [2, 3, 7, 14])
def test_bit_identical_to_scipy(shape, workers):
    x = np.random.default_rng(hash(shape) % 2**32).normal(size=shape)
    ref = sfft.rfftn(x, workers=1)
    ref_back = sfft.irfftn(ref, s=shape, workers=1)
    X = fft_split.rfftn(x, workers)
    assert X.dtype == np.complex128 and X.flags.c_contiguous
    assert np.array_equal(_bits(X.view(np.float64)), _bits(ref.view(np.float64)))
    spectrum = ref.copy()
    back = fft_split.irfftn(spectrum, shape, workers)
    assert np.array_equal(_bits(back), _bits(ref_back))
    assert np.array_equal(spectrum, ref)            # input left as it was


@pytest.mark.parametrize("shape", [(8, 6, 10), (5, 3, 4), (2, 2, 2),
                                   (1, 7, 9), (16, 1, 1)])
def test_small_and_degenerate_grids_split(shape, monkeypatch):
    """More slabs than planes, axes of length 1 or 2, an odd last axis."""
    monkeypatch.setattr(fft_split, "MIN_POINTS", 0)
    x = np.random.default_rng(1).normal(size=shape)
    ref = sfft.rfftn(x, workers=1)
    assert np.array_equal(_bits(fft_split.rfftn(x, 14).view(np.float64)),
                          _bits(ref.view(np.float64)))
    assert np.array_equal(_bits(fft_split.irfftn(ref, shape, 14)),
                          _bits(sfft.irfftn(ref, s=shape, workers=1)))


def test_anything_unusual_is_one_scipy_call():
    rng = np.random.default_rng(2)
    x32 = rng.normal(size=(64, 64, 64)).astype(np.float32)
    assert np.array_equal(fft_split.rfftn(x32, 8), sfft.rfftn(x32))
    x2 = rng.normal(size=(512, 512))
    assert np.array_equal(fft_split.rfftn(x2, 8), sfft.rfftn(x2))
    X = sfft.rfftn(rng.normal(size=(64, 64, 64)))
    padded = (70, 64, 64)                           # s that pads axis 0
    assert np.array_equal(fft_split.irfftn(X, padded, 8),
                          sfft.irfftn(X, s=padded))


def _spy_scipy(monkeypatch):
    calls = []
    for name in ("rfftn", "irfftn", "rfft", "irfft", "fft", "ifft"):
        real = getattr(sfft, name)

        def spy(*a, _real=real, _name=name, **k):
            calls.append((_name, k.get("workers")))
            return _real(*a, **k)
        monkeypatch.setattr(sfft, name, spy)
    return calls


def test_backend_threads_the_transform_off_the_main_thread(monkeypatch):
    """Off the main thread scipy only ever sees workers=1 -- its pool is
    never used -- and the result is scipy's."""
    monkeypatch.delenv("LINAC_GEN_FFT_WORKERS", raising=False)
    be = _CpuBackend()
    x = np.random.default_rng(3).normal(size=(64, 64, 64))
    ref = sfft.rfftn(x, workers=1)
    calls = _spy_scipy(monkeypatch)
    out = {}
    with kept_thread_setting() as threads:
        threads.set_threads(4)

        def run():
            out["X"] = be.rfftn(x)
            out["x"] = be.irfftn(out["X"], x.shape)
        t = threading.Thread(target=run)
        t.start(); t.join()
    assert np.array_equal(out["X"], ref)
    assert np.array_equal(out["x"], sfft.irfftn(ref, s=x.shape, workers=1))
    names = {n for n, _ in calls}
    assert {"rfft", "fft", "ifft", "irfft"} <= names     # the split ran
    assert all(w == 1 for _, w in calls), calls


def test_backend_splits_on_the_main_thread_too(monkeypatch):
    """scipy's own threading is used nowhere: in some builds its result
    depends on the worker count."""
    monkeypatch.delenv("LINAC_GEN_FFT_WORKERS", raising=False)
    be = _CpuBackend()
    x = np.random.default_rng(6).normal(size=(64, 64, 64))
    ref = sfft.rfftn(x, workers=1)
    calls = _spy_scipy(monkeypatch)
    with kept_thread_setting() as threads:
        threads.set_threads(3)
        X = be.rfftn(x)
    assert np.array_equal(X, ref)
    assert {"rfft", "fft"} <= {n for n, _ in calls}
    assert all(w == 1 for _, w in calls), calls


@pytest.mark.parametrize("n, lines", [(64, 64), (96, 49), (65, 63), (80, 41),
                                      (97, 89), (33, 47), (5, 3), (2, 1),
                                      (130, 66), (7, 8)])
def test_slabs_start_on_multiples_of_8_lines(n, lines):
    """Why the split equals the one-thread call in every build: pocketfft
    batches a call's lines 2, 4 or 8 at a time from its first line, and a
    slab that starts on a multiple of 8 lines batches them the same way."""
    for k in range(1, 20):
        slabs = fft_split._slabs(n, k, lines)
        assert slabs[0][0] == 0 and slabs[-1][1] == n
        assert all(b0 == a1 for (_, b0), (a1, _) in zip(slabs, slabs[1:]))
        assert all(a < b for a, b in slabs)
        assert all((a * lines) % fft_split.ALIGN == 0 for a, _ in slabs)
        assert len(slabs) <= k


@pytest.mark.skipif(os.name != "posix", reason="fork is POSIX")
def test_subprocesses_during_transforms(monkeypatch):
    """The failure the split exists for: a fork while a worker thread is
    in a transform (scipy's pool is shut down and restarted around it)."""
    monkeypatch.delenv("LINAC_GEN_FFT_WORKERS", raising=False)
    be = _CpuBackend()
    x = np.random.default_rng(4).normal(size=(64, 64, 64))
    ref = sfft.rfftn(x, workers=1)
    errors, done = [], threading.Event()
    count = [0]

    def transforms():
        try:
            while not done.is_set():
                if not np.array_equal(be.rfftn(x), ref):
                    errors.append("different result")
                count[0] += 1
        except Exception as exc:                    # noqa: BLE001
            errors.append(repr(exc))

    with kept_thread_setting() as threads:
        threads.set_threads(4)
        t = threading.Thread(target=transforms, daemon=True)
        t.start()
        try:
            for _ in range(40):
                subprocess.run([sys.executable, "-c", "pass"], check=True)
        finally:
            done.set()
            t.join(timeout=120)
    assert not t.is_alive(), "a transform hung"
    assert not errors, errors
    assert count[0] > 0
