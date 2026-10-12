"""A whole space-charge run gives the same bits for any thread count, on the
main thread (as the CLI runs it) and in a worker thread (as the GUI does):
the CIC deposit (one-thread loop vs parallel planes), the CIC and TSC
gathers, the FFT (one scipy call on one thread, the split on more, on a grid
large enough to split) and the torch space-charge engine (torch on one
thread during runs)."""
from __future__ import annotations

import threading

import numpy as np
import pytest

from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.simulation import Simulation
from linac_gen.pic import fft_split
from tests.helpers import kept_thread_setting
from tests.test_simulation_e2e import _fodo_lattice, _gaussian_beam


def _run(sc_kwargs, n_threads, in_worker, threads):
    threads.set_threads(n_threads)
    box = {}

    def go():
        sim = Simulation(_fodo_lattice(),
                         _gaussian_beam(n=3000, current=20.0, seed=4),
                         space_charge=SpaceChargeConfig(**sc_kwargs))
        box["rec"] = sim.run()
        box["particles"] = sim.beam.particles.copy()
    if in_worker:
        t = threading.Thread(target=go)
        t.start(); t.join()
    else:
        go()
    rec = box["rec"]
    series = {k: np.asarray(v, float) for k, v in vars(rec).items()
              if isinstance(v, list) and v
              and isinstance(v[0], (float, int, np.floating))}
    return series, box["particles"], rec.threads_info


def _same(a, b):
    sa, pa, _ = a
    sb, pb, _ = b
    assert sa.keys() == sb.keys()
    for k in sa:
        assert np.array_equal(sa[k], sb[k], equal_nan=True), k
    assert np.array_equal(pa, pb, equal_nan=True)


# nx = 32: the solver's doubled grid has 64**3 points, enough to split
CASES = {
    "cic": dict(nx=32, ny=32, nz=32, grid_extent=4.0, kernel="cic",
                grid_mode="fixed"),
    "tsc": dict(nx=32, ny=32, nz=32, grid_extent=4.0, kernel="tsc",
                grid_mode="fixed"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_numpy_engine(case, monkeypatch):
    assert (64 ** 3) >= fft_split.MIN_POINTS
    split_calls = []
    real = fft_split.rfftn

    def spy(x, workers):
        split_calls.append((x.size, workers))
        return real(x, workers)
    monkeypatch.setattr(fft_split, "rfftn", spy)
    monkeypatch.delenv("LINAC_GEN_FFT_WORKERS", raising=False)
    with kept_thread_setting() as threads:
        one = _run(CASES[case], 1, False, threads)
        assert split_calls and all(w == 1 for _, w in split_calls)
        n = len(split_calls)
        four = _run(CASES[case], 4, True, threads)
        assert all(w == 4 for _, w in split_calls[n:])
        n = len(split_calls)
        every = _run(CASES[case], "all", False, threads)    # main thread too
        assert all(w == threads.available_threads() for _, w in split_calls[n:])
    assert one[2]["threads"] == 1 and four[2]["threads"] == 4
    _same(one, four)
    _same(one, every)


def test_torch_engine():
    """torch's CPU work runs on one thread during a run, whatever torch's
    own count: torch splits its sums by thread and adds into the grid from
    several threads at once (torch 2.14 on Linux: different last bits at 1
    and 4 threads)."""
    torch = pytest.importorskip("torch")
    sc = dict(nx=16, ny=16, nz=16, grid_extent=4.0, sc_backend="torch",
              grid_mode="adaptive")
    with kept_thread_setting() as threads:
        torch.set_num_threads(4)                    # torch's own count
        one = _run(sc, 1, False, threads)
        assert torch.get_num_threads() == 4         # back after the run
        four = _run(sc, 4, True, threads)
    assert one[2]["torch"] == 1 and four[2]["torch"] == 1
    _same(one, four)
