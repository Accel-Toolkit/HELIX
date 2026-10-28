"""The CUDA session's scripts (run on the GPU machine) work: the
verification in numba's CUDA simulator at its tiny size, and the roofline's
operation count on the CPU."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_the_verification_script_in_the_simulator(tmp_path):
    pytest.importorskip("numba")
    env = dict(os.environ)
    env.update({"NUMBA_ENABLE_CUDASIM": "1", "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1"})
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_cuda_ring.py"), "--tiny",
                        "--out", str(tmp_path)], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=1800)
    assert r.returncode == 0 and "ALL PASS" in r.stdout, (r.stdout[-2000:], r.stderr[-2000:])
    rep = json.loads(next(tmp_path.glob("*_tiny.json")).read_text())
    assert rep["all_pass"] and rep["device"]["name"] == "numba CUDA simulator"
    assert rep["ramp, CPU vs GPU and restart"]["reference_bitwise"]
    sc = rep["space charge, CPU vs GPU and restart"]
    assert sc["deposit_bitwise"] and sc["restart_bitwise"] and sc["run_to_run_bitwise"]
    assert sc["z_bins_same"] and sc["z_bins_used"][0][1] == 256      # (z_bins "auto")
    inj = rep["injection, CPU vs GPU and restart"]
    assert inj["record_same"] and inj["restart_bitwise"] and inj["run_to_run_bitwise"]
    # (two injection turns of three particles: the run restarted between them)
    assert inj["injected"] == 6


def test_the_roofline_counts_the_kernels_work():
    pytest.importorskip("numba")
    saved = list(sys.path)
    sys.path.insert(0, str(ROOT / "benchmarks" / "ring"))
    try:
        import gpu_roofline as gr
    finally:
        # (the module puts the repository root first when imported: the
        # whole path is put back as it was)
        sys.path[:] = saved
    import copy

    from linac_gen.cli.common import load_lattice
    from linac_gen.ring.compile import compile_from_config
    from linac_gen.ring.config import RingConfig
    rl = compile_from_config(copy.deepcopy(load_lattice(str(gr.DECK))), RingConfig())
    n = gr.flops_per_particle_turn(rl)
    # (the simple Booster, exact bends with automatic steps: millions of
    # operations per particle and turn, the same count every time)
    assert 1e6 < n < 1e8 and n == gr.flops_per_particle_turn(rl)
