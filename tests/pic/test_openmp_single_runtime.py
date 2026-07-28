"""Single-OpenMP-runtime invariant for the C++ PIC kernels.

Regression pin for the 2026-07-28 fresh-clone audit: a pip-built kernel on
macOS used to link its own libomp.dylib (pip's PEP 517 build isolation hid
torch from setup.py's probe, which fell through to Homebrew's copy), and the
process then held two OpenMP runtimes — libomp aborts the whole process with
``OMP: Error #15`` at the first space-charge kick.  The fix links NO libomp
into the extension; its OpenMP symbols resolve at load time against the
runtime torch already brought into the process.

These tests run the killer sequence in clean subprocesses so an abort can
never take the pytest process down with it.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
import torch  # noqa: F401  — must load libomp BEFORE the kernel import below;
#               without it the kernel's dynamic-lookup omp symbols have nothing
#               to bind to and the import below fails (cleanly) even on a tree
#               where the extension IS built.

pytest.importorskip(
    "linac_gen._pic_kernels",
    reason="C++ kernels not built in this tree (source-run dev checkout)",
)

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("KMP_DUPLICATE_LIB_OK", None)   # never let the unsafe override mask a dup
    env["PYTHONPATH"] = _REPO + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=300, env=env, cwd=_REPO,
    )


_KILLER_SEQUENCE = """
import torch                                    # loads the ONE libomp
import numpy as np
from linac_gen._pic_kernels import deposit_cic  # must borrow, not bring, a runtime
rng = np.random.default_rng(0)
n = 20000                                       # big enough to spawn a real team
coords = rng.normal(0.0, 1e-3, (n, 3))
charges = np.full(n, 1e-12)
rho = deposit_cic(coords, charges,
                  np.array([-5e-3] * 3), np.array([5e-3] * 3),
                  np.array([32, 32, 32]))
assert np.isfinite(rho).all() and rho.sum() > 0
print("SINGLE_RUNTIME_OK", rho.sum())
"""


def test_torch_plus_kernel_in_one_process():
    """The exact fresh-user sequence: torch first, then a parallel deposit."""
    res = _run(_KILLER_SEQUENCE)
    assert "OMP: Error" not in res.stderr, res.stderr
    assert res.returncode == 0, f"aborted (rc={res.returncode}):\n{res.stderr}"
    assert "SINGLE_RUNTIME_OK" in res.stdout


def test_solver_consumer_path_survives():
    """The in-package consumer (pic_solver preloads torch) must not abort."""
    res = _run(
        "import numpy as np\n"
        "from linac_gen.pic import pic_solver as ps\n"
        "rng = np.random.default_rng(1)\n"
        "coords = rng.normal(0.0, 1e-3, (20000, 3))\n"
        "charges = np.full(20000, 1e-12)\n"
        "g = (np.array([-5e-3]*3), np.array([5e-3]*3), np.array([32]*3))\n"
        "rho = ps.deposit_cic(coords, charges, *g)\n"
        "import torch\n"                       # late import must also be safe
        "rho2 = ps.deposit_cic(coords, charges, *g)\n"
        "assert np.array_equal(rho, rho2)\n"
        "print('CONSUMER_OK')\n"
    )
    assert "OMP: Error" not in res.stderr, res.stderr
    assert res.returncode == 0, f"aborted (rc={res.returncode}):\n{res.stderr}"
    assert "CONSUMER_OK" in res.stdout


def test_cpp_matches_python_deposit():
    """External anchor: the C++ deposit agrees with the pure-Python one."""
    from linac_gen._pic_kernels import deposit_cic as cpp
    from linac_gen.pic.charge_deposition import deposit_cic as py

    rng = np.random.default_rng(2)
    coords = rng.normal(0.0, 1e-3, (5000, 3))
    charges = rng.uniform(0.5e-12, 1.5e-12, 5000)
    args = (np.array([-5e-3] * 3), np.array([5e-3] * 3), np.array([32, 32, 32]))
    np.testing.assert_allclose(
        cpp(coords, charges, *args), py(coords, charges, *args),
        rtol=1e-12, atol=0.0,
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="Mach-O check is macOS-only")
@pytest.mark.skipif("conda" in sys.prefix.lower() or "anaconda" in sys.prefix.lower(),
                    reason="conda envs share one runtime; a legacy locally-built "
                           "kernel may link conda/Homebrew libomp legitimately")
def test_kernel_links_no_private_libomp():
    """The .so must not carry its own libomp LC_LOAD entry (the old bug)."""
    import linac_gen._pic_kernels as k

    out = subprocess.run(
        ["otool", "-L", k.__file__], capture_output=True, text=True, check=True
    ).stdout
    assert "libomp" not in out, f"kernel links a private libomp again:\n{out}"
