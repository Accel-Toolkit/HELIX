"""Isolated Python execution for assistant analysis (`run_python`).

Ported from the MIRAGE assistant.  Safety model: the child process gets
an **empty PYTHONPATH**, a scratch HOME/cwd, AND ``linac_gen`` /
``linac_gen_gui`` are hard-blocked in the preamble (``None`` poisoning
in ``sys.modules`` — necessary because an installed or editable
linac_gen registers a ``.pth`` import finder that survives
``python -I``).  Assistant-written code can never drive the simulator;
it only sees the data explicitly handed to it.  Plus a wall-clock
timeout, and on POSIX a CPU rlimit (25 s soft / 30 s hard) and a 50 MB
file-size rlimit.  It still runs real code on this machine with
numpy/scipy/matplotlib available — a *machine-safe analysis sandbox*,
not a hardened security jail.

Data channels into the child:
* ``data``  — small JSON (≤ 200 kB) via the ``HELIX_DATA`` env var,
  exposed as the ``data`` variable;
* ``arrays`` — full-precision numpy arrays written to ``data.npz`` in
  the child's cwd (results columns the caller selected), exposed as the
  ``arrays`` dict-like when present.

Output channels: stdout/stderr/returncode, plus the newest ``*.png`` in
the scratch dir returned as base64 — the caller shows it to the model
as a real image (the existing ``img_b64`` plumbing).
"""
from __future__ import annotations

import base64
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

_PREAMBLE = """\
import json as _json, os as _os, sys as _sys
# HARD-BLOCK the simulator: an installed/editable linac_gen (a .pth
# finder survives `python -I`) would otherwise be importable even with
# an empty PYTHONPATH.  None in sys.modules makes `import linac_gen`
# raise ImportError regardless of any finder.
_sys.modules["linac_gen"] = None
_sys.modules["linac_gen_gui"] = None
_sys.meta_path = [f for f in _sys.meta_path
                  if "linac_gen" not in getattr(f, "__module__", "")]
try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass
data = _json.loads(_os.environ.get("HELIX_DATA", "null"))
arrays = None
if _os.path.exists("data.npz"):
    try:
        import numpy as _np
        arrays = _np.load("data.npz")
    except Exception:
        arrays = None
def plot(path="plot.png"):
    import matplotlib.pyplot as _plt
    _plt.savefig(path, dpi=110, bbox_inches="tight")
    _plt.close("all")
# --- assistant code below ---
"""


def _limits():                                # POSIX; runs in the child
    import resource
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (25, 30))
    except Exception:                                       # noqa: BLE001
        pass
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE,
                           (50 * 1024 * 1024, 50 * 1024 * 1024))
    except Exception:                                       # noqa: BLE001
        pass


def run_python_sandbox(code: str, data=None, arrays=None,
                       timeout: float = 30.0) -> dict:
    """Execute ``code`` in the isolated subprocess; see module docstring.

    ``arrays``: optional dict of name → 1-D numpy array, delivered as
    ``data.npz`` (full precision — no JSON size cap).
    Returns ``{stdout, returncode[, stderr|error_output][, img_b64,
    mime, caption][, error]}``.
    """
    timeout = min(max(float(timeout), 1.0), 60.0)
    scratch = tempfile.mkdtemp(prefix="helix_py_")
    try:
        if arrays:
            import numpy as np
            np.savez(os.path.join(scratch, "data.npz"),
                     **{k: np.asarray(v) for k, v in arrays.items()})
        script = os.path.join(scratch, "_run.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(_PREAMBLE + str(code))
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
               "HOME": scratch, "TMPDIR": scratch, "MPLBACKEND": "Agg",
               "PYTHONPATH": "",               # cannot import linac_gen
               "HELIX_DATA": json.dumps(data, default=str)[:200_000],
               "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2",
               "MPLCONFIGDIR": scratch}
        try:
            p = subprocess.run(
                [sys.executable, "-I", "_run.py"], cwd=scratch, env=env,
                capture_output=True, text=True, timeout=timeout,
                preexec_fn=_limits if os.name == "posix" else None)
            out, err, rc = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired:
            return {"error": f"timed out after {timeout:.0f} s — the "
                             "code ran too long", "stdout": "",
                    "returncode": -1}
        result: dict = {"stdout": (out or "")[-6000:], "returncode": rc}
        if rc != 0 and err:
            result["error_output"] = err[-2000:]
        elif err:
            result["stderr"] = err[-1000:]
        pngs = sorted(glob.glob(os.path.join(scratch, "*.png")),
                      key=os.path.getmtime)
        if pngs:
            try:
                with open(pngs[-1], "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                cap = "python analysis output"
                if result["stdout"].strip():
                    cap += ":\n" + result["stdout"].strip()[-1500:]
                result.update(img_b64=b64, mime="image/png", caption=cap)
            except OSError:
                pass
        return result
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
