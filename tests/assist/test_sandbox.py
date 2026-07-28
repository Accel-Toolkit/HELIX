"""The run_python sandbox: isolation properties, limits, data channels.
The subprocess is local (no sockets), so the zero-network conftest is
satisfied throughout."""
from __future__ import annotations

import pytest

import numpy as np

from linac_gen.assist.sandbox import run_python_sandbox


def test_computes_and_returns_stdout():
    out = run_python_sandbox("print(6 * 7)")
    assert out["returncode"] == 0
    assert out["stdout"].strip() == "42"


def test_cannot_import_linac_gen():
    """THE machine-safety property: empty PYTHONPATH means assistant
    code can never reach the simulator (or the GUI)."""
    out = run_python_sandbox("import linac_gen")
    assert out["returncode"] != 0
    assert "ModuleNotFoundError" in out.get("error_output", "")
    out2 = run_python_sandbox("import linac_gen_gui")
    assert out2["returncode"] != 0


def test_timeout_kills_runaway_code():
    out = run_python_sandbox("while True:\n    pass", timeout=1.0)
    assert "timed out" in out.get("error", "")


def test_png_is_captured_as_image():
    pytest.importorskip("matplotlib")
    code = (
        "import matplotlib.pyplot as plt\n"
        "plt.plot([0, 1], [0, 1])\n"
        "plot()\n"
        "print('drawn')\n")
    out = run_python_sandbox(code, timeout=45.0)
    assert out["returncode"] == 0
    assert out.get("img_b64")
    assert out.get("mime") == "image/png"
    assert "drawn" in out.get("caption", "")


def test_arrays_channel_round_trip():
    code = (
        "print(int(arrays['s'].size), float(arrays['sigma_x'].max()))\n")
    out = run_python_sandbox(
        code, arrays={"s": np.arange(10.0),
                      "sigma_x": np.array([1.0, 4.5, 2.0])})
    assert out["returncode"] == 0
    assert out["stdout"].split() == ["10", "4.5"]


def test_data_json_channel():
    out = run_python_sandbox("print(data['a'] + data['b'])",
                             data={"a": 2, "b": 3})
    assert out["stdout"].strip() == "5"


def test_no_arrays_means_arrays_is_none():
    out = run_python_sandbox("print(arrays is None)")
    assert out["stdout"].strip() == "True"
