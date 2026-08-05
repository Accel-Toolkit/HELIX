"""Optional C++ extension build (setup.py).

A machine with no working compiler must still be able to install HELIX
(the runtime pure-Python fallback already exists; before this guard the
unconditional ext_modules failed the whole pip transaction — external
Windows install report, issue 2).  Exercised via ``setup.py build_ext``
in a scratch copy with a compiler that always fails — never via
``pip install -e .``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="uses CC=/usr/bin/false to force a failure")


def _scratch_repo(tmp_path: Path) -> Path:
    root = tmp_path / "pkg"
    (root / "linac_gen" / "csrc").mkdir(parents=True)
    shutil.copy(REPO / "setup.py", root / "setup.py")
    shutil.copy(REPO / "pyproject.toml", root / "pyproject.toml")
    for src in (REPO / "linac_gen" / "csrc").glob("*.cpp"):
        shutil.copy(src, root / "linac_gen" / "csrc" / src.name)
    (root / "linac_gen" / "__init__.py").write_text("", encoding="utf-8")
    return root


def _build(root: Path, extra_env: dict) -> subprocess.CompletedProcess:
    env = dict(os.environ, CC="/usr/bin/false", CXX="/usr/bin/false",
               **extra_env)
    env.pop("LINAC_GEN_REQUIRE_CPP", None)
    env.update(extra_env)
    return subprocess.run(
        [sys.executable, "setup.py", "build_ext", "-b", "build", "-t", "build"],
        cwd=root, env=env, capture_output=True, text=True, timeout=300)


def test_compiler_failure_degrades_to_fallback(tmp_path):
    root = _scratch_repo(tmp_path)
    proc = _build(root, {})
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "pure-Python fallback" in proc.stderr


def test_require_cpp_makes_failure_fatal(tmp_path):
    root = _scratch_repo(tmp_path)
    proc = _build(root, {"LINAC_GEN_REQUIRE_CPP": "1"})
    assert proc.returncode != 0


def test_real_build_produces_importable_kernels(tmp_path):
    """The pip artifact itself must WORK, not merely build: compile both
    extensions for real, then import them in a fresh process with torch
    loaded first (the single-OpenMP-image discipline field_map_3d.py and
    pic_solver.py enforce).  Caught live on macOS: the kernel links no
    libomp by design, so without a torch-first import its ___kmpc_*
    symbols are unresolved and the fused sampler silently degraded to
    scipy RGI in every pip install."""
    root = _scratch_repo(tmp_path)
    env = dict(os.environ)
    env.pop("LINAC_GEN_REQUIRE_CPP", None)
    proc = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "-b", "build", "-t",
         "build/t"],
        cwd=root, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "pure-Python fallback" not in proc.stderr, proc.stderr[-2000:]
    # Import from cwd=build so sys.path[0] resolves linac_gen to the
    # freshly built artifacts — an editable-install .pth on the host
    # would otherwise silently substitute the dev tree's kernels and
    # turn this test into a false positive (caught live).  The __file__
    # assertions pin the provenance.
    (root / "build" / "linac_gen" / "__init__.py").write_text(
        "", encoding="utf-8")
    check = subprocess.run(
        [sys.executable, "-c",
         "import torch\n"
         "from linac_gen import _pic_kernels, _fieldmap_kernels\n"
         "for m in (_pic_kernels, _fieldmap_kernels):\n"
         "    assert 'build' in m.__file__, m.__file__\n"
         "print('kernels-ok')"],
        cwd=root / "build", env=env,
        capture_output=True, text=True, timeout=120)
    assert check.returncode == 0, (check.stdout + check.stderr)[-2000:]
    assert "kernels-ok" in check.stdout
