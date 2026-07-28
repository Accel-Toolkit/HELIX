"""Custom build for the C++ PIC kernel extension via pybind11.

The rest of the project metadata lives in pyproject.toml. This file exists
only so `pip install` compiles linac_gen/csrc/pic_kernels.cpp into
linac_gen/_pic_kernels<ext>.so on the user's machine (no pre-built binary
ships in the wheel).

OpenMP linking
--------------
The kernel uses ``#pragma omp parallel for collapse(3)`` for parallel
deposit/interp.  ``collapse(3)`` requires OpenMP 3.0+.

- **Linux / gcc / clang:** ``-fopenmp``.  Picked statically below.
- **macOS / Apple clang:** ``-fopenmp`` is not recognised natively; use
  ``-Xpreprocessor -fopenmp`` for compilation, and link NO libomp at all —
  the extension's OpenMP symbols resolve at load time (``-undefined
  dynamic_lookup``, the standard macOS extension link mode) against the
  libomp that torch, a hard dependency, has already loaded.  Exactly one
  OpenMP runtime ever exists in the process.  Only ``omp.h`` is probed
  (env include dir, then Homebrew); without it the kernel builds serial.
- **Windows / MSVC:** ``/openmp:llvm`` (Visual Studio 2019 v16.10+) for the
  full OpenMP 3.0+ runtime that supports ``collapse(3)``.  The classic
  ``/openmp`` is OpenMP 2.0 and silently ignores ``collapse(3)`` — building
  with it would give a single-threaded outer-loop kernel and look like it
  worked.  Set ``LINAC_GEN_OPENMP_FALLBACK=1`` to force ``/openmp`` for
  older toolchains; you'll lose ``collapse(3)`` parallelism but it'll build.
- **Windows / MinGW or clang-cl:** ``-fopenmp`` like Linux.
- Windows flags are injected at build-extension time (we don't know the
  compiler type until ``build_ext`` runs), so the static ``_openmp_flags``
  returns ``-O3`` only for Windows; the ``_WindowsOpenMP`` build_ext adds
  the OpenMP switch.
"""
from __future__ import annotations

import os
import platform
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


def _macos_omp_header_dir():
    """Return the directory holding ``omp.h``, or None.

    Only the *header* is probed — the runtime library is deliberately NOT
    linked on macOS (see the Darwin branch of :func:`_openmp_flags`).
    """
    inc_candidates = [
        os.path.join(sys.prefix, "include"),
        "/opt/homebrew/opt/libomp/include",
        "/usr/local/opt/libomp/include",
    ]
    return next(
        (d for d in inc_candidates if os.path.isfile(os.path.join(d, "omp.h"))),
        None,
    )


def _openmp_flags():
    """Pick (extra_compile_args, extra_link_args, include_dirs) for OpenMP.

    ``-march=native`` is added on Unix toolchains so the compiler can emit
    host-specific SIMD (AVX2/AVX-512 on x86_64, full NEON on aarch64).
    Typical gain on the tight per-particle inner loops in pic_kernels.cpp
    is 1.3–1.8× over plain ``-O3``.  Binaries become host-specific, which
    is exactly what ``pip install -e .`` (the documented install path)
    wants; do *not* set this for release wheels intended to ship across
    machines.  Windows MSVC equivalent (`/arch:AVX2`) is injected by
    :class:`_Pybind11WithOpenMP` at build-extension time, where we know
    whether the compiler is MSVC.
    """
    if sys.platform == "darwin":
        inc = _macos_omp_header_dir()
        if inc is None:
            print(
                "setup.py: WARNING — omp.h not found; building the PIC "
                "kernels single-threaded.  `brew install libomp` (headers "
                "only are used) enables the parallel build.",
                file=sys.stderr,
            )
            return ["-O3", "-march=native"], [], []
        # Compile WITH OpenMP pragmas but link NO libomp.  macOS extension
        # modules are linked ``-undefined dynamic_lookup`` (that is how they
        # find Python's own symbols), so the kernel's ``omp_*``/``__kmpc_*``
        # references bind at load time to the libomp image torch has already
        # loaded into the process.  Linking a private copy is what aborted
        # fresh installs (OMP Error #15, two runtimes in one process): under
        # pip's PEP 517 build isolation torch is invisible at build time, so
        # the old torch-first library probe silently fell through to
        # Homebrew's libomp.  The single-runtime invariant is enforced by
        # importing torch before the kernel module — see the guarded imports
        # in linac_gen/pic/pic_solver.py and linac_gen/io/hdf5_output.py.
        return (
            ["-O3", "-march=native", "-Xpreprocessor", "-fopenmp", f"-I{inc}"],
            [],
            [inc],
        )
    if platform.system() == "Linux":
        return ["-O3", "-march=native", "-fopenmp"], ["-fopenmp"], []
    return ["-O3"], [], []


_compile, _link, _includes = _openmp_flags()


class _Pybind11WithOpenMP(build_ext):
    """Inject OpenMP flags at build time for compilers we can't detect statically.

    macOS and Linux paths are decided by :func:`_openmp_flags` at module load
    because the compiler family is fixed by ``sys.platform``.  Windows is
    different: ``setuptools`` may dispatch to MSVC, MinGW-w64, or clang-cl
    depending on what's installed, and each wants a different OpenMP switch.
    We pick the right one here, once ``self.compiler`` is constructed.

    Behaviour on non-Windows hosts: this method is a no-op (the flags are
    already set by :func:`_openmp_flags`).  So macOS and Linux builds are
    bit-identical to before this subclass existed.
    """

    def build_extensions(self) -> None:
        if sys.platform == "win32":
            self._inject_windows_openmp_flags()
        super().build_extensions()

    def _inject_windows_openmp_flags(self) -> None:
        ctype = getattr(self.compiler, "compiler_type", "")
        if ctype == "msvc":
            # /openmp:llvm needs Visual Studio 2019 v16.10+.  Older MSVC
            # rejects the colon-suffix form and the user can opt into the
            # legacy /openmp at the cost of losing collapse(3) parallelism.
            if os.environ.get("LINAC_GEN_OPENMP_FALLBACK") == "1":
                flag = "/openmp"
            else:
                flag = "/openmp:llvm"
            # ``/arch:AVX2`` is the MSVC equivalent of ``-march=native`` for
            # the common case (x86_64 desktops / Fermilab analysis machines
            # from the last decade).  Unset ``LINAC_GEN_NO_NATIVE_ARCH=1``
            # to skip if the binary must run on pre-Haswell CPUs (rare).
            native_flags = []
            if os.environ.get("LINAC_GEN_NO_NATIVE_ARCH") != "1":
                native_flags = ["/arch:AVX2"]
            for ext in self.extensions:
                ext.extra_compile_args = list(ext.extra_compile_args or []) + [
                    flag, *native_flags,
                ]
                # MSVC links the OpenMP runtime automatically when the switch
                # is set; no extra linker args needed.
        elif ctype in ("mingw32", "cygwin"):
            for ext in self.extensions:
                ext.extra_compile_args = list(ext.extra_compile_args or []) + [
                    "-fopenmp", "-march=native",
                ]
                ext.extra_link_args = list(ext.extra_link_args or []) + [
                    "-fopenmp",
                ]
        # else: unknown Windows toolchain, leave the kernel single-threaded.


ext_modules = [
    Pybind11Extension(
        "linac_gen._pic_kernels",
        ["linac_gen/csrc/pic_kernels.cpp"],
        extra_compile_args=_compile,
        extra_link_args=_link,
        include_dirs=_includes,
        cxx_std=14,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": _Pybind11WithOpenMP},
)
