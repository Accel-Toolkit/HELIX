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
- **Windows / MSVC:** ``/openmp`` (classic vcomp) BY DEFAULT.  ``/openmp:llvm``
  would give the full OpenMP 3.0+ runtime with ``collapse(3)``, and was the
  default until an external Windows install report proved it a guaranteed
  abort: torch (a hard dependency) ships Intel's ``libiomp5md.dll``, and
  ``pic_solver.py`` deliberately imports torch first, so an
  ``/openmp:llvm``-linked kernel initialises a second (LLVM) OpenMP runtime
  in the same process and the two ``abort()`` on the first PIC kick.
  Microsoft's vcomp coexists with libiomp.  The cost is that vcomp is
  OpenMP 2.0 and silently ignores ``collapse(3)`` (outer-loop-only
  parallelism) — measured on the reporting user's machine the kernel was
  still ~20x over pure Python.  Set ``LINAC_GEN_OPENMP_LLVM=1`` to opt back
  into ``/openmp:llvm`` for torch-free embedding processes (or if you know
  what ``KMP_DUPLICATE_LIB_OK`` does and accept the consequences).
  ``LINAC_GEN_OPENMP_FALLBACK=1`` (the old escape hatch) is obsolete: it is
  now the default and prints a notice.
- **Windows / MinGW or clang-cl:** ``-fopenmp`` like Linux (GNU-flavour
  runtime — shares the same dual-runtime caveat as ``:llvm`` next to torch).
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
from setuptools.errors import (
    CCompilerError,
    CompileError,
    ExecError,
    LinkError,
    PlatformError,
)

#: Set LINAC_GEN_REQUIRE_CPP=1 to make a failed C++ kernel build FATAL
#: (CI / packagers).  Default: degrade to the pure-Python fallback that
#: linac_gen/pic/pic_solver.py and elements/field_map_3d.py already
#: implement at runtime — a machine with no compiler can still install.
_require_cpp = os.environ.get("LINAC_GEN_REQUIRE_CPP") == "1"


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

    def run(self) -> None:
        try:
            super().run()
        except PlatformError as exc:
            # Compiler CONSTRUCTION failures ("Microsoft Visual C++ 14.0 or
            # greater is required", missing cc) can surface before
            # build_extensions(); degrade like a per-extension failure.
            if _require_cpp:
                raise
            self._warn_fallback("the C++ toolchain", exc)

    def build_extensions(self) -> None:
        if sys.platform == "win32":
            self._inject_windows_openmp_flags()
        super().build_extensions()

    def build_extension(self, ext) -> None:
        try:
            super().build_extension(ext)
        except (CCompilerError, CompileError, ExecError, LinkError,
                PlatformError) as exc:
            if _require_cpp:
                raise
            self._warn_fallback(ext.name, exc)

    @staticmethod
    def _warn_fallback(what, exc) -> None:
        bar = "=" * 70
        print(
            f"\n{bar}\n"
            f"WARNING: building {what} failed:\n    {exc}\n"
            "HELIX installs anyway and runs with the pure-Python fallback\n"
            "(slower, numerically equivalent).  Install a C++ compiler and\n"
            "reinstall to regain the fast kernels, or set\n"
            "LINAC_GEN_REQUIRE_CPP=1 to make this failure fatal.\n"
            f"{bar}\n",
            file=sys.stderr,
        )

    def _inject_windows_openmp_flags(self) -> None:
        ctype = getattr(self.compiler, "compiler_type", "")
        if ctype == "msvc":
            # Default: classic /openmp (vcomp).  It coexists with torch's
            # bundled Intel libiomp5md.dll; /openmp:llvm guarantees an
            # abort in any process that also imports torch (see the module
            # docstring).  /openmp:llvm remains available for torch-free
            # embedders via LINAC_GEN_OPENMP_LLVM=1.
            if os.environ.get("LINAC_GEN_OPENMP_FALLBACK") == "1":
                print(
                    "setup.py: NOTE — LINAC_GEN_OPENMP_FALLBACK is obsolete: "
                    "/openmp (vcomp) is now the Windows default.",
                    file=sys.stderr,
                )
            if os.environ.get("LINAC_GEN_OPENMP_LLVM") == "1":
                flag = "/openmp:llvm"
            else:
                flag = "/openmp"
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
                if ext.name.endswith("_fieldmap_kernels"):
                    # The field-map kernel's bit-identity contract with scipy
                    # RGI (tests/elements/test_fieldmap_kernels.py) forbids FMA
                    # contraction.  VS2022's /fp:precise emits no contractions
                    # by default (/fp:contract is the explicit opt-in we never
                    # pass); on older toolchains the bit-identity test fails
                    # loudly rather than skipping, and the runtime kill-switch
                    # is LINAC_GEN_FIELDMAP_KERNEL=0.
                    ext.extra_compile_args.append("/fp:precise")
        elif ctype in ("mingw32", "cygwin"):
            for ext in self.extensions:
                if ext.name.endswith("_fieldmap_kernels"):
                    # GCC defaults to -ffp-contract=fast in C++ — the
                    # same contraction ban as every other toolchain.
                    ext.extra_compile_args = list(
                        ext.extra_compile_args or []) + ["-ffp-contract=off"]
                ext.extra_compile_args = list(ext.extra_compile_args or []) + [
                    "-fopenmp", "-march=native",
                ]
                ext.extra_link_args = list(ext.extra_link_args or []) + [
                    "-fopenmp",
                ]
        # else: unknown Windows toolchain, leave the kernel single-threaded.


#: Contraction control for _fieldmap_kernels (Unix toolchains): REQUIRED by
#: its bit-identity contract with scipy RGI — mirrors csrc/CMakeLists.txt.
#: The MSVC counterpart (/fp:precise) is injected at build-extension time.
_fieldmap_compile = ([f for f in _compile] + ["-ffp-contract=off"]
                     if sys.platform != "win32" else list(_compile))

ext_modules = [
    Pybind11Extension(
        "linac_gen._pic_kernels",
        ["linac_gen/csrc/pic_kernels.cpp"],
        extra_compile_args=_compile,
        extra_link_args=_link,
        include_dirs=_includes,
        cxx_std=14,
        optional=not _require_cpp,
    ),
    Pybind11Extension(
        "linac_gen._fieldmap_kernels",
        ["linac_gen/csrc/fieldmap_kernels.cpp"],
        extra_compile_args=_fieldmap_compile,
        extra_link_args=_link,
        include_dirs=_includes,
        cxx_std=14,
        optional=not _require_cpp,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": _Pybind11WithOpenMP},
)
