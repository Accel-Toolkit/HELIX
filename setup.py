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
  ``-Xpreprocessor -fopenmp`` plus an absolute-path link to libomp.dylib.
  Probed at PyTorch's bundled copy first (so the kernel shares one OMP
  image with torch), then the conda env, then Homebrew.  Picked statically.
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


def _macos_libomp_paths():
    """Return (include_dir, lib_dir) for libomp, or (None, None).

    Priority order:
    1. **PyTorch's bundled libomp** (``site-packages/torch/lib/libomp.dylib``).
       If torch is installed in this Python, it will load *its own* libomp
       into the process for MPS / MKL operations.  Loading a *second*
       libomp from anywhere else into the same process segfaults on the
       first parallel region.  So when torch is present we link against
       its libomp to share a single image.
    2. The conda env's main libomp at ``sys.prefix/lib/libomp.dylib``
       (matches scipy / sklearn when torch is absent).
    3. Homebrew libomp at ``/opt/homebrew`` or ``/usr/local`` (system Python).
    Headers (``omp.h``) are taken from whichever location supplies them; we
    prefer ``sys.prefix/include`` since torch does not ship the header.
    """
    # Header probe: prefer the env's include dir, then Homebrew.
    inc_candidates = [
        os.path.join(sys.prefix, "include"),
        "/opt/homebrew/opt/libomp/include",
        "/usr/local/opt/libomp/include",
    ]
    include_dir = next(
        (d for d in inc_candidates if os.path.isfile(os.path.join(d, "omp.h"))),
        None,
    )
    if include_dir is None:
        return None, None
    # Library probe — pin to torch's libomp first to share its OMP image.
    torch_lib = os.path.join(
        sys.prefix, "lib", "python3."
    )
    # Find torch/lib via the import path rather than guessing site-packages.
    lib_candidates = []
    try:
        import importlib.util
        spec = importlib.util.find_spec("torch")
        if spec is not None and spec.origin is not None:
            torch_lib_dir = os.path.join(os.path.dirname(spec.origin), "lib")
            if os.path.isfile(os.path.join(torch_lib_dir, "libomp.dylib")):
                lib_candidates.append(torch_lib_dir)
    except Exception:
        pass
    lib_candidates.extend([
        os.path.join(sys.prefix, "lib"),
        "/opt/homebrew/opt/libomp/lib",
        "/usr/local/opt/libomp/lib",
    ])
    lib_dir = next(
        (d for d in lib_candidates
         if os.path.isfile(os.path.join(d, "libomp.dylib"))),
        None,
    )
    if lib_dir is None:
        return None, None
    return include_dir, lib_dir


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
        inc, lib = _macos_libomp_paths()
        if inc is None:
            print(
                "setup.py: WARNING — libomp not found; building without "
                "OpenMP.  Install with `brew install libomp` for parallel "
                "PIC kernels.",
                file=sys.stderr,
            )
            return ["-O3", "-march=native"], [], []
        # Link against the *exact* libomp.dylib at ``lib`` by absolute path.
        # We bypass ``-lomp`` and rpath search entirely so the LC_LOAD_DYLIB
        # entry points unambiguously at the chosen file — necessary because
        # multiple libomp copies coexist on conda+torch hosts and rpath search
        # order is fragile (conda's own ``-rpath $CONDA_PREFIX/lib`` injects
        # itself ahead of any we add).  Subsequent ``install_name_tool -change``
        # in the build then resolves the LC_LOAD entry to the absolute path.
        libfile = os.path.join(lib, "libomp.dylib")
        return (
            ["-O3", "-march=native", "-Xpreprocessor", "-fopenmp", f"-I{inc}"],
            ["-Xpreprocessor", "-fopenmp", libfile,
             f"-Wl,-rpath,{lib}"],
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
