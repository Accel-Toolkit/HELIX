"""Entry point: python -m linac_gen_gui.interphase."""
# Cap BLAS thread counts BEFORE numpy/scipy load.  OpenBLAS's parallel LU
# (`dgetrf_parallel`) allocates a large workspace on the calling thread's
# stack and SIGBUSes any QThread (which has a ~544 KB stack on macOS).
# These env vars are only consulted at BLAS library init, so they MUST be
# set before the first numpy import.  See:
# https://github.com/numpy/numpy/issues/24555
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from linac_gen_gui.interphase.app import main   # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
