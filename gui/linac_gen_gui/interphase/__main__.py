"""Entry point: python -m linac_gen_gui.interphase."""
# Cap BLAS thread counts BEFORE numpy/scipy load.  OpenBLAS's parallel LU
# (`dgetrf_parallel`) allocates a large workspace on the calling thread's
# stack and SIGBUSes any QThread (which has a ~544 KB stack on macOS).
# These env vars are only consulted at BLAS library init, so they MUST be
# set before the first numpy import.  See:
# https://github.com/numpy/numpy/issues/24555
# (HELIX's own kernels, FFT and torch take the "CPU threads" setting, not
# these -- see app.py.)
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"
    # the GUI's cap, not the user's: HELIX's own threads ignore it
    os.environ["HELIX_GUI_OMP_CAP"] = "1"
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from linac_gen_gui.interphase.app import main   # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
