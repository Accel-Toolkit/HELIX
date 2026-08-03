"""Linac_Gen: Particle tracking with PIC space charge for proton/ion linacs."""
import os as _os

# Reproducibility: the C++ PIC deposit kernel is bit-deterministic only
# at a FIXED OpenMP thread count; dynamic team-size adjustment would let
# it drift run-to-run.  The OpenMP runtime reads OMP_DYNAMIC once at
# initialization — which happens as early as `import torch` — so the
# pin must precede EVERY import that can pull libomp.  Package-init is
# the only place that reliably runs first (setdefault: an explicit user
# setting always wins).  No effect if the process imported torch before
# linac_gen.
_os.environ.setdefault("OMP_DYNAMIC", "FALSE")

__version__ = "1.5.1"
