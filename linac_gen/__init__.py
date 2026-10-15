"""Linac_Gen: Particle tracking with PIC space charge for proton/ion linacs."""
import os as _os

# The OpenMP runtime must give each kernel loop the team it asks for (the
# HELIX thread setting, linac_gen.parallel.threads); with dynamic
# adjustment it may hand out fewer threads.  The results do not depend on
# the team size -- the speed does.  The runtime reads OMP_DYNAMIC once at
# initialization -- which happens as early as `import torch` -- so the pin
# must precede EVERY import that can pull libomp.  Package-init is the
# only place that reliably runs first (setdefault: an explicit user
# setting always wins).  No effect if the process imported torch before
# linac_gen.
_os.environ.setdefault("OMP_DYNAMIC", "FALSE")

__version__ = "1.19.0"


def __getattr__(name):
    # linac_gen.set_threads / get_threads, loaded on first use
    if name in ("set_threads", "get_threads"):
        from linac_gen.parallel import threads
        return getattr(threads, name)
    raise AttributeError(f"module 'linac_gen' has no attribute {name!r}")
