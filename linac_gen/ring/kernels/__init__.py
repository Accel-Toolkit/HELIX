"""Ring-mode particle kernels.

* :mod:`.cartesian` — element maps with the bend multipoles expanded in
  Cartesian coordinates (ported code, Apache-2.0: see THIRD_PARTY_NOTICES.md);
* :mod:`.curvilinear` — exact element maps with the bend multipoles in the
  curved frame (ported code, BSD-3-Clause-LBNL: see THIRD_PARTY_NOTICES.md);
* :mod:`.nodes` — the node table and the per-particle step through any node;
* :mod:`.drivers` — the loops over particles, turns and nodes;
* :mod:`.cpu` — those loops compiled by numba (serial and multi-core), or run
  as Python without numba;
* :mod:`.cuda` — the same loops as CUDA kernels.

Each map is written once, on plain floats, and decorated with
:func:`._jit.jitable`, so the CPU, the CUDA GPU and plain Python execute the
same source.
"""
