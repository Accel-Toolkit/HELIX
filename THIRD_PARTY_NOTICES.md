# Third-party notices

HELIX is GPL-3.0-or-later.  The following third-party material is included or
adapted; each keeps its own licence, reproduced or referenced here.

| Material | Where in HELIX | Source | Licence |
|---|---|---|---|
| Simplified Fermilab Booster lattice (`sbbooster-cooked-rfon.madx`), a public test deck: the upstream file with a three-line provenance header added | `tests/ring/fixtures/simple_booster.madx` | ImpactX `examples/simple_booster/` (BLAST-ImpactX/impactx @ 472938df) | BSD-3-Clause-LBNL — full text in `tests/ring/fixtures/IMPACTX_LICENSE.txt`, ImpactX's notices in `tests/ring/fixtures/IMPACTX_NOTICE.txt` |
| Element maps of ring mode (drifts, thick magnets and their integrators, multipole kicks, pole faces — linear, wedge, dipole and multipole fringe —, frame rotations and shifts, RF cavity, the energy helpers), translated from C to Python and modified | `linac_gen/ring/kernels/cartesian.py`; `exact_edge` in `linac_gen/ring/kernels/nodes.py`; `magnet_config` in `linac_gen/ring/compile/blocks.py` | Xsuite: xtrack v0.103.5 (`beam_elements/elements_src/track_drift.h`, `track_magnet_drift.h`, `track_magnet_kick.h`, `track_magnet.h`, `track_magnet_edge.h`, `track_magnet_configure.h`, `track_dipole_edge_linear.h`, `track_yrotation.h`, `track_wedge.h`, `track_dipole_fringe.h`, `track_mult_fringe.h`, `track_srotation.h`, `track_xyshift.h`, `track_rf.h`, `particles/local_particle_custom_api.h`), source identical to git tag v0.104.1 | Apache-2.0 — full text in `licenses/Xsuite-Apache-2.0.txt`; the changes are listed in each file's header |
| ImpactX's exact elements for ring mode (ExactQuad, ExactMultipole, ExactCFbend, the linear DipEdge, the symplectic integrators), translated from C++ to Python and modified (the sector bend and the curvilinear kicks evaluated without cancellation) | `linac_gen/ring/kernels/curvilinear.py`; the symp2/4/6 integrators of `magnet_body` in `linac_gen/ring/kernels/cartesian.py` | ImpactX, development branch @ 472938df (2026-05-14, version 26.04+): `src/elements/ExactDrift.H`, `ExactQuad.H`, `ExactMultipole.H`, `ExactCFbend.H`, `DipEdge.H`, `src/particles/integrators/Integrators.H`; checked against the ImpactX 26.08 release | BSD-3-Clause-LBNL — full text in `licenses/ImpactX-BSD-3-Clause-LBNL.txt`, ImpactX's notices in `licenses/ImpactX-NOTICE.txt` |

No PyORBIT3 or Synergia2 code is included: ring mode's space charge
(`linac_gen/ring/sc/`) is written from the published methods — the integrated
Green's function (Qiang et al. 2006), the convolution on a doubled grid
(Hockney and Eastwood 1988), the pipe boundary fitted by least squares
(F. W. Jones, EPAC 2000) — and checked against analytic solutions and, for
benchmark numbers only, against other codes run separately.
The MAD8 / PTC form of the quadrupole wedge (`linac_gen/ring/kernels/nodes.py`)
is written from its formula.  Three of the Xsuite element maps above record
their own origin: the drift in a curved frame (`polar_drift`, from PTC's
`Sprotr` and MAD-NG's `curex_drift`) and the dipole and multipole fringe maps
(`dipole_fringe`, `mult_fringe`, from MAD-NG); they came to HELIX as part of
xtrack, under its Apache-2.0 licence.
