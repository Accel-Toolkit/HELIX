# ===========================================================================
# Simple FODO + bend cell — HELIX SciBmad import demo
#
# The same cell as examples/madx/fodo.madx as Julia source for the SciBmad
# package Beamlines.jl, written by the lattix translator.  Open it in HELIX
# via  File -> Open Lattice...  (*.jl / *.scibmad are in the file filter)
# or on the command line:
#     python -m linac_gen run examples/scibmad/fodo.jl --energy 800 --freq 352.21
# HELIX parses the restricted Julia subset that lattice files use (element
# constructors, @elements blocks, Beamline/Branch, simple arithmetic) in pure
# Python — Julia is not needed and nothing in the file is executed.  The
# import needs the optional 'lattix' package (not yet published: install lattix into this environment, or
# HELIX_LATTIX_ROOT pointing at a checkout).  Kn1 is normalised by the
# signed rigidity of the deck's reference particle (an 800 MeV proton here).
# ===========================================================================
# lattix 0.0.1 from madx
# Beamlines.jl 0.10 dialect: phi0 in radians (PhaseRef.Accelerating = crest), every cavity
# tracked with SaganCavity, energy jumps as dE_ref on the element that starts each section
using Beamlines

@elements begin
  lat_begin = Marker(species_ref = Species("proton"), E_ref = 1738272000.0)
  qf = Quadrupole(L = 0.3, Kn1 = 0.6)
  drift_0 = Drift(L = 0.7)
  b1 = SBend(L = 1.0, g_ref = 0.1, Kn0 = 0.1, e1 = 0.05, e2 = 0.05)
  drift_1 = Drift(L = 1.0)
  qd = Quadrupole(L = 0.3, Kn1 = -0.6)
  drift_2 = Drift(L = 1.0)
  drift_3 = Drift(L = 1.3)
  m1 = Marker()
end

lattice = Beamline([lat_begin, qf, drift_0, b1, drift_1, qd, drift_2, b1, drift_3, m1])

# track it:  using BeamTracking; bl = [lattice]
#   b = Bunch(v; species = bl[1].species_ref, p_over_q_ref = bl[1].p_over_q_ref)
#   for s in bl; track!(b, s); end
