# Element models

Ring mode tracks with its own element maps ("kernels"): a ring needs maps
that depend on each particle's momentum and stay symplectic over 10⁴ turns,
which the linac's linear pushes are not.  It reads HELIX's own lattice — every
importer and the lattice editor apply — and never modifies it.

## Coordinates

Canonical: x, y [m] and px = Px/P0, py = Py/P0 (MAD-X's transverse
variables); ζ = s − β0 c t [m] (positive ahead of the reference) and
δ = (P − P0)/P0; with the energy deviation p_τ = (E − E0)/(P0 c) (MAD-X's
PT).

## The two models

**Cartesian multipoles** (`cartesian`, the default) — a bend's multipoles
expanded in Cartesian coordinates, as a MAD-X deck defines them:

| Element | Map |
|---|---|
| drift | exact (default), or expanded to second order in px and py (`drift_model = "expanded"`) |
| quadrupole | the thick chromatic map of the MAD-X Hamiltonian, its higher multipoles as kicks |
| combined-function bend | an exact drift in the curved frame with the dipole field as nested kicks ("rot-kick-rot", a fourth-order Forest–Ruth composition), the quadrupole and sextupole components as kicks with the curvature terms of the MAD-X Hamiltonian, 7 per step in Yoshida's sixth-order composition: fourth order overall |
| thin multipole, corrector, thin lens | kicks on the canonical momenta |
| RF cavity | thin; the energy gain V sin(lag − 2πf ζ/(β0c)) relative to the reference particle |

**Curvilinear multipoles** (`curvilinear`) — exact drifts, the exact
quadrupole and multipole Hamiltonians, and the combined-function bend with
*curvilinear* multipoles — the field expansion that satisfies Maxwell's
equations in the curved frame — integrated with symplectic schemes of order
2, 4 or 6, and linear pole faces.

The two differ in one place: the quadrupole field inside a bend.  To the
first order beyond the Cartesian expansion (the cubic terms of the
Hamiltonian), the curvilinear field is the Cartesian one plus a normal
sextupole of strength −h·k1 (h the curvature, k1 the bend's gradient); higher
orders differ further, by less.  On the simple Fermilab Booster this moves
the chromaticity by about −1.0 horizontally and −0.17 vertically; adding that
sextupole to Cartesian bends reproduces the curvilinear value.  The linear
optics are the same in both.

## Pole faces

| Model | What it does | Tested against |
|---|---|---|
| exact (default for the Cartesian model) | rotation into the pole face, the hard-edge dipole fringe (also at a face angle of zero), the quadrupole field of the wedge between face and sector boundary as the kick of H = −θ k1 (x³/3 − x y²) (MAD8's form), the dipole wedge | MAD-X's `TWISS` and `PTC_TWISS` (`model=2, method=6, nst=10, exact=true`, fringe off) |
| full | as exact, with the wedge's quadrupole kick Δpx = θ k1 (x² − y²/2), Δpy = −θ k1 x y (first order in θ), and the quadrupole's own hard-edge fringe | reference code A's full pole faces |
| linear | thin kicks h tan(e)·x, −h tan(e − ψ)·y (with the fringe integral and gap) | reference code A's linear pole faces; the curvilinear model's own, against reference code B |

θ is the face angle e.  The exact pole-face rotation and the hard-edge
fringe maps are those of Forest's treatment of exact maps; the fringe
correction ψ of the linear pole face is Brown's.  The reference codes are
those of the [ring benchmarks](../12_validation/06_ring_benchmarks.md).  On the simple Fermilab Booster (pole faces of half the bend
angle) the choice moves the horizontal chromaticity from −8.65 (exact, = MAD-X's
`TWISS` and `PTC_TWISS`) to −8.24 (full) and −7.01 (linear): the second-order
terms of a rotated pole face act, on the dispersive orbit, like a chromatic
quadrupole.

## Integration

Bends are integrated in steps: automatic by default (enough that the
focusing phase and the bend angle per step stay below 0.05 rad and 10 mrad —
the tunes are converged to 10⁻⁸), or set in the configuration
(`bend_steps`).  The Cartesian model's integrator is the 7-kick Yoshida
composition (`yoshida4`, the default), the TEAPOT thin-lens splitting,
uniform kicks, or the symplectic schemes symp2, symp4 and symp6; the
curvilinear model takes symp2, symp4 or symp6 (Yoshida's compositions of
order 2, 4 and 6).

Three formulas are evaluated in a rearranged but identical form, because
their textbook form loses digits: the exact sector bend of the curvilinear
model (x = −ρ + (ρ + x) cos φ + ρ(…), a difference of terms of order the
bending radius, about 10⁻¹⁴ m of rounding per step), the curvilinear
multipole kicks (log(1 + x/ρ) and its companions), and the Cartesian model's
expanded bend map (`bend_model = "expanded"`) where its horizontal focusing
K = h k0 + k1 nearly vanishes (a field index near 1; at 1, K is a rounding
residue): (cos √K L − 1)/K and its two companions come from their series in
K L² when |K| L² < 2×10⁻³, L the length of a step.  Against a 40-digit
evaluation of the same maps ring mode is then exact to rounding.  For the same reason the
Cartesian model integrates bends with the "rot-kick-rot" drift rather than
the exact curved-bend formula.

## Strengths and the reference

A quadrupole's gradient G becomes k1 = G/Bρ with the rigidity HELIX itself
uses for the lattice — the beam the deck's magnets were set for (its `BEAM`;
for a deck that states no beam, the importer's default, and for a TraceWin
deck, which gives fields, the run's beam) with HELIX's constants — so a
deck's normalized strengths come back exactly, whatever constants the ring
reference uses and, for a deck that states no beam, at whatever energy the
ring runs.  Bends are geometric
(h = θ/L, k1 = −N/ρ² from the field index N, k2 as imported); a field error
B(1 + δB) scales all of a bend's fields, and a bend with no field has no
pole-face kicks.  At another energy than the deck's (`design_energy_mev`),
the magnets follow: the same normalized strengths.  The rigidity carries the
beam's charge, so an H⁻ deck's strengths are those an H⁻ beam feels, and its
cavities give it energy with its charge (−V), as in linac tracking.

The **constants** of the reference particle: HELIX's (CODATA 2018, also
MAD-X 5.09's), CODATA 2010 (MAD-X up to 5.08), or CODATA 2022 (the masses
formed from their values in kilograms, m/(e·10⁶/c²)).

A cavity with a harmonic number (MAD-X `HARMON`) runs at h times the ring's
revolution frequency; one without keeps its own frequency.  HELIX's RF phase
is the cosine phase: ring mode's lag is the phase + 90°.

## Misalignments

dx, dy and a roll of a straight element are applied as a shift and a
rotation of the frame around it, as in linac tracking; dx and dy of a bend
the same way.  A rolled bend is refused for now.  dz, pitch and yaw are not
modelled (as in linac tracking) and are reported.

## Not modelled yet

Solenoids, field maps, RFQ cells, NCELLS cavities, electrostatic steerers
and matrix elements coupled to the longitudinal plane are refused with a
message naming the element.  A stripping foil is a marker where the foil
acts ([Injection](07_injection.md)); bump magnets of an injection are exact
dipoles in a straight frame.  Lattice
commands (matching directives, SET_* cards), space-charge compensation cards
and grid directives have no map in a ring and are skipped, with a note.

## Backends

Each map is written once and runs as plain Python, compiled for a multi-core
CPU (numba, the optional `[ring]` extra) and as a CUDA kernel — the same
source.  The compiled CPU kernels give the same result as the plain Python
bit for bit, whatever the number of threads.  Without numba, ring mode runs
the plain Python (slowly).

A particle for which a map is not defined — a transverse momentum beyond the
total momentum, where a square root would be of a negative number — is lost
as *invalid* at that element, keeping its coordinates at the element's
entrance; every backend loses it at the same element.

## References

* H. Yoshida, "Construction of higher order symplectic integrators",
  Phys. Lett. A **150**, 262 (1990) — the compositions of order 2, 4 and 6.
* E. Forest and R. D. Ruth, "Fourth-order symplectic integration",
  Physica D **43**, 105 (1990) — the composition inside the bend's drift.
* L. Schachinger and R. Talman, "TEAPOT: a thin-element accelerator program
  for optics and tracking", Part. Accel. **22**, 35 (1987) — the thin-lens
  splitting.
* É. Forest, *Beam Dynamics: A New Attitude and Framework* (Harwood
  Academic, 1998) — exact maps: the exact drift and sector bend, the pole-face
  rotation, the hard-edge fringe.
* K. L. Brown, "A first- and second-order matrix theory for the design of
  beam transport systems and charged particle spectrometers", SLAC-75,
  Rev. 4 (1982) — the fringe-field correction of a pole face.
