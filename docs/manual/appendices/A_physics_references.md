# Appendix A — Physics references

Citations for every paper that informs HELIX's physics implementation.
Useful when you need to cite HELIX (cite the underlying physics
paper directly) or want to understand a particular module's
provenance.

## Space charge & PIC

* **Qiang, J.; Lidia, S.; Ryne, R.D.; Limborg-Deprey, C.**
  "Three-dimensional quasistatic model for high brightness beam
  dynamics simulation".  *Phys. Rev. ST Accel. Beams* **9**, 044204
  (2006); erratum **10**, 129901 (2007).  *Source of HELIX's IGF
  Green's function.*  doi:10.1103/PhysRevSTAB.9.044204
* **Hockney, R.W.; Eastwood, J.W.** *Computer Simulation Using
  Particles*, Adam Hilger (1988).  Foundational text on PIC.
* **Wangler, T.P.** *RF Linear Accelerators*, 2nd ed., Wiley-VCH
  (2008).  Standard reference for envelope SC models.
* **Lapostolle, P.M.** "Possible emittance increase through
  filamentation due to space charge in continuous beams".
  *IEEE Trans. Nucl. Sci.* NS-18, 1101 (1971).

## Continuous-beam envelope (Sacherer ODE)

* **Sacherer, F.J.** "RMS envelope equations with space charge".
  *IEEE Trans. Nucl. Sci.* NS-18, 1105 (1971).
* **Kapchinskij, I.M.; Vladimirskij, V.V.** "Limitations of
  proton beam current in a strong focusing linear accelerator
  associated with the beam space charge".  Proc. 2nd Int. Conf.
  on High-Energy Accelerators (1959).

## H⁻ stripping

* **Folsom, B.; Eshraqi, M.; Blaskovic-Kraljevic, N.;
  Gålnander, B.** "Stripping mechanisms and remediation for
  H⁻ beams".  *Phys. Rev. Accel. Beams* **24**, 074201 (2021).
  arXiv:2103.16195.  *Source of HELIX's magnetic-stripping rate
  formula.*
* **Stinson, G.M. et al.** (1969).  Original empirical fit
  for the A₁ / A₂ Lorentz-stripping constants (refit by Scherk
  1979, used by Folsom 2021).

## Eigenemittances

* **Balandin, V.; Decking, W.; Golubeva, N.** "On the calculation
  of generalized four-dimensional and six-dimensional
  eigen-emittances".  IPAC 2013 / arXiv:1305.1532.
  *Source of the Balandin trace invariants and eigenemittance
  cubic formula.*
* **IMPACT-X** code documentation — same eigenemittance convention
  used in HELIX.

## RFQ design

* **Crandall, K.R.; Stokes, R.H.; Wangler, T.P.**
  "RF quadrupole beam dynamics design studies".  *Linac 1979*.
  *Source of the Crandall 2-term potential expansion.*
* **Toutatis** code documentation — RFQ-design code shipped in
  `Tracewin_code/Toutatis_*.pdf` references.

## Field maps

* **TraceWin documentation (CEA Saclay)** — definitive reference
  for the FIELD_MAP `geom` 5-digit encoding and file-naming
  convention (see the FIELD_MAP and field-map-file sections of the
  TraceWin manual shipped with the code).

## Symplectic integrators

* **Yoshida, H.** "Construction of higher order symplectic
  integrators".  *Phys. Lett. A* **150**, 262 (1990).
* **Forest, E.** "Geometric integration for particle accelerators".
  *J. Phys. A: Math. Gen.* **39**, 5321 (2006).

## Ring mode

* **Yoshida, H.** (1990), above — the compositions of order 2, 4 and
  6 of both element models.
* **Forest, E.; Ruth, R.D.** "Fourth-order symplectic integration".
  *Physica D* **43**, 105 (1990).  The composition inside a bend's
  drift.
* **Schachinger, L.; Talman, R.** "TEAPOT: a thin-element
  accelerator program for optics and tracking".  *Part. Accel.*
  **22**, 35 (1987).  The thin-lens splitting.
* **Forest, E.** *Beam Dynamics: A New Attitude and Framework*.
  Harwood Academic (1998).  Exact maps: the exact drift and sector
  bend, the pole-face rotation, the hard-edge fringe.
* **Brown, K.L.** "A first- and second-order matrix theory for the
  design of beam transport systems and charged particle
  spectrometers".  SLAC-75, Rev. 4 (1982).  The fringe-field
  correction of a pole face.
* **Laskar, J.** "Frequency analysis for multi-dimensional systems.
  Global dynamics and diffusion".  *Physica D* **67**, 257 (1993).
  NAFF, the tunes of tracked particles.
* **Qiang, J.; Lidia, S.; Ryne, R.D.; Limborg-Deprey, C.** (2006), above —
  the integrated Green's function, here in two dimensions for ring mode's
  2.5-D space charge.
* **Hockney, R.W.; Eastwood, J.W.** (1988), above — the convolution on a
  doubled grid (open boundary).
* **Bassetti, M.; Erskine, G.A.** "Closed expression for the electrical
  field of a two-dimensional Gaussian charge".  CERN-ISR-TH/80-06 (1980).
  The reference field of the space-charge checks.
* **Jones, F.W.** "A method for incorporating image forces in
  multiparticle tracking with space charge".  Proc. EPAC 2000, p. 1381.
  The pipe boundary fitted to a rectangle or an ellipse.
* **Laslett, L.J.** "On intensity limitations imposed by transverse
  space-charge effects in circular particle accelerators".  BNL-7534,
  p. 324 (1963).  The incoherent tune shift.
* **Qiang, J.** *Phys. Rev. Accel. Beams* **28**, 114602 (2025),
  doi:10.1103/npx1-nclt.  The analytic 2.5-D kick of a Gaussian bunch
  (eqs. 31–32), a space-charge check.
* **Highland, V.L.** "Some practical remarks on multiple scattering".
  *Nucl. Instrum. Methods* **129**, 497 (1975); **Lynch, G.R.; Dahl,
  O.I.** "Approximations to multiple Coulomb scattering".  *Nucl.
  Instrum. Methods B* **58**, 6 (1991).  The stripping foil's
  scattering width θ0 (fitted to the central 98 % of the angles).
* **Gulley, M.S. et al.** "Measurement of H⁻, H⁰, and H⁺ yields
  produced by foil stripping of 800-MeV H⁻ ions".  *Phys. Rev. A*
  **53**, 3201 (1996).  Measured stripping yields, close to the two
  design values the two-step stripping's cross sections are calibrated
  to.

## Boris pusher

* **Boris, J.P.** "Relativistic plasma simulation — optimization
  of a hybrid code".  Proc. Conf. Numerical Simulation Plasmas
  (1970), 3-67.

## Halo

* **Wangler, T.P.; Crandall, K.R.; Mills, R.S.; Reiser, M.**
  "Relation between field energy and rms emittance in intense
  particle beams".  *IEEE Trans. Nucl. Sci.* NS-32, 2196 (1985).
  Background for kurtosis-style halo measures.  Note HELIX's
  recorded halo parameter is the kurtosis form
  h = ⟨x⁴⟩/⟨x²⟩² − 1 (Gaussian ⇒ 2, KV ⇒ 1) — see
  [Halo analysis](../09_diagnostics/03_halo.md).

## Adjacent codes for cross-reference

* **TraceWin** — Saclay, closed-source.  HELIX inherits its `.dat`
  format and matching language.
* **IMPACT-X** — Berkeley Lab, open-source C++.  Source of HELIX's
  Misalignment mixin pattern and eigenemittance convention.
* **PyORBIT** — open-source.  Same H⁻ stripping fit.
* **OPAL** — open-source.  Same Hockney+IGF PIC algorithm.

← [Surrogates → Training guide](../13_surrogates/05_training_guide.md) ·
[Continue to Appendix B →](B_keyword_cheatsheet.md)
