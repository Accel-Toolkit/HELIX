# Appendix A — Physics references

Citations for every paper that informs HELIX's physics implementation.
Useful when you need to cite HELIX (cite the underlying physics
paper directly) or want to understand a particular module's
provenance.

## Space charge & PIC

* **Qiang, J.; Lim, R.; Ryne, R.D.** "Self-consistent
  three-dimensional space-charge effect calculation in a
  high-intensity ion linac".  *Phys. Rev. ST Accel. Beams* **9**,
  044204 (2006).  *Source of HELIX's IGF Green's function.*
  doi:10.1103/PhysRevSTAB.9.044204
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
