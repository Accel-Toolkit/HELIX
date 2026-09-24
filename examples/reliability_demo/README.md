# reliability_demo — the self-test lattice of the Reliability Study mode

`reliability_demo.dat` / `reliability_demo.lgproj` is a 2.8 m proton channel
(20 MeV, 162.5 MHz, 0 mA, 1 % duty): three cells of `ADJUST_STEERER` +
`THIN_STEERING`, `SOLENOID` (150 mm, 1.5 T), `GAP` (0.8–1.2 MV at −25…−20°),
`APERTURE` (8 mm) and `DIAG_POSITION`, then a fourth gap, a collimator and a
600 µg/cm² carbon stripper foil.  Every element type the reliability legs touch
is present: cavities and solenoids to fail, steerers with correction cards,
BPM markers, apertures for losses, a foil for the stripping leg.  The parser
auto-labels the elements `GAP_00n`, `SOL_00n`, `STEER_00n`, `BPM_00n`,
`APER_00n`, `FOIL_001`.

All four files are written by `make_reliability_demo.py`; edit the generator,
not the outputs.  `circuits.json` groups the gaps into two cryomodules and four
RF stations and names the sections and landmarks the engine reports on.

`truth.json` holds answers computed by the generator's own kinematics, with no
HELIX import, so the built-in self-test (`python -m linac_gen reliability
selftest`) compares the code against an independent reference:

| entry | value | how it is known |
|---|---|---|
| re-phased cavity-OFF deficits | `V cos φ_s` per gap (0.725, 0.927, 0.940, 1.128 MeV) | thin gap, T = 1 |
| frozen-phase cavity-OFF deficits | 0.543, 0.799, 0.905, 1.128 MeV | the slower reference reaches every downstream gap later by `360 L/(βλ)` per drift and gains `V cos(φ_s + Δφ)`; the chain is integrated in the generator |
| clock phase at every gap entrance | `clock_at_gap_entrance_deg` | the same drift sum on the design pass (the pins) |
| exit energy | 23.7196 MeV after the gaps; the envelope solver reports 1.047 keV less because it decelerates the reference by the foil's mean loss | kinematics + the MIP loss constant |
| solenoid OFF | ΔW = 0 exactly | a solenoid carries no RF |
| top single-cavity fault, k = 1 neighbours of `GAP_004` | `GAP_004`; `GAP_003`, `GAP_002` | deck design |
| foil | mean loss 1.047 keV, Highland θ_rms 0.626 mrad at 23.7 MeV | PDG constants |
| availability | 100/101, its square, 1 − (1/101)², 3a² − 2a³ | closed forms |
| orbit correction | planted solenoid offsets +0.20/−0.15/+0.10 mm; expect `converged`, residual < 1 % of the initial rms, 100 % transmission | one-to-one pairing on three BPMs |

Checked against HELIX when the deck was generated: zero parser warnings, the
envelope exit energy and the four re-phased deficits agree bit for bit, the
multiparticle clock at each gap entrance agrees to 1e-15, transmission is
100 % (2000 particles), and the corrector converges one-to-one on the planted
offsets.  The foil straggling is pinned to `gaussian` because 23.7 MeV protons
sit in the Vavilov regime, where the automatic dispatch would warn on every run.

Manual: `docs/manual/16_reliability/`.
