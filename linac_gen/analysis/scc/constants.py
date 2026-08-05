"""Physical constants, species, gas properties and cross sections.

PROVENANCE — read this before quoting any number from this file.

This docstring used to cite Tabata & Shirai, Rudd and Tawara as the source of the
tabulated cross-section MAGNITUDES. That was wrong. Published proton-impact
ionisation peaks near 1e-16 cm^2; the old 5 keV entry here was 1.55e-15 cm^2. The
magnitudes are roughly an order of magnitude above bare beam-impact values, and
they are kept that way deliberately, because they are what makes this code
reproduce the one benchmark it is tied to.

Every numeric input is therefore tagged with one of four provenance labels, and
`tests/physics/test_cross_sections.py` fails if any of them is missing:

  [SOURCED]                from a named external reference
  [DERIVED]                computed from SOURCED inputs, no free parameters
  [ASSUMED]                a modelling choice, not measured
  [INHERITED-CALIBRATION]  tuned so the code reproduces a specific published
                           result; NOT a literature value and not to be cited
                           as one

Genuinely sourced:
  * Rudd, Kim, Madison & Gay, Rev. Mod. Phys. 64, 441 (1992) — the reciprocal-sum
    functional form used for ionisation, and its Bethe ln(E)/E asymptote.
  * NIST Atomic Spectra Database — first ionisation potentials, which set where
    each gas's ionisation cross-section peaks.
  * ICRU Report 37 — mean excitation energies, used inside the Bethe logarithm
    (and nowhere else: they are inner-shell-dominated stopping-power quantities,
    so using them to set the peak would put argon's maximum at 690 keV).
  * Oppenheimer-Brinkman-Kramers / Thomas — the steep electron-capture asymptote.
  * CODATA 2018 / AME2020 — nuclide masses.

Inherited calibration:
  * The absolute ionisation and stripping magnitudes, fixed at 45 keV
    H-equivalent to reproduce Valerio-Lizarraga CERN-THESIS-2015-121 Fig. 3.1
    (H2) and Table 3.1 (N2, Kr). See docs/CALIBRATION.md.

Cross sections are keyed on H-EQUIVALENT kinetic energy: the energy a proton
would have at the same velocity. `h_equivalent_energy_keV` performs the mapping,
with the proton as reference (it was previously H-, a 0.12 % error that also let
a 200 keV H+ beam cross silently into the extrapolation branch).

All values SI unless a name says otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log


# ----- fundamental SI constants -----
C_LIGHT = 2.99792458e8         # m/s
Q_E = 1.602176634e-19          # C
M_E = 9.1093837015e-31         # kg  (electron)
M_U = 1.66053906660e-27        # kg  (atomic mass unit)
EPS0 = 8.8541878128e-12        # F/m
MU0 = 1.25663706212e-6         # N/A^2
K_BOLTZ = 1.380649e-23         # J/K
H_PLANCK = 6.62607015e-34      # J*s

# pressure conversion
MBAR_TO_PA = 100.0             # 1 mbar = 100 Pa

# ----- nuclide rest masses (CODATA 2018, in unified atomic mass units) -----
# These are NUCLIDE masses, not standard atomic weights. The distinction matters:
# the standard atomic weight of hydrogen is 1.00794 u, which is (a) isotope
# averaged, so it carries ~0.0115 % deuterium, and (b) already includes one bound
# electron. Building H- as `1.00794 u + m_e` therefore double-counts an electron
# and mixes in deuterium -- it was +114 ppm heavy. D- was built with only ONE
# extra electron, making it numerically the neutral D atom, -272 ppm light.
#
# Every ion mass below is assembled explicitly from a nuclide plus or minus the
# right number of electrons. Electron binding energies (13.6 eV, 0.75 eV) are
# ~1e-8 of the rest mass and are neglected.
M_PROTON_U = 1.007276466621     # CODATA 2018 proton mass
M_DEUTERON_U = 2.013553212745   # CODATA 2018 deuteron mass
M_HELIUM4_U = 4.002603254       # He-4 neutral ATOM (AME2020)
M_E_U = M_E / M_U               # electron mass in u (5.48579909e-4)


# ----- beam particle: H- -----
@dataclass(frozen=True)
class Species:
    """Ion species descriptor (mass, charge, name).

    `can_strip` indicates whether the species can lose an electron via
    collisions with the residual gas (H- -> H0 + e). Positive ions can't
    strip in LEBT-relevant collisions; this flips off the stripping loss.
    """
    name: str
    mass: float          # kg
    charge: float        # C (signed)
    z_symbol: str        # short label
    can_strip: bool = False

    @property
    def rest_energy(self) -> float:
        """mc^2 in joules."""
        return self.mass * C_LIGHT * C_LIGHT

    @property
    def rest_energy_eV(self) -> float:
        return self.rest_energy / Q_E


HMINUS = Species(
    name="H-",
    mass=(M_PROTON_U + 2.0 * M_E_U) * M_U,     # proton + 2 electrons
    charge=-Q_E,
    z_symbol="H⁻",
    can_strip=True,
)

PROTON = Species(
    name="H+",
    mass=M_PROTON_U * M_U,                     # bare proton
    charge=+Q_E,
    z_symbol="H⁺",
    can_strip=False,
)

DEUTERON = Species(
    name="D+",
    mass=M_DEUTERON_U * M_U,                   # bare deuteron
    charge=+Q_E,
    z_symbol="D⁺",
    can_strip=False,
)

DMINUS = Species(
    name="D-",
    mass=(M_DEUTERON_U + 2.0 * M_E_U) * M_U,   # deuteron + 2 electrons
    charge=-Q_E,
    z_symbol="D⁻",
    can_strip=True,
)

# He+ is the neutral He-4 ATOM minus one electron. This one was already correct;
# it is written out explicitly so it does not get "fixed" alongside H-/D-.
HE_PLUS = Species(
    name="He+",
    mass=(M_HELIUM4_U - M_E_U) * M_U,
    charge=+Q_E,
    z_symbol="He⁺",
    can_strip=False,
)


# registry for the GUI drop-down
BEAM_SPECIES: dict[str, Species] = {
    "H-": HMINUS,
    "H+": PROTON,
    "D-": DMINUS,
    "D+": DEUTERON,
    "He+": HE_PLUS,
}


# ----- residual gas library -----
#
# Cross-section tables are stored as (E_keV → σ_m²) anchor points keyed on
# *H-equivalent kinetic energy* (i.e. if the projectile is at velocity v,
# we look up σ at the kinetic energy a proton at the same v would have).
# This is exact in the Born regime for any singly-charged projectile.
#
# Three processes are tracked per gas, and they produce DIFFERENT things -- the
# detector model depends on the distinction:
#   "ion"   : ionisation of the gas       X + beam → X⁺ + e⁻    one ion AND one e⁻
#   "strip" : projectile electron loss    H⁻ + X   → H⁰ + X + e⁻  one free e⁻ only
#   "cap"   : projectile electron capture H⁺ + X   → H⁰ + X⁺      one ION only
#
# "ion" is the analytic model built below. "strip" and "cap" remain tabulated;
# their anchors are unchanged from the original tables, whose SHAPE is correct
# (see the discussion above `CrossSectionModel`). As with the ionisation
# magnitudes, treat the stripping/capture magnitudes as an inherited calibration
# rather than as digitised literature values -- the sources named in the module
# docstring supply the functional forms, not these numbers.

@dataclass(frozen=True)
class Gas:
    name: str
    symbol: str
    molecular_mass: float                       # amu (neutral)
    I_mean: float               # ICRU 37 mean excitation energy (eV)  [SOURCED]
    first_ionisation_eV: float  # NIST ASD first ionisation potential  [SOURCED]
    color: str
    dominant_ion_mass: float = 0.0              # amu of the trapped ion species
                                                  # (0 → use molecular_mass)
    # Cross-section evaluators, filled in below. Ionisation is the analytic Rudd
    # model; stripping and capture stay tabulated. All three are callables taking
    # H-equivalent keV and returning m^2.
    sigma_ion: object = None
    sigma_strip: object = None
    sigma_cap: object = None

    def ion_mass_kg(self) -> float:
        m = self.dominant_ion_mass or self.molecular_mass
        return m * M_U


# ---------------------------------------------------------------------------
# Cross-section evaluation
#
# Two kinds of channel, and they are deliberately NOT forced through one formula.
#
#   IONISATION is analytic (`CrossSectionModel`). Its tabulated anchors had an
#   INVERTED energy dependence: every *_ION table peaked at its first or second
#   anchor (5 or 10 keV) and fell monotonically, whereas proton-impact ionisation
#   rises to a maximum near 50-150 keV. sigma(200)/sigma(5) was 0.19-0.30 for six
#   chemically dissimilar targets -- the signature of one curve rescaled per gas,
#   not six measurements.
#
#   STRIPPING and CAPTURE stay tabulated. Their falloff over 5-200 keV is correct
#   physics: H- detachment has a 0.754 eV binding energy so its peak sits near
#   3.4 keV, BELOW the tabulated range, and capture falls steeply by the
#   Oppenheimer-Brinkman-Kramers argument. Re-fitting a correct shape would have
#   been change for its own sake.
#
# What both share is the high-energy extrapolation rule. A single Bethe ln(E)/E
# tail used to be applied to all three channels, so sigma(1 MeV) was exactly
# 0.23 x sigma(200 keV) for every table -- roughly 3000x too high for capture,
# whose own tabulated slope is E^-4.3 to E^-5.9. Tabulated channels now continue
# with their OWN measured last-decade slope.
# ---------------------------------------------------------------------------

M_ELECTRON_OVER_M_PROTON = (M_E / M_U) / M_PROTON_U


def _bethe_log(E_keV: float, I_eV: float) -> float:
    """ln(1 + 4 T_e / I) with T_e the energy of an electron at the projectile speed.

    `I_eV` is the target's ICRU mean excitation energy. Each gas carries its own
    (`Gas.I_mean`) -- previously 19.0 eV, H2's value, was hardcoded for all six
    while the correct per-gas numbers sat in the dataclass unread.
    """
    T_e_eV = M_ELECTRON_OVER_M_PROTON * E_keV * 1e3
    return log(1.0 + 4.0 * T_e_eV / I_eV)


@dataclass(frozen=True)
class CrossSectionModel:
    """Rudd reciprocal-sum form for an ionisation cross-section.

        sigma(E) = S / [ u^-D + u * Lambda(E0)/Lambda(E) ],   u = E / E0

    Asymptotics are exactly what is required, with no branches:

        E << E0 :  sigma -> S * u^D          rises from zero
        E ~  E0 :  single smooth maximum
        E >> E0 :  sigma -> S * (E0/E) * Lambda(E)/Lambda(E0)   Bethe tail

    Because the low-energy limit is built in, the flat-hold-below-the-first-anchor
    branch disappears structurally, and with it the bug where that branch's
    threshold was in H-equivalent keV while the GUI's floor was in species keV.

    Provenance of each field is tagged; see docs/CALIBRATION.md.
        S_m2    INHERITED-CALIBRATION  fixed by sigma(45 keV) = the frozen anchor
        E0_keV  DERIVED                solved so argmax sigma = E_peak_keV
        D       ASSUMED                low-energy rise exponent, one global value
        I_mean_eV SOURCED              ICRU 37
    """
    S_m2: float
    E0_keV: float
    D: float
    I_mean_eV: float
    E_peak_keV: float = 0.0     # informational: where the maximum sits

    def __call__(self, E_keV: float) -> float:
        # The only guard needed is against E == 0, where u**-D divides by zero.
        # Keep it far below any physically meaningful energy (1e-12 keV = 1 neV)
        # so it never truncates the low-energy asymptote itself: a larger floor
        # silently flattens sigma(E) below it, which is the very behaviour the
        # analytic form exists to remove.
        E = max(E_keV, 1e-12)
        u = E / self.E0_keV
        denom = (u ** (-self.D)
                 + u * _bethe_log(self.E0_keV, self.I_mean_eV)
                 / _bethe_log(E, self.I_mean_eV))
        return self.S_m2 / denom


@dataclass(frozen=True)
class TabulatedCrossSection:
    """(E_keV, sigma_m2) anchors: flat below, log-log between, power law above.

    `high_E_exponent` is the channel's OWN measured log-log slope over its last
    decade, so the extrapolation continues the data instead of imposing a
    universal Bethe tail on a channel that does not obey one.
    """
    table: tuple
    high_E_exponent: float
    def __call__(self, E_keV: float) -> float:
        table = self.table
        if not table:
            return 0.0
        E = max(E_keV, 1e-6)
        Es = [pt[0] for pt in table]
        Ss = [pt[1] for pt in table]
        if E <= Es[0]:
            # Conservative hold. Audible via CrossSectionRangeWarning; the GUI's
            # species-aware energy floor keeps it out of reach interactively.
            return Ss[0]
        if E >= Es[-1]:
            return Ss[-1] * (Es[-1] / E) ** self.high_E_exponent
        for i in range(len(Es) - 1):
            if Es[i] <= E <= Es[i + 1]:
                x = log(E / Es[i]) / log(Es[i + 1] / Es[i])
                return Ss[i] * (Ss[i + 1] / Ss[i]) ** x
        return Ss[-1]


def _last_decade_exponent(table: tuple) -> float:
    """Log-log slope n of the final two anchors, so sigma ~ E^-n above the table."""
    (e0, s0), (e1, s1) = table[-2], table[-1]
    return -log(s1 / s0) / log(e1 / e0)


def _shape_slope_numerator(E_keV: float, E0_keV: float, D: float,
                           I_eV: float) -> float:
    """A function whose sign matches d(sigma)/dE, for locating the peak exactly.

    Writing the model denominator as h(E) = A + B with

        A = u^-D,   B = u * Lambda(E0) / Lambda(E),   u = E / E0

    sigma = S/h is maximised where h is minimised, i.e. where E*h'(E) = 0:

        E h'(E) = -D*A + B * (1 - x / ((1+x) ln(1+x))),   x = 4 T_e(E) / I

    This returns that expression. It goes from -infinity at small E to positive at
    large E with a single sign change, so bisecting it locates the peak to FULL
    machine precision.

    Why not just search for the maximum directly? Because near a maximum the
    function is quadratically flat, so comparing values only locates it to about
    sqrt(machine epsilon) ~ 1e-8. A ternary search was used here first, and the
    cross-language parity vectors immediately caught it: Python and C++ agreed on
    sigma only to ~5e-9, purely from the two implementations landing on slightly
    different E0. Root-finding on the derivative removes that entirely -- the
    deterministic path is now exact in both languages.
    """
    u = max(E_keV, 1e-300) / E0_keV
    x = 4.0 * M_ELECTRON_OVER_M_PROTON * E_keV * 1e3 / I_eV
    lam = log(1.0 + x)
    A = u ** (-D)
    B = u * _bethe_log(E0_keV, I_eV) / lam
    return -D * A + B * (1.0 - x / ((1.0 + x) * lam))


def _argmax_energy(E0_keV: float, D: float, I_eV: float,
                   lo_keV: float, hi_keV: float) -> float:
    """Energy at which the model peaks, by bisection on the exact slope.

    Bounds are deliberately generous: a too-narrow bracket returns the bracket
    edge rather than the maximum, which silently poisons the E0 solve downstream
    (it produced three chemically different gases with identical fitted E0).
    """
    lo, hi = lo_keV, hi_keV
    for _ in range(200):
        mid = (lo * hi) ** 0.5          # geometric midpoint: the scale spans decades
        if _shape_slope_numerator(mid, E0_keV, D, I_eV) < 0.0:
            lo = mid
        else:
            hi = mid
    return (lo * hi) ** 0.5


def _solve_E0_for_peak(E_peak_keV: float, D: float, I_eV: float) -> float:
    """Bisect on E0 so that the model's maximum lands exactly at E_peak_keV.

    argmax is monotonically increasing in E0. Pure-Python bisection with no numpy
    or scipy, so the C++ port runs the identical algorithm and agrees to
    round-off rather than re-deriving the constants by hand.
    """
    lo, hi = 1e-6 * E_peak_keV, 1e3 * E_peak_keV
    if _argmax_energy(lo, D, I_eV, 1e-9 * E_peak_keV, 1e6 * E_peak_keV) > E_peak_keV:
        raise ValueError(
            f"peak {E_peak_keV} keV is unreachable for I_mean={I_eV} eV, D={D}")
    for _ in range(200):
        mid = (lo * hi) ** 0.5
        got = _argmax_energy(mid, D, I_eV, 1e-9 * E_peak_keV, 1e6 * E_peak_keV)
        lo, hi = (mid, hi) if got < E_peak_keV else (lo, mid)
    return (lo * hi) ** 0.5


# Cross-section anchor points (units: keV → m²)
# Below ≈ 5 keV the Born approximation breaks down; values for E < 5 keV
# extrapolate the lowest tabulated point (flat).

# H₂  — Tabata & Shirai (2000) σ_ion (H+ on H2, eq.3 of their paper)
#       and σ_strip (H− on H2, table 5)
# ---------------------------------------------------------------------------
# IONISATION — analytic model, replacing the former *_ION anchor tables.
#
# The magnitudes below are the ONLY external tie this code has. They are the
# value each old table took at 45 keV H-equivalent, preserved exactly so that the
# Valerio-Lizarraga benchmark does not move. They are EFFECTIVE cross-sections,
# roughly an order of magnitude above published bare proton-impact sigma_ion.
# Read docs/CALIBRATION.md before touching them.
# ---------------------------------------------------------------------------

# [INHERITED-CALIBRATION] sigma_ion at 45 keV H-equivalent, m^2.
A_ION_45KEV = {
    "H2": 9.047862e-20, "He": 7.787406e-20, "N2": 1.934971e-19,
    "Ar": 2.513040e-19, "Kr": 3.149894e-19, "Xe": 3.987583e-19,
}

# [ASSUMED] Low-energy rise exponent, one global value for all gases. Chosen so
# sigma(5 keV)/sigma(peak) ~ 0.32 for H2, close to the ~0.36 that measured
# H+ + H2 ionisation shows. A shape fit to digitised Rudd data would upgrade this
# to SOURCED-SHAPE; until then it is an assumption and is labelled as one.
RISE_EXPONENT_ION = 0.70

# [SOURCED] Velocity-matching constant. The ionisation cross-section peaks when
# the projectile speed matches the bound electron's orbital speed, i.e. at
# E_peak = kappa * (M_p/m_e) * B with B the binding energy. kappa is fixed once,
# by the measured peak of H+ + H2 ionisation at 70 keV.
PEAK_VELOCITY_FACTOR = 2.4714
H2_STRIP = (
    (5.0,   2.0e-19),    # H− + H2 → H0 + H2 + e   (Tabata-Shirai T5)
    (10.0,  1.5e-19),
    (20.0,  1.05e-19),
    (30.0,  8.5e-20),
    (50.0,  6.5e-20),
    (100.0, 4.4e-20),
    (200.0, 2.6e-20),
)
H2_CAP = (
    (5.0,   2.0e-19),    # H+ + H2 → H0 + H2+      (Tabata-Shirai)
    (10.0,  1.4e-19),
    (20.0,  6.0e-20),
    (30.0,  2.4e-20),
    (50.0,  6.0e-21),
    (100.0, 3.0e-22),
    (200.0, 5.0e-24),
)

# He
HE_STRIP = (
    (5.0,   1.5e-19),
    (10.0,  1.1e-19),
    (20.0,  7.5e-20),
    (30.0,  5.5e-20),
    (50.0,  3.8e-20),
    (100.0, 2.3e-20),
    (200.0, 1.4e-20),
)
HE_CAP = (
    (5.0,   8.0e-20),
    (10.0,  4.0e-20),
    (20.0,  1.5e-20),
    (30.0,  6.0e-21),
    (50.0,  1.5e-21),
    (100.0, 1.0e-22),
    (200.0, 3.0e-24),
)

# N2  — Rudd 1985, Phelps 1990
N2_STRIP = (
    (5.0,   3.5e-19),
    (10.0,  2.8e-19),
    (20.0,  2.0e-19),
    (30.0,  1.55e-19),
    (50.0,  1.10e-19),
    (100.0, 7.0e-20),
    (200.0, 4.5e-20),
)
N2_CAP = (
    (5.0,   1.5e-19),
    (10.0,  8.0e-20),
    (20.0,  3.0e-20),
    (30.0,  1.2e-20),
    (50.0,  3.0e-21),
    (100.0, 2.0e-22),
    (200.0, 1.0e-23),
)

# Ar
AR_STRIP = (
    (5.0,   3.8e-19),
    (10.0,  3.0e-19),
    (20.0,  2.2e-19),
    (30.0,  1.7e-19),
    (50.0,  1.20e-19),
    (100.0, 7.5e-20),
    (200.0, 4.8e-20),
)
AR_CAP = (
    (5.0,   2.5e-19),
    (10.0,  1.4e-19),
    (20.0,  5.0e-20),
    (30.0,  2.0e-20),
    (50.0,  4.5e-21),
    (100.0, 3.0e-22),
    (200.0, 1.5e-23),
)

# Kr — Tawara NIFS-DATA-79
KR_STRIP = (
    (5.0,   4.5e-19),
    (10.0,  3.5e-19),
    (20.0,  2.6e-19),
    (30.0,  2.0e-19),
    (50.0,  1.40e-19),
    (100.0, 8.5e-20),
    (200.0, 5.5e-20),
)
KR_CAP = (
    (5.0,   3.0e-19),
    (10.0,  1.7e-19),
    (20.0,  6.0e-20),
    (30.0,  2.5e-20),
    (50.0,  6.0e-21),
    (100.0, 5.0e-22),
    (200.0, 2.0e-23),
)

# Xe
XE_STRIP = (
    (5.0,   5.0e-19),
    (10.0,  4.0e-19),
    (20.0,  3.0e-19),
    (30.0,  2.3e-19),
    (50.0,  1.60e-19),
    (100.0, 1.00e-19),
    (200.0, 6.5e-20),
)
XE_CAP = (
    (5.0,   3.5e-19),
    (10.0,  2.0e-19),
    (20.0,  7.5e-20),
    (30.0,  3.0e-20),
    (50.0,  8.0e-21),
    (100.0, 6.0e-22),
    (200.0, 2.5e-23),
)


# ---------------------------------------------------------------------------
# Library assembly
#
# Ionisation models are BUILT here rather than pasted in as fitted constants:
# E0 is solved from the sourced peak position and S from the frozen 45 keV
# anchor, so there is no opportunity for a hand-copied number to drift out of
# step with its inputs. Both solves are pure-Python bisection, so the C++ port
# can run the identical algorithm instead of re-deriving the constants by hand.
# ---------------------------------------------------------------------------

# Dominant ion species: H2+ (m=2.016), He+ (m≈4), N2+ (m=28), Ar+ (40),
# Kr+ (84), Xe+ (131). Dissociative ionisation channels (e.g. H+ from H2) are
# minor (~10-20%) below 100 keV and are folded into the molecular-ion mass for
# the mean radial-drift estimate.
_GAS_SPEC = (
    # key,  name,                 symbol, M_amu,    I_mean, first_ion_eV, colour,   ion_mass
    ("H2", "Molecular hydrogen",  "H₂",   2.01588,   19.2,  15.4259, "#7ac9ff", 2.016),
    ("He", "Helium",              "He",   4.002602,  41.8,  24.5874, "#ffda6a", 4.002),
    ("N2", "Molecular nitrogen",  "N₂",  28.0134,    82.0,  15.5810, "#6af7a7", 28.014),
    ("Ar", "Argon",               "Ar",  39.948,    188.0,  15.7596, "#ff9d6a", 39.948),
    ("Kr", "Krypton",             "Kr",  83.798,    352.0,  13.9996, "#c66af7", 83.798),
    ("Xe", "Xenon",               "Xe", 131.293,    482.0,  12.1298, "#ff6ac6", 131.293),
)

_STRIP_TABLES = {"H2": H2_STRIP, "He": HE_STRIP, "N2": N2_STRIP,
                 "Ar": AR_STRIP, "Kr": KR_STRIP, "Xe": XE_STRIP}
_CAP_TABLES = {"H2": H2_CAP, "He": HE_CAP, "N2": N2_CAP,
               "Ar": AR_CAP, "Kr": KR_CAP, "Xe": XE_CAP}


def build_ionisation_model(first_ionisation_eV: float, I_mean_eV: float,
                           sigma_45keV_m2: float,
                           D: float = RISE_EXPONENT_ION) -> CrossSectionModel:
    """Assemble the analytic ionisation model for one gas.

    E_peak comes from the sourced binding energy; E0 is solved so the model peaks
    there; S is solved so the model reproduces the frozen 45 keV anchor exactly.

    Note `I_mean` enters only the Bethe logarithm, which is where it physically
    belongs -- it is a stopping-power quantity dominated by inner shells (Ar 188 eV,
    Xe 482 eV) that a 45 keV proton cannot reach. Using it to set the peak instead
    would put argon's maximum at 690 keV.
    """
    E_peak = PEAK_VELOCITY_FACTOR * first_ionisation_eV * 1e-3 / M_ELECTRON_OVER_M_PROTON
    E0 = _solve_E0_for_peak(E_peak, D, I_mean_eV)
    unit = CrossSectionModel(1.0, E0, D, I_mean_eV, E_peak)
    return CrossSectionModel(sigma_45keV_m2 / unit(45.0), E0, D, I_mean_eV, E_peak)


GAS_LIBRARY: dict[str, Gas] = {}
for _key, _name, _sym, _mass, _imean, _bion, _col, _ionmass in _GAS_SPEC:
    GAS_LIBRARY[_key] = Gas(
        name=_name, symbol=_sym, molecular_mass=_mass,
        I_mean=_imean, first_ionisation_eV=_bion, color=_col,
        dominant_ion_mass=_ionmass,
        sigma_ion=build_ionisation_model(_bion, _imean, A_ION_45KEV[_key]),
        sigma_strip=TabulatedCrossSection(
            _STRIP_TABLES[_key], _last_decade_exponent(_STRIP_TABLES[_key])),
        sigma_cap=TabulatedCrossSection(
            _CAP_TABLES[_key], _last_decade_exponent(_CAP_TABLES[_key])),
    )
del _key, _name, _sym, _mass, _imean, _bion, _col, _ionmass


# Lowest tabulated / modelled energy, in H-equivalent keV. Below this the code
# holds the cross-section flat, which is a conservative guess, not data.
MODEL_VALIDITY_FLOOR_KEV = 5.0


class CrossSectionRangeWarning(UserWarning):
    """Raised when a cross-section is requested below the tabulated range.

    The flat-hold below the lowest anchor used to be silent. It is reachable from
    the GUI for the heavier species, because the energy floor there is in
    *species* keV while the tables are keyed in *H-equivalent* keV: a He+ beam at
    5-19.85 keV, or D+/D- at 5-9.99 keV, maps below 5 keV H-equivalent. The GUI
    floor is now species-aware (see `min_energy_keV`), so this warning should only
    fire on programmatic use.
    """


def h_equivalent_energy_keV(species: "Species", T_keV: float) -> float:
    """Kinetic energy of a PROTON travelling at the same velocity as `species`.

    The cross-section tables are keyed on this quantity. Non-relativistic at LEBT
    energies (gamma - 1 < 1e-4), so T_equiv = T * m_p / m_species.

    The reference is the proton, as the module docstring has always said. It was
    coded as H- -- a 0.12 % error, harmless in itself, but it also meant a 200 keV
    H+ beam (the GUI maximum) mapped to 200.24 keV and silently crossed into the
    high-energy extrapolation branch.
    """
    return T_keV * (M_PROTON_U * M_U / species.mass)


def min_energy_keV(species: "Species") -> float:
    """Lowest species kinetic energy whose H-equivalent stays in tabulated range."""
    return MODEL_VALIDITY_FLOOR_KEV * species.mass / (M_PROTON_U * M_U)


def sigma_at_H_equivalent(channel, E_keV_H_equivalent: float) -> float:
    """Evaluate a cross-section channel at an H-equivalent energy, no species map.

    This is the reference-free accessor. Use it when the quantity of interest is a
    property of the channel itself -- for instance the calibration anchors in
    docs/CALIBRATION.md, which must not move when the species mapping changes.

    Accepts a `CrossSectionModel`, a `TabulatedCrossSection`, or a bare
    (E_keV, sigma_m2) tuple.
    """
    if isinstance(channel, tuple):
        channel = TabulatedCrossSection(channel, _last_decade_exponent(channel))
    return channel(E_keV_H_equivalent)


def sigma_lookup(channel, species: "Species", T_keV: float,
                 warn_out_of_range: bool = True) -> float:
    """Evaluate a cross-section channel at the projectile's velocity."""
    if channel is None:
        return 0.0
    T_equiv = h_equivalent_energy_keV(species, T_keV)
    # Only the tabulated channels have a hard low-energy floor; the analytic
    # ionisation model is valid down to zero by construction.
    if (warn_out_of_range and isinstance(channel, TabulatedCrossSection)
            and T_equiv < MODEL_VALIDITY_FLOOR_KEV):
        import warnings
        warnings.warn(
            f"{species.name} at {T_keV:.3g} keV is {T_equiv:.3g} keV H-equivalent, "
            f"below the {MODEL_VALIDITY_FLOOR_KEV} keV floor of the tabulated "
            f"cross-section; the value is held flat and is not data "
            f"(min energy for {species.name} is {min_energy_keV(species):.3g} keV)",
            CrossSectionRangeWarning, stacklevel=3)
    return sigma_at_H_equivalent(channel, T_equiv)


# Convenience getters --------------------------------------------------------
def sigma_ionization(gas_key: str, species: "Species", T_keV: float) -> float:
    return sigma_lookup(GAS_LIBRARY[gas_key].sigma_ion, species, T_keV)


def sigma_stripping(gas_key: str, species: "Species", T_keV: float) -> float:
    if not species.can_strip:
        return 0.0
    return sigma_lookup(GAS_LIBRARY[gas_key].sigma_strip, species, T_keV)


def sigma_capture(gas_key: str, species: "Species", T_keV: float) -> float:
    """Electron-capture cross-section H+/D+ + X → H0/D0 + X+ . Zero for negative
    species (capture would form H−, much smaller σ; ignored)."""
    if species.charge < 0:
        return 0.0
    return sigma_lookup(GAS_LIBRARY[gas_key].sigma_cap, species, T_keV)


# Backward-compatible aliases (keep old API working, now velocity-correct)
def sigma_at_velocity(sigma_ref_unused: float, species: "Species",
                       T_keV: float, I_eV_unused: float = 0.0,
                       E_ref_keV_unused: float = 30.0) -> float:
    """Legacy shim: ignored ref values, uses tabulated data instead.

    Old call sites passed a reference σ at 30 keV plus the gas mean
    excitation energy. We now interpolate the proper Tabata/Rudd table.
    """
    # The caller almost always passed gas.sigma_ioniz_30keV; we cannot
    # tell which gas without more context, so fall back to a Bethe scaling
    # of the ref value. Code paths that need real numbers should switch to
    # sigma_ionization() / sigma_stripping() instead.
    if sigma_ref_unused == 0.0:
        return 0.0
    T_equiv_keV = T_keV * (HMINUS.mass / species.mass)
    E = max(T_equiv_keV, 1.0) * 1e3
    Er = E_ref_keV_unused * 1e3
    if I_eV_unused <= 0 or E <= I_eV_unused or Er <= I_eV_unused:
        return sigma_ref_unused
    num = log(4 * E / I_eV_unused) / E
    den = log(4 * Er / I_eV_unused) / Er
    if den <= 0 or num <= 0:
        return sigma_ref_unused
    return sigma_ref_unused * (num / den)


def sigma_at_energy(sigma_ref: float, E_keV: float, I_eV: float,
                    E_ref_keV: float = 30.0) -> float:
    """Legacy Bethe-log scaling. Kept for any external callers."""
    E = max(E_keV, 1.0) * 1e3
    Er = E_ref_keV * 1e3
    if I_eV <= 0 or E <= I_eV or Er <= I_eV:
        return sigma_ref
    num = log(4 * E / I_eV) / E
    den = log(4 * Er / I_eV) / Er
    if den <= 0 or num <= 0:
        return sigma_ref
    return sigma_ref * (num / den)


# Characteristic current I0 = 4 pi eps0 m c^3 / |q|  (same magnitude for H- and H+)
def characteristic_current(species: Species) -> float:
    return 4.0 * 3.141592653589793 * EPS0 * species.mass * C_LIGHT ** 3 / abs(species.charge)
