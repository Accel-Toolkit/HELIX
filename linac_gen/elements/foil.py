"""Stripper / scattering foil element.

A thin material foil that applies, per macroparticle:
  * Multiple Coulomb scattering (Highland 1991 formula) on the transverse
    angles xp, yp.
  * Mean ionisation energy loss (Bethe-Bloch tabulated at minimum-ionising
    rates per material) plus energy-loss straggling on dw.  The straggling
    *shape* is regime-dependent (``straggling`` parameter):

      - ``"landau"``   — Landau distribution, correct for thin absorbers
        (kappa = xi/T_max < 0.01, e.g. a 600 ug/cm2 stripper foil at
        800 MeV has kappa ~ 3e-5).  A Gaussian is badly wrong here: it
        assigns ~44 % of particles a net energy GAIN, unphysical for a
        passive absorber.
      - ``"gaussian"`` — Bohr Gaussian, correct for thick absorbers
        (kappa > 10) and kept for backward compatibility.
      - ``"auto"``     — (default) dispatch on kappa: < 0.01 -> Landau,
        > 10 -> Gaussian, intermediate -> Gaussian with a warning
        (Vavilov not implemented; see :meth:`resolved_straggling_model`).

Solver dispatch
---------------
*Multi-particle tracker*: ``apply_kick`` draws the stochastic kicks per
particle — same dispatch path as :class:`linac_gen.elements.steerer.
Steerer`.  The mean energy loss is carried by the particles' ``dw``
(the reference particle is NOT decelerated in MP mode).

*Envelope solver*: the MEAN map of the foil is the identity (the kicks
are zero-mean, so ``kick_matrix`` is the 6x6 identity), but the foil is
NOT invisible at second-moment level: a zero-mean random kick is a
diffusion process.  The envelope solver applies

    Sigma <- M Sigma M^T + D,   with M = I and
    D = diag(0, theta0^2, 0, theta0^2, 0, sigma_E^2)

(theta0 = Highland rms plane angle in mrad, sigma_E = Bohr straggling
width in MeV — see :meth:`envelope_diffusion`), and the Bethe-Bloch
mean loss is applied to the reference energy (the natural bookkeeping
for a centred-Sigma description; equivalent to the MP convention to
first order since <dE> / W ~ 1e-6 here).

*Matrix solver*: identity — a one-pass linear map carries no second
moments, so there is nothing for it to see.

Materials supported (areal-density basis, ug/cm2):
  C, Be, Al, Cu, Mo, W

Used in HELIX primarily for the BTL -> Booster charge-exchange stripper
in the PIP-II model. Round-trips through TraceWin .dat as a comment line
``; HELIX_FOIL <name> <material> <thickness_ug_cm2> [straggling]``.
"""
from __future__ import annotations

import functools
import math
import warnings

import numpy as np

from linac_gen.core.constants import M_ELECTRON
from linac_gen.core.particle import H_MINUS
from linac_gen.elements.base import ThinKickElement


# Material database. Sources:
#   X0 (g/cm²) — PDG 2024 Table 6.1 (radiation length for low-Z and
#     selected metals).
#   <dE/dx>_min (MeV·cm²/g) — PDG 2024 Table 35.1 minimum-ionising rates
#     for protons / heavy charged particles.  NOTE the model is the MIP
#     CONSTANT with no β-dependence: protons only reach the MIP plateau
#     near ~2 GeV, so below ~1 GeV the true dE/dx is HIGHER than this
#     table (grossly so at injection energies — tens of × at 5 MeV).
#     For a thin stripper foil (∼600 μg/cm²) the absolute loss is tiny
#     either way; treat the mean loss as a lower bound, not a <5 %
#     prediction.
#   I (eV) — PDG mean excitation energies (used by the Landau
#     most-probable-loss parametrization; the tabulated-MIP mean above
#     does not need it).
#   Z, A — standard atomic data.
_MATERIALS: dict[str, dict[str, float]] = {
    "C":  {"X0_g_cm2": 42.70, "dEdx_min_MeVcm2_g": 1.745, "Z":  6, "A":  12.011, "I_eV":  78.0},
    "Be": {"X0_g_cm2": 65.19, "dEdx_min_MeVcm2_g": 1.594, "Z":  4, "A":   9.0122, "I_eV":  63.7},
    "Al": {"X0_g_cm2": 24.01, "dEdx_min_MeVcm2_g": 1.615, "Z": 13, "A":  26.982, "I_eV": 166.0},
    "Cu": {"X0_g_cm2": 12.86, "dEdx_min_MeVcm2_g": 1.403, "Z": 29, "A":  63.546, "I_eV": 322.0},
    "Mo": {"X0_g_cm2":  9.80, "dEdx_min_MeVcm2_g": 1.274, "Z": 42, "A":  95.95,  "I_eV": 424.0},
    "W":  {"X0_g_cm2":  6.76, "dEdx_min_MeVcm2_g": 1.145, "Z": 74, "A": 183.84,  "I_eV": 727.0},
}

_STRAGGLING_MODES = ("auto", "landau", "gaussian")
_STRIP_MODELS = ("off", "two_step")
_DEDX_MODELS = ("mip", "bethe")

# Bethe constant K = 4π N_A r_e² m_e c² (MeV cm²/mol), PDG.
_K_BETHE = 0.307075
_N_AVOGADRO = 6.02214076e23

# Two-step charge-exchange model H⁻ → H⁰ → p on carbon: cross sections
# σ(-1→0) and σ(0→+1) in cm² per atom at 800 MeV, CALIBRATED (not
# measured here) so that the stripped fraction reproduces the two PIP-II
# design anchors — 99.956 % at 600 µg/cm² and 99.1 % at 380 µg/cm² (FDR
# §9.4.1.3) — and scaled to other energies as 1/β² relative to 800 MeV
# (the Bethe-like fall of the electron-loss cross sections with velocity;
# Gulley et al., PRA 53, 3201 (1996) measure 6.76e-19 / 2.64e-19 cm² at
# 800 MeV, close to the values solved from the anchors).  Other
# materials scale the carbon values by sqrt(Z/6), a convention, not
# data; treat any non-carbon stripping fraction as indicative.
_STRIP_ANCHORS = ((600.0, 0.99956), (380.0, 0.991))
_STRIP_BETA_800 = math.sqrt(1.0 - 1.0 / (1.0 + 800.0 / H_MINUS.mass) ** 2)   # β of an 800 MeV H⁻


def _solve_strip_cross_sections() -> tuple[float, float]:
    """(σ_-10, σ_01) in cm² per carbon atom reproducing _STRIP_ANCHORS
    exactly (2-D Newton on the two-step stripped fraction)."""
    a_c = _MATERIALS["C"]["A"]

    def f_p(s10, s01, ug):
        n = ug * 1e-6 / a_c * _N_AVOGADRO
        e10, e01 = math.exp(-s10 * n), math.exp(-s01 * n)
        h0 = s10 / (s10 - s01) * (e01 - e10)
        return 1.0 - e10 - h0

    s10, s01 = 6.76e-19, 2.64e-19
    for _ in range(60):
        r = [f_p(s10, s01, ug) - want for ug, want in _STRIP_ANCHORS]
        if max(abs(v) for v in r) < 1e-14:
            break
        h = 1e-24
        j = [[(f_p(s10 + h, s01, ug) - f_p(s10, s01, ug)) / h,
              (f_p(s10, s01 + h, ug) - f_p(s10, s01, ug)) / h]
             for ug, _w in _STRIP_ANCHORS]
        det = j[0][0] * j[1][1] - j[0][1] * j[1][0]
        d10 = (r[0] * j[1][1] - r[1] * j[0][1]) / det
        d01 = (j[0][0] * r[1] - j[1][0] * r[0]) / det
        s10, s01 = s10 - d10, s01 - d01
    return float(s10), float(s01)


_STRIP_SIGMA_800 = _solve_strip_cross_sections()

# Mode (most probable value) of scipy.stats.landau's standard pdf,
# located numerically (scipy 1.17: argmax of landau.pdf to 1e-5).
_LANDAU_MODE = -0.42931

# scipy.stats.landau is the UNIT-SCALE stable(alpha=1, beta=1) law, which
# is NARROWER than Landau's original phi(lambda) — the distribution the
# energy-loss theory (and the xi / dE_mp parametrization) is written in —
# by a factor pi/2.  Verified numerically against the defining integral
# phi(lambda) = (1/pi) int_0^inf exp(-t ln t - lambda t) sin(pi t) dt:
#   * FWHM:  scipy 2.55835 vs phi 4.01856   (ratio 0.63672 = 2/pi)
#   * tail:  scipy sf(x)*x -> 0.63666 = 2/pi vs phi's sf(x)*x -> 1
#   * peak:  scipy pdf(mode) 0.28377 = (pi/2) * phi's 0.18066
# (pure horizontal compression).  Location conventions cancel by
# anchoring mode to mode:
#   (lambda_phi - mode_phi) = (pi/2) * (lambda_scipy - mode_scipy),
# so a physical Landau of width parameter xi is sampled from scipy's
# variable with the effective scale  xi_eff = (pi/2) * xi  — this gives
# the canonical core FWHM = 4.02*xi and single-collision (Rutherford)
# tail P(loss > E) ~ xi/E, both pinned by tests.
_LANDAU_SCIPY_TO_PHI = math.pi / 2.0

# Kappa-regime boundaries for the "auto" straggling dispatch
# (kappa = xi / T_max, Vavilov's thickness parameter).
_KAPPA_LANDAU_MAX = 0.01     # below: Landau regime
_KAPPA_GAUSSIAN_MIN = 10.0   # above: Gaussian (Bohr) regime


def supported_materials() -> list[str]:
    """Return the list of material keys the Foil element accepts."""
    return list(_MATERIALS.keys())


@functools.lru_cache(maxsize=128)
def _landau_truncated_mean(lam_max: float) -> float:
    """Mean of scipy's standard Landau conditioned on lambda <= lam_max.

    The unconditional Landau mean is DIVERGENT (pdf tail ~ 1/lambda^2),
    so any finite-mean bookkeeping must truncate.  The physical cutoff
    is the maximum single-collision energy transfer T_max, i.e.
    ``lam_max = mode + (T_max - dE_mp) / xi_eff`` in scipy's variable
    (see :meth:`Foil._landau_lam_max`).  The conditional mean grows like
    ln(lam_max); it is evaluated once per lam_max by quadrature
    (segments sized for the 1/lambda integrand tail) and cached.
    """
    from scipy import integrate
    from scipy.stats import landau

    lam_max = float(lam_max)
    f = lambda u: u * landau.pdf(u)   # noqa: E731
    # Peak segment, then geometrically growing tail segments (the
    # integrand decays like 1/lambda out to lam_max ~ 1/kappa ~ 1e4+).
    edges = [-8.0, 20.0]
    while edges[-1] < lam_max:
        edges.append(min(edges[-1] * 20.0, lam_max))
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        val, _ = integrate.quad(f, a, b, limit=200)
        total += val
    return total / float(landau.cdf(lam_max))


class Foil(ThinKickElement):
    """Stripper / scattering foil.

    Parameters
    ----------
    name : str
        Element name in the lattice.
    material : str
        One of the keys in :data:`_MATERIALS` (``"C"``, ``"Be"``, ``"Al"``,
        ``"Cu"``, ``"Mo"``, ``"W"``).  Case-sensitive; the convention is
        the chemical symbol with first letter uppercase.
    thickness_ug_cm2 : float
        Areal density (μg/cm²).  Typical PIP-II stripper foils are
        300–700 μg/cm² of carbon.
    aperture : float, optional
        Aperture radius in mm (default 0 = unrestricted).
    seed : int or None, optional
        Seed for the per-particle random kick generator.  ``None`` (default)
        means each call draws from a fresh non-deterministic stream — fine
        for production but use a fixed seed in tests.  A property: setting
        it (also through ``--set FOIL.seed=…``) rebuilds the generator.
    straggling : str, optional
        Energy-loss straggling model: ``"auto"`` (default — dispatch on
        the Vavilov thickness parameter kappa = xi/T_max, see
        :meth:`resolved_straggling_model`), ``"landau"`` (force the
        thin-absorber Landau shape), or ``"gaussian"`` (force the Bohr
        Gaussian — the pre-2026 behaviour, kept for backward
        compatibility).
    strip_model : str, optional
        ``"off"`` (default — a passive scatterer, the historical element)
        or ``"two_step"``: an H⁻ beam is converted H⁻ → H⁰ → p with the
        calibrated cross sections of :meth:`stripping_fractions`; the
        ions left as H⁻ / H⁰ are recorded on the beam's
        ``unstripped_table`` (they stay alive — not a loss).  Inert for
        a positive species.
    dedx_model : str, optional
        Mean ionisation loss: ``"mip"`` (default — the tabulated
        minimum-ionising rate, a floor below ~2 GeV) or ``"bethe"`` (the
        β-dependent Bethe formula without density/shell corrections).
    dx, dy : float, optional
        Foil offset (mm) — the tracker/envelope wrap translates the beam
        into the foil frame like any other misaligned element.
    extent_mm : None | float | (float, float), optional
        Foil half-size (mm; one value = square).  ``None`` = infinite.
        With an extent, particles outside it MISS the foil: no kick, no
        loss, recorded as ``"missed"`` on ``unstripped_table``.
    """

    # kick_matrix is constant identity (the MEAN map of zero-mean random
    # kicks).  Cache key tuple is empty so any cached lookup treats two
    # Foils of differing material/thickness as equivalent for the matrix
    # solver — which is correct (both → identity).  The envelope solver
    # does NOT stop at this identity: it adds the diffusion matrix from
    # :meth:`envelope_diffusion` (see module docstring).
    _cache_keys: tuple[str, ...] = ()

    def __init__(self, name: str, material: str = "C",
                 thickness_ug_cm2: float = 600.0,
                 aperture: float = 0.0,
                 seed: int | None = None,
                 straggling: str = "auto",
                 strip_model: str = "off",
                 dedx_model: str = "mip",
                 dx: float = 0.0, dy: float = 0.0,
                 extent_mm=None) -> None:
        super().__init__(name=name, aperture=aperture)
        if material not in _MATERIALS:
            raise ValueError(
                f"Foil material {material!r} not in supported set "
                f"{list(_MATERIALS.keys())}"
            )
        if straggling not in _STRAGGLING_MODES:
            raise ValueError(
                f"Foil straggling {straggling!r} not in supported set "
                f"{list(_STRAGGLING_MODES)}"
            )
        if strip_model not in _STRIP_MODELS:
            raise ValueError(
                f"Foil strip_model {strip_model!r} not in supported set "
                f"{list(_STRIP_MODELS)}")
        if dedx_model not in _DEDX_MODELS:
            raise ValueError(
                f"Foil dedx_model {dedx_model!r} not in supported set "
                f"{list(_DEDX_MODELS)}")
        self.material = material
        self.thickness_ug_cm2 = float(thickness_ug_cm2)
        self.straggling = straggling
        self.strip_model = strip_model
        self.dedx_model = dedx_model
        self.dx = float(dx)
        self.dy = float(dy)
        self._extent_mm = None
        self.extent_mm = extent_mm
        # ``np.random.default_rng(None)`` already seeds from the OS entropy
        # pool — exactly what we want when ``seed`` is None.
        self._seed = None
        self.seed = seed

    @staticmethod
    def _norm_extent(extent):
        """None, a number (square), a pair, or the deck / override string
        ``"x,y"`` / ``"x"`` → ``None`` or ``(half_x, half_y)``."""
        if extent is None:
            return None
        if isinstance(extent, str):
            s = extent.strip()
            if s.lower() in ("", "none"):
                return None
            parts = [float(v) for v in s.split(",")]
            extent = parts[0] if len(parts) == 1 else parts[:2]
        if isinstance(extent, (int, float)):
            ex = ey = float(extent)
        else:
            ex, ey = (float(v) for v in extent)
        if ex <= 0.0 or ey <= 0.0:
            raise ValueError(f"Foil extent_mm must be positive, got {extent!r}")
        return (ex, ey)

    @property
    def extent_mm(self):
        """Foil half-size (mm) as ``(half_x, half_y)`` or None; the setter
        accepts a number, a pair or the ``"x,y"`` string an override or the
        deck line carries."""
        return self._extent_mm

    @extent_mm.setter
    def extent_mm(self, value) -> None:
        self._extent_mm = self._norm_extent(value)

    @property
    def seed(self):
        """Seed of the per-particle kick generator (None = unseeded)."""
        return self._seed

    @seed.setter
    def seed(self, value) -> None:
        if value is not None:
            value = int(value)
        self._seed = value
        self._rng = np.random.default_rng(value)

    # -- physics helpers ----------------------------------------------------
    #
    # All helpers accept either a Beam (uses beam.ref) or a
    # ReferenceParticle directly, so the envelope solver — which has no
    # Beam — can reuse the same formulas.

    @staticmethod
    def _ref_of(beam_or_ref):
        """Normalise a Beam or ReferenceParticle argument to the ref."""
        return getattr(beam_or_ref, "ref", beam_or_ref)

    def _thickness_over_X0(self) -> float:
        """Fractional thickness in radiation-length units (x/X0)."""
        mat = _MATERIALS[self.material]
        # thickness in μg/cm² → g/cm²:  × 1e-6
        x_g_cm2 = self.thickness_ug_cm2 * 1e-6
        return x_g_cm2 / mat["X0_g_cm2"]

    def _highland_theta_rms(self, beam_or_ref) -> float:
        """Highland (1991) RMS scattering angle in **radians** for this foil
        and the current reference particle.

        θ_rms = (13.6 MeV / βcp) · |z| · √(x/X0) · [1 + 0.038·ln(x/X0)]

        where ``z`` is the projectile charge (in electron units), ``p`` is
        the momentum in MeV/c, and ``x/X0`` the fractional thickness.
        """
        ref = self._ref_of(beam_or_ref)
        x_over_X0 = self._thickness_over_X0()
        if x_over_X0 <= 0.0:
            return 0.0
        # |charge| in e units — Highland's |z| factor.
        z_proj = abs(ref.species.charge)
        # p in MeV/c.  ReferenceParticle stores bg = β·γ; p = bg · m.
        p_MeVc = ref.bg * ref.species.mass
        beta = ref.beta
        if p_MeVc <= 0.0 or beta <= 0.0:
            return 0.0
        # 13.6 MeV / (β · p_MeVc) — the leading constant in Highland.
        base = 13.6 / (beta * p_MeVc)
        log_correction = 1.0 + 0.038 * math.log(x_over_X0)
        theta = base * z_proj * math.sqrt(x_over_X0) * log_correction
        return float(theta)

    def _mean_energy_loss_MeV(self, beam_or_ref) -> float:
        """Mean Bethe-Bloch energy loss per particle traversing the foil.

        Uses the tabulated minimum-ionising ⟨dE/dx⟩ for the material and
        scales linearly with thickness.  Off-MIP corrections (β-dependent)
        are ignored, so below the proton MIP plateau (~2 GeV) this
        UNDERESTIMATES the true mean loss — increasingly so at low
        energy (the 1/β² rise).  For the thin foils this element
        targets the absolute error stays far below the beam energy;
        treat it as a MIP-floor estimate, not a percent-level one.

        This value is the element's PHYSICAL mean loss in every
        straggling mode: the Landau sampler in :meth:`apply_kick` is
        constructed so its ensemble mean equals this number exactly
        (see :meth:`_sample_energy_loss_MeV`).
        """
        ref = self._ref_of(beam_or_ref)
        mat = _MATERIALS[self.material]
        x_g_cm2 = self.thickness_ug_cm2 * 1e-6
        # |charge|² scaling on dE/dx (Bethe-Bloch).
        z_proj = abs(ref.species.charge)
        if getattr(self, "dedx_model", "mip") == "bethe":
            return float(self._bethe_dedx_MeVcm2_g(ref) * x_g_cm2)
        return float(mat["dEdx_min_MeVcm2_g"] * x_g_cm2 * z_proj ** 2)

    def _bethe_dedx_MeVcm2_g(self, beam_or_ref) -> float:
        """Mean stopping power (MeV cm²/g) from the Bethe formula
        [PDG Eq. 34.5] without density or shell corrections:
        ``K z² (Z/A) / β² [½ ln(2 m_e c² β²γ² T_max / I²) − β²]``."""
        ref = self._ref_of(beam_or_ref)
        mat = _MATERIALS[self.material]
        beta2 = ref.beta * ref.beta
        if beta2 <= 0.0:
            return 0.0
        z_proj = abs(ref.species.charge)
        t_max = self._t_max_MeV(ref)
        i_mev = mat["I_eV"] * 1e-6
        arg = 2.0 * M_ELECTRON * ref.bg * ref.bg * t_max / (i_mev * i_mev)
        val = (_K_BETHE * z_proj ** 2 * mat["Z"] / mat["A"] / beta2
               * (0.5 * math.log(arg) - beta2))
        return float(max(val, 0.0))

    # ---- charge exchange -------------------------------------------------
    def strip_cross_sections_cm2(self, beam_or_ref) -> tuple[float, float]:
        """(σ_-10, σ_01) per atom at the reference velocity — the carbon
        values calibrated to the PIP-II anchors, scaled 1/β², and by
        sqrt(Z/6) for other materials (a convention)."""
        ref = self._ref_of(beam_or_ref)
        beta = ref.beta
        if beta <= 0.0:
            return (0.0, 0.0)
        s10, s01 = _STRIP_SIGMA_800
        scale = (_STRIP_BETA_800 / beta) ** 2
        z_scale = math.sqrt(_MATERIALS[self.material]["Z"] / 6.0)
        return (s10 * scale * z_scale, s01 * scale * z_scale)

    def stripping_fractions(self, beam_or_ref, thickness_ug_cm2=None):
        """``(f_p, f_H0, f_Hminus)`` of a negative-ion beam after the foil
        (two-step model; a positive species returns ``(1, 0, 0)``).
        ``thickness_ug_cm2`` overrides the element's thickness (curves)."""
        ref = self._ref_of(beam_or_ref)
        if ref.species.charge >= 0:
            return (1.0, 0.0, 0.0)
        x = self.thickness_ug_cm2 if thickness_ug_cm2 is None else float(thickness_ug_cm2)
        if x <= 0.0:
            return (0.0, 0.0, 1.0)
        n_atoms = x * 1e-6 / _MATERIALS[self.material]["A"] * _N_AVOGADRO
        s10, s01 = self.strip_cross_sections_cm2(ref)
        e10 = math.exp(-s10 * n_atoms)
        e01 = math.exp(-s01 * n_atoms)
        if abs(s10 - s01) < 1e-30:
            h0 = s10 * n_atoms * e10
        else:
            h0 = s10 / (s10 - s01) * (e01 - e10)
        f_p = max(0.0, 1.0 - e10 - h0)
        return (f_p, float(h0), float(e10))

    def _energy_loss_sigma_MeV(self, beam_or_ref) -> float:
        """Gaussian-approximation straggling width (Bohr's formula).

        σ_E² (MeV²)  =  K · z² · (Z/A) · x  ·  (mass terms)

        where K ≈ 0.1535 MeV·cm²/g.  We use the high-energy ≪ relativistic
        ≫ approximation σ_E = √(K · z² · Z/A · x) MeV.  This is exact at
        ultra-relativistic and slightly under-estimates straggling at
        non-relativistic energies — acceptable for a first-pass model.

        NOTE: this is the width of the GAUSSIAN straggling model (and the
        envelope solver's Σ_WW diffusion term).  In the Landau regime the
        loss distribution has no meaningful finite width parameter — its
        variance is dominated by rare large single-collision transfers
        (formally divergent, T_max-truncated in the sampler) — so this σ
        should be read as conventional rms-level bookkeeping there.
        """
        ref = self._ref_of(beam_or_ref)
        mat = _MATERIALS[self.material]
        x_g_cm2 = self.thickness_ug_cm2 * 1e-6
        z_proj = abs(ref.species.charge)
        if x_g_cm2 <= 0.0:
            return 0.0
        var = 0.1535 * (z_proj ** 2) * (mat["Z"] / mat["A"]) * x_g_cm2
        return float(math.sqrt(var))

    # -- Landau / Vavilov straggling machinery -------------------------------

    def _landau_xi_MeV(self, beam_or_ref) -> float:
        """Landau's thickness parameter ξ (MeV).

        ξ = 0.1535 (MeV·cm²/g) · z² · (Z/A) / β² · x[g/cm²]

        (0.1535 = K/2 with K the Bethe constant 0.307075 MeV·cm²/g —
        the same constant the Bohr σ above uses.)
        """
        ref = self._ref_of(beam_or_ref)
        mat = _MATERIALS[self.material]
        x_g_cm2 = self.thickness_ug_cm2 * 1e-6
        if x_g_cm2 <= 0.0:
            return 0.0
        z_proj = abs(ref.species.charge)
        beta = ref.beta
        if beta <= 0.0:
            return 0.0
        return float(
            0.1535 * (z_proj ** 2) * (mat["Z"] / mat["A"]) * x_g_cm2
            / (beta * beta)
        )

    def _t_max_MeV(self, beam_or_ref) -> float:
        """Maximum single-collision energy transfer to an electron (MeV).

        T_max = 2 m_e c² β²γ² / (1 + 2γ m_e/M + (m_e/M)²)   [PDG Eq. 34.4]
        """
        ref = self._ref_of(beam_or_ref)
        bg = ref.bg
        if bg <= 0.0:
            return 0.0
        m_ratio = M_ELECTRON / ref.species.mass
        return float(
            2.0 * M_ELECTRON * bg * bg
            / (1.0 + 2.0 * ref.gamma * m_ratio + m_ratio * m_ratio)
        )

    def kappa(self, beam_or_ref) -> float:
        """Vavilov thickness parameter κ = ξ / T_max (dimensionless).

        κ < 0.01: Landau regime (thin absorber, skewed loss distribution).
        κ > 10:   Gaussian (Bohr) regime (thick absorber, CLT applies).
        Intermediate: Vavilov regime (not implemented — see
        :meth:`resolved_straggling_model`).
        """
        t_max = self._t_max_MeV(beam_or_ref)
        if t_max <= 0.0:
            return 0.0
        return self._landau_xi_MeV(beam_or_ref) / t_max

    def resolved_straggling_model(self, beam_or_ref) -> str:
        """The straggling model :meth:`apply_kick` will actually use.

        Explicit ``straggling="landau"`` / ``"gaussian"`` win outright.
        ``"auto"`` dispatches on κ = ξ/T_max:

          κ < 0.01  → ``"landau"``
          κ > 10    → ``"gaussian"`` (Bohr — correct thick-absorber limit)
          otherwise → ``"gaussian"`` with a warning: the correct model in
                      this regime is Vavilov's, which is NOT implemented;
                      the Gaussian is used as the nearest available shape.
        """
        if self.straggling != "auto":
            return self.straggling
        k = self.kappa(beam_or_ref)
        if k < _KAPPA_LANDAU_MAX:
            return "landau"
        if k > _KAPPA_GAUSSIAN_MIN:
            return "gaussian"
        warnings.warn(
            f"Foil {self.name!r}: kappa = xi/T_max = {k:.3g} is in the "
            f"Vavilov regime ({_KAPPA_LANDAU_MAX} <= kappa <= "
            f"{_KAPPA_GAUSSIAN_MIN}); Vavilov straggling is not "
            "implemented — falling back to the Gaussian (Bohr) shape.",
            stacklevel=2,
        )
        return "gaussian"

    def _landau_delta_e_mp_MeV(self, beam_or_ref) -> float:
        """Theoretical Landau most-probable energy loss (MeV).

        Standard thin-absorber parametrization [PDG Eq. 34.12]:

        ΔE_mp = ξ [ ln(2 m_e c² β²γ² / I) + ln(ξ/I) + 0.2 − β² − δ ]

        with I the material's mean excitation energy.  The density
        correction δ is set to 0 — consistent with the tabulated-MIP
        mean-loss convention above, which also carries no density
        correction (δ ≈ 0.1–0.3 at βγ ≈ 1.6 shifts ΔE_mp by ≲ 2 %,
        below the parametrization's own O(ξ) convention slop).
        """
        ref = self._ref_of(beam_or_ref)
        xi = self._landau_xi_MeV(beam_or_ref)
        if xi <= 0.0:
            return 0.0
        mat = _MATERIALS[self.material]
        I_MeV = mat["I_eV"] * 1e-6
        bg = ref.bg
        beta2 = ref.beta * ref.beta
        return float(xi * (
            math.log(2.0 * M_ELECTRON * bg * bg / I_MeV)
            + math.log(xi / I_MeV) + 0.2 - beta2
        ))

    def _landau_scale_MeV(self, beam_or_ref) -> float:
        """Effective scale ξ_eff = (π/2)·ξ of the loss distribution in
        scipy's standard-Landau variable.

        Landau's theory is written for phi(lambda) with width parameter
        ξ; scipy's landau is the same law compressed by 2/π (see the
        :data:`_LANDAU_SCIPY_TO_PHI` note), so sampling scipy's variable
        with scale ξ_eff reproduces the physical shape: core FWHM
        2.558·ξ_eff = 4.02·ξ and Rutherford tail P(loss>E) ≈ ξ/E.
        """
        return _LANDAU_SCIPY_TO_PHI * self._landau_xi_MeV(beam_or_ref)

    def most_probable_loss_MeV(self, beam_or_ref) -> float:
        """Most probable energy loss of the distribution actually sampled.

        - Gaussian model: the mode is the Bethe-Bloch mean loss.
        - Landau model: the mode of the mean-pinned truncated-Landau
          construction (see :meth:`_sample_energy_loss_MeV`):
          ⟨ΔE⟩_BB + ξ_eff·(λ_mode − ⟨λ⟩_trunc).  It differs from the
          standalone ΔE_mp parametrization of
          :meth:`_landau_delta_e_mp_MeV` by exactly the amount the
          tabulated-MIP mean ⟨ΔE⟩_BB differs from the truncated
          construction's raw mean (an algebraic identity — the pinning
          shifts the whole curve); tests pin both that identity and the
          raw mean's O(ξ) agreement with the full Bethe formula.
        """
        model = self.resolved_straggling_model(beam_or_ref)
        de_mean = self._mean_energy_loss_MeV(beam_or_ref)
        if model != "landau":
            return de_mean
        scale = self._landau_scale_MeV(beam_or_ref)
        if scale <= 0.0:
            return de_mean
        lam_max = self._landau_lam_max(beam_or_ref)
        lam_mean = _landau_truncated_mean(round(lam_max, 3))
        return float(de_mean + scale * (_LANDAU_MODE - lam_mean))

    def _landau_lam_max(self, beam_or_ref) -> float:
        """Truncation point (scipy's standard-Landau variable) at which the
        raw sampled loss ΔE_mp + ξ_eff·(λ − mode) reaches T_max."""
        scale = self._landau_scale_MeV(beam_or_ref)
        if scale <= 0.0:
            return 0.0
        t_max = self._t_max_MeV(beam_or_ref)
        de_mp = self._landau_delta_e_mp_MeV(beam_or_ref)
        return _LANDAU_MODE + (t_max - de_mp) / scale

    def _sample_energy_loss_MeV(self, ref, n: int) -> np.ndarray:
        """Draw ``n`` per-particle energy LOSSES (positive = loss, MeV)
        from the resolved straggling model.

        Gaussian ("gaussian", and "auto" outside the Landau regime):

            ΔE_i = ⟨ΔE⟩_BB + N(0, σ_Bohr)

        Landau ("landau", and "auto" with κ < 0.01) — what is sampled,
        precisely:

            λ_i  ~ scipy.stats.landau (standard), REJECTED above
                   λ_max = mode + (T_max − ΔE_mp)/ξ_eff  (single-collision
                   kinematic cutoff; P(λ > λ_max) ≈ κ, i.e. ~1e-5 here);
            ΔE_i = ⟨ΔE⟩_BB + ξ_eff·(λ_i − ⟨λ⟩_trunc)

        with ξ_eff = (π/2)·ξ (scipy's landau is 2/π narrower than
        Landau's φ — see :data:`_LANDAU_SCIPY_TO_PHI`) and ⟨λ⟩_trunc the
        analytic mean of the truncated standard Landau
        (:func:`_landau_truncated_mean`).  By construction the ensemble
        mean is EXACTLY the Bethe-Bloch ⟨ΔE⟩_BB of
        :meth:`_mean_energy_loss_MeV` — the raw Landau mean is divergent,
        so the physical mean is pinned rather than emergent.  The shape
        is the truncated Landau's: core FWHM ≈ 4.02 ξ, right skew,
        Rutherford tail P(loss > E) ≈ ξ/E up to T_max.  The mode sits at
        ⟨ΔE⟩_BB + ξ_eff(λ_mode − ⟨λ⟩_trunc); cross-checks against the
        standard ΔE_mp parametrization and the full Bethe mean are in
        tests/elements/test_foil.py.
        """
        de_mean = self._mean_energy_loss_MeV(ref)
        model = self.resolved_straggling_model(ref)

        if model == "gaussian":
            sigma = self._energy_loss_sigma_MeV(ref)
            noise = (self._rng.normal(0.0, sigma, size=n)
                     if sigma > 0.0 else 0.0)
            return de_mean + noise

        # Landau branch
        from scipy.stats import landau
        scale = self._landau_scale_MeV(ref)
        if scale <= 0.0:
            return np.full(n, de_mean)
        lam_max = self._landau_lam_max(ref)
        if lam_max < 10.0:
            # Only reachable by forcing straggling="landau" far outside
            # its validity (the "auto" dispatch guarantees κ < 0.01 ⇒
            # λ_max ≳ 100): the truncated distribution would be mostly
            # cutoff artefact, and the rejection loop below would stall.
            raise ValueError(
                f"Foil {self.name!r}: straggling='landau' requested but "
                f"kappa = {self.kappa(ref):.3g} is far outside the "
                "thin-absorber regime (lam_max < 10); use "
                "straggling='gaussian' (or 'auto')."
            )
        lam = landau.rvs(size=n, random_state=self._rng)
        # Kinematic cutoff at T_max: redraw the ~κ·N samples above it.
        oversized = lam > lam_max
        for _ in range(100):
            n_bad = int(np.count_nonzero(oversized))
            if n_bad == 0:
                break
            lam[oversized] = landau.rvs(size=n_bad,
                                        random_state=self._rng)
            oversized = lam > lam_max
        else:  # pragma: no cover — P(fail) ~ kappa^100
            lam = np.minimum(lam, lam_max)
        lam_mean = _landau_truncated_mean(round(lam_max, 3))
        return de_mean + scale * (lam - lam_mean)

    # -- ThinKickElement API -----------------------------------------------

    def apply_kick(self, beam) -> None:
        """Apply stochastic scattering + ionisation loss to alive particles.

        With an ``extent_mm`` only the particles inside the foil (in the
        foil frame the tracker translated them into) are kicked; the rest
        are recorded as ``"missed"``.  With ``strip_model="two_step"`` on
        a negative-ion beam, one uniform draw per hit particle — AFTER the
        scattering and loss draws, so the historical kicks are unchanged
        for a given seed — assigns its charge state; H⁰ and H⁻ survivors
        are recorded on ``beam.unstripped_table`` and stay alive.
        """
        alive = beam.alive_mask
        hit = alive
        extent = getattr(self, "extent_mm", None)
        if extent is not None:
            ex, ey = extent
            inside = ((np.abs(beam.particles[:, 0]) <= ex)
                      & (np.abs(beam.particles[:, 2]) <= ey))
            hit = alive & inside
            missed = np.flatnonzero(alive & ~inside)
            if missed.size:
                beam.record_unstripped(missed, "missed", self.name)
        n_hit = int(np.count_nonzero(hit))
        if n_hit == 0:
            return

        theta_rms = self._highland_theta_rms(beam)            # radians

        # Particle layout: columns are (x, xp, y, yp, dphi, dw).
        # Highland gives plane-projected scatter: independent draws on xp,yp
        # (the small-angle plane projection of an isotropic 3-D Gaussian).
        if theta_rms > 0.0:
            d_xp = self._rng.normal(0.0, theta_rms, size=n_hit)
            d_yp = self._rng.normal(0.0, theta_rms, size=n_hit)
            # xp / yp are stored in **mrad** (HELIX convention — same as
            # Steerer's ``* 1e3``).  Convert radians → mrad.
            beam.particles[hit, 1] += d_xp * 1e3
            beam.particles[hit, 3] += d_yp * 1e3

        if self.thickness_ug_cm2 > 0.0:
            losses = self._sample_energy_loss_MeV(beam.ref, n_hit)
            # dw is stored in MeV; a LOSS decreases dw.
            beam.particles[hit, 5] -= losses

        if (getattr(self, "strip_model", "off") == "two_step"
                and beam.ref.species.charge < 0):
            f_p, f_h0, _f_hm = self.stripping_fractions(beam.ref)
            u = self._rng.uniform(0.0, 1.0, size=n_hit)
            ids = np.flatnonzero(hit)
            h0 = ids[(u >= f_p) & (u < f_p + f_h0)]
            hm = ids[u >= f_p + f_h0]
            if h0.size:
                beam.record_unstripped(h0, "H0", self.name)
            if hm.size:
                beam.record_unstripped(hm, "H-", self.name)

    def kick_matrix(self, ref) -> np.ndarray:
        """Identity — the MEAN map of zero-mean random kicks.

        Same convention as Steerer's zero-strength branch.  The envelope
        solver must NOT stop here: it adds the second-moment diffusion
        from :meth:`envelope_diffusion` on top of this identity.
        """
        return np.eye(6)

    # -- envelope (second-moment) interface ----------------------------------

    def envelope_diffusion(self, ref) -> tuple[np.ndarray, float]:
        """Second-moment update for the envelope solver.

        Returns ``(D, dE_mean_MeV)`` where ``D`` is the 6×6 diffusion
        matrix to ADD to Σ (envelope units: mm, mrad, deg, MeV):

            D[1,1] = D[3,3] = (θ₀ · 10³)²   [mrad²]  (Highland rms plane
                                                       angle θ₀ in rad)
            D[5,5] = σ_E²                    [MeV²]   (Bohr straggling)

        and ``dE_mean_MeV`` is the Bethe-Bloch mean loss to subtract
        from the reference energy.  Physics: for a zero-mean stochastic
        kick with second moments D, the exact moment update is
        Σ ← M Σ Mᵀ + D with M the mean map (here the identity).

        Σ_WW uses the Gaussian (Bohr) variance in EVERY straggling mode:
        in the Landau regime the loss distribution has no meaningful
        finite second moment (tail-truncation-dominated), so the Bohr
        value is kept as the conventional rms-level diffusion
        coefficient — see :meth:`_energy_loss_sigma_MeV`.
        """
        theta_mrad = self._highland_theta_rms(ref) * 1e3
        sigma_E = self._energy_loss_sigma_MeV(ref)
        D = np.zeros((6, 6))
        D[1, 1] = theta_mrad * theta_mrad
        D[3, 3] = theta_mrad * theta_mrad
        D[5, 5] = sigma_E * sigma_E
        return D, self._mean_energy_loss_MeV(ref)

    # -- diagnostics --------------------------------------------------------

    def option_tokens(self) -> list[str]:
        """The non-default ``key=value`` tokens of the deck line
        (``; HELIX_FOIL name material thickness [straggling] [key=value…]``)."""
        toks: list[str] = []
        if getattr(self, "strip_model", "off") != "off":
            toks.append(f"strip_model={self.strip_model}")
        if getattr(self, "dedx_model", "mip") != "mip":
            toks.append(f"dedx_model={self.dedx_model}")
        if getattr(self, "dx", 0.0):
            toks.append(f"dx={self.dx!r}")
        if getattr(self, "dy", 0.0):
            toks.append(f"dy={self.dy!r}")
        ext = getattr(self, "extent_mm", None)
        if ext is not None:
            toks.append(f"extent_mm={ext[0]!r},{ext[1]!r}")
        if getattr(self, "seed", None) is not None:
            toks.append(f"seed={int(self.seed)}")
        return toks

    def apply_option_token(self, key: str, value: str) -> None:
        """Set one ``key=value`` option from a deck line (parser side)."""
        if key == "strip_model":
            if value not in _STRIP_MODELS:
                raise ValueError(f"Foil strip_model {value!r} not in {list(_STRIP_MODELS)}")
            self.strip_model = value
        elif key == "dedx_model":
            if value not in _DEDX_MODELS:
                raise ValueError(f"Foil dedx_model {value!r} not in {list(_DEDX_MODELS)}")
            self.dedx_model = value
        elif key in ("dx", "dy"):
            setattr(self, key, float(value))
        elif key == "extent_mm":
            self.extent_mm = value
        elif key == "seed":
            self.seed = int(float(value))
        else:
            raise ValueError(f"Foil: unknown option {key!r}")

    def __repr__(self) -> str:
        return (
            f"Foil(name={self.name!r}, material={self.material!r}, "
            f"thickness_ug_cm2={self.thickness_ug_cm2:g}, "
            f"straggling={self.straggling!r})"
        )
