"""Physical-constant profiles for ring mode.

``helix`` (the default) uses HELIX's own constants: CODATA 2018, the values
every linac run uses and also MAD-X 5.09's ``pmass``/``emass``.
``codata2010`` uses the older proton and electron masses of MAD-X up to 5.08
and of Synergia: lattices and programs generated with those tools (the ImpactX
Booster model's RF frequency, 44.704409985 MHz at 800 MeV) are reproduced
bit-exactly only with them.  The two proton masses differ by 4.2e-5 MeV
(4.5e-8 relative), a relative frequency shift of 8.5e-9 at 800 MeV.
``codata2022`` uses CODATA 2022 masses computed from kilograms exactly as
ImpactX 26 does (m / (e·10^6 / c^2)): a ring reference with this profile is
ImpactX's reference particle to the last bit.
"""
from __future__ import annotations

from dataclasses import dataclass

from linac_gen.core import constants as _c


@dataclass(frozen=True)
class ConstantsProfile:
    """Rest masses in MeV/c²; the other constants are SI and shared."""

    name: str
    m_proton: float
    m_electron: float
    c_light: float = _c.C_LIGHT
    e_charge: float = _c.E_CHARGE
    epsilon_0: float = _c.EPSILON_0

    @property
    def m_hminus(self) -> float:
        """H⁻ rest mass: proton + 2 electrons (binding of 0.75 eV neglected, as
        in ``linac_gen.core.particle.H_MINUS``)."""
        return self.m_proton + 2.0 * self.m_electron

    def mass_of(self, species: str) -> float:
        s = species.lower()
        if s in ("proton", "p", "h+"):
            return self.m_proton
        if s in ("h-", "hminus", "h_minus"):
            return self.m_hminus
        raise ValueError(f"ring mode: unknown species {species!r} (proton or H-)")


def _mev_from_kg(m_kg: float) -> float:
    """Rest energy in MeV of a mass in kg, as ImpactX computes it."""
    return m_kg / (_c.E_CHARGE * 1e6 / (_c.C_LIGHT * _c.C_LIGHT))


PROFILES: dict[str, ConstantsProfile] = {
    "helix": ConstantsProfile("helix", _c.M_PROTON, _c.M_ELECTRON),
    "codata2010": ConstantsProfile("codata2010", 938.272046, 0.510998928),
    "codata2022": ConstantsProfile("codata2022", _mev_from_kg(1.67262192595e-27),
                                   _mev_from_kg(9.1093837139e-31)),
}


def profile(name: str) -> ConstantsProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"ring mode: constants profile {name!r} is not one of "
                         f"{sorted(PROFILES)}") from None
