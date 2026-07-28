import pytest
import math
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.particle import PROTON
from linac_gen.core.constants import C_LIGHT

def test_create_reference():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    assert ref.w_kin == 3.0
    assert ref.phi_s == 0.0
    assert ref.s == 0.0

def test_gamma_from_kinetic_energy():
    ref = ReferenceParticle(species=PROTON, w_kin=938.272, frequency=352.21)
    assert abs(ref.gamma - 2.0) < 0.001

def test_beta_from_gamma():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    gamma = 1.0 + 3.0 / PROTON.mass
    beta = math.sqrt(1.0 - 1.0 / gamma**2)
    assert abs(ref.beta - beta) < 1e-10

def test_bg():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    assert abs(ref.bg - ref.beta * ref.gamma) < 1e-10

def test_brho():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    p_mev = PROTON.mass * ref.bg
    brho_expected = p_mev * 1e6 / (C_LIGHT * abs(PROTON.charge))
    assert abs(ref.brho - brho_expected) < 1e-6

def test_wavelength():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    wl_expected = C_LIGHT / (352.21e6) * 1000.0
    assert abs(ref.wavelength - wl_expected) < 0.01

def test_update_energy():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    old_gamma = ref.gamma
    ref.w_kin = 10.0
    assert ref.gamma > old_gamma
    assert abs(ref.gamma - (1.0 + 10.0 / PROTON.mass)) < 1e-10

def test_advance_s():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    ref.s += 100.0
    assert ref.s == 100.0

def test_update_frequency_recomputes_wavelength():
    """Changing frequency must update wavelength immediately."""
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    wl_old = ref.wavelength
    ref.frequency = 704.42  # double the frequency
    assert abs(ref.wavelength - wl_old / 2.0) < 0.01
    assert ref.frequency == 704.42
