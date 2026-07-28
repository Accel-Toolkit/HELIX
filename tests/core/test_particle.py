# tests/core/test_particle.py
import pytest
from linac_gen.core.particle import Particle, PROTON, DEUTERON, H_MINUS

def test_proton_mass():
    assert abs(PROTON.mass - 938.272) < 0.01

def test_proton_charge():
    assert PROTON.charge == 1

def test_deuteron():
    assert abs(DEUTERON.mass - 1875.613) < 0.01
    assert DEUTERON.charge == 1

def test_h_minus():
    assert H_MINUS.charge == -1
    # H⁻ ion = proton + 2 electrons (binding 0.75 eV negligible) = 939.294 MeV.
    # The old assertion pinned mass == m_p — that WAS the bug (0.05% fast TOF,
    # 0.05% low Bρ), caught by the TraceWin PIP-II HB650 validation.
    assert H_MINUS.mass == pytest.approx(PROTON.mass + 2 * 0.51099895, abs=1e-6)
    assert H_MINUS.mass == pytest.approx(939.294, abs=0.001)

def test_custom_ion():
    carbon = Particle(mass=12 * 931.494, charge=6, name="C12_6+")
    assert carbon.charge == 6
    assert carbon.name == "C12_6+"

def test_particle_repr():
    assert "proton" in repr(PROTON)
