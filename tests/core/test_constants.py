# tests/core/test_constants.py
from linac_gen.core.constants import C_LIGHT, E_CHARGE, M_PROTON, AMU, EPSILON_0, PI

def test_speed_of_light_mm_per_ns():
    assert abs(C_LIGHT - 299792458.0) < 1.0

def test_proton_mass_mev():
    assert abs(M_PROTON - 938.272) < 0.01

def test_amu_mev():
    assert abs(AMU - 931.494) < 0.01

def test_elementary_charge():
    assert abs(E_CHARGE - 1.602176634e-19) < 1e-25
