"""Opt-in linearised MAD-X export (``write_madx(..., linearize=True)``):
field maps / explicit-matrix elements / thin lenses become MAD-X ``MATRIX``
elements in MAD-X's canonical basis, and ``MATRIX`` imports back to a
``MatrixElement``.

Conventions pinned against MAD-X 5.09.03 (cpymad) in
tests/io/test_madx_oracle.py; here the basis change is checked against
the analytic MAD-X drift (R56 = L/β²γ²), the export/import pair is
checked as an exact inverse in BOTH regimes (ΔW = 0 and ΔW ≠ 0), and the
warnings that document MAD-X's limits (fixed reference energy, TWISS
symplectification) are asserted to fire.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from linac_gen.core.constants import C_LIGHT
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import H_MINUS, PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.elements.foil import Foil
from linac_gen.elements.matrix_element import MatrixElement
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.steerer import Steerer
from linac_gen.elements.thin_lens import ThinLens
from linac_gen.io.field_map_data import FieldChannel, FieldMapData
from linac_gen.io.madx_parser import parse_madx
from linac_gen.io.madx_writer import write_madx
from linac_gen.io.tracewin_geom import Channel
from linac_gen.tracking.longitudinal_coords import (
    madx_transform, matrix_from_madx, matrix_to_madx, vector_from_madx,
    vector_to_madx)
from linac_gen.tracking.matrix_tracking import (compute_transfer_matrix,
                                                get_element_matrix)


def _ref(species=PROTON, w=5.0, f=352.21):
    return ReferenceParticle(species=species, w_kin=w, frequency=f)


def _box_field(kind, value, L_mm=200.0, f_MHz=0.0):
    """Uniform-field 3-D box map (deliberately hard-edged: its map is NOT
    symplectic, which the export must flag)."""
    n, nz = 3, 11
    x = np.linspace(-10.0, 10.0, n)
    z = np.linspace(0.0, L_mm, nz)
    fd = FieldMapData(z=z, frequency=f_MHz)
    zeros = np.zeros((n, n, nz))
    fd.channels[kind] = FieldChannel(geometry=7, x=x, y=x.copy(), z=z,
                                     Fx=zeros, Fy=zeros.copy(),
                                     Fz=np.full((n, n, nz), value))
    return fd


def _solenoid_map(name="SOL"):
    return FieldMap3D(name=name, length=200.0,
                      field_data=_box_field(Channel.STAT_B, 0.5), n_steps=50)


def _rf_map(name="CAV"):
    return FieldMap3D(name=name, length=200.0,
                      field_data=_box_field(Channel.RF_E, 2.0, f_MHz=352.21),
                      phase=-30.0, frequency=352.21, n_steps=50)


def _symplectic_4x4(k=0.8, phi=0.6):
    M = np.eye(6)
    M[0, 0] = M[1, 1] = np.cos(phi)
    M[0, 1] = np.sin(phi) / k
    M[1, 0] = -k * np.sin(phi)
    M[2, 2] = M[3, 3] = np.cosh(phi)
    M[2, 3] = np.sinh(phi) / k
    M[3, 2] = k * np.sinh(phi)
    return M


def _matrix_elements(lat):
    return [e for e in lat.elements if isinstance(e, MatrixElement)]


# ---------------------------------------------------------------------------
# basis change
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("species", [PROTON, H_MINUS])
def test_madx_transform_pins_the_measured_conventions(species):
    ref = _ref(species)
    T = madx_transform(ref)
    lambda_free_m = C_LIGHT / (ref.frequency * 1e6)
    assert T[0, 0] == T[1, 1] == T[2, 2] == T[3, 3] == 1000.0
    # t = -c·Δt (ahead > 0) with Δφ = 360·f·Δt  ⇒  Δφ = -360/λ_free · t
    assert T[4, 4] == pytest.approx(-360.0 / lambda_free_m, rel=1e-12)
    # pt = ΔE/(p0 c)  ⇒  ΔW = β0γ0 m · pt
    assert T[5, 5] == pytest.approx(ref.bg * species.mass, rel=1e-12)
    # exit angles rescaled by the local momentum ratio
    T2 = madx_transform(ref, p_local_over_p0=2.0)
    assert T2[1, 1] == T2[3, 3] == 500.0 and T2[0, 0] == 1000.0


def test_drift_converts_to_the_analytic_madx_drift():
    """HELIX drift → MAD-X canonical: R12 = L and R56 = L/(β²γ²)
    (MAD-X's drift, verified bit-for-bit against cpymad in the oracle)."""
    ref = _ref()
    d = Drift("d", length=1000.0)
    R = matrix_to_madx(get_element_matrix(d, ref.copy()), ref)
    assert R[0, 1] == pytest.approx(1.0, rel=1e-12)
    assert R[4, 5] == pytest.approx(1.0 / (ref.beta ** 2 * ref.gamma ** 2),
                                    rel=1e-12)
    assert R[4, 4] == 1.0 and R[5, 5] == 1.0 and R[5, 4] == 0.0


def test_matrix_and_vector_conversions_are_inverses():
    ref = _ref()
    rng = np.random.default_rng(7)
    M = rng.normal(size=(6, 6))
    r_in, r_out = ref.copy(), ref.copy()
    r_in.w_kin = 5.4
    r_out.w_kin = 6.1
    back = matrix_from_madx(matrix_to_madx(M, ref, r_in, r_out), ref, r_in, r_out)
    np.testing.assert_allclose(back, M, rtol=0, atol=1e-13)
    v = rng.normal(size=6)
    np.testing.assert_allclose(vector_from_madx(vector_to_madx(v, ref, r_out), ref, r_out),
                               v, rtol=0, atol=1e-13)


# ---------------------------------------------------------------------------
# export / import — exact inverse in both regimes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("with_offset", [False, True])
@pytest.mark.parametrize("species", [PROTON, H_MINUS])
def test_explicit_matrix_round_trips_exactly(tmp_path, species, with_offset):
    ref = _ref(species)
    M = _symplectic_4x4()
    M[0, 5] = 0.02          # dispersion-like coupling (mm per MeV)
    M[4, 1] = -0.3          # Δφ per mrad
    off = np.array([0.1, -0.2, 0.05, 0.0, 1.5, 0.0]) if with_offset else None
    lat = Lattice()
    lat.add(Drift("d1", length=100.0))
    lat.add(MatrixElement("mx", M, length=250.0, offset=off))
    lat.add(Quadrupole("q", length=100.0, gradient=2.0))
    out = tmp_path / "lin.madx"
    warnings = write_madx(lat, out, ref, linearize=True)
    txt = out.read_text()
    assert "mx: MATRIX, l=0.25" in txt and "MARKER" not in txt
    assert [w for w in warnings if "linearised as MAD-X MATRIX" in w]
    assert not [w for w in warnings if "reference energy gain" in w]
    back, meta = parse_madx(str(out))
    (me,) = _matrix_elements(back)
    assert me.length == pytest.approx(250.0, rel=1e-12)
    np.testing.assert_allclose(me.matrix, M, rtol=0, atol=1e-13)
    if with_offset:
        np.testing.assert_allclose(me.offset, off, rtol=0, atol=1e-13)
        assert not meta["warnings"]          # a transverse/phase offset is not an energy kick
    else:
        assert me.offset is None
    # whole-line matrices agree (the quad after it is unaffected)
    M0 = compute_transfer_matrix(lat, ref.copy())
    M1 = compute_transfer_matrix(back, copy.deepcopy(meta["reference"]))
    np.testing.assert_allclose(M1, M0, rtol=1e-12, atol=1e-13)


def test_explicit_matrix_energy_offset_is_a_kick6(tmp_path):
    """An explicit-matrix element that shifts every particle's energy exports
    that shift as kick6; the importer replays the same visible energy so a
    later MATRIX converts with the same momentum scaling (exact inverse)."""
    ref = _ref()
    M = _symplectic_4x4()
    off = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.25])       # +0.25 MeV
    lat = Lattice()
    lat.add(MatrixElement("m1", M, length=100.0, offset=off))
    lat.add(Drift("d", length=100.0))
    lat.add(MatrixElement("m2", _symplectic_4x4(k=0.5, phi=0.3), length=100.0))
    out = tmp_path / "off.madx"
    write_madx(lat, out, ref, linearize=True)
    p0c = ref.bg * PROTON.mass
    assert f"kick6={0.25 / p0c!r}" in out.read_text()
    back, meta = parse_madx(str(out))
    m1, m2 = _matrix_elements(back)
    np.testing.assert_allclose(m1.matrix, M, rtol=0, atol=1e-13)
    np.testing.assert_allclose(m1.offset, off, rtol=0, atol=1e-13)
    np.testing.assert_allclose(m2.matrix, lat.elements[2].matrix, rtol=0, atol=1e-13)
    assert m2.offset is None
    assert any("energy kick (kick6)" in w for w in meta["warnings"])


def test_field_map_static_regime(tmp_path):
    """ΔW = 0: no kick6, no energy warning; re-import equals the fitted
    matrix; the hard-edged box field is flagged as non-symplectic."""
    ref = _ref()
    lat = Lattice()
    lat.add(Drift("d1", length=100.0))
    lat.add(_solenoid_map())
    lat.add(Drift("d2", length=100.0))
    out = tmp_path / "sol.madx"
    warnings = write_madx(lat, out, ref, linearize=True)
    assert len(warnings) == 1
    assert "linearised as MAD-X MATRIX" in warnings[0]
    assert "reference energy gain" not in warnings[0]
    assert "symplectic error" in warnings[0]
    assert "kick" not in out.read_text().split("MATRIX")[1].split(";")[0]
    back, meta = parse_madx(str(out))
    assert meta["warnings"] == []
    (me,) = _matrix_elements(back)
    r = ref.copy()
    r.s += 100.0
    r.phi_s += 360.0 * 100.0 / (r.beta * r.wavelength)
    expected = get_element_matrix(lat.elements[1], r)
    np.testing.assert_allclose(me.matrix, expected, rtol=0, atol=1e-13)
    assert me.offset is None
    np.testing.assert_allclose(compute_transfer_matrix(back, copy.deepcopy(meta["reference"])),
                               compute_transfer_matrix(lat, ref.copy()), rtol=1e-12, atol=1e-13)


def test_field_map_accelerating_regime(tmp_path):
    """ΔW ≠ 0: the gain becomes kick6 = ΔW/(p0 c) and both warnings fire;
    the canonical form has det = 1 in MAD-X coordinates; re-import is
    exact and carries ΔW as an energy offset."""
    ref = _ref()
    lat = Lattice()
    lat.add(Drift("d1", length=100.0))
    cav = _rf_map()
    lat.add(cav)
    lat.add(Drift("d2", length=300.0))
    out = tmp_path / "cav.madx"
    warnings = write_madx(lat, out, ref, linearize=True)
    (w,) = [w for w in warnings if "linearised as MAD-X MATRIX" in w]
    assert "reference energy gain" in w and "kick6" in w and "BEAM energy fixed" in w
    (note,) = [w for w in warnings if "linearised export:" in w]      # line-level pt note
    assert "1 MATRIX element(s)" in note and len(warnings) == 2
    # in-lattice map and gain, exactly as compute_transfer_matrix evaluates them
    r = ref.copy()
    r.s += 100.0
    r.phi_s += 360.0 * 100.0 / (r.beta * r.wavelength)
    cav.reset_run_state()
    M = cav.fitted_matrix(r.copy())
    cav.advance_ref(r)
    dW = r.w_kin - ref.w_kin
    assert dW != 0.0
    assert np.linalg.det(M[:2, :2]) != pytest.approx(1.0, abs=1e-6)   # adiabatic damping
    p0c = ref.bg * PROTON.mass
    assert f"kick6={dW / p0c!r}" in out.read_text()
    # canonical form: the exit-angle rescaling removes the adiabatic
    # damping factor p_in/p_out from the x block (what remains, ~1e-5, is
    # the hard-edged box field's own non-Maxwellian / integration error).
    R = matrix_to_madx(M, ref, ref, r)
    assert abs(np.linalg.det(R[:2, :2]) - 1.0) < 1e-4
    assert abs(np.linalg.det(R[:2, :2]) - 1.0) < 0.05 * abs(np.linalg.det(M[:2, :2]) - 1.0)
    back, meta = parse_madx(str(out))
    assert any("energy kick (kick6)" in m for m in meta["warnings"])
    (me,) = _matrix_elements(back)
    np.testing.assert_allclose(me.matrix, M, rtol=0, atol=1e-13)
    assert me.offset[5] == pytest.approx(dW, rel=1e-12)
    assert np.all(me.offset[:5] == 0.0)


def test_linearize_does_not_touch_supported_elements(tmp_path):
    """With nothing to linearise, linearize=True is byte-identical."""
    from linac_gen.io.tracewin_parser import parse_tracewin
    lat = parse_tracewin("examples/fodo_cell.dat")[0]
    ref = _ref(H_MINUS, 3.0)
    a, b = tmp_path / "a.madx", tmp_path / "b.madx"
    kw = dict(sequence_name="fodo", title="fodo")
    assert write_madx(lat, a, ref, **kw) == []
    assert write_madx(lat, b, ref, linearize=True, **kw) == []
    assert a.read_text() == b.read_text()


def test_thin_lens_is_linearised_but_foil_and_electric_steerer_are_not(tmp_path):
    ref = _ref()
    lat = Lattice()
    lat.add(ThinLens("tl", fx=2000.0, fy=-3000.0))
    lat.add(Drift("d", length=50.0))
    lat.add(Steerer("es", bx_l=1e-3, by_l=0.0, elec=True))
    lat.add(Foil("foil", thickness=1.0)) if _foil_ok() else None
    out = tmp_path / "mix.madx"
    warnings = write_madx(lat, out, ref, linearize=True)
    txt = out.read_text()
    assert "tl: MATRIX" in txt
    assert "es: MARKER" in txt
    assert any("es" in w and "MARKER" in w for w in warnings)
    back, _ = parse_madx(str(out))
    (me,) = _matrix_elements(back)
    np.testing.assert_allclose(me.matrix, get_element_matrix(lat.elements[0], ref.copy()),
                               rtol=0, atol=1e-13)
    if _foil_ok():
        assert "foil: MARKER" in txt


def _foil_ok() -> bool:
    try:
        Foil("f", thickness=1.0)
        return True
    except TypeError:
        return False


def test_strict_mode_still_refuses_what_cannot_be_linearised(tmp_path):
    ref = _ref()
    lat = Lattice()
    lat.add(_solenoid_map())
    lat.add(Steerer("es", bx_l=1e-3, by_l=0.0, elec=True))
    with pytest.raises(ValueError, match="refused"):
        write_madx(lat, tmp_path / "x.madx", ref, linearize=True,
                   on_unsupported="error")
    lat2 = Lattice()
    lat2.add(_solenoid_map())
    assert write_madx(lat2, tmp_path / "y.madx", ref, linearize=True,
                      on_unsupported="error")          # linearised ⇒ no refusal


def test_large_accumulated_gain_is_flagged_as_invalid_for_madx(tmp_path):
    """pt is a deviation about the ENTRANCE momentum: a line whose
    linearised gain is a sizeable fraction of p0c gets an explicit
    'MAD-X will not reproduce this' warning (measured limit, see the
    writer); a small gain gets the quantitative note only."""
    ref = _ref()
    lat = Lattice()
    lat.add(Drift("d1", length=100.0))
    for k in range(40):                       # 40 box cavities: ~0.5 MeV at 5 MeV
        lat.add(_rf_map(f"CAV{k}"))
        lat.add(Drift(f"d{k}", length=50.0))
    warnings = write_madx(lat, tmp_path / "many.madx", ref, linearize=True)
    note = [w for w in warnings if "linearised export:" in w]
    assert len(note) == 1
    assert "40 MATRIX element(s)" in note[0] and "entrance momentum" in note[0]
    p0c = ref.bg * PROTON.mass
    pt_total = float(note[0].split("= pt ")[1].split(" of")[0])
    gains = [float(w.split("ΔW=")[1].split(" MeV")[0]) for w in warnings
             if "reference energy gain" in w]
    assert len(gains) == 40
    assert pt_total == pytest.approx(sum(gains) / p0c, rel=1e-2)   # sum of the per-element kick6
    assert ("may fail" in note[0]) == (1e-2 <= abs(pt_total) < 0.3)
    assert ("will NOT reproduce" in note[0]) == (abs(pt_total) >= 0.3)
    # a single small cavity: quantitative note, no alarm wording
    lat2 = Lattice()
    lat2.add(Drift("d1", length=100.0))
    lat2.add(_rf_map())
    lat2.add(Drift("d2", length=100.0))
    (w,) = [w for w in write_madx(lat2, tmp_path / "one.madx", ref, linearize=True)
            if "linearised export:" in w]
    assert "may fail" not in w and "will NOT" not in w
