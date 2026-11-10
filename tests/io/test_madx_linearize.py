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
    txt = out.read_text(encoding="utf-8")
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
    that shift as kick6; the importer reads a gain as the reference's
    (``ref_gain``: the particles' energies are the same, and the reference
    follows them), and a later MATRIX converts at that energy (exact inverse
    of its map)."""
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
    kick6 = float(out.read_text(encoding="utf-8").split("kick6=")[1].split(",")[0])
    assert kick6 == pytest.approx(0.25 / p0c, rel=1e-12)
    back, meta = parse_madx(str(out))
    m1, m2 = _matrix_elements(back)
    np.testing.assert_allclose(m1.matrix, M, rtol=0, atol=1e-13)
    assert m1.offset is None and m1.ref_gain == pytest.approx(0.25, rel=1e-12)
    np.testing.assert_allclose(m2.matrix, lat.elements[2].matrix, rtol=0, atol=1e-13)
    assert m2.offset is None and m2.ref_gain == pytest.approx(0.0, abs=1e-12)
    assert any("change the energy (kick6)" in w for w in meta["warnings"])


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
    assert "kick" not in out.read_text(encoding="utf-8").split("MATRIX")[1].split(";")[0]
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
    exact and carries ΔW as the reference's gain."""
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
    from linac_gen.tracking.tracker import field_map_reference_states
    field_map_reference_states(lat, cav, r)      # the tracker's reference walk
    dW = r.w_kin - ref.w_kin
    assert dW != 0.0
    assert np.linalg.det(M[:2, :2]) != pytest.approx(1.0, abs=1e-6)   # adiabatic damping
    p0c = ref.bg * PROTON.mass
    kick6 = float(out.read_text(encoding="utf-8").split("kick6=")[1].split(",")[0])
    assert kick6 == pytest.approx(dW / p0c, rel=1e-12)
    # canonical form: the exit-angle rescaling removes the adiabatic
    # damping factor p_in/p_out from the x block (what remains, ~1e-5, is
    # the hard-edged box field's own non-Maxwellian / integration error).
    R = matrix_to_madx(M, ref, ref, r)
    assert abs(np.linalg.det(R[:2, :2]) - 1.0) < 1e-4
    assert abs(np.linalg.det(R[:2, :2]) - 1.0) < 0.05 * abs(np.linalg.det(M[:2, :2]) - 1.0)
    back, meta = parse_madx(str(out))
    assert any("change the energy (kick6)" in m for m in meta["warnings"])
    (me,) = _matrix_elements(back)
    np.testing.assert_allclose(me.matrix, M, rtol=0, atol=1e-13)
    assert me.ref_gain == pytest.approx(dW, rel=1e-12)
    assert me.offset is None


def test_linearize_does_not_touch_supported_elements(tmp_path):
    """With nothing to linearise, linearize=True is byte-identical."""
    from linac_gen.io.tracewin_parser import parse_tracewin
    lat = parse_tracewin("examples/fodo_cell.dat")[0]
    ref = _ref(H_MINUS, 3.0)
    a, b = tmp_path / "a.madx", tmp_path / "b.madx"
    kw = dict(sequence_name="fodo", title="fodo")
    assert write_madx(lat, a, ref, **kw) == []
    assert write_madx(lat, b, ref, linearize=True, **kw) == []
    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")


def test_thin_lens_is_linearised_but_foil_and_electric_steerer_are_not(tmp_path):
    ref = _ref()
    lat = Lattice()
    lat.add(ThinLens("tl", fx=2000.0, fy=-3000.0))
    lat.add(Drift("d", length=50.0))
    lat.add(Steerer("es", bx_l=1e-3, by_l=0.0, elec=True))
    lat.add(Foil("foil", thickness=1.0)) if _foil_ok() else None
    out = tmp_path / "mix.madx"
    warnings = write_madx(lat, out, ref, linearize=True)
    txt = out.read_text(encoding="utf-8")
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
    """pt is a deviation about the ENTRANCE momentum: MAD-X TWISS expands
    the drifts between the maps to second order about the orbit, its R56
    off by about 7.5·(pt/β₀)² (measured, see the writer).  A line whose gain
    puts that above 0.1 % says by how much, and above 10 % that MAD-X will
    not reproduce it; a small gain gets the quantitative note only."""
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
    err = 7.5 * (pt_total / ref.beta) ** 2
    assert ("may be off" in note[0]) == (1e-3 <= err < 0.1)
    assert ("will NOT reproduce" in note[0]) == (err >= 0.1)
    # gains whose estimate is a few % and well above 10 %
    for step, word in ((0.5, "may be off by about"), (5.0, "will NOT reproduce")):
        big = Lattice()
        big.add(MatrixElement("m", np.eye(6), length=100.0, offset=np.array([0, 0, 0, 0, 0, step])))
        (w_big,) = [w for w in write_madx(big, tmp_path / "big.madx", ref, linearize=True)
                    if "linearised export:" in w]
        assert word in w_big, (step, w_big)
        if step == 0.5:                       # the estimate itself
            pct = float(w_big.split("about ")[1].split(" %")[0])
            assert pct == pytest.approx(100.0 * 7.5 * (step / p0c / ref.beta) ** 2, rel=0.03)
    # a single small cavity: quantitative note, no alarm wording
    lat2 = Lattice()
    lat2.add(Drift("d1", length=100.0))
    lat2.add(_rf_map())
    lat2.add(Drift("d2", length=100.0))
    (w,) = [w for w in write_madx(lat2, tmp_path / "one.madx", ref, linearize=True)
            if "linearised export:" in w]
    assert "may be off" not in w and "will NOT" not in w


# ---------------------------------------------------------------------------
# after a frequency change: Δφ is in degrees of the clock at its own point
# ---------------------------------------------------------------------------
def _on_clock(ref, f):
    r = ref.copy()
    r.frequency = f
    return r


def test_the_basis_takes_the_clock_at_its_point():
    ref = _ref(f=162.5)
    T1 = madx_transform(ref)
    T2 = madx_transform(ref, wavelength_local_mm=_on_clock(ref, 325.0).wavelength)
    assert T2[4, 4] == pytest.approx(2.0 * T1[4, 4], rel=1e-15)
    assert np.delete(np.diag(T2), 4).tolist() == np.delete(np.diag(T1), 4).tolist()


def test_a_freq_card_is_nothing_in_madx_coordinates():
    """MAD-X's t is c·Δt and holds no clock, so the card's change of coordinate is
    the identity there: the reason the export can leave it as a comment."""
    ref = _ref(f=162.5)
    D = np.eye(6)
    D[4, 4] = 2.0
    R = matrix_to_madx(D, ref, ref.copy(), _on_clock(ref, 325.0))
    np.testing.assert_allclose(R, np.eye(6), rtol=0, atol=1e-15)


def test_a_static_map_reads_the_same_in_madx_on_either_clock():
    """A magnet does not know the clock: R56 = L/β²γ² on any of them."""
    ref = _ref(f=162.5)
    R = {}
    for f in (162.5, 325.0, 650.0):
        r = _on_clock(ref, f)
        fm = _solenoid_map()
        R[f] = matrix_to_madx(get_element_matrix(fm, r.copy()), ref, r, r)
    np.testing.assert_allclose(R[325.0], R[162.5], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(R[650.0], R[162.5], rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("f_in, f_out", [(162.5, 325.0), (325.0, 650.0), (325.0, 325.0)])
def test_the_conversions_are_inverses_across_clocks(f_in, f_out):
    ref = _ref(f=162.5)
    rng = np.random.default_rng(7)
    M = rng.normal(size=(6, 6))
    r_in, r_out = _on_clock(ref, f_in), _on_clock(ref, f_out)
    r_in.w_kin = 5.4
    r_out.w_kin = 6.1
    back = matrix_from_madx(matrix_to_madx(M, ref, r_in, r_out), ref, r_in, r_out)
    np.testing.assert_allclose(back, M, rtol=0, atol=1e-13)
    v = rng.normal(size=6)
    np.testing.assert_allclose(vector_from_madx(vector_to_madx(v, ref, r_out), ref, r_out),
                               v, rtol=0, atol=1e-13)


def test_on_one_clock_the_conversion_is_unchanged_to_the_last_bit():
    """What the code did before it took a clock, written out."""
    ref = _ref(f=162.5)
    rng = np.random.default_rng(11)
    M = rng.normal(size=(6, 6))
    r_in, r_out = ref.copy(), ref.copy()
    r_in.w_kin = 5.4
    r_out.w_kin = 6.1
    p0 = ref.bg * ref.species.mass

    def old(r):
        T = np.eye(6)
        T[0, 0] = T[2, 2] = 1000.0
        T[1, 1] = T[3, 3] = 1000.0 / ((r.bg * r.species.mass) / p0)
        T[4, 4] = -360.0 / (ref.wavelength / 1000.0)
        T[5, 5] = p0
        return T
    want = np.diag(1.0 / np.diag(old(r_out))) @ M @ old(r_in)
    assert matrix_to_madx(M, ref, r_in, r_out).tobytes() == want.tobytes()
    assert matrix_to_madx(M, ref).tobytes() == (np.diag(1.0 / np.diag(old(ref))) @ M @ old(ref)).tobytes()


@pytest.mark.parametrize("regime", ["static", "rf"])
def test_a_line_with_only_a_freq_card_comes_back_on_one_clock(tmp_path, regime):
    """MAD-X has no FREQ, so the imported line stays on the reference's clock, and its
    maps come back in that clock's degrees: the same physics, ``D_out⁻¹ M D_in``."""
    from linac_gen.elements.lattice_commands import Freq
    from tests.tracking.test_envelope_reference_matches_tracker import _ncells
    ref = _ref(f=162.5, w=20.0)
    lat = Lattice()
    lat.add(Drift("d1", length=100.0))
    lat.add(Freq("F", 325.0))
    lat.add(Drift("d2", length=100.0))
    lat.add(MatrixElement("mx", _symplectic_4x4(), length=250.0) if regime == "static" else _ncells())
    lat.add(Drift("d3", length=100.0))
    out = tmp_path / "lin.madx"
    write_madx(lat, out, ref, linearize=True)
    back, meta = parse_madx(str(out))
    M0 = compute_transfer_matrix(lat, ref.copy())
    ref_back = copy.deepcopy(meta["reference"])
    M1 = compute_transfer_matrix(back, ref_back.copy())
    D_in, D_out = np.eye(6), np.eye(6)
    D_in[4, 4] = ref_back.frequency / 162.5
    D_out[4, 4] = ref_back.frequency / 325.0
    want = D_out @ M0 @ np.linalg.inv(D_in)
    # the reference follows the gain (ref_gain), so the drifts after it too
    tol = 1e-12
    scale = np.maximum(np.abs(want), 1e-3)
    assert np.abs((M1 - want) / scale)[[0, 1, 2, 3], :][:, [0, 1, 2, 3]].max() < tol
    assert M1[4, 4] == pytest.approx(want[4, 4], rel=tol)
    assert M1[4, 5] == pytest.approx(want[4, 5], rel=tol)
    assert M1[5, 4] == pytest.approx(want[5, 4], rel=tol, abs=1e-12)


def test_a_matrix_after_a_cavity_at_another_frequency_comes_back_on_its_clock(tmp_path):
    """An RFCAVITY does move the clock of the imported line, so a MATRIX after one is
    in that clock's degrees, as it was written."""
    from linac_gen.elements.rf_gap import RFGap
    ref = _ref(f=162.5, w=20.0)
    M = _symplectic_4x4()
    M[4, 1] = -0.3
    M[0, 5] = 0.02
    lat = Lattice()
    lat.add(Drift("d1", length=100.0))
    lat.add(RFGap("g1", voltage=0.5, phase=-90.0, frequency=162.5))
    lat.add(Drift("d2", length=100.0))
    lat.add(RFGap("g2", voltage=0.5, phase=-90.0, frequency=325.0))
    lat.add(Drift("d3", length=100.0))
    lat.add(MatrixElement("mx", M, length=250.0))
    lat.add(Drift("d4", length=100.0))
    out = tmp_path / "lin.madx"
    write_madx(lat, out, ref, linearize=True)
    back, meta = parse_madx(str(out))
    (me,) = _matrix_elements(back)
    np.testing.assert_allclose(me.matrix, M, rtol=0, atol=1e-13)
    np.testing.assert_allclose(compute_transfer_matrix(back, copy.deepcopy(meta["reference"])),
                               compute_transfer_matrix(lat, ref.copy()), rtol=1e-11, atol=1e-11)


# ---------------------------------------------------------------------------
# The clock where the importer has lost the energy
# ---------------------------------------------------------------------------
def _parsed(tmp_path, body, energy=20.0):
    deck = tmp_path / "x.madx"
    deck.write_text(f"beam, particle=proton, energy={(energy + PROTON.mass) / 1000.0!r};\n" + body)
    lat, meta = parse_madx(str(deck))
    return [e for e in lat.elements if isinstance(e, MatrixElement)][0], meta


def _m45_at(f, rm56=0.5, w=20.0):
    ref = ReferenceParticle(species=PROTON, w_kin=w, frequency=352.21)
    at = ReferenceParticle(species=PROTON, w_kin=w, frequency=f)
    R = np.eye(6)
    R[4, 5] = rm56
    return matrix_from_madx(R, ref, at, at)[4, 5]


CAVITIES = """c1: rfcavity, l=0, volt=0, lag=0, freq=162.5;
c2: rfcavity, l=0, volt=0, lag=0, freq=325;
"""
MATRIX = "mx: matrix, l=1, rm11=1, rm22=1, rm33=1, rm44=1, rm55=1, rm66=1, rm56=0.5"


def test_a_map_after_a_cavity_with_no_frequency_keeps_the_clock(tmp_path):
    """The energy after that cavity is unknown, and the clock is not."""
    body = CAVITIES + "c3: rfcavity, l=0, volt=1, lag=0;\n" + MATRIX + """;
s: sequence, refer=entry, l=5;
 c1, at=0; c2, at=1; c3, at=2; mx, at=3;
endsequence;
use, sequence=s;
"""
    el, _meta = _parsed(tmp_path, body)
    assert el.matrix[4, 5] == pytest.approx(_m45_at(325.0), rel=1e-13)


def test_a_map_whose_own_kick_loses_the_reference_keeps_the_clock(tmp_path):
    body = CAVITIES + MATRIX + """, kick6=-5;
s: sequence, refer=entry, l=5;
 c1, at=0; c2, at=1; mx, at=3;
endsequence;
use, sequence=s;
"""
    el, _meta = _parsed(tmp_path, body)
    assert el.matrix[4, 5] == pytest.approx(_m45_at(325.0), rel=1e-13)


def _linearised_ncells(tmp_path, nc, w=20.0):
    """Export drift + ``nc`` + drift linearised and read it back.  Returns the export's
    note on the cavity, the map read back (== matrix mode's) and the multiparticle
    tracker's central-difference Jacobian, all on the importer's clock (352.21 MHz,
    so no clock factor enters)."""
    import warnings

    from linac_gen.core.beam import Beam
    from linac_gen.tracking.tracker import Tracker
    lat = Lattice()
    for e in (Drift("d0", 100.0), nc, Drift("d1", 80.0)):
        lat.add(e)
    ref = _ref(w=w, f=352.21)
    out = tmp_path / "nc.madx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        msgs = write_madx(lat, out, ref, linearize=True)
        back, meta = parse_madx(str(out))
        M_back = compute_transfer_matrix(back, meta["reference"].copy())
        M_matrix_mode = compute_transfer_matrix(lat, ref.copy())
    np.testing.assert_allclose(M_back, M_matrix_mode, rtol=1e-12, atol=1e-12)
    h = np.array([1e-4, 1e-5, 1e-4, 1e-5, 1e-3, 1e-6])
    beam = Beam(ref.copy(), n_particles=12, current=0.0)
    beam.particles[:] = 0.0
    for k in range(6):
        beam.particles[2 * k, k], beam.particles[2 * k + 1, k] = h[k], -h[k]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Tracker(copy.deepcopy(lat), beam).run()
    J = np.column_stack([(beam.particles[2 * k] - beam.particles[2 * k + 1]) / (2 * h[k]) for k in range(6)])
    note = [m for m in msgs if m.startswith(f"'{nc.name}' (NCells): linearised as MAD-X MATRIX")]
    assert len(note) == 1
    return note[0], M_back, J


@pytest.mark.parametrize("synchronism", [True, False])
def test_a_linearised_ncells_is_matrix_modes_map(tmp_path, synchronism):
    """A linearised NCELLS is matrix mode's map, which carries TraceWin's matrix-engine
    synchronism factor; the multiparticle tracker, like TraceWin's PARTRAN, applies none.
    The export says so.  With the factor off, the map read back is the tracker's
    Jacobian."""
    from tests.tracking.test_envelope_reference_matches_tracker import _ncells
    nc = _ncells(frequency=352.21)
    nc.synchronism = synchronism
    note, M_back, J = _linearised_ncells(tmp_path, nc)
    if synchronism:
        assert "matrix mode's map, with TraceWin's matrix-engine synchronism factor" in note
        assert ("setting the cavity's synchronism attribute to False in Python gives the "
                "tracker's map") in note
        assert abs(M_back[5, 5] - J[5, 5]) > 1e-2       # the factor: 0.795 against 0.779
    else:
        assert "synchronism" not in note
        np.testing.assert_allclose(M_back, J, rtol=1e-7, atol=1e-8)


_SKIPS = {                              # (cavity, entrance energy in MeV)
    "beta_g 0": (dict(beta_g=0.0), 20.0),
    "transit-time table": (dict(beta_g=0.2, ttf="table"), 20.0),
    "no field": (dict(beta_g=0.2, eot_v_per_m=0.0), 20.0),
    "below beta_g/2": (dict(beta_g=0.2), 3.8),
    "past the clamp": (dict(beta_g=0.2), 5.2),
}


@pytest.mark.parametrize("case", list(_SKIPS))
def test_a_linearised_ncells_the_factor_skips_is_the_trackers_map(tmp_path, case):
    """The factor is skipped, with the flag on, for βg <= 0, a βs != 0 transit-time
    table, a cavity with no field, a beam below βg/2 and one past the factor's clamp.
    There the note says nothing of it, and the map read back is the tracker's
    Jacobian."""
    from linac_gen.elements.ncells import NCells, TTFSet, TTFTable
    kw, w = dict(_SKIPS[case][0]), _SKIPS[case][1]
    if kw.get("ttf") == "table":
        t = TTFSet(Ts=0.8, kTp=0.1, k2Tpp=0.0)
        kw["ttf"] = TTFTable(beta_s=0.2, middle=t, input=t, output=t)
    kw.setdefault("eot_v_per_m", 2.0e6)
    nc = NCells("NC", mode=1, n_cells=6, theta_s_deg=-30.0, aperture_mm=15.0, p_flag=1,
                frequency_mhz=352.21, **kw)
    assert nc.synchronism
    note, M_back, J = _linearised_ncells(tmp_path, nc, w=w)
    assert "synchronism" not in note
    np.testing.assert_allclose(M_back, J, rtol=1e-6, atol=1e-7)


def _quad_map(name="QFM", **misalignment):
    """A 200 mm 3-D map of a 1 T/m quadrupole field (Bx = G y, By = G x)."""
    n, nz, L = 5, 21, 200.0
    x = np.linspace(-10.0, 10.0, n)
    z = np.linspace(0.0, L, nz)
    X, Y, _ = np.meshgrid(x, x, z, indexing="ij")
    G = 1e-3                                             # T/mm
    fd = FieldMapData(z=z, frequency=0.0)
    fd.channels[Channel.STAT_B] = FieldChannel(geometry=7, x=x, y=x.copy(), z=z,
                                               Fx=G * Y, Fy=G * X, Fz=np.zeros_like(X))
    return FieldMap3D(name=name, length=L, field_data=fd, n_steps=50, **misalignment)


@pytest.mark.parametrize("attr, value", [("tilt_deg", 20.0), ("dx", 0.5)])
def test_a_linearised_misaligned_map_is_written_aligned_and_warned(tmp_path, attr, value):
    """A misaligned map is written as the element aligned, with the warning every other
    element gives; it went without one.  Matrix mode tilts the map, so the tilt was lost
    silently."""
    import warnings
    ref = _ref(w=20.0, f=352.21)
    lats = []
    for mis in ({attr: value}, {}):
        lat = Lattice()
        for e in (Drift("d0", 100.0), _quad_map(**mis), Drift("d1", 80.0)):
            lat.add(e)
        lats.append(lat)
    out = tmp_path / "q.madx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        msgs = write_madx(lats[0], out, ref, linearize=True)
        back, meta = parse_madx(str(out))
    assert [m for m in msgs if "misalignment" in m] == [
        f"'QFM': misalignment {attr}={value:g} has no MAD-X element attribute (use EALIGN) "
        "— exported aligned"]
    M_back = compute_transfer_matrix(back, meta["reference"].copy())
    M_aligned = compute_transfer_matrix(lats[1], ref.copy())
    np.testing.assert_allclose(M_back, M_aligned, rtol=1e-12, atol=1e-12)
    coupling = np.abs(compute_transfer_matrix(lats[0], ref.copy())[0:2, 2:4]).max()
    assert coupling == pytest.approx(0.198, abs=1e-3) if attr == "tilt_deg" else coupling == 0.0
    assert np.abs(M_aligned[0:2, 2:4]).max() == 0.0


def test_a_linearised_curved_cluster_keeps_its_own_tilt(tmp_path):
    """A curved field-map cluster moves its own maps, so its map holds its tilt, which
    matrix mode leaves to it: the MATRIX keeps the tilt, and nothing is warned of."""
    import warnings

    from tests.elements.test_curved_field_map import FREQ, W, _sector
    cl = _sector(30.0)
    cl.tilt_deg = 5.0
    lat = Lattice()
    for e in (Drift("d0", 50.0), cl, Drift("d1", 100.0)):
        lat.add(e)
    ref = ReferenceParticle(species=PROTON, w_kin=W, frequency=FREQ)
    out = tmp_path / "c.madx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        msgs = write_madx(lat, out, ref, linearize=True)
        back, meta = parse_madx(str(out))
    assert not [m for m in msgs if "misalignment" in m], msgs
    M = compute_transfer_matrix(lat, ref.copy())
    M_back = compute_transfer_matrix(back, meta["reference"].copy())
    assert np.abs(M[0:2, 2:4]).max() > 0.1                  # the tilt couples x and y
    np.testing.assert_allclose(M_back, M, rtol=1e-9, atol=1e-9)
