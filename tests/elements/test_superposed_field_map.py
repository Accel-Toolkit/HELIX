# tests/elements/test_superposed_field_map.py
"""SuperposedFieldMap — TraceWin SUPERPOSE_MAP cluster container.

Anchor strategy (external-anchor house rule — no round-trip
cancellation):

* IDENTITY: a 1-child z0=0 container must reproduce the plain FieldMap
  bit-for-bit (same slicing, same kernel path).
* SPLIT-MAP: a smooth profile is split into two files with
  complementary linear ramps on COINCIDENT grid nodes — the sum of the
  two linear interpolants equals the original interpolant EXACTLY at
  every z, so the container tracking a genuine 2-child overlap must
  reproduce the single-map result to integrator round-off.
* Analytic: half-strength × 2 fully-overlapped ≡ full-strength.
"""
import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.field_map import FieldMap
from linac_gen.elements.superposed_field_map import SuperposedFieldMap
from linac_gen.io.field_map_reader import FieldMapData


def _ref(w_kin=3.0, freq=352.21, phi_s=0.0):
    return ReferenceParticle(species=PROTON, w_kin=w_kin,
                             frequency=freq, phi_s=phi_s)


def _beam(n=16, seed=7, **kw):
    b = Beam(ref=_ref(**kw), n_particles=n, current=0.0)
    rng = np.random.default_rng(seed)
    b.particles[:, 0] = rng.normal(0.0, 2.0, n)      # x mm
    b.particles[:, 1] = rng.normal(0.0, 1.0, n)      # x' mrad
    b.particles[:, 2] = rng.normal(0.0, 2.0, n)
    b.particles[:, 3] = rng.normal(0.0, 1.0, n)
    b.particles[:, 4] = rng.normal(0.0, 5.0, n)      # dphi deg
    b.particles[:, 5] = rng.normal(0.0, 0.01, n)     # dW MeV
    return b


def _track_full(elem, beam):
    n = max(elem.n_steps, 1)
    ds = elem.length / n
    elem.reset_run_state()
    for _ in range(n):
        elem.track_rk4(beam, ds)
    return beam


def _rf_bump(length=120.0, nz=121, ez=2.5):
    z = np.linspace(0.0, length, nz)
    return FieldMapData(z=z, Ez=ez * np.sin(np.pi * z / length),
                        symmetry="1d")


def _static_sol(length=300.0, nz=151, bz=0.4):
    z = np.linspace(0.0, length, nz)
    prof = bz * (0.5 - 0.5 * np.cos(2 * np.pi * z / length))
    return FieldMapData(z=z, Bz=prof, symmetry="1d")


# ── identity anchors ─────────────────────────────────────────────────────
def test_single_child_z0_zero_identity_mp():
    fd = _rf_bump()
    plain = FieldMap("FM", length=120.0, field_data=fd,
                     phase=-25.0, frequency=352.21, n_steps=120)
    child = FieldMap("FMc", length=120.0, field_data=fd,
                     phase=-25.0, frequency=352.21, n_steps=120)
    cont = SuperposedFieldMap("SUP", [(0.0, child)])
    assert cont.length == plain.length
    assert cont.n_steps == plain.n_steps

    b1 = _track_full(plain, _beam())
    b2 = _track_full(cont, _beam())
    assert np.array_equal(b1.particles, b2.particles)
    assert b1.ref.w_kin == b2.ref.w_kin
    assert b1.ref.phi_s == b2.ref.phi_s


def test_single_child_advance_ref_identity():
    fd = _rf_bump()
    plain = FieldMap("FM", length=120.0, field_data=fd,
                     phase=-25.0, frequency=352.21, n_steps=120)
    child = FieldMap("FMc", length=120.0, field_data=fd,
                     phase=-25.0, frequency=352.21, n_steps=120)
    cont = SuperposedFieldMap("SUP", [(0.0, child)])
    r1, r2 = _ref(), _ref()
    plain.reset_run_state()
    cont.reset_run_state()
    plain.advance_ref(r1)
    cont.advance_ref(r2)
    assert r1.w_kin == pytest.approx(r2.w_kin, rel=1e-14)
    assert r1.phi_s == pytest.approx(r2.phi_s, rel=1e-14)


def test_single_child_sync_phase_identity():
    """a′: the container-aware SET_SYNC_PHASE calibration must
    degenerate exactly to the standalone scan for a z0=0 single RF
    child."""
    fd = _rf_bump()
    plain = FieldMap("FM", length=120.0, field_data=fd,
                     phase=-30.0, frequency=352.21, n_steps=120, p_flag=1)
    child = FieldMap("FMc", length=120.0, field_data=fd,
                     phase=-30.0, frequency=352.21, n_steps=120, p_flag=1)
    cont = SuperposedFieldMap("SUP", [(0.0, child)])
    plain.reset_run_state()
    cont.reset_run_state()
    plain.advance_ref(_ref())
    cont.advance_ref(_ref())
    assert child._sync_offset_deg == pytest.approx(
        plain._sync_offset_deg, abs=0.01)


def test_grid_not_starting_at_zero_identity():
    """Pins the z_file arithmetic for maps whose file grid starts at
    z[0] != 0 (the _z_map_start offset must not double-apply)."""
    length = 100.0
    z = np.linspace(25.0, 25.0 + length, 101)
    fd = FieldMapData(z=z, Ez=1.5 * np.sin(np.pi * (z - 25.0) / length),
                      symmetry="1d")
    plain = FieldMap("FM", length=length, field_data=fd,
                     phase=0.0, frequency=352.21, n_steps=100)
    child = FieldMap("FMc", length=length, field_data=fd,
                     phase=0.0, frequency=352.21, n_steps=100)
    cont = SuperposedFieldMap("SUP", [(0.0, child)])
    b1 = _track_full(plain, _beam())
    b2 = _track_full(cont, _beam())
    assert np.array_equal(b1.particles, b2.particles)


# ── split-map reconstruction anchors ─────────────────────────────────────
def _split(profile_kw, length=150.0, nz=151, m1_frac=0.4, m2_frac=0.7):
    """Split a smooth profile into two 1-D files with complementary
    linear ramps on coincident nodes: interp(A) + interp(B) ≡
    interp(F) at EVERY z.

    Child B is PADDED with two leading zero-nodes so its np.gradient
    stencil (used by the paraxial Fr = −(r/2)·dFz/dz off-axis
    expansion) is central — not one-sided — at the ramp start,
    matching the full-grid derivative EXACTLY (np.gradient is linear,
    and the ramp's first two (1−w) values are zero anyway)."""
    z = np.linspace(0.0, length, nz)
    kind, F = profile_kw
    m1 = int(round(m1_frac * (nz - 1)))
    m2 = int(round(m2_frac * (nz - 1)))
    assert m1 >= 2
    w = np.ones(nz)
    w[m1:m2 + 1] = np.linspace(1.0, 0.0, m2 - m1 + 1)
    w[m2:] = 0.0

    pad = m1 - 2                       # B's file starts 2 nodes early
    kwA = {kind: F * w}
    kwB = {kind: (F * (1.0 - w))[pad:]}
    fdA = FieldMapData(z=z, symmetry="1d", **kwA)
    zB = z[pad:] - z[pad]
    fdB = FieldMapData(z=zB, symmetry="1d", **kwB)
    fdF = FieldMapData(z=z, symmetry="1d", **{kind: F})
    return fdA, fdB, fdF, float(z[pad]), length


@pytest.mark.parametrize("kind", ["Bz", "Ez"])
def test_split_map_reconstruction(kind):
    length = 150.0
    nz = 151
    z = np.linspace(0.0, length, nz)
    if kind == "Bz":
        F = 0.35 * (0.5 - 0.5 * np.cos(2 * np.pi * z / length))
        mk = dict(phase=0.0, frequency=0.0)
    else:
        F = 2.0 * np.sin(np.pi * z / length)
        mk = dict(phase=-20.0, frequency=352.21)
    fdA, fdB, fdF, z1, _ = _split((kind, F), length=length, nz=nz)

    single = FieldMap("FULL", length=length, field_data=fdF,
                      n_steps=nz - 1, **mk)
    chA = FieldMap("A", length=length, field_data=fdA,
                   n_steps=nz - 1, **mk)
    n_B = nz - 1 - int(round(z1 / (length / (nz - 1))))
    chB = FieldMap("B", length=length - z1, field_data=fdB,
                   n_steps=n_B, **mk)
    cont = SuperposedFieldMap("SUP", [(0.0, chA), (z1, chB)])
    assert cont.length == pytest.approx(length)
    assert cont.n_steps == nz - 1

    b1 = _track_full(single, _beam())
    b2 = _track_full(cont, _beam())
    assert np.allclose(b1.particles, b2.particles, atol=1e-10), \
        np.max(np.abs(b1.particles - b2.particles))
    assert b1.ref.w_kin == pytest.approx(b2.ref.w_kin, rel=1e-12)

    M1 = single.fitted_matrix(_ref())
    M2 = cont.fitted_matrix(_ref())
    assert np.allclose(M1, M2, atol=1e-8), np.max(np.abs(M1 - M2))


def test_split_map_n_steps_invariance():
    """(f): the reconstruction identity must hold at refined slicing —
    both sides use the same n, so agreement is slicing-independent."""
    length, nz = 150.0, 151
    z = np.linspace(0.0, length, nz)
    F = 2.0 * np.sin(np.pi * z / length)
    fdA, fdB, fdF, z1, _ = _split(("Ez", F), length=length, nz=nz)
    for mult in (2, 4):
        n = (nz - 1) * mult
        single = FieldMap("FULL", length=length, field_data=fdF,
                          phase=-20.0, frequency=352.21, n_steps=n)
        chA = FieldMap("A", length=length, field_data=fdA,
                       phase=-20.0, frequency=352.21, n_steps=n)
        chB = FieldMap("B", length=length - z1, field_data=fdB,
                       phase=-20.0, frequency=352.21, n_steps=10)
        cont = SuperposedFieldMap("SUP", [(0.0, chA), (z1, chB)],
                                  n_steps=n)
        b1 = _track_full(single, _beam())
        b2 = _track_full(cont, _beam())
        assert np.allclose(b1.particles, b2.particles, atol=1e-10)


def test_negative_z0_tracks_map_tail():
    """TraceWin: 'to start at the Z position in a field map, set
    Z0=−Z' — the pre-entrance part of the map is outside the span."""
    length, nz = 150.0, 151
    z = np.linspace(0.0, length, nz)
    F = 2.0 * np.sin(np.pi * z / length)
    fdF = FieldMapData(z=z, Ez=F, symmetry="1d")
    z1 = 45.0
    # Reference: a trimmed map whose grid is the tail of F, shifted to 0.
    keep = z >= z1 - 1e-9
    fd_tail = FieldMapData(z=z[keep] - z1, Ez=F[keep], symmetry="1d")
    n_tail = int(np.sum(keep)) - 1

    trimmed = FieldMap("TAIL", length=length - z1, field_data=fd_tail,
                       phase=-20.0, frequency=352.21, n_steps=n_tail)
    child = FieldMap("FULLC", length=length, field_data=fdF,
                     phase=-20.0, frequency=352.21, n_steps=nz - 1)
    cont = SuperposedFieldMap("SUP", [(-z1, child)])
    assert cont.length == pytest.approx(length - z1)

    b1 = _track_full(trimmed, _beam())
    b2 = _track_full(cont, _beam())
    # Energy (on-axis Fz interp) is exact; the transverse coords carry
    # a small REFERENCE-construction artifact: the trimmed comparison
    # map's np.gradient is one-sided at its first node while the
    # container samples the full map with central differences (the
    # container side is the more accurate one).
    assert b1.ref.w_kin == pytest.approx(b2.ref.w_kin, rel=1e-12)
    assert np.allclose(b1.particles, b2.particles, atol=5e-5), \
        np.max(np.abs(b1.particles - b2.particles))


# ── analytic overlap anchors ─────────────────────────────────────────────
def test_two_half_strength_solenoids_equal_one():
    fd_full = _static_sol(bz=0.4)
    fd_half = _static_sol(bz=0.2)
    single = FieldMap("SOL", length=300.0, field_data=fd_full, n_steps=150)
    c1 = FieldMap("H1", length=300.0, field_data=fd_half, n_steps=150)
    c2 = FieldMap("H2", length=300.0, field_data=fd_half, n_steps=150)
    cont = SuperposedFieldMap("SUP", [(0.0, c1), (0.0, c2)])
    b1 = _track_full(single, _beam())
    b2 = _track_full(cont, _beam())
    assert np.allclose(b1.particles, b2.particles, atol=1e-12)
    M1 = single.fitted_matrix(_ref())
    M2 = cont.fitted_matrix(_ref())
    assert np.allclose(M1, M2, atol=1e-9)


def test_two_half_ke_cavities_equal_one():
    fd = _rf_bump(ez=2.5)
    single = FieldMap("CAV", length=120.0, field_data=fd,
                      phase=-25.0, frequency=352.21, n_steps=120)
    c1 = FieldMap("K1", length=120.0, field_data=fd, ke=0.5,
                  phase=-25.0, frequency=352.21, n_steps=120)
    c2 = FieldMap("K2", length=120.0, field_data=fd, ke=0.5,
                  phase=-25.0, frequency=352.21, n_steps=120)
    cont = SuperposedFieldMap("SUP", [(0.0, c1), (0.0, c2)])
    b1 = _track_full(single, _beam())
    b2 = _track_full(cont, _beam())
    assert np.allclose(b1.particles, b2.particles, atol=1e-12)


# ── construction rules ───────────────────────────────────────────────────
def test_mixed_rf_frequencies_refused():
    fd = _rf_bump()
    c1 = FieldMap("C1", length=120.0, field_data=fd,
                  phase=0.0, frequency=162.5, n_steps=60)
    c2 = FieldMap("C2", length=120.0, field_data=fd,
                  phase=0.0, frequency=325.0, n_steps=60)
    with pytest.raises(ValueError, match="mixes RF frequencies"):
        SuperposedFieldMap("SUP", [(0.0, c1), (0.0, c2)])


def test_static_children_exempt_from_frequency_rule():
    sol = FieldMap("SOL", length=300.0, field_data=_static_sol(),
                   n_steps=150)
    cav = FieldMap("CAV", length=120.0, field_data=_rf_bump(),
                   phase=-25.0, frequency=162.5, n_steps=120)
    cont = SuperposedFieldMap("SUP", [(0.0, sol), (90.0, cav)])
    assert cont.effective_frequency == pytest.approx(162.5)
    assert cont.length == pytest.approx(300.0)


def test_span_aperture_and_carrier():
    sol = FieldMap("SOL", length=1000.0, field_data=_static_sol(1000.0),
                   aperture=100.0, n_steps=500)
    quad_fd = _static_sol(100.0, bz=0.1)
    quad = FieldMap("Q", length=100.0, field_data=quad_fd,
                    aperture=42.0, n_steps=50)
    cont = SuperposedFieldMap("SUP", [(400.0, quad), (0.0, sol)])
    assert cont.length == pytest.approx(1000.0)       # manual example
    assert cont.aperture == pytest.approx(100.0)      # z0==0 carrier
    assert cont.field_data is sol.field_data


def test_nonpositive_span_refused():
    fm = FieldMap("C", length=100.0, field_data=_rf_bump(100.0, 101),
                  phase=0.0, frequency=352.21)
    with pytest.raises(ValueError, match="non-positive span"):
        SuperposedFieldMap("SUP", [(-200.0, fm)])


# ── run-state hygiene ────────────────────────────────────────────────────
def test_reset_run_state_reproduces_exactly():
    fdA, fdB, fdF, z1, length = _split(
        ("Ez", 2.0 * np.sin(np.pi * np.linspace(0, 1, 151))), nz=151)
    chA = FieldMap("A", length=length, field_data=fdA,
                   phase=-20.0, frequency=352.21, n_steps=150, p_flag=1)
    chB = FieldMap("B", length=length - z1, field_data=fdB,
                   phase=-20.0, frequency=352.21, n_steps=100)
    cont = SuperposedFieldMap("SUP", [(0.0, chA), (z1, chB)])
    b1 = _track_full(cont, _beam())
    assert chA._sync_offset_deg is not None
    b2 = _track_full(cont, _beam())        # reset inside _track_full
    assert np.array_equal(b1.particles, b2.particles)
    cont.reset_run_state()
    assert cont._z_cursor == 0.0 and cont._z_history == []
    assert chA._sync_offset_deg is None    # child state cleared too


def test_fitted_matrix_slice_chain_matches_full():
    """Chained fitted_matrix_slice with the explicit z-cursor must
    compose to (approximately) the full-element matrix — the envelope
    SC bundle contract."""
    fd = _rf_bump()
    child = FieldMap("FMc", length=120.0, field_data=fd,
                     phase=-25.0, frequency=352.21, n_steps=120)
    cont = SuperposedFieldMap("SUP", [(0.0, child)])
    cont.reset_run_state()
    M_full = cont.fitted_matrix(_ref())

    cont.reset_run_state()
    ref = _ref()
    M = np.eye(6)
    n_bundles, ds = 4, 120.0 / 4
    z = 0.0
    for _ in range(n_bundles):
        M = cont.fitted_matrix_slice(ref, ds, _z_from_mm=z) @ M
        cont.advance_ref_over(ref, z, z + ds)
        z += ds
    assert np.allclose(M, M_full, rtol=2e-2, atol=1e-4)
