"""NCELLS import / round-trip / tracking on real + synthetic decks.

The headline regression: TraceWin ``NCELLS`` cards used to import as skipped
"unknown card" warnings; now they build first-class :class:`NCells` elements
that track and round-trip.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.io.tracewin_writer import write_tracewin
from linac_gen.elements.ncells import NCells
from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle

FNALSCL = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "examples", "piplattice", "fnalscl.dat"))
_HAVE_FNALSCL = os.path.isfile(FNALSCL)


# ----------------------------------------------------------- synthetic deck
def _synthetic_deck(tmp_path):
    """A minimal absolute-phase π-mode NCELLS deck (2 cavities + a drift)."""
    dat = tmp_path / "nc.dat"
    dat.write_text(
        "FREQ 804.96\n"
        "NCELLS 1 16 0.4566 6.82062e6 62 15 1 0 0 0 0\n"
        "DRIFT 300 20\n"
        "NCELLS 1 16 0.4703 6.65229e6 242 15 1 0 0 0 0\n"
        "END\n"
    )
    return str(dat)


def test_synthetic_import_builds_ncells(tmp_path):
    lat, meta = parse_tracewin(_synthetic_deck(tmp_path))
    ncells = [e for e in lat.elements if isinstance(e, NCells)]
    assert len(ncells) == 2
    assert not any("NCELLS" in w or "unknown" in w.lower()
                   for w in meta["warnings"])
    e = ncells[0]
    assert e.mode == 1 and e.n_cells == 16 and e.p_flag == 1
    assert e.eot_v_per_m == pytest.approx(6.82062e6)
    assert e.frequency_mhz == pytest.approx(804.96)


def test_synthetic_roundtrip_preserves_operands(tmp_path):
    lat, _ = parse_tracewin(_synthetic_deck(tmp_path))
    out = tmp_path / "rt.dat"
    write_tracewin(lat, str(out))
    lat2, _ = parse_tracewin(str(out))
    a = [e for e in lat.elements if isinstance(e, NCells)]
    b = [e for e in lat2.elements if isinstance(e, NCells)]
    assert len(a) == len(b) == 2
    for x, y in zip(a, b):
        for f in ("mode", "n_cells", "beta_g", "eot_v_per_m", "theta_s_deg",
                  "aperture", "p_flag", "k_eot_i", "k_eot_o", "dz_i_mm", "dz_o_mm"):
            assert float(getattr(x, f)) == pytest.approx(float(getattr(y, f)))


def test_ttf_tail_roundtrips(tmp_path):
    """A βs≠0 transit-time tail survives write → re-parse."""
    dat = tmp_path / "ttf.dat"
    dat.write_text(
        "FREQ 804.96\n"
        "NCELLS 1 5 0.6579 1.1e7 -12 100 0 0.18 0.29 0 0 "
        "0.6 0.85 0.1 0.02 0.80 0.05 0.01 0.82 0.06 0.015\n"
        "END\n"
    )
    lat, _ = parse_tracewin(str(dat))
    nc = [e for e in lat.elements if isinstance(e, NCells)][0]
    assert nc._ttf is not None and nc.beta_s == pytest.approx(0.6)
    out = tmp_path / "ttf_rt.dat"
    write_tracewin(lat, str(out))
    lat2, _ = parse_tracewin(str(out))
    nc2 = [e for e in lat2.elements if isinstance(e, NCells)][0]
    assert nc2._ttf is not None
    assert nc2.beta_s == pytest.approx(0.6)
    assert nc2._ttf.middle.Ts == pytest.approx(0.85)
    assert nc2._ttf.output.k2Tpp == pytest.approx(0.015)


def test_matched_section_accelerates_monotonically(tmp_path):
    """A short absolute-phase section, injected at its βg-matched energy with a
    phase that puts the first cavity near crest, accelerates monotonically."""
    beta = 0.4566
    w0 = (1.0 / np.sqrt(1 - beta * beta) - 1.0) * H_MINUS.mass
    # two identical cavities one drift apart; θs=0 = crest for the |q| convention
    dat = tmp_path / "sec.dat"
    dat.write_text(
        "FREQ 804.96\n"
        "NCELLS 1 16 0.4566 6.82062e6 0 15 0 0 0 0 0\n"   # P=0 (relative), crest
        "DRIFT 100 20\n"
        "NCELLS 1 16 0.4566 6.82062e6 0 15 0 0 0 0 0\n"
        "END\n"
    )
    lat, _ = parse_tracewin(str(dat))
    ref = ReferenceParticle(species=H_MINUS, w_kin=w0, frequency=804.96)
    energies = [ref.w_kin]
    for e in lat.elements:
        if isinstance(e, NCells):
            e.reset_run_state()
            e.advance_ref(ref)
            energies.append(ref.w_kin)
        elif hasattr(e, "length") and e.length > 0:
            ref.s += e.length
            if ref.beta > 0 and ref.wavelength > 0:
                ref.phi_s += 360.0 * e.length / (ref.beta * ref.wavelength)
    gains = np.diff(energies)
    assert np.all(gains > 0), gains          # every cavity accelerates
    assert energies[-1] > energies[0] + 5.0  # meaningful net gain (MeV)


# -------------------------------------------------------- real fnalscl deck
@pytest.mark.skipif(not _HAVE_FNALSCL, reason="fnalscl.dat not present")
def test_fnalscl_imports_all_cavities_without_warnings():
    lat, meta = parse_tracewin(FNALSCL)
    ncells = [e for e in lat.elements if isinstance(e, NCells)]
    # 30 accelerating cavities before the deck's END marker (the 31st card
    # sits after END and is correctly excluded).
    assert len(ncells) == 30
    assert not any("NCELLS" in w or "unknown" in w.lower()
                   for w in meta["warnings"])
    # all πmode, absolute phase, HB650 βg ramp
    assert all(e.mode == 1 and e.p_flag == 1 for e in ncells)
    assert ncells[0].beta_g < ncells[-1].beta_g       # β increases downstream


@pytest.mark.skipif(not _HAVE_FNALSCL, reason="fnalscl.dat not present")
def test_fnalscl_roundtrips_operands():
    lat, _ = parse_tracewin(FNALSCL)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "rt.dat")
        write_tracewin(lat, out)
        lat2, _ = parse_tracewin(out)
    a = [e for e in lat.elements if isinstance(e, NCells)]
    b = [e for e in lat2.elements if isinstance(e, NCells)]
    assert len(a) == len(b) == 30
    for x, y in zip(a, b):
        assert x.mode == y.mode and x.n_cells == y.n_cells
        assert x.beta_g == pytest.approx(y.beta_g)
        assert x.eot_v_per_m == pytest.approx(y.eot_v_per_m)
        assert x.theta_s_deg == pytest.approx(y.theta_s_deg)


@pytest.mark.skipif(not _HAVE_FNALSCL, reason="fnalscl.dat not present")
def test_fnalscl_energy_matches_tracewin():
    """GROUND TRUTH: tracking fnalscl (H⁻, 116.1 MeV, absolute-phase NCELLS)
    reproduces the TraceWin ENV+SC reference run — the beam accelerates to
    ~404.8 MeV and the transverse envelope stays bounded (rms < 15 mm), which
    only holds with the validated |q| + (phi_s − θs) absolute-phase convention.

    Longitudinal Twiss: TraceWin's displayed α_z (−0.50) is in its z-like
    convention; HELIX's Δφ runs opposite to z (late particle: Δφ>0, z<0), so
    ⟨Δφ·ΔW⟩ = −⟨z·δ⟩ and the SAME physical beam needs α_z = +0.50 here.  With
    the wrong sign σ_φ collapses to ~3.8° (spuriously damped synchrotron
    oscillation) instead of TraceWin's sustained ~7°.
    """
    from linac_gen.tracking.envelope import EnvelopeSolver
    lat, _ = parse_tracewin(FNALSCL)
    ref = ReferenceParticle(species=H_MINUS, w_kin=116.1, frequency=804.6)
    bg = ref.bg
    initial = dict(alpha_x=-0.49, beta_x=9.03, emit_x=0.9 / bg,
                   alpha_y=0.35, beta_y=1.89, emit_y=0.8 / bg,
                   alpha_z=+0.50, beta_z=61.219316, emit_z=0.9079413)
    res = EnvelopeSolver(lat, ref, initial, current=23.7).run()
    sx = np.array(res.sigma_x)          # already mm
    sy = np.array(res.sigma_y)
    sphi = np.array(res.sigma_phi)      # deg
    # TraceWin: 116.1 -> 404.803 MeV; rms_x/y max ~5.6/4.6 mm.  The tight
    # tolerance is deliberate: it gates the H⁻ ion mass (the old proton-mass
    # bug gives 404.631 — outside) as well as the phase conventions.
    assert res.ref_w_kin[-1] == pytest.approx(404.803, abs=0.05)
    assert sx.max() < 15.0 and sy.max() < 15.0        # bounded, no blow-up
    assert np.all(np.diff(res.ref_w_kin) >= -1.0)     # monotone (bunchers aside)
    # Longitudinal: TraceWin σ_φ max 7.83°, final 7.14° (sustained synchrotron
    # oscillation).  The α_z sign regression damps it to 3.8° final.
    assert 7.0 < sphi.max() < 9.0
    assert sphi[-1] > 4.5
