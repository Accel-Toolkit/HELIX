"""End-to-end phase-advance cross-check against TraceWin exports.

Two tiers, both gated on the presence of the (never-committed) TraceWin
reference data under ``Tracewin_code/mebtplushwr/``:

* FAST — a *convention* pin.  Feed TraceWin's OWN β(s) columns into
  HELIX's phase-advance integrator (``_integrate_inverse_beta_split``)
  and confirm it reproduces TraceWin's reported RMS phase advance to a
  fraction of a degree.  This locks the ∫ds/β convention (contiguous
  trapz, (180/π)·1e0 rad→deg) to the reference, independent of HELIX's
  own tracking.
* SLOW (``@pytest.mark.slow``) — run HELIX's envelope on
  ``examples/mebt_plus_hwr.dat`` and check the machine RMS phase advance
  and tune depression land near the measured anchors.

Reference anchors (user's TraceWin run, 162.5 MHz, ~2.12 MeV injection):
    I = 0    : μ_x = 1304.8°, μ_y = 1375.4°
    5 mA env : μ_x = 1030.2°, μ_y = 1091.8°
    machine η_x = μ_x(5 mA)/μ_x(0) ≈ 0.79
"""
from __future__ import annotations

from tests.dataguard import needs, require  # noqa: E402

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
TW_DIR = REPO / "Tracewin_code" / "mebtplushwr"
EXAMPLE_DAT = REPO / "examples" / "mebt_plus_hwr.dat"

# TraceWin envelope-export column layout (header:
# position gam-1 centroid(10) rms_size(10) dispX dispY betX betY):
_COL_S, _COL_BETX, _COL_BETY = 0, 24, 25

# (file, anchor μ_x, anchor μ_y) — the projected-β RMS phase advance
# TraceWin itself reports for this line.
_TW_CASES = [
    ("nospacecharge_MP_env.txt", 1304.8, 1375.4),      # I = 0
    ("envenvelopewithspacecharge.txt", 1030.2, 1091.8),  # 5 mA
]

_needs_tw = pytest.mark.skipif(
    not TW_DIR.is_dir(),
    reason="TraceWin reference data (Tracewin_code/) not present — never committed",
)


def _load_tw_beta(path: Path) -> np.ndarray:
    """(s, betX, betY) rows from a TraceWin envelope export."""
    rows = []
    with open(path) as fh:
        fh.readline()                                   # header
        for line in fh:
            parts = line.split()
            if len(parts) <= _COL_BETY:
                continue
            try:
                rows.append((float(parts[_COL_S]),
                             float(parts[_COL_BETX]),
                             float(parts[_COL_BETY])))
            except ValueError:
                continue
    return np.asarray(rows, dtype=float)


@_needs_tw
@pytest.mark.parametrize("fname,mu_x_ref,mu_y_ref", _TW_CASES)
def test_integrator_reproduces_tracewin_phase_advance(fname, mu_x_ref, mu_y_ref):
    """HELIX's ∫ds/β integrator, fed TraceWin's own β(s), reproduces
    TraceWin's RMS phase advance to < 0.5°."""
    from linac_gen.analysis.phase_advance import _integrate_inverse_beta_split

    data = _load_tw_beta(TW_DIR / fname)
    assert data.shape[0] > 100, f"{fname}: too few rows parsed"
    s = data[:, 0]
    deg = 180.0 / np.pi

    bx = data[:, 1]
    by = data[:, 2]
    mu_x = _integrate_inverse_beta_split(s, bx, bx > 0) * deg
    mu_y = _integrate_inverse_beta_split(s, by, by > 0) * deg

    assert mu_x == pytest.approx(mu_x_ref, abs=0.5), (mu_x, mu_x_ref)
    assert mu_y == pytest.approx(mu_y_ref, abs=0.5), (mu_y, mu_y_ref)


@_needs_tw
def test_integrator_matches_plain_trapz():
    """The contiguous-run integrator equals a single trapz for a
    fully-valid β span (no fabricated phase across gaps here)."""
    from linac_gen.analysis.phase_advance import _integrate_inverse_beta_split

    data = _load_tw_beta(TW_DIR / _TW_CASES[0][0])
    s, bx = data[:, 0], data[:, 1]
    m = bx > 0
    split = _integrate_inverse_beta_split(s, bx, m)
    plain = np.trapz(1.0 / bx[m], s[m])
    assert split == pytest.approx(plain, rel=1e-12)


# ---------------------------------------------------------------------------
# SLOW end-to-end HELIX run on examples/mebt_plus_hwr.dat.
#
# Canonical matched input (== compare_mebt_hwr_sc.build_cfg): H- at
# 2.1226695 MeV, 162.5 MHz, with the rms-derived input Twiss and
# NATIVE-unit longitudinal Twiss (deg·MeV / deg-per-MeV).  Feeding a
# mm·mrad longitudinal Twiss instead corrupts the 3-D SC form factors and
# erases the transverse depression, so the units below are load-bearing.
#
# Machine RMS phase advance uses WHOLE-LINE, BOUNDARY-ONLY sampling
# (record_substeps=False, the default): projected β_x inside the 8 HWR
# solenoid interiors is not a normal-mode tune, so substep sampling
# over-counts the whole-line μ by +14-20% (see the projected_only guard).
# Measured HELIX vs TraceWin (boundary-only): I=0 μ_x 1303.6 (-0.1%),
# μ_y 1388.3 (+0.9%); 5 mA μ_x 1010.5 (-1.9%), μ_y 1104.2 (+1.1%);
# η_x 0.775 vs TW 0.790.  A 5% tolerance brackets all with >2.5× margin.
# ---------------------------------------------------------------------------

_ANCHORS = {0.0: (1304.8, 1375.4), 5.0: (1030.2, 1091.8)}


def _canonical_cfg(current: float):
    from linac_gen.core.config import BeamConfig
    return BeamConfig(
        species="H-", energy=2.1226695, frequency=162.5,
        current=current, duty_cycle=100.0,
        n_particles=64, distribution="gaussian", cutoff=4.0,
        emit_nx=0.21, alpha_x=1.228, beta_x=0.316,
        emit_ny=0.21, alpha_y=-0.095394, beta_y=0.113,
        emit_z=0.06231832, alpha_z=0.0, beta_z=819.05492,
    )


def _run_machine_mu(current: float):
    """Whole-line boundary-only RMS phase advance (deg) via HELIX's own
    envelope β(s) and integrator."""
    from linac_gen.analysis.phase_advance import _integrate_inverse_beta_split
    from linac_gen.distributions.factory import create_beam
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.tracking.envelope import EnvelopeSolver

    lat, meta = parse_tracewin(str(EXAMPLE_DAT))
    # The deck's ERROR_CAV_NCPL_stat card uses r=0 (constant amplitude),
    # which HELIX approximates — since the honesty round (2026-07-19)
    # that downgrade is REPORTED instead of silently swallowed.  It is
    # the only expected warning; anything else is a regression.
    unexpected = [w for w in meta["warnings"]
                  if "r=0 (constant amplitude)" not in w]
    assert unexpected == [], unexpected

    cfg = _canonical_cfg(current)
    beam = create_beam(cfg, seed=42)
    bg = beam.ref.bg
    initial = dict(
        alpha_x=cfg.alpha_x, beta_x=cfg.beta_x, emit_x=cfg.emit_nx / bg,
        alpha_y=cfg.alpha_y, beta_y=cfg.beta_y, emit_y=cfg.emit_ny / bg,
        alpha_z=cfg.alpha_z, beta_z=cfg.beta_z, emit_z=cfg.emit_z,
    )
    res = EnvelopeSolver(lat, beam.ref, initial, current=current).run()

    # res.s is in mm but res.beta is in m — convert s to metres so ds/β
    # is dimensionless (the TraceWin anchor uses metres throughout).
    s = np.asarray(res.s, dtype=float) * 1e-3
    bx = np.asarray(res.beta_x, dtype=float)
    by = np.asarray(res.beta_y, dtype=float)
    deg = 180.0 / np.pi
    mu_x = _integrate_inverse_beta_split(s, bx, bx > 0) * deg
    mu_y = _integrate_inverse_beta_split(s, by, by > 0) * deg
    return mu_x, mu_y


@pytest.mark.slow
@pytest.mark.skipif(not EXAMPLE_DAT.exists(),
                    reason="example lattice missing")
@needs("examples/mebt_plus_hwr.dat", "examples/piplattice/fnalscl.lgproj")
def test_helix_machine_phase_advance_matches_tracewin():
    """End-to-end: HELIX's envelope + integrator reproduce TraceWin's
    machine RMS phase advance on MEBT+HWR at I=0 and 5 mA within 5%, with
    the correct tune depression."""
    mu = {}
    for current in (0.0, 5.0):
        mu_x, mu_y = _run_machine_mu(current)
        mu[current] = (mu_x, mu_y)
        ref_x, ref_y = _ANCHORS[current]
        assert mu_x == pytest.approx(ref_x, rel=0.05), (current, mu_x, ref_x)
        assert mu_y == pytest.approx(ref_y, rel=0.05), (current, mu_y, ref_y)

    # Tune depression η_x = μ_x(5 mA)/μ_x(0): TraceWin 0.790, HELIX ~0.775.
    eta_x = mu[5.0][0] / mu[0.0][0]
    assert eta_x == pytest.approx(0.790, rel=0.05), eta_x
    assert 0.70 < eta_x < 0.85            # genuine depression, not ~1


@pytest.mark.slow
@pytest.mark.skipif(not EXAMPLE_DAT.exists(),
                    reason="example lattice missing")
@needs("examples/mebt_plus_hwr.dat", "examples/piplattice/fnalscl.lgproj")
def test_substep_sampling_overshoots_through_solenoids():
    """Regression pin for the projected-β-in-solenoid artifact: substep
    sampling over-counts the whole-line phase advance vs boundary-only by
    >10% (localized to the HWR solenoid interiors), and beam_phase_advance
    flags it via projected_only."""
    from linac_gen.analysis.period_detect import detect_periods
    from linac_gen.analysis.phase_advance import (
        _integrate_inverse_beta_split, beam_phase_advance,
    )
    from linac_gen.distributions.factory import create_beam
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.tracking.envelope import EnvelopeSolver

    lat, _ = parse_tracewin(str(EXAMPLE_DAT))
    cfg = _canonical_cfg(0.0)
    beam = create_beam(cfg, seed=42)
    bg = beam.ref.bg
    initial = dict(
        alpha_x=cfg.alpha_x, beta_x=cfg.beta_x, emit_x=cfg.emit_nx / bg,
        alpha_y=cfg.alpha_y, beta_y=cfg.beta_y, emit_y=cfg.emit_ny / bg,
        alpha_z=cfg.alpha_z, beta_z=cfg.beta_z, emit_z=cfg.emit_z,
    )
    deg = 180.0 / np.pi

    def _mu_x(record_substeps):
        res = EnvelopeSolver(lat, beam.ref.copy(), initial, current=0.0,
                             record_substeps=record_substeps).run()
        s = np.asarray(res.s, float) * 1e-3      # mm → m (β is in m)
        bx = np.asarray(res.beta_x, float)
        return _integrate_inverse_beta_split(s, bx, bx > 0) * deg, res

    mu_boundary, _ = _mu_x(False)
    mu_substep, res_sub = _mu_x(True)
    assert mu_substep > mu_boundary * 1.10   # solenoid-interior overshoot

    # The beam-integral helper flags the coupled interior on the substep run.
    pa = beam_phase_advance(res_sub, detect_periods(lat)[0])
    for key in ("mu_x_deg", "matched", "projected_only",
                "n_samples", "resolution_ok"):
        assert key in pa


# ── NCELLS per-cavity matrix anchor (2026-07-21 synchronism round) ───────
_PIPTW = REPO / "Tracewin_code" / "piplatticetracewin"
_needs_piptw = pytest.mark.skipif(
    not (_PIPTW / "Transfer_matrix1.dat").is_file(),
    reason="TraceWin piplattice matrix export not present — never committed",
)


@_needs_piptw
@pytest.mark.slow
@needs("examples/mebt_plus_hwr.dat", "examples/piplattice/fnalscl.lgproj")
def test_ncells_longitudinal_matrices_match_tracewin():
    """Every NCELLS cavity's longitudinal 2x2 must match TraceWin's own
    exported per-element matrices to the sub-0.5% level — the anchor
    that pinned (and now guards) the geometric synchronism factor.
    Without it the diagonals were off by up to 1.8%."""
    import re
    import warnings
    from types import SimpleNamespace

    from linac_gen.io.project import load_project
    from linac_gen.cli.common import load_lattice, build_ref
    from linac_gen.tracking.matrix_tracking import get_element_matrix
    from linac_gen.elements.base import FieldMapElement, ThinKickElement
    from linac_gen.elements.lattice_commands import Freq
    from linac_gen.elements.ncells import NCells

    MC2 = 939.294308
    LAM = 299792458.0 / 804.96e6
    proj = load_project(str(REPO / "examples/piplattice/fnalscl.lgproj"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat = load_lattice(proj.lattice_path)
        ref = build_ref(proj.beam)
        cav = []
        s = 0.0
        for elem in lat.elements:
            L = float(getattr(elem, "length", 0.0) or 0.0)
            if isinstance(elem, NCells):
                elem.reset_run_state()
                Mh = get_element_matrix(elem, ref.copy())
                cav.append((s / 1e3, (s + L) / 1e3,
                            np.array(Mh[np.ix_([4, 5], [4, 5])])))
            if isinstance(elem, Freq):
                elem.apply_command(SimpleNamespace(ref=ref, beam=None))
            elif isinstance(elem, FieldMapElement):
                elem.reset_run_state()
                elem.advance_ref(ref)
            else:
                ref.s += L
                if L > 0:
                    ref.phi_s += 360.0 * L / (ref.beta * ref.wavelength)
                if isinstance(elem, ThinKickElement):
                    elem.advance_ref(ref)
            s += L
    assert ref.w_kin == pytest.approx(404.803, abs=0.01)   # walk sanity

    mats, s_at = [], []
    with open(_PIPTW / "Transfer_matrix1.dat") as fh:
        rows = []
        for line in fh:
            m = re.match(r"\s*ELE#\s*(\d+)\s*:\s*([-\d.eE+]+)\s*m", line)
            if m:
                s_at.append(float(m.group(2))); rows = []
                continue
            if line.strip():
                rows.append([float(v) for v in line.split()])
                if len(rows) == 6:
                    mats.append(np.array(rows)); rows = []
    s_at = np.asarray(s_at)
    twe = np.genfromtxt(_PIPTW / "MP+SC.txt", skip_header=1)

    ratios = {k: [] for k in ("M54", "M55", "M65", "M66")}
    for s_in, s_out, Mh in cav:
        j = int(np.argmin(np.abs(s_at - s_out)))
        if abs(s_at[j] - s_out) > 0.005 or j == 0:
            continue
        Mtw = (mats[j] @ np.linalg.inv(mats[j - 1]))[np.ix_([4, 5], [4, 5])]
        g_i = 1.0 + np.interp(s_in, twe[:, 0], twe[:, 1])
        g_o = 1.0 + np.interp(s_out, twe[:, 0], twe[:, 1])
        b_i = np.sqrt(1 - 1 / g_i**2); b_o = np.sqrt(1 - 1 / g_o**2)
        Ti = np.diag([-360.0 / (b_i * LAM), b_i**2 * g_i * MC2])
        To = np.diag([-360.0 / (b_o * LAM), b_o**2 * g_o * MC2])
        Mc = To @ Mtw @ np.linalg.inv(Ti)
        ratios["M54"].append(Mh[0, 1] / Mc[0, 1])
        ratios["M55"].append(Mh[0, 0] / Mc[0, 0])
        ratios["M65"].append(Mh[1, 0] / Mc[1, 0])
        ratios["M66"].append(Mh[1, 1] / Mc[1, 1])
    assert len(ratios["M65"]) == 30
    for k, v in ratios.items():
        v = np.asarray(v)
        assert abs(v.mean() - 1.0) < 0.002, (k, v.mean())
        assert v.std() < 0.005, (k, v.std())
