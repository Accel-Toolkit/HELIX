"""Regressions for two 2026-07 error-machinery fixes.

1. ``*_rel`` errors on elements that carry the perturbation slot itself
   (FieldMap / FieldMap3D ``voltage_rel``, Dipole ``field_rel``) used to be
   SILENTLY DROPPED: ``_apply_errors`` stripped the suffix and required a
   base attribute (``voltage`` / ``field``) those elements don't have.
   Every ERROR_CAV amplitude error on an SRF field-map cavity and every
   ERROR_BEND dg error was a no-op — tolerance studies ran with zero RF
   amplitude jitter.  Now the draw lands on the ``_rel`` slot, which
   tracking consumes via ``effective_ke/kb`` / ``effective_angle``.

2. ``ADJUST_STEERER_BX/_BY`` plane masks were computed but never passed to
   ``apply_correction`` (both planes always corrected) — and the mask
   assignment itself was inverted vs the matcher's canonical mapping
   (a BX card authorizes the ``bx_l`` knob, which kicks y′).
"""
from __future__ import annotations

from tests.dataguard import needs, require  # noqa: E402

import warnings

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import (
    AdjustSteerer, AdjustSteererBx, AdjustSteererBy,
)
from linac_gen.elements.marker import Marker
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.steerer import Steerer
from linac_gen.errors.correction import run_correction_from_lattice
from linac_gen.errors.error_model import ErrorDef, ErrorStudy


# ---------------------------------------------------------------------
# 1. _rel errors on slot-carrying elements
# ---------------------------------------------------------------------
def _study_with(lattice, pattern, parameter, sigma=0.10):
    study = ErrorStudy(lattice, BeamConfig(), n_seeds=1)
    study._errors.append(ErrorDef(pattern=pattern, parameter=parameter,
                                  sigma=sigma, distribution="gaussian",
                                  cutoff=3.0))
    return study


@pytest.fixture(scope="module")
def hwr_lattice():
    """Real PIP-II lattice with both FieldMap (1-D) and FieldMap3D."""
    from linac_gen.io.tracewin_parser import parse_tracewin
    lat, _ = parse_tracewin("examples/pipii/mebt+hwr/mebt+hwr.dat")
    return lat


@needs("examples/pipii/mebt+hwr/mebt+hwr.dat")
def test_voltage_rel_perturbs_fieldmap_and_fieldmap3d(hwr_lattice):
    from linac_gen.elements.field_map import FieldMap
    from linac_gen.elements.field_map_3d import FieldMap3D

    fm = next(e for e in hwr_lattice.elements if isinstance(e, FieldMap))
    fm3 = next(e for e in hwr_lattice.elements if isinstance(e, FieldMap3D))
    for el in (fm, fm3):
        study = _study_with(hwr_lattice, el.name, "voltage_rel")
        perturbed = study._apply_errors(seed=7)
        pe = next(e for e in perturbed.elements if e.name == el.name)
        # The slot itself moved …
        assert pe.voltage_rel != el.voltage_rel, \
            f"voltage_rel error still a no-op on {type(el).__name__}"
        # … and so did the amplitude tracking actually consumes.  A pure
        # magnetic map (solenoid-type: ke=0) is exercised through kb, an
        # accelerating map through ke — check whichever channel is live.
        assert el.ke != 0.0 or el.kb != 0.0, "degenerate field map in fixture"
        if el.ke != 0.0:
            assert pe.effective_ke != pytest.approx(el.effective_ke), \
                f"effective_ke unchanged on {type(el).__name__}"
        if el.kb != 0.0:
            assert pe.effective_kb != pytest.approx(el.effective_kb), \
                f"effective_kb unchanged on {type(el).__name__}"


def test_field_rel_perturbs_dipole_effective_angle():
    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    lat.add(Dipole(name="BEND_1", angle=15.0, rho=800.0))
    study = _study_with(lat, "BEND_1", "field_rel")
    perturbed = study._apply_errors(seed=3)
    pd = next(e for e in perturbed.elements if e.name == "BEND_1")
    assert pd.field_rel != 0.0
    assert pd.effective_angle != pytest.approx(15.0)


def test_gradient_rel_legacy_multiply_unchanged():
    """The pre-existing path (base attribute present) must keep the exact
    multiply-the-design-slot behaviour the test suite depends on."""
    lat = Lattice()
    lat.add(Quadrupole(name="QUAD_1", length=200.0, gradient=12.0))
    study = _study_with(lat, "QUAD_1", "gradient_rel", sigma=0.50)
    perturbed = study._apply_errors(seed=12345)
    q = next(e for e in perturbed.elements if e.name == "QUAD_1")
    assert q.gradient != 12.0                     # applied …
    assert getattr(q, "gradient_rel", 0.0) == 0.0  # … via the base slot


def test_mistargeted_rel_error_warns_once():
    """An error that can land nowhere must warn (once), never silently
    vanish."""
    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    study = _study_with(lat, "D1", "voltage_rel")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        study._apply_errors(seed=1)
        study._apply_errors(seed=2)   # second seed: no duplicate warning
    hits = [w for w in rec if "NOT applied" in str(w.message)]
    assert len(hits) == 1


# ---------------------------------------------------------------------
# 2. Plane masks in orbit correction
# ---------------------------------------------------------------------
def _factory(offset_x=2.0, offset_y=1.0, n=300, seed=42):
    def f():
        ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
        beam = Beam(ref=ref, n_particles=n, current=0.0)
        rng = np.random.default_rng(seed)
        beam.particles[:, 0] = rng.normal(offset_x, 0.5, n)
        beam.particles[:, 1] = rng.normal(0, 0.2, n)
        beam.particles[:, 2] = rng.normal(offset_y, 0.5, n)
        beam.particles[:, 3] = rng.normal(0, 0.2, n)
        return beam
    return f


def _card_lattice(card_cls):
    lat = Lattice()
    lat.add(Drift("D_pre", 100.0))
    lat.add(card_cls("ADJ_1", diag_n=1, vmax=0.0, first_step=1e-4))
    lat.add(Steerer("STEER_1", bx_l=0.0, by_l=0.0))
    lat.add(Drift("D_post", 100.0))
    lat.add(Marker("BPM_1", is_bpm=True))
    return lat


def _run(card_cls, method=None):
    lat = _card_lattice(card_cls)
    run_correction_from_lattice(lat, _factory(), n_iter=1,
                                override_method=method)
    steer = next(e for e in lat.elements if isinstance(e, Steerer))
    return steer


def test_bx_card_touches_only_bx_l():
    """ADJUST_STEERER_BX authorizes the Bx knob → y-plane only.  With an
    offset in BOTH planes, by_l must remain exactly zero."""
    steer = _run(AdjustSteererBx)
    assert steer.by_l == 0.0, "BX card must not drive the by_l (x-plane) knob"
    assert steer.bx_l != 0.0, "BX card should correct the y plane"


def test_by_card_touches_only_by_l():
    steer = _run(AdjustSteererBy)
    assert steer.bx_l == 0.0, "BY card must not drive the bx_l (y-plane) knob"
    assert steer.by_l != 0.0, "BY card should correct the x plane"


def test_plain_card_corrects_both_planes():
    steer = _run(AdjustSteerer)
    assert steer.bx_l != 0.0 and steer.by_l != 0.0


def test_plane_mask_enforced_in_svd_path_too():
    steer = _run(AdjustSteererBx, method="svd")
    assert steer.by_l == 0.0
    assert steer.bx_l != 0.0


# ---------------------------------------------------------------------
# 3. Truncated-gaussian draws + fnmatch-safe element names
# ---------------------------------------------------------------------
def test_truncated_gaussian_redraws_instead_of_clipping():
    """Clipping piled ~16 % of draws (cutoff 1σ) onto EXACTLY ±cutoff·σ;
    the redraw convention (TraceWin) never lands exactly on the limit."""
    from linac_gen.errors.error_model import _draw_truncated_gaussian

    rng = np.random.default_rng(7)
    lim = 1.0 * 1.0
    draws = [_draw_truncated_gaussian(rng, 1.0, 1.0) for _ in range(5000)]
    assert all(abs(v) <= lim for v in draws)
    assert not any(abs(v) == lim for v in draws), \
        "mass piled exactly on ±cutoff·σ — still clipping, not redrawing"
    # cutoff <= 0 keeps the legacy zero-draw semantics.
    assert _draw_truncated_gaussian(rng, 1.0, 0.0) == 0.0


def test_error_directive_matches_element_with_glob_metachars():
    """An element literally named 'Q[1]' must receive its own errors —
    the exact name is used as an fnmatch pattern downstream and was not
    escaped, so it never matched itself."""
    from linac_gen.io.tracewin_error_parsing import TraceWinErrorState

    state = TraceWinErrorState()
    #                       N  r   dx  dy φx φy φz  dG g3 g4 g5 g6
    state.add_quad_directive([1, 2, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    state.on_element("QUAD", "Q[1]")
    assert state.errors, "directive must have produced an ErrorDef"

    lat = Lattice()
    lat.add(Quadrupole(name="Q[1]", length=100.0, gradient=10.0))
    lat.errors = list(state.errors)
    study = ErrorStudy(lat, BeamConfig(), n_seeds=1)
    perturbed = study._apply_errors(seed=11)
    q = perturbed.elements[0]
    assert q.dx != 0.0, \
        "dx error on 'Q[1]' silently no-oped (fnmatch metacharacters)"
