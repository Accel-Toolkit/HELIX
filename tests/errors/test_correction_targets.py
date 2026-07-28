# tests/errors/test_correction_targets.py
"""Target-aware orbit correction (steer-to-DIAG_POSITION set-points).

House rules honoured here:
* rule 2 — ``targets=None`` (the default) must be BIT-IDENTICAL to the
  historical flatten-to-zero behaviour;
* rule 4 — both regimes tested: steer-to-zero AND steer-to-recorded-
  orbit, plus the sentinel-disabled-plane branch.
"""
import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.marker import Marker
from linac_gen.elements.steerer import Steerer
from linac_gen.errors.correction import (apply_correction,
                                         apply_diagnostic_matching)
from linac_gen.tracking.tracker import Tracker


def _factory(n=300, offset_x=2.0, offset_y=1.0, seed=42):
    def factory():
        ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
        beam = Beam(ref=ref, n_particles=n, current=0.0)
        rng = np.random.default_rng(seed)
        beam.particles[:, 0] = rng.normal(offset_x, 0.5, n)
        beam.particles[:, 1] = rng.normal(0, 0.2, n)
        beam.particles[:, 2] = rng.normal(offset_y, 0.5, n)
        beam.particles[:, 3] = rng.normal(0, 0.2, n)
        return beam
    return factory


def _simple_lattice(x_t=None, y_t=None):
    lat = Lattice()
    lat.add(Steerer("STEER_1", bx_l=0.0, by_l=0.0))
    lat.add(Drift("D1", 500.0))
    lat.add(Marker("BPM_1", is_bpm=True,
                   diag_family=None if (x_t is None and y_t is None) else 1,
                   x_target_mm=x_t, y_target_mm=y_t))
    return lat


def _bpm_reading(lat, factory):
    rec = Tracker(lat, factory()).run()
    idx = next(i for i, e in enumerate(lat.elements)
               if getattr(e, "is_bpm", False))
    c = rec.centroid[rec.element_exit_idx[idx]]
    return float(c[0]), float(c[2])


def test_targets_none_bit_identical_to_explicit_zero():
    """targets=None ≡ targets={name: (0,0)} — identical kicks, seeded."""
    kicks = {}
    for key, tgt in (("none", None), ("zero", {"BPM_1": (0.0, 0.0)})):
        lat = _simple_lattice()
        apply_correction(lat, _factory(), method="svd", n_iter=2,
                         noise_seed=3, targets=tgt)
        s = next(e for e in lat.elements if isinstance(e, Steerer))
        kicks[key] = (s.bx_l, s.by_l)
    assert kicks["none"] == kicks["zero"]          # exact float equality


def test_one_to_one_steers_to_nonzero_target():
    lat = _simple_lattice()
    factory = _factory(offset_x=2.0, offset_y=1.0)
    apply_correction(lat, factory, method="one_to_one", n_iter=3,
                     targets={"BPM_1": (0.5, -0.3)})
    cx, cy = _bpm_reading(lat, factory)
    assert cx == pytest.approx(0.5, abs=0.1)
    assert cy == pytest.approx(-0.3, abs=0.1)


def test_svd_steers_to_nonzero_targets_multi_bpm():
    """Two steerers, two BPMs — guards the raw-response trap: had the
    response matrix been built from target-subtracted orbits, the solve
    would not land the readings on the targets."""
    lat = Lattice()
    lat.add(Steerer("STEER_1"))
    lat.add(Drift("D1", 300.0))
    lat.add(Marker("BPM_1", is_bpm=True))
    lat.add(Steerer("STEER_2"))
    lat.add(Drift("D2", 300.0))
    lat.add(Marker("BPM_2", is_bpm=True))
    factory = _factory(offset_x=1.5, offset_y=-1.0)
    apply_correction(lat, factory, method="svd", n_iter=3,
                     targets={"BPM_1": (0.4, 0.2), "BPM_2": (-0.2, 0.6)})
    rec = Tracker(lat, factory()).run()
    rows = [rec.element_exit_idx[i] for i, e in enumerate(lat.elements)
            if getattr(e, "is_bpm", False)]
    r1, r2 = (rec.centroid[r] for r in rows)
    assert r1[0] == pytest.approx(0.4, abs=0.1)
    assert r1[2] == pytest.approx(0.2, abs=0.1)
    assert r2[0] == pytest.approx(-0.2, abs=0.1)
    assert r2[2] == pytest.approx(0.6, abs=0.1)


def test_disabled_plane_left_free():
    """y target None (TW 1e50 sentinel) → the y plane is untouched."""
    lat = _simple_lattice()
    factory = _factory(offset_x=2.0, offset_y=1.0)
    apply_correction(lat, factory, method="svd", n_iter=2,
                     targets={"BPM_1": (0.0, None)})
    s = next(e for e in lat.elements if isinstance(e, Steerer))
    assert s.bx_l == 0.0                      # y-plane knob never moved
    cx, cy = _bpm_reading(lat, factory)
    assert cx == pytest.approx(0.0, abs=0.1)  # x corrected
    assert cy == pytest.approx(1.0, abs=0.2)  # y untouched (launch offset)


def test_deck_targets_via_apply_diagnostic_matching(tmp_path):
    """End-to-end through the parser: DIAG_POSITION targets + plain
    ADJUST family cards on THIN_STEERING drive the SVD special case."""
    from linac_gen.io.tracewin_parser import parse_tracewin
    dat = tmp_path / "diag.dat"
    dat.write_text(
        "FREQ 352.21\n"
        "ADJUST 5 2\n"                       # v=2 → by_l → x plane
        "T1: THIN_STEERING 0 0 50 0\n"
        "ADJUST 5 1\n"                       # v=1 → bx_l → y plane
        "T2: THIN_STEERING 0 0 50 0\n"
        "DRIFT 500 50 0\n"
        "B1: DIAG_POSITION 5 0.5 -0.3\n"
        "END\n")
    lat, meta = parse_tracewin(str(dat))
    assert not meta["warnings"]
    factory = _factory(offset_x=2.0, offset_y=1.0)
    res = apply_diagnostic_matching(lat, factory, n_iter=3)
    assert res["n_steerers"] == 2 and res["n_bpms"] == 1
    assert res["method"] == "svd"
    cx, cy = _bpm_reading(lat, factory)
    assert cx == pytest.approx(0.5, abs=0.1)
    assert cy == pytest.approx(-0.3, abs=0.1)


def test_families_none_excludes_passive_and_targetless_bpms(tmp_path):
    """families=None = 'families that have steerer ADJUSTs' — a passive
    family (no ADJUST) must not enter the solve, matching the
    constraints route's passive-monitor semantics."""
    from linac_gen.io.tracewin_parser import parse_tracewin
    dat = tmp_path / "diag.dat"
    dat.write_text(
        "FREQ 352.21\n"
        "ADJUST 5 2\n"
        "T1: THIN_STEERING 0 0 50 0\n"
        "DRIFT 400 50 0\n"
        "B1: DIAG_POSITION 5 0.5 -0.3\n"
        "B2: DIAG_POSITION 9 7.7 7.7\n"    # family 9: NO adjuster
        "BPM :\n"                          # target-less BPM card
        "END\n")
    lat, _ = parse_tracewin(str(dat))
    res = apply_diagnostic_matching(lat, _factory(), n_iter=1)
    assert res["n_bpms"] == 1              # only the family-5 monitor


def test_diagnostic_matching_skips_quad_families(tmp_path):
    """A family whose ADJUSTs bind a QUAD (fnalscl-style) is NOT consumed
    by the linear driver — that case belongs to the matching engine."""
    from linac_gen.io.tracewin_parser import parse_tracewin
    dat = tmp_path / "diag.dat"
    dat.write_text(
        "FREQ 352.21\n"
        "ADJUST 5 2\n"
        "Q1: QUAD 80 5.0 50 0\n"
        "DRIFT 500 50 0\n"
        "B1: DIAG_POSITION 5 0.5 -0.3\n"
        "END\n")
    lat, _ = parse_tracewin(str(dat))
    res = apply_diagnostic_matching(lat, _factory(), n_iter=1)
    assert res["kicks"] == {} and res["n_steerers"] == 0


def _beam_config(cx=2.0, cy=1.0):
    from linac_gen.core.config import BeamConfig
    return BeamConfig(species="proton", energy=3.0, frequency=352.21,
                      current=0.0, n_particles=300,
                      distribution="waterbag",
                      emit_nx=0.25, alpha_x=0.0, beta_x=2.0,
                      emit_ny=0.25, alpha_y=0.0, beta_y=2.0,
                      emit_z=0.3, alpha_z=0.0, beta_z=10.0,
                      centroid_x=cx, centroid_y=cy)


def test_envelope_backend_steers_to_target():
    """Envelope readings (deterministic, no particles) drive the same
    solve — and land EXACTLY on the target, no sampling noise."""
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.tracking.envelope import EnvelopeSolver
    lat = _simple_lattice()
    cfg = _beam_config()
    apply_correction(lat, beam_factory=None, method="svd", n_iter=2,
                     reading_backend="envelope", beam_config=cfg,
                     targets={"BPM_1": (0.5, -0.3)})
    ref = ReferenceParticle(species=cfg_species(cfg), w_kin=3.0,
                            frequency=352.21)
    res = EnvelopeSolver(lat, ref,
                         dict(alpha_x=0.0, beta_x=2.0, emit_x=1.0,
                              alpha_y=0.0, beta_y=2.0, emit_y=1.0,
                              alpha_z=0.0, beta_z=10.0, emit_z=0.3,
                              centroid=[2.0, 0, 1.0, 0, 0, 0]),
                         current=0.0).run()
    idx = next(i for i, e in enumerate(lat.elements)
               if getattr(e, "is_bpm", False))
    c = res.centroid[res.element_exit_idx[idx]]
    assert c[0] == pytest.approx(0.5, abs=1e-6)
    assert c[2] == pytest.approx(-0.3, abs=1e-6)


def cfg_species(cfg):
    from linac_gen.core.particle import PROTON
    return PROTON


def test_backend_equivalence_mp_vs_envelope():
    """Both backends steer the same lattice to the same kicks (within
    the MP backend's sampling noise) — dual-regime rule."""
    kicks = {}
    for backend in ("mp", "envelope"):
        lat = _simple_lattice()
        cfg = _beam_config()
        apply_correction(
            lat,
            beam_factory=(_factory(n=20000, offset_x=2.0, offset_y=1.0)
                          if backend == "mp" else None),
            method="svd", n_iter=2,
            reading_backend=backend,
            beam_config=cfg if backend == "envelope" else None,
            targets={"BPM_1": (0.5, -0.3)})
        s = next(e for e in lat.elements if isinstance(e, Steerer))
        kicks[backend] = (s.bx_l, s.by_l)
    assert kicks["mp"] == pytest.approx(kicks["envelope"], abs=2e-5)


def test_envelope_backend_requires_beam_config():
    with pytest.raises(ValueError, match="beam_config"):
        apply_correction(_simple_lattice(), beam_factory=None,
                         reading_backend="envelope")


def test_backend_equivalence_on_accelerating_lattice():
    """Review RISK closure: the two backends use DIFFERENT rigidities
    (mp: legacy exit, envelope: entrance) — benign only because brho
    cancels exactly in the kick algebra (probe and apply share it).
    The magnetostatic toy above cannot catch a regression in that
    cancellation; here the beam accelerates between steerer and BPM,
    so any rigidity-semantics breakage shows up as diverging kicks."""
    from linac_gen.elements.rf_gap import RFGap
    kicks = {}
    for backend in ("mp", "envelope"):
        lat = Lattice()
        lat.add(Steerer("STEER_1", bx_l=0.0, by_l=0.0))
        lat.add(Drift("D1", 250.0))
        for k in range(3):
            lat.add(RFGap(f"GAP_{k}", voltage=1.5, phase=-30.0,
                          frequency=352.21))
            lat.add(Drift(f"DG_{k}", 100.0))
        lat.add(Marker("BPM_1", is_bpm=True, diag_family=1,
                       x_target_mm=0.4, y_target_mm=-0.2))
        cfg = _beam_config()
        apply_correction(
            lat,
            beam_factory=(_factory(n=20000, offset_x=2.0, offset_y=1.0)
                          if backend == "mp" else None),
            method="svd", n_iter=2,
            reading_backend=backend,
            beam_config=cfg if backend == "envelope" else None,
            targets={"BPM_1": (0.4, -0.2)})
        s = next(e for e in lat.elements if isinstance(e, Steerer))
        kicks[backend] = (s.bx_l, s.by_l)
    assert kicks["mp"] == pytest.approx(kicks["envelope"], abs=5e-5)
    assert any(abs(v) > 1e-6 for v in kicks["envelope"])   # genuinely steered


def test_error_study_envelope_backend_end_to_end():
    """Review gap closure: nothing exercised ErrorStudy with
    reading_backend='envelope' — pins the per-seed hand-over
    (beam_config passed alongside the errored lattice copy) and that
    the correction genuinely engages (non-trivial kicks recorded)."""
    from linac_gen.elements.lattice_commands import AdjustSteerer
    from linac_gen.elements.quadrupole import Quadrupole
    from linac_gen.errors.error_model import ErrorStudy
    lat = Lattice()
    # ErrorStudy corrections are driven by ADJUST_STEERER cards (the
    # card's diag_n names the BPM; the partner steerer is the next
    # Steerer after the card — TraceWin convention).
    lat.add(AdjustSteerer("ADJ_1", diag_n=1))
    lat.add(Steerer("STEER_1", bx_l=0.0, by_l=0.0))
    lat.add(Drift("D1", 250.0))
    lat.add(Quadrupole("QUAD_1", 100.0, 8.0))
    lat.add(Drift("D2", 250.0))
    lat.add(Marker("BPM_1", is_bpm=True))
    study = ErrorStudy(lattice=lat, beam_config=_beam_config(),
                       n_seeds=2)
    study.add_error(pattern="QUAD*", parameter="dx",
                    distribution="gaussian", sigma=0.5)
    study.enable_correction(method="svd", n_iter=1,
                            reading_backend="envelope")
    res = study.run()

    def _flat(v):
        if isinstance(v, dict):
            for x in v.values():
                yield from _flat(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                yield from _flat(x)
        elif isinstance(v, (int, float)):
            yield float(v)

    for seed in range(2):
        kicks = res.corrected_kicks(seed)
        assert kicks, f"seed {seed}: correction did not run"
        vals = list(_flat(kicks))
        assert vals and any(abs(x) > 1e-9 for x in vals), \
            f"seed {seed}: correction produced all-zero kicks: {kicks}"


def test_error_study_enable_correction_stores_targets():
    from linac_gen.errors.error_model import ErrorStudy
    study = ErrorStudy.__new__(ErrorStudy)     # kwargs plumbing only
    study.enable_correction(method="svd", targets="deck")
    assert study._correction_kwargs["targets"] == "deck"
    assert study._correction_enabled
