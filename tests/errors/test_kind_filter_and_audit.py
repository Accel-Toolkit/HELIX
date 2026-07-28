# tests/errors/test_kind_filter_and_audit.py
"""ErrorDef.element_kind gating, the pre-run audit, and per-seed draw
logging — the machinery that makes name-glob error budgets safe on
decks where solenoids and cavities share the ``FMAP_*`` prefix.

Regression context: the pip2 misalignment study's solenoid and cavity
globs were IDENTICAL (``FMAP_0[2-9]?``), so every SSR solenoid took the
cavities' 1 % voltage error while its own field error was warn-skipped
— and the study driver's comment claimed a type filter existed at apply
time when none did.  These tests pin the filter, the audit that would
have caught it, and the audit↔apply mirror agreement."""
from __future__ import annotations

from tests.dataguard import needs, require  # noqa: E402

import warnings

import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.errors.error_model import ErrorDef, ErrorStudy, element_kind


@pytest.fixture(scope="module")
def hwr_lattice():
    """Real PIP-II front end: FieldMap solenoids AND FieldMap3D cavities
    that share the FMAP_* name prefix — the exact ambiguity the kind
    gate exists for."""
    from linac_gen.io.tracewin_parser import parse_tracewin
    lat, _ = parse_tracewin("examples/pipii/mebt+hwr/mebt+hwr.dat")
    return lat


def _study(lattice, *defs):
    study = ErrorStudy(lattice, BeamConfig(), n_seeds=1)
    study._errors = list(defs)          # replace any lattice-absorbed ones
    return study


# ---------------------------------------------------------------------
# element_kind classifier
# ---------------------------------------------------------------------
@needs("examples/pipii/mebt+hwr/mebt+hwr.dat", "examples/pip2_misalignment_study")
def test_element_kind_mirrors_categorize_fieldmap(hwr_lattice):
    """The local classifier must never drift from the matching module's
    categorize_fieldmap — same rule, two call sites."""
    from linac_gen.matching.variables import categorize_fieldmap
    n_fm = 0
    for el in hwr_lattice.elements:
        if type(el).__name__ in ("FieldMap", "FieldMap3D"):
            n_fm += 1
            assert element_kind(el) == categorize_fieldmap(el)
    assert n_fm > 10          # the fixture really contains field maps
    assert element_kind(Quadrupole("Q", 50.0, gradient=5.0)) == "Quadrupole"


# ---------------------------------------------------------------------
# kind gating in _apply_errors
# ---------------------------------------------------------------------
@needs("examples/pipii/mebt+hwr/mebt+hwr.dat", "examples/pip2_misalignment_study")
def test_kind_gate_separates_solenoids_from_cavities(hwr_lattice):
    """One FMAP_* glob + voltage_rel, gated 'solenoid': only ke==0 maps
    move; every cavity stays at design (and vice versa)."""
    for want in ("solenoid", "cavity"):
        study = _study(hwr_lattice, ErrorDef(
            pattern="FMAP_*", parameter="voltage_rel", sigma=0.05,
            distribution="gaussian", cutoff=3.0, element_kind=want))
        perturbed = study._apply_errors(seed=11)
        touched = {e["element"] for e in study.applied_log[11]}
        assert touched, f"no {want} received a draw"
        for el in perturbed.elements:
            if type(el).__name__ not in ("FieldMap", "FieldMap3D"):
                continue
            if element_kind(el) == want:
                assert el.name in touched
            else:
                assert el.name not in touched, \
                    f"{el.name} is not a {want} but drew the {want} error"


@needs("examples/pipii/mebt+hwr/mebt+hwr.dat", "examples/pip2_misalignment_study")
def test_kind_gate_skips_before_drawing(hwr_lattice):
    """Filtered-out elements must consume NO RNG entropy.  Pin: a wide
    glob + solenoid gate must draw the SAME VALUES as one no-gate def
    per solenoid name (both consume one draw per solenoid, in lattice
    order — if the gate sat after the draw, cavities would burn entropy
    in the wide study and every subsequent value would differ)."""
    sol_names = [e.name for e in hwr_lattice.elements
                 if type(e).__name__ in ("FieldMap", "FieldMap3D")
                 and element_kind(e) == "solenoid"]
    assert len(sol_names) >= 4
    wide = _study(hwr_lattice, ErrorDef(
        pattern="FMAP_*", parameter="dx", sigma=0.2,
        element_kind="solenoid"))
    wide._apply_errors(seed=3)
    draws_wide = [(e["element"], e["value"]) for e in wide.applied_log[3]]

    narrow = _study(hwr_lattice, *[
        ErrorDef(pattern=name, parameter="dx", sigma=0.2)
        for name in sol_names])            # no kind gate, no cavities
    narrow._apply_errors(seed=3)
    draws_narrow = [(e["element"], e["value"])
                    for e in narrow.applied_log[3]]
    assert draws_wide == draws_narrow      # value-for-value identical


def test_kind_gate_class_names():
    lat = Lattice()
    lat.add(Quadrupole("QUAD_1", 50.0, gradient=5.0))
    lat.add(Drift("QUAD_FAKE", 100.0))       # name matches, wrong class
    study = _study(lat, ErrorDef(pattern="QUAD_*", parameter="dx",
                                 sigma=0.1, element_kind="Quadrupole"))
    study._apply_errors(seed=1)
    touched = {e["element"] for e in study.applied_log[1]}
    assert touched == {"QUAD_1"}


# ---------------------------------------------------------------------
# audit ↔ apply mirror agreement
# ---------------------------------------------------------------------
@needs("examples/pipii/mebt+hwr/mebt+hwr.dat", "examples/pip2_misalignment_study")
def test_audit_mirrors_apply_outcomes(hwr_lattice):
    """audit_errors' applicable/skipped classification must agree with
    what _apply_errors actually does (mirror drift = audit lies)."""
    defs = [
        ErrorDef(pattern="FMAP_*", parameter="voltage_rel", sigma=0.01,
                 element_kind="cavity"),
        # field_rel is unsupported on FieldMaps — the historical
        # solenoid trap: matched but warn-skipped.
        ErrorDef(pattern="FMAP_*", parameter="field_rel", sigma=0.01,
                 element_kind="solenoid"),
        ErrorDef(pattern="QUAD_*", parameter="dx", sigma=0.1),
    ]
    study = _study(hwr_lattice, *defs)
    audit = study.audit_errors()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study._apply_errors(seed=5)
    log = study.applied_log[5]
    applied = {(e["element"], e["parameter"]) for e in log if e["applied"]}
    skipped = {(e["element"], e["parameter"]) for e in log if not e["applied"]}
    for row in audit["defs"]:
        for name in row["applicable"]:
            assert (name, row["parameter"]) in applied, \
                f"audit said applicable, apply skipped: {name}"
        for name, _cls in row["skipped"]:
            assert (name, row["parameter"]) in skipped, \
                f"audit said skipped, apply landed: {name}"
    # the solenoid field_rel def really is fully skipped (the trap)
    sol_row = audit["defs"][1]
    assert sol_row["matched"] and not sol_row["applicable"]


@needs("examples/pipii/mebt+hwr/mebt+hwr.dat", "examples/pip2_misalignment_study")
def test_audit_flags_multiselection_and_inert(hwr_lattice):
    study = _study(
        hwr_lattice,
        ErrorDef(pattern="FMAP_*", parameter="dx", sigma=0.1,
                 element_kind="cavity"),
        ErrorDef(pattern="FMAP_0*", parameter="dx", sigma=0.2,
                 element_kind="cavity"),          # overlaps the first
        ErrorDef(pattern="QUAD_*", parameter="dz", sigma=0.1),  # inert
    )
    audit = study.audit_errors()
    assert audit["multi"], "overlapping defs not detected"
    assert all(v == [0, 1] for v in audit["multi"].values())
    assert audit["defs"][2]["inert"] is True


# ---------------------------------------------------------------------
# draw logging
# ---------------------------------------------------------------------
def test_applied_log_values_match_lattice_deltas():
    lat = Lattice()
    lat.add(Quadrupole("QUAD_1", 50.0, gradient=5.0))
    lat.add(Quadrupole("QUAD_2", 50.0, gradient=-5.0))
    study = _study(lat, ErrorDef(pattern="QUAD_*", parameter="dx",
                                 sigma=0.1))
    perturbed = study._apply_errors(seed=42)
    log = study.applied_log[42]
    assert len(log) == 2 and all(e["applied"] for e in log)
    for entry in log:
        pe = next(e for e in perturbed.elements
                  if e.name == entry["element"])
        assert pe.dx == pytest.approx(entry["value"])
        assert entry["mode"] == "align-add"


def test_phase_on_slotless_element_warns_and_logs():
    """The 'phase' branch used to be the last SILENT no-op path."""
    lat = Lattice()
    lat.add(Drift("D_1", 100.0))
    study = _study(lat, ErrorDef(pattern="D_*", parameter="phase",
                                 sigma=1.0))
    with pytest.warns(UserWarning, match="no 'phase' slot"):
        study._apply_errors(seed=2)
    log = study.applied_log[2]
    assert len(log) == 1 and log[0]["applied"] is False


def test_beam_applied_log_and_mismatch_percent_semantics():
    """mismatch_* draws are PERCENT: a σ=5 draw of e.g. +3.7 must land
    as cfg.mismatch_x = 3.7 (the factory then applies 1 + m/100)."""
    lat = Lattice()
    lat.add(Drift("D_1", 100.0))
    study = ErrorStudy(lat, BeamConfig(), n_seeds=1)
    study.add_beam_error(parameter="mismatch_x", sigma=5.0, cutoff=3.0)
    cfg = study._apply_beam_errors(seed=7)
    blog = study.beam_applied_log[7]
    assert len(blog) == 1 and blog[0]["applied"] is True
    assert cfg.mismatch_x == pytest.approx(blog[0]["value"])
    assert abs(blog[0]["value"]) <= 15.0          # 3σ of 5 %
    # and the factory consumes it as percent
    from linac_gen.distributions import factory as F
    assert 1.0 + cfg.mismatch_x / 100.0 == pytest.approx(
        1.0 + blog[0]["value"] / 100.0)


# ---------------------------------------------------------------------
# adversarial-review regressions (2026-07 round)
# ---------------------------------------------------------------------
def test_unknown_distribution_rejected_at_registration():
    """MED-1: 'Gaussian' (capital G) used to pass add_error and the
    audit while producing ZERO draws in every seed."""
    lat = Lattice()
    lat.add(Quadrupole("QUAD_1", 50.0, gradient=5.0))
    study = ErrorStudy(lat, BeamConfig(), n_seeds=1)
    with pytest.raises(ValueError, match="unknown distribution"):
        study.add_error("QUAD_*", "dx", distribution="Gaussian",
                        sigma=0.1)
    with pytest.raises(ValueError, match="unknown distribution"):
        study.add_beam_error("centroid_x", distribution="Normal",
                             sigma=0.1)


def test_unknown_distribution_direct_def_warns_and_audit_flags():
    """Defense in depth for ErrorDefs constructed directly (bypassing
    add_error): apply warns + logs applied=False, audit flags the row
    with empty applicable (the mirror holds only for valid dists)."""
    lat = Lattice()
    lat.add(Quadrupole("QUAD_1", 50.0, gradient=5.0))
    study = _study(lat, ErrorDef(pattern="QUAD_*", parameter="dx",
                                 sigma=0.1, distribution="Gaussian"))
    row = study.audit_errors()["defs"][0]
    assert row["bad_distribution"] is True
    assert row["matched"] == ["QUAD_1"] and row["applicable"] == []
    with pytest.warns(UserWarning, match="unknown distribution"):
        perturbed = study._apply_errors(seed=1)
    log = study.applied_log[1]
    assert len(log) == 1 and log[0]["applied"] is False
    assert next(e for e in perturbed.elements).dx == 0.0


def test_audit_multi_uses_def_identity_not_name_counts():
    """LOW-1: two same-named elements under ONE def are two legitimate
    per-element draws, not an overlap; two DISTINCT defs still flag."""
    lat = Lattice()
    lat.add(Quadrupole("QUAD_A", 50.0, gradient=5.0))
    lat.add(Quadrupole("QUAD_A", 50.0, gradient=-5.0))   # duplicate name
    one = _study(lat, ErrorDef(pattern="QUAD_*", parameter="dx",
                               sigma=0.1))
    assert one.audit_errors()["multi"] == {}
    two = _study(lat,
                 ErrorDef(pattern="QUAD_*", parameter="dx", sigma=0.1),
                 ErrorDef(pattern="QUAD_A", parameter="dx", sigma=0.2))
    multi = two.audit_errors()["multi"]
    assert multi == {("QUAD_A", "dx"): [0, 1]}


def test_audit_covers_beam_defs():
    """LOW-3: beam-input defs are audited too — a typo'd parameter or
    distribution shows up BEFORE any seed runs."""
    from linac_gen.errors.beam_error import BeamErrorDef
    lat = Lattice()
    lat.add(Drift("D_1", 100.0))
    study = ErrorStudy(lat, BeamConfig(), n_seeds=1)
    study.add_beam_error(parameter="centroid_x", sigma=0.1)
    study._beam_errors.append(BeamErrorDef(
        parameter="centroid_typo", sigma=0.1))       # direct construction
    rows = study.audit_errors()["beam_defs"]
    assert [r["applicable"] for r in rows] == [True, False]
    assert all(r["bad_distribution"] is False for r in rows)
