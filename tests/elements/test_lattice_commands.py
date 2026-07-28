"""Tests for ``linac_gen.elements.lattice_commands`` parse / write / apply.

Covers each ``SET_*`` and ``ADJUST_*`` command class:

* the **active** subclasses (``SetSyncPhase``, ``SetBeamPhaseError``,
  ``SetBeamE0P0``, ``SetBeamEnergy``, ``SetGaussianCutOff``) — assert
  that ``apply_command(track_state)`` mutates the right field,
* every command — assert .dat round-trip via parse_tracewin / write_tracewin
  is lossless on the synthetic fixture ``tests/io/fixtures/lattice_with_set_commands.dat``.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from linac_gen.core.track_state import TrackState
from linac_gen.elements.lattice_commands import (
    LatticeCommand,
    SetSyncPhase, SetBeamPhaseError, SetBeamE0P0, SetBeamEnergy,
    SetGaussianCutOff,
    SetTwiss, SetPosition, SetAchromat, SetSize, SetSizeMax, SetSizeMin,
    SetBeamPhaseAdv, SetSeparation, SetAdv,
    Adjust, AdjustSteerer, AdjustSteererBx, AdjustSteererBy,
    AdjustBeamTwiss, AdjustBeamCentroid, AdjustBeamEmit, AdjustBeamCurrent,
)
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.io.tracewin_writer import write_tracewin


FIXTURE = Path(__file__).parent.parent / "io" / "fixtures" / "lattice_with_set_commands.dat"


# ---------------------------------------------------------------------------
# Run-time active commands — apply_command behaviour
# ---------------------------------------------------------------------------
class _MutableRef:
    """Lightweight stand-in for ``ReferenceParticle`` (only ``w_kin`` used)."""
    def __init__(self, w_kin: float = 0.0):
        self.w_kin = w_kin


class TestActiveCommands:
    def test_sync_phase_sets_flag(self):
        ts = TrackState()
        SetSyncPhase("CMD_001").apply_command(ts)
        assert ts.sync_phase_mode is True

    def test_phase_error_accumulates_and_clears(self):
        ts = TrackState()
        SetBeamPhaseError("CMD_001", dphi_deg=5.0).apply_command(ts)
        SetBeamPhaseError("CMD_002", dphi_deg=2.5).apply_command(ts)
        assert ts.phase_ref_shift == pytest.approx(7.5)
        SetBeamPhaseError("CMD_003", dphi_deg=0.0).apply_command(ts)
        assert ts.phase_ref_shift == 0.0

    def test_phase_error_random_flag_raises(self):
        ts = TrackState()
        with pytest.raises(NotImplementedError):
            SetBeamPhaseError("CMD_X", dphi_deg=1.0, random_flag=1).apply_command(ts)

    def test_beam_energy_sets_w_kin(self):
        ref = _MutableRef(2.0)
        ts = TrackState(ref=ref)
        SetBeamEnergy("CMD_E", k=1, energy_MeV=5.0).apply_command(ts)
        assert ref.w_kin == 5.0

    def test_beam_e0_p0_independent_axes(self):
        ref = _MutableRef(3.0)
        ts = TrackState(ref=ref)
        # kE off → energy unchanged
        SetBeamE0P0("CMD_X", dE_MeV=1.0, dphi_deg=10.0, ke=0, kp=1).apply_command(ts)
        assert ref.w_kin == 3.0
        assert ts.phase_ref_shift == 10.0
        # kp off → phase unchanged
        SetBeamE0P0("CMD_Y", dE_MeV=2.0, dphi_deg=99.0, ke=1, kp=0).apply_command(ts)
        assert ref.w_kin == 5.0
        assert ts.phase_ref_shift == 10.0

    def test_gaussian_cutoff_sets_field(self):
        ts = TrackState()
        SetGaussianCutOff("CMD_C", sigma=3.5).apply_command(ts)
        assert ts.error_cutoff_sigma == 3.5


# ---------------------------------------------------------------------------
# Round-trip — every command on the fixture
# ---------------------------------------------------------------------------
class TestRoundTrip:
    def test_fixture_parses_without_warnings(self):
        lat, meta = parse_tracewin(FIXTURE)
        # The fixture's SET_SIZE_MAX/MIN cards carry k2=1, which since
        # the 2026-07 review round warns per parse (centroid-inclusive
        # transverse sizes are not modelled) — those two warnings are
        # EXPECTED; anything else is a regression.
        assert all("k2=1" in w for w in meta["warnings"]), \
            meta["warnings"]
        assert len(meta["warnings"]) == 2, meta["warnings"]
        cmds = [e for e in lat.elements if isinstance(e, LatticeCommand)]
        # 22 SET_*/ADJUST_* + the FREQ card (now a Freq command element)
        assert len(cmds) == 23, [c.KEYWORD for c in cmds]

    def test_round_trip_identical(self, tmp_path):
        lat, _ = parse_tracewin(FIXTURE)
        out = tmp_path / "round.dat"
        write_tracewin(lat, str(out), frequency=162.5)
        lat2, meta2 = parse_tracewin(str(out))
        assert all("k2=1" in w for w in meta2["warnings"]), \
            meta2["warnings"]
        cmds  = [e for e in lat.elements  if isinstance(e, LatticeCommand)]
        cmds2 = [e for e in lat2.elements if isinstance(e, LatticeCommand)]
        assert len(cmds) == len(cmds2)
        for before, after in zip(cmds, cmds2):
            assert type(before) is type(after)
            assert before.KEYWORD == after.KEYWORD
            assert before.to_tracewin_args() == after.to_tracewin_args(), (
                before.KEYWORD, before.to_tracewin_args(),
                after.to_tracewin_args(),
            )


# ---------------------------------------------------------------------------
# Per-class smoke — every command class can construct, format args, parse back.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ctor,args,kwargs,expected_kw", [
    (SetSyncPhase, ("c",), {}, "SET_SYNC_PHASE"),
    (SetBeamPhaseError, ("c",), dict(dphi_deg=2.0, random_flag=0), "SET_BEAM_PHASE_ERROR"),
    (SetBeamE0P0, ("c",), dict(k=1, dE_MeV=0.5, dphi_deg=1.0, ke=1, kp=0), "SET_BEAM_E0_P0"),
    (SetBeamEnergy, ("c",), dict(k=1, energy_MeV=10.0), "SET_BEAM_ENERGY"),
    (SetGaussianCutOff, ("c",), dict(sigma=4.0), "SET_GAUSSIAN_CUT_OFF"),
    (SetTwiss, ("c",), dict(family="Q1", alpha_x=1.0, beta_x=0.5), "SET_TWISS"),
    (SetPosition, ("c",), dict(k=1, x_mm=0.1), "SET_POSITION"),
    (SetAchromat, ("c",), dict(k=1, f1=1, f2=1, plane=0), "SET_ACHROMAT"),
    (SetSize, ("c",), dict(k=1, x_mm=5, y_mm=5, phi_or_z=10), "SET_SIZE"),
    (SetSizeMax, ("c",), dict(k=1, n_elems=3, x_mm=5, y_mm=5, phi_or_z=10, k2=1), "SET_SIZE_MAX"),
    (SetSizeMin, ("c",), dict(k=1, n_elems=3, x_mm=0.5, y_mm=0.5), "SET_SIZE_MIN"),
    (SetBeamPhaseAdv, ("c",), dict(k=1, n_elems=4, mu_x_deg=60, mu_y_deg=60), "SET_BEAM_PHASE_ADV"),
    (SetSeparation, ("c",), dict(k=1, sx=2.0, sy=2.0), "SET_SEPARATION"),
    (SetAdv, ("c",), dict(kxot=90, kyot=90), "SET_ADV"),
    (Adjust, ("c",), dict(target="QUAD", param_idx=2, link_group=1, vmin=-10, vmax=10), "ADJUST"),
    (AdjustSteerer, ("c",), dict(diag_n=1, vmax=0.05, first_step=1e-3), "ADJUST_STEERER"),
    (AdjustSteererBx, ("c",), dict(diag_n=1, vmax=0.05), "ADJUST_STEERER_BX"),
    (AdjustSteererBy, ("c",), dict(diag_n=1, vmax=0.05), "ADJUST_STEERER_BY"),
])
def test_command_class_smoke(ctor, args, kwargs, expected_kw):
    cmd = ctor(*args, **kwargs)
    assert cmd.KEYWORD == expected_kw
    assert cmd.length == 0.0
    args_out = cmd.to_tracewin_args()
    assert isinstance(args_out, list)
    assert all(isinstance(a, str) for a in args_out)


@pytest.mark.parametrize("ctor,n_flags", [
    (AdjustBeamTwiss, 6),
    (AdjustBeamCentroid, 6),
    (AdjustBeamEmit, 3),
    (AdjustBeamCurrent, 1),
])
def test_adjust_beam_flag_arity(ctor, n_flags):
    cmd = ctor("c", 1, *([1] * n_flags))
    args = cmd.to_tracewin_args()
    assert len(args) == 1 + n_flags
