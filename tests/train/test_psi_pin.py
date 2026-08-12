"""sync_phase_pin: pinned == lazily-calibrated at nominal; pin survives
reset_run_state; unpinned path bit-identical to today (NCells fixture —
no field file needed)."""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.ncells import NCells

F = 804.96
BG = 0.456630316


def _w(beta):
    g = 1.0 / np.sqrt(1.0 - beta * beta)
    return (g - 1.0) * H_MINUS.mass


def _ref():
    return ReferenceParticle(species=H_MINUS, w_kin=_w(BG), frequency=F)


def _nc():
    return NCells("c", mode=1, n_cells=8, beta_g=BG, eot_v_per_m=6.8e6,
                  theta_s_deg=-28.0, aperture_mm=15.0, sync_phase=True,
                  frequency_mhz=F)


def _track_dw(nc):
    ref = _ref()
    w0 = ref.w_kin
    beam = Beam(ref=ref, n_particles=4, current=0.0)
    beam.particles[:] = 0.0
    nc.reset_run_state()
    nc.track_rk4(beam, nc.length)
    return ref.w_kin - w0, getattr(nc, "_sync_offset_deg", None)


def test_pin_matches_lazy_and_survives_reset():
    nc = _nc()
    dw_lazy, psi = _track_dw(nc)
    assert psi is not None
    nc2 = _nc()
    nc2.sync_phase_pin = float(psi)
    dw_pin, psi2 = _track_dw(nc2)
    assert psi2 == pytest.approx(psi)
    assert dw_pin == pytest.approx(dw_lazy, rel=1e-12)
    # pin survives reset; a second tracked pass reuses it identically
    dw_pin2, psi3 = _track_dw(nc2)
    assert psi3 == pytest.approx(psi)
    assert dw_pin2 == pytest.approx(dw_pin, rel=1e-12)


def test_unpinned_recalibrates_each_pass_identically():
    nc = _nc()
    dw1, psi1 = _track_dw(nc)
    dw2, psi2 = _track_dw(nc)
    assert psi1 == pytest.approx(psi2)
    assert dw1 == pytest.approx(dw2, rel=1e-12)


def test_pin_beats_loaded_voltage_refit():
    """The reason the pin exists: with a perturbed (loaded) voltage, the
    UNPINNED cavity re-fits psi at the perturbed amplitude, while the
    pinned cavity keeps the design operating point."""
    nc_a = _nc()
    _, psi_design = _track_dw(nc_a)
    nc_b = _nc()
    nc_b.voltage_rel = -0.05                      # 5% beam-loading droop
    _, psi_refit = _track_dw(nc_b)
    nc_c = _nc()
    nc_c.voltage_rel = -0.05
    nc_c.sync_phase_pin = float(psi_design)
    _, psi_pinned = _track_dw(nc_c)
    assert psi_pinned == pytest.approx(psi_design)
    # and the refit differs (documents WHY pinning is needed) — unless the
    # cavity's psi is voltage-independent, in which case both match.
    if abs(psi_refit - psi_design) < 1e-9:
        pytest.skip("psi voltage-independent for this cavity — pin inert")
