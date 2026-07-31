"""GUI round-trip of the `periodic_phase` beam flag.

The checkbox is deliberately coupled to `Continuous beam`: the tracker
only folds a beam that was injected DC and has since been bunched, so
ticking it on a bunched beam would be a setting that silently does
nothing.  These tests pin the coupling AND that the flag survives a
save/load cycle — a dropped flag would leave a project quietly
producing the train-wide σ_φ the user thought they had turned off.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")


def _cfg(**kw):
    from linac_gen.core.config import BeamConfig
    base = dict(species="proton", energy=0.03, frequency=162.5,
                current=0.0, n_particles=1000)
    base.update(kw)
    return BeamConfig(**base)


def test_periodic_phase_round_trip(qapp):
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.beam_tab import BeamTab

    tab = BeamTab(AppState())
    try:
        tab.set_beam_config(_cfg(continuous=True, periodic_phase=True))
        assert tab._periodic_phase.isChecked() is True
        assert tab._periodic_phase.isEnabled() is True
        assert tab._build_cfg().periodic_phase is True

        # Default off, and an older project without the key still loads.
        tab.set_beam_config(_cfg(continuous=True))
        assert tab._periodic_phase.isChecked() is False
        assert tab._build_cfg().periodic_phase is False
        legacy = _cfg(continuous=True)
        del legacy.periodic_phase          # instance attr shadowing default
        tab.set_beam_config(legacy)
        assert tab._build_cfg().periodic_phase is False
    finally:
        tab.deleteLater()


def test_periodic_phase_survives_a_continuous_toggle(qapp):
    """Untick DC and re-tick it: the setting must come back.

    An earlier version CLEARED the box on untick, so an accidental
    double-toggle silently destroyed the user's choice and the next
    save wrote `periodic_phase: false`.  The box now greys out but
    keeps its tick, and `_build_cfg` ANDs with the DC state so a
    greyed-out tick can never reach the project.
    """
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.beam_tab import BeamTab

    tab = BeamTab(AppState())
    try:
        tab.set_beam_config(_cfg(continuous=True, periodic_phase=True))
        assert tab._periodic_phase.isChecked()
        tab._continuous.setChecked(False)          # the real user action
        assert tab._periodic_phase.isEnabled() is False
        assert tab._build_cfg().periodic_phase is False   # not written
        tab._continuous.setChecked(True)                  # ...and back
        assert tab._periodic_phase.isEnabled() is True
        assert tab._build_cfg().periodic_phase is True    # restored
    finally:
        tab.deleteLater()


def test_bunched_project_cannot_smuggle_the_flag_in(qapp):
    """A project file that sets periodic_phase on a BUNCHED beam is
    inconsistent; whatever the widget shows, the built config must not
    carry it — the tracker would ignore it anyway."""
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.beam_tab import BeamTab

    tab = BeamTab(AppState())
    try:
        tab.set_beam_config(_cfg(continuous=False, periodic_phase=True))
        assert tab._periodic_phase.isEnabled() is False
        assert tab._build_cfg().periodic_phase is False
    finally:
        tab.deleteLater()


def test_reset_defaults_clears_the_flag(qapp):
    """ADVERSARIAL FIND: `Reset defaults` rebuilt 30-odd widgets but
    skipped the beam-mode toggles, then called `_apply`, so a ticked
    Periodic phase went straight into the project state — a reset that
    silently kept changing the tracked coordinates."""
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.beam_tab import BeamTab

    tab = BeamTab(AppState())
    try:
        tab.set_beam_config(_cfg(continuous=True, periodic_phase=True,
                                 dc_energy_spread_keV=3.0))
        tab._reset()
        assert tab._periodic_phase.isChecked() is False
        assert tab._continuous.isChecked() is False
        assert tab._dc_dw.value() == 0.0
        cfg = tab._build_cfg()
        assert cfg.periodic_phase is False and cfg.continuous is False
    finally:
        tab.deleteLater()


def test_flag_reaches_a_real_run_through_the_tab(qapp):
    """End-to-end through the real entry path: tick the box on the Beam
    tab, build the config the way Apply does, generate the beam through
    the factory and track it — the run's results must carry the
    provenance the Results tab reads."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.core.simulation import Simulation
    from linac_gen.distributions.factory import create_beam
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.rf_gap import RFGap
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.beam_tab import BeamTab

    lat = Lattice()
    lat.add(Drift("D0", length=100.0, aperture=50.0))
    lat.add(RFGap("B", voltage=0.002, phase=-90.0, frequency=162.5,
                  ttf=1.0, aperture=50.0))
    for i in range(6):
        lat.add(Drift(f"D{i}", length=200.0, aperture=50.0))

    tab = BeamTab(AppState())
    try:
        tab.set_beam_config(_cfg(continuous=True, energy=0.03,
                                 n_particles=200, emit_z=0.0))
        tab._periodic_phase.setChecked(True)      # the user's click
        cfg = tab._build_cfg()
        assert cfg.periodic_phase is True
        beam = create_beam(cfg, seed=3)
        res = Simulation(lat, beam, space_charge="off").run()
        assert beam.bunch_train is True
        assert res.periodic_phase is True
        assert abs(beam.particles[beam.alive_mask, 4]).max() <= 180.0
    finally:
        tab.deleteLater()


def test_incompatible_run_fails_cleanly_instead_of_crashing(qapp):
    """The CSR guard raises inside Tracker.__init__, i.e. inside
    Simulation.run() — the GUI worker's try/except must turn that into a
    `failed` signal, not an unhandled exception on a QThread."""
    import pytest as _pytest
    from linac_gen.core.config import SpaceChargeConfig
    from linac_gen.core.lattice import Lattice
    from linac_gen.core.simulation import Simulation
    from linac_gen.distributions.factory import create_beam
    from linac_gen.elements.drift import Drift

    lat = Lattice()
    lat.add(Drift("D", length=100.0, aperture=50.0))
    beam = create_beam(_cfg(continuous=True, energy=0.03, n_particles=64,
                            emit_z=0.0, periodic_phase=True), seed=1)
    sc = SpaceChargeConfig(nx=16, ny=16, nz=16, use_gpu="cpu",
                           grid_mode="adaptive", csr_enabled=True)
    with _pytest.raises(ValueError, match="csr_enabled"):
        Simulation(lat, beam, space_charge=sc).run()
