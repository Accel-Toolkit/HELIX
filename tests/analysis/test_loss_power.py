"""Loss-power accounting (`analysis.loss_power`): watts tables, W/m
profiles, plane power density, HDF5 round-trip, and the end-to-end MP
path (Simulation attaches the loss table; the CLI prints the block).
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.analysis import loss_power as LP
from linac_gen.core.beam import LOSS_DTYPE


def _table(rows):
    return np.array(rows, dtype=LOSS_DTYPE)


# ---------------------------------------------------------------------------
# power arithmetic (hand-checked numbers)
# ---------------------------------------------------------------------------
def test_watts_per_macroparticle_hand_numbers():
    # 5 mA CW, 100 macroparticles: I_avg/N = 5e-5 A; at 2 MeV -> 100 W
    w = LP.watts_per_macroparticle(2.0, 5.0, 100.0, 100)
    assert w == pytest.approx(100.0)


def test_pip2_booster_anchor():
    # 2 mA x 550 us x 20 Hz = 1.1% duty at 800 MeV -> 17.6 kW total
    n = 12345
    w_total = n * LP.watts_per_macroparticle(800.0, 2.0, 1.1, n)
    assert w_total == pytest.approx(17600.0, rel=1e-12)


def test_zero_and_negative_n_macro_rejected():
    with pytest.raises(ValueError):
        LP.watts_per_macroparticle(1.0, 1.0, 100.0, 0)


# ---------------------------------------------------------------------------
# per-element table
# ---------------------------------------------------------------------------
def test_loss_power_table_groups_and_sorts():
    lt = _table([
        (0, 1000.0, 1.0, 0.0, 2.0, "SCRAPER1"),
        (1, 1010.0, -1.0, 0.5, 2.0, "SCRAPER1"),
        (2, 9000.0, 0.0, 3.0, 10.0, "APERTURE7"),
    ])
    tab = LP.loss_power_table(lt, current_mA=5.0, duty_pct=100.0,
                              n_macro=100)
    # per-macro at 2 MeV = 100 W, at 10 MeV = 500 W (5 mA CW, N=100):
    # APERTURE7 = 1x500 = 500 W  >  SCRAPER1 = 2x100 = 200 W
    assert list(tab["element_name"]) == ["APERTURE7", "SCRAPER1"]
    assert tab["watts"][0] == pytest.approx(500.0)
    assert tab["watts"][1] == pytest.approx(200.0)
    assert tab["n_lost"][1] == 2
    assert tab["s_min_mm"][1] == 1000.0 and tab["s_max_mm"][1] == 1010.0
    assert tab["e_mean_mev"][0] == pytest.approx(10.0)


def test_loss_power_table_empty():
    assert LP.loss_power_table(_table([]), current_mA=5.0,
                               n_macro=10).size == 0


# ---------------------------------------------------------------------------
# W/m profile
# ---------------------------------------------------------------------------
def test_loss_power_profile_reads_in_w_per_m():
    # three 100 W losses inside the first metre, none elsewhere
    lt = _table([(i, 100.0 + 200.0 * i, 0, 0, 2.0, "A") for i in range(3)])
    s, wpm = LP.loss_power_profile(lt, current_mA=5.0, duty_pct=100.0,
                                   n_macro=100, s_end_mm=3000.0)
    assert len(s) == 3
    assert wpm[0] == pytest.approx(300.0)          # 3 x 100 W / 1 m
    assert wpm[1] == wpm[2] == 0.0


# ---------------------------------------------------------------------------
# plane power density
# ---------------------------------------------------------------------------
def test_plane_power_density_integral_conserves_power():
    rng = np.random.default_rng(7)
    x, y = rng.normal(0, 3.0, 5000), rng.normal(0, 2.0, 5000)
    H, xe, ye = LP.plane_power_density(
        x, y, energy_mev=800.0, current_mA=2.0, duty_pct=1.1,
        n_macro=5000, bins=48)
    area_cm2 = (np.diff(xe)[0] * 0.1) * (np.diff(ye)[0] * 0.1)
    total = H.sum() * area_cm2
    # all 5000 launched macroparticles arrive -> full 17.6 kW on plane
    assert total == pytest.approx(17600.0, rel=1e-9)


def test_plane_power_density_empty_raises():
    with pytest.raises(ValueError):
        LP.plane_power_density(np.array([]), np.array([]),
                               energy_mev=1.0, current_mA=1.0, n_macro=1)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def test_summary_delivered_and_lost():
    lt = _table([(0, 500.0, 0, 0, 2.0, "SCR")])
    s = LP.loss_power_summary(lt, current_mA=5.0, duty_pct=100.0,
                              n_macro=100, w_exit_mev=800.0, n_alive=99)
    assert s["i_avg_ma"] == pytest.approx(5.0)
    assert s["n_lost"] == 1
    assert s["lost_w"] == pytest.approx(100.0)
    assert s["delivered_w"] == pytest.approx(99 * 5e-5 * 800e6)
    assert s["top"][0][0] == "SCR"


# ---------------------------------------------------------------------------
# HDF5 round-trip
# ---------------------------------------------------------------------------
def test_hdf5_roundtrip_of_loss_table(tmp_path):
    from linac_gen.io.hdf5_output import (load_results_hdf5,
                                          save_results_hdf5)

    class _Rec:                                    # minimal recorder
        s = [0.0, 1.0]
        sigma_x = [1.0, 1.0]
        transmission = [100.0, 99.0]
        ref_w_kin = [2.0, 2.0]

    rec = _Rec()
    rec.loss_table = _table([
        (3, 1234.5, 0.1, -0.2, 1.75, "MEBT_SCRAPER"),
        (9, 2000.0, 2.0, 0.0, 2.10, "APERTURE2"),
    ])
    rec.n_macro = 2000
    fp = tmp_path / "r.h5"
    save_results_hdf5(rec, str(fp))
    out = load_results_hdf5(str(fp))
    lt = out["loss_table"]
    assert out["n_macro"] == 2000
    assert lt.shape == (2,)
    assert lt["element_name"][0] == "MEBT_SCRAPER"
    assert lt["energy"][1] == pytest.approx(2.10)
    assert lt["s"][0] == pytest.approx(1234.5)


# ---------------------------------------------------------------------------
# end-to-end: MP run on a tiny lattice with a scraping aperture
# ---------------------------------------------------------------------------
def test_mp_run_attaches_loss_table_and_powers_balance():
    from linac_gen.core.beam import Beam
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.core.simulation import Simulation
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.aperture import Aperture

    lat = Lattice()
    lat.add(Drift("D1", length=200.0, aperture=50.0))
    lat.add(Aperture("APER", dx=2.0, dy=2.0, aperture_type=0))
    lat.add(Drift("D2", length=200.0, aperture=50.0))

    ref = ReferenceParticle(species=H_MINUS, w_kin=2.0, frequency=162.5)
    rng = np.random.default_rng(11)
    beam = Beam(ref=ref, n_particles=2000, current=5.0)
    beam.particles[:, 0] = rng.normal(0.0, 2.0, 2000)   # x mm
    beam.particles[:, 2] = rng.normal(0.0, 2.0, 2000)   # y mm

    res = Simulation(lat, beam).run()
    lt = res.loss_table
    assert res.n_macro == 2000
    assert lt.size > 100                       # sigma=2 on +/-2 mm cut
    assert set(np.unique(lt["element_name"])) == {"APER"}
    # power bookkeeping closes: lost + delivered == launched
    from linac_gen.analysis.loss_power import (loss_power_table,
                                               watts_per_macroparticle)
    tab = loss_power_table(lt, current_mA=5.0, duty_pct=100.0,
                           n_macro=2000)
    w_one = watts_per_macroparticle(2.0, 5.0, 100.0, 2000)
    lost_w = tab["watts"].sum()
    delivered_w = beam.n_alive * w_one
    assert lost_w + delivered_w == pytest.approx(2000 * w_one, rel=1e-9)
