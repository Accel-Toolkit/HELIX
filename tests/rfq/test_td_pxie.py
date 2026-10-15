"""The time-domain RFQ solver against Toutatis on the PXIE RFQ.

Both codes start from Toutatis's own input particles (7786 alive at the
RFQ entrance, 30 keV H-, DC) and track the same vanes (RFQ_GEOM .vane
file) at 0 and 5 mA.  The vane file and the Toutatis runs live outside
git; every test here skips when they are absent.

Toutatis reference (its own output files, read here):
  transmission 98.37 % / 80.40 %, exit <W> 2.1187 / 2.1211 MeV.
Its PXIE runs used 12 steps per RF period; the time-domain solver at 12
steps moves the 5 mA transmission by about +1.8 points, so the reference
itself carries about +-2 points -- hence the tolerances below.
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

from tests.rfq import ref_loaders as rl

pytestmark = pytest.mark.slow

N_FILE = 10000


def _read_dst(path):
    """Toutatis .dst: 23-byte header, then 7 float64 per particle
    (x cm, x' rad, y cm, y' rad, phase rad, W MeV, lost flag)."""
    raw = path.read_bytes()[23:23 + N_FILE * 7 * 8]
    return np.frombuffer(raw, dtype="<f8").reshape(N_FILE, 7)


def _ref_dir(case):
    d = rl.toutatis_ref_dir(case)
    if d is None:
        pytest.skip(f"Toutatis PXIE {case} reference run not present")
    return d


def _vane():
    p = rl.reference_path("pxie-rfq.vane")
    if p is None:
        pytest.skip("PXIE vane file not present on this machine")
    return p


def _deck(tmp_path, vane):
    """The RFQ section of the in-repo lebt2sol_plus_rfq_phifix deck."""
    src = rl.REPO_ROOT / "examples" / "lebt_plus_rfq" / \
        "lebt2sol_plus_rfq_phifix.dat"
    from tests.dataguard import require
    require(str(src))                   # not in the public repository
    lines = src.read_text().splitlines()
    i0 = next(k for k, ln in enumerate(lines)
              if ln.startswith("RFQ_GAP_RMS_FFS"))
    i1 = max(k for k, ln in enumerate(lines) if ln.startswith("RFQ_CELL"))
    body = ([f"RFQ_GEOM 1 {vane}"] + ["DRIFT 0.00001 5"] * 3
            + lines[i0:i1 + 1] + ["END"])
    deck = tmp_path / "pxie_rfq.dat"
    deck.write_text("\n".join(body) + "\n")
    return deck


def _run(case, tmp_path, *, n=None, solver="time_domain", cards=False,
         periodic=True):
    """``n``: the first n input particles only; ``cards``: no vane file
    (per-cell: the cards' tips with the quadrant boundary)."""
    from linac_gen.core.config import BeamConfig, SpaceChargeConfig
    from linac_gen.core.simulation import Simulation
    from linac_gen.distributions.factory import create_beam
    from linac_gen.io.tracewin_parser import parse_tracewin

    ref = _ref_dir(case)
    vane = _vane()
    if solver == "time_domain" and not cards:
        pytest.importorskip("pyamg")
    t_in = _read_dst(ref / "toutatis.dst")
    alive_in = t_in[:, 6] == 0
    # the current of the particles alive at the entrance (I / f / N per
    # macro-particle, so a subset carries the same beam)
    current = {"0mA": 0.0, "5mA": 5.0 * alive_in.mean()}[case]
    if n is not None:
        alive_in &= np.cumsum(alive_in) <= n
    p = t_in[alive_in]
    cfg = BeamConfig(species="H-", energy=0.030, frequency=162.5,
                     current=current, duty_cycle=100.0,
                     n_particles=int(alive_in.sum()), distribution="gaussian",
                     cutoff=4.0, emit_nx=0.1, alpha_x=0.0, beta_x=0.1,
                     emit_ny=0.1, alpha_y=0.0, beta_y=0.1, emit_z=0.0,
                     alpha_z=0.0, beta_z=1.0, continuous=True,
                     dc_energy_spread_keV=0.0, periodic_phase=periodic)
    beam = create_beam(cfg, seed=42)
    beam.particles[:, 0] = p[:, 0] * 10.0
    beam.particles[:, 1] = p[:, 1] * 1e3
    beam.particles[:, 2] = p[:, 2] * 10.0
    beam.particles[:, 3] = p[:, 3] * 1e3
    beam.particles[:, 4] = np.degrees(p[:, 4])
    beam.particles[:, 5] = p[:, 5] - 0.030
    beam.lost[:] = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, _ = parse_tracewin(str(_deck(tmp_path, vane)))
    lat.step_config = dataclasses.replace(lat.step_config,
                                          rfq_solver=solver)
    if cards:
        lat.rfq_geom_file = None
        for e in lat.elements:
            if type(e).__name__ == "RfqCell":
                e._geom_boundary = True
                e._wall_mm = 0.0
    sc = (SpaceChargeConfig(nx=64, ny=64, nz=64, grid_extent=6.0,
                            grid_mode="adaptive") if current > 0 else "off")
    res = Simulation(lat, beam, space_charge=sc, rfq_geometry="off").run()
    cells = [e.name for e in lat.elements if type(e).__name__ == "RfqCell"]
    return dict(res=res, beam=beam, cells=cells, n_in=int(alive_in.sum()),
                alive_in=alive_in, ref=ref)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = {}
    for case in ("0mA", "5mA"):
        out[case] = _run(case, tmp_path_factory.mktemp(case))
    return out


def _toutatis(run):
    t_out = _read_dst(run["ref"] / "distrib.dst")
    ok = run["alive_in"] & (t_out[:, 6] == 0)
    table = np.loadtxt(run["ref"] / "toutatis.out", skiprows=10,
                       usecols=range(23))
    table = table[table[:, 0] >= 1]             # one row per cell exit
    assert np.array_equal(table[:, 0], np.arange(1, table.shape[0] + 1))
    return ok, t_out, table


@pytest.mark.parametrize("case, lo, hi", [("0mA", 97.8, 99.0),
                                          ("5mA", 78.9, 81.9)])
def test_transmission_matches_toutatis(runs, case, lo, hi):
    run = runs[case]
    ok, _t_out, _tab = _toutatis(run)
    t_ref = 100.0 * ok.sum() / run["n_in"]
    t_td = 100.0 * run["beam"].alive_mask.mean()
    assert lo <= t_ref <= hi                  # the reference itself
    assert lo <= t_td <= hi, (t_td, t_ref)


@pytest.mark.parametrize("case", ["0mA", "5mA"])
def test_exit_energy_within_3_keV(runs, case):
    run = runs[case]
    ok, t_out, _tab = _toutatis(run)
    b = run["beam"]
    w_td = float(np.mean(b.ref.w_kin + b.particles[b.alive_mask, 5]))
    w_ref = float(np.mean(t_out[ok, 5]))
    assert abs(w_td - w_ref) < 3e-3, (w_td, w_ref)


@pytest.mark.parametrize("case", ["0mA", "5mA"])
def test_losses_along_the_rfq_follow_toutatis(runs, case):
    """Particles alive at cell exits outside the buncher's loss burst
    (Toutatis's npart column) agree within 1 % of the input count, and at
    5 mA the burst sits at the same place: the position where half of all
    losses have happened agrees within 0.1 m (the time-domain losses trail
    by about 5 cm, 2-3 % of the input mid-burst)."""
    run = runs[case]
    _ok, _t_out, tab = _toutatis(run)
    res = run["res"]
    names = list(res.element_names)
    rows = [names.index(c) for c in run["cells"]]
    n_in = run["n_in"]
    alive_td = np.asarray(res.transmission)[rows] / 100.0 * n_in
    alive_ref = tab[:len(rows), 21]
    z = tab[:len(rows), 1]
    outside = np.array([50, 75, 125, 150, 175, 200]) - 1
    dev = np.abs(alive_td[outside] - alive_ref[outside]) / n_in
    assert dev.max() < 0.01, dict(zip(outside + 1, np.round(100 * dev, 2)))
    if case == "5mA":
        def z_half(alive):
            lost = alive[0] - alive
            return float(np.interp(0.5 * lost[-1], lost, z))
        assert abs(z_half(alive_td) - z_half(alive_ref)) < 0.1


def test_beam_sizes_follow_toutatis_at_5mA(runs):
    """sigma_x and sigma_y at the cell exits (z > 0.25 m) lie within a
    median of 12 % of Toutatis's table.  The table holds fixed-time
    snapshots and HELIX fixed-z crossings, which accounts for most of
    the difference."""
    run = runs["5mA"]
    _ok, _t_out, tab = _toutatis(run)
    res = run["res"]
    names = list(res.element_names)
    rows = [names.index(c) for c in run["cells"]]
    z = tab[:len(rows), 1]
    m = z > 0.25
    for col, key in ((9, "sigma_x"), (10, "sigma_y")):
        td = np.asarray(getattr(res, key))[rows][m]
        ref = tab[:len(rows), col][m]
        med = float(np.median(np.abs(td / ref - 1.0)))
        assert med < 0.12, (key, med)


def test_regression_mode_reproduces_the_per_cell_energy_gain(tmp_path,
                                                             monkeypatch):
    """Regression mode: the time-domain integrator and clock driven by the
    per-cell model's own analytic two-term field (cards_geometry.
    two_term_field_map) gain the same energy as the per-cell model, from
    the same particles at 0 mA -- so the cards' synchronous phases agree
    with a global RF clock, and the solvers' differences come from the
    field (real vanes against two-term) and the transverse model."""
    from linac_gen.rfq_td import build
    from linac_gen.rfq_td.cards_geometry import two_term_field_map
    cells_run = _run("0mA", tmp_path, n=1000, solver="cells", cards=True)
    maps = {}

    def two_term(electrodes, edges, **kw):
        if "m" not in maps:
            from linac_gen.io.tracewin_parser import parse_tracewin
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lat, _ = parse_tracewin(str(_deck(tmp_path, _vane())))
            cells = [e for e in lat.elements
                     if type(e).__name__ == "RfqCell"]
            # at 1 V: the solver scales card-built maps by the voltage
            maps["m"] = two_term_field_map(cells, voltage=1.0)
        return maps["m"]

    monkeypatch.setattr(build, "get_field_map", two_term)
    td_run = _run("0mA", tmp_path, n=1000, cards=True)

    def accelerated(run):
        b = run["beam"]
        w = b.ref.w_kin + b.particles[b.alive_mask, 5]
        return w[w > 2.0]

    w_c, w_t = accelerated(cells_run), accelerated(td_run)
    assert abs(w_t.mean() - w_c.mean()) < 3e-3, (w_t.mean(), w_c.mean())
    t_c = 100.0 * w_c.size / cells_run["n_in"]
    t_t = 100.0 * w_t.size / td_run["n_in"]
    assert abs(t_t - t_c) < 2.5, (t_t, t_c)


def test_one_bunch_at_the_exit_without_periodic_phase(tmp_path):
    """The time-domain bucket does not line up with the DC injection
    window (part of the beam is captured one period later); every particle
    takes its whole periods from its injection phase, so without the
    periodic fold the accelerated particles still leave as one bunch."""
    run = _run("0mA", tmp_path, periodic=False)
    b = run["beam"]
    al = b.alive_mask
    w = b.ref.w_kin + b.particles[al, 5]
    ph = b.particles[al, 4][w > 2.0]
    assert ph.size > 7000
    assert (np.abs(ph) > 180.0).mean() < 0.01
    assert np.std(ph[np.abs(ph) < 180.0]) < 15.0
