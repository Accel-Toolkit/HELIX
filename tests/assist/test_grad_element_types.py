"""grad_sensitivities on every element type a magnet line carries.

Markers, apertures, compensation and grid cards, steerers, thin lenses, multipoles, matrix
elements and foils stopped it with a TypeError from the torch matrix path (a deck with
diagnostic markers failed at its first one), and an RFQ cell got past its refusal, whose name
test was case-sensitive, to the same TypeError.  Anchored on the numpy envelope at zero current,
element by element, and on central differences through it.
"""
from __future__ import annotations

import copy
import importlib
import inspect
import pkgutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from linac_gen.assist.tools import TOOLS, WorkContext  # noqa: E402
from linac_gen.cli.common import _envelope_initial, build_ref  # noqa: E402
from linac_gen.core.config import BeamConfig  # noqa: E402
from linac_gen.core.lattice import Lattice  # noqa: E402
from linac_gen.elements.aperture import Aperture  # noqa: E402
from linac_gen.elements.base import Element, FieldMapElement  # noqa: E402
from linac_gen.elements.dipole import Dipole  # noqa: E402
from linac_gen.elements.drift import Drift  # noqa: E402
from linac_gen.elements.edge import Edge  # noqa: E402
from linac_gen.elements.foil import Foil  # noqa: E402
from linac_gen.elements.lattice_commands import (  # noqa: E402
    Freq,
    SetBeamPhaseError,
    SetReferenceEnergy,
)
from linac_gen.elements.marker import Marker  # noqa: E402
from linac_gen.elements.matrix_element import MatrixElement  # noqa: E402
from linac_gen.elements.multipole import Multipole  # noqa: E402
from linac_gen.elements.quadrupole import Quadrupole  # noqa: E402
from linac_gen.elements.rf_gap import RFGap  # noqa: E402
from linac_gen.elements.rfq_cell import RfqCell  # noqa: E402
from linac_gen.elements.sc_grid import ScGridDirective  # noqa: E402
from linac_gen.elements.solenoid import Solenoid  # noqa: E402
from linac_gen.elements.space_charge_comp import SpaceChargeComp  # noqa: E402
from linac_gen.elements.steerer import Steerer  # noqa: E402
from linac_gen.elements.thin_lens import ThinLens  # noqa: E402
from linac_gen.tracking.envelope import EnvelopeSolver  # noqa: E402

_IDX = {"sigma_x": 0, "sigma_y": 2, "sigma_phi": 4}


def _cfg():
    return BeamConfig(species="proton", energy=3.0, frequency=352.21, current=0.0,
                      n_particles=1000, alpha_x=-0.5, beta_x=1.0, emit_nx=0.25, alpha_y=0.7,
                      beta_y=1.5, emit_ny=0.3, alpha_z=0.2, beta_z=0.8, emit_z=0.3)


def _tool(lat, **kw):
    ctx = WorkContext()
    ctx.lattice = lat
    ctx.beam_config = _cfg()
    return TOOLS["grad_sensitivities"].fn(ctx, **kw)


def _envelope(lat):
    ref = build_ref(_cfg())
    return EnvelopeSolver(lat, ref, _envelope_initial(_cfg(), ref), current=0.0).run()


def _lattice(elements):
    lat = Lattice()
    for e in elements:
        lat.add(e)
    return lat


def _coupled_map():
    m = np.eye(6)
    m[0, 1] = 0.15; m[1, 0] = -0.2; m[2, 3] = 0.15; m[3, 0] = 0.05; m[1, 2] = 0.03   # noqa: E702
    m[4, 5] = 10.0; m[0, 5] = 3.0; m[4, 0] = 0.2; m[5, 4] = -0.001                   # noqa: E702
    return m


def _every_type(dedx="mip", edit=None):
    """Every element type the tool takes, the knobs among them; a matrix element counted on
    another clock with a reference gain, a foil, a clock card and an energy card."""
    els = [Drift("D1", length=200.0), Quadrupole("QF", length=100.0, gradient=8.0),
           Marker("MK"), Aperture("AP", dx=10.0, dy=10.0), SpaceChargeComp("SCC", factor=0.5),
           ScGridDirective("GRID", extent_sigma=10.0), Steerer("ST", bx_l=1e-4, by_l=2e-4),
           Drift("D2", length=150.0), ThinLens("TL", fx=2.0, fy=-3.0),
           Multipole("MP", knl=[1e-3, 0.3, 5.0], ksl=[0.0, 0.1], tilt_deg=7.0),
           Drift("D3", length=100.0), Solenoid("SOL", length=150.0, field=0.3),
           Edge("E1", pole_rotation=5.0, rho=1000.0, gap=30.0),
           Dipole("B1", angle=10.0, rho=1000.0, e1=5.0, e2=5.0),
           Edge("E2", pole_rotation=5.0, rho=1000.0, gap=30.0), Drift("D4", length=200.0),
           MatrixElement("ME", _coupled_map(), length=150.0, clock_mhz=704.42, ref_gain=0.5),
           Quadrupole("QD", length=100.0, gradient=-8.0, tilt_deg=3.0),
           Foil("FOIL", material="C", thickness_ug_cm2=600.0, dedx_model=dedx),
           Drift("D5", length=300.0), Freq("F", frequency_mhz=704.42),
           MatrixElement("ME2", _coupled_map(), length=50.0, clock_mhz=704.42, ref_gain=-0.2),
           SetReferenceEnergy("E", energy_MeV=0.1, shift=True),
           Quadrupole("QF2", length=100.0, gradient=6.0), Drift("D6", length=200.0)]
    for e in els:
        if edit is not None and e.name == edit[0]:
            setattr(e, edit[1], getattr(e, edit[1]) + edit[2])
    return _lattice(els)


@pytest.mark.parametrize("make", [
    lambda: Marker("X"),
    lambda: Marker("X", is_bpm=True, diag_family=1, x_target_mm=0.0, y_target_mm=0.0),
    lambda: Aperture("X", dx=10.0, dy=5.0), lambda: SpaceChargeComp("X", factor=0.5),
    lambda: ScGridDirective("X", extent_sigma=12.0),
    lambda: Steerer("X", bx_l=1e-4, by_l=-2e-4)],
    ids=["marker", "bpm with targets", "aperture", "compensation card", "grid card", "steerer"])
def test_an_element_that_leaves_the_sizes_alone_changes_nothing(make):
    """The element is the identity on the beam's second moments in the zero-current envelope (a
    steerer moves the centroid, which the rms size does not see): the answer is the line's
    without it, to the last bit."""
    def line(extra):
        return _lattice([Drift("D1", length=200.0), Quadrupole("QF", length=100.0, gradient=8.0),
                         *extra, Drift("D2", length=400.0),
                         Quadrupole("QD", length=100.0, gradient=-8.0),
                         Dipole("B", angle=8.0, rho=900.0), Drift("D3", length=200.0)])

    def rows(d):
        return [(r["name"], r["value"], r["sens"], r["s_m"]) for r in d["ranked"]]

    for kpi in _IDX:
        got = _tool(line([make(), make()]), kpi=kpi)
        assert got["status"] == "ok", got
        want = _tool(line([]), kpi=kpi)["data"]
        assert rows(got["data"]) == rows(want) and got["data"]["kpi_value"] == want["kpi_value"]


@pytest.mark.parametrize("dedx", ["mip", "bethe"])
def test_every_type_follows_the_numpy_envelope_after_each_element(dedx):
    lat = _every_type(dedx)
    env = _envelope(lat)
    for k in range(1, len(lat.elements)):
        sigma = env.sigma_matrix[env.element_exit_idx[k]]
        for kpi, i in _IDX.items():
            res = _tool(_every_type(dedx), kpi=kpi, at_index=k)
            assert res["status"] == "ok", (k, res)
            # the tool rounds to 6 digits
            assert res["data"]["kpi_value"] == pytest.approx(np.sqrt(sigma[i, i]), abs=5e-7), \
                (k, kpi)


_STEP = {"quadrupole": 1e-3, "solenoid": 1e-3, "dipole": 1e-3}


@pytest.mark.parametrize("probe", [9, 16, 18, 19, 21, 24])
@pytest.mark.parametrize("dedx", ["mip", "bethe"])
def test_every_type_gradients_match_central_differences_of_the_envelope(probe, dedx):
    """Richardson-extrapolated central differences through the numpy envelope (h and h/2)."""
    for kpi, i in _IDX.items():
        res = _tool(_every_type(dedx), kpi=kpi, at_index=probe, top_n=50)
        assert res["status"] == "ok", res
        for row in res["data"]["ranked"]:
            def size(dv, row=row, i=i):
                env = _envelope(_every_type(dedx, (row["name"], row["param"], dv)))
                return float(np.sqrt(env.sigma_matrix[env.element_exit_idx[probe]][i, i]))
            h = _STEP[row["kind"]]
            d1 = (size(h) - size(-h)) / (2 * h)
            d2 = (size(h / 2) - size(-h / 2)) / h
            fd = d2 + (d2 - d1) / 3.0
            assert row["sens"] == pytest.approx(fd, rel=1e-6, abs=1e-9), \
                (kpi, row["name"], row["sens"], fd)


def test_a_foil_adds_its_scattering_and_slows_the_reference(monkeypatch):
    """Against the envelope (above) and against the line without it: the foil must matter,
    downstream of it through its scattering and through the quadrupole that sees the slower
    reference (the last part against the same foil with its loss taken away)."""
    def line(foil):
        return _lattice([Drift("D1", length=200.0), Quadrupole("QF", length=100.0, gradient=8.0),
                         *foil, Drift("D2", length=300.0),
                         Quadrupole("QD", length=100.0, gradient=-8.0), Drift("D3", length=200.0)])

    def qd(d):
        return {r["name"]: r["sens"] for r in d["ranked"]}["QD"]

    foil = Foil("F", material="C", thickness_ug_cm2=600.0, dedx_model="bethe")
    for kpi, i in _IDX.items():
        got = _tool(line([foil]), kpi=kpi)["data"]
        env = _envelope(line([copy.deepcopy(foil)]))
        assert got["kpi_value"] == pytest.approx(np.sqrt(env.sigma_matrix[-1][i, i]), abs=5e-7)
        bare = _tool(line([]), kpi=kpi)["data"]
        assert abs(got["kpi_value"] - bare["kpi_value"]) > 1e-3 * bare["kpi_value"], kpi
        if kpi != "sigma_phi":            # (an upright quadrupole does not move sigma_phi here)
            assert qd(got) != pytest.approx(qd(bare), rel=1e-6), kpi
    real = {kpi: qd(_tool(line([foil]), kpi=kpi)["data"]) for kpi in ("sigma_x", "sigma_y")}
    diffusion = Foil.envelope_diffusion
    monkeypatch.setattr(Foil, "envelope_diffusion",
                        lambda self, ref: (diffusion(self, ref)[0], 0.0))
    for kpi, sens in real.items():
        assert sens != pytest.approx(qd(_tool(line([foil]), kpi=kpi)["data"]), rel=1e-6), kpi


@pytest.mark.parametrize("stop", [
    lambda: Foil("F", material="W", thickness_ug_cm2=5e7, dedx_model="bethe"),
    lambda: MatrixElement("M", np.eye(6), ref_gain=-5.0)],
    ids=["a foil that stops the reference", "a matrix element that takes its energy"])
def test_an_element_that_leaves_the_reference_with_no_energy_is_refused(stop):
    """As a card that does it: the numpy envelope carries on below zero energy behind such a
    foil."""
    lat = _lattice([Drift("D1", length=200.0), Quadrupole("Q", length=100.0, gradient=8.0),
                    stop(), Quadrupole("Q2", length=100.0, gradient=-8.0)])
    res = _tool(lat, kpi="sigma_y")
    assert res["status"] == "refused" and "no energy" in res["data"]["message"], res


def test_an_element_that_leaves_the_reference_with_exactly_no_energy_is_refused():
    """At the boundary itself, where the sizes would be NaN: a gain of -W, and a foil whose mean
    loss is W to the last bit (carbon at the minimum-ionisation loss, which does not depend on
    the energy, and the beam at that energy)."""
    foil = Foil("F", material="C", thickness_ug_cm2=1e6, dedx_model="mip")
    w_foil = foil.envelope_diffusion(build_ref(_cfg()))[1]
    for stop, energy in ((MatrixElement("M", np.eye(6), ref_gain=-3.0), 3.0), (foil, w_foil)):
        cfg = _cfg()
        cfg.energy = energy
        ref = build_ref(cfg)
        assert ref.w_kin == energy
        if stop is foil:
            assert foil.envelope_diffusion(ref)[1] == ref.w_kin
        ctx = WorkContext()
        ctx.lattice = _lattice([Drift("D1", length=200.0),
                                Quadrupole("Q", length=100.0, gradient=8.0), stop,
                                Quadrupole("Q2", length=100.0, gradient=-8.0)])
        ctx.beam_config = cfg
        res = TOOLS["grad_sensitivities"].fn(ctx, kpi="sigma_y")
        assert res["status"] == "refused" and "no energy" in res["data"]["message"], res


def test_a_continuous_beam_follows_the_envelope_too():
    """A continuous beam has no bunch, and the envelope seeds no longitudinal block for one; the
    tool took the config's bunch settings, which Continuous only greys out in the Beam tab, so
    behind a bend σ_x and σ_y came out wrong and the ranking flipped (5.02 mm against the
    envelope's 1.88 at 30 keV, 2026-11-08)."""
    cfg = BeamConfig(species="proton", energy=0.03, frequency=162.5, current=0.0,
                     n_particles=1000, continuous=True, alpha_x=-0.5, beta_x=1.0, emit_nx=0.25,
                     alpha_y=0.7, beta_y=1.5, emit_ny=0.3, alpha_z=0.2, beta_z=0.8, emit_z=0.3)
    lat = _lattice([Drift("D1", length=200.0), Solenoid("S1", length=150.0, field=0.2),
                    Drift("D2", length=200.0), Edge("E1", pole_rotation=5.0, rho=500.0, gap=30.0),
                    Dipole("B", angle=20.0, rho=500.0, e1=5.0, e2=5.0),
                    Edge("E2", pole_rotation=5.0, rho=500.0, gap=30.0), Drift("D3", length=200.0),
                    Solenoid("S2", length=150.0, field=0.2), Drift("D4", length=200.0)])
    ref = build_ref(cfg)
    env = EnvelopeSolver(lat, ref, _envelope_initial(cfg, ref), current=0.0).run()
    for kpi, i in _IDX.items():
        ctx = WorkContext()
        ctx.lattice, ctx.beam_config = lat, cfg
        res = TOOLS["grad_sensitivities"].fn(ctx, kpi=kpi)
        assert res["status"] == "ok", res
        assert res["data"]["kpi_value"] == pytest.approx(
            float(np.sqrt(env.sigma_matrix[-1][i, i])), abs=5e-7), kpi


def _rfq_cell():
    return RfqCell("C1", voltage_V=60e3, r0_mm=3.0, A10=0.38, modulation=1.5, length_mm=40.0,
                   phi_s_deg=-30.0, cell_type=2)


def test_an_rfq_cell_is_refused_by_name():
    lat = _lattice([Drift("D1", length=100.0), Quadrupole("Q", length=100.0, gradient=5.0),
                    _rfq_cell()])
    res = _tool(lat, kpi="sigma_y")
    assert res["status"] == "refused" and "RfqCell" in res["data"]["message"], res


def test_a_refusal_lists_what_it_cannot_represent_and_nothing_else():
    lat = _lattice([Drift("D1", length=100.0), Marker("M"),
                    Quadrupole("Q", length=100.0, gradient=5.0),
                    RFGap("G", voltage=0.5, phase=-30.0, frequency=352.21),
                    Steerer("S", by_l=1e-4), _rfq_cell(), Foil("F")])
    res = _tool(lat, kpi="sigma_y")
    assert res["status"] == "refused", res
    msg = res["data"]["message"]
    assert "RFGap×1, RfqCell×1" in msg
    assert not any(n in msg for n in ("Marker", "Steerer", "Foil")), msg


def test_a_class_it_does_not_know_is_refused_whatever_it_is_called():
    """An allowlist, not a scan of names: a kicker with a map of its own and a field map, defined
    outside HELIX's element modules and named like none of the refused families, are refused by
    name next to an RF gap, and the walk itself never takes one as the identity."""
    from linac_gen.assist.tools_grad import _probe_sigma
    from linac_gen.elements.base import ThinKickElement

    class OctupoleKicker(ThinKickElement):
        def apply_kick(self, beam):
            pass

        def kick_matrix(self, ref):
            m = np.eye(6)
            m[1, 0], m[3, 2] = -0.5, 0.5
            return m

    class Buncher(FieldMapElement):
        def __init__(self, name):
            super().__init__(name=name, length=0.0, aperture=0.0, n_steps=1)

        def track_rk4(self, beam, ds):
            pass

    lat = _lattice([Drift("D1", length=200.0), Quadrupole("Q", length=100.0, gradient=8.0),
                    OctupoleKicker("K"), Buncher("BUN"),
                    RFGap("G", voltage=0.5, phase=-30.0, frequency=352.21),
                    Drift("D2", length=300.0)])
    res = _tool(lat, kpi="sigma_y")
    assert res["status"] == "refused", res
    assert "Buncher×1, OctupoleKicker×1, RFGap×1" in res["data"]["message"], res
    with pytest.raises(ValueError, match="OctupoleKicker"):
        _probe_sigma(lat, build_ref(_cfg()), torch.eye(6, dtype=torch.float64), 2, {})


def test_a_probe_upstream_of_what_it_cannot_represent_answers():
    """Only the span up to the probe is read: upstream of an RF gap and an RFQ cell the answer is
    the line's cut at the probe."""
    def line(rest):
        return _lattice([Drift("D1", length=200.0), Quadrupole("Q", length=100.0, gradient=8.0),
                         Drift("D2", length=300.0), *rest])

    for kpi in _IDX:
        got = _tool(line([RFGap("G", voltage=0.5, phase=-30.0, frequency=352.21), _rfq_cell()]),
                    kpi=kpi, at_index=2)
        assert got["status"] == "ok", got
        want = _tool(line([]), kpi=kpi)
        assert got["data"]["ranked"] == want["data"]["ranked"], kpi
        assert got["data"]["kpi_value"] == want["data"]["kpi_value"], kpi


def _classes():
    import linac_gen.elements as pkg
    for m in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(f"linac_gen.elements.{m.name}")
    import linac_gen.surrogates.base  # noqa: F401

    def walk(c):
        for s in c.__subclasses__():
            yield s
            yield from walk(s)
    return [c for c in dict.fromkeys(walk(Element)) if not inspect.isabstract(c)
            and c.__module__.startswith("linac_gen.")]


_REAL = {
    "Drift": lambda: Drift("X", length=100.0),
    "Quadrupole": lambda: Quadrupole("X", length=100.0, gradient=3.0),
    "Solenoid": lambda: Solenoid("X", length=100.0, field=0.2),
    "Dipole": lambda: Dipole("X", angle=10.0, rho=1e3),
    "Edge": lambda: Edge("X", pole_rotation=5.0, rho=1000.0, gap=30.0),
    "MatrixElement": lambda: MatrixElement("X", _coupled_map(), clock_mhz=704.42, ref_gain=0.1),
    "Aperture": lambda: Aperture("X", dx=10.0, dy=10.0), "Marker": lambda: Marker("X"),
    "Steerer": lambda: Steerer("X", bx_l=1e-4, by_l=2e-4),
    "Multipole": lambda: Multipole("X", knl=[0.0, 0.3, 5.0]),
    "ThinLens": lambda: ThinLens("X", fx=2.0, fy=-3.0), "Foil": lambda: Foil("X"),
    "SpaceChargeComp": lambda: SpaceChargeComp("X", factor=0.5),
    "ScGridDirective": lambda: ScGridDirective("X", extent_sigma=10.0),
    "SetBeamPhaseError": lambda: SetBeamPhaseError("X", dphi_deg=2.0),
    "SetReferenceEnergy": lambda: SetReferenceEnergy("X", energy_MeV=4.0),
    "Freq": lambda: Freq("X", frequency_mhz=704.42),
}


def test_no_element_class_makes_it_fail():
    """Every element class answers or is refused by name, never an error: the RF and field-map
    family (whose maps follow the reference energy they change) is refused, everything else
    answers."""
    from linac_gen.elements.lattice_commands import LatticeCommand
    for cls in _classes():
        if cls.__name__ in _REAL:
            x = _REAL[cls.__name__]()
        elif issubclass(cls, LatticeCommand):
            x = cls("X")
        else:                                    # refused before any of it is read
            x = cls.__new__(cls)
            Element.__init__(x, name="X", length=100.0, aperture=0.0, n_steps=1)
        lat = _lattice([Drift("D1", length=200.0), Quadrupole("Q", length=100.0, gradient=8.0), x,
                        Drift("D2", length=300.0)])
        res = _tool(lat, kpi="sigma_y")
        rf = issubclass(cls, (FieldMapElement, RFGap))
        assert res["status"] == ("refused" if rf else "ok"), (cls.__name__, res)
        if rf:
            assert cls.__name__ in res["data"]["message"], (cls.__name__, res)


def test_a_magnet_s_subclass_is_refused_and_a_passive_element_s_answers():
    """A magnet is taken by its exact class: a subclass's torch matrix would be its base class's,
    whatever map it has (a quadrupole subclass with a map of its own came out with the plain
    quadrupole's sizes, 2026-11-08).  A subclass of a passive element enters with its own map,
    as in the other modes."""
    class FringedQuad(Quadrupole):
        pass

    class SlantedFace(Edge):                     # an Edge is a passive element as well
        pass

    class Screen(Marker):
        pass

    for x, refused in ((FringedQuad("X", length=100.0, gradient=3.0), True),
                       (SlantedFace("X", pole_rotation=5.0, rho=1000.0, gap=30.0), True),
                       (Screen("X"), False)):
        lat = _lattice([Drift("D1", length=200.0), Quadrupole("Q", length=100.0, gradient=8.0), x,
                        Drift("D2", length=300.0)])
        res = _tool(lat, kpi="sigma_y")
        assert res["status"] == ("refused" if refused else "ok"), (type(x).__name__, res)
        if refused:
            assert f"{type(x).__name__}×1" in res["data"]["message"], res


_DAT = """\
; a transfer line with the diagnostic cards a real one carries
MARKER
DRIFT 200 20
BPM
QUAD 100 8 20
DIAG_POSITION 1 0 0 1
DRIFT 150 20
APERTURE 15 15 0
THIN_STEERING 0.0001 -0.0002 20
LATTICE 4 0
DRIFT 150 20
QUAD 100 -8 20
DIAG_SIZE
DRIFT 150 20
LATTICE_END
SPACE_CHARGE_COMP 0.5
; HELIX_SC_GRID 20
EDGE 4 1000 30 0.45 2.8 20 0
BEND 8 1000 0 20 0
EDGE 4 1000 30 0.45 2.8 20 0
DRIFT 200 20
; HELIX_FOIL F1 C 600
MARKER
QUAD 100 6 20
DRIFT 200 20
END
"""


def _strip(lat):
    keep = [e for e in lat.elements
            if not isinstance(e, (Marker, Aperture, SpaceChargeComp, ScGridDirective, Steerer))]
    return _lattice(keep)


def _deck_case(ctx):
    cfg = ctx.beam_config
    ref = build_ref(cfg)
    env = EnvelopeSolver(ctx.lattice, ref, _envelope_initial(cfg, ref), current=0.0).run()
    stripped = WorkContext()
    stripped.lattice, stripped.beam_config = _strip(ctx.lattice), cfg

    def rows(d):
        return [(r["name"], r["value"], r["sens"]) for r in d["ranked"]]

    for kpi, i in _IDX.items():
        got = TOOLS["grad_sensitivities"].fn(ctx, kpi=kpi, top_n=1000)
        assert got["status"] == "ok", got
        assert got["data"]["kpi_value"] == pytest.approx(
            float(np.sqrt(env.sigma_matrix[-1][i, i])), abs=5e-7)
        want = TOOLS["grad_sensitivities"].fn(stripped, kpi=kpi, top_n=1000)["data"]
        assert rows(got["data"]) == rows(want)


def test_a_line_with_diagnostic_cards_loaded_from_a_deck(tmp_path):
    """The parser's markers, monitors, aperture, steerer and cards reach the tool as they do
    from a deck."""
    deck = tmp_path / "line.dat"
    deck.write_text(_DAT)
    ctx = WorkContext()
    assert TOOLS["load_lattice"].fn(ctx, path=str(deck))["status"] == "ok"
    kinds = {type(e).__name__ for e in ctx.lattice.elements}
    assert {"Marker", "Aperture", "Steerer", "SpaceChargeComp", "ScGridDirective",
            "Foil"} <= kinds, kinds
    ctx.beam_config = _cfg()
    _deck_case(ctx)


def test_a_line_of_the_types_it_took_before_gives_the_matrix_path_s_answer_to_the_last_bit():
    """No foil and no constant map: the walk composes the map as compute_transfer_matrix_torch
    does, so Σ and every gradient are what the tool gave before, bit for bit."""
    from linac_gen.assist.tools_grad import _probe_sigma
    from linac_gen.distributions.factory import create_beam, geometric_emittances
    from linac_gen.tracking.envelope import _build_sigma_matrix
    from linac_gen.tracking.torch_matrices import F64
    from linac_gen.tracking.torch_tracking import compute_transfer_matrix_torch

    def line():
        return _lattice([
            Drift("D1", length=200.0), Quadrupole("QF", length=100.0, gradient=8.0, tilt_deg=2.0),
            Solenoid("SOL", length=150.0, field=0.3),
            Edge("E1", pole_rotation=5.0, rho=1000.0, gap=30.0),
            Dipole("B1", angle=10.0, rho=1000.0, e1=5.0, e2=5.0),
            Edge("E2", pole_rotation=5.0, rho=1000.0, gap=30.0), Freq("F", frequency_mhz=704.42),
            SetReferenceEnergy("E", energy_MeV=0.1, shift=True),
            Quadrupole("QD", length=100.0, gradient=-8.0), Drift("D2", length=300.0)])

    cfg = _cfg()
    out = []
    for path in ("walk", "matrix"):
        lat = line()
        knobs, overrides = [], {}
        for e in lat.elements:
            attr = {"Quadrupole": "gradient", "Solenoid": "field", "Dipole": "angle"}.get(
                type(e).__name__)
            if attr:
                t = torch.tensor(float(getattr(e, attr)), dtype=F64, requires_grad=True)
                overrides[id(e)] = t
                knobs.append(t)
        ref = create_beam(cfg, seed=42).ref
        ex, ey, ez = geometric_emittances(cfg, max(float(ref.bg), 1e-9))
        s0 = torch.as_tensor(_build_sigma_matrix(cfg.alpha_x, cfg.beta_x, ex, cfg.alpha_y,
                                                 cfg.beta_y, ey, cfg.alpha_z, cfg.beta_z, ez),
                             dtype=F64)
        last = len(lat.elements) - 1
        if path == "walk":
            S = _probe_sigma(lat, ref, s0, last, overrides)
        else:
            M = compute_transfer_matrix_torch(lat, ref, start=0, end=last, overrides=overrides,
                                              on_nonlinear="error")
            S = M @ s0 @ M.T
        grads = torch.autograd.grad(torch.sqrt(S[2, 2]), knobs)
        out.append((S.detach().numpy().copy(), [float(g) for g in grads]))
    assert np.array_equal(out[0][0], out[1][0]) and out[0][1] == out[1][1]
