"""Test rings for injection (ring mode, milestone M5): the simple Booster with a
foil marker, and with a closed four-magnet bump (thick kickers, kicks +theta,
-theta, -theta, +theta) around the foil in its first long straight."""
from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DECK = ROOT / "tests" / "ring" / "fixtures" / "simple_booster.madx"
C_LIGHT = 299792458.0
#: the simple Booster at 800 MeV (its deck's beam): circumference, beta0
LENGTH_M = 474.202752
BETA0 = 0.841810706


def revolution_time() -> float:
    return LENGTH_M / (BETA0 * C_LIGHT)


def load_booster():
    from linac_gen.cli.common import load_lattice
    return load_lattice(str(DECK))


def booster_beam(**kw):
    from linac_gen.core.config import BeamConfig
    d = dict(species="proton", energy=800.0, frequency=44.7044096, current=0.0)
    d.update(kw)
    return BeamConfig(**d)


def with_foil_marker(lat, name: str = "foil", index: int = 1):
    """The lattice with a marker ``name`` inserted at ``index``."""
    from linac_gen.elements.marker import Marker
    lat.elements.insert(index, Marker(name=name))
    return lat


def bumped(theta: float = 0.02, length: float = 0.3, gap12: float = 1.0, gap23: float = 2.0,
           plane: str = "y", lat=None):
    """The simple Booster's first long straight ('dlong') with kicker centres at a,
    a + gap12, a + gap12 + gap23, a + 2 gap12 + gap23 [m] (thick kickers of
    ``length``, written as the importers write them: X_din, X, X_dout), named
    kb1..kb4, and a marker 'foil' midway between kb2 and kb3."""
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.marker import Marker
    from linac_gen.elements.steerer import Steerer
    from linac_gen.ring.compile import lattice_rigidity
    lat = load_booster() if lat is None else lat
    els = list(lat.elements)
    i = next(k for k, e in enumerate(els) if str(e.name) == "dlong")
    total = els[i].length                       # mm
    brho = lattice_rigidity("proton", 800.0)
    lm = length * 1e3
    a = (total - (2 * gap12 + gap23 + length) * 1e3) / 2.0

    def kick(name, th):
        hk, vk = (th, 0.0) if plane == "x" else (0.0, th)
        return [Drift(name=f"{name}_din", length=lm / 2),
                Steerer(name=name, by_l=hk * brho, bx_l=vk * brho),
                Drift(name=f"{name}_dout", length=lm / 2)]

    seq = [Drift(name="bump_a", length=a)] + kick("kb1", theta)
    seq += [Drift(name="bump_b1", length=gap12 * 1e3 - lm)] + kick("kb2", -theta)
    seq += [Drift(name="bump_m1", length=gap23 * 1e3 / 2 - lm / 2), Marker(name="foil"),
            Drift(name="bump_m2", length=gap23 * 1e3 / 2 - lm / 2)]
    seq += kick("kb3", -theta) + [Drift(name="bump_b2", length=gap12 * 1e3 - lm)]
    seq += kick("kb4", theta) + [Drift(name="bump_c", length=a)]
    assert abs(sum(e.length for e in seq) - total) < 1e-9
    lat.elements[i:i + 1] = seq
    return lat


def bump_height(theta: float, length: float, gap12: float) -> float:
    """The exact height of that bump at the foil for a particle of delta 0:
    through the first magnet (a uniform field, px from 0 to theta) its offset
    grows by L theta/(pz + 1), then by (gap12 - L) theta/pz over the drift,
    and again by L theta/(pz + 1) through the second (theta back to 0)."""
    pz = math.sqrt(1.0 - theta * theta)
    return 2.0 * length * theta / (pz + 1.0) + (gap12 - length) * theta / pz
