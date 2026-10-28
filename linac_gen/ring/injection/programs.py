"""Bump and painting magnets: their waveforms written into the compiled ring
turn by turn.

Each magnet group (:class:`~linac_gen.ring.injection.config.MagnetGroup`)
names elements of the ring — every element of that name, wherever it is
placed — and how their strength follows the group's waveform w(t):

* ``"kick"`` — a kicker's deflection: kick_rad x w(t) in the group's plane
  (without kick_rad, the element's own kick x w(t)); with a plane, the other
  plane keeps the element's own setting;
* ``"dipole"`` — a bump magnet compiled as an exact dipole
  (:mod:`linac_gen.ring.compile`): its field k0 = -kick/L x w(t) in the body
  and at both faces;
* ``"scale"`` — the element's own kicks and multipole strengths x w(t) (the
  error multipoles of a bump magnet, for one).

The strengths are normalized (deflections at the reference momentum of the
moment), as a ramp's magnets are: a waveform gives the deflection whatever
the energy.
"""
from __future__ import annotations

import numpy as np

from linac_gen.ring.injection.config import InjectionError
from linac_gen.ring.kernels import cartesian as cart
from linac_gen.ring.kernels import nodes as nd


class MagnetPrograms:
    """The magnet groups of an injection on a compiled ring: :meth:`apply`
    writes them at time t into a parameter buffer."""

    def __init__(self, groups, rl):
        self.groups = tuple(groups)
        self.names = [g.name for g in self.groups]
        index = {}
        for k, name in enumerate(rl.element_names):
            index.setdefault(name.lower(), []).append(k)
        node_elem = np.asarray(rl.node_element)
        self._slots = []          # per group: (indices into params, base values)
        self.summary = []
        for g in self.groups:
            idx, base = [], []
            for k, name in enumerate(g.elements):
                elems = index.get(name.lower())
                if not elems:
                    raise InjectionError(f"injection: magnet group {g.name!r}: element {name!r} "
                                         "is not in the ring")
                kick = None if g.kick_rad is None else float(g.kick_rad[k])
                if g.model == "dipole":
                    got = [b for b in rl.bumps if b["element"].lower() == name.lower()]
                    if not got:
                        raise InjectionError(f"injection: magnet group {g.name!r}: {name!r} was "
                                             "not compiled as a bump dipole")
                    for b in got:
                        j_in, j_body, j_out = b["nodes"]
                        o = int(rl.offsets[j_body])
                        idx.append(o + cart.MG_K0_DRIFT)
                        base.append(b["k0"])
                        for j in (j_in, j_out):
                            o = int(rl.offsets[j])
                            for off in (nd.EE_K0_FRINGE, nd.EE_K0_WEDGE, nd.EE_KN):
                                idx.append(o + off)
                                base.append(b["k0"])
                    continue
                for e in elems:
                    nodes = np.flatnonzero(node_elem == e)
                    kinds = np.asarray(rl.kinds)[nodes]
                    if g.model == "kick":
                        ks = nodes[kinds == nd.N_KICK]
                        if not ks.size:
                            raise InjectionError(f"injection: magnet group {g.name!r}: {name!r} "
                                                 "is not a kicker (the 'kick' model)")
                        for j in ks:
                            o = int(rl.offsets[j])
                            dpx, dpy = float(rl.params[o]), float(rl.params[o + 1])
                            plane = g.plane
                            if kick is not None and plane is None:
                                if dpx and dpy:
                                    raise InjectionError(
                                        f"injection: magnet group {g.name!r}: {name!r} kicks "
                                        "in both planes — give the group's plane")
                                if not dpx and not dpy:
                                    raise InjectionError(
                                        f"injection: magnet group {g.name!r}: {name!r} has "
                                        "no kick of its own to tell its plane — give the "
                                        "group's plane")
                                plane = "x" if dpx else "y"
                            if plane is None:
                                # (the element's own kicks, both planes)
                                idx += [o, o + 1]
                                base += [dpx, dpy]
                            else:
                                # (only the plane's deflection follows the waveform:
                                # the other keeps the element's own setting)
                                idx.append(o if plane == "x" else o + 1)
                                base.append(kick if kick is not None else
                                            (dpx if plane == "x" else dpy))
                    else:
                        found = False
                        for j, kd in zip(nodes, kinds, strict=True):
                            o = int(rl.offsets[j])
                            if kd == nd.N_KICK:
                                idx += [o, o + 1]
                                base += [float(rl.params[o]), float(rl.params[o + 1])]
                                found = True
                            elif kd == nd.N_MULTIPOLE:
                                order = int(rl.params[o])
                                for c in range(2 * (order + 1)):
                                    idx.append(o + 2 + c)
                                    base.append(float(rl.params[o + 2 + c]))
                                found = True
                        if not found:
                            raise InjectionError(f"injection: magnet group {g.name!r}: {name!r} "
                                                 "has no kick or multipole to scale (the "
                                                 "'scale' model)")
            self._slots.append((np.asarray(idx, dtype=np.int64), np.asarray(base, dtype=float)))
            self.summary.append({"name": g.name, "model": g.model, "elements": list(g.elements),
                                 "slots": len(idx)})

    def values(self, t: float) -> np.ndarray:
        """Each group's waveform at time t."""
        return np.array([g.waveform(t) for g in self.groups], dtype=float)

    def apply(self, t: float, params) -> np.ndarray:
        """Write every group at time t into ``params`` (in place); returns the
        waveforms' values."""
        w = self.values(t)
        for (idx, base), wk in zip(self._slots, w, strict=True):
            params[idx] = base * wk
        return w
