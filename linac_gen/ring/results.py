"""Ring results in HDF5 (``run_type = "ring"``, schema 3) and back.

Layout (ring coordinates, SI units)::

    /                      attrs: run_type, schema_version, engine, n_turns_requested,
                           truncated, wall_s
    /turns/                turn (k), n_alive (k), mean (k, 6), cov (k, 6, 6),
                           beta0gamma0 (k), p0c_mev (k): the reference of each row
    /ramp/                 a ramp's turn-by-turn record (per turn, at its start): t_s,
                           kinetic_mev, gain_mev, voltage_mv, phi_deg, frequency_hz,
                           above_transition, capped; attrs: transition_turn, gamma_t,
                           n_capped, the end, the program (JSON) — absent at fixed energy
    /snapshots/turn_<t>/   coords (6, m), pid (m); attrs: the reference and RF of the turn
                           (a ramp)
    /injection/            with a foil: per turn injected, stripped, h0, hminus, missed,
                           injected_charge, foil_hits, foil_hits_injected, deposit_mev,
                           electron_mev (real particles and energies: the ring's);
                           closed_orbit_foil (turns + 1, 4), waveforms (turns + 1, groups)
                           and waveform_names, hit_map and deposit_map_mev (nx, ny) with
                           map_extent_m, hits_histogram; attrs: the foil, the settings, the
                           source, the waste, mean_hits_circulating, buckets, notes
    /losses/               pid, turn, node, element, s, cause, coords (6, n)
    /probes/               tbt (n, min(turns, 4096) + 1, 6), amplitudes_sigma (n, 2),
                           tunes (n, 3), diffusion (n), state (n)
    /tunes                 attrs: the NAFF tunes of the smallest probe (qx, qy, qs)
    /optics                attrs: the periodic optics of the compiled ring
    /optics_table/         the run's periodic optics at every element's exit: name, type
                           (strings), s, betx, alfx, bety, alfy, mux, muy, dx, dpx, dy, dpy,
                           x, px, y, py — absent in files written before 2026-10-25
    /ledger                attrs: injected / circulating / lost / waste
    /meta                  attrs (scalars), config and options as JSON (with space
                           charge, its settings, kicks and analytic tune shift as
                           ``space_charge``); element_names, element_s, element_length
    /beam_config/          the beam settings the particles came from (attrs), when given
    /provenance/           code version, git commit, lattice hash, seed, threads

The coordinates are those of :mod:`linac_gen.ring.coords`: x, y [m],
px, py = P/P0, zeta = s - beta0 c t [m], delta = (P - P0)/P0.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 3
#: the per-turn arrays of a ramp's record
_RAMP_ARRAYS = ("t_s", "kinetic_mev", "gain_mev", "voltage_mv", "phi_deg", "frequency_hz",
                "above_transition", "capped")
#: the arrays of the injection's record (the rest of it: attributes)
_INJECTION_ARRAYS = ("injected", "stripped", "h0", "hminus", "missed", "injected_charge",
                     "foil_hits", "foil_hits_injected", "deposit_mev", "electron_mev",
                     "closed_orbit_foil", "waveforms", "hit_map", "deposit_map_mev",
                     "map_extent_m", "hits_histogram")


def _attrs(g, d: dict) -> None:
    """Scalars as attributes, lists of numbers as datasets, anything else as JSON."""
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, (bool, np.bool_)):
            g.attrs[k] = bool(v)
        elif isinstance(v, (int, float, str, np.integer, np.floating)):
            g.attrs[k] = v
        elif isinstance(v, (list, tuple, np.ndarray)) and len(v) and all(
                isinstance(x, (int, float, np.integer, np.floating)) for x in np.ravel(v)):
            g.create_dataset(k, data=np.asarray(v, dtype=float))
        else:
            g.attrs[k + "__json"] = json.dumps(v, default=_json_default)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def _read_attrs(g, datasets: bool = True) -> dict:
    """A group's attributes (and, with ``datasets``, its datasets as lists)."""
    out = {}
    for k, v in g.attrs.items():
        if k.endswith("__json"):
            out[k[:-6]] = json.loads(v)
        elif isinstance(v, np.generic):
            out[k] = v.item()
        else:
            out[k] = v
    if datasets:
        for k, ds in g.items():
            out[k] = ds[()].tolist() if ds.ndim else ds[()]
    return out


def survival_text(n0: int, n1: int) -> str:
    """'99.996 % (4 of 100000 lost)': the percentage with the digits the count
    needs — a loss never shows as 100 %."""
    if n0 <= 0:
        return "n/a"
    lost = n0 - n1
    if lost == 0:
        return "100 % (none lost)"
    pct = 100.0 * n1 / n0
    digits = 1
    while digits < 12 and float(f"{pct:.{digits}f}") >= 100.0:
        digits += 1
    return f"{pct:.{digits}f} % ({lost} of {n0} lost)"


def save_ring_hdf5(res, path, *, lattice_path=None, beam_config=None) -> Path:
    """Write ``res`` (a :class:`~linac_gen.ring.engine.RingResults`) to ``path``
    (to a temporary file first, renamed when complete); ``lattice_path`` goes
    into the provenance (with the file's hash), ``beam_config`` (the
    BeamConfig the particles came from) into ``beam_config/``."""
    import h5py

    from linac_gen.io.hdf5_output import _write_provenance
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with h5py.File(tmp, "w") as f:
        f.attrs["run_type"] = "ring"
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["engine"] = res.engine
        f.attrs["n_turns_requested"] = int(res.n_turns_requested)
        f.attrs["truncated"] = bool(res.truncated)
        f.attrs["wall_s"] = float(res.wall_s)
        t = f.create_group("turns")
        t.create_dataset("turn", data=np.asarray(res.turns, dtype=np.int64))
        t.create_dataset("n_alive", data=np.asarray(res.n_alive, dtype=np.int64))
        t.create_dataset("mean", data=np.asarray(res.mean, dtype=float))
        t.create_dataset("cov", data=np.asarray(res.cov, dtype=float))
        if getattr(res, "row_beta0gamma0", None) is not None:
            t.create_dataset("beta0gamma0", data=np.asarray(res.row_beta0gamma0, dtype=float))
            t.create_dataset("p0c_mev", data=np.asarray(res.row_p0c_mev, dtype=float))
        ramp = dict(getattr(res, "ramp", None) or {})
        if ramp:
            rg = f.create_group("ramp")
            for k in _RAMP_ARRAYS:
                if k in ramp:
                    v = np.asarray(ramp.pop(k))
                    rg.create_dataset(k, data=v.astype(bool) if v.dtype == bool else
                                      v.astype(float))
            _attrs(rg, ramp)
        s = f.create_group("snapshots")
        for turn, snap in sorted(res.snapshots.items()):
            g = s.create_group(f"turn_{int(turn)}")
            g.create_dataset("coords", data=np.asarray(snap["coords"], dtype=float))
            g.create_dataset("pid", data=np.asarray(snap["pid"], dtype=np.int64))
            if snap.get("reference"):
                _attrs(g, dict(snap["reference"]))
        inj = dict(getattr(res, "injection", None) or {})
        if inj:
            ig = f.create_group("injection")
            for k in _INJECTION_ARRAYS:
                v = inj.pop(k, None)
                if v is not None:
                    ig.create_dataset(k, data=np.asarray(v, dtype=np.int64 if k == "hits_histogram"
                                                         else float))
            names = inj.pop("waveform_names", None)
            if names is not None:
                ig.create_dataset("waveform_names", data=np.array(
                    [str(n) for n in names], dtype=h5py.string_dtype()))
            _attrs(ig, inj)
        lg = f.create_group("losses")
        for k, v in (res.losses or {}).items():
            lg.create_dataset(k, data=np.asarray(v))
        pg = f.create_group("probes")
        for k, v in (res.probes or {}).items():
            if v is not None:
                pg.create_dataset(k, data=np.asarray(v))
        _attrs(f.create_group("tunes"), dict(res.tunes or {}))
        _attrs(f.create_group("optics"), dict(res.optics or {}))
        table = dict(getattr(res, "optics_table", None) or {})
        if table:
            og = f.create_group("optics_table")
            for k, v in table.items():
                if k in ("name", "type"):
                    og.create_dataset(k, data=np.array([str(x) for x in v],
                                                       dtype=h5py.string_dtype()))
                else:
                    og.create_dataset(k, data=np.asarray(v, dtype=float))
        _attrs(f.create_group("ledger"), dict(res.ledger or {}))
        meta = dict(res.meta or {})
        mg = f.create_group("meta")
        names = meta.pop("element_names", None)
        if names is not None:
            mg.create_dataset("element_names", data=np.array([str(n) for n in names],
                                                             dtype=h5py.string_dtype()))
        for k in ("element_s", "element_length"):
            v = meta.pop(k, None)
            if v is not None:
                mg.create_dataset(k, data=np.asarray(v, dtype=float))
        _attrs(mg, meta)
        if beam_config is not None:
            bg = f.create_group("beam_config")
            for key, val in vars(beam_config).items():
                if val is None:
                    continue
                try:
                    bg.attrs[key] = val
                except TypeError:                          # (not encodable: as text)
                    bg.attrs[key] = str(val)
        cfg = meta.get("config") or {}
        _write_provenance(f, lattice_path=lattice_path, seed=cfg.get("seed"))
    tmp.replace(path)
    return path


def load_ring_hdf5(path):
    """A :class:`~linac_gen.ring.engine.RingResults` from a file written by
    :func:`save_ring_hdf5`."""
    import h5py

    from linac_gen.ring.engine import RingResults
    with h5py.File(path, "r") as f:
        if f.attrs.get("run_type") != "ring":
            raise ValueError(f"{path}: not a ring-mode results file "
                             f"(run_type {f.attrs.get('run_type')!r})")
        ver = int(f.attrs.get("schema_version", 0))
        if ver > SCHEMA_VERSION:
            raise ValueError(f"{path}: ring results schema {ver} is newer than this HELIX's "
                             f"({SCHEMA_VERSION})")
        t = f["turns"]
        snaps = {}
        for name, g in f["snapshots"].items():
            snap = {"coords": g["coords"][()], "pid": g["pid"][()]}
            # (the reference: attributes only — the particles are read once)
            ref = _read_attrs(g, datasets=False)
            if ref:
                snap["reference"] = ref
            snaps[int(name.split("_", 1)[1])] = snap
        ramp = {}
        if "ramp" in f:
            ramp = _read_attrs(f["ramp"], datasets=False)
            for k in _RAMP_ARRAYS:
                if k in f["ramp"]:
                    ramp[k] = f["ramp"][k][()]
            for k in ("transition_turn", "turns_to_end", "gamma_t",
                      "above_transition_at_start"):              # (None is not written)
                ramp.setdefault(k, None)
        injection = {}
        if "injection" in f:
            g = f["injection"]
            injection = _read_attrs(g, datasets=False)
            for k in _INJECTION_ARRAYS:
                if k in g:
                    injection[k] = g[k][()]
            if "map_extent_m" in injection:
                injection["map_extent_m"] = [float(v) for v in injection["map_extent_m"]]
            if "waveform_names" in g:
                injection["waveform_names"] = [n.decode() if isinstance(n, bytes) else str(n)
                                               for n in g["waveform_names"][()]]
        losses = {k: v[()] for k, v in f["losses"].items()}
        optics_table = {}
        if "optics_table" in f:
            for k, v in f["optics_table"].items():
                optics_table[k] = ([x.decode() if isinstance(x, bytes) else str(x) for x in v[()]]
                                   if k in ("name", "type") else v[()])
        probes = {k: v[()] for k, v in f["probes"].items()}
        meta = _read_attrs(f["meta"])
        if "element_names" in f["meta"]:
            meta["element_names"] = [n.decode() if isinstance(n, bytes) else str(n)
                                     for n in f["meta"]["element_names"][()]]
        for k in ("element_s", "element_length"):
            if k in f["meta"]:
                meta[k] = f["meta"][k][()]
        res = RingResults(
            engine=str(f.attrs["engine"]), n_turns_requested=int(f.attrs["n_turns_requested"]),
            turns=t["turn"][()], n_alive=t["n_alive"][()], mean=t["mean"][()], cov=t["cov"][()],
            snapshots=snaps, losses=losses, probes=probes, tunes=_read_attrs(f["tunes"]),
            optics=_read_attrs(f["optics"]), ledger=_read_attrs(f["ledger"]),
            truncated=bool(f.attrs["truncated"]), wall_s=float(f.attrs["wall_s"]), meta=meta,
            row_beta0gamma0=t["beta0gamma0"][()] if "beta0gamma0" in t else None,
            row_p0c_mev=t["p0c_mev"][()] if "p0c_mev" in t else None, ramp=ramp,
            injection=injection, optics_table=optics_table)
    return res
