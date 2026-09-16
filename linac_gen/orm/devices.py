"""Device ↔ element mapping for orbit-response matrices.

Measured matrices are keyed by control-system device names (``L:D21BPH``,
``L:D21TMH``); the lattice knows BPM markers (``Marker.is_bpm``) and steerers
(``Steerer``) by their deck labels (``D21BPM``, ``D21T`` — ``element.label``)
or generated names.  A :class:`DeviceMap` says which element each device
reads or drives; :func:`resolve_devices` turns it into index tables
(:class:`OrmSelection`) every later stage consumes.  Unmatched devices and
unmatched elements are always listed, never dropped silently.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from linac_gen.elements.marker import Marker
from linac_gen.elements.steerer import Steerer

#: Fermilab linac naming: ``L:DnnBPH`` / ``L:DnnBPV`` read BPM marker ``DnnBPM``,
#: ``L:DnnTMH`` / ``L:DnnTMV`` drive trim ``DnnT``; the tank-5 exit BPM is the
#: SCL deck's first BPM.
FNAL_SCL_ALIASES = {"L:BPH5OT": "D01BPM", "L:BPV5OT": "D01BPM"}
_RULES = (
    (re.compile(r"^L:(D\d\d)BP[HV]$"), r"\1BPM", "bpm"),
    (re.compile(r"^L:(D\d\d)TM[HV]$"), r"\1T", "trim"),
)
_PLANE_OF = {"H": "x", "V": "y"}


def element_label(e) -> str:
    """Deck label when the element has one, else its name."""
    return getattr(e, "label", None) or getattr(e, "name", "")


def device_plane(device: str) -> str | None:
    """``"x"`` for a horizontal device (…H), ``"y"`` for a vertical one (…V)."""
    return _PLANE_OF.get(device.strip()[-1:].upper())


@dataclass
class DeviceMap:
    """Editable mapping (JSON on the CLI, table in the GUI)."""

    bpms: dict = field(default_factory=dict)        # device -> element label or name
    trims: dict = field(default_factory=dict)
    bpm_sign: dict = field(default_factory=dict)    # device -> +1 / -1 (inverted polarity)
    exclude_bpms: set = field(default_factory=set)  # devices (or labels) left out of compare/fit
    exclude_trims: set = field(default_factory=set)
    rules: str = "pip2_forma"

    def to_dict(self) -> dict:
        return {"__kind__": "helix_orm_device_map", "__version__": 1, "rules": self.rules,
                "bpms": dict(self.bpms), "trims": dict(self.trims),
                "bpm_sign": {k: int(v) for k, v in self.bpm_sign.items()},
                "exclude_bpms": sorted(self.exclude_bpms), "exclude_trims": sorted(self.exclude_trims)}

    @classmethod
    def from_dict(cls, d: dict) -> "DeviceMap":
        if d.get("__kind__") != "helix_orm_device_map":
            raise ValueError("not a HELIX ORM device map (missing __kind__)")
        return cls(bpms=dict(d.get("bpms", {})), trims=dict(d.get("trims", {})),
                   bpm_sign={k: int(v) for k, v in (d.get("bpm_sign") or {}).items()},
                   exclude_bpms=set(d.get("exclude_bpms", [])), exclude_trims=set(d.get("exclude_trims", [])),
                   rules=str(d.get("rules", "custom")))

    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def from_json(cls, path) -> "DeviceMap":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def default_device_map(lattice, measured: dict, *, aliases=None) -> DeviceMap:
    """Apply the naming rules to every device of ``measured`` (``{"x": MeasuredOrm, …}``).

    Devices no rule covers are left out of the map (they appear in
    :attr:`OrmSelection.unmatched_devices`).  When the lattice carries no
    labels at all (hand-built lattices), devices equal to element names map
    to themselves.
    """
    aliases = FNAL_SCL_ALIASES if aliases is None else aliases
    names = {getattr(e, "name", None) for e in lattice.elements}
    labels = {getattr(e, "label", None) for e in lattice.elements} - {None}
    dm = DeviceMap()
    for m in measured.values():
        for dev in m.trims:
            key = _rule(dev, "trim", aliases)
            if key is None and (dev in labels or dev in names):
                key = dev
            if key is not None:
                dm.trims[dev] = key
        for rp in m.read_planes():
            for dev in m.bpms[rp]:
                key = _rule(dev, "bpm", aliases)
                if key is None and (dev in labels or dev in names):
                    key = dev
                if key is not None:
                    dm.bpms[dev] = key
    return dm


def _rule(device: str, kind: str, aliases: dict):
    if device in aliases:
        return aliases[device]
    for rx, repl, k in _RULES:
        if k == kind and rx.match(device):
            return rx.sub(repl, device)
    return None


@dataclass
class OrmSelection:
    """Index tables of the matched BPMs / trims (lattice order) plus everything that did not match."""

    bpm_idx: np.ndarray                 # element indices of the matched BPM markers
    bpm_labels: list
    bpm_devices: dict                   # read plane -> [device or None per BPM]
    bpm_sign: dict                      # read plane -> (+1/-1) array
    bpm_include: dict                   # read plane -> bool array (False = excluded or absent)
    trim_idx: np.ndarray
    trim_labels: list
    trim_devices: dict                  # kick plane -> [device or None per trim]
    trim_include: dict                  # kick plane -> bool array
    s_bpm_m: np.ndarray
    s_trim_m: np.ndarray
    down: np.ndarray                    # (n_bpm, n_trim): BPM strictly downstream of the trim
    unmatched_devices: dict             # {"bpm": [...], "trim": [...]}
    unmatched_elements: dict            # {"bpm": [labels], "trim": [labels]} of lattice devices without data
    dead_devices: list
    notes: list

    @property
    def n_bpm(self) -> int:
        return int(self.bpm_idx.size)

    @property
    def n_trim(self) -> int:
        return int(self.trim_idx.size)

    def summary(self) -> str:
        return (f"{self.n_bpm} BPMs and {self.n_trim} trims matched; unmatched devices: "
                f"{len(self.unmatched_devices['bpm'])} BPM, {len(self.unmatched_devices['trim'])} trim; "
                f"lattice devices without data: {len(self.unmatched_elements['bpm'])} BPM, "
                f"{len(self.unmatched_elements['trim'])} trim" + (f"; {len(self.notes)} note(s)" if self.notes else ""))


def _s_entrance_m(lattice) -> np.ndarray:
    lengths = [float(getattr(e, "length", 0.0) or 0.0) for e in lattice.elements]
    return np.concatenate([[0.0], np.cumsum(lengths)]) / 1000.0


def _lookup(lattice, key: str, cls, s_entr, notes: list, kind: str, device: str):
    """Element index for ``key`` (label first, then name); None when unmatched or ambiguous."""
    by_label = [i for i, e in enumerate(lattice.elements) if getattr(e, "label", None) == key and isinstance(e, cls)]
    if not by_label:
        by_name = [i for i, e in enumerate(lattice.elements) if getattr(e, "name", None) == key and isinstance(e, cls)]
        cand = by_name
    else:
        cand = by_label
    if not cand:
        wrong = [i for i, e in enumerate(lattice.elements)
                 if key in (getattr(e, "label", None), getattr(e, "name", None))]
        if wrong:
            notes.append(f"{device}: '{key}' is a {type(lattice.elements[wrong[0]]).__name__}, not a {kind}")
        return None
    if len(cand) > 1:
        s = [s_entr[i] for i in cand]
        if max(s) - min(s) < 1e-9:
            notes.append(f"{device}: label '{key}' is carried by {len(cand)} elements at the same s "
                         f"({', '.join(lattice.elements[i].name for i in cand)}) — using {lattice.elements[cand[0]].name}")
            return cand[0]
        notes.append(f"{device}: label '{key}' is ambiguous ({', '.join(lattice.elements[i].name for i in cand)} at different s) "
                     "— map the device to an element name explicitly")
        return None
    return cand[0]


def _is_bpm(e) -> bool:
    return isinstance(e, Marker) and bool(getattr(e, "is_bpm", False))


class _BpmMarker:            # isinstance helper: BPM markers only
    @classmethod
    def __instancecheck__(cls, obj):
        return _is_bpm(obj)


def resolve_devices(lattice, measured: dict, device_map: DeviceMap) -> OrmSelection:
    """Resolve a :class:`DeviceMap` against ``lattice`` for the loaded ``measured`` blocks."""
    s_entr = _s_entrance_m(lattice)
    notes: list = []
    dead: list = []
    for m in measured.values():
        dead += list(m.meta.get("dead_devices", []) or [])
    bpm_rows: dict = {}        # element index -> {read plane: device}
    trim_cols: dict = {}       # element index -> {kick plane: device}
    unmatched = {"bpm": [], "trim": []}
    excl_b = set(device_map.exclude_bpms); excl_t = set(device_map.exclude_trims)
    for kick, m in measured.items():
        for dev in m.trims:
            key = device_map.trims.get(dev)
            i = _lookup(lattice, key, Steerer, s_entr, notes, "steerer", dev) if key else None
            if i is None:
                unmatched["trim"].append(dev); continue
            trim_cols.setdefault(i, {})
            if kick in trim_cols[i]:
                notes.append(f"{dev}: trim {lattice.elements[i].name} already driven by {trim_cols[i][kick]} in plane {kick} — ignoring {dev}")
                unmatched["trim"].append(dev); continue
            trim_cols[i][kick] = dev
        for rp in m.read_planes():
            for dev in m.bpms[rp]:
                key = device_map.bpms.get(dev)
                i = _lookup(lattice, key, Marker, s_entr, notes, "BPM marker", dev) if key else None
                if i is not None and not _is_bpm(lattice.elements[i]):
                    notes.append(f"{dev}: '{key}' is a plain marker, not a BPM (DIAG_POSITION/BPM card)"); i = None
                if i is None:
                    if dev not in unmatched["bpm"]:
                        unmatched["bpm"].append(dev)
                    continue
                bpm_rows.setdefault(i, {})
                if rp in bpm_rows[i] and bpm_rows[i][rp] != dev:
                    notes.append(f"{dev}: BPM {lattice.elements[i].name} already read by {bpm_rows[i][rp]} in plane {rp} — ignoring {dev}")
                    unmatched["bpm"].append(dev); continue
                bpm_rows[i][rp] = dev
    bpm_idx = np.array(sorted(bpm_rows), dtype=int); trim_idx = np.array(sorted(trim_cols), dtype=int)
    planes = ("x", "y")
    bpm_devices = {p: [bpm_rows[i].get(p) for i in bpm_idx] for p in planes}
    trim_devices = {p: [trim_cols[i].get(p) for i in trim_idx] for p in planes}
    bpm_sign = {p: np.array([int(device_map.bpm_sign.get(d, 1)) if d else 1 for d in bpm_devices[p]], dtype=float) for p in planes}
    lab_b = [element_label(lattice.elements[i]) for i in bpm_idx]; lab_t = [element_label(lattice.elements[i]) for i in trim_idx]
    bpm_include = {p: np.array([d is not None and d not in excl_b and lab not in excl_b and d not in dead
                                for d, lab in zip(bpm_devices[p], lab_b)], dtype=bool) for p in planes}
    trim_include = {p: np.array([d is not None and d not in excl_t and lab not in excl_t for d, lab in zip(trim_devices[p], lab_t)], dtype=bool) for p in planes}
    all_bpm = [i for i, e in enumerate(lattice.elements) if _is_bpm(e)]
    all_trim = [i for i, e in enumerate(lattice.elements) if isinstance(e, Steerer)]
    def _tag(i):          # label plus name when they differ (fnalscl: the second ``D01T`` steerer)
        e = lattice.elements[i]; lab = element_label(e)
        return lab if lab == e.name else f"{lab} ({e.name})"
    unmatched_el = {"bpm": [_tag(i) for i in all_bpm if i not in bpm_rows],
                    "trim": [_tag(i) for i in all_trim if i not in trim_cols]}
    down = np.array([[bi > ti for ti in trim_idx] for bi in bpm_idx], dtype=bool).reshape(len(bpm_idx), len(trim_idx))
    return OrmSelection(bpm_idx=bpm_idx, bpm_labels=lab_b, bpm_devices=bpm_devices, bpm_sign=bpm_sign, bpm_include=bpm_include,
                        trim_idx=trim_idx, trim_labels=lab_t, trim_devices=trim_devices, trim_include=trim_include,
                        s_bpm_m=s_entr[bpm_idx] if bpm_idx.size else np.zeros(0), s_trim_m=s_entr[trim_idx] if trim_idx.size else np.zeros(0),
                        down=down, unmatched_devices=unmatched, unmatched_elements=unmatched_el, dead_devices=dead, notes=notes)


def align_measured(measured: dict, sel: OrmSelection, kick_plane: str, read_plane: str):
    """``(values, errors)`` of one measured block re-indexed onto ``sel`` (NaN where no device / excluded)."""
    nb, nt = sel.n_bpm, sel.n_trim
    A = np.full((nb, nt), np.nan); E = np.full((nb, nt), np.nan)
    m = measured.get(kick_plane)
    if m is None or read_plane not in m.values:
        return A, E
    col = {d: j for j, d in enumerate(m.trims)}; row = {d: i for i, d in enumerate(m.bpms[read_plane])}
    for j, dev in enumerate(sel.trim_devices[kick_plane]):
        if dev is None or dev not in col or not sel.trim_include[kick_plane][j]:
            continue
        for i, bdev in enumerate(sel.bpm_devices[read_plane]):
            if bdev is None or bdev not in row or not sel.bpm_include[read_plane][i]:
                continue
            A[i, j] = sel.bpm_sign[read_plane][i] * m.values[read_plane][row[bdev], col[dev]]
            E[i, j] = m.errors[read_plane][row[bdev], col[dev]]
    return A, E


def upstream_rows(measured: dict, sel: OrmSelection, kick_plane: str, read_plane: str) -> np.ndarray:
    """Measured values at devices the lattice does not carry (tank BPMs …): the measured noise floor."""
    m = measured.get(kick_plane)
    if m is None or read_plane not in m.values:
        return np.zeros((0, 0))
    known = {d for d in sel.bpm_devices[read_plane] if d}
    rows = [i for i, d in enumerate(m.bpms[read_plane]) if d not in known]
    return m.values[read_plane][rows] if rows else np.zeros((0, len(m.trims)))
