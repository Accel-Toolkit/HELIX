"""Configuration for the OPT-IN multibunch / pulse study.

This study is rarely run and strictly opt-in: nothing here is touched by
normal single-bunch workflows, and with every physics flag at its default
(off) an N-bunch train is defined to be bit-identical to N independent
single-bunch runs (the zero-coupling contract, enforced by
tests/train/test_zero_coupling.py).

Required parameters are validated loudly at construction: switching a
physics channel on without its inputs is an error listing exactly what is
missing — physics inputs never get silent defaults.

The same honesty cuts the other way (M8): knobs that no v1 runner
consumes (``charge_scale``, ``jitter``, and explicit ``select_bunches``
outside hybrid mode) are REFUSED at construction rather than silently
carried — a study must never accept an input and then ignore it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


class PulsePattern:
    """A pulse's bunch-slot fill pattern at the bunch frequency.

    Canonical storage is a boolean array over the slot axis (True =
    bunch present, False = chopped/empty slot) plus a run-length-encoded
    string form for provenance.  PIP-II scale (0.55 ms @ 162.5 MHz ~
    89,375 slots) is a trivially small array.
    """

    def __init__(self, filled: np.ndarray):
        arr = np.asarray(filled, dtype=bool)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("pattern must be a non-empty 1-D bool array")
        if not arr.any():
            raise ValueError("pattern has no filled slots — nothing to run")
        self.filled = arr

    # ---- constructors -------------------------------------------------
    @classmethod
    def from_array(cls, filled) -> "PulsePattern":
        return cls(np.asarray(filled, dtype=bool))

    @classmethod
    def from_rle(cls, rle: str) -> "PulsePattern":
        """Parse '1*10 0*54 1*26 ...' (value*count tokens, whitespace-sep)."""
        chunks = []
        for tok in rle.split():
            try:
                val, cnt = tok.split("*")
                val = val.strip()
                if val not in ("0", "1"):
                    raise ValueError(f"RLE value must be 0 or 1, got {val!r}")
                chunks.append(np.full(int(cnt), val == "1", bool))
            except ValueError:
                raise
            except Exception as exc:                          # noqa: BLE001
                raise ValueError(f"bad RLE token {tok!r} in pattern") from exc
        if not chunks:
            raise ValueError("empty RLE pattern string")
        return cls(np.concatenate(chunks))

    @classmethod
    def uniform(cls, n_slots: int) -> "PulsePattern":
        return cls(np.ones(int(n_slots), bool))

    @classmethod
    def from_duty(cls, n_slots: int, keep: int, period: int) -> "PulsePattern":
        """Periodic chopping: keep the first ``keep`` of every ``period``."""
        if not 0 < keep <= period:
            raise ValueError("need 0 < keep <= period")
        idx = np.arange(int(n_slots))
        return cls((idx % period) < keep)

    # ---- views --------------------------------------------------------
    @property
    def n_slots(self) -> int:
        return int(self.filled.size)

    @property
    def n_bunches(self) -> int:
        return int(self.filled.sum())

    @property
    def filled_slots(self) -> np.ndarray:
        return np.flatnonzero(self.filled)

    def to_rle(self) -> str:
        vals, starts = self.filled, np.flatnonzero(
            np.r_[True, self.filled[1:] != self.filled[:-1]])
        lengths = np.diff(np.r_[starts, vals.size])
        return " ".join(f"{int(vals[s])}*{int(l)}"
                        for s, l in zip(starts, lengths))

    def pulse_length_us(self, bunch_frequency_MHz: float) -> float:
        return self.n_slots / bunch_frequency_MHz


@dataclass
class TrainPhysics:
    """Bunch-coupling channels.  ALL OFF by default (opt-in study)."""
    direct_sc: bool = False       # M5
    beam_loading: bool = False    # M2
    hom: bool = False             # M4


@dataclass
class TrainJitter:
    """Per-bunch jitter (optional; seeded)."""
    phase_deg_rms: float = 0.0
    amplitude_rel_rms: float = 0.0
    charge_rel_rms: float = 0.0
    seed: int = 0


@dataclass
class TrainConfig:
    """Top-level switch + parameters for the multibunch study."""
    bunch_frequency_MHz: float
    pattern: PulsePattern
    mode: str = "mp"                       # "mp" | "envelope" | "fast" | "hybrid"
    physics: TrainPhysics = field(default_factory=TrainPhysics)
    cavity_params: Optional[str] = None    # sidecar file (R/Q, Q_L, HOMs)
    charge_scale: Optional[Callable[[int], float]] = None
    jitter: Optional[TrainJitter] = None
    select_bunches: object = "auto"        # hybrid replay selection
    seed: int = 42
    keep_full_results: bool = True         # False: summary-only (big trains)
    # ---- physics.direct_sc knobs (M5; consulted only when it is on) ----
    # Neighbour model for the ±1 bunch-train images:
    #   "images"   — exact copies of the live bunch, scaled by the
    #                pattern factors (M5a; the Toutatis-validated
    #                machinery made chopped-gap-aware);
    #   "distinct" — the LEADING image (the previously tracked bunch)
    #                is its recorded subsampled snapshot; the trailing
    #                image stays a self-copy (not tracked yet — causal
    #                approximation).  With identical bunches this
    #                reproduces "images" within deposition noise (the
    #                anchor test), so it pays off only once bunches
    #                genuinely differ (losses, loading-shifted refs).
    direct_sc_neighbors: str = "images"
    # Force the PIC's σφ engagement gate ON for every bunch pass: an
    # explicit train study must not be silently disengaged downstream
    # (or never engaged, for a born-bunched pulse).  Costs self-field
    # resolution for short bunches (3×-span grid) — off by default; a
    # run whose images never engage warns loudly instead.
    direct_sc_force_engage: bool = False
    # Snapshot subsample size per SC kick for "distinct" (bounded ring
    # buffer; memory ≈ 2 · n_kicks_engaged · n_sub · 3 · 8 B).
    direct_sc_subsample: int = 1024

    def __post_init__(self):
        if self.bunch_frequency_MHz <= 0:
            raise ValueError("bunch_frequency_MHz must be > 0")
        if not isinstance(self.pattern, PulsePattern):
            raise TypeError("pattern must be a PulsePattern")
        if self.mode not in ("mp", "envelope", "fast", "hybrid"):
            raise ValueError(f"unknown train mode {self.mode!r}")
        if self.mode == "hybrid":
            # DOCUMENTED CHOICE (M6, tests/train/test_replay.py): hybrid
            # with no coupling physics is REFUSED, not run trivially — a
            # two-pass replay of bit-identical bunches is pure waste and
            # a physics-off "hybrid study" would be indistinguishable
            # from a mislabelled single-bunch run.  beam_loading is the
            # minimum (it is what pass 1 records per slot); hom is
            # optional on top; direct_sc composes into the pass-2 MP
            # replays ("images" neighbours only — an independent replay
            # has no previously tracked bunch to snapshot).
            if not self.physics.beam_loading:
                raise ValueError(
                    "mode='hybrid' requires physics.beam_loading=True "
                    "(the pass-1 fast recursion records per-cavity "
                    "loading state for the replays; hom is optional, "
                    "direct_sc composes).  With all coupling physics "
                    "off, run mode='mp' directly — every bunch is "
                    "identical (zero-coupling contract)")
            if self.physics.direct_sc \
                    and self.direct_sc_neighbors == "distinct":
                raise ValueError(
                    "mode='hybrid' supports only "
                    "direct_sc_neighbors='images': pass-2 replays are "
                    "independent, so there is no previously tracked "
                    "bunch to snapshot as a distinct neighbour")
            self.select_bunches = self._validate_select_bunches()
        else:
            # select_bunches is a HYBRID-replay knob; every other mode
            # runs every filled slot.  Accepting a selection and then
            # tracking everything anyway would be silent inertness.
            if not (isinstance(self.select_bunches, str)
                    and self.select_bunches == "auto"):
                raise ValueError(
                    f"select_bunches={self.select_bunches!r} is a "
                    f"hybrid-replay knob; mode={self.mode!r} tracks "
                    "every filled slot of the pattern — drop it or use "
                    "mode='hybrid'")
        # ---- M8 honesty guards ---------------------------------------
        # These fields are declared for forward compatibility but NOT
        # consumed by any v1 runner (tracked, fast and hybrid all run
        # identical nominal-charge bunches).  Refuse loudly instead of
        # running a study that would silently ignore its inputs.
        if self.charge_scale is not None:
            raise ValueError(
                "TrainConfig.charge_scale is not consumed by any v1 "
                "runner (every bunch runs at the nominal beam charge) — "
                "remove it; per-bunch charge modulation is a planned "
                "extension and will be refused until it actually acts")
        if self.jitter is not None:
            raise ValueError(
                "TrainConfig.jitter is not consumed by any v1 runner "
                "(bunches are identical by construction) — remove it; "
                "per-bunch jitter is a planned extension and will be "
                "refused until it actually acts")
        missing = []
        if self.physics.beam_loading or self.physics.hom:
            if not self.cavity_params:
                missing.append(
                    "cavity_params (sidecar file with R/Q, Q_L"
                    + (", HOM table" if self.physics.hom else "") + ")")
        if missing:
            raise ValueError(
                "multibunch physics enabled but required inputs missing: "
                + "; ".join(missing))
        if self.direct_sc_neighbors not in ("images", "distinct"):
            raise ValueError(
                f"direct_sc_neighbors={self.direct_sc_neighbors!r}: "
                "expected \"images\" or \"distinct\"")
        if int(self.direct_sc_subsample) < 8:
            raise ValueError(
                f"direct_sc_subsample={self.direct_sc_subsample} is too "
                "small to represent a neighbour bunch (need >= 8)")
        if self.physics.direct_sc and self.mode not in ("mp", "hybrid"):
            raise ValueError(
                "physics.direct_sc requires mode='mp' (or 'hybrid', "
                "where it acts in the pass-2 MP replays): direct bunch-"
                "to-bunch space charge is a particle/PIC effect — "
                f"mode={self.mode!r} tracks no macroparticles "
                "(the runner additionally requires an sc_config "
                "with the numpy 3-D PIC)")

    def _validate_select_bunches(self):
        """Normalise/validate the hybrid replay selection (M6).

        "auto" passes through (resolved against the pattern by
        ``linac_gen.train.replay.auto_select_bunches``); anything else
        must be an iterable of ABSOLUTE slot indices, each a filled slot
        of the pattern — replaying an empty (chopped) slot is a
        contradiction and refused loudly.  Returns "auto" or a sorted,
        de-duplicated list[int].
        """
        sb = self.select_bunches
        if isinstance(sb, str):
            if sb != "auto":
                raise ValueError(
                    f"select_bunches={sb!r}: expected \"auto\" or a "
                    "list of absolute slot indices")
            return sb
        try:
            slots = sorted({int(s) for s in sb})
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"select_bunches={sb!r}: expected \"auto\" or an "
                "iterable of integer slot indices") from exc
        if not slots:
            raise ValueError("select_bunches is empty — nothing to replay")
        filled = self.pattern.filled
        bad = [s for s in slots
               if not (0 <= s < filled.size) or not filled[s]]
        if bad:
            raise ValueError(
                f"select_bunches contains slot(s) {bad} that are out of "
                f"range or empty (chopped) in the pattern "
                f"({filled.size} slots) — only filled slots can be "
                "replayed")
        return slots
