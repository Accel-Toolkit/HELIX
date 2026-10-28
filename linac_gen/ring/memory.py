"""The memory a ring run will hold, estimated before it starts — so that a run
that cannot fit is refused at once (the GUI's Run button and the assistant's
``run_ring`` refuse beyond MEMORY_SHARE of the machine's memory)."""
from __future__ import annotations

import os
import sys

#: a run is refused when it would need more than this share of the memory
MEMORY_SHARE = 0.7
#: the run's memory per particle (bytes): the particles with their energy
#: variables and bookkeeping, and the arrays made while they are generated
#: (the largest rise measured: 567 for a matched beam, 414 for the Beam tab's)
BYTES_PER_PARTICLE = 600


def physical_memory() -> int | None:
    """The machine's memory in bytes (None when it cannot be read)."""
    try:
        if sys.platform == "win32":
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            ("total_phys", ctypes.c_ulonglong),
                            ("avail_phys", ctypes.c_ulonglong),
                            ("total_page", ctypes.c_ulonglong),
                            ("avail_page", ctypes.c_ulonglong),
                            ("total_virtual", ctypes.c_ulonglong),
                            ("avail_virtual", ctypes.c_ulonglong),
                            ("avail_extended", ctypes.c_ulonglong)]
            st = _Status()
            st.length = ctypes.sizeof(_Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.total_phys)
            return None
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (OSError, ValueError, AttributeError):
        return None


def run_memory(n_particles: int, n_snapshots: int, n_rows: int, n_probes: int,
               n_record: int) -> int:
    """Bytes a run holds, roughly: the particles, a copy of them per snapshot
    (56 bytes each), the statistics rows (about 600 bytes each) and the
    probes' record (48 bytes per probe and turn)."""
    return int(n_particles * (BYTES_PER_PARTICLE + 56 * n_snapshots) + 600 * n_rows
               + 48 * n_probes * n_record)


def injection_record_memory(injection: dict, n_turns: int) -> int:
    """Bytes an injection's turn-by-turn record holds, kept and copied into
    the results: its ten numbers a turn, the closed orbit at the foil (four)
    and the magnets' waveforms (one a group), 8 bytes each."""
    per_turn = 8 * (10 + 4 + len(injection.get("magnets") or ()))
    return 2 * per_turn * (int(n_turns) + 1)


def space_charge_memory(sc: dict | None, n_starts: int | None = None) -> int:
    """Bytes space charge adds, at most: the Green's functions kept (their cap,
    or one grid's when it is larger), the deposit's grids, a solve's working
    arrays, and the frozen model's fields (their cap) with each kick's line
    density — of the kicks placed, at most ``n_starts`` (the ring's elements)
    when given."""
    if not sc:
        return 0
    from linac_gen.ring.sc import greens, solver
    from linac_gen.ring.sc.beams import DEPOSIT_BYTES
    nx, ny = sc["grid"]
    one_grid = 6 * 16 * nx * ny                    # (G_phi, G_x, G_y transforms)
    work = 12 * 16 * nx * ny                       # (a solve's spectra and outputs)
    frozen = 0
    if sc.get("model") == "frozen":
        # (its fields, to their cap, and each kick's line density and slope —
        # of the kicks placed: no more than the ring's elements)
        kicks = int(sc["kicks_per_turn"])
        if n_starts:
            kicks = min(kicks, int(n_starts))
        # ("auto" bins: as many as they may become)
        nz = solver.AUTO_NZ_MAX if sc.get("z_bins") == "auto" else int(sc["z_bins"])
        frozen = solver.FROZEN_CACHE_BYTES + kicks * 2 * 8 * nz
    return int(max(greens.CACHE_BYTES, one_grid) + DEPOSIT_BYTES + work + frozen)


def run_need(config, beam_particles: int = 0, n_elements: int | None = None) -> int:
    """Bytes the run of a RingConfig ``config`` will hold, roughly: its
    particles (a matched source's; none to start with an empty ring; the beam
    settings' ``beam_particles`` for a beam source; those the injection
    brings), snapshots, statistics rows, the probes' record, a ramp's record,
    space charge (of a ring of ``n_elements``) and the injection's record."""
    from linac_gen.ring.config import snapshot_turn_set
    from linac_gen.ring.engine import PROBE_RECORD_TURNS
    n = int(config.n_turns)
    n_part = (int(config.n_particles) if config.source == "matched"
              else 0 if config.source == "empty" else int(beam_particles or 0))
    inj = config.injection
    if inj:
        n_part += int(inj["turns"]) * int(inj["particles_per_turn"])
    probes = config.tune_probes ** 2 if config.tune_probes > 1 else config.tune_probes
    return (run_memory(n_part, len(snapshot_turn_set(config.snapshot_turns, n)),
                       n // int(config.stats_every) + 2, probes,
                       min(n, PROBE_RECORD_TURNS) + 1) + (66 * n if config.ramp else 0)
            + space_charge_memory(config.space_charge, n_starts=n_elements)
            + (injection_record_memory(inj, n) if inj else 0))
