# Results files

What a ring run writes, and how to read it back.

| File | Written by | Contents |
|---|---|---|
| `<stem>_ring.h5` (command line, in `--out`), `<time>_ring.h5` (GUI, in the calc directory), `<lattice>_<time>_ring.h5` (the assistant's `run_ring`, in the calc directory) | every run — a stopped one too, unless it stopped before its first turn | the results, below |
| `<stem>_ring_checkpoint.npz` | `ring run --checkpoint-every K` (in `--out`); in Python, `RingConfig(checkpoint_every=K)` with `run(checkpoint_path=...)`, to that path | everything a run needs to continue bit for bit ([Checkpoints](04_run.md#checkpoints)) |
| a CSV of the reference turn by turn | `ring ramp --csv` | time, energy, gain, voltage, phase, frequency |
| a program table (`.csv`) | the Programs page's **Export CSV…** (read back with **Import CSV…**) | an energy, voltage or phase program, the times in s |
| a Twiss table (`.csv` or `.json`) | `ring optics --twiss-out` | the periodic optics per element |

## The HDF5 layout

`run_type = "ring"`, schema 3 (files of schema 1 and 2, written before the
ramp and before injection, are read too).  Ring coordinates, SI units: x, y
[m], px, py = P/P0, ζ = s − β0ct [m], δ = (P − P0)/P0.

A group's numbers and words are attributes, its lists of numbers datasets;
anything else — a dictionary, a list of words — is JSON text in an attribute
whose name ends in `__json` (the space-charge record, for example, is
`meta.attrs["space_charge__json"]`); the optics table's element names and
types are string datasets.  `load_ring_hdf5` reads them all back (a file
written before 2026-10-25 has no `/optics_table`: it reads as an empty
table).

| Path | Datasets | Attributes |
|---|---|---|
| `/` | | `run_type`, `schema_version`, `engine`, `n_turns_requested`, `truncated`, `wall_s` |
| `/turns/` | per statistics row: `turn`, `n_alive`, `mean` (rows × 6), `cov` (rows × 6 × 6), the row's reference `beta0gamma0` and `p0c_mev` | |
| `/ramp/` (a ramp) | per turn, at its start: `t_s`, `kinetic_mev`, `gain_mev`, `voltage_mv`, `phi_deg`, `frequency_hz`, `above_transition`, `capped` | `transition_turn` (when crossed), `gamma_t`, `n_capped`, `turns_to_end`, `kinetic_end_mev`, `gamma_end`, `t_end_s`, `above_transition_at_start`, `phase_offset_deg`, `voltage_eff_factor`, `program__json` |
| `/snapshots/turn_<t>/` | `coords` (6 × m, ring coordinates), `pid` (m) | in a ramp, the reference and RF of the turn after the snapshot: `kinetic_mev`, `energy_mev`, `beta0`, `t_s`, `voltage_mv`, `voltage_eff_mv`, `phi_deg`, `phi_eff_deg`, `frequency_hz`, `harmonic`, `eta`, `above_transition`, `length_m` |
| `/injection/` (a foil) | per turn: `injected`, `stripped`, `h0`, `hminus`, `missed`, `injected_charge`, `foil_hits`, `foil_hits_injected`, `deposit_mev`, `electron_mev` (real particles and energies: the ring's); `closed_orbit_foil` (turns + 1 × 4), `waveforms` (turns + 1 × groups) and `waveform_names` (with magnets), `hit_map` and `deposit_map_mev` (nx × ny), `map_extent_m`, `hits_histogram` | `buckets`, `mean_hits_circulating`, `foil__json`, `settings__json`, `source__json`, `waste__json`, `notes__json` |
| `/losses/` | `pid`, `turn` (the turns completed before the loss), `node`, `element`, `s`, `cause`, `coords` (6 × n) | |
| `/probes/` | `tbt` (probes × (min(turns, 4096) + 1) × 6), `amplitudes_sigma`, `tunes`, `diffusion`, `state` | |
| `/tunes` | | the NAFF tunes of the smallest probe: `qx`, `qy`, `qs` (none after fewer than 15 turns) |
| `/optics` | `closed_orbit_start`, `eigen_tunes_6d` | `qx`, `qy`, `dqx`, `dqy`, `alpha_c`, `eta`, `gamma_t`, `qs`, `gamma0`, `beta0`, `length_m`, `mode`, the start's Twiss and dispersion (`betx_start`, `alfx_start`, `dx_start`, `dpx_start`, … ), `bucket__json` (6-D optics) |
| `/optics_table/` | the run's periodic optics at every element's exit (at the start of a ramp): `name`, `type` (strings), `s` [m], `betx`, `alfx`, `bety`, `alfy`, `mux`, `muy` (the phase advance from the start, in units of 2π), `dx`, `dpx`, `dy`, `dpy` (per δ), `x`, `px`, `y`, `py` (the closed orbit) | |
| `/ledger` | | `injected`, `circulating`, `lost`, `waste` and their real particles (`…_charge`) |
| `/meta` | `element_names`, `element_s`, `element_length`, `closed_orbit`, `eps0_normal_modes`, `start_mismatch` (the starting beam's BMAG per mode: x, y and, with the RF, z; absent for an empty ring or a continued run; `start_mismatch__json` with a null for a mode of no emittance) | the reference (`kinetic_mev`, `p0c_mev`, `beta0`, `gamma0`, `beta0gamma0`, `mass_mev`, `species`), `length_m`, `harmonic`, `revolution_time_s`, `rf_on`, `zeta_wrap_m`, `device`, `threads`, `stats_every`, `n_particles`, `config__json` (the settings), `options__json`, `notes__json`, `warnings__json`; with space charge `space_charge__json`: its settings, kicks, analytic tune shift, the line-density bins used (`z_bins_used`, `z_bins_changes`) and `line_density_resolution` |
| `/beam_config/` | | the beam settings the particles came from, when given |
| `/provenance/` | | the code version and commit, the lattice file and its hash, the seed, the threads |

## Reading it

In Python, back into the results object the run made:

    from linac_gen.ring.results import load_ring_hdf5
    res = load_ring_hdf5("runs/simple_booster_ring.h5")
    res.turns, res.n_alive          # the statistics rows
    res.cov[-1]                     # the last row's covariance (6 x 6)
    res.emit_normalized             # normalized rms emittances per row (m rad)
    res.snapshots[1000]["coords"]   # a snapshot's particles (6, m)
    res.losses["turn"], res.ramp, res.injection, res.meta["notes"]
    res.optics_table["s"], res.optics_table["betx"]   # the periodic optics

Or with h5py alone:

    import json
    import h5py
    with h5py.File("runs/simple_booster_ring.h5") as f:
        turn = f["turns/turn"][()]
        cov = f["turns/cov"][()]           # (rows, 6, 6)
        x = f["snapshots/turn_1000/coords"][0]
        qx = f["optics"].attrs["qx"]
        sc = json.loads(f["meta"].attrs["space_charge__json"])   # (with space charge)
