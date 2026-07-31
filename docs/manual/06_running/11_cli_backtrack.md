# CLI: `backtrack`

Backward tracking from the shell — walk a distribution from the exit
of an element range to its entrance.  Physics background and
invertibility limits:
[Concepts → Backward tracking](../02_concepts/03_tracking_modes.md#backward-tracking).

```
python -m linac_gen backtrack <input> [options]
```

`<input>` is a `.dat` / `.madx` lattice or a `.lgproj` project.  Two
source modes:

* **`--dst EXIT.dst`** — *reconstruction*: the measured / exported exit
  distribution is loaded and walked upstream.  The file is
  authoritative for its reference energy and frequency (same
  convention as forward `.dst` input).
* **no `--dst`** — *design targeting*: the project's Twiss / emittance
  describe the **desired exit** beam; it is generated at the design
  exit energy (from the forward replay of the entrance reference) and
  backtracked to find the input that produces it.

## Options

| Option | Argument | Default | Meaning |
|---|---|---|---|
| `--dst` | path | — | exit-plane `.dst` to reconstruct from (omit for design mode) |
| `--mode` | `mp` \| `envelope` | `mp` | multi-particle walk or RMS σ-matrix walk |
| `--from-element` | int | `0` | range start — the entrance the walk targets |
| `--to-element` | int | last | range end — the exit the walk starts from |
| `--write-dst` | path | — | write the reconstructed entrance distribution (mp) |
| `--fieldmap-inverse` | `rk4` \| `linear` | `rk4` | field-map inverse: `rk4` undoes each forward integration step in closed form (round trips close at float round-off, ~1e-13); `linear` is the v1 inverted fitted-matrix path (~2 % closure through strong bunchers), kept for comparison |
| `--validate` | flag | off | forward re-track the reconstruction and report exit residuals; exits non-zero if the closure is worse than 5 % (mp) |
| `--allow-dc-crossing` | flag | off | assert the beam was DC (continuous) upstream of the first RF bunching element |
| `--energy-tolerance` | rel. | `1e-3` | warn when the exit beam's energy deviates from the design exit by more |
| `--energy-hard-limit` | rel. | `5e-2` | refuse above this deviation |
| `--sc-on` | flag | off | enable space charge on the backward walk |
| `--out`, `--format` | | `.`, `hdf5` | results directory / format (as `run`) |
| `--energy`, `--freq`, `--species`, `--current`, `--n-particles`, `--beam`, `--nx`, `--grid-extent`, `--step1`, `--step2`, `--kernel`, `--backend`, `--sc`, `--seed`, `-q` | | | same contract as [`run`](07_cli_run.md) |

The results file (`<stem>_backtrack.h5`) is an ordinary HELIX results
file with `s` **increasing**: index 0 is the reconstructed entrance,
the last row is the supplied exit distribution.  It opens in the GUI
Results tab and all exporters unchanged.

## Example: reconstruct the PIP-II MEBT input from an exit `.dst`

Produce an exit distribution with a forward run, then walk it back and
check the closure (run from the repository root):

```bash
# 1. forward run -> writes mebt_final.dst at the MEBT exit
python -m linac_gen run examples/pipii/mebt/mebt.dat --mode mp \
    --species H- --energy 2.1 --freq 162.5 --n-particles 1000 \
    --beam emit_nx=0.21 --beam alpha_x=1.228  --beam beta_x=0.316 \
    --beam emit_ny=0.21 --beam alpha_y=-0.095 --beam beta_y=0.113 \
    --beam emit_z=0.0623 --beam beta_z=819.05 \
    --write-dst --out /tmp/bt -q

# 2. backtrack the exit .dst to the entrance + forward closure check
python -m linac_gen backtrack examples/pipii/mebt/mebt.dat \
    --dst /tmp/bt/mebt_final.dst \
    --species H- --energy 2.1 --freq 162.5 \
    --beam emit_nx=0.21 --beam alpha_x=1.228  --beam beta_x=0.316 \
    --beam emit_ny=0.21 --beam alpha_y=-0.095 --beam beta_y=0.113 \
    --beam emit_z=0.0623 --beam beta_z=819.05 \
    --write-dst /tmp/bt/mebt_entrance.dst --validate --out /tmp/bt
```

Typical output of step 2 (426 elements, 4 bunchers):

```
[backtrack] mode=mp  input=examples/pipii/mebt/mebt.dat  source=/tmp/bt/mebt_final.dst
  range          = [0, 425] (exit → entrance)
  entrance s     = 0.000 mm
  entrance W_kin = 2.100000 MeV
  reconstructed sigma: x=1.0064 mm  y=0.5941 mm  dphi=7.4258 deg  dW=0.008266 MeV
[validate] forward closure over [0, 425] (999 particles):
  sigma_x/mm     recovered=1.61566   original=1.61718   rel=9.411e-04
  sigma_y/mm     recovered=1.57804   original=1.58115   rel=1.972e-03
  sigma_phi/deg  recovered=6.18469   original=6.06787   rel=1.925e-02
  sigma_w/MeV    recovered=0.00994   original=0.01011   rel=1.664e-02
[validate] OK (worst relative sigma residual 1.925e-02)
```

The sample output above was measured with `--fieldmap-inverse linear`
(the v1 path): the ~2 % longitudinal residual is the linear-inverse
fidelity of the four bunchers.  With the default **`rk4` exact
inverse** the same `.dst` round trip closes at float round-off —
measured worst relative σ residual **7×10⁻¹⁵** on the MEBT — because
every forward integration step is undone in closed form.  The
`--validate` gate (5 %) is then only ever tripped by genuine input
problems (heavy losses, wrong energy, unmatched design-mode Twiss).

!!! note "That float-round-off figure is the space-charge-free closure"
    The 7×10⁻¹⁵ above is the optics + RF closure — the field-map/RF
    inverse itself is exact.  Under **strong space charge** the
    reconstruction recovers the beam *envelope* (σ/ε/Twiss close to
    ~0.2–2.4 % over the full MEBT+HWR at 5 mA) but **not** individual
    particle trajectories (~0.14 mm median per-particle error at that
    scale): the PIC field solve is not bit-reversible and the
    collective, effectively-chaotic SC dynamics amplify the seed error
    through the ~10³ kicks of a full section.  `--validate` reports the
    **σ** closure, which is the quantity that stays accurate — so it
    remains a good gate.  Backtrack the *beam*, not per-particle
    histories, when SC is significant.

!!! warning "Use the real machine Twiss"
    Design mode and the `--validate` interpretation both assume the
    beam is reasonably matched.  An unmatched beam debunches, drives
    the bunchers far outside their linear range, and the linear
    backward inverse (like any linearisation) becomes meaningless —
    σ_φ in the thousands of degrees is the telltale.

## Example: design targeting (what input gives this exit beam?)

The project Twiss is read as the **desired exit** state:

```bash
python -m linac_gen backtrack examples/pipii/mebt/mebt.dat \
    --mode envelope --species H- --energy 2.1 --freq 162.5 \
    --beam emit_nx=0.21 --beam alpha_x=1.228  --beam beta_x=0.316 \
    --beam emit_ny=0.21 --beam alpha_y=-0.095 --beam beta_y=0.113 \
    --beam emit_z=0.0623 --beam beta_z=819.05 --out /tmp/bt
```

```
[backtrack] mode=envelope  input=examples/pipii/mebt/mebt.dat  range=[0, 425]
  entrance Twiss: alpha_x=-0.9483  beta_x=0.1240  alpha_y=+2.9056  beta_y=0.2222
  entrance sigma: x=0.6237 mm  y=0.8349 mm
```

`--mode mp` on the same design-mode inputs generates the exit
distribution from the Twiss and walks the particles back instead —
use it when you want the full reconstructed distribution
(`--write-dst`) rather than the RMS answer.

## Guards you may encounter

| Message | Meaning |
|---|---|
| `aperture losses are non-invertible …` | reconstruction is valid for surviving particles only |
| `backtrack skips N lattice command(s) …` | SET_* / ADJUST_* effects are honoured via the replay table, not re-executed |
| `… uses grid_mode='fixed' … ADAPTIVE grids instead` | SC undo is approximate without the forward run's solver — see the tip in [Python API](01_python_api.md#backward-tracking) |
| `… crosses the DC↔bunched transition …` | pass `--allow-dc-crossing` if the machine really has a DC front end |
| `… deviates … from the design exit energy` | the `.dst` and the lattice/energy pairing disagree — check `--energy` |
| `… tracked with periodic phase coordinates … non-invertible` | **hard error, not a warning.** The run used [`periodic_phase`](../04_beam/03_beam_config.md), which discards which bunch of the train each particle landed in, so the undo would be wrong by whole bunch spacings.  Re-run the forward pass with `periodic_phase=False` to backtrack it. |

## See also

* [Python API → Backward tracking](01_python_api.md#backward-tracking)
* [Concepts → Backward tracking](../02_concepts/03_tracking_modes.md#backward-tracking)
* [.dst I/O](../04_beam/04_dst_io.md) — the exchange format both ends use

← [CLI: twiss](10_cli_twiss.md) ·
[Continue to Matching →](../07_matching/01_overview.md)
