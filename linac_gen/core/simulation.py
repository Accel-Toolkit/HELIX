"""Simulation facade: wires lattice + beam + config into a single run API."""
from linac_gen.core.lattice import Lattice
from linac_gen.core.beam import Beam
from linac_gen.core.config import SpaceChargeConfig

class Simulation:
    """Top-level simulation controller."""

    def __init__(self, lattice: Lattice, beam: Beam,
                 space_charge: SpaceChargeConfig = None,
                 snapshot_locations: list = None,
                 snapshot_every_n: int = None,
                 snapshot_elements=None,
                 record_substeps: bool = False,
                 density_axes: tuple = (),
                 density_n_bins: int = 200,
                 density_extent: dict | None = None,
                 tail_fractions: tuple = (),
                 progress_callback=None, should_abort=None,
                 phase_probe: bool = False):
        self.lattice = lattice
        self.beam = beam
        # ``space_charge="off"`` is the explicit no-SC sentinel: same
        # physics as None but declares intent (suppresses the tracker's
        # NoSpaceChargeWarning for current-carrying bunched beams).
        if isinstance(space_charge, str) and space_charge != "off":
            raise ValueError(
                f"space_charge={space_charge!r}: the only string "
                f"sentinel is \"off\" (lowercase); pass a "
                f"SpaceChargeConfig to enable space charge")
        self._sc_explicitly_off = (space_charge == "off")
        self.sc_config = None if self._sc_explicitly_off else space_charge
        self.snapshot_locations = snapshot_locations
        self.snapshot_every_n = snapshot_every_n
        self.snapshot_elements = snapshot_elements
        self.record_substeps = record_substeps
        # Envelope-mode phase probe: record the solver's exact per-
        # element tangent maps (channel-tune extraction).  MP tracking
        # has no probe (the PIC field carries no tangent map).
        self.phase_probe = bool(phase_probe)
        # 2-D density-vs-s recording is opt-in.  ``density_axes`` enumerates
        # which beam coordinates to histogram per step (e.g. ``("x", "y")``).
        # Empty → no recording (default, zero overhead).
        self.density_axes = tuple(density_axes or ())
        self.density_n_bins = int(density_n_bins)
        self.density_extent = dict(density_extent or {})
        # Tail-quantile recording (fractional emittances / radial
        # quantiles per step) is opt-in.  Empty -> zero overhead.
        self.tail_fractions = tuple(tail_fractions or ())
        self.progress_callback = progress_callback
        self.should_abort = should_abort
        self._results = None
        # Design entrance reference, captured before any tracking mutates
        # beam.ref — run_backtrack() defaults to it for the forward
        # replay of the machine's fields and phases.
        self._entrance_ref = beam.ref.copy()
        # The PIC solver of the last forward run() — run_backtrack()
        # reuses it so a fixed-grid solver undoes its own kicks exactly.
        self._pic_solver = None

    def run(self):
        from linac_gen.tracking.tracker import Tracker
        pic = None
        csr = None
        if self.sc_config:
            if getattr(self.sc_config, "sc_backend", "numpy") == "torch":
                from linac_gen.pic.torch.solver import TorchPicSolver
                pic = TorchPicSolver(self.sc_config)
            elif getattr(self.sc_config, "sc_backend", "numpy") == "halo":
                from linac_gen.pic.ml.solver import HaloPicSolver
                pic = HaloPicSolver(self.sc_config)
                cdir = (getattr(self.sc_config, "halo", None) or {}).get(
                    "corrector_dir")
                if cdir:
                    pic.load(cdir)
            else:
                from linac_gen.pic.pic_solver import PicSolver
                pic = PicSolver(self.sc_config)
            # CSR rides the same config object and the same current>0
            # gate as space charge.  Build a kicker only when the user
            # opted in via SpaceChargeConfig.csr_enabled.
            if getattr(self.sc_config, "csr_enabled", False):
                from linac_gen.tracking.csr import CsrKicker
                csr = CsrKicker(
                    n_bins=int(getattr(self.sc_config, "csr_bins", 200)),
                    model=str(getattr(self.sc_config, "csr_model",
                                      "1d_steady")),
                )
        tracker = Tracker(
            self.lattice, self.beam,
            pic_solver=("off" if self._sc_explicitly_off else pic),
            snapshot_locations=self.snapshot_locations,
            snapshot_every_n=self.snapshot_every_n,
            snapshot_elements=self.snapshot_elements,
            record_substeps=self.record_substeps,
            progress_callback=self.progress_callback,
            should_abort=self.should_abort,
            csr_kicker=csr,
        )
        if self.density_axes:
            tracker.recorder.configure_density(
                axes=self.density_axes,
                extent=self.density_extent,
                n_bins=self.density_n_bins,
            )
        if self.tail_fractions:
            tracker.recorder.configure_tail(self.tail_fractions)
        self._pic_solver = pic
        self._results = tracker.run()
        return self._results

    def run_backtrack(self, *, entrance_ref=None, start: int = 0,
                      end: int | None = None,
                      allow_dc_crossing: bool = False,
                      approximate_backtracking: bool = False,
                      energy_tolerance: float = 1e-3,
                      energy_hard_limit: float = 5e-2,
                      field_map_mode: str = "rk4"):
        """Backtrack ``self.beam`` (an exit-plane distribution) to the
        entrance of ``lattice.elements[start]``.

        Two workflows:

        * **Undo a forward run** — call :meth:`run` first, then this;
          the beam left at the exit walks back to the entrance, reusing
          the forward run's PIC solver (exact SC undo, frozen grid and
          all).
        * **Reconstruct from a measured .dst** — build the Simulation
          with the loaded exit distribution as ``beam`` and pass
          ``entrance_ref`` (the design entrance state used to evaluate
          the machine's fields; defaults to the ref the beam carried at
          construction, which is only correct if that WAS the entrance
          state).

        Returns the reversed-to-increasing-s ``DiagnosticRecorder``
        (index 0 = reconstructed entrance), tagged
        ``direction="backward"``; also stored in :meth:`get_results`.
        """
        from linac_gen.tracking.backtrack import backtrack_distribution
        if entrance_ref is None:
            entrance_ref = self._entrance_ref
        self._results = backtrack_distribution(
            self.lattice, self.beam, entrance_ref,
            start=start, end=end,
            # Forward the explicit-off intent (same mapping as run()):
            # without it a DC space_charge="off" run would un-kick
            # forward but re-kick on the backward walk.
            sc_config=self.sc_config,
            pic_solver=("off" if self._sc_explicitly_off
                        else self._pic_solver),
            record_substeps=self.record_substeps,
            progress_callback=self.progress_callback,
            should_abort=self.should_abort,
            allow_dc_crossing=allow_dc_crossing,
            approximate_backtracking=approximate_backtracking,
            energy_tolerance=energy_tolerance,
            energy_hard_limit=energy_hard_limit,
            field_map_mode=field_map_mode)
        return self._results

    def run_envelope(self):
        from linac_gen.tracking.envelope import EnvelopeSolver, EnvelopeResults
        # Support an explicit dict of Twiss + emittance parameters attached to the
        # simulation, or fall back to sensible defaults.
        initial = getattr(self, "beam_envelope_params", None)
        if initial is None:
            # Placeholder fallback (β=1 m, ε=1): Simulation holds no
            # BeamConfig, so nothing better can be derived here.  Every
            # in-repo caller sets ``beam_envelope_params`` first — warn
            # loudly so an external API caller can't mistake the
            # placeholder envelope for their beam.
            import warnings
            warnings.warn(
                "Simulation.run_envelope(): no beam_envelope_params set "
                "— using a PLACEHOLDER beam (β=1 m, ε=1 mm·mrad in all "
                "planes), NOT your tracked beam's Twiss.  Set "
                "sim.beam_envelope_params (or use EnvelopeSolver "
                "directly) for real envelope results.",
                stacklevel=2)
            initial = {
                "alpha_x": 0.0, "beta_x": 1.0,
                "alpha_y": 0.0, "beta_y": 1.0,
                "alpha_z": 0.0, "beta_z": 1.0,
                "emit_x": 1.0, "emit_y": 1.0, "emit_z": 1.0,
            }
        solver = EnvelopeSolver(
            self.lattice,
            self.beam.ref,
            initial,
            current=self.beam.current,
            progress_callback=self.progress_callback,
            should_abort=self.should_abort,
            record_substeps=self.record_substeps,
            phase_probe=self.phase_probe,
        )
        self._results = solver.run()
        return self._results

    def get_results(self):
        if self._results is None:
            raise RuntimeError("No simulation has been run yet")
        return self._results
