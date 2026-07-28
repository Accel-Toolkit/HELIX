"""HaloPicSolver: coarse PIC + learned delta-rho defect correction with
fine-grid anchor monitoring.

Drop-in replacement for :class:`PicSolver` (selected via
``SpaceChargeConfig.sc_backend = "halo"``).  Per kick:

  coarse deposit -> coarse IGF solve -> [+ alpha * sum_k c_k E_k] ->
  gather -> kick

Every K-th kick additionally runs the SAME particles through a
fine-grid deposit+solve on the same physical grid box (the anchor):

  * the beam receives the fine field (anchors are free accuracy),
  * the defect (fine - coarse, at coarse nodes) is projected onto the
    basis and logged as a training pair (features, coeffs),
  * the running defect norm drives the trust-region K controller and
    the alpha kill-switch.

Training happens OFFLINE (`train_offline.py`) on the logged pairs —
never inside the loop (determinism + speed; see the HALO-PIC plan).

With ``alpha = 0`` and anchors disabled the kick is bit-identical to
``PicSolver.kick`` (pinned by test).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from linac_gen.pic.pic_solver import PicSolver, _select_kernel
from linac_gen.pic.coordinates import beam_to_spatial
from linac_gen.pic.lorentz_boost import boost_to_rest_frame
from linac_gen.pic.poisson_solver import PoissonSolverFFT
from linac_gen.pic.ml.basis import BasisFieldCache, default_mode_indices
from linac_gen.pic.ml.features import FEATURE_DIM, kick_features


class KController:
    """Trust-region anchor-cadence controller.

    e < tau_lo  -> K doubles (up to K_max)
    e > tau_hi  -> K halves (down to K_min)
    e > tau_hard-> alpha = 0 (corrector disabled; anchors continue)
    """

    def __init__(self, k_init=8, k_min=2, k_max=64,
                 tau_lo=0.02, tau_hi=0.10, tau_hard=0.30):
        self.k = int(k_init)
        self.k_min, self.k_max = int(k_min), int(k_max)
        self.tau_lo, self.tau_hi, self.tau_hard = tau_lo, tau_hi, tau_hard
        self.alpha_killed = False

    def update(self, e: float, allow_kill: bool = True) -> None:
        if e > self.tau_hard and allow_kill:
            self.alpha_killed = True
        if e > self.tau_hi:
            self.k = max(self.k // 2, self.k_min)
        elif e < self.tau_lo:
            self.k = min(self.k * 2, self.k_max)


class HaloPicSolver(PicSolver):
    """Coarse PIC with learned defect correction + fine anchors."""

    def __init__(self, config):
        super().__init__(config)
        h = getattr(config, "halo", {}) or {}
        self.alpha = float(h.get("alpha", 1.0))
        self.warmup_kicks = int(h.get("warmup_kicks", 0))
        self.fine_factor = int(h.get("fine_factor", 4))
        self.anchors_enabled = bool(h.get("anchors", True))
        self.collect_training = bool(h.get("collect", True))
        self.weighted_projection = bool(h.get("weighted", True))
        self.gate_enabled = bool(h.get("gate", True))
        self.kc = KController(
            k_init=int(h.get("k_init", 8)), k_min=int(h.get("k_min", 2)),
            k_max=int(h.get("k_max", 64)),
            tau_lo=float(h.get("tau_lo", 0.02)),
            tau_hi=float(h.get("tau_hi", 0.10)),
            tau_hard=float(h.get("tau_hard", 0.30)))
        # Guard the fine-anchor cost up front: with the DEFAULT
        # SpaceChargeConfig grid (96^3) and fine_factor=4 the anchor
        # solver would be 384^3 -> a ~tens-of-GB IGF build on the first
        # kick.  The halo backend is meant to run on a COARSE grid; fail
        # fast with guidance instead of an OOM deep in the first anchor.
        if self.anchors_enabled:
            n_fine = (int(config.nx) * self.fine_factor,
                      int(config.ny) * self.fine_factor,
                      int(config.nz) * self.fine_factor)
            if max(n_fine) > 192:
                raise ValueError(
                    f"halo backend: fine anchor grid {n_fine} exceeds "
                    f"192^3 (grid {config.nx}x{config.ny}x{config.nz} x "
                    f"fine_factor {self.fine_factor}). Use a coarse grid "
                    f"(e.g. nx=ny=nz=24..48) and/or a smaller "
                    f"fine_factor, or disable anchors.")
        self.basis = BasisFieldCache(
            default_mode_indices(int(h.get("basis_degree", 4))),
            gauss_width_sigma=float(h.get("basis_width", 1.0)))
        self.net = None                    # set by load()/attach_net()
        self.net_scale_normalized = False  # net outputs coeffs per |E|_w
        self._fine_solver = None
        self._kick_index = 0
        self._s_mm = 0.0
        # per-interval alpha gate: only correct between anchors if the
        # last anchor showed the net beats no-correction; a persistent
        # failure (4 consecutive anchors) kills alpha globally
        self._alpha_gate = True
        self._bad_streak = 0
        self.period_len_mm = float(h.get("period_len_mm", 400.0))
        # telemetry + training log
        self.log = {"e": [], "k": [], "kick_of_anchor": [],
                    "features": [], "coeffs": [], "applied_alpha": []}

    # -- corrector weights -------------------------------------------------
    def attach_net(self, net) -> None:
        self.net = net

    def load(self, directory: str | Path) -> None:
        import torch
        from linac_gen.surrogates.base import MlpHead
        d = Path(directory)
        meta = json.loads((d / "metadata.json").read_text())
        net = MlpHead(meta["input_dim"], meta["output_dim"],
                      tuple(meta["hidden_dims"]))
        net.load_state_dict(torch.load(d / "weights.pt",
                                       map_location="cpu",
                                       weights_only=True))
        net.eval()
        self.net = net
        self.net_scale_normalized = bool(meta.get("scale_normalized", False))
        if "basis_degree" in meta:
            self.basis = BasisFieldCache(
                default_mode_indices(int(meta["basis_degree"])),
                gauss_width_sigma=float(meta.get("basis_width", 1.0)))

    def save_log(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            e=np.asarray(self.log["e"]),
            k=np.asarray(self.log["k"]),
            kick_of_anchor=np.asarray(self.log["kick_of_anchor"]),
            features=np.asarray(self.log["features"]),
            coeffs=np.asarray(self.log["coeffs"]),
            e_raw=np.asarray(self.log.get("e_raw", [])),
            capture=np.asarray(self.log.get("capture", [])),
            scale=np.asarray(self.log.get("scale", [])),
            r_after=np.asarray(self.log.get("r_after", [])),
        )

    # -- the corrected kick --------------------------------------------------
    def kick(self, beam, ds: float) -> None:                     # noqa: C901
        if beam.current <= 0 or beam.n_alive < 2:
            # NOTE: counters (_kick_index/_s_mm) intentionally do not
            # advance on skipped kicks — mirrors PicSolver's no-op; the
            # internal s-clock tracks SC kicks actually applied.
            return
        alive_mask = beam.alive_mask
        coords_lab = beam_to_spatial(beam)
        gamma = beam.ref.gamma
        coords_rest = boost_to_rest_frame(coords_lab, gamma)
        self._setup_grid(coords_rest)

        n_alive = beam.n_alive
        n_total = beam.n_particles
        current_A = beam.current * 1e-3
        freq_Hz = beam.bunch_frequency * 1e6
        macro_charge = (current_A / freq_Hz) / n_total
        charges = np.full(n_alive, macro_charge)

        deposit_fn, interpolate_fn = _select_kernel(
            getattr(self.config, "kernel", "cic"))
        rho = deposit_fn(coords_rest, charges,
                         self._grid_min, self._grid_max, self._n_grid)
        Ex, Ey, Ez = self._solver.solve(rho)

        is_anchor = (self.anchors_enabled
                     and self._kick_index % max(self.kc.k, 1) == 0)
        alpha_now = 0.0
        if (self.net is not None and not self.kc.alpha_killed
                and self._alpha_gate
                and self._kick_index >= self.warmup_kicks):
            alpha_now = self.alpha

        entry = None
        if alpha_now > 0.0 or is_anchor:
            sigma_now = coords_rest.std(axis=0)
            entry = self.basis.get(self._solver, self._grid_min,
                                   self._grid_max, self._n_grid,
                                   sigma_now)

        feats = None
        # On anchors, features are needed whenever a net exists (to score
        # the net against the anchor and drive the alpha gate / K
        # controller) — NOT only when collecting training data.  Gating
        # this on collect_training left eval runs unable to ever re-open
        # the alpha gate after a failed first anchor (net never scored
        # again; r_after stayed nan; corrector permanently off).
        if alpha_now > 0.0 or (is_anchor and (self.collect_training
                                              or self.net is not None)):
            feats = kick_features(
                coords_rest, self._s_mm, n_alive, self._n_grid,
                0.0, beam.current, self.period_len_mm)

        if is_anchor:
            dEx, dEy, dEz = self._anchor_defect(coords_rest, charges,
                                                Ex, Ey, Ez)
            # Beam-weighted everything: projection, norms, capture.  The
            # depressed tune is set by the core field; an unweighted L2
            # fit over the box spends the basis on the (huge, empty)
            # vacuum region and leaves the core grid error uncorrected
            # (measured: unweighted removed only ~20% of the 24^3 tune
            # error).  The density-weighted inner product is the one the
            # collective dynamics actually uses.
            if self.weighted_projection:
                w = self.basis.cell_weights(rho)
                c_star = self.basis.project_defect_weighted(
                    entry, dEx, dEy, dEz, w)
            else:
                w = np.ones(int(np.prod(self._n_grid)))
                c_star = self.basis.project_defect(entry, dEx, dEy, dEz)
            scale = self.basis.weighted_norm(w, Ex, Ey, Ez)
            d_norm = self.basis.weighted_norm(w, dEx, dEy, dEz)
            e_raw = d_norm / max(scale, 1e-30)
            # capture fraction: 1 - |residual|_w / |defect|_w
            pEx, pEy, pEz = self.basis.correction_field(entry, c_star)
            res = self.basis.weighted_norm(w, dEx - pEx, dEy - pEy,
                                           dEz - pEz)
            capture = float(np.sqrt(max(
                1.0 - (res / max(d_norm, 1e-300)) ** 2, 0.0)))
            # trust-region control variable = NET-vs-ANCHOR disagreement
            # in the captured subspace (the part the controller can act
            # on).  The uncapturable residual is a property of the grid
            # operating point, not of the net — throttling K cannot fix
            # it, so it must not drive K.  It stays visible via
            # e_raw/capture telemetry.
            r_after = np.nan
            if self.net is not None and feats is not None:
                c_net = self._net_coeffs(feats, scale)
                rEx, rEy, rEz = self.basis.correction_field(entry, c_net)
                e_ctl = self.basis.weighted_norm(
                    w, rEx - pEx, rEy - pEy, rEz - pEz) / max(scale, 1e-30)
                # alpha gate: correct the next interval only if the net
                # beats no-correction at THIS anchor.  The global kill
                # requires persistent failure ON ANCHORS THAT MATTER
                # (large raw defect): a sloppy prediction at a
                # small-defect anchor is already neutralized by the gate
                # and says nothing about the net where correction counts.
                r_after = self.basis.weighted_norm(
                    w, dEx - rEx, dEy - rEy, dEz - rEz) / max(scale, 1e-30)
                if self.gate_enabled:
                    self._alpha_gate = bool(r_after < e_raw)
                    if self._alpha_gate:
                        self._bad_streak = 0
                    elif e_raw > 0.15:
                        self._bad_streak += 1
                    if self._bad_streak >= 4:
                        self.kc.alpha_killed = True
            else:
                e_ctl = e_raw
            # kill authority stays with the r_after/e_raw streak logic
            # above whenever a net is present; e_ctl only paces K
            self.kc.update(e_ctl, allow_kill=self.net is None)
            self.log["e"].append(e_ctl)
            self.log.setdefault("e_raw", []).append(e_raw)
            self.log.setdefault("capture", []).append(capture)
            self.log.setdefault("scale", []).append(scale)
            self.log.setdefault("r_after", []).append(r_after)
            self.log["k"].append(self.kc.k)
            self.log["kick_of_anchor"].append(self._kick_index)
            if self.collect_training:
                self.log["features"].append(feats.copy())
                self.log["coeffs"].append(np.asarray(c_star))
            # the beam gets the fine field on anchor steps (free accuracy)
            Ex, Ey, Ez = Ex + dEx, Ey + dEy, Ez + dEz
        elif alpha_now > 0.0:
            scale_now = 1.0
            if self.net_scale_normalized:
                # Must use the SAME weighting regime as the anchor branch
                # (which the alpha gate validated against) — otherwise the
                # applied correction magnitude differs from the validated
                # one when weighted=False.
                if self.weighted_projection:
                    w_now = self.basis.cell_weights(rho)
                else:
                    w_now = np.ones(int(np.prod(self._n_grid)))
                scale_now = self.basis.weighted_norm(w_now, Ex, Ey, Ez)
            c = self._net_coeffs(feats, scale_now)
            cEx, cEy, cEz = self.basis.correction_field(entry, c)
            Ex = Ex + alpha_now * cEx
            Ey = Ey + alpha_now * cEy
            Ez = Ez + alpha_now * cEz
        self.log["applied_alpha"].append(alpha_now)

        E_at_particles = interpolate_fn(Ex, Ey, Ez, coords_rest,
                                        self._grid_min, self._grid_max,
                                        self._n_grid)
        ds_m = ds * 1e-3
        beta = beam.ref.beta
        mass = beam.ref.species.mass
        E_si = E_at_particles * 1e6
        z_state = abs(beam.ref.species.charge)
        factor_t = z_state * ds_m * 1e3 / (mass * 1e6 * beta**2 * gamma**2)
        factor_z = z_state * ds_m * 1e-6
        alive_indices = np.where(alive_mask)[0]
        beam.particles[alive_indices, 1] += factor_t * E_si[:, 0]
        beam.particles[alive_indices, 3] += factor_t * E_si[:, 1]
        beam.particles[alive_indices, 5] += factor_z * E_si[:, 2]

        self._kick_index += 1
        self._s_mm += ds

    def _net_coeffs(self, feats: np.ndarray, scale: float) -> np.ndarray:
        """Net forward pass -> basis coefficients (rescaled by the local
        field norm when the net was trained on scale-normalized targets)."""
        import torch
        with torch.no_grad():
            c = self.net(torch.from_numpy(feats).unsqueeze(0)
                         ).squeeze(0).numpy()
        if self.net_scale_normalized:
            c = c * scale
        return c

    # -- anchor: fine-grid solve of the SAME particles ---------------------
    def _anchor_defect(self, coords_rest, charges, Ex_c, Ey_c, Ez_c):
        """(fine - coarse) field at the coarse nodes."""
        n_fine = self._n_grid * self.fine_factor
        if self._fine_solver is None:
            self._fine_solver = PoissonSolverFFT(
                self._grid_min, self._grid_max, n_fine,
                use_gpu=getattr(self.config, "use_gpu", "auto"),
                green_kind=getattr(self.config, "green_kind", "igf"))
        elif not (np.array_equal(self._fine_solver.grid_min, self._grid_min)
                  and np.array_equal(self._fine_solver.grid_max,
                                     self._grid_max)):
            # Only when the box actually moved (adaptive grid mode) — the
            # IGF Green's rebuild costs seconds at fine resolution, and in
            # the default fixed-grid mode the box never moves after kick 0.
            self._fine_solver.update_grid(self._grid_min, self._grid_max)
        deposit_fn, interpolate_fn = _select_kernel(
            getattr(self.config, "kernel", "cic"))
        rho_f = deposit_fn(coords_rest, charges,
                           self._grid_min, self._grid_max, n_fine)
        Exf, Eyf, Ezf = self._fine_solver.solve(rho_f)
        # restrict the fine field to the coarse node positions by using
        # the gather kernel with the coarse nodes as "particles"
        axes = [np.linspace(self._grid_min[i], self._grid_max[i],
                            int(self._n_grid[i])) for i in range(3)]
        gx, gy, gz = np.meshgrid(*axes, indexing="ij")
        nodes = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        E_nodes = interpolate_fn(Exf, Eyf, Ezf, nodes,
                                 self._grid_min, self._grid_max, n_fine)
        shape = tuple(int(n) for n in self._n_grid)
        return (E_nodes[:, 0].reshape(shape) - Ex_c,
                E_nodes[:, 1].reshape(shape) - Ey_c,
                E_nodes[:, 2].reshape(shape) - Ez_c)

    def free_memory(self) -> None:
        if hasattr(super(), "free_memory"):
            super().free_memory()
        self._fine_solver = None
        # Basis entries are up to ~E(n_basis,3,n^3) each and the telemetry
        # lists grow per kick/anchor — release both (the Simulation keeps
        # the solver alive for backtrack reuse).
        self.basis._entries.clear()
        for lst in self.log.values():
            lst.clear()
