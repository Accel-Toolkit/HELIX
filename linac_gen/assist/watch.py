"""Proactive run watching — the assistant speaks up on its own.

MIRAGE's MachineWatch discipline (one-shot alerts with re-arm
hysteresis, stale-baseline honesty, zero tokens until something fires)
without its 30 s poller: a simulator has nothing ticking, so the hook
is the app's ``results_changed`` signal — every completed run,
INCLUDING user-initiated ones, gets inspected by pure numpy in ~ms.

Alert channels:
* transmission — the final transmission fell vs the previous run
  (s-localized via the shared onset helper);
* sigma excursion — σ blew up vs the run's own early-lattice scale;
* emittance growth — end/start ≥ 2× in any plane;
* baseline drift — exit KPIs moved vs the stored anomaly baseline
  (identity-checked; a stale baseline is announced ONCE, never
  silently ignored).

Every channel is one-shot-per-arm with re-arm on recovery, plus a
per-channel rate limit so a parameter scan cannot spam.  Alerts are
plain strings the caller feeds to ``AgentSession.submit_event`` —
rendered immediately, narrated when idle narration is on.
"""
from __future__ import annotations

import time


class RunWatch:
    #: (alert threshold, re-arm threshold) per channel
    TRANSMISSION_DROP_PTS = 0.5
    TRANSMISSION_REARM_PTS = 0.25
    SIGMA_FACTOR = 6.0
    SIGMA_REARM_FACTOR = 4.0
    EMIT_FACTOR = 2.0
    EMIT_REARM_FACTOR = 1.5
    BASELINE_Z = 1.5
    BASELINE_REARM_Z = 1.0

    def __init__(self, baseline_loader=None, clock=time.monotonic,
                 min_interval_s: float = 10.0):
        """``baseline_loader()`` -> the assist_baseline.json payload (or
        None); injectable clock for tests."""
        self._load_baseline = baseline_loader or (lambda: None)
        self._clock = clock
        self._min_interval = float(min_interval_s)
        self._armed: dict[str, bool] = {}
        self._last_fire: dict[str, float] = {}
        self._prev_final_tr: float | None = None
        self._stale_notified = False

    # ------------------------------------------------------------------
    def _fire(self, channel: str, text: str, out: list[str]) -> None:
        now = self._clock()
        if not self._armed.get(channel, True):
            return
        if now - self._last_fire.get(channel, -1e9) < self._min_interval:
            return
        self._armed[channel] = False        # one shot per arm
        self._last_fire[channel] = now
        out.append(text)

    def _rearm(self, channel: str, recovered: bool,
               out: list[str], recovery_text: str = "") -> None:
        if recovered and not self._armed.get(channel, True):
            self._armed[channel] = True
            if recovery_text:
                out.append(recovery_text)

    # ------------------------------------------------------------------
    def inspect(self, results, identity: dict | None = None) -> list[str]:
        """Pure numpy over one finished run; returns alert strings."""
        import numpy as np
        out: list[str] = []
        if results is None:
            return out
        s = np.asarray(getattr(results, "s", []), dtype=float)
        if s.size == 0:
            return out

        # -- transmission vs the previous run --------------------------
        # Semantics: a drop vs the previous run fires ONCE; while
        # disarmed, a level that STABILIZES re-arms silently (so further
        # degradation alerts again) and a genuine recovery above the
        # alert level announces itself.
        tr = getattr(results, "transmission", None)
        if tr is not None and len(tr):
            tr = np.asarray(tr, dtype=float)
            final = float(tr[-1])
            prev = self._prev_final_tr
            if prev is not None:
                drop = prev - final
                if self._armed.get("transmission", True):
                    if drop > self.TRANSMISSION_DROP_PTS:
                        from linac_gen.assist.tools_analysis import (
                            first_drop_onset,
                        )
                        onset = first_drop_onset(s, tr)
                        where = (f" (loss starts near s = "
                                 f"{onset / 1000.0:.1f} m)"
                                 if onset is not None else "")
                        self._fire("transmission",
                                   f"run watch: transmission fell "
                                   f"{prev:.2f} % -> {final:.2f} % "
                                   f"({drop:.2f} pts vs the previous "
                                   "run)" + where, out)
                        self._alert_tr = final
                else:
                    if final > getattr(self, "_alert_tr", final) \
                            + self.TRANSMISSION_REARM_PTS:
                        self._armed["transmission"] = True
                        out.append("run watch: transmission recovered "
                                   f"({final:.2f} %)")
                    elif abs(drop) <= self.TRANSMISSION_REARM_PTS:
                        # stabilized at the new level — re-arm silently
                        self._armed["transmission"] = True
            self._prev_final_tr = final

        # -- sigma excursion vs the run's own early scale ---------------
        for plane in ("sigma_x", "sigma_y"):
            sig = getattr(results, plane, None)
            if sig is None or not len(sig) or len(sig) != s.size:
                continue
            sig = np.asarray(sig, dtype=float)
            ref = float(np.median(sig[:max(5, sig.size // 20)]))
            if ref <= 0:
                continue
            peak = float(np.max(sig))
            ch = f"excursion_{plane}"
            if peak > self.SIGMA_FACTOR * ref:
                i = int(np.argmax(sig))
                self._fire(ch,
                           f"run watch: {plane} blows up to "
                           f"{peak:.2f} mm ({peak / ref:.0f}× its "
                           f"early value) near s = "
                           f"{float(s[i]) / 1000.0:.1f} m", out)
            else:
                self._rearm(ch, peak < self.SIGMA_REARM_FACTOR * ref,
                            out)

        # -- emittance growth ------------------------------------------
        for em in ("emit_x", "emit_y", "emit_z"):
            e = getattr(results, em, None)
            if e is None or len(e) < 2:
                continue
            e0, e1 = float(e[0]), float(e[-1])
            if e0 <= 0:
                continue
            ratio = e1 / e0
            ch = f"growth_{em}"
            if ratio > self.EMIT_FACTOR:
                self._fire(ch, f"run watch: {em} grew ×{ratio:.1f} "
                               "start → end", out)
            else:
                self._rearm(ch, ratio < self.EMIT_REARM_FACTOR, out)

        # -- drift vs the stored anomaly baseline ----------------------
        base = None
        try:
            base = self._load_baseline()
        except Exception:                                   # noqa: BLE001
            base = None
        if base:
            ident = base.get("identity", {})
            if identity is not None and (
                    ident.get("lattice_path")
                    != identity.get("lattice_path")
                    or ident.get("n_elements")
                    != identity.get("n_elements")):
                if not self._stale_notified:
                    self._stale_notified = True
                    out.append(
                        "run watch: the anomaly baseline was made for "
                        "a DIFFERENT lattice — baseline drift checks "
                        "are OFF until anomaly_baseline is re-run")
            else:
                self._stale_notified = False
                fp = base.get("fingerprint", {})
                from linac_gen.assist.tools_analysis import _TOLERANCES
                worst, worst_key = 0.0, ""
                for key, (rel, floor) in _TOLERANCES.items():
                    b = fp.get(f"exit_{key}")
                    col = getattr(results, key, None)
                    if b is None or col is None or not len(col):
                        continue
                    import numpy as np
                    c = float(np.asarray(col)[-1])
                    z = abs(c - float(b)) / max(rel * abs(b), floor)
                    if z > worst:
                        worst, worst_key = z, key
                if worst > self.BASELINE_Z:
                    self._fire("baseline",
                               f"run watch: {worst_key} is "
                               f"{worst:.1f} tolerance-bands off the "
                               "healthy baseline", out)
                else:
                    self._rearm("baseline",
                                worst < self.BASELINE_REARM_Z, out)
        return out
