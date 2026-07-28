"""Proactive RunWatch: alert channels, one-shot/re-arm hysteresis,
rate limiting, stale-baseline honesty, headless session hook."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from linac_gen.assist.watch import RunWatch


def _res(n=100, tr_final=100.0, sigma_peak=None, emit_ratio=1.0,
         exit_sigma_x=2.0):
    s = np.linspace(0.0, 5000.0, n)
    tr = np.linspace(100.0, tr_final, n)
    sx = np.full(n, exit_sigma_x)
    sy = np.full(n, 1.5)
    if sigma_peak is not None:
        sy[n // 2] = sigma_peak
    return SimpleNamespace(
        s=s, transmission=tr, sigma_x=sx, sigma_y=sy,
        emit_x=np.linspace(0.25, 0.25 * emit_ratio, n),
        emit_y=np.full(n, 0.25), emit_z=np.full(n, 0.30),
        ref_w_kin=np.full(n, 3.0))


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_transmission_drop_fires_once_then_silently_rearms():
    clk = _Clock()
    w = RunWatch(clock=clk, min_interval_s=0.0)
    assert w.inspect(_res(tr_final=97.3)) == []      # first run: no ref
    clk.t += 100
    alerts = w.inspect(_res(tr_final=95.0))
    assert len(alerts) == 1
    assert "transmission fell" in alerts[0]
    assert "loss starts near s" in alerts[0]
    clk.t += 100
    # same bad level again: stabilized — NO repeat alert, silent re-arm
    assert w.inspect(_res(tr_final=95.0)) == []
    clk.t += 100
    # further degradation from the new level alerts again
    again = w.inspect(_res(tr_final=92.0))
    assert any("transmission fell 95.00" in a for a in again)


def test_transmission_genuine_recovery_announces():
    clk = _Clock()
    w = RunWatch(clock=clk, min_interval_s=0.0)
    w.inspect(_res(tr_final=100.0))
    clk.t += 100
    assert len(w.inspect(_res(tr_final=95.0))) == 1  # fire
    clk.t += 100
    rec = w.inspect(_res(tr_final=96.5))             # clearly recovered
    assert any("recovered" in a for a in rec)


def test_sigma_excursion_and_emit_growth_channels():
    w = RunWatch(min_interval_s=0.0)
    alerts = w.inspect(_res(sigma_peak=40.0, emit_ratio=3.0))
    kinds = " | ".join(alerts)
    assert "sigma_y blows up" in kinds
    assert "emit_x grew" in kinds
    # both channels one-shot
    assert w.inspect(_res(sigma_peak=40.0, emit_ratio=3.0)) == []


def test_rate_limit_suppresses_scan_spam():
    clk = _Clock()
    w = RunWatch(clock=clk, min_interval_s=60.0)
    w.inspect(_res(tr_final=100.0))
    clk.t += 1.0
    a1 = w.inspect(_res(tr_final=95.0))
    assert len(a1) == 1
    # recovery + immediate second drop within the rate window: silent
    clk.t += 1.0
    w.inspect(_res(tr_final=95.05))
    clk.t += 1.0
    assert w.inspect(_res(tr_final=90.0)) == []
    # after the window, fires again
    clk.t += 120.0
    w.inspect(_res(tr_final=95.05))
    clk.t += 1.0
    assert len(w.inspect(_res(tr_final=90.0))) == 1


def test_baseline_drift_and_stale_notice_once():
    base = {"identity": {"lattice_path": "A.dat", "n_elements": 4},
            "fingerprint": {"exit_sigma_x": 2.0}}
    w = RunWatch(baseline_loader=lambda: base, min_interval_s=0.0)
    me = {"lattice_path": "A.dat", "n_elements": 4}
    # healthy: no alert
    assert w.inspect(_res(), identity=me) == []
    # drifted exit sigma_x: 2.0 -> 2.5 = 25 % >> 5 % tolerance
    alerts = w.inspect(_res(exit_sigma_x=2.5), identity=me)
    assert any("off the\nhealthy baseline" in a.replace("the "
               "healthy", "the\nhealthy") or "healthy baseline" in a
               for a in alerts)
    # stale identity: ONE notice, never repeated
    other = {"lattice_path": "B.dat", "n_elements": 9}
    n1 = w.inspect(_res(), identity=other)
    assert any("DIFFERENT lattice" in a for a in n1)
    assert w.inspect(_res(), identity=other) == []


def test_none_and_empty_results_are_safe():
    w = RunWatch()
    assert w.inspect(None) == []
    assert w.inspect(SimpleNamespace(s=np.array([]))) == []


def test_headless_session_hook_emits_events(ctx, assist_config):
    """attach_watch: a compute tool completion feeds the watcher and
    alerts ride the event channel."""
    from linac_gen.assist.agent import AgentSession, Decision
    from linac_gen.assist.testing import (
        MockProvider, ScriptedApprover, turn_text, turn_tools,
    )
    events = []
    sess = AgentSession(
        assist_config, ctx,
        approver=ScriptedApprover([Decision.APPROVE, Decision.APPROVE]),
        provider=MockProvider([
            turn_tools(("diagnose", {})),        # read: no inspect
            turn_text("looked"),
        ]),
        on_event=events.append)
    clk = _Clock()
    w = RunWatch(clock=clk, min_interval_s=0.0)
    sess.attach_watch(w)
    # seed the previous-run reference, then hand the session a bad run
    w.inspect(_res(tr_final=100.0))
    ctx.set_results(_res(tr_final=90.0))
    # simulate a compute completion path directly
    sess._inspect_results()
    assert any(e.get("type") == "event"
               and "transmission fell" in e.get("text", "")
               for e in events)
    sess.close()
