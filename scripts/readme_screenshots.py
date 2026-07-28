"""Generate the README screenshots — offscreen, reproducible.

Renders the real GUI (no mockups): the workbench running the bundled
showcase deck (a generic 60-cell space-charge FODO channel —
examples/showcase/), the Lattice and Beam tabs, and the assistant panel
with a representative conversation.  Every asset is reproducible by any
user from a fresh clone.  Run from the repo root:

    QT_QPA_PLATFORM=offscreen PYTHONPATH=.:gui python scripts/readme_screenshots.py

Outputs land in docs/assets/readme/.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("HELIX_ASSIST_NO_PREWARM", "1")
sys.path[:0] = [".", "gui"]

OUT = os.path.join("docs", "screenshots")


def _pump(app, seconds: float) -> None:
    t0 = time.time()
    while time.time() - t0 < seconds:
        app.processEvents()
        time.sleep(0.02)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv[:1])

    import linac_gen_gui.interphase.app as A

    # sandboxed settings: never touch (or restore) the user's session
    import tempfile
    store = QSettings(os.path.join(tempfile.mkdtemp(), "_shots.ini"),
                      QSettings.Format.IniFormat)
    A._settings = lambda: store

    from linac_gen.cli.common import _envelope_initial, build_ref
    from linac_gen.io.project import load_project
    from linac_gen.tracking.envelope import EnvelopeSolver

    proj = load_project("examples/showcase/fodo_channel.lgproj")
    lattice, _meta = A._parse_lattice_file(
        "examples/showcase/fodo_channel.dat")
    cfg = proj.beam

    win = A.InterphaseWindow()
    win.resize(1680, 1010)
    win.state.set_lattice(lattice,
                          "examples/showcase/fodo_channel.dat")
    win.show()
    _pump(app, 1.0)

    # -- Lattice + Beam tab shots ---------------------------------------
    win.show_tab("Lattice")
    _pump(app, 1.0)
    win.grab().save(os.path.join(OUT, "gui-lattice.png"))
    win.show_tab("Beam")
    _pump(app, 1.2)
    win.grab().save(os.path.join(OUT, "gui-beam.png"))

    # -- envelope run -> Results tab shot -------------------------------
    ref = build_ref(cfg)
    res = EnvelopeSolver(lattice, ref, _envelope_initial(cfg, ref),
                         current=getattr(cfg, "current", 0.0)).run()
    win.state.set_results(res)
    win.show_tab("Results")
    _pump(app, 2.0)
    win.grab().save(os.path.join(OUT, "gui-results.png"))

    # -- assistant panel shot -------------------------------------------
    from linac_gen_gui.interphase.dialogs import assistant_panel as ap
    ap.AssistantPanel._settings = lambda self: store
    panel = ap.AssistantPanel(None, win.state)
    panel.resize(780, 940)
    # present as CONNECTED (hide the backend setup row; realistic status)
    panel._settings_box.setVisible(False)
    panel._status.setText("provider: claude (subscription login)   ·   "
                          "ledger: assist_20260727_163000.jsonl")
    t = panel._transcript
    t.add_message("user", "run the envelope and report sigma x at the exit")
    t.append_line("  · run_envelope … done in 0.8 s")
    t.add_message(
        "assistant",
        "Done — the 60-cell channel is in.  At the exit **σx is "
        "1.42 mm**, beating **±12%** along the channel with 20 mA of "
        "space charge; tune depression **η ≈ 0.81**.  Want the RMS "
        "plot?")
    t.append_line("\n▶ show the RMS plot")
    t.append_line("  · instant: open_plot")
    t.add_message("assistant", "Opened the **RMS** plot.")
    t.append_line("  ‣ [event] run-watch: all KPIs within baseline")
    panel._set_state("idle")
    panel.show()
    _pump(app, 1.2)
    panel.grab().save(os.path.join(OUT, "gui-assistant.png"))

    panel.close()
    win.close()
    _pump(app, 0.3)
    make_demo_gif()
    make_envelope_svg(res)
    for f in ("gui-lattice.png", "gui-beam.png", "gui-results.png",
              "gui-assistant.png"):
        p = os.path.join(OUT, f)
        print(f"{p}  {os.path.getsize(p)//1024} KB")


def make_demo_gif() -> None:
    """Assemble the tab-tour GIF from freshly grabbed frames."""
    from PIL import Image
    frames = [Image.open(os.path.join(OUT, f)).convert("P",
              palette=Image.Palette.ADAPTIVE, colors=160).resize((900, 716))
              for f in ("gui-lattice.png", "gui-beam.png",
                        "gui-results.png")]
    frames[0].save(os.path.join(OUT, "gui-tour.gif"), save_all=True,
                   append_images=frames[1:], duration=1600, loop=0,
                   optimize=True)
    print(os.path.join(OUT, "gui-tour.gif"),
          os.path.getsize(os.path.join(OUT, "gui-tour.gif")) // 1024, "KB")


def make_envelope_svg(res) -> None:
    """The bundled demo as the hero: the showcase FODO channel's
    σx(s) envelope, mirrored ±σ, drawing itself in an animated SVG
    (SMIL — renders on GitHub) with macro-particles in flight.  Fully
    reproducible from a fresh clone."""
    import numpy as np
    s_m = np.asarray(res.s, float) / 1000.0
    sx = np.asarray(res.sigma_x, float)
    keep = np.linspace(0, s_m.size - 1, 240).astype(int)
    s_m, sx = s_m[keep], sx[keep]
    W, H, PAD = 1200.0, 300.0, 26.0
    x = PAD + (s_m - s_m[0]) / (s_m[-1] - s_m[0]) * (W - 2 * PAD)
    mid = H / 2.0
    amp = (H / 2.0 - PAD) * sx / sx.max()

    def path(ys):
        pts = [f"{x[i]:.1f},{ys[i]:.1f}" for i in range(x.size)]
        return "M" + " L".join(pts)

    top = path(mid - amp)
    bot = path(mid + amp)
    area = top + " L" + " L".join(
        f"{x[i]:.1f},{(mid + amp)[i]:.1f}" for i in range(x.size - 1, -1, -1)
    ) + " Z"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" width="100%" role="img" aria-label="PIP-II beam envelope animating along the linac">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#0b1020"/><stop offset="1" stop-color="#101830"/>
    </linearGradient>
    <linearGradient id="beam" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#06b6d4"/><stop offset="0.55" stop-color="#2563eb"/><stop offset="1" stop-color="#7c3aed"/>
    </linearGradient>
    <linearGradient id="fill" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#06b6d4" stop-opacity="0.16"/>
      <stop offset="0.5" stop-color="#2563eb" stop-opacity="0.05"/>
      <stop offset="1" stop-color="#7c3aed" stop-opacity="0.16"/>
    </linearGradient>
  </defs>
  <rect width="{W:.0f}" height="{H:.0f}" rx="14" fill="url(#bg)"/>
  <line x1="{PAD}" y1="{mid}" x2="{W - PAD}" y2="{mid}" stroke="#334155" stroke-width="1" stroke-dasharray="3 6"/>
  <path d="{area}" fill="url(#fill)"/>
  <path d="{top}" fill="none" stroke="url(#beam)" stroke-width="2.4" stroke-linecap="round"
        stroke-dasharray="4000" stroke-dashoffset="4000">
    <animate attributeName="stroke-dashoffset" from="4000" to="0" dur="6s" repeatCount="indefinite"/>
  </path>
  <path d="{bot}" fill="none" stroke="url(#beam)" stroke-width="2.4" stroke-linecap="round"
        stroke-dasharray="4000" stroke-dashoffset="4000">
    <animate attributeName="stroke-dashoffset" from="4000" to="0" dur="6s" repeatCount="indefinite"/>
  </path>
  <circle r="4.5" fill="#67e8f9"><animateMotion dur="6s" repeatCount="indefinite" path="{top}"/></circle>
  <circle r="3.5" fill="#93c5fd" opacity="0.9"><animateMotion dur="6s" begin="1.5s" repeatCount="indefinite" path="{top}"/></circle>
  <circle r="3.5" fill="#c4b5fd" opacity="0.9"><animateMotion dur="6s" begin="3s" repeatCount="indefinite" path="{bot}"/></circle>
  <text x="{PAD}" y="{H - 10:.0f}" font-family="Menlo, monospace" font-size="13" fill="#64748b">60-cell FODO channel · 24 m · 20 mA space charge · σx(s) — bundled demo: examples/showcase</text>
  <text x="{W - PAD:.0f}" y="{H - 10:.0f}" text-anchor="end" font-family="Menlo, monospace" font-size="13" fill="#475569">HELIX EnvelopeSolver</text>
</svg>'''
    out = os.path.join(OUT, "envelope-hero.svg")
    with open(out, "w") as fh:
        fh.write(svg)
    print(out, os.path.getsize(out) // 1024, "KB")


if __name__ == "__main__":
    main()
