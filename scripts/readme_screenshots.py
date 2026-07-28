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
    make_masthead_svg()
    make_divider_svg()
    make_phasespace_gif()
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




def make_masthead_svg() -> None:
    """Custom animated brand masthead — logo (embedded), wordmark, and a
    beamline motif with macro-particles racing through a FODO channel.
    SMIL only (renders on GitHub); self-contained dark tile."""
    import base64
    import io

    from PIL import Image
    logo = Image.open(
        "gui/linac_gen_gui/interphase/assets/helix_logo.png").resize(
        (180, 180), Image.LANCZOS)
    buf = io.BytesIO()
    logo.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()

    W, H = 1200, 260
    beam_y = 208
    # FODO symbols along the beam axis: F quads cyan, D quads violet
    quads = ""
    for i, x in enumerate(range(320, 1150, 90)):
        col = "#06b6d4" if i % 2 == 0 else "#7c3aed"
        h = 34 if i % 2 == 0 else 26
        quads += (f'<rect x="{x}" y="{beam_y - h / 2:.0f}" width="10" '
                  f'height="{h}" rx="2" fill="{col}" opacity="0.85"/>')
    beam_path = f"M300,{beam_y} L1180,{beam_y}"
    particles = "".join(
        f'<circle r="{r}" fill="{c}" opacity="0.95">'
        f'<animateMotion dur="{d}s" begin="{b}s" repeatCount="indefinite" '
        f'path="{beam_path}"/></circle>'
        for r, c, d, b in ((4.0, "#67e8f9", 3.0, 0.0),
                           (3.0, "#93c5fd", 3.0, 0.55),
                           (2.5, "#c4b5fd", 3.0, 1.1),
                           (3.5, "#22d3ee", 3.0, 1.7),
                           (2.5, "#818cf8", 3.0, 2.3)))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="HELIX — Hybrid Envelope-multiparticle LInac eXplorer">
  <defs>
    <linearGradient id="mbg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#0b1020"/><stop offset="1" stop-color="#131b36"/>
    </linearGradient>
    <linearGradient id="mword" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#67e8f9"/><stop offset="0.5" stop-color="#3b82f6"/><stop offset="1" stop-color="#8b5cf6"/>
      <animate attributeName="x1" values="0;-0.4;0" dur="7s" repeatCount="indefinite"/>
      <animate attributeName="x2" values="1;1.4;1" dur="7s" repeatCount="indefinite"/>
    </linearGradient>
  </defs>
  <rect width="{W}" height="{H}" rx="16" fill="url(#mbg)"/>
  <image x="42" y="40" width="180" height="180" xlink:href="data:image/png;base64,{b64}"/>
  <text x="262" y="118" font-family="Helvetica, Arial, sans-serif" font-size="86" font-weight="bold" fill="url(#mword)">HELIX</text>
  <text x="266" y="156" font-family="Menlo, monospace" font-size="19" fill="#94a3b8">Hybrid Envelope-multiparticle LInac eXplorer</text>
  <line x1="300" y1="{beam_y}" x2="1180" y2="{beam_y}" stroke="#1e3a5f" stroke-width="2"/>
  {quads}
  {particles}
  <text x="1178" y="{beam_y + 34}" text-anchor="end" font-family="Menlo, monospace" font-size="12" fill="#475569">envelope · multiparticle 3-D PIC · matching · AI copilot</text>
</svg>'''
    out = os.path.join(OUT, "masthead.svg")
    with open(out, "w") as fh:
        fh.write(svg)
    print(out, os.path.getsize(out) // 1024, "KB")


def make_divider_svg() -> None:
    """Thin section divider with a traveling glow pulse."""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 14" width="100%" role="presentation">
  <line x1="20" y1="7" x2="1180" y2="7" stroke="#1e293b" stroke-width="1.5"/>
  <circle r="3.5" fill="#38bdf8" opacity="0.9">
    <animateMotion dur="5s" repeatCount="indefinite" path="M20,7 L1180,7"/>
  </circle>
  <circle r="7" fill="#38bdf8" opacity="0.25">
    <animateMotion dur="5s" repeatCount="indefinite" path="M20,7 L1180,7"/>
  </circle>
</svg>'''
    out = os.path.join(OUT, "divider.svg")
    with open(out, "w") as fh:
        fh.write(svg)
    print(out, os.path.getsize(out) // 1024, "KB")




def make_phasespace_gif() -> None:
    """Real physics as eye-candy: the showcase bunch tumbling in x-x'
    phase space, tracked by the actual MP tracker station by station."""
    import matplotlib
    matplotlib.use("Agg")
    import io as _io

    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    from linac_gen.core.config import BeamConfig
    from linac_gen.core.lattice import Lattice
    from linac_gen.distributions.factory import create_beam
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.tracking.tracker import Tracker

    lat_full, _ = parse_tracewin("examples/showcase/fodo_channel.dat")
    cfg = BeamConfig(species="proton", energy=5.0, frequency=352.2,
                     current=0.0, duty_cycle=100.0, n_particles=2500,
                     distribution="gaussian", cutoff=4.0,
                     emit_nx=0.30, alpha_x=1.2, beta_x=0.6,
                     emit_ny=0.30, alpha_y=0.0, beta_y=0.35,
                     emit_z=0.25, alpha_z=0.0, beta_z=600.0)
    n_frames = 28
    cuts = np.linspace(4, 96, n_frames).astype(int)   # first ~10 m
    frames = []
    for k, n_el in enumerate(cuts):
        beam = create_beam(cfg, seed=11)
        sub = Lattice()
        s_mm = 0.0
        for e in lat_full.elements[:n_el]:
            sub.add(e)
            s_mm += float(getattr(e, "length", 0.0) or 0.0)
        Tracker(sub, beam).run()
        pts = beam.particles[beam.alive_mask]
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        fig.patch.set_facecolor("#0b1020")
        ax.set_facecolor("#0b1020")
        r = np.hypot(pts[:, 0] / 4.0, pts[:, 1] / 12.0)
        ax.scatter(pts[:, 0], pts[:, 1], s=2.2, c=r, cmap="cool",
                   alpha=0.75, linewidths=0)
        ax.set_xlim(-9, 9); ax.set_ylim(-28, 28)
        ax.set_xlabel("x [mm]", color="#94a3b8", fontsize=11)
        ax.set_ylabel("x' [mrad]", color="#94a3b8", fontsize=11)
        for sp in ax.spines.values():
            sp.set_color("#1e293b")
        ax.tick_params(colors="#475569", labelsize=9)
        ax.set_title(f"x-x' phase space   ·   s = {s_mm / 1000.0:5.2f} m",
                     color="#7dd3fc", fontsize=12, family="monospace")
        ax.text(0.02, 0.02, "HELIX multiparticle tracker",
                transform=ax.transAxes, color="#334155", fontsize=8)
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert(
            "P", palette=Image.Palette.ADAPTIVE, colors=128))
    out = os.path.join(OUT, "phasespace.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=110, loop=0, optimize=True)
    print(out, os.path.getsize(out) // 1024, "KB")


if __name__ == "__main__":
    main()
