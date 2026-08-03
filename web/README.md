# HELIX landing site (`web/`)

A hand-written, dependency-free static site for HELIX — separate from the
MkDocs **manual** in `docs/`. It is designed to be served on GitHub Pages at
the **`/home/`** sub-path, alongside the manual which stays at the Pages root.

## Pages

| File | Purpose |
|------|---------|
| `index.html`        | Home — hero, solver ladder, workbench + copilot teasers |
| `features.html`     | Full feature matrix, architecture pipeline, space-charge/GPU table, I/O |
| `workbench.html`    | GUI gallery (tour gif, results/beam/lattice, phase space) |
| `ai-copilot.html`   | AI assistant deep-dive (voice, tour/drills, backends, MCP) |
| `get-started.html`  | Install / CLI / Python API / GUI / testing |
| `about.html`        | Project, citation (`#cite`), license (`#license`), links |
| `assets/css/helix.css` | The whole design system (one file) |
| `assets/js/helix.js`   | Nav toggle, hero ticker, copy buttons (progressive enhancement) |
| `assets/img/`          | Copies of the real figures from `docs/screenshots/` |

No build step, no framework. Every figure is a copy of a real HELIX output
already shipped in `docs/screenshots/` — the site is self-contained.

## Preview locally

```bash
cd web
python3 -m http.server 8791
# open http://localhost:8791/index.html
```

The **Manual** links (`href="../"`) resolve to the Pages root only once the
site is deployed under `/home/`; locally they'll 404, which is expected.

## Deploy

The site ships in the *same* GitHub Pages artifact as the manual. The
`.github/workflows/docs.yml` workflow builds the manual into `site/`; a single
added step copies this folder into `site/home/` before the artifact is
uploaded, so:

- `…github.io/HELIX/`        → the manual (unchanged)
- `…github.io/HELIX/home/`   → this landing site

## Design

"Beam-diagnostics console" — dark oscilloscope ink with an engineering
graticule, physics-notation eyebrows, and an instrument "scope panel" frame as
the recurring signature. Type: Space Grotesk (display) / IBM Plex Sans (body) /
JetBrains Mono (data & code). Fonts load from Google Fonts.
