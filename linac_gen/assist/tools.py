"""Declarative tool registry — the assistant's ONLY capabilities.

Every tool is a thin wrapper over an EXISTING audited HELIX API; the
agent orchestrates, it never computes, and there is deliberately no
shell / eval / arbitrary-file tool.  The capability boundary of the
whole assistant is exactly this registry.

Tiers (gate confirmation in the agent loop):
* ``read``    — no side effects; auto-approved.
* ``compute`` — expensive (runs a solver) and stores its RESULTS in
                the session; requires confirmation unless the session
                enables auto-approve; executed through the job manager
                so the loop never blocks.  Compute tools never mutate
                the user's INPUTS (lattice, beam) — a run that would
                (matching rewrites the lattice) is tagged ``mutate``.
* ``mutate``  — replaces user INPUT state (lattice/beam) or writes
                files; ALWAYS requires confirmation (even when compute
                auto-approve is on) with the resolved call echoed.

Every tool returns a ``ToolResult`` dict::

    {"status": "ok" | "error" | "refused",
     "data": <JSON-serializable>,
     "provenance": {...},        # where the numbers came from
     "warnings": [str, ...]}

HELIX's own refuse-loudly guards (matching audits, backtrack opt-ins,
parser strict mode) surface here as ``status="refused"`` with the
original message — the assistant inherits the code's honesty.
"""
from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Session context — the assistant's working state
# ---------------------------------------------------------------------------
@dataclass
class WorkContext:
    """What the assistant is currently working on.

    The CLI REPL owns a plain instance; the GUI subclasses it to proxy
    ``AppState`` so assistant actions stay in sync with the tabs.
    """
    lattice: object = None
    lattice_path: str = ""
    beam_config: object = None
    results: object = None
    results_path: str = ""
    calc_dir: str = "."

    # Overridable hooks (GUI adapter reroutes through AppState setters)
    def set_lattice(self, lattice, path: str) -> None:
        self.lattice = lattice
        self.lattice_path = path

    def set_beam_config(self, cfg) -> None:
        self.beam_config = cfg

    def set_results(self, results, path: str = "") -> None:
        self.results = results
        self.results_path = path

    # GUI navigation hooks — base has no GUI, so these are inert; the GUI
    # adapter overrides them to drive the app's tabs/plots via a queued
    # signal.  All return "nothing available" outside the desktop GUI.
    def available_tabs(self) -> list:
        return []

    def available_subtabs(self) -> dict:
        return {}

    def available_plots(self) -> list:
        return []

    def show_tab(self, tab: str, subtab=None):
        return None                  # -> tool reports "GUI only"

    def open_plot(self, name: str):
        return None                  # -> tool reports "GUI only"

    def highlight_element(self, index: int, s_mm: float) -> bool:
        return False                 # -> tool notes "GUI not available"

    def set_cursor(self, s_mm: float) -> bool:
        return False                 # -> tool reports "GUI only"

    def gui_context(self) -> dict:
        return {}                    # -> tool reports "GUI only"

    def grab_plot(self, name: str):
        return None                  # -> {"img_b64","mime","w","h","label"}

    def grab_screen(self):
        return None                  # -> {"img_b64","mime","w","h","label"}

    def run_gui_simulation(self, kind: str):
        return None                  # GUI: "started" | "busy" | error text


@dataclass
class Tool:
    name: str
    description: str
    schema: dict                     # JSON schema for the parameters
    tier: str                        # "read" | "compute" | "mutate"
    fn: Callable                     # fn(ctx, **params) -> ToolResult
    #: optional custom confirmation echo — render(params) -> str.  Used
    #: by plan-shaped tools (run_campaign) whose whole value is that the
    #: ONE confirmation shows the full numbered plan.
    render: Callable | None = None


TOOLS: dict[str, Tool] = {}


def _tool(name, description, schema, tier, render=None):
    def deco(fn):
        TOOLS[name] = Tool(name=name, description=description,
                           schema=schema, tier=tier, fn=fn,
                           render=render)
        return fn
    return deco


def _ok(data, provenance=None, warnings=None):
    return {"status": "ok", "data": data,
            "provenance": provenance or {}, "warnings": warnings or []}


def _err(msg, warnings=None):
    return {"status": "error", "data": {"message": str(msg)},
            "provenance": {}, "warnings": warnings or []}


def _refused(msg):
    return {"status": "refused", "data": {"message": str(msg)},
            "provenance": {}, "warnings": []}


_URL_SCHEMES = ("http://", "https://", "ftp://", "ftps://", "file://",
                "s3://", "gs://", "sftp://", "data:")


def _local_path(path):
    """Refuse anything that isn't a plain local filesystem path.

    HELIX itself makes no network connection; several readers (notably
    ``numpy.genfromtxt``) will silently GET a URL, which would be an
    SSRF / data-exfil egress.  Every file-taking tool passes its path
    through here first, so a URL is refused loudly (ValueError ->
    status='refused') instead of fetched."""
    p = str(path)
    low = p.strip().lower()
    for scheme in _URL_SCHEMES:
        if low.startswith(scheme):
            raise ValueError(
                f"refusing a non-local path {p!r}: HELIX tools read "
                f"local files only (no URLs — HELIX makes no network "
                f"connection).")
    return p


def _need(ctx, *what):
    missing = [w for w in what
               if getattr(ctx, w, None) is None]
    if missing:
        return _err(f"no {'/'.join(missing)} in the session — load one "
                    f"first (load_lattice / set_beam_config / "
                    f"load_results)")
    return None


def _ctx_provenance(ctx) -> dict:
    p = {}
    if ctx.lattice_path:
        p["lattice_path"] = str(ctx.lattice_path)
    if ctx.results_path:
        p["results_path"] = str(ctx.results_path)
    return p


def _capture(fn, *args, **kwargs):
    """Run *fn* capturing python warnings into the ToolResult channel;
    ValueError (HELIX's refusal idiom) maps to status='refused'."""
    with _warnings.catch_warnings(record=True) as rec:
        _warnings.simplefilter("always")
        try:
            out = fn(*args, **kwargs)
        except ValueError as exc:
            return None, _refused(exc), []
        except Exception as exc:               # noqa: BLE001
            return None, _err(f"{type(exc).__name__}: {exc}"), []
    return out, None, [str(w.message) for w in rec]


# ---------------------------------------------------------------------------
# READ tools
# ---------------------------------------------------------------------------
@_tool("get_status",
       "Current session state: loaded lattice, beam, results, jobs.",
       {"type": "object", "properties": {}, "required": []},
       "read")
def _get_status(ctx):
    lat = ctx.lattice
    data = {
        "lattice": ({"path": ctx.lattice_path,
                     "n_elements": len(lat.elements),
                     "length_m": sum(float(getattr(e, "length", 0.0) or 0)
                                     for e in lat.elements) / 1000.0}
                    if lat is not None else None),
        "beam_config": (getattr(ctx.beam_config, "__dict__", None)
                        and {k: v for k, v in vars(ctx.beam_config).items()
                             if isinstance(v, (int, float, str, bool))})
                       if ctx.beam_config is not None else None,
        "results_loaded": ctx.results is not None,
        "results_path": ctx.results_path,
        "calc_dir": ctx.calc_dir,
    }
    return _ok(data, _ctx_provenance(ctx))


@_tool("get_lattice_info",
       "Overview of the loaded lattice: element type counts, total "
       "length, RF frequency cards.",
       {"type": "object", "properties": {}, "required": []},
       "read")
def _get_lattice_info(ctx):
    gate = _need(ctx, "lattice")
    if gate:
        return gate
    counts: dict = {}
    freq = []
    total = 0.0
    for e in ctx.lattice.elements:
        counts[type(e).__name__] = counts.get(type(e).__name__, 0) + 1
        total += float(getattr(e, "length", 0.0) or 0.0)
        if type(e).__name__ == "Freq":
            freq.append(float(getattr(e, "frequency_mhz", 0.0) or
                              getattr(e, "frequency", 0.0)))
    return _ok({"element_counts": counts, "length_m": total / 1000.0,
                "freq_cards_mhz": freq,
                "n_elements": len(ctx.lattice.elements)},
               _ctx_provenance(ctx))


@_tool("describe_lattice",
       "Physicist-level rollup of the loaded lattice: how many RF "
       "cavities (powered vs parked), solenoids, quads, dipoles, "
       "correctors, BPMs…, lengths per kind, RF frequency sections.  "
       "THE tool for 'how many cavities' questions — field maps are "
       "classified by their field CHANNELS (RF vs static), not by "
       "class name or amplitude.",
       {"type": "object", "properties": {}, "required": []},
       "read")
def _describe_lattice(ctx):
    gate = _need(ctx, "lattice")
    if gate:
        return gate
    from linac_gen.lattice_semantics import summarize_lattice
    out, refusal, warns = _capture(summarize_lattice, ctx.lattice)
    if refusal:
        return refusal
    return _ok(out, _ctx_provenance(ctx), warns)


_PARAM_ATTRS = ("gradient", "field", "angle", "rho", "ke", "kb", "phase",
                "frequency", "voltage", "bx_l", "by_l")


def _element_params(e) -> dict:
    out = {}
    for attr in _PARAM_ATTRS:
        v = getattr(e, attr, None)
        if isinstance(v, (int, float)) and float(v) != 0.0:
            out[attr] = round(float(v), 6)
    return out


@_tool("list_lattice_elements",
       "List elements with physics semantics: name, class, KIND "
       "(cavity/solenoid/quad/dipole/corrector/diagnostic…), decoded "
       "field-map type + dimensionality, key parameters (ke/kb/"
       "gradient/angle/phase/frequency…), length.  'filter' matches "
       "kind OR class OR name substring; reports total_matched.",
       {"type": "object",
        "properties": {"filter": {"type": "string"},
                       "limit": {"type": "integer", "default": 40}},
        "required": []},
       "read")
def _list_elements(ctx, filter: str = "", limit: int = 40):
    gate = _need(ctx, "lattice")
    if gate:
        return gate
    from linac_gen.lattice_semantics import classify_element
    want = (filter or "").strip().lower()
    # semantic synonyms an operator uses
    kind_alias = {"cavities": "cavity", "cavs": "cavity", "rf": "cavity",
                  "solenoids": "solenoid", "quads": "quad",
                  "quadrupoles": "quad", "bends": "dipole",
                  "dipoles": "dipole", "steerers": "corrector",
                  "correctors": "corrector", "bpms": "diagnostic",
                  "diagnostics": "diagnostic"}
    want = kind_alias.get(want, want)
    rows = []
    total = 0
    limit = max(1, int(limit))
    import os as _os
    for i, e in enumerate(ctx.lattice.elements):
        cls = type(e).__name__
        name = str(getattr(e, "name", "?"))
        sem = classify_element(e)
        # the parser SYNTHESIZES names (FMAP_001…) and discards deck
        # labels — the field-map FILE name is the surviving human label
        ffile = _os.path.basename(str(getattr(e, "field_file", "")
                                      or ""))
        if want and (want not in cls.lower()
                     and want not in name.lower()
                     and want not in ffile.lower()
                     and want != sem["kind"]):
            continue
        total += 1                    # count EVERY match, past the limit
        if len(rows) < limit:
            row = {"index": i, "name": name, "class": cls,
                   "kind": sem["kind"],
                   "length_mm": float(getattr(e, "length", 0.0) or 0.0)}
            if sem["field_type"]:
                row["field_type"] = (f"{sem['field_type']}, "
                                     f"{sem['dims']}")
                row["powered"] = sem["powered"]
                geom = getattr(e, "geom", None)
                if geom is not None:
                    row["geom"] = int(geom)
                if ffile:
                    row["field_file"] = ffile
            params = _element_params(e)
            if params:
                row["params"] = params
            rows.append(row)
    return _ok({"elements": rows,
                "total_matched": total,
                "truncated": total > len(rows)},
               _ctx_provenance(ctx))


@_tool("read_file",
       "READ-ONLY windowed view of a local text file — defaults to the "
       "loaded lattice .dat, so the raw element cards (FIELD_MAP geom "
       "codes, SET_* commands…) are directly inspectable.  Page with "
       "start_line/max_lines, or pass 'pattern' (case-insensitive "
       "regex/substring) to get matching lines with numbers.  Local "
       "files only; binary files return a summary, never bytes; no "
       "writes ever.",
       {"type": "object",
        "properties": {
            "path": {"type": "string",
                     "description": "file to read; empty = the loaded "
                                    "lattice file"},
            "start_line": {"type": "integer", "default": 1},
            "max_lines": {"type": "integer", "default": 120,
                          "description": "lines per page (cap 200)"},
            "pattern": {"type": "string",
                        "description": "return only lines matching this "
                                       "case-insensitive regex"}},
        "required": []},
       "read")
def _read_file(ctx, path: str = "", start_line: int = 1,
               max_lines: int = 120, pattern: str = ""):
    import os as _os
    import re as _re
    p = str(path or "").strip() or (ctx.lattice_path or "")
    if not p:
        return _err("no path given and no lattice is loaded — pass "
                    "'path' or load_lattice first")
    try:
        p = _local_path(p)
    except ValueError as exc:
        return _refused(exc)
    p = _os.path.expanduser(p)
    if not _os.path.isfile(p):
        return _err(f"not a file: {p}")
    size = _os.path.getsize(p)
    with open(p, "rb") as f:
        if b"\x00" in f.read(4096):
            return _ok({"path": p, "binary": True, "size_bytes": size,
                        "message": "binary file — no bytes shown "
                                   "(HDF5 results: use read_provenance "
                                   "/ load_results)"},
                       _ctx_provenance(ctx))
    max_lines = max(1, min(int(max_lines), 200))
    start_line = max(1, int(start_line))
    rx = None
    if pattern:
        try:
            rx = _re.compile(pattern, _re.I)
        except _re.error:
            rx = _re.compile(_re.escape(pattern), _re.I)
    lines: list[str] = []
    total = 0
    matches = 0
    scan_cap = 500_000                    # huge-file safety
    with open(p, encoding="utf-8", errors="replace") as f:
        for n, raw in enumerate(f, start=1):
            total = n
            if n > scan_cap:
                break
            txt = raw.rstrip("\n")
            if rx is not None:
                if rx.search(txt):
                    matches += 1
                    if len(lines) < max_lines:
                        lines.append(f"{n:5d}| {txt[:300]}")
            elif start_line <= n < start_line + max_lines:
                lines.append(f"{n:5d}| {txt[:300]}")
    data = {"path": p, "total_lines": total, "lines": lines}
    if rx is not None:
        data["matches_total"] = matches
        data["truncated"] = matches > len(lines)
    else:
        data["truncated"] = total > start_line + max_lines - 1
    return _ok(data, _ctx_provenance(ctx))


@_tool("result_summary",
       "Compact scalar summary of the session results (final sigmas, "
       "emittances, transmission, output energy).  Multibunch train "
       "results get the per-bunch train summary (bunch counts, W_exit "
       "droop numbers) instead.",
       {"type": "object", "properties": {}, "required": []},
       "read")
def _result_summary(ctx):
    gate = _need(ctx, "results")
    if gate:
        return gate
    if _is_train_results(ctx.results):
        out, refusal, warns = _capture(_train_summary_data, ctx.results)
        if refusal:
            return refusal
        return _ok(out, _ctx_provenance(ctx), warns)
    from linac_gen.cli.common import result_summary
    out, refusal, warns = _capture(result_summary, ctx.results)
    if refusal:
        return refusal
    return _ok(out, _ctx_provenance(ctx), warns)


@_tool("query_results",
       "Value of a recorded quantity at position s (metres), linearly "
       "interpolated.  Quantities: sigma_x, sigma_y, sigma_phi, sigma_w, "
       "emit_x, emit_y, emit_z, emit_z_mmmrad, alpha_x, beta_x, alpha_y, "
       "beta_y, alpha_z, beta_z, transmission, ref_w_kin.",
       {"type": "object",
        "properties": {"quantity": {"type": "string"},
                       "s_m": {"type": "number"}},
        "required": ["quantity", "s_m"]},
       "read")
def _query_results(ctx, quantity: str, s_m: float):
    gate = _need(ctx, "results")
    if gate:
        return gate
    import numpy as np
    res = ctx.results
    ys = getattr(res, quantity, None)
    s = getattr(res, "s", None)
    if ys is None or s is None or not len(ys):
        return _err(f"results carry no '{quantity}'")
    s_mm = float(s_m) * 1000.0                  # results record s in mm
    val = float(np.interp(s_mm, np.asarray(s, dtype=float),
                          np.asarray(ys, dtype=float)))
    return _ok({"quantity": quantity, "s_m": float(s_m), "value": val},
               _ctx_provenance(ctx))


@_tool("summarize_beam",
       "Full beam-parameter table (Twiss all planes, RMS, centroid, "
       "emittances, halo, extents) of the exit distribution or a "
       "snapshot at s_m (multiparticle results only).",
       {"type": "object",
        "properties": {"s_m": {"type": "number",
                               "description": "snapshot position; omit "
                                              "for the exit beam"}},
        "required": []},
       "read")
def _summarize_beam(ctx, s_m: float | None = None):
    gate = _need(ctx, "results")
    if gate:
        return gate
    from linac_gen.diagnostics.beam_summary import summarize_particles
    res = ctx.results
    particles, ref, n_total, where = None, None, None, "exit"
    if s_m is not None:
        s_mm = float(s_m) * 1000.0
        try:
            particles = res.alive_at(s_mm)
            all_p, ref = res.beam_at(s_mm)
            n_total = len(all_p)
            where = f"snapshot s={s_m:g} m"
        except (KeyError, AttributeError):
            return _err(f"no snapshot at s={s_m} m (available: "
                        f"{sorted(getattr(res, '_snapshots', {})) or 'none'}"
                        f" [mm])")
    else:
        beam = getattr(res, "beam", None)
        if beam is None:
            return _err("results carry no exit particle distribution "
                        "(envelope run?) — use result_summary instead")
        particles, ref = beam.alive_particles, beam.ref
        n_total = beam.particles.shape[0]
    rows = summarize_particles(particles, ref, n_total=n_total,
                               location=where)
    return _ok({"rows": [{"group": g, "name": n, "value": v, "unit": u}
                         for (g, n, v, u) in rows]},
               _ctx_provenance(ctx))


@_tool("read_provenance",
       "Provenance attributes of a results HDF5 file (code commit, "
       "lattice/field-map hashes, seed, solver config, surrogates).",
       {"type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"]},
       "read")
def _read_provenance(ctx, path: str):
    import h5py
    try:
        path = _local_path(path)
    except ValueError as exc:
        return _refused(exc)
    try:
        with h5py.File(path, "r") as f:
            if "provenance" not in f:
                return _err(f"{path} has no provenance/ group")
            attrs = {k: (v.item() if hasattr(v, "item") else str(v))
                     for k, v in f["provenance"].attrs.items()}
    except OSError as exc:
        return _err(exc)
    return _ok(attrs, {"results_path": path})


@_tool("list_runs",
       "List saved result files (*.h5) in the calc directory, newest "
       "first, with timestamps.",
       {"type": "object",
        "properties": {"limit": {"type": "integer", "default": 20}},
        "required": []},
       "read")
def _list_runs(ctx, limit: int = 20):
    import os
    from pathlib import Path
    d = Path(ctx.calc_dir)
    if not d.is_dir():
        return _err(f"calc dir {d} does not exist")
    files = sorted(d.glob("**/*.h5"), key=os.path.getmtime, reverse=True)
    out = [{"path": str(p),
            "mtime": os.path.getmtime(p),
            "size_kb": round(p.stat().st_size / 1024, 1)}
           for p in files[:max(1, int(limit))]]
    return _ok({"runs": out, "calc_dir": str(d)})


@_tool("matched_input_twiss",
       "Matched INPUT Twiss for a transfer line: periodic solution of "
       "the FODO cell back-propagated to the entrance (transverse).",
       {"type": "object",
        "properties": {"cell_start": {"type": "integer"},
                       "cell_end": {"type": "integer"}},
        "required": []},
       "read")
def _matched_twiss(ctx, cell_start: int | None = None,
                   cell_end: int | None = None):
    gate = _need(ctx, "lattice", "beam_config")
    if gate:
        return gate
    from linac_gen.cli.common import build_ref
    from linac_gen.matching.periodic import (
        find_fodo_cells, find_matched_input_twiss,
    )
    ref = build_ref(ctx.beam_config)
    if cell_start is None or cell_end is None:
        cells = find_fodo_cells(ctx.lattice)
        if not cells:
            return _err("no FODO cell detected — pass cell_start/"
                        "cell_end explicitly")
        cell_start, cell_end = cells[0]
    out, refusal, warns = _capture(find_matched_input_twiss, ctx.lattice,
                                   ref, int(cell_start), int(cell_end))
    if refusal:
        return refusal
    data = {k: (float(v) if isinstance(v, (int, float)) else v)
            for k, v in out.items()}
    return _ok(data, _ctx_provenance(ctx), warns)


@_tool("web_search",
       "Search the web (DISABLED unless explicitly enabled in the "
       "assistant settings — HELIX is offline by default).",
       {"type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"]},
       "read")
def _web_search(ctx, query: str):
    # The agent injects the live config as ctx._assist_config.
    cfg = getattr(ctx, "_assist_config", None)
    if cfg is None or not getattr(cfg, "web_search_enabled", False):
        return _refused("web search is disabled (enable it explicitly "
                        "in the assistant settings)")
    import json as _json
    import urllib.parse
    import urllib.request
    url = ("https://duckduckgo.com/api?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "no_html": 1}))
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            payload = _json.loads(r.read().decode("utf-8", "replace"))
    except Exception as exc:                    # noqa: BLE001
        return _err(f"web search failed: {exc}")
    related = [t.get("Text", "") for t in payload.get("RelatedTopics", [])
               if isinstance(t, dict)][:5]
    return _ok({"abstract": payload.get("AbstractText", ""),
                "related": related, "source": "duckduckgo"})


@_tool("list_tabs",
       "List the desktop-GUI tabs (and their subtabs) you can switch to — "
       "Beam, Lattice, Matching, Numerics, Results, ...  Empty outside "
       "the GUI.",
       {"type": "object", "properties": {}, "required": []},
       "read")
def _list_tabs(ctx):
    return _ok({"tabs": list(ctx.available_tabs()),
                "subtabs": dict(ctx.available_subtabs())})


@_tool("show_tab",
       "Switch the desktop GUI to a tab (e.g. 'Results', 'Beam', "
       "'Matching', 'Lattice', 'Numerics') and optionally a subtab within "
       "it (e.g. Lattice→'Breakdown', Numerics→'Plot', Error Study→'Beam "
       "errors').  Use when the user asks to show/open/pull up a tab or "
       "view.  Names match case-insensitively; no-op outside the GUI.",
       {"type": "object",
        "properties": {
            "tab": {"type": "string",
                    "description": "tab name, e.g. 'Results'"},
            "subtab": {"type": "string",
                       "description": "optional subtab within the tab"}},
        "required": ["tab"]},
       "read")
def _show_tab(ctx, tab: str, subtab: str = None):
    shown = ctx.show_tab(str(tab), subtab)
    if shown:
        return _ok({"shown": shown})
    tabs = list(ctx.available_tabs())
    if tabs:
        return _refused(f"no tab matches {tab!r}; available: "
                        + ", ".join(tabs))
    return _refused("tab navigation is only available in the desktop GUI")


@_tool("list_plots",
       "List the result plots you can open in the GUI's Results tab "
       "(RMS, emittance, Twiss, phase space, energy, losses, ...).  Empty "
       "outside the GUI.",
       {"type": "object", "properties": {}, "required": []},
       "read")
def _list_plots(ctx):
    return _ok({"plots": list(ctx.available_plots())})


# ---------------------------------------------------------------------------
# manual search (offline; mkdocs index with markdown fallback)
# ---------------------------------------------------------------------------
_MANUAL_CACHE: dict = {}


def _manual_roots():
    """Candidate repo roots holding ``site/`` and/or ``docs/manual/``."""
    import os
    roots = [os.getcwd()]
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    roots.append(os.path.dirname(here))          # repo root above linac_gen/
    return roots


def _load_manual_sections():
    """[(title, location, plain_text)] — prebuilt mkdocs search index when
    present (1 section per heading), else raw ``docs/manual/**/*.md`` split
    on top headings.  Cached on (path, mtime)."""
    import glob
    import json as _json
    import os
    import re as _re

    for root in _manual_roots():
        idx = os.path.join(root, "site", "search", "search_index.json")
        if os.path.isfile(idx):
            key = (idx, os.path.getmtime(idx))
            if _MANUAL_CACHE.get("key") == key:
                return _MANUAL_CACHE["sections"]
            with open(idx, encoding="utf-8") as f:
                docs = _json.load(f).get("docs", [])
            tag = _re.compile(r"<[^>]+>")
            sections = [(d.get("title", ""), d.get("location", ""),
                         tag.sub(" ", d.get("text", "")))
                        for d in docs]
            _MANUAL_CACHE.update(key=key, sections=sections)
            return sections
    # fallback: raw markdown, one section per file
    for root in _manual_roots():
        man = os.path.join(root, "docs", "manual")
        if os.path.isdir(man):
            files = sorted(glob.glob(
                os.path.join(man, "**", "*.md"), recursive=True))
            newest = max((os.path.getmtime(p) for p in files), default=0)
            key = ("md", man, len(files), newest)   # invalidate on edits
            if _MANUAL_CACHE.get("key") == key:
                return _MANUAL_CACHE["sections"]
            sections = []
            for path in files:
                try:
                    text = open(path, encoding="utf-8").read()
                except OSError:
                    continue
                first = next((ln.lstrip("# ").strip()
                              for ln in text.splitlines()
                              if ln.startswith("#")), "")
                rel = os.path.relpath(path, man)
                sections.append((first or rel, rel, text))
            _MANUAL_CACHE.update(key=key, sections=sections)
            return sections
    return []


@_tool("search_manual",
       "Search the HELIX user manual (offline, local index) and return the "
       "best-matching sections with snippets.  Use to answer questions "
       "about HELIX features, elements, conventions, file formats — and "
       "cite the section titles you used.",
       {"type": "object",
        "properties": {
            "query": {"type": "string", "description": "search terms"},
            "k": {"type": "integer", "default": 5,
                  "description": "max results (1-10)"}},
        "required": ["query"]},
       "read")
def _search_manual(ctx, query: str, k: int = 5):
    sections = _load_manual_sections()
    if not sections:
        return _err("no manual found (neither site/search/search_index.json "
                    "nor docs/manual/*.md is reachable from here)")
    terms = [t for t in str(query).lower().split() if len(t) > 1]
    if not terms:
        return _refused("give at least one search term")
    scored = []
    for title, location, text in sections:
        tl, xl = title.lower(), text.lower()
        score = sum(3 * tl.count(t) + xl.count(t) for t in terms)
        if score > 0:
            scored.append((score, title, location, text))
    scored.sort(key=lambda r: -r[0])
    out = []
    for score, title, location, text in scored[:max(1, min(int(k), 10))]:
        pos = min((text.lower().find(t) for t in terms
                   if t in text.lower()), default=0)
        lo = max(0, pos - 80)
        # 700 chars, not 280: syntax answers are often TABLES (the
        # FIELD_MAP geom-digit table) that a narrow window clipped
        snippet = " ".join(text[lo:lo + 700].split())
        out.append({"title": title, "location": location,
                    "snippet": snippet})
    if not out:
        return _ok({"results": [], "note": f"no section matches {query!r}"})
    return _ok({"results": out, "sections_searched": len(sections)})


def _resolve_element(lattice, ident):
    """Resolve an element by index, exact name, or unique substring.
    Returns ``(index, element, [])`` on success, else
    ``(None, None, candidates)`` (empty candidates = nothing matched)."""
    els = lattice.elements
    try:
        i = int(ident)
        if 0 <= i < len(els):
            return i, els[i], []
    except (TypeError, ValueError):
        pass
    want = str(ident).strip().casefold()
    named = [(i, e) for i, e in enumerate(els)
             if getattr(e, "name", "")]
    exact = [(i, e) for i, e in named if e.name.casefold() == want]
    if len(exact) == 1:
        return exact[0][0], exact[0][1], []
    subs = [(i, e) for i, e in named
            if want and want in e.name.casefold()]
    if len(subs) == 1:
        return subs[0][0], subs[0][1], []
    pool = exact or subs
    return None, None, [e.name for _i, e in pool[:8]]


@_tool("highlight_element",
       "Locate a lattice element (by name, unique substring, or index) and "
       "highlight it in the GUI: switches to the Lattice tab, selects it "
       "(white outline + halo, listing scrolls to it) and moves the "
       "s-cursor to its entrance.  Outside the GUI it still reports the "
       "element's position.",
       {"type": "object",
        "properties": {"element": {"type": "string",
                                   "description": "element name, unique "
                                   "name fragment, or integer index"}},
        "required": ["element"]},
       "read")
def _highlight_element(ctx, element: str):
    gate = _need(ctx, "lattice")
    if gate:
        return gate
    idx, el, cand = _resolve_element(ctx.lattice, element)
    if idx is None:
        if cand:
            return _refused(f"ambiguous element {element!r} — candidates: "
                            + ", ".join(cand))
        return _refused(f"no element matches {element!r}")
    s_start, s_end = ctx.lattice.get_s_positions()
    s0, s1 = float(s_start[idx]), float(s_end[idx])
    shown = ctx.highlight_element(idx, s0)
    return _ok({"name": getattr(el, "name", f"#{idx}"),
                "type": type(el).__name__, "index": idx,
                "s_start_m": round(s0 * 1e-3, 4),
                "s_end_m": round(s1 * 1e-3, 4),
                "length_mm": round(s1 - s0, 3),
                "highlighted_in_gui": bool(shown)},
               _ctx_provenance(ctx))


@_tool("set_cursor",
       "Move the GUI's s-position cursor to a location along the lattice "
       "(metres).  Use to point at a position you are discussing, e.g. "
       "where the beam is largest.  GUI only.",
       {"type": "object",
        "properties": {"s_m": {"type": "number",
                               "description": "position along the lattice "
                               "in metres"}},
        "required": ["s_m"]},
       "read")
def _set_cursor(ctx, s_m: float):
    s_mm = max(0.0, float(s_m)) * 1e3
    if ctx.lattice is not None:
        try:
            s_mm = min(s_mm, float(ctx.lattice.total_length))
        except Exception:                                   # noqa: BLE001
            pass
    if ctx.set_cursor(s_mm):
        return _ok({"s_m": round(s_mm * 1e-3, 4)})
    return _refused("the s-cursor is only available in the desktop GUI")


@_tool("get_gui_context",
       "What the user currently sees in the desktop GUI: active tab, "
       "selected element, s-cursor position, open plot windows, and what "
       "is loaded.  Use to answer 'what am I looking at?' and to ground "
       "answers in the visible context.  GUI only.",
       {"type": "object", "properties": {}, "required": []},
       "read")
def _get_gui_context(ctx):
    snap = ctx.gui_context()
    if snap:
        return _ok(snap)
    return _refused("GUI context is only available in the desktop GUI")


def _save_capture(ctx, snap: dict, stem: str) -> dict:
    """Persist a grabbed image under calc_dir/assist_captures and return
    the tool payload (b64 + metadata).  The SDK backend turns img_b64 into
    an MCP image block; other transports strip it and keep the saved path."""
    import base64
    import itertools
    import os as _os
    out = dict(snap)
    try:
        cap = _os.path.join(getattr(ctx, "calc_dir", ".") or ".",
                            "assist_captures")
        _os.makedirs(cap, exist_ok=True)
        ext = ".jpg" if snap.get("mime") == "image/jpeg" else ".png"
        for n in itertools.count(len(_os.listdir(cap))):
            path = _os.path.join(cap, f"{stem}_{n:03d}{ext}")
            if not _os.path.exists(path):       # no overwrite on reuse
                break
        with open(path, "wb") as f:
            f.write(base64.b64decode(snap["img_b64"]))
        out["saved_to"] = path
    except Exception:                                       # noqa: BLE001
        pass
    return out


@_tool("look_at_plot",
       "LOOK at a result plot: opens it in the GUI, captures the rendered "
       "figure as an image, and returns it so you can analyse the visual "
       "structure (matching, halo, filamentation, oscillations).  Use "
       "before answering questions about how a plot LOOKS.  GUI only.",
       {"type": "object",
        "properties": {"name": {"type": "string",
                                "description": "plot name, e.g. 'phase "
                                "space', 'RMS'"}},
        "required": ["name"]},
       "read")
def _look_at_plot(ctx, name: str):
    snap = ctx.grab_plot(str(name))
    if not snap:
        plots = list(ctx.available_plots())
        if plots:
            return _refused(f"could not capture {name!r}; available plots: "
                            + ", ".join(plots))
        return _refused("looking at plots is only available in the "
                        "desktop GUI")
    return _ok(_save_capture(ctx, snap, "plot"), _ctx_provenance(ctx))


@_tool("look_at_screen",
       "LOOK at what the user currently sees: captures the active GUI tab "
       "(or frontmost plot window) as an image and returns it.  Use for "
       "'what am I looking at?' questions.  GUI only.",
       {"type": "object", "properties": {}, "required": []},
       "read")
def _look_at_screen(ctx):
    snap = ctx.grab_screen()
    if not snap:
        return _refused("looking at the screen is only available in the "
                        "desktop GUI")
    return _ok(_save_capture(ctx, snap, "screen"), _ctx_provenance(ctx))


@_tool("run_in_gui",
       "Start a simulation EXACTLY as the user's Run button does — same "
       "GUI settings (Numerics-tab grid/steps/integrator/backend, "
       "recording options), same progress bar, results into the tabs.  "
       "PREFER this over run_envelope/run_mp when working in the desktop "
       "GUI and the user hasn't asked for setting overrides.  Returns "
       "immediately; the run continues in the GUI (poll get_status).",
       {"type": "object",
        "properties": {"kind": {"type": "string",
                                "enum": ["envelope", "mp"],
                                "description": "'envelope' or 'mp' "
                                "(multiparticle)"}},
        "required": ["kind"]},
       "compute")
def _run_in_gui(ctx, kind: str):
    gate = _need(ctx, "lattice", "beam_config")
    if gate:
        return gate
    kind = str(kind).strip().lower()
    if kind not in ("envelope", "mp"):
        return _refused(f"kind must be 'envelope' or 'mp', not {kind!r}")
    status = ctx.run_gui_simulation(kind)
    if status is None:
        return _refused("run_in_gui presses the desktop GUI's Run button — "
                        "outside the GUI use run_envelope / run_mp instead")
    if status == "started":
        return _ok({"kind": kind, "state": "started",
                    "note": "launched exactly like the GUI Run button, "
                            "with the GUI's own settings; results will "
                            "appear in the tabs — check get_status /"
                            " result_summary when the run finishes"},
                   _ctx_provenance(ctx))
    if status == "busy":
        return _refused("a run is already in progress in the GUI — stop "
                        "it first or wait for it to finish")
    return _err(status)


@_tool("open_plot",
       "Open a result plot window in the GUI's Results tab by name (e.g. "
       "'phase space', 'RMS', 'emittance', 'Twiss', 'energy', 'loss "
       "profile', 'tune depression').  Use when the user asks to show/open/"
       "plot a specific result.  Switches to Results first; no-op outside "
       "the GUI.",
       {"type": "object",
        "properties": {"name": {"type": "string",
                                "description": "plot name, e.g. 'phase "
                                "space'"}},
        "required": ["name"]},
       "read")
def _open_plot(ctx, name: str):
    opened = ctx.open_plot(str(name))
    if opened:
        return _ok({"opened": opened})
    plots = list(ctx.available_plots())
    if plots:
        return _refused(f"no plot matches {name!r}; available: "
                        + ", ".join(plots))
    return _refused("plots can only be opened in the desktop GUI")


@_tool("notebook_note",
       "Write a note into the persistent lab notebook (survives across "
       "sessions).  Use when the user says 'note that down' or when a "
       "conclusion is worth remembering.",
       {"type": "object",
        "properties": {"text": {"type": "string",
                                "description": "the note to record"}},
        "required": ["text"]},
       "read")
def _notebook_note(ctx, text: str):
    if not str(text).strip():
        return _refused("empty note")
    from linac_gen.assist.notebook import append_note
    try:
        path = append_note(getattr(ctx, "calc_dir", ".") or ".", str(text))
    except Exception as exc:                                # noqa: BLE001
        return _err(f"could not write the notebook: {exc}")
    return _ok({"noted": str(text).strip(), "notebook": path})


@_tool("read_notebook",
       "Read the most recent lab-notebook entries (past sessions' runs, "
       "results and conclusions).  The last few entries are already in "
       "your context; use this to look further back.",
       {"type": "object",
        "properties": {"k_entries": {"type": "integer", "default": 5,
                                     "description": "how many entries"}},
        "required": []},
       "read")
def _read_notebook(ctx, k_entries: int = 5):
    from linac_gen.assist.notebook import load_tail, notebook_path
    calc_dir = getattr(ctx, "calc_dir", ".") or "."
    tail = load_tail(calc_dir, k_entries=max(1, min(int(k_entries), 20)),
                     max_chars=6000)
    if not tail:
        return _ok({"entries": "",
                    "note": f"no notebook yet at "
                            f"{notebook_path(calc_dir)}"})
    return _ok({"entries": tail})


@_tool("generate_report",
       "Write a markdown run report — lattice, provenance, exit-KPI table "
       "and (in the GUI) embedded figures — to <calc_dir>/reports/ and "
       "return its path.  Reports on the current in-session results, or "
       "on a results .h5 file if a path is given.",
       {"type": "object",
        "properties": {
            "path": {"type": "string",
                     "description": "optional results .h5 file (default: "
                     "current results)"},
            "title": {"type": "string",
                      "description": "optional report title"}},
        "required": []},
       "read")
def _generate_report(ctx, path: str = "", title: str = ""):
    import datetime as _dt
    import os as _os
    if path:
        try:
            p = _local_path(path)
        except ValueError as exc:
            return _refused(exc)
        from linac_gen.io.hdf5_output import load_results_hdf5
        try:
            res = load_results_hdf5(p)
        except Exception as exc:                            # noqa: BLE001
            return _err(f"{type(exc).__name__}: {exc}")
        source = p
    else:
        if ctx.results is None:
            return _refused("no results in the session — run or load "
                            "first, or give a results file path")
        res, source = ctx.results, "current session results"
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# {title or 'HELIX run report'}", "",
             f"*Generated {stamp} by the HELIX assistant*", "",
             f"- **Results**: {source}",
             f"- **Lattice**: {getattr(ctx, 'lattice_path', '') or '?'}"]
    prov = _ctx_provenance(ctx)
    for k, v in prov.items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## Exit KPIs", "", "| quantity | value |", "|---|---|"]
    units = {"sigma_x": "mm", "sigma_y": "mm", "sigma_phi": "deg",
             "sigma_w": "MeV", "emit_x": "mm·mrad", "emit_y": "mm·mrad",
             "emit_z": "deg·MeV", "transmission": "%", "ref_w_kin": "MeV"}
    for key in _COMPARE_KEYS:
        v = _rlast(res, key)
        if v is not None:
            lines.append(f"| {key} | {v:.6g} {units.get(key, '')} |")
    # Figures come from the LIVE GUI plots, i.e. the current session's
    # results — embed them only when reporting on those same results.
    # (Reporting on a *file* with the session's figures would falsify
    # the record.)  GUI-less contexts silently skip.
    figures = []
    if not path:
        for plot in ("RMS", "emittance", "energy"):
            snap = ctx.grab_plot(plot)
            if snap:
                figures.append((snap.get("label", plot), snap))
    rep_dir = _os.path.join(getattr(ctx, "calc_dir", ".") or ".",
                            "reports")
    _os.makedirs(rep_dir, exist_ok=True)
    import itertools as _it
    for n in _it.count(len([f for f in _os.listdir(rep_dir)
                            if f.endswith(".md")])):
        rep_path = _os.path.join(rep_dir, f"report_{n:03d}.md")
        if not _os.path.exists(rep_path):        # never overwrite
            break
    if figures:
        import base64 as _b64
        lines += ["", "## Figures", ""]
        for label, snap in figures:
            ext = ".jpg" if snap.get("mime") == "image/jpeg" else ".png"
            img = rep_path[:-3] + "_" + "".join(
                c if c.isalnum() else "_" for c in label)[:32] + ext
            try:
                with open(img, "wb") as f:
                    f.write(_b64.b64decode(snap["img_b64"]))
                lines.append(f"![{label}]({_os.path.basename(img)})")
                lines.append("")
            except Exception:                               # noqa: BLE001
                continue
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return _ok({"report": rep_path, "figures": len(figures),
                "kpis": sum(1 for k in _COMPARE_KEYS
                            if _rlast(res, k) is not None)}, prov)


def _rget(res, name):
    """Read a results field from either an attribute-style results object
    (in-session recorder/envelope) or a dict loaded by load_results_hdf5."""
    if isinstance(res, dict):
        return res.get(name)
    return getattr(res, name, None)


def _rlast(res, name):
    arr = _rget(res, name)
    try:
        return float(arr[-1]) if arr is not None and len(arr) else None
    except (TypeError, IndexError, ValueError):
        return None


_COMPARE_KEYS = ("sigma_x", "sigma_y", "sigma_phi", "sigma_w",
                 "emit_x", "emit_y", "emit_z", "transmission", "ref_w_kin")


@_tool("compare_runs",
       "Compare two result sets: exit KPIs side-by-side with deltas, plus "
       "max |Δσx|/|Δσy| along s when the grids match.  Each side is a "
       "results .h5 path or 'current' for the in-session results.",
       {"type": "object",
        "properties": {
            "run_a": {"type": "string",
                      "description": "results file path, or 'current'"},
            "run_b": {"type": "string",
                      "description": "results file path, or 'current'"}},
        "required": ["run_a", "run_b"]},
       "read")
def _compare_runs(ctx, run_a: str, run_b: str):
    def load(spec):
        if str(spec).strip().lower() == "current":
            if ctx.results is None:
                raise ValueError("no in-session results — run or load first")
            return ctx.results, "current session"
        path = _local_path(spec)
        from linac_gen.io.hdf5_output import load_results_hdf5
        return load_results_hdf5(path), path
    try:
        ra, name_a = load(run_a)
        rb, name_b = load(run_b)
    except ValueError as exc:
        return _refused(exc)
    except Exception as exc:                                # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")
    table = {}
    for key in _COMPARE_KEYS:
        a, b = _rlast(ra, key), _rlast(rb, key)
        if a is None and b is None:
            continue
        row = {"a": a, "b": b}
        if a is not None and b is not None:
            row["delta"] = b - a
            if abs(a) > 1e-30:
                row["pct"] = round(100.0 * (b - a) / a, 3)
        table[key] = row
    notes = []
    import numpy as _np
    for key in ("sigma_x", "sigma_y"):
        xa, xb = _rget(ra, key), _rget(rb, key)
        if xa is None or xb is None:
            continue
        xa, xb = _np.asarray(xa, float), _np.asarray(xb, float)
        if xa.shape == xb.shape and xa.size:
            i = int(_np.argmax(_np.abs(xb - xa)))
            sa = _rget(ra, "s")
            s_m = (float(_np.asarray(sa, float)[i]) * 1e-3
                   if sa is not None and len(sa) == len(xa) else None)
            table[f"max_abs_d{key}"] = {
                "value_mm": float(abs(xb - xa)[i]),
                "at_s_m": round(s_m, 3) if s_m is not None else None}
        else:
            notes.append(f"{key}: s-grids differ "
                         f"({xa.size} vs {xb.size} points) — exit only")
    return _ok({"run_a": name_a, "run_b": name_b, "exit_kpis": table,
                "notes": notes}, _ctx_provenance(ctx))


@_tool("parameter_scan",
       "Sweep ONE element parameter over N evenly spaced values, running a "
       "simulation per point (envelope by default — seconds/point), and "
       "return a value -> exit-metrics table (sigma, emittance, "
       "transmission, energy).  Long-running: executes as a background "
       "job.  The lattice is re-read from its FILE for each point; unsaved "
       "in-memory edits are not seen.",
       {"type": "object",
        "properties": {
            "element": {"type": "string",
                        "description": "element name or index"},
            "attribute": {"type": "string",
                          "description": "attribute to sweep, e.g. "
                          "'gradient'"},
            "start": {"type": "number"},
            "stop": {"type": "number"},
            "n_points": {"type": "integer", "default": 7,
                         "description": "2-41 points"},
            "mode": {"type": "string", "enum": ["envelope", "mp"],
                     "default": "envelope"}},
        "required": ["element", "attribute", "start", "stop"]},
       "compute")
def _parameter_scan(ctx, element: str, attribute: str, start: float,
                    stop: float, n_points: int = 7, mode: str = "envelope",
                    progress_callback=None, should_abort=None,
                    _assist_prov=None):
    gate = _need(ctx, "lattice", "beam_config")
    if gate:
        return gate
    lattice_path = getattr(ctx, "lattice_path", "") or ""
    import os as _os
    if not lattice_path or not _os.path.isfile(lattice_path):
        return _refused("parameter_scan re-reads the lattice from its file; "
                        "no readable lattice file path in this session")
    idx, el, cand = _resolve_element(ctx.lattice, element)
    if idx is None:
        if cand:
            return _refused(f"ambiguous element {element!r} — candidates: "
                            + ", ".join(cand))
        return _refused(f"no element matches {element!r}")
    if not hasattr(el, attribute):
        return _refused(f"element {getattr(el, 'name', idx)!r} has no "
                        f"attribute {attribute!r}")
    n = max(2, min(int(n_points), 41))
    import dataclasses as _dc

    import numpy as _np
    from linac_gen.parallel.scan_pool import (
        ScanPoint, _run_one_point_worker,
    )
    cfg = ctx.beam_config
    beam_dict = {f.name: getattr(cfg, f.name)
                 for f in _dc.fields(cfg)} if _dc.is_dataclass(cfg) \
        else dict(vars(cfg))
    step = getattr(ctx.lattice, "step_config", None)
    step1 = float(getattr(step, "integration_steps_per_metre", 100.0))
    step2 = float(getattr(step, "sc_steps_per_metre", 50.0))
    # Prefer the NAME selector when it is unique in the session lattice —
    # robust against structural in-memory edits shifting indices between
    # this lattice and the re-parsed file.  Fall back to @index (1-based).
    el_name = getattr(el, "name", "") or ""
    name_count = sum(1 for e in ctx.lattice.elements
                     if getattr(e, "name", None) == el_name)
    selector = (f"{el_name}.{attribute}" if el_name and name_count == 1
                else f"@{idx + 1}.{attribute}")
    # Int-typed attributes: the override coerces to the current type, so a
    # non-integral sweep value would be silently truncated — refuse loudly.
    if isinstance(getattr(el, attribute), int):
        bad = [float(v) for v in
               _np.linspace(float(start), float(stop), n)
               if abs(v - round(v)) > 1e-9]
        if bad:
            return _refused(
                f"{attribute!r} is integer-typed; sweep values like "
                f"{bad[0]:g} would be silently truncated — choose "
                f"integer start/stop/n_points")
    values = _np.linspace(float(start), float(stop), n)
    rows, aborted = [], False
    # scan_pool's worker setdefaults LINAC_GEN_FFT_WORKERS=1 (meant for
    # subprocess pools); running in-process, snapshot & restore it so one
    # scan doesn't permanently single-thread the host GUI's FFTs.
    _prev_fft = _os.environ.get("LINAC_GEN_FFT_WORKERS")
    try:
        for i, v in enumerate(values):
            if should_abort is not None and should_abort():
                aborted = True
                break
            point = ScanPoint(lattice_path=lattice_path,
                              beam_config=dict(beam_dict),
                              nx=32, grid_extent=5.0,
                              step1=step1, step2=step2,
                              mode=("envelope" if mode != "mp" else "mp"),
                              element_overrides=((selector, float(v)),))
            try:
                m = _run_one_point_worker(point)
            except Exception as exc:                        # noqa: BLE001
                rows.append({"value": float(v),
                             "error": f"{type(exc).__name__}: {exc}"})
                continue
            rows.append({"value": float(v),
                         **{k: m.get(k) for k in
                            ("sigma_x", "sigma_y", "sigma_phi", "emit_x",
                             "emit_y", "transmission", "ref_w_kin",
                             "elapsed")}})
            if progress_callback is not None:
                try:
                    progress_callback(float(i + 1), i + 1, n)
                except Exception:                           # noqa: BLE001
                    pass
    finally:
        if _prev_fft is None:
            _os.environ.pop("LINAC_GEN_FFT_WORKERS", None)
        else:
            _os.environ["LINAC_GEN_FFT_WORKERS"] = _prev_fft
    return _ok({"element": getattr(el, "name", f"#{idx}"),
                "attribute": attribute, "selector": selector,
                "mode": mode, "points": rows, "aborted": aborted,
                "note": f"lattice re-read from {lattice_path} per point"},
               _ctx_provenance(ctx))


# ---------------------------------------------------------------------------
# COMPUTE tools (dispatched through the job manager by the agent loop)
# ---------------------------------------------------------------------------
def _sc_config_from(params: dict):
    if not params.get("space_charge", True):
        return None
    from linac_gen.core.config import SpaceChargeConfig
    return SpaceChargeConfig(
        nx=int(params.get("grid", 32)), ny=int(params.get("grid", 32)),
        nz=int(params.get("grid", 32)),
        grid_extent=float(params.get("grid_extent", 5.0)),
        use_gpu="cpu")


@_tool("run_envelope",
       "Run the linear-envelope simulation (with linear space charge at "
       "the beam current) on the session lattice + beam.  Long-running: "
       "executes as a background job.",
       {"type": "object", "properties": {}, "required": []},
       "compute")
def _run_envelope(ctx, progress_callback=None, should_abort=None,
                  _assist_prov=None):
    gate = _need(ctx, "lattice", "beam_config")
    if gate:
        return gate
    from linac_gen.cli.common import _envelope_initial, build_ref
    from linac_gen.tracking.envelope import EnvelopeSolver
    ref = build_ref(ctx.beam_config)
    out, refusal, warns = _capture(
        lambda: EnvelopeSolver(
            ctx.lattice, ref, _envelope_initial(ctx.beam_config, ref),
            current=getattr(ctx.beam_config, "current", 0.0),
            progress_callback=progress_callback,
            should_abort=should_abort).run())
    if refusal:
        return refusal
    ctx.set_results(out, "")
    from linac_gen.cli.common import result_summary
    return _ok({"summary": result_summary(out)},
               _ctx_provenance(ctx), warns)


@_tool("run_mp",
       "Run the full multiparticle simulation (3-D PIC space charge) on "
       "the session lattice + beam.  Long-running background job.",
       {"type": "object",
        "properties": {
            "n_particles": {"type": "integer",
                            "description": "override macroparticle count"},
            "seed": {"type": "integer", "default": 42},
            "space_charge": {"type": "boolean", "default": True},
            "grid": {"type": "integer", "default": 32,
                     "description": "PIC grid per axis"},
            "grid_extent": {"type": "number", "default": 5.0}},
        "required": []},
       "compute")
def _run_mp(ctx, n_particles: int | None = None, seed: int = 42,
            space_charge: bool = True, grid: int = 32,
            grid_extent: float = 5.0,
            progress_callback=None, should_abort=None,
            _assist_prov=None):
    gate = _need(ctx, "lattice", "beam_config")
    if gate:
        return gate
    import copy
    from linac_gen.core.simulation import Simulation
    from linac_gen.distributions.factory import create_beam
    cfg = ctx.beam_config
    if n_particles:
        cfg = copy.deepcopy(cfg)
        cfg.n_particles = int(n_particles)
    sc = _sc_config_from({"space_charge": space_charge, "grid": grid,
                          "grid_extent": grid_extent})
    def _go():
        beam = create_beam(cfg, seed=int(seed))
        sim = Simulation(ctx.lattice, beam, space_charge=sc,
                         progress_callback=progress_callback,
                         should_abort=should_abort)
        rec = sim.run()
        rec.beam = beam
        return rec
    out, refusal, warns = _capture(_go)
    if refusal:
        return refusal
    ctx.set_results(out, "")
    from linac_gen.cli.common import result_summary
    return _ok({"summary": result_summary(out), "seed": int(seed)},
               _ctx_provenance(ctx), warns)


# ---------------------------------------------------------------------------
# multibunch / pulse study (opt-in; linac_gen.train)
# ---------------------------------------------------------------------------
def _is_train_results(res) -> bool:
    """True for the multibunch containers (TrainResults or the loader's
    namespace) — the routing predicate for train-aware summaries."""
    try:
        from linac_gen.train.results import (LoadedTrainResults,
                                             TrainResults)
    except ImportError:                     # pragma: no cover - core pkg
        return False
    return isinstance(res, (TrainResults, LoadedTrainResults))


def _train_summary_data(res) -> dict:
    """Compact per-run summary of a train result (live or loaded):
    bunch counts, W_exit statistics and signed loading droop numbers."""
    import numpy as np

    live = hasattr(res, "config")            # TrainResults vs loaded ns
    cfg = res.config if live else res
    pattern = cfg.pattern
    fast = res.fast
    # A fine-chopped full pulse RLE-encodes to hundreds of kB — cap the
    # summary copy (the full pattern lives in the HDF5 file).
    rle = pattern.to_rle()
    if len(rle) > 200:
        rle = rle[:200] + f" ...(+{len(rle) - 200} more chars)"
    data = {
        "run_type": "train",
        "mode": str(res.mode),
        "bunch_frequency_MHz": float(cfg.bunch_frequency_MHz),
        "n_slots": int(pattern.n_slots),
        "n_bunches_pattern": int(pattern.n_bunches),
        "n_bunches_tracked": len(res.slots),
        "n_bunches_replayed": len(getattr(res, "replay_bunches", {}) or {}),
        "pattern_rle": rle,
        "pulse_length_us": float(
            pattern.pulse_length_us(cfg.bunch_frequency_MHz)),
        "physics": {"beam_loading": bool(cfg.physics.beam_loading),
                    "hom": bool(cfg.physics.hom),
                    "direct_sc": bool(cfg.physics.direct_sc)},
        "truncated": bool(getattr(res, "truncated", False)
                          or (fast is not None
                              and getattr(fast, "truncated", False))),
    }

    def _w_stats(w, w_design):
        w = np.asarray(w, float)
        w = w[np.isfinite(w)]
        if not w.size:
            return None
        out = {"first": float(w[0]), "last": float(w[-1]),
               "min": float(w.min()), "max": float(w.max())}
        if w_design is not None:
            # Signed dW = W_exit - W_design (loading droop is negative).
            out["w_design_MeV"] = float(w_design)
            out["dw_min_keV"] = float((w.min() - w_design) * 1e3)
            out["dw_max_keV"] = float((w.max() - w_design) * 1e3)
            out["dw_last_keV"] = float((w[-1] - w_design) * 1e3)
        return out

    if fast is not None and len(getattr(fast, "w_exit_MeV", ())):
        data["fast_w_exit_MeV"] = _w_stats(fast.w_exit_MeV,
                                           float(fast.w_design_exit_MeV))
        data["fast_n_bunches"] = int(np.asarray(fast.slot).size)
    summ = res.summary() if callable(getattr(res, "summary", None)) \
        else (res.summary or {})
    wk = summ.get("ref_w_kin")
    if wk is not None and len(wk):
        w_design = None
        dr = getattr(res, "design_result", None)
        if dr is not None and len(getattr(dr, "ref_w_kin", ())):
            w_design = float(dr.ref_w_kin[-1])
        elif getattr(res, "w_design_exit_MeV", None) is not None:
            w_design = float(res.w_design_exit_MeV)
        data["tracked_w_exit_MeV"] = _w_stats(wk, w_design)
        tr = summ.get("transmission")
        if tr is not None and len(tr):
            t = np.asarray(tr, float)
            t = t[np.isfinite(t)]
            if t.size:
                data["transmission_pct"] = {"min": float(t.min()),
                                            "mean": float(t.mean())}
    return data


#: constructor knobs accepted per train mode beyond the shared set —
#: TrainRunner forwards unknown kwargs to Simulation, so mode-foreign
#: knobs must never reach it.
_TRAIN_MODE_KWARGS = {"fast": ("history_stride",),
                      "hybrid": ("history_stride", "replay_parallel")}


@_tool("run_train",
       "Run the OPT-IN multibunch / pulse study (linac_gen.train): a "
       "train of bunches at the bunch frequency with a chopped fill "
       "pattern, sequentially coupled through cavity beam loading, "
       "dipole-HOM wakes and/or direct bunch-to-bunch space charge.  "
       "Strictly opt-in: all physics flags default OFF (an all-off "
       "train is bit-identical to independent single-bunch runs), and "
       "missing physics inputs (e.g. the cavity_params sidecar) are "
       "refused loudly, never defaulted.  Modes: 'mp' (tracked, one "
       "full pass per bunch), 'envelope', 'fast' (per-slot phasor "
       "recursion over the full ~10^5-slot pulse in seconds), 'hybrid' "
       "(fast pass + full-MP replay of selected bunches).  Saves the "
       "train HDF5 (schema: linac_gen/train/results.py) and returns a "
       "compact summary with W_exit droop numbers.  Long-running "
       "background job.",
       {"type": "object",
        "properties": {
            "mode": {"type": "string", "default": "mp",
                     "enum": ["mp", "envelope", "fast", "hybrid"]},
            "bunch_frequency_MHz": {"type": "number",
                                    "description": "bunch-slot rate "
                                    "(the RF bunch frequency)"},
            "pattern": {"type": "string",
                        "description": "RLE fill pattern over the slot "
                        "axis, e.g. '1*10 0*54 1*26' (1=bunch, "
                        "0=chopped)"},
            "n_bunches": {"type": "integer",
                          "description": "alternative to 'pattern': "
                          "uniform train of this many bunches; with "
                          "duty_keep/duty_period it is the total SLOT "
                          "count of a periodically chopped pulse"},
            "duty_keep": {"type": "integer",
                          "description": "periodic chopping: keep the "
                          "first duty_keep of every duty_period slots "
                          "(requires n_bunches as the slot count)"},
            "duty_period": {"type": "integer"},
            "beam_loading": {"type": "boolean", "default": False,
                             "description": "fundamental-mode cavity "
                             "beam loading (needs cavity_params)"},
            "hom": {"type": "boolean", "default": False,
                    "description": "dipole-HOM long-range wakes / "
                    "cumulative BBU (needs cavity_params with "
                    "hom_modes)"},
            "direct_sc": {"type": "boolean", "default": False,
                          "description": "direct bunch-to-bunch space "
                          "charge via the PIC bunch-train images "
                          "(mode='mp'/'hybrid', numpy PIC)"},
            "cavity_params": {"type": "string",
                              "description": "sidecar JSON/YAML path "
                              "with per-cavity R/Q, Q_L, detuning, "
                              "hom_modes (matched to elements by name "
                              "pattern)"},
            "select_bunches": {"type": "array",
                               "items": {"type": "integer"},
                               "description": "hybrid only: absolute "
                               "slot indices to replay full-MP "
                               "(default: auto edge/probe selection)"},
            "replay_parallel": {"type": "boolean", "default": False,
                                "description": "hybrid only: replay "
                                "selected bunches in parallel worker "
                                "processes"},
            "history_stride": {"type": "integer", "default": 1,
                               "description": "fast/hybrid: record "
                               "per-cavity phasor histories every Nth "
                               "bunch"},
            "keep_full_results": {"type": "boolean", "default": True,
                                  "description": "False: summary-only "
                                  "per bunch (big tracked trains)"},
            "seed": {"type": "integer", "default": 42},
            "n_particles": {"type": "integer",
                            "description": "override macroparticle "
                            "count (tracked passes)"},
            "space_charge": {"type": "boolean", "default": True,
                             "description": "in-bunch space charge for "
                             "tracked passes (False = explicit off)"},
            "grid": {"type": "integer", "default": 32},
            "grid_extent": {"type": "number", "default": 5.0},
            "lattice_path": {"type": "string",
                             "description": "deck to run (default: the "
                             "session lattice)"},
            "out_path": {"type": "string",
                         "description": "output train HDF5 (default: "
                         "<calc_dir>/train_<timestamp>.h5)"}},
        "required": ["bunch_frequency_MHz"]},
       "compute")
def _run_train(ctx, bunch_frequency_MHz: float, mode: str = "mp",
               pattern: str = "", n_bunches: int | None = None,
               duty_keep: int | None = None, duty_period: int | None = None,
               beam_loading: bool = False, hom: bool = False,
               direct_sc: bool = False, cavity_params: str = "",
               select_bunches=None, replay_parallel: bool = False,
               history_stride: int = 1, keep_full_results: bool = True,
               seed: int = 42, n_particles: int | None = None,
               space_charge: bool = True, grid: int = 32,
               grid_extent: float = 5.0, lattice_path: str = "",
               out_path: str = "",
               progress_callback=None, should_abort=None,
               _assist_prov=None):
    import copy
    import datetime as _dt
    import os as _os

    gate = _need(ctx, "beam_config") if lattice_path \
        else _need(ctx, "lattice", "beam_config")
    if gate:
        return gate
    # ---- lattice -----------------------------------------------------
    if lattice_path:
        try:
            lattice_path = _local_path(lattice_path)
        except ValueError as exc:
            return _refused(exc)
        if not _os.path.isfile(lattice_path):
            return _err(f"lattice file not found: {lattice_path}")
        from linac_gen.cli.common import load_lattice
        lat, refusal, warns0 = _capture(load_lattice, lattice_path)
        if refusal:
            return refusal
        lat_path = lattice_path
    else:
        lat, warns0 = ctx.lattice, []
        lat_path = ctx.lattice_path or None
    # ---- pattern -----------------------------------------------------
    if pattern and n_bunches is not None:
        return _refused("give either 'pattern' (RLE) or 'n_bunches', "
                        "not both")
    if (duty_keep is None) != (duty_period is None):
        return _refused("duty_keep and duty_period must be given "
                        "together")
    if duty_keep is not None and pattern:
        return _refused("duty chopping composes with 'n_bunches' (the "
                        "slot count), not with an explicit 'pattern'")
    from linac_gen.train import PulsePattern, TrainConfig, TrainPhysics

    def _build_pattern():
        if pattern:
            return PulsePattern.from_rle(pattern)
        if n_bunches is None:
            raise ValueError(
                "no fill pattern: give 'pattern' (RLE string, e.g. "
                "'1*10 0*54 1*26') or 'n_bunches' (uniform train / "
                "slot count for duty chopping)")
        if duty_keep is not None:
            return PulsePattern.from_duty(int(n_bunches), int(duty_keep),
                                          int(duty_period))
        return PulsePattern.uniform(int(n_bunches))

    pat, refusal, _ = _capture(_build_pattern)
    if refusal:
        return refusal
    # ---- TrainConfig (its own loud validation surfaces verbatim) -----
    if cavity_params and not (beam_loading or hom):
        return _refused(
            "cavity_params given but neither beam_loading nor hom is "
            "enabled — the sidecar would be silently unused; enable a "
            "channel or drop it")
    if cavity_params:
        try:
            cavity_params = _local_path(cavity_params)
        except ValueError as exc:
            return _refused(exc)
    tc_kwargs = dict(
        bunch_frequency_MHz=float(bunch_frequency_MHz), pattern=pat,
        mode=str(mode),
        physics=TrainPhysics(direct_sc=bool(direct_sc),
                             beam_loading=bool(beam_loading),
                             hom=bool(hom)),
        cavity_params=(cavity_params or None), seed=int(seed),
        keep_full_results=bool(keep_full_results))
    if select_bunches is not None:
        tc_kwargs["select_bunches"] = list(select_bunches)
    tc, refusal, _ = _capture(lambda: TrainConfig(**tc_kwargs))
    if refusal:
        return refusal
    # ---- beam + space charge ----------------------------------------
    cfg = ctx.beam_config
    if n_particles:
        cfg = copy.deepcopy(cfg)
        cfg.n_particles = int(n_particles)
    # "off" (explicit opt-out) instead of None: a current-carrying train
    # would otherwise warn once per bunch.
    sc = _sc_config_from({"space_charge": space_charge, "grid": grid,
                          "grid_extent": grid_extent}) \
        if space_charge else "off"
    # ---- run ---------------------------------------------------------
    from linac_gen.train import run_train as _run

    def _cb(done, total):
        if progress_callback is not None:
            try:
                progress_callback(float(done), int(done), int(total))
            except Exception:                               # noqa: BLE001
                pass

    # Mode-foreign knobs are refused, not dropped: TrainRunner forwards
    # unknown kwargs to Simulation (TypeError), and silently ignoring a
    # caller's input is the failure mode this study bans.
    allowed = _TRAIN_MODE_KWARGS.get(tc.mode, ())
    if replay_parallel and "replay_parallel" not in allowed:
        return _refused(f"replay_parallel applies to mode='hybrid' only "
                        f"(got mode={tc.mode!r})")
    if int(history_stride) != 1 and "history_stride" not in allowed:
        return _refused(f"history_stride applies to the fast/hybrid "
                        f"modes only (got mode={tc.mode!r})")
    mode_kwargs = {}
    if "history_stride" in allowed:
        mode_kwargs["history_stride"] = max(1, int(history_stride))
    if "replay_parallel" in allowed and replay_parallel:
        mode_kwargs["replay_parallel"] = True
    out, refusal, warns = _capture(
        _run, lat, cfg, tc, sc_config=sc, lattice_path=lat_path,
        progress_callback=_cb, should_abort=should_abort, **mode_kwargs)
    if refusal:
        return refusal
    warns = list(warns0) + warns
    # ---- save + summarize -------------------------------------------
    if out_path:
        try:
            out_path = _local_path(out_path)
        except ValueError as exc:
            return _refused(exc)
    else:
        ts = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        calc = getattr(ctx, "calc_dir", ".") or "."
        _os.makedirs(calc, exist_ok=True)
        out_path = _os.path.join(calc, f"train_{ts}.h5")
    try:
        out.save_hdf5(str(out_path))
    except Exception as exc:                                # noqa: BLE001
        return _err(f"train ran but saving {out_path} failed: "
                    f"{type(exc).__name__}: {exc}", warns)
    ctx.set_results(out, str(out_path))
    data = _train_summary_data(out)
    data["output_path"] = str(out_path)
    prov = {"results_path": str(out_path)}
    if lat_path:
        prov["lattice_path"] = str(lat_path)
    return _ok(data, prov, warns)


@_tool("run_match",
       "Run the card-driven matcher (ADJUST/SET_* cards in the loaded "
       "lattice) on the session beam.  MUTATES the loaded lattice in "
       "place — the matched knob values replace the current ones — so "
       "it always asks for confirmation.  Long-running background job.",
       {"type": "object",
        "properties": {
            "algorithm": {"type": "string", "default": "least_squares",
                          "enum": ["least_squares", "cmaes", "bo",
                                   "differential_evolution",
                                   "dual_annealing", "gradient",
                                   "sequential_scan"]},
            "max_iter": {"type": "integer", "default": 100},
            "cost_solver": {"type": "string", "default": "envelope",
                            "enum": ["envelope", "mp"]}},
        "required": []},
       "mutate")
def _run_match(ctx, algorithm: str = "least_squares", max_iter: int = 100,
               cost_solver: str = "envelope",
               progress_callback=None, should_abort=None,
               _assist_prov=None):
    gate = _need(ctx, "lattice", "beam_config")
    if gate:
        return gate
    from linac_gen.matching import match
    out, refusal, warns = _capture(
        match, ctx.lattice, ctx.beam_config, algorithm=algorithm,
        max_iter=int(max_iter), cost_solver=cost_solver)
    if refusal:
        return refusal
    data = {
        "success": bool(out.success),
        "message": str(out.message),
        "cost": float(out.cost),
        "baseline_cost": (float(out.baseline_cost)
                          if out.baseline_cost is not None else None),
        "n_iter": int(out.n_iter),
        "z_penalty_infeasible": bool(out.z_penalty_infeasible),
        "x0": [float(v) for v in out.x0],
        "x_final": [float(v) for v in out.x_final],
        "report": out.report(),
    }
    return _ok(data, _ctx_provenance(ctx), warns)


@_tool("compare_to_tracewin",
       "Pearson correlation of session results vs a TraceWin export "
       "file for sigma_x/sigma_y/sigma_phi/sigma_w and energy.",
       {"type": "object",
        "properties": {"tracewin_file": {"type": "string"},
                       "mass_mev": {"type": "number",
                                    "default": 939.294308}},
        "required": ["tracewin_file"]},
       "compute")
def _compare_tw(ctx, tracewin_file: str, mass_mev: float = 939.294308,
                progress_callback=None, should_abort=None,
                _assist_prov=None):
    gate = _need(ctx, "results")
    if gate:
        return gate
    import numpy as np
    try:
        tw = np.genfromtxt(_local_path(tracewin_file), skip_header=1)
    except ValueError as exc:
        return _refused(exc)
    except OSError as exc:
        return _err(exc)
    if tw.ndim != 2 or tw.shape[1] < 22:
        return _err("unrecognised TraceWin export layout "
                    f"(shape {tw.shape})")
    res = ctx.results
    hs = np.asarray(res.s, dtype=float)          # mm
    ts = tw[:, 0] * 1000.0                       # m -> mm
    cols = {"sigma_x": tw[:, 12] * 1000.0, "sigma_y": tw[:, 14] * 1000.0,
            "sigma_phi": tw[:, 19], "sigma_w": tw[:, 21],
            "ref_w_kin": tw[:, 1] * float(mass_mev)}
    out = {}
    for name, tcol in cols.items():
        h = getattr(res, name, None)
        if h is None or not len(h):
            continue
        ti = np.interp(hs, ts, tcol)
        out[name] = float(np.corrcoef(np.asarray(h, float), ti)[0, 1])
    return _ok({"correlations": out, "tracewin_file": tracewin_file},
               _ctx_provenance(ctx))


# ---------------------------------------------------------------------------
# MUTATE tools (always confirmed)
# ---------------------------------------------------------------------------
@_tool("load_lattice",
       "Load a lattice file (.dat/.madx/.lat/.lte) or .lgproj project "
       "into the session (a project also loads its beam).",
       {"type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"]},
       "mutate")
def _load_lattice(ctx, path: str):
    from pathlib import Path as _P
    try:
        path = _local_path(path)
    except ValueError as exc:
        return _refused(exc)
    if not _P(path).exists():
        return _err(f"{path} not found")
    def _go():
        if str(path).endswith(".lgproj"):
            from linac_gen.cli.common import load_lattice
            from linac_gen.io.project import load_project
            proj = load_project(path)
            lat = load_lattice(proj.lattice_path)
            return lat, str(proj.lattice_path), proj.beam
        from linac_gen.cli.common import load_lattice
        return load_lattice(path), str(path), None
    out, refusal, warns = _capture(_go)
    if refusal:
        return refusal
    lat, lat_path, beam = out
    ctx.set_lattice(lat, lat_path)
    if beam is not None:
        ctx.set_beam_config(beam)
    return _ok({"n_elements": len(lat.elements),
                "beam_loaded": beam is not None,
                "parse_warnings": list(getattr(lat, "parse_warnings",
                                               []))[:10]},
               {"lattice_path": lat_path}, warns)


@_tool("load_results",
       "Load a previously saved results HDF5 into the session (as a "
       "plain arrays object for querying).  Multibunch train files "
       "(provenance run_type='train') are auto-detected and loaded "
       "through the train loader; result_summary then reports the "
       "per-bunch train summary.",
       {"type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"]},
       "mutate")
def _load_results(ctx, path: str):
    from types import SimpleNamespace
    from linac_gen.io.hdf5_output import load_results_hdf5
    try:
        path = _local_path(path)
    except ValueError as exc:
        return _refused(exc)
    # Train files carry none of the single-bunch envelope/reference
    # groups — load_results_hdf5 would return an EMPTY namespace
    # (silent uselessness).  Route on the train/ group instead.
    def _is_train_file(p):
        import h5py
        with h5py.File(p, "r") as f:
            return "train" in f
    is_train, refusal, _ = _capture(_is_train_file, path)
    if refusal:
        return refusal
    if is_train:
        from linac_gen.train.results import load_train_results
        out, refusal, warns = _capture(load_train_results, path)
        if refusal:
            return refusal
        ctx.set_results(out, str(path))
        data = _train_summary_data(out)
        data["note"] = ("multibunch train results loaded (per-bunch "
                        "summary via result_summary; python API: "
                        "linac_gen.train.load_train_results)")
        return _ok(data, {"results_path": str(path)}, warns)
    out, refusal, warns = _capture(load_results_hdf5, path)
    if refusal:
        return refusal
    ctx.set_results(SimpleNamespace(**out), str(path))
    return _ok({"arrays": sorted(out.keys())},
               {"results_path": str(path)}, warns)


@_tool("set_beam_config",
       "Update fields of the session beam configuration (partial "
       "update; unknown fields are refused).  NOTE conventions: emit_nx/"
       "emit_ny normalized pi.mm.mrad; emit_z deg.MeV; alpha_z is "
       "HELIX-internal = MINUS the TraceWin value.",
       {"type": "object",
        "properties": {"fields": {"type": "object"}},
        "required": ["fields"]},
       "mutate")
def _set_beam(ctx, fields: dict):
    import copy
    from linac_gen.core.config import BeamConfig
    cfg = (copy.deepcopy(ctx.beam_config)
           if ctx.beam_config is not None else BeamConfig())
    unknown = [k for k in fields if not hasattr(cfg, k)]
    if unknown:
        return _refused(f"unknown BeamConfig fields: {unknown}")
    for k, v in fields.items():
        setattr(cfg, k, v)
    ctx.set_beam_config(cfg)
    return _ok({"updated": sorted(fields.keys())})


@_tool("set_element_param",
       "Set one numeric parameter on a named lattice element "
       "(e.g. gradient on a quadrupole).",
       {"type": "object",
        "properties": {"element_name": {"type": "string"},
                       "param": {"type": "string"},
                       "value": {"type": "number"}},
        "required": ["element_name", "param", "value"]},
       "mutate")
def _set_param(ctx, element_name: str, param: str, value: float):
    gate = _need(ctx, "lattice")
    if gate:
        return gate
    elem = next((e for e in ctx.lattice.elements
                 if getattr(e, "name", None) == element_name), None)
    if elem is None:
        return _err(f"element '{element_name}' not in lattice")
    if not hasattr(elem, param):
        return _refused(f"'{type(elem).__name__}' has no parameter "
                        f"'{param}'")
    old = getattr(elem, param)
    if not isinstance(old, (int, float)):
        return _refused(f"'{param}' is not numeric (is {type(old).__name__})")
    setattr(elem, param, float(value))
    return _ok({"element": element_name, "param": param,
                "old": float(old), "new": float(value)},
               _ctx_provenance(ctx))


@_tool("write_results",
       "Write the session results to disk (hdf5 / openpmd / partran), "
       "with full provenance.",
       {"type": "object",
        "properties": {"path": {"type": "string"},
                       "format": {"type": "string", "default": "hdf5",
                                  "enum": ["hdf5", "openpmd", "partran"]}},
        "required": ["path"]},
       "mutate")
def _write_results(ctx, path: str, format: str = "hdf5",
                   _assist_prov=None):
    gate = _need(ctx, "results")
    if gate:
        return gate
    from linac_gen.cli.common import write_results
    try:
        path = _local_path(path)
    except ValueError as exc:
        return _refused(exc)
    out, refusal, warns = _capture(
        write_results, ctx.results, path, format, ctx.beam_config,
        ctx.lattice, lattice_path=ctx.lattice_path or None)
    if refusal:
        return refusal
    return _ok({"written": str(out)},
               {"results_path": str(out),
                **_ctx_provenance(ctx)}, warns)


# ---------------------------------------------------------------------------
def render_call(name: str, params: dict) -> str:
    """The echo-back line shown at every confirmation: the EXACT
    resolved call, one param per line.  A tool with a custom ``render``
    (plan-shaped tools) supplies its own echo — the ONE confirmation
    must show the full numbered plan."""
    tool = TOOLS.get(name)
    if tool is not None and tool.render is not None:
        try:
            return tool.render(params or {})
        except Exception:                                   # noqa: BLE001
            pass                     # fall through to the generic echo
    tier = tool.tier if tool else "?"
    lines = [f"{name}  [{tier}]"]
    for k, v in (params or {}).items():
        lines.append(f"    {k} = {v!r}")
    return "\n".join(lines)


def provider_tool_specs() -> list[dict]:
    """The registry rendered as provider-neutral tool specs."""
    return [{"name": t.name, "description": t.description,
             "input_schema": t.schema} for t in TOOLS.values()]


# ---------------------------------------------------------------------------
# extension registries — importing registers their tools (side-effect by
# design: one import site, every transport + MCP picks them up)
# ---------------------------------------------------------------------------
from linac_gen.assist import tools_analysis  # noqa: E402,F401  (registry)
from linac_gen.assist import tools_campaign  # noqa: E402,F401  (registry)
from linac_gen.assist import tools_training  # noqa: E402,F401  (registry)
from linac_gen.assist import tools_grad  # noqa: E402,F401  (registry)
