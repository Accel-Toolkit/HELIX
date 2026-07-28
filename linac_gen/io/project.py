"""Standalone ``.lgproj`` project-file loader for the headless CLI.

A ``.lgproj`` is plain JSON written by the GUI.  This module reconstructs
a :class:`~linac_gen.core.config.BeamConfig` and a convergence-settings
dict from it **without importing any GUI code** — the CLI must run in a
GUI-free environment.

Unknown keys are ignored, so the loader tolerates project files written
by newer or older GUI versions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

from linac_gen.core.config import BeamConfig

__all__ = ["ProjectConfig", "load_project"]


@dataclass
class ProjectConfig:
    """Parsed contents of a ``.lgproj`` file.

    Attributes
    ----------
    lattice_path :
        Absolute path to the lattice file (a relative path in the
        project is resolved against the project-file directory).
    beam :
        The reconstructed :class:`BeamConfig`.
    convergence :
        The raw ``convergence`` sub-dict (step density, PIC grid,
        kernel, backend, csr_enabled, …) — kept as a plain dict; the
        CLI resolves it against command-line overrides.
    source_path :
        Path of the ``.lgproj`` file itself.
    """

    lattice_path: str
    beam: BeamConfig
    convergence: dict = field(default_factory=dict)
    source_path: str = ""


def _beam_from_dict(d: dict) -> BeamConfig:
    """Build a BeamConfig from a project ``beam`` dict, ignoring any keys
    that are not BeamConfig fields (tolerates schema drift)."""
    valid = {f.name for f in fields(BeamConfig)}
    return BeamConfig(**{k: v for k, v in d.items() if k in valid})


def load_project(path: str | Path) -> ProjectConfig:
    """Load a ``.lgproj`` file into a :class:`ProjectConfig`.

    Raises
    ------
    ValueError
        If the file is not a recognised linac_gen project file.
    """
    p = Path(path)
    data = json.loads(p.read_text())
    if data.get("__kind__") != "linac_gen_project":
        raise ValueError(
            f"{p} is not a linac_gen project file "
            f"(missing '__kind__': 'linac_gen_project')"
        )
    lat = str(data.get("lattice_path", "") or "")
    # Resolve a relative lattice path robustly: PROJECT-DIR FIRST — the
    # GUI now saves paths relative to the .lgproj so projects relocate —
    # then the current directory (legacy repo-root-relative projects);
    # fall back to project-dir-joined so the path is at least absolute.
    if lat and not Path(lat).is_absolute():
        for base in (p.parent, Path.cwd()):
            cand = base / lat
            if cand.is_file():
                lat = str(cand.resolve())
                break
        else:
            lat = str((p.parent / lat).resolve())
    beam_dict = dict(data.get("beam", {}))
    # Same treatment for the beam's distribution file — runtime consumes
    # the path verbatim, so a relative entry must be absolutized here.
    df = beam_dict.get("distribution_file")
    if df and not Path(str(df)).is_absolute():
        for base in (p.parent, Path.cwd()):
            cand = base / str(df)
            if cand.is_file():
                beam_dict["distribution_file"] = str(cand.resolve())
                break
        else:
            beam_dict["distribution_file"] = str((p.parent / str(df)).resolve())
    return ProjectConfig(
        lattice_path=lat,
        beam=_beam_from_dict(beam_dict),
        convergence=dict(data.get("convergence", {})),
        source_path=str(p),
    )
