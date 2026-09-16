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
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

from linac_gen.core.config import BeamConfig

__all__ = ["ProjectConfig", "load_project", "write_project"]


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
    data = json.loads(p.read_text(encoding="utf-8"))
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


def write_project(path: str | Path, *, lattice_path: str | Path,
                  beam: BeamConfig, convergence: dict | None = None,
                  extra: dict | None = None, overwrite: bool = False) -> Path:
    """Write a ``.lgproj`` file without the GUI (the same schema the GUI's
    Save writes: ``__kind__``/``__version__``, ``lattice_path``, ``beam``,
    ``convergence``).

    ``lattice_path`` and the beam's ``distribution_file`` are stored
    relative to the project directory whenever a meaningful relative form
    exists (:func:`linac_gen.io.portable_paths.best_relpath`), so the
    project relocates with its deck; otherwise the absolute path is kept.
    ``extra`` merges further top-level keys (e.g. ``calc_dir``).

    Raises
    ------
    FileExistsError
        ``path`` exists and ``overwrite`` is false.
    """
    from dataclasses import asdict
    from linac_gen.io.portable_paths import best_relpath

    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"{p} exists (pass overwrite=True to replace it)")
    # abspath, not resolve(): best_relpath abspath's the target, and a
    # symlinked directory (macOS /tmp -> /private/tmp) must relativise
    # the same way on both sides — the GUI's _collect_project_dict does
    # the same.
    anchor = os.path.abspath(p.parent)

    def _portable(q):
        if not q:
            return q
        rel, _ok = best_relpath(str(q), anchor)   # absolute when not portable
        return rel

    data: dict = {
        "__kind__": "linac_gen_project",
        "__version__": 1,
        "lattice_path": _portable(str(lattice_path)),
    }
    b = asdict(beam)
    if b.get("distribution_file"):
        b["distribution_file"] = _portable(b["distribution_file"])
    data["beam"] = b
    data["convergence"] = dict(convergence or {})
    if extra:
        data.update(extra)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p
