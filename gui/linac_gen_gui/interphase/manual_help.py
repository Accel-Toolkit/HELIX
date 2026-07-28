"""Per-element manual lookup.

Resolves an element type to its chapter in the HELIX user manual and
opens it in the OS default browser.  Tries a local built copy first
(``<repo_root>/site/...`` in dev, the PyInstaller bundle in the .exe);
falls back to a configured online URL if nothing is on disk.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

# Element class name → manual chapter (relative to ``site/`` root).
# Mirrors dialogs/manual_popup._CHAPTER_MD_BY_TYPE; keep in sync.
_SET_ADJUST_HTML = "07_matching/02_set_adjust.html"
_CHAPTER_BY_TYPE: dict[str, str] = {
    # ---- Physical elements ----------------------------------------
    "Drift":          "03_elements/01_drift.html",
    "Quadrupole":     "03_elements/02_quadrupole.html",
    "Sextupole":      "03_elements/11_multipole.html",
    "Octupole":       "03_elements/11_multipole.html",
    "Multipole":      "03_elements/11_multipole.html",
    "Solenoid":       "03_elements/03_solenoid.html",
    "Dipole":         "03_elements/04_dipole.html",
    "Edge":           "03_elements/05_edge.html",
    "RFGap":          "03_elements/06_rfgap.html",
    "FieldMap":       "03_elements/07_fieldmap.html",
    "FieldMap3D":     "03_elements/08_fieldmap3d.html",
    "RfqCell":        "03_elements/09_rfqcell.html",
    "VaneRFQ":        "03_elements/10_vanerfq.html",
    "Aperture":       "03_elements/12_aperture.html",
    "Marker":         "03_elements/13_marker.html",
    "Steerer":        "03_elements/14_steerer.html",
    "Foil":           "03_elements/15_foil.html",
    "ThinLens":       "03_elements/00_overview.html",
    "SpaceChargeComp": "03_elements/00_overview.html",
    # ---- LatticeCommand subclasses --------------------------------
    "SetSyncPhase":         _SET_ADJUST_HTML,
    "SetBeamPhaseError":    _SET_ADJUST_HTML,
    "SetBeamE0P0":          _SET_ADJUST_HTML,
    "SetBeamEnergy":        _SET_ADJUST_HTML,
    "SetGaussianCutOff":    _SET_ADJUST_HTML,
    "SetTwiss":             _SET_ADJUST_HTML,
    "SetPosition":          _SET_ADJUST_HTML,
    "SetAchromat":          _SET_ADJUST_HTML,
    "SetSize":              _SET_ADJUST_HTML,
    "SetSizeMax":           _SET_ADJUST_HTML,
    "SetSizeMin":           _SET_ADJUST_HTML,
    "SetBeamPhaseAdv":      _SET_ADJUST_HTML,
    "SetSeparation":        _SET_ADJUST_HTML,
    "SetAdv":               _SET_ADJUST_HTML,
    "MinEmitGrowth":        _SET_ADJUST_HTML,
    "MinEmit4DGrowth":      _SET_ADJUST_HTML,
    "MinTransmission":      _SET_ADJUST_HTML,
    "SetKeOutMin":          _SET_ADJUST_HTML,
    "Adjust":               _SET_ADJUST_HTML,
    "AdjustSteerer":        _SET_ADJUST_HTML,
    "AdjustSteererBx":      _SET_ADJUST_HTML,
    "AdjustSteererBy":      _SET_ADJUST_HTML,
    "AdjustBeamTwiss":      _SET_ADJUST_HTML,
    "AdjustBeamCentroid":   _SET_ADJUST_HTML,
    "AdjustBeamEmit":       _SET_ADJUST_HTML,
    "AdjustBeamCurrent":    _SET_ADJUST_HTML,
}

# Set this to the deployed Pages URL if you ship the GUI without a
# bundled manual.  Empty string disables the online fallback.
_ONLINE_BASE: str = ""


def chapter_for(type_name: str) -> str | None:
    """Return the relative path under ``site/`` for *type_name*, or
    ``None`` if no chapter is registered."""
    return _CHAPTER_BY_TYPE.get(type_name)


def _candidate_roots() -> list[Path]:
    """Possible filesystem locations of the built manual (in priority
    order: PyInstaller bundle → repo-relative ``site/`` → cwd)."""
    roots: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass) / "site")
    here = Path(__file__).resolve()
    # walk up looking for a ``site/`` sibling of ``docs/`` or ``gui/``
    for parent in here.parents:
        cand = parent / "site"
        if cand.is_dir():
            roots.append(cand)
            break
    roots.append(Path.cwd() / "site")
    return roots


def _resolve_local(rel: str) -> Path | None:
    for root in _candidate_roots():
        p = root / rel
        if p.is_file():
            return p
    return None


def _is_wsl() -> bool:
    if sys.platform != "linux":
        return False
    try:
        with open("/proc/sys/kernel/osrelease", "r", encoding="utf-8") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def _wsl_to_windows_path(p: Path) -> str | None:
    """``/mnt/c/foo/bar`` → ``C:\\foo\\bar`` if applicable, else ``None``."""
    s = str(p)
    if not s.startswith("/mnt/"):
        return None
    parts = s.split("/")  # ['', 'mnt', 'c', 'foo', 'bar', ...]
    if len(parts) < 4 or len(parts[2]) != 1:
        return None
    drive = parts[2].upper() + ":"
    rest = "\\".join(parts[3:])
    return drive + "\\" + rest if rest else drive + "\\"


def _open_path(p: Path) -> bool:
    """Open *p* in the OS's default app.  Returns ``True`` on success.

    Tries — in order — cmd.exe (WSL → Windows browser), the platform's
    native ``start`` / ``xdg-open`` / ``open``, then ``webbrowser``.
    """
    # WSL: hand off to Windows's default file association via cmd.exe.
    if _is_wsl():
        win = _wsl_to_windows_path(p)
        if win and shutil.which("cmd.exe"):
            try:
                # ``start "" "<path>"`` — empty title arg avoids the path
                # being interpreted as the window title.
                subprocess.Popen(
                    ["cmd.exe", "/c", "start", "", win],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            except OSError:
                pass

    # macOS
    if sys.platform == "darwin" and shutil.which("open"):
        try:
            subprocess.Popen(["open", str(p)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except OSError:
            pass

    # Native Windows
    if sys.platform.startswith("win"):
        try:
            os.startfile(str(p))  # type: ignore[attr-defined]
            return True
        except (OSError, AttributeError):
            pass

    # Linux desktop
    if shutil.which("xdg-open"):
        try:
            subprocess.Popen(["xdg-open", str(p)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except OSError:
            pass

    # Last-resort: Python webbrowser, file:// URL.
    try:
        return webbrowser.open(p.as_uri())
    except Exception:
        return False


def open_for_type(type_name: str) -> tuple[bool, str]:
    """Open the manual chapter for *type_name*.

    Returns ``(ok, message)`` so the caller can route the result into
    the GUI's status bar without owning any presentation logic.
    """
    rel = chapter_for(type_name)
    if rel is None:
        return False, f"no manual chapter registered for '{type_name}'"
    local = _resolve_local(rel)
    if local is not None:
        if _open_path(local):
            return True, f"opened {local.name}"
        return False, f"failed to open {local}"
    if _ONLINE_BASE:
        url = _ONLINE_BASE.rstrip("/") + "/" + rel
        try:
            if webbrowser.open(url):
                return True, f"opened {url} (online)"
        except Exception:
            pass
        return False, f"failed to open online URL {url}"
    return False, (
        f"manual not found locally — build it via "
        f"`mkdocs build` in docs/ to enable per-element help"
    )


def open_for_element(element) -> tuple[bool, str]:
    """Convenience wrapper: open the chapter for the element's class."""
    return open_for_type(type(element).__name__)
