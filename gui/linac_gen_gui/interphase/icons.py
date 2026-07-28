"""Lucide-style SVG icon set, cached as QIcon.

Every icon is defined by an SVG path string on a 24×24 viewbox with
round-linecap / round-linejoin.  ``icon(name, size, color)`` returns a
QIcon rasterised at the requested pixel size; results are cached.
"""
from __future__ import annotations

from functools import lru_cache
from PyQt6.QtCore import Qt, QByteArray, QSize
from PyQt6.QtGui import QIcon, QPixmap, QPainter
from PyQt6.QtSvg import QSvgRenderer

from linac_gen_gui.interphase import theme


# --- Icon path library (ported from icons.jsx) ------------------------------
# Each value is the SVG body inside the <svg> element, without the wrapping
# element.  Strokes use currentColor so we can re-tint cheaply.
_ICONS: dict[str, str] = {
    # Generic
    "plus":       '<path d="M12 5v14M5 12h14"/>',
    "minus":      '<path d="M5 12h14"/>',
    "x":          '<path d="M18 6 6 18M6 6l12 12"/>',
    "check":      '<path d="M20 6 9 17l-5-5"/>',
    "chev_r":     '<path d="M9 18l6-6-6-6"/>',
    "chev_d":     '<path d="M6 9l6 6 6-6"/>',
    "chev_u":     '<path d="M18 15l-6-6-6 6"/>',
    "chev_l":     '<path d="M15 18l-6-6 6-6"/>',
    "dots_v":     '<circle cx="12" cy="5" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="12" cy="19" r="1"/>',
    "dots_h":     '<circle cx="5" cy="12" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/>',
    "search":     '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
    "filter":     '<path d="M3 5h18M6 12h12M10 19h4"/>',
    "settings":   '<circle cx="12" cy="12" r="3"/>'
                  '<path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1'
                  'a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1'
                  'a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1'
                  'a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1'
                  'a1.7 1.7 0 0 0 1.5-1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1'
                  'a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1'
                  'a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1'
                  'a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1'
                  'a1.7 1.7 0 0 0-1.5 1z"/>',
    "save":       '<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/>'
                  '<path d="M17 21v-8H7v8M7 3v5h8"/>',
    "folder":     '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5'
                  'a2 2 0 0 1-2-2V7z"/>',
    "file":       '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
                  '<path d="M14 2v6h6"/>',
    "refresh":    '<path d="M21 12a9 9 0 1 1-3.3-6.95M21 3v6h-6"/>',
    # Run controls
    "play":       '<polygon points="5 3 19 12 5 21 5 3"/>',
    "pause":      '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>',
    "stop":       '<rect x="5" y="5" width="14" height="14" rx="1"/>',
    "step":       '<path d="M5 4v16l10-8z"/><path d="M19 5v14"/>',
    "rewind":     '<polygon points="11 19 2 12 11 5 11 19"/>'
                  '<polygon points="22 19 13 12 22 5 22 19"/>',
    # Physics / domain
    "atom":       '<circle cx="12" cy="12" r="1"/>'
                  '<path d="M20.2 20.2c2.04-2.03.02-7.36-4.5-11.9-4.54-4.52-9.87-6.54-11.9-4.5'
                  '-2.04 2.03-.02 7.36 4.5 11.9 4.54 4.52 9.87 6.54 11.9 4.5z"/>'
                  '<path d="M15.7 15.7c4.52-4.54 6.54-9.87 4.5-11.9-2.03-2.04-7.36-.02-11.9 4.5'
                  '-4.52 4.54-6.54 9.87-4.5 11.9 2.03 2.04 7.36.02 11.9-4.5z"/>',
    "beam":       '<path d="M3 12h18M3 8h18M3 16h18"/>',
    "wave":       '<path d="M3 12c3-6 6 6 9 0s6-6 9 0"/>',
    "box_3d":     '<path d="M21 16.5V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4'
                  'A2 2 0 0 0 3 8v8.5a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4'
                  'A2 2 0 0 0 21 16.5z"/>'
                  '<path d="M3.3 7 12 12l8.7-5M12 22V12"/>',
    "grid":       '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/>'
                  '<rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/>',
    "chart":      '<path d="M3 3v18h18"/><path d="M7 14l4-4 4 4 5-5"/>',
    "scatter":    '<path d="M3 3v18h18"/>'
                  '<circle cx="8" cy="16" r="1"/><circle cx="11" cy="11" r="1"/>'
                  '<circle cx="14" cy="14" r="1"/><circle cx="17" cy="8" r="1"/>'
                  '<circle cx="10" cy="7" r="1"/><circle cx="6" cy="12" r="1"/>',
    "heatmap":    '<rect x="3" y="3" width="18" height="18" rx="1"/>'
                  '<path d="M3 9h18M3 15h18M9 3v18M15 3v18"/>',
    "sliders":    '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>',
    "target":     '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/>'
                  '<circle cx="12" cy="12" r="1"/>',
    "zap":        '<path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/>',
    "cpu":        '<rect x="4" y="4" width="16" height="16" rx="2"/>'
                  '<rect x="9" y="9" width="6" height="6"/>'
                  '<path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 15h3M1 9h3M1 15h3"/>',
    "terminal":   '<path d="M4 17l6-6-6-6M12 19h8"/>',
    "alert":      '<path d="M10.3 3.86 1.82 18a2 2 0 0 0 1.7 3h16.96a2 2 0 0 0 1.7-3L13.7 3.86'
                  'a2 2 0 0 0-3.4 0z"/>'
                  '<path d="M12 9v4M12 17h.01"/>',
    "info":       '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
    "check_circle":'<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="M22 4 12 14.01l-3-3"/>',
    "xcircle":    '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6M9 9l6 6"/>',
    "eye":        '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8S1 12 1 12z"/>'
                  '<circle cx="12" cy="12" r="3"/>',
    "users":      '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>'
                  '<circle cx="9" cy="7" r="4"/>'
                  '<path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"/>',
    "zoom_in":    '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35M11 8v6M8 11h6"/>',
    "zoom_out":   '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35M8 11h6"/>',
    "expand":     '<path d="M3 3h7v7M14 21h7v-7M3 21h7v-7M14 3h7v7"/>',
    "maximize":   '<path d="M8 3H5a2 2 0 0 0-2 2v3M21 8V5a2 2 0 0 0-2-2h-3M3 16v3'
                  'a2 2 0 0 0 2 2h3M16 21h3a2 2 0 0 0 2-2v-3"/>',
    # Workspace icons
    "lattice":    '<path d="M3 7h18M3 12h18M3 17h18"/>'
                  '<circle cx="7" cy="7" r="1.5"/><circle cx="12" cy="12" r="1.5"/>'
                  '<circle cx="17" cy="17" r="1.5"/>',
    "ring":       '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="3"/>',
    "report":     '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
                  '<path d="M14 2v6h6M8 13h8M8 17h8M8 9h2"/>',
    "queue":      '<rect x="3" y="4" width="18" height="3"/><rect x="3" y="10" width="18" height="3"/>'
                  '<rect x="3" y="16" width="18" height="3"/>',
    "plugin":     '<path d="M14 7h6a1 1 0 0 1 1 1v6a4 4 0 0 1-4 4h-2a1 1 0 0 0-1 1v2'
                  'a1 1 0 0 1-1 1h-3a1 1 0 0 1-1-1v-2a1 1 0 0 0-1-1H6a4 4 0 0 1-4-4V8'
                  'a1 1 0 0 1 1-1h5"/>'
                  '<path d="M8 3v4M14 3v4M8 11h0M14 11h0"/>',
    "magnet":     '<path d="M6 3v8a6 6 0 1 0 12 0V3M6 3h4M14 3h4M6 11h4M14 11h4"/>',
    "beaker":     '<path d="M9 2v7L4.5 19a1 1 0 0 0 .87 1.5h13.26A1 1 0 0 0 19.5 19L15 9V2'
                  'M8 2h8M7 14h10"/>',
    "gauge":      '<path d="M12 14l4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    "help":       '<circle cx="12" cy="12" r="10"/>'
                  '<path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>'
                  '<path d="M12 17h.01"/>',
    "palette":    '<circle cx="13.5" cy="6.5" r=".5"/>'
                  '<circle cx="17.5" cy="10.5" r=".5"/>'
                  '<circle cx="8.5" cy="7.5" r=".5"/>'
                  '<circle cx="6.5" cy="12.5" r=".5"/>'
                  '<path d="M12 2a10 10 0 1 0 10 10c0-2-3-2-3-4a3 3 0 0 0-3-3h-1'
                  'a5 5 0 0 1-3-9z"/>',
    "collision":  '<circle cx="8" cy="12" r="3"/><circle cx="16" cy="12" r="3"/>'
                  '<path d="M11 12h2M4 8l2 2M4 16l2-2M20 8l-2 2M20 16l-2-2"/>',
    "layers":     '<path d="m12 2 10 5-10 5L2 7z"/>'
                  '<path d="m2 17 10 5 10-5M2 12l10 5 10-5"/>',
    "star":       '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 '
                  '5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
    "git":        '<circle cx="6" cy="6" r="2"/><circle cx="18" cy="18" r="2"/>'
                  '<path d="M6 8v8a4 4 0 0 0 4 4h6"/>',
}


def _svg(body: str, color: str, stroke: float = 1.6) -> bytes:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
    ).encode("utf-8")


@lru_cache(maxsize=512)
def icon(name: str, size: int = 16, color: str | None = None) -> QIcon:
    """Return a QIcon of the requested lucide icon, tinted to ``color``.

    Unknown names return an empty QIcon rather than raising — helps the
    app stay alive while icon names are being wired up.
    """
    body = _ICONS.get(name)
    if body is None:
        return QIcon()
    color = color or theme.TEXT_1
    svg = _svg(body, color)

    renderer = QSvgRenderer(QByteArray(svg))
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    renderer.render(painter)
    painter.end()
    return QIcon(pix)


def available() -> list[str]:
    return sorted(_ICONS)
