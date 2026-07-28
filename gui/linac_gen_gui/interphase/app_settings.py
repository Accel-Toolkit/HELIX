"""Shared QSettings factory — the sandbox seam for persisted GUI state.

Every QSettings the GUI creates goes through :func:`make_settings`.
When ``HELIX_QSETTINGS_DIR`` is set (the pytest conftests do this),
settings live in throwaway INI files under that directory instead of
the platform-native store, so test runs can exercise the real
save/load paths without polluting the developer's actual recent
projects, window geometry, or last-used paths.

The env-var seam exists because the documented alternative does not
work: on macOS, ``QSettings(org, app)`` resolves to NativeFormat
(CFPreferences) regardless of ``QSettings.setDefaultFormat`` — verified
empirically — and ``setPath`` has no effect on the native format.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import QSettings


def make_settings(org: str, app: str) -> QSettings:
    root = os.environ.get("HELIX_QSETTINGS_DIR")
    if root:
        return QSettings(os.path.join(root, f"{org}_{app}.ini"),
                         QSettings.Format.IniFormat)
    return QSettings(org, app)
