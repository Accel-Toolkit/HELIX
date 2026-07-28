"""Runtime registry for trained surrogate elements.

TRACKING LOOKUP IS BY ELEMENT NAME **GUARDED BY THE ELEMENT IN HAND**:
the envelope solver, the MP tracker and the backtracker all call
:func:`get_for_element`, which resolves by name (last registration
wins) and then verifies the registered surrogate actually wraps the
element being tracked — identity first, structural fingerprint (class,
length, field file, drive parameters) as fallback for a reloaded copy
of the same deck.  A mismatch (e.g. two decks' auto-named ``FMAP_001``
colliding in one process) skips the surrogate with a once-per-name
warning and the native element tracks.  The ``(lattice_hash,
element_key)`` key exists alongside for on-disk weight paths and GUI
presence checks only.  If an engaged surrogate is admissible (input in
training scope) it is used; otherwise the wrapped element tracks
normally.

Module-level singleton.  Use :func:`clear` in tests.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from linac_gen.surrogates.base import SurrogateFieldMap


_REGISTRY: dict[tuple[str, str], SurrogateFieldMap] = {}
# Name-only index for fast envelope-hook lookup (M3).  Without a
# per-process "current lattice" context, the envelope solver looks up
# surrogates by element.name alone; the (lattice_hash, name) key in
# _REGISTRY stays for safe disambiguation when the user registers
# multiple surrogates for the same element across different lattices.
_BY_NAME: dict[str, SurrogateFieldMap] = {}

# Master MP-engagement flag (M7).  When False (default), the MP
# tracker's surrogate hook is a no-op -- registered surrogates only
# act in envelope-mode runs, matching M1-M6 behaviour.  When True,
# the hook routes per-element track_rk4 calls through the surrogate's
# hybrid linear-anchor + RK4-residual path.  Toggled from the GUI
# "Multi-particle surrogates" section.
_MP_ENGAGED: bool = False

# Second opt-in (M7-followup): the actual linear-matrix fast path
# that bypasses wrapped.track_rk4 in the inner loop.  Default False
# so MP-engagement alone keeps the bit-identical safe delegate.
# When True (and MP-engagement is also True), `SurrogateFieldMap.
# track_rk4` switches to the analytic ref-advance + batched
# M_slice @ particles implementation.  Targets ~10-15x speedup at
# the cost of linear-matrix accuracy (depends on training quality).
_FAST_PATH_ENABLED: bool = False


# ---------------------------------------------------------------------------
def hash_lattice_file(path: str | Path) -> str:
    """SHA256 hex digest of a lattice file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
def register(surrogate: SurrogateFieldMap,
             lattice_hash: str | None = None,
             element_key: str | None = None) -> None:
    """Register ``surrogate`` by ``(lattice_hash, element_key)``.

    Defaults: read from ``surrogate.metadata``.
    """
    from linac_gen.elements.superposed_field_map import SuperposedFieldMap
    if isinstance(getattr(surrogate, "_wrapped", None), SuperposedFieldMap):
        raise ValueError(
            "surrogates cannot wrap a SUPERPOSE cluster container "
            "(train/register the individual maps' physics instead)"
        )
    lh = lattice_hash if lattice_hash is not None else surrogate.metadata.lattice_hash
    ek = element_key if element_key is not None else surrogate.metadata.element_key
    # Cross-lattice collision: tracking lookup is name-scoped and
    # last-write-wins, so a surrogate trained for lattice A's 'CAV1'
    # would silently serve lattice B's 'CAV1'.  Full hash-scoped lookup
    # is a documented architectural deferral — until then, be loud.
    prior = _BY_NAME.get(ek)
    if prior is not None:
        prior_lh = getattr(getattr(prior, "metadata", None),
                           "lattice_hash", None)
        # reversed(): dict order is insertion order, and the MOST
        # RECENT registration of the prior object is what name-scoped
        # tracking is actually serving — cite that hash, not the first.
        for (reg_lh, reg_ek) in reversed(_REGISTRY):
            if reg_ek == ek and _REGISTRY[(reg_lh, reg_ek)] is prior:
                prior_lh = reg_lh
                break
        if prior_lh is not None and prior_lh != lh:
            import warnings
            warnings.warn(
                f"surrogate '{ek}' is already registered for a "
                f"DIFFERENT lattice (hash {str(prior_lh)[:12]}… vs "
                f"{str(lh)[:12]}…) — tracking lookup is name-scoped "
                f"and last-write-wins, so this registration replaces "
                f"the other lattice's surrogate for every '{ek}'.",
                stacklevel=2)
    _REGISTRY[(lh, ek)] = surrogate
    _BY_NAME[ek] = surrogate
    _GENERATION[0] += 1              # invalidate pinned decisions
    _MISMATCH_WARNED.discard(ek)     # a fresh mismatch re-warns


def unregister(lattice_hash: str, element_key: str) -> None:
    _REGISTRY.pop((lattice_hash, element_key), None)
    # If no other registration shares this name, drop it from the
    # name-only index too.
    if not any(name == element_key for (_, name) in _REGISTRY):
        _BY_NAME.pop(element_key, None)
    _GENERATION[0] += 1
    _MISMATCH_WARNED.discard(element_key)


def clear() -> None:
    """Clear all registered surrogates (use in tests)."""
    _REGISTRY.clear()
    _BY_NAME.clear()
    _MISMATCH_WARNED.clear()
    _GENERATION[0] += 1


# ---------------------------------------------------------------------------
def set_mp_enabled(enabled: bool) -> None:
    """Toggle the MP-mode surrogate engagement (M7)."""
    global _MP_ENGAGED
    _MP_ENGAGED = bool(enabled)


def is_mp_enabled() -> bool:
    """Return whether MP-mode surrogate engagement is currently on."""
    return _MP_ENGAGED


def set_fast_path_enabled(enabled: bool) -> None:
    """Toggle the linear-matrix fast path (M7-followup, opt-in)."""
    global _FAST_PATH_ENABLED
    _FAST_PATH_ENABLED = bool(enabled)


def is_fast_path_enabled() -> bool:
    """Return whether the experimental linear-matrix fast path is on."""
    return _FAST_PATH_ENABLED


def get(lattice_hash: str, element_key: str) -> SurrogateFieldMap | None:
    """Return the registered surrogate, or ``None``."""
    return _REGISTRY.get((lattice_hash, element_key))


def get_by_element_name(name: str) -> SurrogateFieldMap | None:
    """Lookup a registered surrogate by element name only.

    Used by the envelope-mode hook (M3).  Returns ``None`` if no
    surrogate matches.  If multiple surrogates share the same name
    across lattices, the most recently registered wins.

    NOTE: tracking seams should prefer :func:`get_for_element`, which
    additionally verifies the registered surrogate actually wraps the
    element being tracked (auto-generated names like ``FMAP_001``
    collide across decks BY CONSTRUCTION, so a bare name hit can
    dispatch the wrong lattice's physics).
    """
    return _BY_NAME.get(name)


#: element attributes that identify a field map for dispatch purposes —
#: the things that would be WRONG if another lattice's surrogate were
#: substituted.  Superset of FieldMap._cache_keys (the repo's
#: authoritative matrix-affecting list: ke, kb, phase, frequency,
#: scale, p_flag, voltage_rel, phase_offset, frequency_offset) plus
#: geometry/data identity (length, field_file, geom, aperture — the
#: surrogate element copies the WRAPPED element's aperture, so a
#: false engage would apply the wrong lattice's aperture cuts).
_FINGERPRINT_ATTRS = ("length", "field_file", "geom", "aperture",
                     "ke", "kb", "ki", "ka", "phase", "frequency",
                     "scale", "p_flag", "voltage_rel", "phase_offset",
                     "frequency_offset", "n_steps")

_MISMATCH_WARNED: set[str] = set()

#: bumped on every register/unregister/clear — engagement decisions
#: memoized on elements are valid only within one generation, so a
#: mid-run ADJUST mutation can never flip-flop engagement (the
#: decision is pinned at first lookup), while any registry change
#: invalidates all pins immediately.
_GENERATION: list[int] = [0]


def _fingerprint(elem) -> tuple:
    out = [type(elem).__name__]
    for a in _FINGERPRINT_ATTRS:
        v = getattr(elem, a, None)
        if isinstance(v, float):
            v = round(v, 9)
        out.append(v)
    return tuple(out)


def get_for_element(element) -> SurrogateFieldMap | None:
    """Name lookup guarded by the element in hand (ground truth).

    Auto-generated element names (``FMAP_001`` …) are per-deck
    counters, so two different lattices both contain an ``FMAP_001``
    by construction and the process-global registry would otherwise
    silently dispatch lattice A's learned map while tracking lattice B.

    Guard: the registered surrogate engages only if it wraps THIS
    element (identity — true in every shipped register-then-track
    flow) or an element structurally identical to it (fingerprint:
    class, length, field file, drive parameters — so re-loading the
    SAME deck still engages).  On a mismatch the surrogate is skipped,
    the native element tracks, and a warning fires once per element
    name.
    """
    name = getattr(element, "name", None)
    if name is None:
        return None
    surr = _BY_NAME.get(name)
    if surr is None:
        return None
    # Per-element pinned decision: the first lookup within a registry
    # generation decides engage/skip and the decision sticks for the
    # rest of the run — an ADJUST mutating drive parameters mid-match
    # must not silently flip engagement (discontinuous objective,
    # contaminated finite-difference Jacobians).  Any register/
    # unregister/clear bumps the generation and invalidates the pin.
    pin = getattr(element, "_surr_binding", None)
    if pin is not None and pin[0] == _GENERATION[0]:
        return pin[1]

    wrapped = getattr(surr, "_wrapped", None)
    decision = None
    if wrapped is element:
        decision = surr
    elif wrapped is None:
        if name not in _MISMATCH_WARNED:
            _MISMATCH_WARNED.add(name)
            import warnings
            warnings.warn(
                f"registered surrogate '{name}' carries no wrapped "
                f"element, so it cannot be verified against the "
                f"element being tracked — skipping it; the native "
                f"element tracks instead.",
                stacklevel=3)
    elif _fingerprint(wrapped) == _fingerprint(element):
        decision = surr
    else:
        if name not in _MISMATCH_WARNED:
            _MISMATCH_WARNED.add(name)
            import warnings
            warnings.warn(
                f"registered surrogate '{name}' wraps a DIFFERENT "
                f"element than the one being tracked (auto-generated "
                f"names collide across decks) — skipping the "
                f"surrogate for every '{name}' in this lattice; the "
                f"native element tracks instead.  Re-register the "
                f"surrogate for the current lattice to engage it.",
                stacklevel=3)
    try:
        element._surr_binding = (_GENERATION[0], decision)
    except Exception:                                       # noqa: BLE001
        pass                     # slotted/frozen element: no pinning
    return decision


def list_registered() -> list[tuple[str, str]]:
    """Snapshot of registered keys."""
    return list(_REGISTRY.keys())
