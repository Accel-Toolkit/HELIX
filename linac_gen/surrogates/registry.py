"""Runtime registry for trained surrogate elements.

TRACKING LOOKUP IS BY ELEMENT NAME **GUARDED BY THE ELEMENT IN HAND**:
the envelope solver, the MP tracker and the backtracker all call
:func:`get_for_element`, which resolves by name (last registration
wins) and then verifies the registered surrogate actually wraps the
element being tracked — identity first, then its structure (class,
length, field file, geometry, aperture, steps, p_flag) for a copy of it:
a reloaded deck, an error or failure study's copy.  How the element is
set (its inputs, its errors, its other settings) is checked when the
weights are asked (``SurrogateFieldMap._refusal``).  A mismatch (e.g.
two decks' auto-named ``FMAP_001`` colliding in one process) skips the
surrogate with a once-per-name warning and the native element tracks.  The ``(lattice_hash,
element_key)`` key exists alongside for on-disk weight paths and GUI
presence checks only.  If an engaged surrogate is admissible (input in
training scope) it is used; otherwise the wrapped element tracks
normally.

Module-level singleton.  Use :func:`clear` in tests.
"""
from __future__ import annotations

import contextlib
import hashlib
import threading
from pathlib import Path

from linac_gen.surrogates.base import SurrogateFieldMap, structure_of


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
# the hook routes per-element track_rk4 calls through the surrogate
# (a safe delegate unless _FAST_PATH_ENABLED below is also True).
# Toggled from the GUI "Multi-particle surrogates" section.
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


_LOCAL = threading.local()


@contextlib.contextmanager
def suspended():
    """Inside it, in this thread, no registered surrogate engages: :func:`get_for_element`
    gives None and pins nothing, so the native elements track and the registrations stay as
    they were.  For a study that runs native (``study_check.study_surrogates``)."""
    before = getattr(_LOCAL, "suspended", False)
    _LOCAL.suspended = True
    try:
        yield
    finally:
        _LOCAL.suspended = before


def is_suspended() -> bool:
    """Whether :func:`suspended` is in force in this thread."""
    return bool(getattr(_LOCAL, "suspended", False))


@contextlib.contextmanager
def native_for(elements):
    """Inside it, each of ``elements`` runs native: its decision is pinned to none for the
    registry generation in force; after it, every one is pinned as it was.  For a match whose
    variables move a setting of a surrogated element that its weights do not take as an input:
    deciding once for the whole run keeps the objective one model (a change of such a setting
    has the weights refuse, so they would answer the first evaluation and not the next)."""
    saved = []
    for element in elements:
        saved.append((element, getattr(element, "_surr_binding", None)))
        with contextlib.suppress(Exception):               # slotted/frozen: no pinning
            element._surr_binding = _Pin(_GENERATION[0], None, id(element))
    try:
        yield
    finally:
        for element, pin in saved:
            with contextlib.suppress(Exception):
                element._surr_binding = pin


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


_MISMATCH_WARNED: set[str] = set()


def _no_pin():
    return None


class _Pin:
    """An engagement decision pinned on an element: valid in the registry generation it
    was made in, and for that very element.  A copy carries none (deepcopy, as error and
    failure studies copy the lattice, and pickle give None; a shallow copy shares it,
    and the owner check sends that copy back to a fresh decision): the copy decides for
    itself.  Within a run, an ADJUST on the element keeps the decision on which weights
    stand for it; how the element is set is checked when they are asked."""
    __slots__ = ("generation", "decision", "owner")

    def __init__(self, generation: int, decision, owner: int) -> None:
        self.generation, self.decision, self.owner = generation, decision, owner

    def __getitem__(self, i):
        """As the (generation, decision) tuple the pin used to be."""
        return (self.generation, self.decision, self.owner)[i]

    def __deepcopy__(self, memo):
        return None

    def __reduce__(self):
        return (_no_pin, ())

#: bumped on every register/unregister/clear — lookup decisions
#: memoized on elements are valid only within one generation, so a
#: mid-run ADJUST never changes which weights stand for an element (the
#: decision is pinned at first lookup; how the element is set, and its
#: structure, are checked when the weights are asked), while any registry
#: change invalidates all pins immediately.
_GENERATION: list[int] = [0]


def _fingerprint(elem) -> tuple:
    """What identifies a field map for dispatch (``base.structure_of``: its class and
    ``STRUCTURE``): what it is (the surrogate element copies the WRAPPED element's aperture,
    so a false engage would apply the wrong lattice's aperture cuts), not how it is set.
    Its settings are checked when the weights are asked: its inputs live, its error knobs
    folded, its misalignments around it, anything else equal to what the weights hold
    (``SurrogateFieldMap._settings_reason``), and its structure again (``_structure_reason``)."""
    return structure_of(elem)


def get_for_element(element) -> SurrogateFieldMap | None:
    """Name lookup guarded by the element in hand (ground truth).

    Auto-generated element names (``FMAP_001`` …) are per-deck
    counters, so two different lattices both contain an ``FMAP_001``
    by construction and the process-global registry would otherwise
    silently dispatch lattice A's learned map while tracking lattice B.

    Guard: the registered surrogate engages only if it wraps THIS
    element (identity — true in every shipped register-then-track
    flow) or an element structurally identical to it (fingerprint:
    ``base.structure_of``, class and ``STRUCTURE`` — so re-loading the
    SAME deck still engages; how it is set is checked when the weights
    are asked).  On a mismatch the surrogate is skipped,
    the native element tracks, and a warning fires once per element
    name.
    """
    if is_suspended():
        return None
    name = getattr(element, "name", None)
    if name is None:
        return None
    surr = _BY_NAME.get(name)
    if surr is None:
        return None
    # Per-element pinned decision: the first lookup within a registry
    # generation decides which weights stand for the element and the
    # decision sticks for the rest of the run.  The weights check how the
    # element is set, and its structure, at each query (a match pins a
    # held-setting variable's element native for the whole run,
    # ``native_for``).  Any register/unregister/clear bumps the
    # generation and invalidates the pin.
    pin = getattr(element, "_surr_binding", None)
    if (isinstance(pin, _Pin) and pin.generation == _GENERATION[0]
            and pin.owner == id(element)):
        return pin.decision

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
        # the same element in another object (the deck reloaded, a study's copy): a view
        # of the surrogate that reads ITS parameters and keeps its own traversal state
        decision = surr.bound_to(element)
    else:
        if name not in _MISMATCH_WARNED:
            _MISMATCH_WARNED.add(name)
            import warnings
            warnings.warn(
                f"registered surrogate '{name}' wraps a DIFFERENT "
                f"element than the one being tracked (another field "
                f"file, length, geometry, aperture, steps or p_flag: "
                f"auto-generated names collide across decks) — skipping "
                f"the surrogate for every such '{name}'; the native "
                f"element tracks instead.  Re-register the surrogate for "
                f"the current lattice to engage it.",
                stacklevel=3)
    try:
        element._surr_binding = _Pin(_GENERATION[0], decision, id(element))
    except Exception:                                       # noqa: BLE001
        pass                     # slotted/frozen element: no pinning
    return decision


def list_registered() -> list[tuple[str, str]]:
    """Snapshot of registered keys."""
    return list(_REGISTRY.keys())
