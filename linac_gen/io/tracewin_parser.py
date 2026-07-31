"""TraceWin .dat lattice file parser.

Reads a TraceWin-format beamline file and returns a :class:`Lattice` populated
with the corresponding element objects.

Supported cards
---------------
TITLE, FREQ, DRIFT, QUAD, SOLENOID, GAP, FIELD_MAP, THIN_STEERING,
BEND, APERTURE, MARKER, DIAG_SIZE, DIAG_EMIT, DIAG_PHASE,
SPACE_CHARGE_COMP, END

All lengths are in mm, matching the internal Linac_Gen convention.
"""

import os
import shlex
import warnings
from pathlib import Path

from linac_gen.core.lattice import Lattice
from linac_gen.elements.aperture import Aperture
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.edge import Edge
from linac_gen.elements.field_map import FieldMap
from linac_gen.elements.lattice_commands import COMMAND_CLASSES, Freq
from linac_gen.elements.marker import Marker
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.elements.solenoid import Solenoid
from linac_gen.elements.space_charge_comp import SpaceChargeComp
from linac_gen.elements.steerer import Steerer
from linac_gen.io.tracewin_syntax import SCHEMA, parse_positionals

# Default RF frequency used when a GAP or FIELD_MAP is encountered before
# any FREQ card.
_DEFAULT_FREQ_MHZ = 352.21


def parse_tracewin(filepath, strict=False, base_dir=None):
    """Parse a TraceWin .dat lattice file.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.dat`` file.
    strict : bool, optional
        If *True*, raise :exc:`ValueError` on unknown or malformed cards
        AND on every physics downgrade: superposition/beam-mutating
        cards HELIX cannot model, coupled/dynamic error groups sampled
        as independent/static, ERROR_* alignment components no tracker
        applies, negative (second-order) field-map geometry codes, and
        missing/rejected field-map files (which permissive mode DROPS
        from the lattice).  A strict parse never returns a lattice
        whose physics silently differs from the deck.
        If *False* (default), each downgrade is recorded in
        ``metadata["warnings"]`` (surfaced by the GUI status bar and
        echoed to stderr by the CLI loaders) and parsing continues.
    base_dir : str or Path, optional
        Directory used to resolve ``FIELD_MAP`` file references.  Defaults
        to the directory containing *filepath*.

    Returns
    -------
    lattice : Lattice
        Lattice populated with parsed elements.
    metadata : dict
        Dictionary with keys:

        ``"title"``   – string from the ``TITLE`` card (empty if absent).
        ``"warnings"`` – list of warning strings for non-fatal parse issues.
    """
    filepath = str(filepath)
    if base_dir is None:
        base_dir = str(Path(filepath).parent)
    else:
        base_dir = str(base_dir)

    lattice = Lattice()
    metadata = {"warnings": [], "title": ""}

    # ERROR_* state machine — tracks pending ERROR_QUAD/CAV/BEND directives
    # and assigns them to subsequent matching elements.  See
    # :mod:`linac_gen.io.tracewin_error_parsing`.
    from linac_gen.io.tracewin_error_parsing import TraceWinErrorState
    error_state = TraceWinErrorState()

    # ── Physics-downgrade reporting ────────────────────────────────────
    # A card whose physical meaning HELIX cannot honour is a DOWNGRADE:
    # permissive mode records it in metadata["warnings"] (surfaced by the
    # GUI status bar and echoed by the CLI), strict mode refuses — a
    # strict parse must never return a lattice whose physics silently
    # differs from the deck's intent.
    def _downgrade(msg: str) -> None:
        if strict:
            raise ValueError(msg)
        metadata["warnings"].append(msg)

    # ERROR_* directive caveats (inert alignment DOF, r=0 approximations,
    # ignored ratios) are emitted by tracewin_error_parsing via
    # warnings.warn — historically stderr-only, invisible to the GUI.
    # Capture them into the downgrade channel as well, re-emitting so API
    # users keep the stderr signal.
    def _error_directive(line_num, fn, nums):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fn(nums)
        for w in caught:
            warnings.warn(str(w.message), w.category, stacklevel=3)
        for w in caught:
            _downgrade(f"Line {line_num}: {w.message}")

    # Low-stakes recognised no-ops (TraceWin fit hints / RFQ vane-setup
    # cards): counted and reported once per deck instead of per card.
    noop_hint_counts: dict = {}

    # ── SUPERPOSE_MAP cluster assembly ─────────────────────────────────
    # A cluster = consecutive SUPERPOSE_MAP+FIELD_MAP pairs, closed by
    # the next ordinary element (TraceWin semantics).  Cards may appear
    # in any order/position within the cluster.
    open_cluster: list = []       # [(z0_mm, element)] in card order
    pending_superpose: list = []  # [z0] of one unbound SUPERPOSE_MAP
    # SHIFT_IN_FIELD_MAP: diagnostics placed INSIDE the span of the
    # following FIELD_MAP/cluster (TraceWin: dz > 0, several allowed).
    pending_shift_dz: list = []      # [dz] of one unbound SHIFT card
    pending_interior: list = []      # [(dz, Marker)] awaiting a map
    # The cards a SHIFT_IN_FIELD_MAP may bind to: genuine diagnostics /
    # hardware markers only.  Keyword-gated so structural marker cards
    # (LATTICE, fit hints, READ_DST …) are never captured by mistake.
    # BPM-flagged cards (BPM, DIAG_POSITION) are refused at capture —
    # orbit correction and diagnostic matching resolve BPMs ORDINALLY
    # from lattice.elements, so consuming one would silently renumber
    # every downstream BPM.
    shift_capturable = (
        "MARKER", "DIAG_SIZE", "DIAG_EMIT", "DIAG_PHASE", "DIAG_POSITION",
        "BPM", "XCOR", "YCOR", "ACCT", "COL", "RPU", "FFC", "DPI",
        "DCCT", "RWCM", "ASCN", "FASTGV", "LASERPROFILE", "MEBTABSORBER",
        "CHOPPER",
    )

    def _register_cav(elem) -> None:
        # FIELD_MAP cavities (any element with an electric channel)
        # consume ERROR_CAV_NCPL_STAT directives; static-only maps do
        # not.  Registration happens at ADD time so cluster children
        # never register individually (the container carries the
        # cluster-wide ERROR_CAV binding).
        if (hasattr(elem, "_has_electric_channel")
                and elem._has_electric_channel()):
            error_state.on_element("CAV", elem.name)

    def _take_interior(span: float, line_num) -> list:
        """Validate and consume the pending SHIFT_IN_FIELD_MAP markers
        for a map/cluster of physical *span* mm."""
        if pending_shift_dz:
            _downgrade(
                f"Line {line_num}: SHIFT_IN_FIELD_MAP not followed by a "
                "diagnostic element — the shift is ignored."
            )
            pending_shift_dz.clear()
        taken = []
        for dz, mk in pending_interior:
            if not (0.0 < dz <= span + 1e-9):
                _downgrade(
                    f"Line {line_num}: SHIFT_IN_FIELD_MAP dz={dz:g} mm "
                    f"is outside the following map span ({span:g} mm) — "
                    f"diagnostic '{mk.name}' dropped."
                )
                continue
            taken.append((dz, mk))
        pending_interior.clear()
        return taken

    def _flush_cluster(line_num) -> None:
        if pending_superpose:
            _downgrade(
                f"Line {line_num}: SUPERPOSE_MAP not followed by a "
                "FIELD_MAP card — the dangling placement is ignored."
            )
            pending_superpose.clear()
        if not open_cluster:
            return
        children = list(open_cluster)
        open_cluster.clear()
        span = max(z0 + c.length for z0, c in children)
        interior = _take_interior(span, line_num)
        if (len(children) == 1 and abs(children[0][0]) < 1e-12
                and interior):
            # Interior diagnostics need the container walk even for a
            # lone z0=0 map.
            from linac_gen.elements.superposed_field_map import (
                SuperposedFieldMap,
            )
            cont = SuperposedFieldMap(next_name("SUPERP"), children,
                                      interior_markers=interior)
            lattice.add(cont)
            _register_cav(cont)
            return
        if len(children) == 1 and abs(children[0][0]) < 1e-12:
            # The dominant "SUPERPOSE_MAP 0 as furniture" idiom: emit
            # the plain element (bit-identical physics; every existing
            # dispatch — surrogates, ADJUST, backtrack — keeps working)
            # with writer provenance.
            elem = children[0][1]
            elem.superpose_z0 = 0.0
            lattice.add(elem)
            _register_cav(elem)
            return
        from linac_gen.elements.superposed_field_map import (
            SuperposedFieldMap,
        )
        try:
            cont = SuperposedFieldMap(next_name("SUPERP"), children,
                                      interior_markers=interior)
        except ValueError as exc:
            # e.g. mixed RF frequencies (unsupported v1).  Permissive
            # mode falls back to today's end-to-end physics — loudly;
            # strict refuses (via _downgrade).
            _downgrade(
                f"Line {line_num}: SUPERPOSE cluster cannot be built "
                f"({exc}) — falling back to END-TO-END sequential maps "
                "(fields are NOT superposed; physics differs from "
                "TraceWin)."
            )
            for _z0, child in children:
                lattice.add(child)
                _register_cav(child)
            for _dz, mk in interior:
                # Best effort: restore consumed diagnostics as ordinary
                # markers so their readings aren't lost entirely.
                lattice.add(mk)
            return
        if cont._pos0_child is None:
            metadata["warnings"].append(
                f"Line {line_num}: SUPERPOSE cluster '{cont.name}' has "
                "no map at position 0 — no aperture/current carrier "
                "(TraceWin convention: include an empty map at position "
                "0 carrying the aperture and/or current maps)."
            )
        lattice.add(cont)
        _register_cav(cont)

    def _build_field_map_card(kw, line_num, *, in_cluster: bool):
        """Build (but do NOT add) the element for one FIELD_MAP card.

        Returns the element, or ``None`` when the card is dropped
        (missing/rejected field file — warnings/strict-raise exactly as
        the historical inline branch).  ``in_cluster`` children leave a
        pending SET_SYNC_PHASE flag for the first ELECTRIC child
        (static solenoids inside a cluster don't consume it)."""
        nonlocal freq, pending_sync_phase
        if freq is None:
            freq = _DEFAULT_FREQ_MHZ

        # Filename resolution: absolute path > FIELD_MAP_PATH > base_dir.
        raw_name = kw["filename"]
        if os.path.isabs(raw_name):
            prefix_abs = raw_name
        elif field_map_path is not None:
            prefix_abs = os.path.join(field_map_path, raw_name)
        else:
            prefix_abs = os.path.join(base_dir, raw_name)

        # Negative geom = TraceWin's second-order off-axis expansion
        # flag, which HELIX does not implement (first-order paraxial
        # reconstruction).  The flag only MEANS anything for channels
        # whose data is on-axis-only (digit 1 = 1-D, digit 9 = G(z)):
        # 2-D/3-D channels (digits 4-7) are sampled directly from the
        # map at full fidelity, so a negative all-multi-D geom (e.g.
        # -7700) must NOT warn or strict-refuse.  Checked BEFORE the
        # read-try: the inner except would otherwise re-wrap a strict
        # refusal as a misleading "FIELD_MAP rejected" read failure.
        if float(kw["geom"]) < 0:
            from linac_gen.io.tracewin_geom import decode_geom as _dg
            _code = _dg(int(float(kw["geom"])))
            if any(d in (1, 9) for d in (_code.stat_E, _code.stat_B,
                                         _code.rf_E, _code.rf_B)):
                _downgrade(
                    f"Line {line_num}: FIELD_MAP geometry code "
                    f"{int(float(kw['geom']))} < 0 requests TraceWin's "
                    "second-order off-axis expansion on a 1-D channel, "
                    "which HELIX does not implement (permissive mode "
                    "proceeds with the first-order paraxial "
                    "reconstruction)."
                )

        try:
            from linac_gen.io.tracewin_fieldmap_reader import (
                read_tracewin_fieldmap,
            )
            from linac_gen.io.tracewin_geom import decode_geom
            from linac_gen.elements.field_map_factory import (
                make_field_map_element,
            )

            fdata = read_tracewin_fieldmap(
                geom=kw["geom"],
                prefix=prefix_abs,
                base_dir=None,
                frequency=freq,
                Ki=kw["ki"],
                Ka=kw["ka"],
            )
            code = decode_geom(kw["geom"])
            # SET_SYNC_PHASE before this card ⇒ force absolute phase
            # interpretation (θᵢ locked to the synchronous particle).
            # Consumes the flag — except for static-only children inside
            # a cluster, where it stays pending for the first ELECTRIC
            # child (the sync scan is meaningless on a solenoid map).
            p_flag = kw["p_flag"]
            if pending_sync_phase:
                has_e = any(ch.is_electric for ch in fdata.channels)
                if (not in_cluster) or has_e:
                    p_flag = 1
                    pending_sync_phase = False
            # Default n_steps: match the field map's own z-grid so RK4
            # lands on every grid point (ds = Δz of the map, no
            # interpolation between integration nodes).
            try:
                first_ch = next(iter(fdata.channels.values()))
                grid_n = len(first_ch.z) if first_ch.z is not None else 0
            except StopIteration:
                grid_n = 0
            default_steps = max(grid_n - 1, 50)  # floor so tiny maps
                                                 # still get enough steps
            fmname = next_name("FMAP")
            return make_field_map_element(
                name=fmname,
                code=code,
                length_mm=kw["length"],
                field_data=fdata,
                kb=kw["kb"], ke=kw["ke"],
                ki=kw["ki"], ka=kw["ka"],
                phase=kw["phase"], frequency=freq,
                aperture=kw["aperture"],
                p_flag=p_flag,
                n_steps=default_steps,
                geom=kw["geom"],
                # Store an ABSOLUTE prefix so the written card re-parses
                # correctly no matter what directory the output .dat
                # lands in.
                field_file=os.path.abspath(prefix_abs),
            )
        except FileNotFoundError as exc:
            msg = f"Line {line_num}: FIELD_MAP file missing: {exc}"
            if strict:
                # A strict parse must never hand back a lattice with the
                # cavity silently removed.
                raise ValueError(msg) from exc
            metadata["warnings"].append(
                msg + " — ELEMENT DROPPED from the lattice; downstream "
                "energy and optics will be wrong."
            )
            return None
        except (ValueError, NotImplementedError) as exc:
            msg = f"Line {line_num}: FIELD_MAP rejected: {exc}"
            if strict:
                raise ValueError(msg) from exc
            metadata["warnings"].append(
                msg + " — ELEMENT DROPPED from the lattice."
            )
            return None

    # Stateful RF frequency (set by FREQ card)
    freq = None

    # Stateful field map path (set by FIELD_MAP_PATH card)
    field_map_path: str | None = None

    # SET_SYNC_PHASE applies to the **next** FIELD_MAP: its θᵢ is then
    # interpreted as the synchronous phase (locked to the reference
    # particle, i.e. p_flag=1 absolute) instead of a relative offset.
    # Cleared after it's consumed by the following FIELD_MAP.
    pending_sync_phase: bool = False

    # Per-type naming counters
    counters = {}

    def next_name(prefix):
        counters[prefix] = counters.get(prefix, 0) + 1
        return f"{prefix}_{counters[prefix]:03d}"

    # Deck labels are not guaranteed unique (fnalscl reuses ``D01T`` for two
    # THIN_STEERING cards) but several consumers key dicts by element name
    # (correction pairing, GUI kick-apply).  De-duplicate labels used as
    # element names by suffixing ``_2``, ``_3``, …
    seen_labels = set()

    def unique_label(lab):
        if lab not in seen_labels:
            seen_labels.add(lab)
            return lab
        n = 2
        while f"{lab}_{n}" in seen_labels:
            n += 1
        out = f"{lab}_{n}"
        seen_labels.add(out)
        return out

    # ``;@LG key=value`` comment-directives let users set Linac_Gen-specific
    # options (e.g. ``dc_kernel``) without polluting the .dat with non-TW
    # keywords.  TraceWin ignores all comment lines, so these directives are
    # invisible to TW and the file stays portable.  Stored on
    # metadata["lg_options"] for the run script / GUI to consume when
    # constructing the SpaceChargeConfig.
    metadata["lg_options"] = {}

    import re
    _LG_DIRECTIVE = re.compile(r"^\s*;\s*@LG\s+(\S+?)\s*=\s*(.+?)\s*$")
    # ``; HELIX_FOIL <name> <material> <thickness_ug_cm2>`` — HELIX-specific
    # extension that lives as a TraceWin-invisible comment but materialises
    # as a :class:`linac_gen.elements.foil.Foil` element when re-loaded.
    # Trailing whitespace plus an optional second comment (``; …``) are
    # ignored so authors can annotate the line.
    # Optional 4th token: straggling model ("auto"|"landau"|"gaussian");
    # absent means the element default ("auto" — kappa-regime dispatch).
    _HELIX_FOIL = re.compile(
        r"^\s*;\s*HELIX_FOIL\s+(\S+)\s+(\S+)\s+([\d.eE+-]+)"
        r"(?:\s+(auto|landau|gaussian))?\s*(?:;.*)?$"
    )
    # ``; HELIX_SC_GRID <extent_sigma>`` — HELIX-specific comment card:
    # from this position onward the 3-D bunched PIC solver uses the given
    # grid extent (in beam sigmas).  Motivating case: linac at the default
    # extent + BTL at 20 sigma in one combined deck (examples/MEBT_To_Foil).
    _HELIX_SC_GRID = re.compile(
        r"^\s*;\s*HELIX_SC_GRID\s+([\d.eE+-]+)\s*(?:;.*)?$"
    )

    with open(filepath) as fh:
        line_num = 0                       # survives an empty file
        for line_num, raw_line in enumerate(fh, 1):
            # ``;@LG key=value`` directive — extract before stripping comments.
            m = _LG_DIRECTIVE.match(raw_line)
            if m:
                key, value = m.group(1).strip(), m.group(2).strip()
                # Coerce common types so the consumer doesn't have to.
                if value.lower() in ("true", "false"):
                    value = (value.lower() == "true")
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass  # keep as string
                metadata["lg_options"][key] = value
                continue
            # ``; HELIX_FOIL …`` — instantiate the Foil element and skip
            # the regular keyword-parse path.
            m = _HELIX_FOIL.match(raw_line)
            if m:
                from linac_gen.elements.foil import Foil
                lattice.add(Foil(
                    name=m.group(1),
                    material=m.group(2),
                    thickness_ug_cm2=float(m.group(3)),
                    straggling=m.group(4) or "auto",
                ))
                continue
            # ``; HELIX_SC_GRID …`` — zero-length PIC grid-extent directive.
            m = _HELIX_SC_GRID.match(raw_line)
            if m:
                from linac_gen.elements.sc_grid import ScGridDirective
                extent = float(m.group(1))
                if extent > 0.0:
                    lattice.add(ScGridDirective(
                        next_name("SCGRID"), extent_sigma=extent))
                else:
                    msg = (f"Line {line_num}: HELIX_SC_GRID extent must be "
                           f"> 0 sigma, got {extent:g} — card IGNORED.")
                    _downgrade(msg)
                    lattice.add(Marker(next_name("SCGRID"), snapshot=False))
                continue
            # Strip inline comments and whitespace
            line = raw_line.split(";")[0].strip()
            if not line:
                continue

            tokens = [t.strip('"') for t in shlex.split(line, posix=False)]
            # TraceWin allows labels on elements in the form ``NAME : CARD …``
            # (note the space-separated colon), ``NAME: CARD …`` (no space)
            # or ``NAME :CARD …``.  The label is a hint for diagnostics, not
            # a card.  Drop it so the actual keyword stays at tokens[0].
            #
            # *Standalone* diagnostic markers — ``BPM :``, ``ACCT :``,
            # ``RPU :``, ``COL :``, etc. — also use the colon syntax with
            # nothing after.  The colon-strip would silently delete them;
            # detect that here and keep the marker name as the keyword.
            label = None   # element label (``D01BPM``) when the line has one
            if len(tokens) >= 2 and tokens[1] == ":":
                if len(tokens) == 2:
                    # Standalone ``NAME :`` — treat NAME as the card.
                    tokens = [tokens[0]]
                else:
                    # ``NAME : CARD …`` — drop the label, keep the card.
                    label = tokens[0]
                    tokens = tokens[2:]
            elif len(tokens) >= 1 and tokens[0].endswith(":") and \
                    len(tokens[0]) > 1 and tokens[0][0].isalpha():
                # ``NAME: CARD …``  — colon glued to the label.  Gate on
                # "label starts with a letter", NOT "label has no digit":
                # real TraceWin decks name elements with numbers (``Q01:``,
                # ``D02BPM:``, ``M11:``), and the old no-digit guard dropped
                # every such labelled element (QUAD/BEND/EDGE/DIAG/…),
                # mis-reading the label itself as the card keyword.  Only
                # a leading-digit token (e.g. a bare number ``1E-9:``) is
                # now excluded — never a real element name.
                if len(tokens) >= 2:
                    # Drop the label so the card keyword stays at
                    # tokens[0] (mirrors the spaced-colon ``NAME : CARD``
                    # case above).  Covers ``skew: QUAD 200 …`` (PIP-II PDR
                    # placeholders) and ``Q01: QUAD …`` (numbered TraceWin).
                    label = tokens[0][:-1]
                    tokens = tokens[1:]
                # else: standalone ``NAME:`` — fall through to
                # ``tokens[0].rstrip(":").upper()`` below so built-in
                # markers like ``BPM:`` keep working.
            elif ":" in tokens[0] and not tokens[0].startswith(":") \
                    and not tokens[0].endswith(":") \
                    and tokens[0][0].isalpha():
                # ``NAME:CARD param param …``  — colon glued on BOTH sides
                # (no space between label and card).  Used in MAD-style
                # exports of the BTL transfer line: ``BA1011:BEND 6.56 …``,
                # ``QD101:QUAD 200 …``.  Split on the first colon and drop
                # the label so tokens[0] becomes the card keyword.
                _label, _, _card = tokens[0].partition(":")
                if _card:
                    label = _label
                    tokens = [_card] + list(tokens[1:])
            if not tokens:
                continue
            keyword = tokens[0].rstrip(":").upper()
            params = tokens[1:]
            # Some labelled cards still have a trailing colon glued to the
            # second token, e.g. ``SOL1: FIELD_MAP`` parses cleanly via
            # rstrip; ``SOL1 : FIELD_MAP`` was handled by the early skip
            # above.

            n_elems_before = len(lattice.elements)
            try:
                # ── SUPERPOSE cluster close condition ─────────────────
                # The next ORDINARY element ends an open cluster
                # (TraceWin).  Transparent cards: further SUPERPOSE_MAP
                # pairs, path/numerics/title state, SET_*/ADJUST_*
                # command cards (zero-length markers — they land before
                # the container; ADJUST inside a cluster is warn-skipped
                # by the matcher), and ERROR_* directives.  FREQ closes
                # WITH a warning (mixed-frequency clusters are
                # unsupported); everything else closes silently.
                if open_cluster or pending_superpose:
                    transparent = (
                        keyword in ("SUPERPOSE_MAP", "TITLE",
                                    "PARTRAN_STEP", "FIELD_MAP_PATH")
                        or keyword in COMMAND_CLASSES
                        or keyword.startswith("ERROR_")
                        or (keyword == "FIELD_MAP"
                            and bool(pending_superpose))
                    )
                    if keyword == "FREQ":
                        # Physics downgrade, not furniture: the maps
                        # that were meant to OVERLAP become sequential
                        # elements, so the line is LONGER than the deck
                        # intends (span max(z0+L) → ΣL) and every
                        # downstream element shifts.  _downgrade so
                        # strict refuses instead of returning the
                        # mislengthened lattice.
                        _downgrade(
                            f"Line {line_num}: FREQ closes the open "
                            "SUPERPOSE cluster (mixed-frequency "
                            "clusters are unsupported) — the maps are "
                            "laid out SEQUENTIALLY instead of "
                            "overlapped, lengthening the lattice and "
                            "shifting every downstream element."
                        )
                        transparent = False
                    if not transparent:
                        _flush_cluster(line_num)

                # ── SHIFT_IN_FIELD_MAP orphan condition ───────────────
                # Runs AFTER the cluster flush (which consumes pending
                # interior markers).  Anything still pending when an
                # ordinary non-map card arrives has no map to overlap —
                # restore as plain markers, loudly.
                if pending_shift_dz or pending_interior:
                    shift_transparent = (
                        keyword in ("SHIFT_IN_FIELD_MAP", "SUPERPOSE_MAP",
                                    "FIELD_MAP", "TITLE", "PARTRAN_STEP",
                                    "FIELD_MAP_PATH")
                        or keyword in shift_capturable
                        or keyword in COMMAND_CLASSES
                    )
                    if not shift_transparent:
                        if pending_interior:
                            for _dz, mk in pending_interior:
                                lattice.add(mk)
                            pending_interior.clear()
                            _downgrade(
                                f"Line {line_num}: SHIFT_IN_FIELD_MAP "
                                "diagnostics have no following FIELD_MAP "
                                "— restored as ordinary markers at the "
                                "card position (nothing to overlap)."
                            )
                        if pending_shift_dz:
                            pending_shift_dz.clear()
                            _downgrade(
                                f"Line {line_num}: SHIFT_IN_FIELD_MAP "
                                "not followed by a diagnostic element — "
                                "the shift is ignored."
                            )

                # ── Control cards ──────────────────────────────────────────
                if keyword == "END":
                    break

                elif keyword == "TITLE":
                    metadata["title"] = " ".join(params)

                elif keyword == "FREQ":
                    freq = float(params[0])
                    # Materialize as an active Freq command so the machine
                    # RF clock switches AT THE CARD at track time (TraceWin
                    # semantics), not at the first downstream RF element.
                    # The stateful ``freq`` above still stamps every RF
                    # element's own frequency (unchanged behaviour).
                    lattice.add(Freq(next_name("FREQ"), frequency_mhz=freq))

                elif keyword == "PARTRAN_STEP":
                    if len(params) < 2:
                        raise ValueError(
                            f"line {line_num}: PARTRAN_STEP needs 2 numbers "
                            f"(step1, step2), got {len(params)}"
                        )
                    from linac_gen.core.step_config import StepConfig
                    lattice.step_config = StepConfig(
                        integration_steps_per_metre=float(params[0]),
                        sc_steps_per_metre=float(params[1]),
                    )

                # ── Drift ──────────────────────────────────────────────────
                elif keyword == "DRIFT":
                    kw = parse_positionals(SCHEMA["DRIFT"], params)
                    lattice.add(Drift(
                        next_name("DRIFT"),
                        length=kw["length"],
                        aperture=kw["aperture"],
                        aperture_y=kw["aperture_y"],
                        x_shift=kw["x_shift"],
                        y_shift=kw["y_shift"],
                    ))

                # ── Quadrupole ─────────────────────────────────────────────
                elif keyword == "QUAD":
                    kw = parse_positionals(SCHEMA["QUAD"], params)
                    qname = next_name("QUAD")
                    lattice.add(Quadrupole(
                        qname,
                        length=kw["length"],
                        gradient=kw["gradient"],
                        aperture=kw["aperture"],
                        skew_angle=kw["skew_angle"],
                        g3=kw["g3"], g4=kw["g4"], g5=kw["g5"], g6=kw["g6"],
                        gfr=kw["gfr"],
                    ))
                    error_state.on_element("QUAD", qname)

                # ── Solenoid ───────────────────────────────────────────────
                elif keyword == "SOLENOID":
                    kw = parse_positionals(SCHEMA["SOLENOID"], params)
                    lattice.add(Solenoid(
                        next_name("SOL"),
                        length=kw["length"],
                        field=kw["field"],
                        aperture=kw["aperture"],
                    ))

                # ── RF Gap ─────────────────────────────────────────────────
                elif keyword == "GAP":
                    kw = parse_positionals(SCHEMA["GAP"], params)
                    if freq is None:
                        metadata["warnings"].append(
                            f"Line {line_num}: GAP before FREQ, using {_DEFAULT_FREQ_MHZ} MHz"
                        )
                        freq = _DEFAULT_FREQ_MHZ
                    # p_flag is stored on RFGap for round-trip but the thin
                    # gap's apply_kick never reads it — only FieldMap/
                    # FieldMap3D implement absolute/sync-phase modes.
                    if int(kw["p_flag"]) != 0:
                        # Physics downgrade -> _downgrade so strict mode
                        # RAISES (was a bare warnings-append that strict
                        # silently accepted; 2026-07-25 review, claim 5).
                        _downgrade(
                            f"Line {line_num}: GAP p_flag={int(kw['p_flag'])} "
                            "(absolute/sync-phase mode) is not implemented "
                            "on the thin GAP — the phase is treated as a "
                            "relative offset.  Use a FIELD_MAP for absolute-"
                            "phase cavities."
                        )
                    # SET_SYNC_PHASE binds to the NEXT FIELD_MAP only; if
                    # one is pending when a GAP appears, TraceWin would
                    # have applied it to this GAP but HELIX will instead
                    # apply it to a later FIELD_MAP — tell the deck author.
                    if pending_sync_phase:
                        metadata["warnings"].append(
                            f"Line {line_num}: SET_SYNC_PHASE is pending but "
                            "the next cavity is a thin GAP — HELIX applies "
                            "SET_SYNC_PHASE only to the next FIELD_MAP, so "
                            "this GAP keeps its relative phase and the "
                            "sync-phase flag will bind to a LATER field map."
                        )
                    # TraceWin voltage is in V; existing RFGap expects MV.
                    voltage_mv = kw["e0tl"] * 1e-6
                    gname = next_name("GAP")
                    lattice.add(RFGap(
                        gname,
                        voltage=voltage_mv,
                        phase=kw["phase"],
                        frequency=freq,
                        ttf=1.0,           # TraceWin embeds the TTF in E0TL
                        aperture=kw["aperture"],
                        p_flag=kw["p_flag"],
                    ))
                    error_state.on_element("CAV", gname)

                # ── Field map path ────────────────────────────────────────
                elif keyword == "FIELD_MAP_PATH":
                    kw = parse_positionals(SCHEMA["FIELD_MAP_PATH"], params)
                    p = kw["path"]
                    if not os.path.isabs(p):
                        p = os.path.join(base_dir, p)
                    field_map_path = p

                # ── Field map ──────────────────────────────────────────────
                elif keyword == "FIELD_MAP":
                    kw = parse_positionals(SCHEMA["FIELD_MAP"], params)
                    if pending_superpose:
                        z0 = pending_superpose.pop()
                        elem = _build_field_map_card(kw, line_num,
                                                     in_cluster=True)
                        if elem is not None:
                            open_cluster.append((z0, elem))
                        else:
                            metadata["warnings"].append(
                                f"Line {line_num}: SUPERPOSE cluster "
                                "child dropped — the cluster span and "
                                "summed fields no longer match the deck."
                            )
                    else:
                        elem = _build_field_map_card(kw, line_num,
                                                     in_cluster=False)
                        if elem is not None:
                            interior = []
                            if pending_interior or pending_shift_dz:
                                interior = _take_interior(elem.length,
                                                          line_num)
                            if interior:
                                # SHIFT diagnostics inside a PLAIN map
                                # need the container walk too.  The wrap
                                # flag lets the writer re-emit the plain
                                # FIELD_MAP card (no SUPERPOSE_MAP in the
                                # source deck).
                                from linac_gen.elements import (
                                    superposed_field_map as _sfm_mod,
                                )
                                cont = _sfm_mod.SuperposedFieldMap(
                                    next_name("SUPERP"), [(0.0, elem)],
                                    interior_markers=interior)
                                cont._from_plain_wrap = True
                                lattice.add(cont)
                                _register_cav(cont)
                            else:
                                lattice.add(elem)
                                _register_cav(elem)
                        else:
                            if pending_interior or pending_shift_dz:
                                for _dz, mk in pending_interior:
                                    lattice.add(mk)
                                pending_interior.clear()
                                pending_shift_dz.clear()
                                metadata["warnings"].append(
                                    f"Line {line_num}: the FIELD_MAP "
                                    "bound to SHIFT_IN_FIELD_MAP "
                                    "diagnostics was dropped — the "
                                    "diagnostics are restored as "
                                    "ordinary markers at the card "
                                    "position."
                                )

                # ── Thin steering corrector ────────────────────────────────
                elif keyword in ("STEERER", "THIN_STEERING"):
                    kw = parse_positionals(SCHEMA["STEERER"], params)
                    # elec=1: the operands are ∫E·dl in VOLTS and the
                    # kick is same-plane via the electric rigidity
                    # (Steerer handles both flavours).
                    lattice.add(Steerer(
                        next_name("STEER"),
                        bx_l=kw["bl_x"],
                        by_l=kw["bl_y"],
                        elec=bool(kw.get("elec", 0)),
                    ))

                # ── Dipole / sector bend ───────────────────────────────────
                elif keyword == "BEND":
                    kw = parse_positionals(SCHEMA["BEND"], params)
                    bname = next_name("BEND")
                    lattice.add(Dipole(
                        bname,
                        angle=kw["angle"],
                        rho=kw["rho"],
                        field_index=kw["field_index"],
                        aperture=kw["aperture"],
                        hv=kw["hv"],
                    ))
                    error_state.on_element("BEND", bname)

                # ── Edge (pole-face rotation) ──────────────────────────────
                elif keyword == "EDGE":
                    kw = parse_positionals(SCHEMA["EDGE"], params)
                    lattice.add(Edge(
                        next_name("EDGE"),
                        pole_rotation=kw["pole_rotation"],
                        rho=kw["rho"],
                        gap=kw["gap"],
                        k1=kw["k1"],
                        k2=kw["k2"],
                        aperture=kw["aperture"],
                        hv=kw["hv"],
                    ))

                # ── Aperture ───────────────────────────────────────────────
                elif keyword == "APERTURE":
                    kw = parse_positionals(SCHEMA["APERTURE"], params)
                    lattice.add(Aperture(
                        next_name("APER"),
                        dx=kw["dx"],
                        dy=kw["dy"],
                        aperture_type=kw["ap_type"],
                    ))

                # ── Diagnostics / markers ──────────────────────────────────
                elif keyword == "MARKER":
                    lattice.add(Marker(next_name("MARK"), snapshot=False))

                elif keyword in ("DIAG_SIZE", "DIAG_EMIT"):
                    lattice.add(Marker(next_name("MARK"), snapshot=False))

                elif keyword == "DIAG_PHASE":
                    lattice.add(Marker(next_name("MARK"), snapshot=True))

                elif keyword == "DIAG_POSITION":
                    # BPM-equivalent, carrying TraceWin's diagnostic-matching
                    # data: ``DIAG_POSITION N X Y [dm]`` — N = diagnostic
                    # (family) number linking to ``ADJUST N v`` variables,
                    # X/Y = wanted beam centroid (mm; |value| >= 1e50 leaves
                    # that plane unconstrained), dm = accuracy (mm, default
                    # 1).  A bare card (no operands) stays a plain BPM.
                    # Malformed operands degrade to a target-less BPM with
                    # a warning — the MARKER must survive (dropping it
                    # would shift element counts and every legacy
                    # index-based ADJUST downstream).
                    fam, x_t, y_t, dm = None, None, None, 1.0
                    try:
                        if params:
                            fam = int(float(params[0]))
                            if len(params) > 1:
                                v = float(params[1])
                                x_t = None if abs(v) >= 1e50 else v
                            if len(params) > 2:
                                v = float(params[2])
                                y_t = None if abs(v) >= 1e50 else v
                            if len(params) > 3:
                                dm = float(params[3])
                    except ValueError:
                        metadata["warnings"].append(
                            f"Line {line_num}: DIAG_POSITION operands "
                            f"{params!r} not numeric — keeping the BPM "
                            f"marker without targets")
                        fam, x_t, y_t, dm = None, None, None, 1.0
                    name = unique_label(label) if label \
                        else unique_label(next_name("BPM"))
                    lattice.add(Marker(name, snapshot=False, is_bpm=True,
                                       diag_family=fam,
                                       x_target_mm=x_t, y_target_mm=y_t,
                                       accuracy_mm=dm))

                # ── Space-charge compensation ──────────────────────────────
                elif keyword == "SPACE_CHARGE_COMP":
                    factor = float(params[0])
                    lattice.add(SpaceChargeComp(next_name("SCCOMP"), factor))

                # ── RFQ_CELL ───────────────────────────────────────────────
                # Manual: V[V]  Ro[mm]  A10  m  L[mm]  θs[deg]  Type [Tc] [dP]
                # We chain successive cells by remembering the previous
                # cell so its type_next / our type_prev can be set, which
                # the per-cell transverse model (S = -sign(type[n±1]))
                # requires for ±3 / ±4 cells.
                elif keyword == "RFQ_CELL":
                    from linac_gen.elements.rfq_cell import RfqCell
                    if len(params) < 7:
                        raise ValueError(
                            f"Line {line_num}: RFQ_CELL needs at least 7 "
                            f"operands (V Ro A10 m L θs Type), got {len(params)}"
                        )
                    V_volts = float(params[0])
                    r0_mm   = float(params[1])
                    A10     = float(params[2])
                    m_mod   = float(params[3])
                    L_mm    = float(params[4])
                    phi_s   = float(params[5])
                    ctype   = int(float(params[6]))
                    Tc_mm   = float(params[7]) if len(params) > 7 else 0.0
                    dP_deg  = float(params[8]) if len(params) > 8 else 0.0
                    rfq = RfqCell(
                        next_name("RFQ"),
                        voltage_V=V_volts, r0_mm=r0_mm, A10=A10,
                        modulation=m_mod, length_mm=L_mm,
                        phi_s_deg=phi_s, cell_type=ctype,
                        Tc_mm=Tc_mm, dP_deg=dP_deg,
                    )
                    # Back-link the previous RFQ cell so it knows its
                    # forward neighbour, and prime our prev-link from it.
                    # Walk back over zero-length passive cards ("lattice",
                    # "lattice_end", markers…) — genuine TW decks place
                    # them BETWEEN RFQ_CELL lines and the naive
                    # elements[-1] check silently broke the type chain
                    # exactly there (found 2026-07-30: the -4 transcell
                    # after "lattice_end" got a default neighbour and a
                    # plane-swapped transverse matrix).  A thick element
                    # still terminates the chain: separate RFQ sections
                    # must not couple.
                    for prev in reversed(lattice.elements):
                        if isinstance(prev, RfqCell):
                            rfq.type_prev = prev.cell_type
                            prev.type_next = rfq.cell_type
                            break
                        if float(getattr(prev, "length", 0.0) or 0.0) > 0.0:
                            break
                    lattice.add(rfq)

                # ── Multi-gap cavity (NCELLS) ─────────────────────────────
                elif keyword == "NCELLS":
                    from linac_gen.elements.ncells import (
                        NCells, parse_ttf_tail,
                    )
                    if len(params) < 6:
                        raise ValueError(
                            f"Line {line_num}: NCELLS needs at least 6 "
                            f"operands (mode Nc betaG EoT thetaS R), got "
                            f"{len(params)}"
                        )
                    if freq is None:
                        metadata["warnings"].append(
                            f"Line {line_num}: NCELLS before FREQ, using "
                            f"{_DEFAULT_FREQ_MHZ} MHz"
                        )
                        freq = _DEFAULT_FREQ_MHZ
                    nc_mode = int(float(params[0]))
                    nc_n    = int(float(params[1]))
                    nc_bg   = float(params[2])
                    nc_eot  = float(params[3])
                    nc_th   = float(params[4])
                    nc_R    = float(params[5])
                    nc_p    = int(float(params[6])) if len(params) > 6 else 0
                    nc_kEi  = float(params[7]) if len(params) > 7 else 0.0
                    nc_kEo  = float(params[8]) if len(params) > 8 else 0.0
                    nc_dzi  = float(params[9]) if len(params) > 9 else 0.0
                    nc_dzo  = float(params[10]) if len(params) > 10 else 0.0
                    nc_ttf  = (parse_ttf_tail(params[11:])
                               if len(params) > 11 else None)
                    # SET_SYNC_PHASE binds to the NEXT cavity (mirror the
                    # FIELD_MAP path): consume the pending flag; θs then means
                    # the synchronous phase, not the absolute phase.
                    nc_sync = pending_sync_phase
                    if pending_sync_phase:
                        pending_sync_phase = False
                        if nc_p == 1:
                            metadata["warnings"].append(
                                f"Line {line_num}: NCELLS has P=1 (absolute) "
                                "AND a pending SET_SYNC_PHASE — SET_SYNC_PHASE "
                                "wins (theta_s treated as the synchronous phase)."
                            )
                    nname = next_name("NCELLS")
                    lattice.add(NCells(
                        nname, mode=nc_mode, n_cells=nc_n, beta_g=nc_bg,
                        eot_v_per_m=nc_eot, theta_s_deg=nc_th, aperture_mm=nc_R,
                        p_flag=nc_p, k_eot_i=nc_kEi, k_eot_o=nc_kEo,
                        dz_i_mm=nc_dzi, dz_o_mm=nc_dzo, frequency_mhz=freq,
                        sync_phase=nc_sync, ttf=nc_ttf,
                    ))
                    error_state.on_element("CAV", nname)

                # ── SET / ADJUST commands ─────────────────────────────────
                # Each parses into a ``LatticeCommand`` subclass (see
                # ``linac_gen.elements.lattice_commands``).  Some — notably
                # ``SET_SYNC_PHASE`` — are also kept as a one-shot parser
                # flag for back-compat with the existing FIELD_MAP per-cell
                # ``p_flag=1`` interpretation: the runtime ``apply_command``
                # hook only matters for tracking-loop call paths that
                # consult ``TrackState``.
                elif keyword in COMMAND_CLASSES:
                    cmd_cls = COMMAND_CLASSES[keyword]
                    # ADJUST_BEAM_* takes a variable-length flag tail.
                    if keyword in ("ADJUST_BEAM_TWISS", "ADJUST_BEAM_CENTROID",
                                   "ADJUST_BEAM_EMIT", "ADJUST_BEAM_CURRENT"):
                        if not params:
                            raise ValueError(
                                f"line {line_num}: {keyword} requires at least 1 arg"
                            )
                        diag_n = int(float(params[0]))
                        flags = [int(float(t)) for t in params[1:]]
                        cmd = cmd_cls(next_name(keyword), diag_n, *flags)
                    else:
                        schema = SCHEMA.get(keyword)
                        if schema:
                            kw = parse_positionals(schema, params)
                            cmd = cmd_cls(next_name(keyword), **kw)
                        else:
                            cmd = cmd_cls(next_name(keyword))
                    lattice.add(cmd)
                    if (keyword in ("SET_SIZE", "SET_SIZE_MAX",
                                    "SET_SIZE_MIN")
                            and getattr(cmd, "k2", 0)):
                        # k2 = include-the-centroid transverse sizes (TW
                        # manual) — not modelled; per-parse so the GUI
                        # surfaces it on every load (the collect-time
                        # warnings.warn dedupes per process).
                        metadata["warnings"].append(
                            f"Line {line_num}: {keyword} k2="
                            f"{cmd.k2} (transverse sizes including the "
                            "beam centroid) is not modelled — sizes are "
                            "evaluated about the centroid (k2=0 form)."
                        )
                    # NOTE: ADJUST's start_step is parsed/round-tripped but
                    # deliberately NOT warned about — real decks carry it as
                    # routine furniture and it is only an optimizer *hint*
                    # (HELIX optimizers pick their own initial steps; results
                    # are unaffected).  Disclosed in the keyword cheatsheet
                    # and the Adjust docstring instead.
                    if keyword == "SET_SYNC_PHASE":
                        pending_sync_phase = True

                elif keyword in ("LATTICE", "LATTICE_END"):
                    # TraceWin: ``LATTICE n_cells flag`` opens a periodic
                    # block; ``LATTICE_END`` closes it.  We don't simulate
                    # the periodicity (transport is unaffected) but we DO
                    # need to recover the numeric args downstream so that
                    # period-detection can read out the cell count.
                    # The args are stashed on the Marker as
                    # ``lattice_card_args`` (a list of floats); writers
                    # ignore unknown attributes so this is safe for
                    # round-tripping.
                    args: list[float] = []
                    for tok in params:
                        try:
                            args.append(float(tok))
                        except ValueError:
                            break
                    m = Marker(next_name(keyword), snapshot=False)
                    m.lattice_card_args = args
                    lattice.add(m)

                elif keyword == "SUPERPOSE_MAP":
                    # Cluster placement card: Z0 [X0 Y0 θz θx θy].  The
                    # transverse/rotation operands are honoured by
                    # TraceWin ONLY with SUPERPOSE_MAP_OUT (curved
                    # reference frames, unsupported) — ignoring them is
                    # FAITHFUL, so nonzero extras warn without a strict
                    # refusal.
                    if not params:
                        _downgrade(
                            f"Line {line_num}: SUPERPOSE_MAP needs a Z0 "
                            "operand — card ignored."
                        )
                    else:
                        if pending_superpose:
                            _downgrade(
                                f"Line {line_num}: consecutive "
                                "SUPERPOSE_MAP cards without a FIELD_MAP "
                                "— the earlier placement is ignored."
                            )
                            pending_superpose.clear()
                        z0 = float(params[0])
                        extras = []
                        for tok in params[1:7]:
                            try:
                                extras.append(float(tok))
                            except ValueError:
                                break
                        if any(abs(v) > 0.0 for v in extras):
                            metadata["warnings"].append(
                                f"Line {line_num}: SUPERPOSE_MAP "
                                "transverse/rotation operands ignored — "
                                "TraceWin itself honours them only with "
                                "SUPERPOSE_MAP_OUT (curved reference "
                                "frames), which HELIX does not support."
                            )
                        pending_superpose.append(z0)

                elif keyword == "SHIFT_IN_FIELD_MAP":
                    # ``SHIFT_IN_FIELD_MAP dz`` — the FOLLOWING diagnostic
                    # reads the beam dz mm INTO the span of the next
                    # FIELD_MAP/cluster (TraceWin: dz > 0; several pairs
                    # allowed).
                    dz_val = None
                    if params:
                        try:
                            dz_val = float(params[0])
                        except ValueError:
                            dz_val = None
                    if dz_val is None or dz_val <= 0.0:
                        _downgrade(
                            f"Line {line_num}: SHIFT_IN_FIELD_MAP needs "
                            "dz > 0 — card ignored."
                        )
                    else:
                        if pending_shift_dz:
                            _downgrade(
                                f"Line {line_num}: consecutive "
                                "SHIFT_IN_FIELD_MAP cards without a "
                                "diagnostic — the earlier shift is "
                                "ignored."
                            )
                            pending_shift_dz.clear()
                        pending_shift_dz.append(dz_val)

                elif keyword == "SUPERPOSE_MAP_OUT":
                    # Declares a CURVED reference trajectory through the
                    # cluster (dipole field maps), which HELIX cannot
                    # model.  Physics downgrade: warn / strict-refuse.
                    _downgrade(
                        f"Line {line_num}: SUPERPOSE_MAP_OUT is not "
                        "modelled — curved-reference clusters (dipole "
                        "field maps) are unsupported; the exit-frame "
                        "card is ignored."
                    )
                    lattice.add(Marker(next_name(keyword), snapshot=False))

                elif keyword in ("READ_DST", "BEAM_ROT"):
                    # Both mutate the BEAM mid-line in TraceWin (re-read a
                    # distribution / rotate the bunch); HELIX ignores them.
                    _downgrade(
                        f"Line {line_num}: {keyword} is ignored — HELIX "
                        "does not re-load or rotate the beam mid-lattice; "
                        "the tracked beam keeps its upstream state."
                    )
                    lattice.add(Marker(next_name(keyword), snapshot=False))

                elif keyword in ("MATCH_FAM_FIELD", "MATCH_FAM_GRAD",
                                 "MIN_FIELD_VARIATION",
                                 # RFQ_GEOM / RFQ_GAP_RMS_FFS are pre-RFQ
                                 # setup cards; RfqCell elements read
                                 # cell-local parameters off each RFQ_CELL
                                 # line directly.
                                 "RFQ_GEOM", "RFQ_GAP_RMS_FFS"):
                    # TraceWin fit hints / RFQ vane setup: no transport
                    # effect in HELIX.  Reported once per deck (below).
                    noop_hint_counts[keyword] = (
                        noop_hint_counts.get(keyword, 0) + 1)
                    lattice.add(Marker(next_name(keyword), snapshot=False))

                elif keyword in (
                    "BPM", "XCOR", "YCOR", "ACCT", "COL", "RPU", "FFC",
                    "DPI", "DCCT", "RWCM", "ASCN", "FASTGV", "LASERPROFILE",
                    "MEBTABSORBER", "CHOPPER",
                    "END",  # lowercase "end" is common as a lattice terminator
                ):
                    # Recognised diagnostic-hardware no-op — emit as a
                    # marker.  Tag BPM cards so the orbit-correction driver
                    # picks them up alongside DIAG_POSITION; remember the
                    # origin keyword so the writer round-trips ``BPM`` as
                    # ``BPM`` (not ``DIAG_POSITION``).
                    lattice.add(Marker(
                        next_name(keyword), snapshot=False,
                        is_bpm=(keyword == "BPM"),
                        origin_keyword=keyword if keyword == "BPM" else None,
                    ))

                # ── ERROR_* family — translate into ErrorDef list ─────────
                elif keyword.startswith("ERROR_"):
                    try:
                        nums = [float(p) for p in params]
                    except (TypeError, ValueError):
                        nums = []
                    kw_up = keyword.upper()
                    # TraceWin's coupled groups share ONE random draw
                    # across the group; HELIX samples every element and
                    # parameter independently.  The card still takes
                    # effect (as its NCPL equivalent) — declare the
                    # correlation loss.
                    if "_CPL_" in kw_up and "_NCPL_" not in kw_up:
                        _downgrade(
                            f"Line {line_num}: {keyword}: coupled error "
                            "groups are not implemented — every element "
                            "and parameter is sampled INDEPENDENTLY "
                            "(group correlations ignored)."
                        )
                    # Dynamic errors happen AFTER correction in a real
                    # machine; HELIX draws them once per seed BEFORE the
                    # correction pass, which will steer against jitter it
                    # physically could not see.
                    if kw_up.endswith("_DYN") and kw_up != "ERROR_BEAM_DYN":
                        _downgrade(
                            f"Line {line_num}: {keyword}: dynamic errors "
                            "are treated as STATIC (drawn once per seed, "
                            "subject to the orbit-correction pass)."
                        )
                    if kw_up == "ERROR_BEAM_DYN":
                        _downgrade(
                            f"Line {line_num}: ERROR_BEAM_DYN is treated "
                            "as ERROR_BEAM_STAT (static per-seed beam "
                            "error)."
                        )
                    if keyword.startswith("ERROR_QUAD"):
                        _error_directive(line_num,
                                         error_state.add_quad_directive, nums)
                    elif keyword.startswith("ERROR_CAV"):
                        _error_directive(line_num,
                                         error_state.add_cav_directive, nums)
                    elif keyword.startswith("ERROR_BEND"):
                        _error_directive(line_num,
                                         error_state.add_bend_directive, nums)
                    elif kw_up in ("ERROR_BEAM_STAT", "ERROR_BEAM_DYN"):
                        _error_directive(line_num,
                                         error_state.add_beam_directive, nums)
                    elif keyword == "ERROR_GAUSSIAN_CUT_OFF":
                        _error_directive(line_num,
                                         error_state.add_cutoff, nums)
                    elif keyword == "ERROR_SET_RATIO":
                        _error_directive(line_num,
                                         error_state.add_ratios, nums)
                    else:
                        # ERROR_*_FILE / ERROR_RFQ_* etc.: recognised but
                        # NOT implemented — no errors will be applied for
                        # this card.  Visible marker + downgrade warning.
                        _downgrade(
                            f"Line {line_num}: {keyword} is not "
                            "implemented — the card is IGNORED (no errors "
                            "will be applied)."
                        )
                        lattice.add(Marker(next_name(keyword), snapshot=False))

                # ── Unknown card ───────────────────────────────────────────
                else:
                    msg = f"Line {line_num}: unsupported card '{keyword}'"
                    if strict:
                        raise ValueError(msg)
                    metadata["warnings"].append(msg)

                # ── SHIFT_IN_FIELD_MAP capture ─────────────────────────
                # Pull the diagnostic Marker just added after a SHIFT
                # card out of the element stream — it lives INSIDE the
                # following map's span and is attached to the container
                # at flush (the marker keeps its card provenance for the
                # writer).
                if (pending_shift_dz
                        and keyword in shift_capturable
                        and len(lattice.elements) > n_elems_before
                        and isinstance(lattice.elements[-1], Marker)
                        and lattice.elements[-1].length == 0.0):
                    if getattr(lattice.elements[-1], "is_bpm", False):
                        # BPMs must stay lattice elements (ordinal
                        # resolution in orbit correction / diagnostic
                        # matching) — the marker keeps its position, the
                        # shift is dropped.
                        pending_shift_dz.clear()
                        _downgrade(
                            f"Line {line_num}: '{keyword}' after "
                            "SHIFT_IN_FIELD_MAP is a BPM — BPMs are "
                            "resolved ordinally as lattice elements and "
                            "cannot move inside a field map; the BPM "
                            "stays at the card position and the shift "
                            "is ignored."
                        )
                    else:
                        mk = lattice.elements.pop()
                        if getattr(mk, "origin_keyword", None) is None:
                            mk.origin_keyword = keyword
                        mk.origin_params = list(params)
                        pending_interior.append((pending_shift_dz.pop(), mk))

            except (IndexError, ValueError) as exc:
                # Re-raise immediately in strict mode only when we raised it ourselves
                if strict and isinstance(exc, ValueError):
                    raise
                msg = f"Line {line_num}: error parsing '{keyword}': {exc}"
                if strict:
                    raise ValueError(msg) from exc
                metadata["warnings"].append(msg)

    # A cluster still open at END/EOF closes here (TraceWin: the deck
    # end is as good an "ordinary element" as any).
    _flush_cluster(line_num)

    # SHIFT state still pending at EOF (deck without END): nothing left
    # to overlap — same restore-loudly contract as the mid-deck orphan
    # condition (an END card reaches that block; a bare EOF lands here).
    if pending_interior:
        for _dz, mk in pending_interior:
            lattice.add(mk)
        pending_interior.clear()
        _downgrade(
            f"Line {line_num}: SHIFT_IN_FIELD_MAP diagnostics have no "
            "following FIELD_MAP — restored as ordinary markers at the "
            "card position (nothing to overlap)."
        )
    if pending_shift_dz:
        pending_shift_dz.clear()
        _downgrade(
            f"Line {line_num}: SHIFT_IN_FIELD_MAP not followed by a "
            "diagnostic element — the shift is ignored."
        )

    # One grouped line for the low-stakes recognised no-ops (fit hints /
    # RFQ setup) — informative, not strict-fatal.
    if noop_hint_counts:
        cards = ", ".join(f"{k}×{v}"
                          for k, v in sorted(noop_hint_counts.items()))
        metadata["warnings"].append(
            "Recognised no-op cards ignored (TraceWin fit hints / RFQ "
            f"setup; no transport effect in HELIX): {cards}")

    # Attach lg_options to the lattice itself so downstream consumers (GUI,
    # Simulation, run scripts) can find them without holding the metadata
    # dict separately.
    lattice.lg_options = dict(metadata["lg_options"])

    # Push accumulated ERROR_* directives onto lattice.errors / .beam_errors.
    error_state.finalize(lattice)

    return lattice, metadata
