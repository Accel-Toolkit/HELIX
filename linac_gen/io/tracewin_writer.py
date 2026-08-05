"""TraceWin .dat lattice file writer.

Serialises a :class:`Lattice` to a TraceWin-format ``.dat`` text file.

Supported element types
-----------------------
Drift, Quadrupole, Solenoid, RFGap, Dipole, Edge, Steerer, Aperture,
Marker (snapshot → DIAG_PHASE, plain → MARKER), SpaceChargeComp,
FieldMap and FieldMap3D (re-emitted as FIELD_MAP cards using the geom +
field-file provenance the parser stores on the element; a map built
programmatically without that provenance falls back to a comment).

Unknown element types are silently skipped (a comment line is written instead).

Positional-order conventions follow ``linac_gen.io.tracewin_syntax.SCHEMA``
(the authoritative TraceWin ordering).  Trailing fields that match their
schema defaults are elided so the output stays compact and re-parses
cleanly via :func:`linac_gen.io.tracewin_parser.parse_tracewin`.
"""

import os
import re
import warnings

from linac_gen.io.portable_paths import best_relpath

from linac_gen.elements.aperture import Aperture
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.edge import Edge
from linac_gen.elements.field_map import FieldMap
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.elements.foil import Foil
from linac_gen.elements.lattice_commands import LatticeCommand
from linac_gen.elements.sc_grid import ScGridDirective
from linac_gen.elements.lattice_commands import Freq as FreqCommand
from linac_gen.elements.marker import Marker
from linac_gen.elements.ncells import NCells
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.superposed_field_map import SuperposedFieldMap
from linac_gen.elements.rf_gap import RFGap
from linac_gen.elements.rfq_cell import RfqCell
from linac_gen.elements.solenoid import Solenoid
from linac_gen.elements.space_charge_comp import SpaceChargeComp
from linac_gen.elements.steerer import Steerer
from linac_gen.elements.vane_rfq import VaneRFQ


def write_tracewin(lattice, filepath, frequency=None):
    """Write a :class:`Lattice` to a TraceWin ``.dat`` file.

    Parameters
    ----------
    lattice : Lattice
        The lattice to serialise.
    filepath : str or Path
        Destination file path.  Any existing file is overwritten.
    frequency : float or None, optional
        If provided, write a ``FREQ`` header line before any elements.
        The writer also emits a ``FREQ`` card automatically before each RF
        element whose frequency differs from the previously written value,
        so this parameter is usually not needed.

    Notes
    -----
    * Lengths and apertures are written in **mm** to match internal units.
    * Floating-point values are written with ``g`` format; sufficient
      precision is preserved for a parse–write round-trip.
    * ``GAP`` cards emit ``E0TL`` in **volts** (not MV) and a ``p_flag``
      4th positional, per the TraceWin schema.
    * ``SOLENOID`` cards emit only ``length field aperture`` — ``n_steps``
      is a Linac_Gen-only attribute and is NOT round-trippable via .dat.
    """
    filepath = str(filepath)
    out_dir = os.path.dirname(os.path.abspath(filepath))

    # Honesty (adversarial-review finding): the writer serialises no
    # ERROR_* cards, so a lattice carrying an error study loses it on
    # save — and the round-tripped deck would no longer even warn,
    # because the cards are simply gone.  Until ERROR_* emission is
    # implemented, say so loudly instead of silently deleting physics.
    if getattr(lattice, "errors", None) or getattr(lattice,
                                                   "beam_errors", None):
        warnings.warn(
            "write_tracewin: this lattice carries ERROR_* definitions "
            f"({len(getattr(lattice, 'errors', []) or [])} element + "
            f"{len(getattr(lattice, 'beam_errors', []) or [])} beam) "
            "which the writer does NOT serialise — the saved .dat "
            "loses the error study.  Keep the original deck for error "
            "work.",
            stacklevel=2)

    current_freq = None   # last FREQ value written to the file
    emitted_fmp = None    # map directory currently in force via FIELD_MAP_PATH
    warned_map_dirs = set()   # absolute-path fallback warned once per dir

    def _fmt(v):
        """Format a float with enough sig-figs for a round-trip."""
        return f"{v:.10g}"

    def _emit_card(fh, keyword, required, optionals):
        """Emit ``keyword`` followed by ``required`` fields plus trailing
        ``optionals`` up to the last non-default entry.

        Parameters
        ----------
        required : list of str
            Pre-formatted required positional tokens (always emitted).
        optionals : list of (token_str, is_default)
            Trailing optionals; we emit up to the last one whose
            ``is_default`` is ``False`` (everything strictly after that
            point is elided).
        """
        last = -1
        for i, (_, is_default) in enumerate(optionals):
            if not is_default:
                last = i
        tokens = list(required) + [tok for tok, _ in optionals[:last + 1]]
        fh.write(keyword + " " + " ".join(tokens) + "\n")

    def _emit_field_map_card(fh, elem, emitted_fmp):
        """Emit one FIELD_MAP card (with the stateful FIELD_MAP_PATH
        machinery); returns the updated ``emitted_fmp`` state.  Shared
        by plain maps and SUPERPOSE cluster children."""
        geom = getattr(elem, "geom", None)
        field_file = getattr(elem, "field_file", None)
        if geom is None or not field_file:
            # No .dat provenance (e.g. a programmatically-built map):
            # we can't reconstruct a loadable card, so leave a comment
            # rather than emit a broken one.
            fh.write(
                f"; FIELD_MAP (element '{elem.name}') — no source "
                f"file recorded on the element, cannot re-export\n"
            )
            return emitted_fmp
        # Relativize the map location against the OUTPUT .dat's
        # directory so exported lattices are portable.  All maps
        # sharing a directory get ONE stateful FIELD_MAP_PATH card +
        # bare names; the directive is re-emitted whenever the
        # directory changes.
        map_dir, map_name = os.path.split(
            os.path.abspath(str(field_file)))
        rel_dir, rel_ok = best_relpath(map_dir, out_dir)
        if not rel_ok:
            if map_dir not in warned_map_dirs:
                warned_map_dirs.add(map_dir)
                warnings.warn(
                    f"FIELD_MAP field files under {map_dir!r} "
                    f"share no usable ancestor with the output "
                    f"directory {out_dir!r} — writing absolute "
                    "paths; the exported .dat will not be "
                    "relocatable.",
                    UserWarning,
                    stacklevel=2,
                )
            ff = str(field_file)
        elif map_dir == out_dir and emitted_fmp is None:
            ff = map_name
        else:
            if map_dir != emitted_fmp:
                fmp_tok = rel_dir
                if any(c.isspace() for c in fmp_tok):
                    fmp_tok = f'"{fmp_tok}"'
                fh.write(f"FIELD_MAP_PATH {fmp_tok}\n")
                emitted_fmp = map_dir
            ff = map_name
        if any(c.isspace() for c in ff):
            ff = f'"{ff}"'
        required = [
            str(int(geom)),
            _fmt(elem.length),
            _fmt(elem.phase),
            _fmt(elem.aperture),
            _fmt(elem.kb),
            _fmt(elem.ke),
            _fmt(getattr(elem, "ki", 0.0)),
            str(int(getattr(elem, "ka", 1))),
            ff,
        ]
        p_flag = int(getattr(elem, "p_flag", 0) or 0)
        optionals = [(str(p_flag), p_flag == 0)]
        _emit_card(fh, "FIELD_MAP", required, optionals)
        return emitted_fmp

    # latin-1 keeps written decks byte-compatible with TraceWin and
    # round-trips exactly through our latin-1 readers; a stray char
    # outside latin-1 degrades to "?" instead of crashing the writer.
    with open(filepath, "w", encoding="latin-1", errors="replace") as fh:

        # Optional header FREQ
        if frequency is not None:
            fh.write(f"FREQ {_fmt(frequency)}\n")
            current_freq = frequency

        for elem in lattice.elements:

            # Skip Marker(frequency_MHz=…) — the FREQ card is auto-emitted
            # below from the next RF element's `frequency`, so writing both
            # would duplicate.  Real (non-frequency) Markers fall through.
            if isinstance(elem, Marker) and getattr(elem, "frequency_MHz", 0.0) > 0:
                continue

            # Freq command (parsed FREQ card): emit it here and update
            # ``current_freq`` so the auto-emit below does not write a
            # duplicate card before the next RF element.  Skip when the
            # value is already current (e.g. an explicit ``frequency=``
            # header argument wrote it) so a parse→write→parse round-trip
            # does not grow a second identical card.
            if isinstance(elem, FreqCommand):
                if elem.frequency_mhz > 0 and elem.frequency_mhz != current_freq:
                    fh.write(f"FREQ {_fmt(elem.frequency_mhz)}\n")
                    current_freq = elem.frequency_mhz
                continue

            # ── Write FREQ card if this element has an RF frequency ────────
            elem_freq = getattr(elem, "frequency", None)
            if elem_freq is not None and elem_freq != 0 and elem_freq != current_freq:
                fh.write(f"FREQ {_fmt(elem_freq)}\n")
                current_freq = elem_freq

            # ── Element cards ──────────────────────────────────────────────
            if isinstance(elem, Drift):
                # Schema: L R [Ry] [Rx_shift] [Ry_shift]
                # Ry defaults to None (circular); only emit if set.  If we
                # emit Ry and any shift is non-zero we must also emit both
                # shifts so positions stay aligned.
                required = [_fmt(elem.length), _fmt(elem.aperture)]
                optionals = []
                if elem.aperture_y is not None:
                    optionals.append((_fmt(elem.aperture_y), False))
                    if elem.x_shift or elem.y_shift:
                        optionals.append((_fmt(elem.x_shift),
                                          elem.x_shift == 0.0))
                        optionals.append((_fmt(elem.y_shift),
                                          elem.y_shift == 0.0))
                _emit_card(fh, "DRIFT", required, optionals)

            elif isinstance(elem, Quadrupole):
                # Schema: L G R [Θ] [G3] [G4] [G5] [G6] [GFR]
                required = [_fmt(elem.length), _fmt(elem.gradient),
                            _fmt(elem.aperture)]
                trailing = [elem.skew_angle, elem.g3, elem.g4,
                            elem.g5, elem.g6, elem.gfr]
                optionals = [(_fmt(v), v == 0.0) for v in trailing]
                _emit_card(fh, "QUAD", required, optionals)

            elif isinstance(elem, Solenoid):
                # Schema: L B R — n_steps has no .dat slot.
                fh.write(
                    f"SOLENOID {_fmt(elem.length)} {_fmt(elem.field)} "
                    f"{_fmt(elem.aperture)}\n"
                )

            elif isinstance(elem, RFGap):
                # Schema: E0TL(V) phi_s(deg) R [P]
                # The element stores voltage in MV; TraceWin expects volts.
                # TraceWin absorbs the TTF into E0TL on the card, so we
                # fold ttf into E0TL on write (otherwise the effective
                # voltage would not round-trip).
                e0tl_volts = elem.voltage * 1e6 * elem.ttf
                required = [_fmt(e0tl_volts), _fmt(elem.phase),
                            _fmt(elem.aperture)]
                p_flag = int(getattr(elem, "p_flag", 0) or 0)
                optionals = [(str(p_flag), p_flag == 0)]
                _emit_card(fh, "GAP", required, optionals)

            elif isinstance(elem, Dipole):
                # Schema: angle rho [N] [R] [HV]
                # Pole-face (edge) angles are represented by separate EDGE
                # cards in TraceWin, so non-zero e1/e2 on the Dipole do NOT
                # round-trip through a single BEND card — users must emit
                # EDGE ... BEND ... EDGE explicitly if edge focusing is
                # required.
                required = [_fmt(elem.angle), _fmt(elem.rho)]
                trailing = [
                    (_fmt(elem.field_index), elem.field_index == 0.0),
                    (_fmt(elem.aperture),    elem.aperture == 0.0),
                    (str(int(elem.hv)),      int(elem.hv) == 0),
                ]
                _emit_card(fh, "BEND", required, trailing)

            elif isinstance(elem, Edge):
                # Schema: β rho [G] [K1] [K2] [R] [HV]
                # K1/K2 TraceWin defaults: 0.45 / 2.80
                required = [_fmt(elem.pole_rotation), _fmt(elem.rho)]
                trailing = [
                    (_fmt(elem.gap),              elem.gap == 0.0),
                    (_fmt(elem.k1),               elem.k1 == 0.45),
                    (_fmt(elem.k2),               elem.k2 == 2.80),
                    (_fmt(elem.aperture_radius),  elem.aperture_radius == 0.0),
                    (str(int(elem.hv)),           int(elem.hv) == 0),
                ]
                _emit_card(fh, "EDGE", required, trailing)

            elif isinstance(elem, Steerer):
                # Schema: bl_x bl_y [R] [elec].  Both are required.  The
                # Linac_Gen element does not track aperture, so the R
                # slot is emitted (as 0) only when the electric flag
                # needs the fourth position.
                if getattr(elem, "elec", False):
                    fh.write(f"THIN_STEERING {_fmt(elem.bx_l)} "
                             f"{_fmt(elem.by_l)} 0 1\n")
                else:
                    fh.write(f"THIN_STEERING {_fmt(elem.bx_l)} "
                             f"{_fmt(elem.by_l)}\n")

            elif isinstance(elem, Aperture):
                # Schema: dx dy n.  All three required for a clean round-
                # trip (the type flag disambiguates circle vs rectangle).
                fh.write(
                    f"APERTURE {_fmt(elem.dx)} {_fmt(elem.dy)} "
                    f"{int(elem.aperture_type)}\n"
                )

            elif isinstance(elem, LatticeCommand):
                # SET_*/ADJUST_* round-trip: dump KEYWORD + positional args
                # produced by the command's own ``to_tracewin_args``.
                args = elem.to_tracewin_args()
                if args:
                    fh.write(f"{elem.KEYWORD} " + " ".join(args) + "\n")
                else:
                    fh.write(f"{elem.KEYWORD}\n")

            elif isinstance(elem, Marker):
                name = getattr(elem, "name", "") or ""
                card_args = getattr(elem, "lattice_card_args", None)
                if re.match(r"^LATTICE_\d+$", name) and card_args:
                    # Periodicity bracket (from a LATTICE card or the MAD8
                    # importer) — round-trip it, else the declared period
                    # structure is silently lost on resave.
                    fh.write("LATTICE " + " ".join(
                        _fmt(a) for a in card_args) + "\n")
                elif re.match(r"^LATTICE_END_\d+$", name):
                    fh.write("LATTICE_END\n")
                elif elem.snapshot:
                    fh.write("DIAG_PHASE 1\n")
                elif getattr(elem, "is_bpm", False):
                    fam = getattr(elem, "diag_family", None)
                    if fam is not None:
                        # DIAG_POSITION N [X Y] [dm] — None planes emit the
                        # TraceWin 1e50 "unconstrained" sentinel; X/Y are
                        # omitted entirely when both are absent.
                        x_t = getattr(elem, "x_target_mm", None)
                        y_t = getattr(elem, "y_target_mm", None)
                        toks = [str(int(fam))]
                        if x_t is not None or y_t is not None:
                            toks.append(_fmt(x_t) if x_t is not None
                                        else "1e50")
                            toks.append(_fmt(y_t) if y_t is not None
                                        else "1e50")
                        dm = float(getattr(elem, "accuracy_mm", 1.0) or 1.0)
                        if dm != 1.0:
                            if len(toks) == 1:
                                toks += ["1e50", "1e50"]
                            toks.append(_fmt(dm))
                        # Deck labels re-parse through the label branches;
                        # suppress auto-generated BPM_### names so unlabeled
                        # decks round-trip without cosmetic labels, and only
                        # emit labels the parser's grammar can re-read
                        # (leading letter, no spaces/colons) — anything else
                        # would corrupt the card on reload.
                        emit_label = (
                            name
                            and not re.match(r"^BPM_\d+", name)
                            and re.match(r"^[A-Za-z][^\s:]*$", name))
                        prefix = f"{name}: " if emit_label else ""
                        fh.write(prefix + "DIAG_POSITION "
                                 + " ".join(toks) + "\n")
                    elif getattr(elem, "origin_keyword", None) == "BPM":
                        fh.write("BPM\n")
                    else:
                        fh.write("DIAG_POSITION\n")
                else:
                    fh.write("MARKER\n")

            elif isinstance(elem, SpaceChargeComp):
                fh.write(f"SPACE_CHARGE_COMP {_fmt(elem.factor)}\n")

            elif isinstance(elem, RfqCell):
                # Schema: RFQ_CELL  V[V]  Ro[mm]  A10  m  L[mm]  θs[deg]  Type [Tc] [dP]
                required = [
                    _fmt(elem.voltage_V),
                    _fmt(elem.r0_mm),
                    _fmt(elem.A10),
                    _fmt(elem.modulation),
                    _fmt(elem.length),       # length is in mm internally
                    _fmt(elem.phi_s_deg),
                    str(int(elem.cell_type)),
                ]
                trailing = [
                    (_fmt(elem.Tc_mm), elem.Tc_mm == 0.0),
                    (_fmt(elem.dP_deg), elem.dP_deg == 0.0),
                ]
                _emit_card(fh, "RFQ_CELL", required, trailing)

            elif isinstance(elem, VaneRFQ):
                # Reconstruct the original RFQ_CELL chain from the cell list.
                # The .vane file itself is not re-emitted (parser owns its
                # path); a comment flags this so users notice they need to
                # re-attach it before re-running the resulting .dat.
                fh.write(
                    f"; VaneRFQ '{elem.name}' reconstructed from cell list "
                    f"({len(elem.cells)} cells) — re-attach the .vane "
                    f"file to drive the geometry\n"
                )
                for c in elem.cells:
                    required = [
                        _fmt(c.voltage_V),
                        _fmt(c.r0_dat_mm),
                        _fmt(c.A10),
                        _fmt(c.modulation),
                        _fmt(c.length_mm),
                        _fmt(c.phi_s_deg),
                        str(int(c.cell_type)),
                    ]
                    trailing = [
                        (_fmt(c.Tc_mm),  c.Tc_mm  == 0.0),
                        (_fmt(c.dP_deg), c.dP_deg == 0.0),
                    ]
                    _emit_card(fh, "RFQ_CELL", required, trailing)

            elif isinstance(elem, NCells):
                # Schema: mode Nc betaG EoT thetaS R [P kEoTi kEoTo dzi dzo]
                #         [betaS Ts kT's k2T''s Ti kT'i k2T''i To kT'o k2T''o]
                required = [
                    str(int(elem.mode)),
                    str(int(elem.n_cells)),
                    _fmt(elem.beta_g),
                    _fmt(elem.eot_v_per_m),
                    _fmt(elem.theta_s_deg),
                    _fmt(elem.aperture),
                ]
                trailing = [
                    (str(int(elem.p_flag)), elem.p_flag == 0),
                    (_fmt(elem.k_eot_i), elem.k_eot_i == 0.0),
                    (_fmt(elem.k_eot_o), elem.k_eot_o == 0.0),
                    (_fmt(elem.dz_i_mm), elem.dz_i_mm == 0.0),
                    (_fmt(elem.dz_o_mm), elem.dz_o_mm == 0.0),
                ]
                # βs≠0 transit-time-factor tail (non-default → always emitted).
                if elem._ttf is not None:
                    t = elem._ttf
                    for v in (t.beta_s,
                              t.middle.Ts, t.middle.kTp, t.middle.k2Tpp,
                              t.input.Ts, t.input.kTp, t.input.k2Tpp,
                              t.output.Ts, t.output.kTp, t.output.k2Tpp):
                        trailing.append((_fmt(v), False))
                _emit_card(fh, "NCELLS", required, trailing)

            elif isinstance(elem, SuperposedFieldMap):
                # TraceWin cluster: SHIFT_IN_FIELD_MAP + diagnostic per
                # interior marker (verbatim card provenance), then
                # SUPERPOSE_MAP z0 + FIELD_MAP card per child, in
                # ORIGINAL card order (any order is valid per the
                # manual; preserving it keeps round-trips
                # byte-idempotent).  The shared FREQ context was emitted
                # by the generic check above (the container's
                # ``frequency`` property returns the single RF frequency
                # enforced at construction).
                for dz_mk, mk in getattr(elem, "interior_markers", []):
                    fh.write(f"SHIFT_IN_FIELD_MAP {_fmt(dz_mk)}\n")
                    kw_mk = getattr(mk, "origin_keyword", None) or "MARKER"
                    ps_mk = " ".join(getattr(mk, "origin_params", [])
                                     or [])
                    fh.write(kw_mk + ((" " + ps_mk) if ps_mk else "")
                             + "\n")
                if getattr(elem, "_from_plain_wrap", False):
                    # Container created only to host SHIFT diagnostics
                    # around a plain FIELD_MAP card — the source deck had
                    # no SUPERPOSE_MAP line, so don't invent one.
                    for _z0, child in elem.children:
                        emitted_fmp = _emit_field_map_card(fh, child,
                                                           emitted_fmp)
                else:
                    for z0, child in elem.children:
                        fh.write(f"SUPERPOSE_MAP {_fmt(z0)}\n")
                        emitted_fmp = _emit_field_map_card(fh, child,
                                                           emitted_fmp)

            elif isinstance(elem, (FieldMap, FieldMap3D)):
                # Schema: geom L phase R kb ke ki ka filename [p_flag]
                # (kb/ke/phase from the element's CURRENT values so
                # matched amplitudes/phases persist; geom + field_file
                # are parser provenance).  A single-child z0=0 cluster
                # re-parses as the plain element carrying superpose_z0
                # provenance — re-emit its SUPERPOSE_MAP line.
                sz0 = getattr(elem, "superpose_z0", None)
                if sz0 is not None:
                    fh.write(f"SUPERPOSE_MAP {_fmt(sz0)}\n")
                emitted_fmp = _emit_field_map_card(fh, elem, emitted_fmp)

            elif isinstance(elem, Foil):
                # HELIX-specific extension; TraceWin parses this line as a
                # plain comment and ignores it.  On re-import HELIX
                # reconstructs the Foil via :func:`parse_tracewin`'s
                # ``_HELIX_FOIL`` regex.
                # Optional 4th token: non-default straggling model; the
                # default ("auto") is omitted so existing files round-trip
                # byte-identically.
                strag = getattr(elem, "straggling", "auto")
                strag_tok = f" {strag}" if strag != "auto" else ""
                fh.write(
                    f"; HELIX_FOIL {elem.name} {elem.material} "
                    f"{elem.thickness_ug_cm2}{strag_tok}\n"
                )

            elif isinstance(elem, ScGridDirective):
                # HELIX-specific comment card (TraceWin ignores it); the
                # parser's _HELIX_SC_GRID regex reconstructs the element.
                fh.write(f"; HELIX_SC_GRID {elem.extent_sigma:g}\n")

            else:
                # Unknown element: write informative comment, do not crash
                fh.write(
                    f"; unsupported element type: {type(elem).__name__} "
                    f"('{elem.name}')\n"
                )

        fh.write("END\n")
