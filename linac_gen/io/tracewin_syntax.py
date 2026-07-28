"""Positional argument schemas for TraceWin .dat cards.

Each entry in :data:`SCHEMA` is a list of :class:`Field` tuples describing,
in order, the parameter name, its Python type, whether it is required,
and its default when missing.  :func:`parse_positionals` converts a list
of string tokens into a kwargs dict ready to hand to an element
constructor or to a higher-level parser branch.

Entries whose value is ``None`` are recognised keywords that do not use
positional-schema parsing (control cards handled directly in the
dispatcher, e.g. FREQ / PARTRAN_STEP / TITLE / END).
"""
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class Field:
    name: str
    cast: Callable[[str], Any]
    required: bool = False
    default: Any = None


def parse_positionals(fields: list[Field], tokens: list[str]) -> dict:
    """Turn ``tokens`` into ``{name: value}`` per ``fields``.

    Raises ValueError if a required field is missing or a cast fails.
    Extra trailing tokens are ignored (the caller decides whether to warn).
    """
    required_min = sum(1 for f in fields if f.required)
    if len(tokens) < required_min:
        names = ", ".join(f.name for f in fields if f.required)
        raise ValueError(
            f"card requires at least {required_min} positional args "
            f"({names}); got {len(tokens)}"
        )
    out: dict = {}
    for i, field in enumerate(fields):
        if i < len(tokens):
            try:
                out[field.name] = field.cast(tokens[i])
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    f"failed to cast {field.name}={tokens[i]!r} via "
                    f"{field.cast.__name__}: {exc}"
                ) from exc
        else:
            out[field.name] = field.default
    return out


# --------------------------------------------------------------------------
# Per-card schemas (TraceWin-ordering, authoritative)
# --------------------------------------------------------------------------

SCHEMA: dict[str, Optional[list[Field]]] = {
    "DRIFT": [
        Field("length",     float, required=True),
        Field("aperture",   float, default=0.0),
        Field("aperture_y", float, default=None),   # None => circular
        Field("x_shift",    float, default=0.0),
        Field("y_shift",    float, default=0.0),
    ],
    "QUAD": [
        Field("length",     float, required=True),
        Field("gradient",   float, required=True),
        Field("aperture",   float, default=0.0),
        Field("skew_angle", float, default=0.0),    # degrees
        Field("g3",         float, default=0.0),    # sextupole component
        Field("g4",         float, default=0.0),    # octupole
        Field("g5",         float, default=0.0),    # decapole
        Field("g6",         float, default=0.0),    # dodecapole
        Field("gfr",        float, default=0.0),    # good-field radius
    ],
    "SOLENOID": [
        Field("length",   float, required=True),
        Field("field",    float, required=True),
        Field("aperture", float, default=0.0),
    ],
    "GAP": [
        Field("e0tl",     float, required=True),     # effective gap voltage (V)
        Field("phase",    float, required=True),     # deg
        Field("aperture", float, default=0.0),
        Field("p_flag",   int,   default=0),         # 0 relative, 1 absolute, 2,3 variants
    ],
    "FIELD_MAP": [
        Field("geom",      int,   required=True),    # 5-digit geom code
        Field("length",    float, required=True),    # mm
        Field("phase",     float, default=0.0),      # deg
        Field("aperture",  float, default=0.0),
        Field("kb",        float, default=1.0),      # magnetic scale
        Field("ke",        float, default=1.0),      # electric scale
        Field("ki",        float, default=0.0),      # SC compensation
        Field("ka",        int,   default=1),        # aperture flag
        Field("filename",  str,   required=True),
        Field("p_flag",    int,   default=0),
    ],
    "FIELD_MAP_PATH": [
        Field("path", str, required=True),     # abs or relative to .dat dir
    ],
    "BEND": [
        Field("angle",       float, required=True),  # deg
        Field("rho",         float, required=True),  # mm
        Field("field_index", float, default=0.0),
        Field("aperture",    float, default=0.0),
        Field("hv",          int,   default=0),      # 0=horizontal, 1=vertical
    ],
    "EDGE": [
        Field("pole_rotation", float, required=True),  # deg
        Field("rho",           float, required=True),  # mm
        Field("gap",           float, default=0.0),    # mm
        Field("k1",            float, default=0.45),   # fringe
        Field("k2",            float, default=2.80),   # fringe 2
        Field("aperture",      float, default=0.0),
        Field("hv",            int,   default=0),
    ],
    "APERTURE": [
        Field("dx",           float, required=True),  # mm half-width or radius
        Field("dy",           float, default=0.0),    # mm half-width (or separator for pepperpot)
        Field("ap_type",      int,   default=0),      # 0 rect, 1 circle, 2 pepperpot, 3 fraction, 4/5 finger, 6 ring
    ],
    "STEERER": [   # aka THIN_STEERING
        Field("bl_x",   float, required=True),    # T.m  (or V if elec=1)
        Field("bl_y",   float, required=True),
        Field("aperture", float, default=0.0),
        Field("elec",   int,   default=0),         # 0 magnetic, 1 electric
    ],
    # Control cards -- no positional typing.  Listed so the parser can
    # introspect "do we know this keyword".
    "THIN_STEERING": None,
    "FREQ":          None,
    "LATTICE":       None,
    "LATTICE_END":   None,
    "PARTRAN_STEP":  None,
    "TITLE":         None,
    "DIAG_PHASE":    None,
    "DIAG_SIZE":     None,
    "DIAG_POSITION": None,
    "REPEAT_ELE":    None,
    "END":           None,

    # ------------------------------------------------------------------
    # SET / ADJUST family (TraceWin matching language, ref.
    # ``linac_gen.elements.lattice_commands``).  Each parses through
    # ``parse_positionals``; the parser dispatcher in ``tracewin_parser``
    # constructs the corresponding ``LatticeCommand`` subclass.
    # ------------------------------------------------------------------
    "SET_SYNC_PHASE": [],   # no args
    "SET_BEAM_PHASE_ERROR": [
        Field("dphi_deg",     float, default=0.0),
        Field("random_flag",  int,   default=0),
    ],
    "SET_BEAM_E0_P0": [
        Field("k",        int,   default=0),
        Field("dE_MeV",   float, default=0.0),
        Field("dphi_deg", float, default=0.0),
        Field("ke",       int,   default=0),
        Field("kp",       int,   default=0),
    ],
    "SET_BEAM_ENERGY": [
        Field("k",          int,   default=0),
        Field("energy_MeV", float, required=True),
    ],
    "SET_GAUSSIAN_CUT_OFF": [
        Field("sigma", float, default=4.0),
    ],
    "SET_TWISS": [
        Field("family",  str,   default=""),
        Field("alpha_x", float, default=0.0),
        Field("beta_x",  float, default=0.0),
        Field("alpha_y", float, default=0.0),
        Field("beta_y",  float, default=0.0),
        Field("alpha_z", float, default=0.0),
        Field("beta_z",  float, default=0.0),
        Field("kax", int, default=0), Field("kbx", int, default=0),
        Field("kay", int, default=0), Field("kby", int, default=0),
        Field("kaz", int, default=0), Field("kbz", int, default=0),
    ],
    "SET_POSITION": [
        Field("k",        float, default=0.0),
        Field("x_mm",     float, default=0.0),
        Field("xp_mrad",  float, default=0.0),
        Field("y_mm",     float, default=0.0),
        Field("yp_mrad",  float, default=0.0),
    ],
    "SET_ACHROMAT": [
        Field("k",     int, default=0),
        Field("f1",    int, default=0),
        Field("f2",    int, default=0),
        Field("plane", int, default=0),
    ],
    "SET_SIZE": [
        Field("k",        float, default=0.0),
        Field("x_mm",     float, default=0.0),
        Field("y_mm",     float, default=0.0),
        Field("phi_or_z", float, default=0.0),
        Field("k2",       int,   default=0),
    ],
    "SET_SIZE_MAX": [
        Field("k",        float, default=0.0),
        Field("n_elems",  int,   default=1),
        Field("x_mm",     float, default=0.0),
        Field("y_mm",     float, default=0.0),
        Field("phi_or_z", float, default=0.0),
        Field("k2",       int,   default=0),
    ],
    "SET_SIZE_MIN": [
        Field("k",        float, default=0.0),
        Field("n_elems",  int,   default=1),
        Field("x_mm",     float, default=0.0),
        Field("y_mm",     float, default=0.0),
        Field("phi_or_z", float, default=0.0),
        Field("k2",       int,   default=0),
    ],
    "SET_BEAM_PHASE_ADV": [
        Field("k",         float, default=0.0),
        Field("n_elems",   int,   default=1),
        Field("mu_x_deg",  float, default=0.0),
        Field("mu_y_deg",  float, default=0.0),
        Field("mu_z_deg",  float, default=0.0),
    ],
    "SET_SEPARATION": [
        Field("k",  float, default=0.0),
        Field("sx", float, default=0.0),
        Field("sy", float, default=0.0),
    ],
    "SET_ADV": [
        Field("kxot", float, default=0.0),
        Field("kyot", float, default=0.0),
    ],
    "MIN_EMIT_GROWTH": [
        Field("plane",  str,   required=True),   # 'X' | 'Y' | 'Z'
        Field("weight", float, default=1.0),
    ],
    "MIN_EMIT_4D_GROWTH": [
        Field("weight", float, default=1.0),
        Field("tol_4d", float, default=1.0),     # >=1.0; 1.1 = allow 10%
        Field("tol_z",  float, default=1.0),
    ],
    "SET_KE_OUT_MIN": [
        Field("energy_mev", float, required=True),
        Field("weight",     float, default=1.0),
    ],
    "MIN_TRANSMISSION": [
        Field("threshold_pct", float, default=99.0),
        Field("weight",        float, default=1.0),
    ],
    "ADJUST": [
        Field("target",     str,   required=True),
        Field("param_idx",  int,   required=True),
        Field("link_group", int,   default=0),
        Field("vmin",       float, default=0.0),
        Field("vmax",       float, default=0.0),
        Field("start_step", float, default=0.0),
        Field("kn",         int,   default=0),
    ],
    "ADJUST_STEERER": [
        Field("diag_n",     int,   required=True),
        Field("vmax",       float, default=0.0),
        Field("first_step", float, default=0.0),
    ],
    "ADJUST_STEERER_BX": [
        Field("diag_n",     int,   required=True),
        Field("vmax",       float, default=0.0),
        Field("first_step", float, default=0.0),
    ],
    "ADJUST_STEERER_BY": [
        Field("diag_n",     int,   required=True),
        Field("vmax",       float, default=0.0),
        Field("first_step", float, default=0.0),
    ],
    # ADJUST_BEAM_* are handled by a special-case branch in the parser
    # (variable-length flag tail); declare with None to mark "known".
    "ADJUST_BEAM_TWISS":    None,
    "ADJUST_BEAM_CENTROID": None,
    "ADJUST_BEAM_EMIT":     None,
    "ADJUST_BEAM_CURRENT":  None,
}
