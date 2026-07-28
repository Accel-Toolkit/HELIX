"""Design tokens + QSS theme for Interphase.

Mirrors ``interphase/project/styles.css`` from the Claude Design mockup.
Every CSS variable is exposed here as a module constant; ``dark_qss()``
returns the full stylesheet string with tokens substituted.
"""
from __future__ import annotations


# --- Design tokens (hex) -----------------------------------------------------
ACCENT      = "#22d3ee"
ACCENT_2    = "#67e8f9"
ACCENT_DIM  = "#0e7490"
ACCENT_GLOW = "rgba(34, 211, 238, 0.18)"

BG_0      = "#0b0f15"   # window bg (slightly lifted from near-black)
BG_1      = "#11161f"   # panel bg
BG_2      = "#1a2230"   # elevated card (raised one notch above panel)
BG_3      = "#222d3d"   # cards (hover/selected)
BG_4      = "#2c394c"   # hover
BG_INSET  = "#0c1119"   # sunken wells (kept dark for non-input use)
BG_FIELD  = "#26334a"   # value/parameter field bg — slightly *lighter* than panels

# Borders — kept *bright* relative to BG so fields and panel separators
# read clearly on dark backgrounds.  Targets WCAG AA 3:1 against BG_1.
BORDER_0  = "#3a4a5e"   # panel separators
BORDER_1  = "#5b6f87"   # input borders (the box around every field)
BORDER_2  = "#7286a0"   # emphasised borders / focus rims

# Text colours — recalibrated for ≥4.5:1 contrast on BG_0/BG_1.
TEXT_0    = "#f8fafc"   # primary (near-white)
TEXT_1    = "#dbe2eb"   # secondary
TEXT_2    = "#b3becc"   # tertiary — units, status lines (was 94a3b8)
TEXT_3    = "#a1aebe"   # section headers, KPI labels (was 8595a8)
TEXT_DIM  = "#6f7d92"   # placeholder / disabled

# Subtle tint applied to section title backgrounds for visual lift.
SECTION_BG = "#161d29"

# Semantic
OK    = "#4ade80"
WARN  = "#fbbf24"
ERR   = "#f87171"
INFO  = "#60a5fa"
MAG   = "#e879f9"
PINK  = "#f472b6"

# Plot palette
PLOT_PALETTE = ["#22d3ee", "#a3e635", "#fbbf24", "#f472b6", "#a78bfa", "#fb923c"]

# Element type colours (lattice timeline)
EL_COLORS = {
    "Drift":      "#6b7280",
    "Quadrupole": "#22d3ee",
    "Dipole":     "#fb923c",
    "RFGap":      "#a3e635",
    "FieldMap":   "#a3e635",
    "FieldMap3D": "#22d3ee",  # cyan — distinguishes 3-D maps from 1-D/2-D
    "Solenoid":   "#a78bfa",
    "BPM":        "#f472b6",
    "Aperture":   "#f87171",
    "Marker":     "#e5e7eb",
    "Sextupole":  "#84cc16",
    "Octupole":   "#65a30d",
    "ThinLens":   "#22d3ee",
    "Edge":       "#fb923c",
    "RfqCell":    "#67e8f9",
    "VaneRFQ":    "#67e8f9",
    "SpaceChargeComp": "#fbbf24",
    "Steerer":    "#f472b6",
    "Foil":       "#e94f37",   # red-orange — distinct from Drift gray and Aperture pink
}

# Fonts
FONT_SANS = "Inter, 'IBM Plex Sans', 'Segoe UI', sans-serif"
FONT_MONO = "'JetBrains Mono', 'IBM Plex Mono', 'Cascadia Code', Consolas, monospace"
FONT_SIZE = 12


# Chevron icon paths (used by the QSpinBox::up-arrow / down-arrow rules in
# ``dark_qss`` below).  PNG assets live alongside this module; Qt's QSS
# parser handles absolute file paths reliably across macOS / Linux /
# Windows (unlike data-URLs which the parser sometimes silently drops).
import os as _os
_ASSETS_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "assets")
_CHEVRON_UP         = _os.path.join(_ASSETS_DIR, "chevron-up.png")
_CHEVRON_DOWN       = _os.path.join(_ASSETS_DIR, "chevron-down.png")
_CHEVRON_UP_HOVER   = _os.path.join(_ASSETS_DIR, "chevron-up-hover.png")
_CHEVRON_DOWN_HOVER = _os.path.join(_ASSETS_DIR, "chevron-down-hover.png")


def dark_qss(base: int | None = None) -> str:
    """Return the full application stylesheet.

    Uses Qt's object-name selectors: widgets get ``setObjectName('titlebar')``
    and the stylesheet targets them as ``QFrame#titlebar``.  Generic widget
    styling (buttons, inputs, scrollbars) is handled on the base class.

    ``base`` overrides the default ``FONT_SIZE`` so the user can scale the
    whole app via a spinbox.  Tuned font sizes elsewhere in the sheet are
    kept in proportion to ``base``.
    """
    bs = int(base) if base else FONT_SIZE
    # Scale points: base ± offset. Floor at 8 so micro-labels never disappear.
    fs_p2  = bs + 2
    fs_p6  = bs + 6     # KPI value rendering
    fs_p8  = bs + 8     # display-size headers
    fs_m1  = max(8, bs - 1)   # tabs / inspector / table fields
    fs_m2  = max(8, bs - 2)   # chrome paths, statusbar, micro-labels
    fs_m3  = max(8, bs - 3)   # KPI label caps
    # Input-field padding scales with the base font so QLineEdit /
    # QSpinBox / QComboBox don't clip their own text when the user
    # bumps the global font size.  Vertical padding is generous enough
    # to cover Qt 6 macOS' QStyleSheetStyle text-positioning quirk
    # (text baseline lands below the un-padded clip rect).
    # 12 pt → 5 / 8 (just above the original look), 16 pt → 7 / 11,
    # 20 pt → 9 / 14.
    pad_v  = max(5, int(bs * 0.45))
    pad_h  = max(8, int(bs * 0.7))
    # Keep f-string injection limited to hex colour / size constants
    # to avoid any brace-escaping gymnastics.
    return f"""
/* ========= GLOBAL ========= */
QWidget {{
    background: {BG_0};
    color: {TEXT_0};
    font-family: {FONT_SANS};
    font-size: {bs}px;
    selection-background-color: {ACCENT};
    selection-color: #00161c;
}}
QToolTip {{
    background: {BG_4};
    color: {TEXT_0};
    border: 1px solid {BORDER_2};
    padding: 4px 8px;
    font-size: {fs_m1}px;
}}

/* ========= SCROLLBARS ========= */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_1};
    border-radius: 5px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{ background: {BORDER_2}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER_1};
    border-radius: 5px;
    min-width: 20px;
}}
QScrollBar::handle:horizontal:hover {{ background: {BORDER_2}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ========= CHROME FRAMES ========= */
QFrame#titlebar {{
    background: {BG_1};
    border-bottom: 1px solid {BORDER_0};
    /* Height is owned by TitleBar.apply_font_size (setFixedHeight) —
       QSS geometry here is re-applied on every repolish and would
       overwrite the font-derived value, clipping the text. */
}}
QFrame#titlebar QLabel#appname {{
    color: {TEXT_0};
    font-weight: 600;
    letter-spacing: 1.5px;
}}
QFrame#titlebar QLabel#appaccent {{ color: {ACCENT}; font-weight: 600; letter-spacing: 1.5px; }}
QFrame#titlebar QLabel.path {{ color: {TEXT_2}; font-family: {FONT_MONO}; font-size: {fs_m2}px; }}
QFrame#titlebar QLabel.path-b {{ color: {TEXT_0}; font-weight: 500; }}
QFrame#titlebar QLabel.dim {{ color: {TEXT_3}; }}
QFrame#titlebar QLabel.meta {{ color: {TEXT_2}; font-family: {FONT_MONO}; font-size: {fs_m2}px; }}

QFrame#menubar {{
    background: {BG_1};
    border-bottom: 1px solid {BORDER_0};
    /* Height owned by Toolbar.apply_font_size — see #titlebar note.
       (The old 38px max-height contradicted the toolbar's own
       setFixedHeight(max(44, base+30)) even at the default font.) */
}}
QFrame#menubar QPushButton {{
    background: transparent;
    border: 0;
    color: {TEXT_1};
    padding: 3px 10px;
    border-radius: 3px;
    font-size: {fs_m1}px;
}}
QFrame#menubar QPushButton:hover {{ background: {BG_3}; color: {TEXT_0}; }}

QFrame#menubar QPushButton.btn {{
    border: 1px solid {BORDER_1};
    background: {BG_2};
    color: {TEXT_0};
    padding: 4px 10px;
}}
QFrame#menubar QPushButton.btn:hover {{
    border-color: {BORDER_2};
    background: {BG_3};
}}
QFrame#menubar QPushButton.primary {{
    background: {ACCENT};
    color: #00161c;
    border-color: {ACCENT};
    font-weight: 600;
}}
QFrame#menubar QPushButton.primary:hover {{ background: {ACCENT_2}; }}
QFrame#menubar QPushButton.danger {{
    color: {ERR};
    border-color: rgba(248,113,113,0.25);
}}

/* #subbar: dead geometry removed — no widget uses this objectName;
   height pins kept nothing alive and would fight any future owner. */
QFrame#subbar {{
    background: {BG_0};
    border-bottom: 1px solid {BORDER_0};
}}
QFrame#subbar QLabel.crumb {{ color: {TEXT_1}; font-size: {fs_m1}px; }}
QFrame#subbar QLabel.crumb-b {{ color: {TEXT_0}; font-size: {fs_m1}px; font-weight: 500; }}
QFrame#subbar QLabel.crumb-dim {{ color: {TEXT_2}; font-size: {fs_m1}px; }}
QFrame#subbar QLabel.crumb-sel {{ color: {ACCENT}; font-family: {FONT_MONO}; font-size: {fs_m1}px; }}
QFrame#subbar QLabel.scursor {{ color: {TEXT_2}; font-family: {FONT_MONO}; font-size: {fs_m1}px; }}
QFrame#subbar QSlider::groove:horizontal {{
    background: {BORDER_1};
    height: 3px;
    border-radius: 2px;
}}
QFrame#subbar QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-radius: 6px;
}}

QFrame#statusbar {{
    background: qlineargradient(x1:0 y1:0 x2:0 y2:1, stop:0 #0a1420, stop:1 #070c14);
    border-top: 1px solid {BORDER_0};
    color: {TEXT_1};
    /* Height owned by StatusBar.apply_font_size — see #titlebar note. */
    font-family: {FONT_MONO};
    font-size: {fs_m2}px;
}}
QFrame#statusbar QLabel.seg {{
    padding: 0 10px;
    border-right: 1px solid {BORDER_0};
    color: {TEXT_1};
}}
QFrame#statusbar QLabel.seg-acc {{
    background: {ACCENT};
    color: #00161c;
    padding: 0 10px;
    font-weight: 600;
}}
QFrame#statusbar QLabel.seg-warn {{ color: {WARN}; }}

/* ========= MAIN SPLIT ========= */
/* #rail: dead geometry removed — no widget uses this objectName. */
QFrame#rail {{
    background: {BG_1};
    border-right: 1px solid {BORDER_0};
}}
QFrame#rail QPushButton {{
    background: transparent;
    border: 0;
    color: {TEXT_2};
    border-radius: 5px;
    padding: 0;
}}
QFrame#rail QPushButton:hover {{ background: {BG_3}; color: {TEXT_0}; }}
QFrame#rail QPushButton:checked {{
    background: {BG_3};
    color: {ACCENT};
    border-left: 2px solid {ACCENT};
}}

QFrame#sidebar {{
    background: {BG_1};
    border-right: 1px solid {BORDER_0};
    min-width: 180px;
}}
QFrame#sidebar QLabel.section {{
    color: {TEXT_3};
    font-size: {fs_m2}px;
    letter-spacing: 1px;
    padding: 8px 10px 4px;
    font-weight: 600;
}}
QFrame#sidebar QLineEdit {{
    background: {BG_FIELD};
    border: 1px solid {BORDER_1};
    border-radius: 3px;
    color: #ffffff;
    font-family: {FONT_MONO};
    font-size: {fs_m1}px;
    font-weight: 700;
    padding: 4px 8px;
    margin: 2px 8px 6px;
}}
QFrame#sidebar QLineEdit:focus {{ border-color: {ACCENT}; }}

QTreeWidget, QTreeView, QListView {{
    background: {BG_1};
    border: 0;
    font-family: {FONT_MONO};
    font-size: {fs_m1}px;
    color: {TEXT_1};
    outline: 0;
}}
QTreeWidget::item, QTreeView::item {{ padding: 2px 4px; }}
QTreeWidget::item:hover, QTreeView::item:hover {{
    background: {BG_3};
    color: {TEXT_0};
}}
QTreeWidget::item:selected, QTreeView::item:selected {{
    background: rgba(34, 211, 238, 0.18);
    color: {TEXT_0};
}}

QFrame#inspector {{
    background: {BG_1};
    border-left: 1px solid {BORDER_0};
    min-width: 240px;
}}
QFrame#inspector QLabel.inspHead {{
    color: {TEXT_0};
    font-size: {fs_m1}px;
    font-weight: 500;
    padding: 8px 10px;
    border-bottom: 1px solid {BORDER_0};
}}
QFrame#inspector QLabel.sectionh {{
    color: {TEXT_3};
    font-size: {fs_m2}px;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-weight: 600;
    padding: 10px 10px 6px;
}}
QFrame#inspector QLabel.field-lbl {{ color: {TEXT_2}; font-size: {fs_m1}px; }}
QFrame#inspector QLineEdit, QFrame#inspector QDoubleSpinBox, QFrame#inspector QSpinBox,
QFrame#inspector QComboBox {{
    background: {BG_FIELD};
    border: 1px solid {BORDER_1};
    border-radius: 3px;
    color: #ffffff;
    font-family: {FONT_MONO};
    font-size: {fs_m1}px;
    font-weight: 700;
    padding: 3px 6px;
}}
QFrame#inspector QLineEdit:focus, QFrame#inspector QDoubleSpinBox:focus,
QFrame#inspector QSpinBox:focus, QFrame#inspector QComboBox:focus {{
    border-color: {ACCENT};
}}

/* ========= STAGE ========= */
QFrame#stage {{ background: {BG_0}; }}
/* #stageTabs: dead geometry removed — no widget uses this objectName. */
QFrame#stageTabs {{
    background: {BG_1};
    border-bottom: 1px solid {BORDER_0};
}}
QFrame#stageTabs QPushButton {{
    background: transparent;
    border: 0;
    border-right: 1px solid {BORDER_0};
    color: {TEXT_2};
    padding: 0 12px;
    min-height: 30px;
    font-size: {fs_m1}px;
}}
QFrame#stageTabs QPushButton:hover {{ color: {TEXT_0}; background: {BG_2}; }}
QFrame#stageTabs QPushButton:checked {{
    color: {TEXT_0};
    background: {BG_0};
    border-top: 1px solid {ACCENT};
}}

QFrame#stageToolbar {{
    background: {BG_0};
    border-bottom: 1px solid {BORDER_0};
    /* Height owned by the Lattice tab (setFixedHeight) — the old 32px
       max-height clipped its 36px toolbar strip on every repolish. */
}}
QFrame#stageToolbar QPushButton {{
    background: {BG_2};
    border: 1px solid {BORDER_1};
    border-radius: 3px;
    color: {TEXT_1};
    padding: 4px 8px;
    font-size: {fs_m1}px;
}}
QFrame#stageToolbar QPushButton:hover {{
    background: {BG_3};
    color: {TEXT_0};
    border-color: {BORDER_2};
}}
QFrame#stageToolbar QPushButton:checked {{
    background: {BG_3};
    color: {TEXT_0};
    border-color: {ACCENT_DIM};
}}

/* ========= PANELS / CARDS ========= */
QFrame.panel {{
    background: {BG_1};
    border: 1px solid {BORDER_0};
    border-radius: 4px;
}}
QFrame.panel QLabel.ph {{
    color: {TEXT_1};
    font-size: {fs_m2}px;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-weight: 600;
    padding: 6px 10px;
    border-bottom: 1px solid {BORDER_0};
}}
QFrame.kpi {{
    background: {BG_2};
    border: 1px solid {BORDER_0};
    border-radius: 4px;
}}
QFrame.kpi QLabel.kpi-l {{
    color: {TEXT_3};
    font-size: {fs_m3}px;
    letter-spacing: 1px;
    text-transform: uppercase;
}}
QFrame.kpi QLabel.kpi-v {{
    color: {TEXT_0};
    font-family: {FONT_MONO};
    font-size: {fs_p6}px;
    font-weight: 500;
}}
QFrame.kpi QLabel.kpi-u {{ color: {TEXT_2}; font-size: {fs_m1}px; }}
QFrame.kpi QLabel.kpi-d {{ color: {TEXT_2}; font-family: {FONT_MONO}; font-size: {fs_m2}px; }}

/* ========= BOTTOM DOCK ========= */
QFrame#bottomDock {{
    background: {BG_1};
    border-top: 1px solid {BORDER_0};
}}
QFrame#bottomDock QPushButton {{
    background: transparent;
    border: 0;
    color: {TEXT_2};
    padding: 4px 10px;
    font-size: {fs_m1}px;
}}
QFrame#bottomDock QPushButton:hover {{ color: {TEXT_0}; }}
QFrame#bottomDock QPushButton:checked {{
    color: {TEXT_0};
    border-bottom: 1px solid {ACCENT};
}}

/* ========= TWEAK PANEL ========= */
QFrame#tweakPanel {{
    background: {BG_2};
    border: 1px solid {BORDER_2};
    border-radius: 8px;
}}
QFrame#tweakPanel QLabel.tp-t {{ color: {TEXT_0}; font-weight: 600; }}
QFrame#tweakPanel QLabel.tp-l {{
    color: {TEXT_3};
    font-size: {fs_m2}px;
    letter-spacing: 1px;
    text-transform: uppercase;
}}

/* ========= GENERIC INPUTS ========= */
QPushButton {{
    background: {BG_2};
    border: 1px solid {BORDER_1};
    color: {TEXT_1};
    border-radius: 3px;
    padding: 4px 8px;
    font-size: {fs_m1}px;
}}
QPushButton:hover {{ background: {BG_3}; color: {TEXT_0}; border-color: {BORDER_2}; }}
QPushButton:pressed {{ background: {BG_4}; }}
QPushButton:checked {{
    background: rgba(34, 211, 238, 0.18);
    color: {TEXT_0};
    border-color: {ACCENT_DIM};
}}

QLineEdit, QPlainTextEdit, QTextEdit {{
    background: {BG_FIELD};
    border: 1px solid {BORDER_1};
    border-radius: 4px;
    color: #ffffff;
    font-weight: 700;
    padding: {pad_v}px {pad_h}px;
    selection-background-color: {ACCENT};
    selection-color: #00161c;
}}
/* QSpinBox / QDoubleSpinBox / QComboBox on macOS draw their text via
   the native style.  When QSS adds vertical padding the native baseline
   ends up below the styled clip-rect, so the rendered digits get
   top-clipped.  Use horizontal padding only and let the widget's
   native vertical sizing handle the text.  Combined with the global
   font-size scaling via QApplication.setFont, this prevents clipping
   at any base point size. */
QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {BG_FIELD};
    border: 1px solid {BORDER_1};
    border-radius: 4px;
    color: #ffffff;
    font-weight: 700;
    padding-left: {pad_h}px;
    padding-right: {pad_h}px;
    selection-background-color: {ACCENT};
    selection-color: #00161c;
}}
QLineEdit:hover, QDoubleSpinBox:hover, QSpinBox:hover, QComboBox:hover,
QPlainTextEdit:hover, QTextEdit:hover {{
    border-color: {BORDER_2};
}}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {{
    border: 1px solid {ACCENT};
    background: {BG_3};
}}
QLineEdit:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled, QComboBox:disabled {{
    background: {BG_INSET};
    color: {TEXT_DIM};
    font-weight: 400;
    border-color: {BORDER_0};
}}

/* Spin-box steppers — visible up/down chevrons. */
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 16px;
    border-left: 1px solid {BORDER_1};
    background: {BG_2};
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 16px;
    border-left: 1px solid {BORDER_1};
    background: {BG_2};
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {BG_3};
}}
/* Arrow glyphs inside the spin-buttons.  Without these rules the
   QSS-styled button rectangles have no visible glyph on macOS.  PNG
   icons are rendered once into ``assets/`` by ``render_spinbox_chevrons``
   below; the absolute path is substituted at QSS-build time so Qt's
   parser doesn't need to deal with data URLs. */
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url({_CHEVRON_UP});
    width: 10px; height: 6px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url({_CHEVRON_DOWN});
    width: 10px; height: 6px;
}}
QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover {{
    image: url({_CHEVRON_UP_HOVER});
}}
QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover {{
    image: url({_CHEVRON_DOWN_HOVER});
}}

QHeaderView::section {{
    background: {SECTION_BG};
    color: {TEXT_1};
    border: 0;
    border-right: 1px solid {BORDER_0};
    border-bottom: 1px solid {BORDER_1};
    padding: 5px 9px;
    font-size: {fs_m2}px;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-weight: 600;
}}
QTableWidget, QTableView {{
    background: {BG_1};
    alternate-background-color: {BG_2};
    border: 1px solid {BORDER_0};
    gridline-color: {BORDER_0};
    color: {TEXT_1};
    font-family: {FONT_MONO};
    font-size: {fs_m1}px;
}}
QTableView::item, QTableWidget::item {{ padding: 4px 6px; }}
QTableView::item:selected, QTableWidget::item:selected {{
    background: rgba(34, 211, 238, 0.22);
    color: {TEXT_0};
}}

/* QGroupBox — visible title + framed body. */
QGroupBox {{
    background: {BG_1};
    border: 1px solid {BORDER_0};
    border-radius: 5px;
    margin-top: 16px;
    padding-top: 8px;
    color: {TEXT_1};
    font-weight: 500;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    margin-left: 10px;
    color: {ACCENT};
    background: {BG_0};
    font-weight: 700;
    letter-spacing: 0.5px;
}}

/* QCheckBox — readable indicator. */
QCheckBox {{ color: {TEXT_1}; spacing: 6px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {BORDER_1};
    border-radius: 3px;
    background: {BG_INSET};
}}
QCheckBox::indicator:hover {{ border-color: {BORDER_2}; }}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
    image: none;
}}
QCheckBox::indicator:checked:hover {{ background: {ACCENT_2}; }}

/* QLabel — fallback global colour so untyped labels stay readable. */
QLabel {{ color: {TEXT_1}; }}
"""
