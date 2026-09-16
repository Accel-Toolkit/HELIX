"""TraceWin ``.ini`` options-file reader — pinned against external truth.

Ground truth is ``tests/io/fixtures/tracewin_ini/ads.ini`` (LightWin, MIT)
whose beam LightWin also states as a 6×6 σ-matrix in ``lightwin.toml``;
the transverse Twiss, the normalised emittances and the longitudinal
σ_φ / σ_W must come out of the reader's conversion (no round trip).
Layout variants (2017 layout, DC beam, foreign sizes, other particle
rows) are synthesised from the same file by patching bytes.  Fermilab
project files are pinned only when ``HELIX_TW_INI_PRIVATE_DIR`` is set.
"""
from __future__ import annotations

import math
import os
import shutil
import struct
from pathlib import Path

import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.constants import C_LIGHT
from linac_gen.core.particle import H_MINUS, PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.io.tracewin_ini import (
    FIELD_BY_NAME, FIELDS, KNOWN_SIZES, TAG_OFFSET, TraceWinIni,
    beam_header, lattice_first_frequency, load_tracewin_ini, report,
    resolve_tracewin_ini, sibling_ini, species_for, to_beam_config,
    to_project_extras,
)

FIXTURES = Path(__file__).parent / "fixtures" / "tracewin_ini"
ADS = FIXTURES / "ads.ini"

# LightWin src/lightwin/data/ads/lightwin.toml, [beam]
E_MEV = 20.0
E_REST_MEV = 938.27203
F_MHZ = 100.0
SIGMA_X = ((8.409896e-06, 3.548736e-06), (3.548736e-06, 1.607857e-06))
SIGMA_Y = ((2.941564e-06, 6.094860e-07), (6.094860e-07, 4.418911e-07))
SIGMA_Z = ((3.593136e-06, -2.552518e-07), (-2.552518e-07, 5.994771e-07))

OFF = {f.name: f.offset for f in FIELDS}


def _twiss(sig):
    (s00, s01), (_, s11) = sig
    eps = math.sqrt(s00 * s11 - s01 * s01)
    return eps, -s01 / eps, s00 / eps


def _rel(a, b):
    return abs(a - b) / abs(b)


def _d(v):
    return struct.pack("<d", v)


def _i(v):
    return struct.pack("<i", v)


def _variant(tmp_path, name, patches=None, size=None, src=ADS):
    """A copy of ``src`` with byte patches and an optional truncation.
    The layout tag is deliberately NOT rewritten on truncation: that is
    exactly what TraceWin's own ``*.old.ini`` files look like (old size,
    new tag)."""
    buf = bytearray(src.read_bytes())
    if size is not None:
        buf = buf[:size]
    for off, b in (patches or {}).items():
        buf[off:off + len(b)] = b
    p = tmp_path / name
    p.write_bytes(bytes(buf))
    return p


def _private(name):
    d = os.environ.get("HELIX_TW_INI_PRIVATE_DIR")
    if not d:
        pytest.skip("HELIX_TW_INI_PRIVATE_DIR not set")
    p = Path(d) / name
    if not p.is_file():
        pytest.skip(f"{name} not present in HELIX_TW_INI_PRIVATE_DIR")
    return p


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def test_fixture_is_the_2019_layout():
    ini = load_tracewin_ini(ADS)
    assert isinstance(ini, TraceWinIni)
    assert ini.size == 44824 and ini.format_tag == 44824
    assert ini.version == KNOWN_SIZES[44824]
    assert ini.fields["project_name"] == "ads"
    assert ini.fields["freq1"] == 100.0
    assert ini.fields["current1"] == 0.005
    assert ini.fields["energy1"] == 2e7
    assert ini.fields["nbr_part1"] == 50000
    assert ini.fields["particle_index1"] == 2
    assert [p.name for p in ini.particles[:7]] == [
        "Positron", "Electron", "Proton", "H-", "Deuton", "H2+", "H3+"]
    assert ini.particles[3].charge == -1
    assert abs(ini.particles[3].mass_MeV - 939.294308) < 1e-6
    assert ini.warnings == []


def test_field_table_is_consistent():
    slots = [(f.fmt, f.offset) for f in FIELDS]
    assert len(slots) == len(set(slots)), "duplicate slot in FIELDS"
    assert {f.status for f in FIELDS} <= {"verified", "probable"}
    for name in ("freq1", "current1", "energy1", "etnx1", "etny1", "eps_z1",
                 "alpx1", "betx1", "alpy1", "bety1", "alpz1", "betz1",
                 "nbr_part1", "particle_index1"):
        assert FIELD_BY_NAME[name].status == "verified", name
    # beam-2 twins sit at the documented strides
    assert OFF["freq2"] == OFF["freq1"] + 8
    assert OFF["alpx2"] == OFF["alpx1"] + 16
    assert OFF["betz2"] == OFF["betz1"] + 16


def test_raw_scan_separates_known_and_unknown_slots():
    ini = load_tracewin_ini(ADS)
    assert ini.raw["i32@0x2ed0"] == 50000          # nbr_part1
    assert "i32@0x2ed0" not in ini.unknown
    assert "f64@0x2f24" not in ini.unknown         # freq1
    assert all(k.startswith(("i32@", "f64@")) for k in ini.unknown)
    known_offsets = {(f.fmt, f.offset) for f in FIELDS}
    for k in ini.unknown:
        fmt = "<i" if k.startswith("i32") else "<d"
        assert (fmt, int(k.split("@")[1], 16)) not in known_offsets


def test_old_layout_is_a_prefix(tmp_path):
    p = _variant(tmp_path, "old.ini", size=31624)
    ini = load_tracewin_ini(p)
    assert ini.version == KNOWN_SIZES[31624]
    assert ini.format_tag == 44824                 # the *.old.ini shape
    assert any("layout tag" in w for w in ini.warnings)
    cfg_old, _ = to_beam_config(ini)
    cfg_new, _ = to_beam_config(load_tracewin_ini(ADS))
    assert cfg_old == cfg_new


def test_renamed_file_warns_about_project_name(tmp_path):
    p = tmp_path / "renamed.ini"
    shutil.copy(ADS, p)
    ini = load_tracewin_ini(p)
    assert any("renamed" in w and "'ads'" in w for w in ini.warnings)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_tracewin_ini(FIXTURES / "nope.ini")


def test_refuses_bad_magic(tmp_path):
    p = _variant(tmp_path, "bad_magic.ini", {0: b"NotTraceWin_options"})
    with pytest.raises(ValueError, match="not a TraceWin options file"):
        load_tracewin_ini(p)


def test_refuses_unknown_size(tmp_path):
    p = tmp_path / "grown.ini"
    p.write_bytes(ADS.read_bytes() + b"\0" * 16)
    with pytest.raises(ValueError) as ei:
        load_tracewin_ini(p)
    msg = str(ei.value)
    assert "31624" in msg and "44824" in msg and "probe" in msg


def test_tag_size_mismatch_is_a_warning_not_a_refusal(tmp_path):
    """The 0x68 tag is the saving TraceWin's struct size, not the file's:
    a project upgraded by a newer TraceWin keeps <name>.old.ini at the
    old size with the new tag (7 such files on the reference machine)."""
    p = _variant(tmp_path, "oddtag.ini", {TAG_OFFSET: _i(31624)})
    ini = load_tracewin_ini(p)
    assert ini.version == KNOWN_SIZES[44824] and ini.format_tag == 31624
    assert any("layout tag" in w and "31624" in w for w in ini.warnings)
    cfg, _ = to_beam_config(ini)
    assert cfg.energy == 20.0


def test_truncated_file_is_refused(tmp_path):
    p = tmp_path / "cut.ini"
    p.write_bytes(ADS.read_bytes()[:20000])
    with pytest.raises(ValueError):
        load_tracewin_ini(p)


# ---------------------------------------------------------------------------
# Conversion against the σ-matrix
# ---------------------------------------------------------------------------
def test_transverse_matches_lightwin_sigma():
    cfg, warns = to_beam_config(load_tracewin_ini(ADS))
    assert warns == []
    assert cfg.species == "proton"
    assert cfg.energy == 20.0 and cfg.frequency == 100.0
    assert cfg.current == pytest.approx(5.0)
    assert cfg.n_particles == 50000
    ref = ReferenceParticle(PROTON, E_MEV, F_MHZ)
    for sig, emit, alpha, beta in ((SIGMA_X, cfg.emit_nx, cfg.alpha_x, cfg.beta_x),
                                   (SIGMA_Y, cfg.emit_ny, cfg.alpha_y, cfg.beta_y)):
        eps, a, b = _twiss(sig)
        assert _rel(emit, eps * ref.bg * 1e6) < 1e-3     # normalised, π mm mrad
        assert _rel(alpha, a) < 1e-4
        assert _rel(beta, b) < 1e-4                      # m


def test_longitudinal_sigma_phi_and_sigma_w_match_sigma():
    cfg, _ = to_beam_config(load_tracewin_ini(ADS))
    assert not cfg.continuous
    # α_z: TraceWin +0.17661 → HELIX −0.17661 (documented sign convention)
    assert cfg.alpha_z == pytest.approx(-0.17661, abs=1e-12)
    ref = ReferenceParticle(PROTON, E_MEV, F_MHZ)
    (szz, _), (_, sdd) = SIGMA_Z
    f_hz = F_MHZ * 1e6
    sigma_phi_tw = 360.0 * f_hz * math.sqrt(szz) / (ref.beta * C_LIGHT)   # deg
    sigma_w_tw = PROTON.mass * ref.beta ** 2 * ref.gamma * math.sqrt(sdd)  # MeV
    sigma_phi = math.sqrt(cfg.beta_z * cfg.emit_z)
    gamma_z = (1.0 + cfg.alpha_z ** 2) / cfg.beta_z
    sigma_w = math.sqrt(gamma_z * cfg.emit_z)
    assert _rel(sigma_phi, sigma_phi_tw) < 1e-3
    assert _rel(sigma_w, sigma_w_tw) < 1e-3
    # and the normalised emittance identity eps_z1 = βγ·ε_zδ
    eps_zd, _, _ = _twiss(SIGMA_Z)
    assert _rel(3e-7, eps_zd * ref.bg) < 1e-3
    assert cfg.emit_z == pytest.approx(0.033801, rel=1e-4)
    assert cfg.beta_z == pytest.approx(37.110, rel=1e-4)


def test_emit_z_is_energy_independent_and_beta_z_scales(tmp_path):
    cfg20, _ = to_beam_config(load_tracewin_ini(ADS))
    p = _variant(tmp_path, "e40.ini", {OFF["energy1"]: _d(40e6)})
    cfg40, _ = to_beam_config(load_tracewin_ini(p))
    assert cfg40.energy == 40.0
    assert cfg40.emit_z == pytest.approx(cfg20.emit_z, rel=1e-12)
    bg20 = ReferenceParticle(PROTON, 20.0, F_MHZ).bg
    bg40 = ReferenceParticle(PROTON, 40.0, F_MHZ).bg
    assert cfg40.beta_z == pytest.approx(cfg20.beta_z * (bg20 / bg40) ** 3, rel=1e-12)


def test_dc_beam_maps_to_continuous(tmp_path):
    p = _variant(tmp_path, "dc.ini", {OFF["eps_z1"]: _d(0.0), OFF["betz1"]: _d(0.0)})
    base = BeamConfig(emit_z=0.7, beta_z=3.0, dc_energy_spread_keV=1.5)
    cfg, warns = to_beam_config(load_tracewin_ini(p), base=base)
    assert cfg.continuous is True
    assert cfg.alpha_z == 0.0
    assert cfg.emit_z == 0.7 and cfg.beta_z == 3.0          # untouched, ignored
    assert cfg.dc_energy_spread_keV == 1.5
    assert any("DC" in w and "continuous" in w for w in warns)


@pytest.mark.parametrize("slot", ["energy1", "freq1", "etnx1", "alpz1", "betz1", "current1"])
def test_nan_slot_is_refused_not_propagated(tmp_path, slot):
    p = _variant(tmp_path, f"nan_{slot}.ini", {OFF[slot]: _d(float("nan"))})
    with pytest.raises(ValueError, match=f"{slot}.*not finite"):
        to_beam_config(load_tracewin_ini(p))


def test_negative_eps_z_is_refused_not_dc(tmp_path):
    p = _variant(tmp_path, "negz.ini", {OFF["eps_z1"]: _d(-3e-7)})
    with pytest.raises(ValueError, match="negative longitudinal emittance"):
        to_beam_config(load_tracewin_ini(p))


def test_unknown_species_name_is_refused():
    ini = load_tracewin_ini(ADS)
    for bad in ("electron", "Proton", "h-"):
        with pytest.raises(ValueError, match="species must be one of"):
            to_beam_config(ini, species=bad)


def test_alpha_z_sign_from_the_cross_term():
    """σ_φ and σ_W are even in α_z; the (Δφ, ΔW) cross-term is what fixes
    the sign.  With Δφ = −k·z (k = 360 f/(βc)) and ΔW = m·δ (m = mc²β²γ):
    ⟨Δφ·ΔW⟩ = −k·m·σ_zδ, and HELIX's Twiss says ⟨Δφ·ΔW⟩ = −α_z·ε_z."""
    cfg, _ = to_beam_config(load_tracewin_ini(ADS))
    ref = ReferenceParticle(PROTON, E_MEV, F_MHZ)
    (_, s_zd), (_, _) = SIGMA_Z
    k = 360.0 * F_MHZ * 1e6 / (ref.beta * C_LIGHT)
    m = PROTON.mass * ref.beta ** 2 * ref.gamma
    cross_tw = -k * m * s_zd                     # > 0 for this beam
    cross_helix = -cfg.alpha_z * cfg.emit_z
    assert cross_tw > 0
    assert _rel(cross_helix, cross_tw) < 1e-3
    # and the wrong sign would be caught
    assert _rel(-cross_helix, cross_tw) > 1.0


def test_bunched_beam_without_betz_is_refused(tmp_path):
    p = _variant(tmp_path, "nobetz.ini", {OFF["betz1"]: _d(0.0)})
    with pytest.raises(ValueError, match="betz"):
        to_beam_config(load_tracewin_ini(p))


def test_unpopulated_beam2_is_refused():
    ini = load_tracewin_ini(ADS)
    with pytest.raises(ValueError, match="not populated"):
        to_beam_config(ini, beam=2)
    with pytest.raises(ValueError):
        ini.beam_fields(3)


def test_zero_transverse_emittance_is_refused(tmp_path):
    p = _variant(tmp_path, "noeps.ini", {OFF["etny1"]: _d(0.0)})
    with pytest.raises(ValueError, match="transverse emittance"):
        to_beam_config(load_tracewin_ini(p))


def test_zero_particle_count_keeps_base(tmp_path):
    p = _variant(tmp_path, "np0.ini", {OFF["nbr_part1"]: _i(0)})
    cfg, warns = to_beam_config(load_tracewin_ini(p), base=BeamConfig(n_particles=777))
    assert cfg.n_particles == 777
    assert any("nbr_part1" in w for w in warns)


def test_file_source_base_becomes_generate():
    base = BeamConfig(source="file", distribution_file="x.dst")
    cfg, warns = to_beam_config(load_tracewin_ini(ADS), base=base)
    assert cfg.source == "generate" and cfg.distribution_file is None
    assert any("distribution file" in w for w in warns)


# ---------------------------------------------------------------------------
# Species
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("row, expected", [(2, "proton"), (3, "H-"), (4, "deuteron")])
def test_particle_rows_map_to_helix_species(tmp_path, row, expected):
    p = _variant(tmp_path, f"row{row}.ini", {OFF["particle_index1"]: _i(row)})
    cfg, warns = to_beam_config(load_tracewin_ini(p))
    assert cfg.species == expected
    assert warns == []


def test_user_row_falls_back_with_a_warning(tmp_path):
    p = _variant(tmp_path, "row11.ini", {OFF["particle_index1"]: _i(11)})
    ini = load_tracewin_ini(p)
    cfg, warns = to_beam_config(ini)
    assert cfg.species == "proton"
    assert any("row 11" in w and "NOT used" in w for w in warns)
    cfg2, warns2 = to_beam_config(ini, base=BeamConfig(species="H-"))
    assert cfg2.species == "H-"
    cfg3, warns3 = to_beam_config(ini, species="deuteron")
    assert cfg3.species == "deuteron" and warns3 == []


def test_explicit_species_overrides_the_table_with_a_note():
    ini = load_tracewin_ini(ADS)
    cfg, warns = to_beam_config(ini, species="H-")
    assert cfg.species == "H-"
    assert any("'Proton'" in w and "'proton'" in w for w in warns)
    # H- rest mass drives the relativistic factors
    betz1 = ini.fields["betz1"]
    assert cfg.beta_z == pytest.approx(
        betz1 * 360.0 * 1e8 / (ReferenceParticle(H_MINUS, 20.0, 100.0).bg ** 3
                               * C_LIGHT * H_MINUS.mass), rel=1e-9)


def test_species_for_out_of_table():
    sp, warn = species_for(None)
    assert sp is None and "outside" in warn


def test_out_of_range_particle_index(tmp_path):
    p = _variant(tmp_path, "row99.ini", {OFF["particle_index1"]: _i(99)})
    ini = load_tracewin_ini(p)
    assert ini.particle_for_beam(1) is None
    cfg, warns = to_beam_config(ini)
    assert cfg.species == "proton" and any("outside" in w for w in warns)


# ---------------------------------------------------------------------------
# Lattice frequency check
# ---------------------------------------------------------------------------
def test_lattice_frequency_mismatch_warns():
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.lattice_commands import Freq
    lat = Lattice()
    lat.add(Drift("D1", length=100.0))
    lat.add(Freq("FREQ1", frequency_mhz=352.21))
    assert lattice_first_frequency(lat) == 352.21
    ini = load_tracewin_ini(ADS)
    _, warns = to_beam_config(ini, lattice=lat)
    assert any("352.21" in w and "FREQ" in w for w in warns)
    lat2 = Lattice()
    lat2.add(Freq("FREQ1", frequency_mhz=100.0))
    _, warns2 = to_beam_config(ini, lattice=lat2)
    assert warns2 == []
    assert lattice_first_frequency(Lattice()) is None


# ---------------------------------------------------------------------------
# Header, extras, report, sibling resolution
# ---------------------------------------------------------------------------
def test_beam_header_has_dst_keys_and_scalars():
    h = beam_header(load_tracewin_ini(ADS))
    for k in ("n_particles", "current_mA", "frequency_MHz", "mass_MeV",
              "w_kin_ref", "species", "particle_name", "emit_nx", "alpha_x",
              "beta_x", "emit_ny", "alpha_y", "beta_y", "emit_z", "alpha_z",
              "beta_z", "continuous", "warnings"):
        assert k in h, k
    assert h["mass_MeV"] == PROTON.mass and h["w_kin_ref"] == 20.0
    assert h["particle_name"] == "Proton" and h["warnings"] == []


def test_project_extras_record_meshes_not_applied():
    ex = to_project_extras(load_tracewin_ini(ADS))["tracewin_ini"]
    assert ex["file"] == "ads.ini"
    assert ex["picnic_r_mesh"] == 20 and ex["picnic_z_mesh"] == 40
    assert ex["picnic_xy_mesh"] == [9, 9] and ex["nbr_thread"] == 8
    assert "not applied" in ex["status"]


def test_report_lists_every_section():
    ini = load_tracewin_ini(ADS)
    txt = report(ini)
    assert "ads.ini" in txt and "TraceWin 2019+ layout" in txt
    assert "Applied to the HELIX beam" in txt
    assert "sign flipped" in txt and "-0.17661" in txt
    assert "Identified, not applied" in txt and "picnic_r_mesh = 20" in txt
    assert "Unidentified non-zero slots" in txt
    assert "Not decoded" in txt and "input_dist_type" in txt
    assert "Warnings: none" in txt


def test_report_on_unconvertible_beam_explains(tmp_path):
    ini = load_tracewin_ini(ADS)
    txt = report(ini, beam=2)
    assert "NOT convertible" in txt and "not populated" in txt


def test_sibling_and_resolve(tmp_path):
    lat = tmp_path / "deck.dat"
    lat.write_text("DRIFT 100 20 0\nEND\n")
    assert sibling_ini(lat) is None
    assert resolve_tracewin_ini(lat, None) is None
    assert resolve_tracewin_ini(lat, False) is None
    with pytest.raises(ValueError, match="deck.ini"):
        resolve_tracewin_ini(lat, "auto")
    with pytest.raises(ValueError, match="not found"):
        resolve_tracewin_ini(lat, str(tmp_path / "missing.ini"))
    shutil.copy(ADS, tmp_path / "deck.ini")
    assert sibling_ini(lat) == tmp_path / "deck.ini"
    assert resolve_tracewin_ini(lat, True) == tmp_path / "deck.ini"
    assert resolve_tracewin_ini(lat, "auto") == tmp_path / "deck.ini"
    assert resolve_tracewin_ini(lat, str(ADS)) == ADS


# ---------------------------------------------------------------------------
# Private Fermilab pins (skipped unless HELIX_TW_INI_PRIVATE_DIR is set)
# ---------------------------------------------------------------------------
def test_private_scl_project_matches_its_run_inputs():
    """PIP-II SCL project: inputs known from the TraceWin run, β_z pinned
    to the value that reproduces tracewin.out's σ_φ = 7.455°."""
    ini = load_tracewin_ini(_private("fnalsc_APSeminar_07142026.ini"))
    cfg, warns = to_beam_config(ini)
    assert warns == []
    assert cfg.species == "H-"
    assert cfg.energy == pytest.approx(116.1) and cfg.frequency == 804.96
    assert cfg.current == pytest.approx(23.7) and cfg.n_particles == 100000
    assert cfg.emit_nx == pytest.approx(0.918685) and cfg.emit_ny == pytest.approx(0.887854)
    assert cfg.alpha_x == -0.2758 and cfg.beta_x == 6.46037
    assert cfg.alpha_y == 0.213686 and cfg.beta_y == 1.71278
    assert cfg.alpha_z == 0.5                     # TraceWin −0.5
    assert cfg.emit_z == pytest.approx(0.9079411, rel=1e-6)
    assert cfg.beta_z == pytest.approx(61.2193, rel=1e-5)
    assert math.sqrt(cfg.beta_z * cfg.emit_z) == pytest.approx(7.455, abs=2e-3)


def test_private_2017_lebt_project_is_dc():
    ini = load_tracewin_ini(_private("LEBT_PXIE_JAN2017.ini"))
    assert ini.version == KNOWN_SIZES[31624]
    cfg, warns = to_beam_config(ini)
    assert cfg.species == "H-" and cfg.continuous is True
    assert cfg.energy == pytest.approx(0.03) and cfg.frequency == 162.5
    assert cfg.current == pytest.approx(5.0)
    assert cfg.emit_nx == pytest.approx(0.1370111)
    assert any("DC" in w for w in warns)
