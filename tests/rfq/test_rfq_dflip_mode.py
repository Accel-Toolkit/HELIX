"""The experimental ``per_plane_dflip`` mode and the arm-time defocus
audit (2026-08-03 loss-location campaign).

Background: under the per-cell phase reset, the armed profile's
axisymmetric defocus channel D = (gx+gy)/2 accumulates the OPPOSITE
sign vs the analytic RF defocus (measured on PXIE: 100 % of cells at
0.81x magnitude), which made "per_plane" transmission agreement
compensatory.  ``per_plane_dflip`` flips D (and scales it by
``d_scale``); ``apply_rfq_geometry`` now audits every arming and warns.

Synthetic-vane tests run everywhere; the PXIE audit-number pins skip on
public checkouts (vane file is local-only ANL/CEA data).
"""
from __future__ import annotations

import logging
import os
import warnings

import numpy as np
import pytest

from linac_gen.elements.rfq_cell import RfqCell
from linac_gen.io.rfq_geometry_helper import apply_rfq_geometry
from linac_gen.io.tracewin_parser import parse_tracewin

from tests.rfq.test_rfq_geometry_auto import _write_deck, _write_vane

PXIE_VANE = os.path.join(os.path.expanduser(
    "~/Desktop/Projects/PIP_II/Paper/LEBT+RFQ"), "pxie-rfq.vane")
PXIE_DECK = os.path.join(os.path.dirname(__file__), "..", "..",
                         "examples", "lebt_plus_rfq")


def _armed_cells(tmp_path, mode, **kw):
    """Arm a synthetic deck.  The vane/deck are written ONCE per
    directory: rewriting the vane would invalidate the profile's disk
    cache and force a fresh (pyamg, iteratively-converged) solve per
    arming, making cross-mode comparisons differ at ~1e-9 instead of
    being cache-identical."""
    if not (tmp_path / "test.vane").is_file():
        _write_vane(tmp_path / "test.vane")
        _write_deck(tmp_path / "test.dat")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, _ = parse_tracewin(str(tmp_path / "test.dat"))
        n = apply_rfq_geometry(lat, tmp_path / "test.vane",
                               mode=mode, **kw)
    assert n > 0
    return [e for e in lat.elements if isinstance(e, RfqCell)
            and e._geom_z is not None]


class TestModeRegistration:
    def test_unknown_mode_raises(self, tmp_path):
        _write_vane(tmp_path / "test.vane")
        _write_deck(tmp_path / "test.dat")
        lat, _ = parse_tracewin(str(tmp_path / "test.dat"))
        with pytest.raises(ValueError, match="mode must be one of"):
            apply_rfq_geometry(lat, tmp_path / "test.vane", mode="nope")

    def test_bad_d_scale_raises(self, tmp_path):
        _write_vane(tmp_path / "test.vane")
        _write_deck(tmp_path / "test.dat")
        lat, _ = parse_tracewin(str(tmp_path / "test.dat"))
        with pytest.raises(ValueError, match="d_scale"):
            apply_rfq_geometry(lat, tmp_path / "test.vane",
                               mode="per_plane_dflip", d_scale=0.0)

    def test_d_scale_ignored_elsewhere_warns(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING,
                             logger="linac_gen.io.rfq_geometry_helper"):
            _armed_cells(tmp_path, "per_plane", d_scale=1.23)
        assert any("only used by" in r.message for r in caplog.records)

    def test_simulation_accepts_dflip(self):
        """Validation string admits the new mode and the d_scale
        plumb-through exists (source-level: constructing a
        Simulation needs a full lattice+beam)."""
        import inspect
        from linac_gen.core.simulation import Simulation
        src = inspect.getsource(Simulation.__init__)
        assert '"per_plane_dflip"' in src
        assert "rfq_d_scale" in src
        src_arm = inspect.getsource(Simulation._arm_rfq_geometry)
        assert "d_scale" in src_arm


class TestDflipAlgebra:
    def test_dscale_one_is_the_negated_swap(self, tmp_path):
        """gx' = -gy, gy' = -gx exactly (the e41 identity)."""
        pp = _armed_cells(tmp_path, "per_plane")
        df = _armed_cells(tmp_path, "per_plane_dflip", d_scale=1.0)
        for a, b in zip(pp, df):
            np.testing.assert_allclose(b._geom_gx, -a._geom_gy,
                                       rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(b._geom_gy, -a._geom_gx,
                                       rtol=1e-12, atol=1e-14)

    def test_general_dscale_matches_gd_split(self, tmp_path):
        pp = _armed_cells(tmp_path, "per_plane")
        df = _armed_cells(tmp_path, "per_plane_dflip", d_scale=1.23)
        for a, b in zip(pp, df):
            g = 0.5 * (a._geom_gx - a._geom_gy)
            d = 0.5 * (a._geom_gx + a._geom_gy)
            np.testing.assert_allclose(b._geom_gx, g - 1.23 * d,
                                       rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(b._geom_gy, -g - 1.23 * d,
                                       rtol=1e-12, atol=1e-14)

    def test_existing_modes_unchanged(self, tmp_path):
        """antisym stays gy = -gx; per_plane's x-channel untouched."""
        an = _armed_cells(tmp_path, "antisym")
        pp = _armed_cells(tmp_path, "per_plane")
        for a, p in zip(an, pp):
            np.testing.assert_array_equal(a._geom_gy, -a._geom_gx)
            np.testing.assert_allclose(a._geom_gx, p._geom_gx,
                                       rtol=1e-12, atol=1e-14)


class TestAudit:
    def test_synthetic_quad_vane_is_d_free(self, tmp_path, caplog):
        """The ideal-quad synthetic vane has no axisymmetric content:
        the audit must take the D-free branch, not the KNOWN-ISSUE
        one."""
        with caplog.at_level(logging.INFO,
                             logger="linac_gen.io.rfq_geometry_helper"):
            _armed_cells(tmp_path, "per_plane")
        msgs = [r.message for r in caplog.records]
        assert not any("KNOWN ISSUE" in m for m in msgs)
        assert any("no net axisymmetric" in m for m in msgs)

    @pytest.mark.skipif(not os.path.isfile(PXIE_VANE),
                        reason="PXIE vane file not present (public "
                               "checkout / other machine)")
    def test_pxie_audit_pins_the_finding(self, caplog):
        """The 2026-08-03 measurement, pinned: shipped per_plane is
        sign-inverted in ~all scored cells at ~0.81x; dflip at the
        audit-suggested 1.23 verifies at ratio ~1.00."""
        deck = os.path.join(PXIE_DECK, "lebt_plus_rfq.dat")
        if not os.path.isfile(deck):
            pytest.skip("PXIE deck not present")
        from linac_gen.io.rfq_geometry_helper import \
            _audit_defocus_channel
        cwd = os.getcwd()
        os.chdir(PXIE_DECK)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lat, _ = parse_tracewin("lebt_plus_rfq.dat")
                with caplog.at_level(
                        logging.WARNING,
                        logger="linac_gen.io.rfq_geometry_helper"):
                    apply_rfq_geometry(lat, PXIE_VANE, mode="per_plane")
            cells = [e for e in lat.elements if isinstance(e, RfqCell)]
            n, frac, med = _audit_defocus_channel(cells)
        finally:
            os.chdir(cwd)
        assert n > 150
        assert frac < 0.05                      # inverted ~everywhere
        assert 0.70 < med < 0.90                # the 0.81x magnitude
        assert any("KNOWN ISSUE" in r.message for r in caplog.records)

    @pytest.mark.skipif(not os.path.isfile(PXIE_VANE),
                        reason="PXIE vane file not present (public "
                               "checkout / other machine)")
    def test_pxie_dflip_verifies_at_suggested_scale(self, caplog):
        '''Arming dflip at the audit-suggested 1.23 must flip the
        verdict: ~all cells agree with the analytic sign and the
        armed |impulse| ratio lands at ~1.00.'''
        deck = os.path.join(PXIE_DECK, "lebt_plus_rfq.dat")
        if not os.path.isfile(deck):
            pytest.skip("PXIE deck not present")
        from linac_gen.io.rfq_geometry_helper import \
            _audit_defocus_channel
        cwd = os.getcwd()
        os.chdir(PXIE_DECK)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lat, _ = parse_tracewin("lebt_plus_rfq.dat")
                with caplog.at_level(
                        logging.INFO,
                        logger="linac_gen.io.rfq_geometry_helper"):
                    apply_rfq_geometry(lat, PXIE_VANE,
                                       mode="per_plane_dflip",
                                       d_scale=1.23)
            cells = [e for e in lat.elements if isinstance(e, RfqCell)]
            n, frac, med = _audit_defocus_channel(cells)
        finally:
            os.chdir(cwd)
        assert n > 150
        assert frac > 0.95
        assert 0.95 < med < 1.05
        assert any("sign verified" in r.message for r in caplog.records)
