"""The `.dst` header frequency: HELIX's convention vs TraceWin's.

MEASURED 2026-07-31 against six genuine TraceWin PIP-II files
(`part_rfq.dst`, `part_dtl1.dst`, `lbin.dst`, `input.dst`,
`ssr2out.dst`, `output.dst`): every one carries **162.5 MHz**, whether
its particles sit at 30 keV, 166 MeV (an SSR2 plane, 325 MHz RF) or
752 MeV (an HB650 plane, 650 MHz RF).  TraceWin therefore writes the
BUNCH REPETITION RATE, never the local cavity clock.

HELIX writes `ref.frequency` — the local clock — because that is the
clock its Δφ degrees are measured in (`write_dst` does a pure
deg→rad conversion, and `load_dst` the inverse; the frequency never
enters either).  HELIX↔HELIX round-trips are therefore self-consistent,
but a HELIX file written downstream of a frequency jump is NOT
interchangeable with TraceWin.

Writing `bunch_frequency` instead would need the phase column
converted too, and whether TraceWin converts on export is unresolved —
it needs one round-trip through real TraceWin to settle.  Until then
the divergence is WARNED about rather than silently papered over, and
these tests pin that contract so the warning cannot be dropped by
accident.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.io.tracewin_dst import (DstHeaderWarning,
                                       warn_if_nonstandard_dst_header)


def _beam(f_local, f_bunch, n=16):
    b = Beam(ref=ReferenceParticle(species=H_MINUS, w_kin=166.0,
                                   frequency=f_local),
             n_particles=n, current=5.0)
    b.bunch_frequency = f_bunch
    return b


def test_no_warning_before_a_frequency_jump():
    """The overwhelmingly common case: the two agree, header is correct
    by both conventions, nothing to say."""
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        fired = warn_if_nonstandard_dst_header(_beam(162.5, 162.5))
    assert fired is False
    assert not [w for w in wl if issubclass(w.category, DstHeaderWarning)]


def test_warns_downstream_of_a_frequency_jump():
    """162.5 MHz train tracked at 325 MHz: the header will say 325,
    TraceWin would say 162.5.  Must be loud, and must name BOTH numbers
    so the reader can act."""
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        fired = warn_if_nonstandard_dst_header(_beam(325.0, 162.5))
    assert fired is True
    hits = [w for w in wl if issubclass(w.category, DstHeaderWarning)]
    assert len(hits) == 1
    msg = str(hits[0].message)
    assert "325" in msg and "162.5" in msg
    assert "TraceWin" in msg


def test_missing_bunch_frequency_does_not_warn():
    """A cleared bunch_frequency is a different problem and the header
    is not the place to report it — don't add noise.  (A zero
    ref.frequency is unreachable: ReferenceParticle divides by it to get
    the wavelength and raises first.)"""
    b = _beam(325.0, 0.0)
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        assert warn_if_nonstandard_dst_header(b) is False
    assert not [w for w in wl if issubclass(w.category, DstHeaderWarning)]


def test_export_path_emits_the_warning(tmp_path):
    """End-to-end through the real writer used by the CLI."""
    from linac_gen.cli.common import write_final_dst
    b = _beam(325.0, 162.5, n=32)
    b.particles[:, 0] = np.linspace(-1, 1, 32)
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        out = write_final_dst(b, tmp_path / "post_jump.dst")
    assert [w for w in wl if issubclass(w.category, DstHeaderWarning)]

    # And the file really does carry the local clock, so its phases stay
    # interpretable on re-import — the property the warning documents.
    from linac_gen.io.tracewin_dst import load_dst
    _, hdr = load_dst(str(out))
    assert hdr["frequency_MHz"] == pytest.approx(325.0)


def test_helix_round_trip_is_self_consistent(tmp_path):
    """The reason we do NOT just switch the header to the bunch rate:
    today's convention round-trips exactly."""
    from linac_gen.cli.common import write_final_dst
    from linac_gen.io.tracewin_dst import load_dst
    b = _beam(325.0, 162.5, n=64)
    rng = np.random.default_rng(4)
    b.particles[:, 0] = rng.normal(0, 1.0, 64)
    b.particles[:, 4] = rng.normal(0, 3.0, 64)      # deg at 325 MHz
    b.particles[:, 5] = rng.normal(0, 0.01, 64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = write_final_dst(b, tmp_path / "rt.dst")
    parts, hdr = load_dst(out)
    # Δφ comes back in the same degrees it went out in, because the
    # header records the clock those degrees belong to.
    np.testing.assert_allclose(np.sort(parts[:, 4]),
                               np.sort(b.particles[:, 4] - b.particles[:, 4].mean()),
                               atol=1e-9)
    assert hdr["frequency_MHz"] == pytest.approx(325.0)
