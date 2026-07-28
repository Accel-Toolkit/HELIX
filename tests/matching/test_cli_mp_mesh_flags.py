"""--mp-grid / --mp-extent: PIC mesh override for particle-based costs.

2026-07-25 review, claim 8: the CLI exposed --mp-n-particles but no way
to set the SC mesh, so the MP cost solver and the gradient+SC bunch
always fell back to the engine's 32^3 / +-4 sigma defaults.
"""
from __future__ import annotations


def test_parser_accepts_mp_mesh_flags():
    from linac_gen.matching.__main__ import _make_parser
    p = _make_parser()
    a = p.parse_args(["deck.dat", "--mp-grid", "48", "--mp-extent", "6.0"])
    assert a.mp_grid == 48
    assert a.mp_extent == 6.0


def test_parser_mp_mesh_flags_default_to_none():
    from linac_gen.matching.__main__ import _make_parser
    a = _make_parser().parse_args(["deck.dat"])
    assert a.mp_grid is None
    assert a.mp_extent is None
