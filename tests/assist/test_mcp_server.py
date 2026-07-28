# tests/assist/test_mcp_server.py
"""HELIX-as-MCP-server surface.

The pure helpers (spec mapping, tier→annotation, stateful call) need no
network and no ``mcp`` runtime; the end-to-end handshake test skips if
the optional ``mcp`` package is absent.  The MCP server itself makes no
network connection (the no_network conftest fixture would fail it if
it did) — auth lives entirely in the client (Claude Code / Desktop)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from linac_gen.assist.mcp_server import (
    _annotations_for, call, tool_specs,
)
from linac_gen.assist.tools import TOOLS, WorkContext


def test_tier_to_annotation_mapping():
    assert _annotations_for("read") == {"readOnlyHint": True,
                                        "openWorldHint": False}
    mut = _annotations_for("mutate")
    assert mut["destructiveHint"] is True and mut["readOnlyHint"] is False
    comp = _annotations_for("compute")
    assert comp["readOnlyHint"] is False and comp["destructiveHint"] \
        is False


def test_specs_cover_every_tool_with_annotations():
    specs = tool_specs()
    assert {s["name"] for s in specs} == set(TOOLS)
    for s in specs:
        assert s["inputSchema"]["type"] == "object"
        assert "readOnlyHint" in s["annotations"]
    # read tools are flagged read-only so the client auto-approves them
    read_names = {n for n, t in TOOLS.items() if t.tier == "read"}
    for s in specs:
        if s["name"] in read_names:
            assert s["annotations"]["readOnlyHint"] is True
        if TOOLS[s["name"]].tier == "mutate":
            assert s["annotations"]["destructiveHint"] is True


def test_call_is_stateful_across_invocations(ctx):
    # a read
    out = json.loads(call(ctx, "get_status", {}))
    assert out["status"] == "ok"
    assert out["data"]["lattice"]["n_elements"] == 4
    # a compute that mutates session state
    out = json.loads(call(ctx, "run_envelope", {}))
    assert out["status"] == "ok"
    assert ctx.results is not None
    # a follow-up read sees the new state
    out = json.loads(call(ctx, "query_results",
                          {"quantity": "sigma_x", "s_m": 0.2}))
    assert out["status"] == "ok"
    assert np.isfinite(out["data"]["value"])


def test_call_unknown_tool_refuses():
    out = json.loads(call(WorkContext(calc_dir="."), "rm_rf", {}))
    assert out["status"] == "refused"
    assert "unknown tool" in out["data"]["message"]


def test_call_bad_params_refuses_not_crash(ctx):
    out = json.loads(call(ctx, "query_results", {"quantity": "sigma_x"}))
    # missing required s_m → TypeError → refused, never an exception
    assert out["status"] == "refused"


def test_end_to_end_mcp_handshake(ctx):
    """List + call over the real MCP in-memory transport — proves the
    server speaks the protocol.  Runs the coroutine directly (no
    pytest-asyncio dependency).  Skips if 'mcp' isn't installed."""
    pytest.importorskip("mcp")
    import anyio
    from mcp.shared.memory import (
        create_connected_server_and_client_session,
    )
    from linac_gen.assist.mcp_server import build_server

    server, _ctx = build_server(ctx)

    async def _run():
        async with create_connected_server_and_client_session(
                server) as client:
            listed = await client.list_tools()
            names = {t.name for t in listed.tools}
            assert "run_envelope" in names and "get_status" in names
            gs = next(t for t in listed.tools if t.name == "get_status")
            assert gs.annotations.readOnlyHint is True
            result = await client.call_tool("get_status", {})
            payload = json.loads(result.content[0].text)
            assert payload["status"] == "ok"
            assert payload["data"]["lattice"]["n_elements"] == 4

    anyio.run(_run)
