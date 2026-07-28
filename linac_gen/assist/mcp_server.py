"""HELIX as an MCP server — drive it from Claude Code / Claude Desktop
with YOUR OWN Claude subscription login.

This is the sanctioned way to use a Claude subscription with HELIX:
HELIX is a *tool provider*, not an auth broker.  The MCP client (Claude
Code, Claude Desktop, or any MCP client) authenticates with the user's
own login and calls these tools; HELIX itself makes NO network
connection and needs no API key — it only exposes the same audited
tool registry the in-app assistant uses, over stdio.

Confirmation is delegated to the MCP client: tools are annotated
``readOnlyHint`` (read tier) / ``destructiveHint`` (mutate tier) so the
client's own permission system prompts appropriately — the same
"confirm before it runs" guarantee, enforced by the client.

The ``mcp`` package is an OPTIONAL dependency (extra ``assist-mcp``);
the schema-building helpers below are import-safe without it, and only
:func:`serve` requires it.
"""
from __future__ import annotations

import json

from linac_gen.assist.tools import TOOLS, WorkContext


# ---- pure helpers (no ``mcp`` import — unit-testable everywhere) -----
def _annotations_for(tier: str) -> dict:
    """Map HELIX tiers to MCP tool-annotation hints so the client's
    permission model auto-approves reads and confirms writes/runs."""
    if tier == "read":
        return {"readOnlyHint": True, "openWorldHint": False}
    if tier == "mutate":
        return {"readOnlyHint": False, "destructiveHint": True,
                "openWorldHint": False}
    # compute: not read-only (it runs a solver) but not destructive
    return {"readOnlyHint": False, "destructiveHint": False,
            "openWorldHint": False}


def tool_specs() -> list[dict]:
    """The registry as MCP-shaped specs (name, description, schema,
    annotations) — a pure function, no ``mcp`` runtime needed."""
    out = []
    for t in TOOLS.values():
        out.append({"name": t.name, "description": t.description,
                    "inputSchema": t.schema,
                    "annotations": _annotations_for(t.tier)})
    return out


def call(context: WorkContext, name: str, arguments: dict) -> str:
    """Execute one tool against *context*; returns the JSON envelope
    string (the MCP text-content payload).  Unknown names refuse —
    there is no dynamic execution."""
    tool = TOOLS.get(name)
    if tool is None:
        return json.dumps({"status": "refused",
                           "data": {"message": f"unknown tool {name!r}"},
                           "provenance": {}, "warnings": []})
    try:
        env = tool.fn(context, **(arguments or {}))
    except TypeError as exc:
        env = {"status": "refused",
               "data": {"message": f"bad parameters: {exc}"},
               "provenance": {}, "warnings": []}
    except Exception as exc:                                # noqa: BLE001
        env = {"status": "error",
               "data": {"message": f"{type(exc).__name__}: {exc}"},
               "provenance": {}, "warnings": []}
    return json.dumps(env, default=str)


# ---- server runtime (needs ``mcp``) ---------------------------------
def build_server(context: WorkContext | None = None):
    """Construct the MCP ``Server`` bound to a persistent session
    context (so load → run → query share state across calls)."""
    try:
        from mcp.server import Server
        import mcp.types as types
    except ImportError as exc:                             # noqa: BLE001
        raise RuntimeError(
            "the MCP surface needs the optional 'mcp' package: "
            "pip install linac_gen[assist-mcp]") from exc

    import anyio

    ctx = context or WorkContext(calc_dir="runs")
    server = Server("helix")

    @server.list_tools()
    async def _list_tools():
        return [types.Tool(
            name=s["name"], description=s["description"],
            inputSchema=s["inputSchema"],
            annotations=types.ToolAnnotations(**s["annotations"]))
            for s in tool_specs()]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict):
        # run the (blocking) HELIX tool off the event loop
        text = await anyio.to_thread.run_sync(call, ctx, name, arguments)
        return [types.TextContent(type="text", text=text)]

    return server, ctx


def serve(initial_input: str | None = None, calc_dir: str = "runs") -> int:
    """Run the stdio MCP server (blocking).  ``initial_input`` may be a
    project/lattice file to preload into the session."""
    import anyio
    from mcp.server.stdio import stdio_server

    ctx = WorkContext(calc_dir=calc_dir)
    if initial_input:
        env = TOOLS["load_lattice"].fn(ctx, path=initial_input)
        # a failed preload is non-fatal — the client can load one
        if env["status"] != "ok":
            import sys
            print(f"[helix-mcp] could not preload {initial_input}: "
                  f"{env['data'].get('message')}", file=sys.stderr)
    server, ctx = build_server(ctx)

    async def _main():
        async with stdio_server() as (read, write):
            await server.run(read, write,
                             server.create_initialization_options())

    anyio.run(_main)
    return 0
