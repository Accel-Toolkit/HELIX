"""Phase 4 (Sight): capture payload delivery across transports."""
from __future__ import annotations

import json

from linac_gen.assist.config import AssistConfig
from linac_gen.assist.agent import AgentSession, Decision
from linac_gen.assist.sdk_backend import _content
from linac_gen.assist.testing import MockProvider, turn_text, turn_tools
from linac_gen.assist.tools import TOOLS, WorkContext


class _VisionCtx(WorkContext):
    def available_plots(self):
        return ["RMS σ (x · y · z)"]

    def grab_plot(self, name):
        return {"img_b64": "QUJDRA==", "w": 10, "h": 5, "label": "RMS"}

    def grab_screen(self):
        return {"img_b64": "QUJDRA==", "w": 10, "h": 5, "label": "Beam tab"}


def test_sdk_content_emits_image_block():
    env = {"status": "ok", "warnings": [],
           "data": {"img_b64": "QUJDRA==", "w": 10, "h": 5, "label": "x"}}
    out = _content(env)
    kinds = [b["type"] for b in out["content"]]
    assert kinds == ["text", "image"]
    img = out["content"][1]
    assert img["data"] == "QUJDRA==" and img["mimeType"] == "image/png"
    assert "img_b64" not in out["content"][0]["text"]     # b64 not in text


def test_look_at_plot_saves_capture(tmp_path):
    ctx = _VisionCtx(calc_dir=str(tmp_path))
    r = TOOLS["look_at_plot"].fn(ctx, name="rms")
    assert r["status"] == "ok"
    assert r["data"]["label"] == "RMS"
    saved = r["data"]["saved_to"]
    assert saved.endswith(".png")
    assert (tmp_path / "assist_captures").exists()
    assert open(saved, "rb").read() == b"ABCD"            # b64 round-trip


def test_classic_transport_strips_b64(tmp_path):
    """The raw-HTTP agent path replaces the b64 with a saved-file note."""
    provider = MockProvider([
        turn_tools(("look_at_plot", {"name": "rms"})),
        turn_text("done"),
    ])
    s = AgentSession(AssistConfig(provider="anthropic", model="m",
                                  api_key="sk-x"),
                     _VisionCtx(calc_dir=str(tmp_path)),
                     approver=lambda r: Decision.APPROVE,
                     provider=provider)
    s.ask("look")
    tr = next(m for m in s.transcript
              if type(m).__name__ == "ToolResultsMsg")
    payload = json.loads(tr.outcomes[0].content)
    assert "img_b64" not in json.dumps(payload)
    assert "image_note" in payload["data"]
    assert "saved to" in payload["data"]["image_note"]
    s.close()


def test_look_refuses_outside_gui(tmp_path):
    ctx = WorkContext(calc_dir=str(tmp_path))
    assert TOOLS["look_at_plot"].fn(ctx, name="rms")["status"] == "refused"
    assert TOOLS["look_at_screen"].fn(ctx)["status"] == "refused"
