"""Phase 5: lab notebook (cross-session memory) + report generation."""
from __future__ import annotations

import os

import numpy as np

from linac_gen.assist import notebook as NB
from linac_gen.assist.config import AssistConfig
from linac_gen.assist.agent import AgentSession, Decision
from linac_gen.assist.testing import MockProvider, turn_text, turn_tools
from linac_gen.assist.tools import TOOLS, WorkContext


class _ResCtx(WorkContext):
    """Context with synthetic attribute-style results (no simulation)."""
    def __init__(self, calc_dir):
        super().__init__(calc_dir=calc_dir)
        class _R:
            sigma_x = np.array([1.0, 0.9]); sigma_y = np.array([1.0, 1.1])
            sigma_phi = np.array([3.0, 3.5]); emit_x = np.array([.2, .21])
            ref_w_kin = np.array([2.1, 800.0])
        self.results = _R()


# ---- notebook primitives ---------------------------------------------
def test_note_and_tail_roundtrip(tmp_path):
    NB.append_note(str(tmp_path), "alpha")
    NB.append_note(str(tmp_path), "beta")
    tail = NB.load_tail(str(tmp_path))
    assert "alpha" in tail and "beta" in tail


def test_load_tail_missing_and_corrupt(tmp_path):
    assert NB.load_tail(str(tmp_path)) == ""             # missing -> ""
    (tmp_path / "assist_notebook.md").write_bytes(b"\x00\xff garbage")
    assert isinstance(NB.load_tail(str(tmp_path)), str)  # never raises


def test_tail_keeps_only_last_k_sessions(tmp_path):
    p = tmp_path / "assist_notebook.md"
    p.write_text("".join(f"## Session S{i}\n- t{i}\n" for i in range(6)))
    tail = NB.load_tail(str(tmp_path), k_entries=2)
    assert "S4" in tail and "S5" in tail and "S2" not in tail


# ---- session summary + memory ----------------------------------------
def test_session_summary_and_new_session_memory(tmp_path):
    ctx = _ResCtx(str(tmp_path))
    prov = MockProvider([turn_tools(("notebook_note", {"text": "sx ok"})),
                         turn_text("Concluded: matched.")])
    s = AgentSession(AssistConfig(provider="anthropic", model="m",
                                  api_key="k"), ctx,
                     approver=lambda r: Decision.APPROVE, provider=prov)
    s.ask("go")
    s.close()
    nb = (tmp_path / "assist_notebook.md").read_text()
    assert "## Session" in nb and "notebook_note" in nb
    assert "Concluded: matched." in nb
    s2 = AgentSession(AssistConfig(provider="anthropic", model="m",
                                   api_key="k"),
                      WorkContext(calc_dir=str(tmp_path)),
                      approver=lambda r: Decision.APPROVE,
                      provider=MockProvider([turn_text("hi")]))
    assert "sx ok" in s2.notebook_tail
    assert "PAST SESSIONS" in s2._prompt_extra()
    s2.close()


def test_empty_session_writes_no_summary(tmp_path):
    s = AgentSession(AssistConfig(provider="anthropic", model="m",
                                  api_key="k"),
                     WorkContext(calc_dir=str(tmp_path)),
                     approver=lambda r: Decision.APPROVE,
                     provider=MockProvider([]))
    s.close()                                            # zero turns
    assert not (tmp_path / "assist_notebook.md").exists()


# ---- report -----------------------------------------------------------
def test_generate_report_kpis_no_gui(tmp_path):
    ctx = _ResCtx(str(tmp_path))
    r = TOOLS["generate_report"].fn(ctx, title="T1")
    assert r["status"] == "ok" and r["data"]["figures"] == 0
    text = open(r["data"]["report"]).read()
    assert text.startswith("# T1")
    assert "| sigma_x | 0.9 mm |" in text
    assert "| ref_w_kin | 800 MeV |" in text


def test_generate_report_embeds_figures_with_gui_hook(tmp_path):
    class _GuiCtx(_ResCtx):
        def grab_plot(self, name):
            return {"img_b64": "QUJDRA==", "w": 4, "h": 2, "label": name}
    r = TOOLS["generate_report"].fn(_GuiCtx(str(tmp_path)))
    assert r["status"] == "ok" and r["data"]["figures"] == 3
    text = open(r["data"]["report"]).read()
    assert text.count("![") == 3                         # embedded images
    pngs = [f for f in os.listdir(tmp_path / "reports")
            if f.endswith(".png")]
    assert len(pngs) == 3


def test_report_refuses_without_results(tmp_path):
    r = TOOLS["generate_report"].fn(WorkContext(calc_dir=str(tmp_path)))
    assert r["status"] == "refused"
