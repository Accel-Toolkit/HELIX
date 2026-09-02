"""Episode 13 — The voice assistant (series finale), MAXIMUM-DETAIL CUT.

Build:  PYTHONPATH=.:gui:tutorials python3 tutorials/storyboards/ep13_assistant.py
Output: tutorials/rendered/ep13_assistant.mp4 (+ .srt)

v3 (exhaustive cut) — everything is RECORDED REAL BEHAVIOUR:
  * motion clips captured from the live panel with pipeline.record.Recorder,
    encoded at the ACHIEVED capture rate so playback speed is honest,
  * ~20 GENUINE model turns over the keyless claude_sdk path — recorded
    live during the build, nothing scripted or fabricated,
  * the whole confirmation matrix on camera: mutate Approve, mutate Deny,
    the three-button compute gate, Approve (session) ticking the
    auto-approve checkbox, then a compute call running with NO gate,
  * all four quick-action chips driven for real (capability list, Python
    sandbox with the figure returned inline, guided tour with instant
    next/stop, hidden-fault drill through gate → hint → give-up →
    bit-exact restore, verified programmatically after the clip),
  * sight (look_at_plot), diagnose, notebook, search_manual, the
    web_search refusal, the queued inbox + Stop interrupt, the orb state
    montage, the voice-only view and close-hides semantics,
  * the actual session ledger quoted in the closing cards.
Voice/mic hardware cannot run offscreen, so the voice stack stays a
(verified) card + the real mic controls; everything else is live.

Narration honesty: model replies are nondeterministic, so scenes whose
narration depends on a recorded verdict were written AFTER viewing the
frames.  To iterate narration without re-recording (which would change
the replies the narration was written for), set EP13_REUSE=1: the
storyboard then re-renders the video from the clips/stills of the last
capture (metadata in ep13_work/clips.json) instead of recapturing.

Panel gate: `_send` refuses ALL input until a backend is connected (the
session check sits before the instant-command fast path), so the connect
scene comes before every command scene.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time as _time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "gui"), str(ROOT / "tutorials")]
_qcfg = Path(tempfile.mkdtemp()) / "offscreen.json"
_qcfg.write_text('{"screens": [{"name": "tut", "x": 0, "y": 0, '
                 '"width": 1920, "height": 1080, "logicalDpi": 96, '
                 '"physicalDpi": 96}]}')
os.environ.setdefault("QT_QPA_PLATFORM", f"offscreen:configfile={_qcfg}")
os.environ.setdefault("HELIX_QSETTINGS_DIR", tempfile.mkdtemp())
os.environ["HELIX_ASSIST_NO_PREWARM"] = "1"      # no mic, no auto-warm
# the panel's claude_sdk backend spawns the user's own `claude` CLI —
# scrub the variables a nested Claude Code session would inherit
for _v in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "ANTHROPIC_API_KEY"):
    os.environ.pop(_v, None)
if "QT_PLUGIN_PATH" not in os.environ:
    import PyQt6
    os.environ["QT_PLUGIN_PATH"] = os.path.join(
        os.path.dirname(PyQt6.__file__), "Qt6", "plugins")

from pipeline.record import Recorder                    # noqa: E402
from pipeline.render import Scene, build_video          # noqa: E402

WORK = ROOT / "tutorials" / "rendered" / "ep13_work"
SHOTS = WORK / "shots"
FRAMES = WORK / "frames"
META = WORK / "clips.json"
OUT = ROOT / "tutorials" / "rendered" / "ep13_assistant.mp4"

SCENES = [
    Scene("010_title",
          "The final episode, the one this series has been saving. "
          "HELIX has an assistant living inside it: you talk to "
          "it, typed or spoken, and it runs the physics, reads the "
          "plots, drives the interface, and answers from the "
          "manual. This voice, which has narrated every episode, "
          "is the assistant's own. Today you watch it work "
          "exhaustively, every control and every tier of its "
          "safety model on camera, recorded live while this video "
          "was being made.",
          min_s=6.0),
    Scene("020_open",
          "Open it from the toolbar and the assistant appears in "
          "its own window beside the workbench. Top to bottom: the "
          "orb and its lamp, which name the assistant's state; a "
          "status row; the backend row, where you choose the "
          "brain; the conversation view; four quick action chips; "
          "the input row; and the options row. We pressed A plus "
          "twice so the text reads on camera.",
          min_s=7.0),
    Scene("022_anatomy",
          "Up close. The status row: the backend button, which "
          "reopens the provider row after you connect, and a small "
          "latency readout. The provider row: backend selector, "
          "password masked key field, editable model box, Connect. "
          "The chips: guided tour, training drill, Python sandbox, "
          "capability list. The input row: Hold, for push to talk; "
          "the HELIX toggle, for hands free wake word listening; "
          "the input line, Send, and Stop. The options: auto "
          "approve compute for this session, speak replies, "
          "instant commands, narrate events, watch runs, and Text, "
          "which hides the transcript for a voice only view, plus "
          "A minus and A plus for the text size.",
          min_s=8.0),
    Scene("025_orb_states",
          "The orb is the assistant's face, and this is its full "
          "vocabulary, on the real widget. Idle, cyan. Listening, "
          "blue, microphone open. Thinking, amber, while a model "
          "reasons or a tool runs. Responding, bright cyan, with "
          "an equaliser ring, a synthesised voice print, while it "
          "speaks. Awaiting confirm, pink: waiting for your "
          "approval and nothing else. Error, red. Off, grey, when "
          "hands free listening is disabled. The colour morphs "
          "rather than snaps.",
          min_s=8.0),
    Scene("030_live",
          "Before connecting, the three backends. Anthropic by A P "
          "I key: the key goes in this masked field, is stored in "
          "the application settings on this machine, and never "
          "appears in a ledger. Local, or OpenAI compatible: one "
          "box takes the server U R L, or U R L and model "
          "together. And Claude by subscription login, no key at "
          "all, through the Claude agent S D K, with the aliases "
          "fable, opus, sonnet and haiku; blank means the default. "
          "Connect. The status line pins the provider and the "
          "ledger file name, the provider row folds away, and on "
          "a first connection a hello card maps what the "
          "assistant can do. Then a real question: what lattice "
          "is loaded. The orb turns amber, and the reply is the "
          "model's live answer, correct because HELIX grounds it "
          "in the live application state: the deck, the element "
          "count, the results. The workbench, not the model's "
          "imagination.",
          min_s=8.0),
    Scene("033_capability_chip",
          "The question mark chip asks the assistant what it can "
          "do. A chip simply types a real request and sends it "
          "through the normal path, so the answer arriving is the "
          "model's own live summary from this recording, not a "
          "canned list. Read it as the table of contents for the "
          "rest of this episode.",
          min_s=6.0),
    Scene("040_instant",
          "Not every request deserves a model. We type, status. "
          "The answer lands in milliseconds and the transcript "
          "says why: instant, get status. Lattice fodo cell, 22 "
          "elements, 1.6 metres, results loaded. A small set of "
          "unambiguous read only phrasings match the whole "
          "utterance exactly and run the tool directly, "
          "deterministically. Precision over recall: a missed "
          "shortcut costs one model turn, a wrong one would hijack "
          "your question, so only exact phrasings and only read "
          "tier tools qualify. Then, summarise the beam, and the "
          "honesty story: the beam summary tool refuses, because "
          "an envelope run carries no particle distribution, so "
          "the request falls through to the model, which pulls "
          "the scalar summary instead, reports exit sigmas of "
          "about two point six eight millimetres at one M e V, "
          "and says which substitution it made. Finally, list "
          "plots: the catalogue of result windows, a dozen names "
          "at a time. The Instant commands checkbox turns the "
          "whole fast path off.",
          min_s=9.0),
    Scene("042_hud",
          "Responsiveness as a number. The latency readout in the "
          "status row, cropped twice from this recording. After a "
          "model turn it shows the time to the first token, three "
          "and a half seconds here. After an instant command, the "
          "milliseconds the tool took: four. Smoothness measured, "
          "not felt.",
          min_s=7.0),
    Scene("045_nav_pointing",
          "The assistant drives the interface you already know. "
          "Show the beam tab: instant, and the workbench follows. "
          "Show the R M S plot: instant again. The Results tab "
          "comes forward and the R M S window opens as its own "
          "floating window; we untick its aperture overlay so the "
          "millimetre scale envelope fills the axes. Nested views "
          "need a model turn: open the Lattice tab's Breakdown "
          "view, and the sub tab switches. And pointing: highlight "
          "the second quadrupole. The assistant resolves the name, "
          "selects the element in the Lattice tab, and moves the s "
          "cursor to its entrance. Highlight, set cursor and get G "
          "U I context are the pointing tools: read tier, wired "
          "through thread safe queued signals, never touching the "
          "physics core.",
          min_s=9.0),
    Scene("050_gate",
          "And now the most important thirty seconds of this "
          "episode: the safety model, on camera. On the left, the "
          "Lattice tab, first quadrupole selected, gradient five "
          "tesla per metre. We ask for nine. That is a mutation, "
          "and the assistant does not touch the machine. It "
          "proposes the call, the orb turns pink, awaiting "
          "confirm, and the gate appears: Confirm mutate, then the "
          "exact resolved call, tool, element, parameter and "
          "value, one per line. Only two buttons, Approve and "
          "Deny; there is no session pass for mutations. Nothing "
          "happens until a human clicks Approve. We approve, the "
          "tool runs, the reply confirms it, and then the extra "
          "honesty: it warns that the results on screen are now "
          "stale, and offers to rerun the envelope.",
          min_s=11.0),
    Scene("052_deny",
          "The same gate, the other button. We ask for the "
          "gradient to go back to five; the call is proposed, the "
          "gate appears, and we click Deny. The gradient stays at "
          "nine, checked in the lattice object after this "
          "recording, and the denial goes back to the model as an "
          "answer like any other, so it responds rather than "
          "guesses. A gate is never left hanging, either: an "
          "unanswered confirmation auto denies after a hundred "
          "and twenty seconds.",
          min_s=8.0),
    Scene("054_compute_gate",
          "The compute tier. We take the assistant up on its "
          "offer: rerun the envelope in the G U I. Run in G U I "
          "presses the same Run button you would, with your "
          "Numerics settings, results into the same tabs. A "
          "computation asks once, and this gate has grown a third "
          "button: Approve for the session. We click plain "
          "Approve. The run starts, the Results tab refreshes, and "
          "the exit tiles tell the story of a nine tesla per "
          "metre quadrupole in a five tesla per metre cell: sigma "
          "x from two point six seven five to four point seven "
          "one five millimetres, sigma y to five point five nine "
          "one. The loop the assistant opened, the assistant "
          "closed.",
          min_s=9.0),
    Scene("056_diagnose_notebook",
          "Session memory as physics tooling. Diagnose runs an "
          "ordered differential over the results, and its last "
          "check is the session ledger: it knows we changed this "
          "quadrupole minutes ago, and says so. Then we dictate a "
          "note. Notebook note appends it to a markdown lab "
          "notebook that survives across sessions; the most "
          "recent entries are folded into the assistant's context "
          "at every start up. And asking for the latest entry "
          "reads it straight back.",
          min_s=9.0),
    Scene("058_python_sandbox",
          "The Python chip. The assistant proposes real analysis "
          "code, and because run Python is compute tier, the gate "
          "appears with the code echoed for you to read. This "
          "time we click Approve for the session. Watch the auto "
          "approve checkbox along the bottom tick itself. The code "
          "runs in an isolated subprocess: an empty import path, "
          "the simulator itself hard blocked, no network, and a "
          "wall clock cap. It sees only the result columns it "
          "explicitly asked for: s, sigma x and sigma y. Its first "
          "two attempts trip over the sandbox's own conventions, "
          "and it fixes them itself; because compute is now "
          "approved for the session, the retries run with no "
          "further gate. Then the figure it saved comes back into "
          "the conversation as an image the assistant can see, "
          "and it reads the split between the two planes off its "
          "own plot.",
          min_s=9.0),
    Scene("060_nogate_refusal",
          "Session approval, proven. We ask for gradient "
          "sensitivities, a compute tool, and no gate appears: the "
          "call runs straight away, ledgered as approved for the "
          "session. And then the tool refuses. This deck carries "
          "a frequency card the differentiable matrix path cannot "
          "represent, so rather than return silently wrong "
          "derivatives it declines, names the reason, and names "
          "parameter scan as the fallback. Compute runs free now; "
          "mutations still always ask.",
          min_s=8.0),
    Scene("062_sight",
          "Sight. We ask whether the beam looks matched, and the "
          "assistant looks at the R M S plot itself: it opens the "
          "window, captures the rendered figure, and analyses the "
          "image. The capture is saved under assist captures and "
          "its thumbnail lands in the transcript. The verdict is "
          "its own, from the pixels: not matched. Sigma x pinches "
          "to about one millimetre mid cell and flares to nearly "
          "five at the exit while sigma y drifts up to roughly "
          "six; asymmetric and growing rather than periodic.",
          min_s=9.0),
    Scene("064_manual_search",
          "The manual is indexed locally. Ask about barge in, and "
          "search manual returns the sections it matched, and the "
          "answer cites the section titles it used, the G U I "
          "section and the voice section, with no network "
          "involved. "
          "Then we ask it to search the web. Refused: HELIX is "
          "offline by default, web search is disabled unless you "
          "enable it explicitly, and the assistant relays that "
          "refusal rather than quietly reaching out.",
          min_s=8.0),
    Scene("070_tour",
          "The Tour chip starts a guided tour: fifteen stations "
          "over the real tabs, each with pre written narration "
          "and, at many stops, a hands on exercise. Station one is "
          "the welcome. Now watch the station counter: next is an "
          "instant command, but only while a tour is active, so it "
          "can never hijack a normal question. Station two "
          "switches the workbench to the Lattice tab beside; next "
          "again, station three with its Try it exercise; stop "
          "the tour ends it. And watch the lamp between stations: "
          "the follow up microphone opens for ten seconds after "
          "each one, so by voice you just keep saying next.",
          min_s=9.0),
    Scene("072_drill",
          "The Drill chip starts a training drill: a hidden fault "
          "injected into the loaded lattice, and the truth stays "
          "outside the assistant's context until the debrief, so "
          "it can relay hints without being able to spoil the "
          "answer. Injecting the fault is a mutation, so even the "
          "trainer asks first. Approve. We ask for a hint, and the "
          "first rung of the ladder names the element family; two "
          "more rungs exist, which third of the line, then the "
          "class and rough position, each costing eight points of "
          "the economy score. We give up, and the debrief reveals "
          "the truth: quadrupole four, its gradient drifted from "
          "minus five to about minus four point three eight tesla "
          "per metre. The score, thirty nine point seven, "
          "combines diagnosis, speed and hint economy, and the "
          "lattice is restored bit for bit, checked against a "
          "snapshot of every quadrupole taken before the drill.",
          min_s=10.0),
    Scene("074_stop_queue",
          "The interrupt story. We ask for a long reply and, while "
          "it streams, send a second message. It is queued, not "
          "dropped: a small first in, first out inbox of three, "
          "and even an overflow is announced, never silent. Then "
          "Stop, or the Escape key. Streaming halts, speech is "
          "cut, a pending confirmation would be denied, and on "
          "this backend the server side generation itself is "
          "interrupted. Stop also drops the queue, and says so. "
          "The session stays usable: status answers instantly.",
          min_s=9.0),
    Scene("076_watch_options",
          "The options row in close up, after the session "
          "approval. Watch runs inspects every finished run, "
          "including ones you start yourself, with pure local "
          "numpy: transmission drops localised in s, sigma blow "
          "ups, emittance growth of two times or more, drift from "
          "a stored baseline; no tokens are spent until something "
          "fires. Alerts arrive as event lines, and with Narrate "
          "events on, an alert during idle gives the assistant an "
          "unprompted turn to speak up. Instant commands, Watch "
          "runs, Text and the text size persist across sessions; "
          "auto approve is deliberately per session.",
          min_s=8.0),
    Scene("078_voiceonly_close",
          "Untick Text for a voice only view: the transcript "
          "hides, the orb becomes the whole face, and the "
          "conversation keeps recording underneath; tick it again "
          "and the history is intact. The window's close button "
          "hides the panel rather than killing it, and with the "
          "panel out of the way the Lattice inspector is "
          "uncovered, reading nine for the quadrupole we edited. "
          "The assistant is an application level service: it "
          "keeps running, hands free listening included, and "
          "reopening it from the toolbar, or saying HELIX, brings "
          "the same conversation back. And the backend button "
          "reopens the provider row, so you can switch brains mid "
          "session.",
          min_s=9.0),
    Scene("079_voice",
          "Everything you watched was typed, but the panel is "
          "built for speech, and these are the real controls. Hold "
          "the microphone button, or hold Space with the input "
          "line unfocused, and talk; or toggle HELIX and just say "
          "the wake word, which must address the assistant: a "
          "mention mid sentence does not trigger it. A voice "
          "activity detector gates the microphone, Whisper turns "
          "speech to text on your machine, and the reply is "
          "synthesised locally with kokoro, this voice. Talk over "
          "it and it stops and listens; after it answers, a ten "
          "second follow up window listens without the wake word. "
          "Confirmations can be spoken too, matched locally, never "
          "by the model: yes or confirm, no or cancel, no outranks "
          "yes, anything unclear is asked again, so a misheard "
          "word can never approve an action. Spoken numbers are "
          "coarsened while the exact digits stay on screen. "
          "Nothing audio ever leaves the computer.",
          min_s=9.0),
    Scene("080_ledger",
          "Everything you just watched also left a record. This is "
          "the actual ledger of this recording session, selected "
          "rows: every tool call, its tier, who approved it, and "
          "its verdict. Fast path for the instant commands, user "
          "for the gates you saw clicked, denied where we clicked "
          "Deny, auto for read tools the model called on its own, "
          "and auto session once compute was approved for the "
          "session. Results the assistant writes carry assist "
          "attributes in their H D F 5 provenance, tracing every "
          "number back to its ledger entry.",
          min_s=9.0),
    Scene("082_replay_mcp",
          "A ledger can be replayed later without the model: no "
          "provider, no network, just the approved compute and "
          "mutate steps re executed in order, so anything the "
          "assistant did can be audited and reproduced. And the "
          "same registry, fifty three tools, is served over M C P, "
          "so external agents, including the one that helped build "
          "HELIX itself, can drive the simulator; read tools carry "
          "the read only hint and mutations the destructive hint, "
          "so the client's own permission prompts take over. The "
          "model proposes. Deterministic code checks. You decide.",
          min_s=9.0),
    Scene("090_finale",
          "And that is the series. Install and first beam. The "
          "tour. Every tab in depth. The physics of space charge. "
          "And the assistant you just watched working. Everything "
          "shown here lives in the manual's assistant chapter. The "
          "machine explained itself, and every claim you heard was "
          "checked against the pixels on screen before it was "
          "spoken. Thank you for watching. Now go run a beam.",
          min_s=7.0),
]


def scene(name: str) -> Scene:
    return next(sc for sc in SCENES if sc.name == name)


# ---------------------------------------------------------------------------
# cards that depend only on files (regenerated in both capture and reuse)
# ---------------------------------------------------------------------------
def _ledger_rows(led: Path) -> list[tuple[str, str]]:
    """Selected tool rows of a real ledger: cover every approved_by
    value that occurred (fast_path / user / denied / auto_session /
    auto), chronological, distinct tools within a class."""
    recs = []
    for line in led.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(line)
        except Exception:                                   # noqa: BLE001
            continue
        if r.get("event") == "tool" and r.get("tool"):
            recs.append(r)
    picked: list[dict] = []
    for who, n in (("fast_path", 3), ("user", 3), ("denied", 1),
                   ("auto_session", 2), ("auto", 2)):
        seen: set = set()
        for r in recs:
            if r.get("approved_by") != who or r["tool"] in seen:
                continue
            seen.add(r["tool"])
            picked.append(r)
            if len(seen) >= n:
                break
    picked.sort(key=lambda r: str(r.get("ts", "")))
    rows = [("out", f"{'tool':<20} {'tier':<8} {'approved_by':<13} status"),
            ("gap", "")]
    for r in picked[:11]:
        rows.append(("out", f"{r['tool']:<20} {r.get('tier', '?'):<8} "
                            f"{r.get('approved_by', '?'):<13} "
                            f"{r.get('status', '')}"))
    return rows


def make_cards(s: dict, ledger: str | None) -> None:
    from pipeline import cards
    cards.title_card(s["010_title"], "The Voice Assistant",
                     "Series finale — recorded live, nothing staged")
    cards.outro_card(s["090_finale"], [
        "Thirteen episodes — install to autonomy",
        "Every claim checked against the screen",
        "github.com/Accel-Toolkit/HELIX",
    ])
    led_lines: list = []
    if ledger and Path(ledger).exists():
        try:
            led_lines = _ledger_rows(Path(ledger))
        except Exception as exc:                            # noqa: BLE001
            print(f"[warn] ledger read failed: {exc}")
    if led_lines:
        cards.terminal_card(s["080_ledger"], [
            ("out", f"— {Path(ledger).name} · written during THIS "
                    "recording —"),
            ("gap", ""), *led_lines,
        ], title="assist_sessions ledger — selected rows")
    else:                       # honest fallback if no ledger was found
        cards.title_card(s["080_ledger"], "The Session Ledger",
                         "every call typed, tiered, logged — "
                         "replayable model-free",
                         kicker="AUDIT TRAIL")
    cards.terminal_card(s["082_replay_mcp"], [
        ("out", "— replay: the approved steps, re-executed with no model —"),
        ("cmd", "python -m linac_gen assist --replay "
                "runs/assist_sessions/<id>.jsonl"),
        ("gap", ""),
        ("out", "— serve the same 53 tools to an external agent over MCP —"),
        ("cmd", "python -m linac_gen assist --mcp <project>.lgproj"),
        ("gap", ""),
        ("out", ".mcp.json"),
        ("out", '{ "mcpServers": { "helix": {'),
        ("out", '      "command": "python",'),
        ("out", '      "args": ["-m", "linac_gen", "assist", "--mcp"] } } }'),
        ("gap", ""),
        ("out", "read tools: readOnlyHint    mutations: destructiveHint"),
    ], title="replay and MCP")


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------
def capture_visuals() -> dict:
    from PyQt6.QtCore import QEventLoop, QRect, Qt
    from PyQt6.QtGui import QColor, QImage, QPainter
    from PyQt6.QtWidgets import QApplication

    from pipeline import cards as _cards

    SHOTS.mkdir(parents=True, exist_ok=True)
    s = {sc.name: str(SHOTS / f"{sc.name}.png") for sc in SCENES}
    meta: dict = {"clips": {}, "ledger": None}

    app = QApplication.instance() or QApplication([])

    from linac_gen_gui.interphase.app import (InterphaseWindow,
                                              _parse_lattice_file)
    win = InterphaseWindow()
    win.resize(1920, 1080)
    win.show()

    def settle(n: int = 4) -> None:
        for _ in range(n):
            QApplication.processEvents(
                QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    # ---- machine: matched FODO with a fresh envelope run ------------
    deck = str(ROOT / "examples/fodo_cell.dat")
    lattice, _meta = _parse_lattice_file(deck)
    win.state.set_lattice(lattice, deck)
    from linac_gen.core.config import BeamConfig
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.matching import find_fodo_cells, find_matched_input_twiss
    ref = ReferenceParticle(species=PROTON, w_kin=1.0, frequency=352.21)
    tw = find_matched_input_twiss(lattice, ref, *find_fodo_cells(lattice)[0])
    win.state.set_beam_config(BeamConfig(
        species="proton", energy=1.0, frequency=352.21, current=0.0,
        alpha_x=tw["alpha_x"], beta_x=tw["beta_x"],
        alpha_y=tw["alpha_y"], beta_y=tw["beta_y"]))
    settle(8)
    win._run_envelope()
    for _ in range(40):
        settle(6)

    def quad(name: str):
        return next(e for e in win.state.lattice.elements if e.name == name)

    def quad_gradients() -> dict:
        return {e.name: float(e.gradient) for e in win.state.lattice.elements
                if type(e).__name__ == "Quadrupole"}

    def lattice_view(subtab: str = "Sequence", select: str = "QUAD_001"):
        win.show_tab("Lattice", subtab)
        win.state.set_selected(quad(select))
        settle(10)

    lattice_view()

    # ---- the real panel --------------------------------------------
    win._open_assistant()
    settle(14)
    panel = win._assistant_panel
    panel.resize(1240, 1000)
    panel._font_plus.click()
    panel._font_plus.click()
    settle(10)
    panel.grab().save(s["020_open"])

    # ---- crops & composite cards (real widgets, real labels) --------
    def crop_of(widgets, pad: int = 8) -> QImage:
        settle(3)
        img = panel.grab().toImage()
        r = QRect(widgets[0].geometry())
        for w in widgets[1:]:
            r = r.united(w.geometry())
        r = r.adjusted(-pad, -pad, pad, pad).intersected(img.rect())
        return img.copy(r)

    def stack_card(path: str, items, max_scale: float = 1.5) -> None:
        """Vertically stacked crops with small captions on a 1920x1080
        canvas (HELIX dark palette)."""
        from PyQt6.QtCore import QRectF
        canvas = QImage(1920, 1080, QImage.Format.Format_RGB32)
        canvas.fill(QColor(_cards.BG))
        p = QPainter(canvas)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        scaled = []
        for img, label in items:
            f = min(max_scale, 1800 / max(1, img.width()))
            si = img.scaled(int(img.width() * f), int(img.height() * f),
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
            scaled.append((si, label))
        total = sum(si.height() + 44 + 40 for si, _ in scaled) - 40
        y = max(40, (1080 - total) // 2)
        for si, label in scaled:
            p.setPen(QColor(_cards.ACCENT))
            p.setFont(_cards._font(22, bold=True))
            p.drawText(QRectF(0, y, 1920, 36),
                       Qt.AlignmentFlag.AlignHCenter, label)
            y += 44
            x = (1920 - si.width()) // 2
            p.fillRect(x - 4, y - 4, si.width() + 8, si.height() + 8,
                       QColor("#2b3b55"))
            p.drawImage(x, y, si)
            y += si.height() + 40
        p.end()
        canvas.save(path)

    status_widgets = [panel._status, panel._perf, panel._backend_btn]
    stack_card(s["022_anatomy"], [
        (crop_of(status_widgets + [panel._settings_box]),
         "status row  ·  provider row"),
        (crop_of(list(panel._chip_btns) + [panel._mic_btn, panel._stop_btn]),
         "quick-action chips  ·  input row"),
        (crop_of([panel._auto, panel._font_plus]), "options row"),
    ])
    stack_card(s["079_voice"], [
        (crop_of([panel._mic_btn, panel._stop_btn]),
         "the real voice controls (offscreen build: no microphone)"),
    ])

    # ---- recording machinery ---------------------------------------
    rec = Recorder(FRAMES, settle)

    def tr_text() -> str:
        try:
            return str(panel._transcript.toPlainText())
        except Exception:                                   # noqa: BLE001
            return ""

    def busy() -> bool:
        for w in (getattr(panel, "_worker", None),
                  getattr(panel, "_fast_worker", None)):
            try:
                if w is not None and w.isRunning():
                    return True
            except RuntimeError:
                pass
        return False

    def gate_up() -> bool:
        return bool(panel._btn_approve.isVisible())

    def panel_frame() -> QImage:
        return panel.grab().toImage()

    def visible_popups() -> list:
        out = []
        for _k, dlg in getattr(win.results_tab, "_popups", {}).items():
            try:
                if dlg.isVisible():
                    out.append(dlg)
            except RuntimeError:
                continue
        return out

    def _canvas() -> tuple:
        canvas = QImage(1920, 1080, QImage.Format.Format_RGB32)
        canvas.fill(QColor("#0b1220"))
        p = QPainter(canvas)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        return canvas, p

    def _draw_panel(p, width: int = 720) -> None:
        if not panel.isVisible():
            return
        pimg = panel.grab().toImage().scaledToWidth(
            width, Qt.TransformationMode.SmoothTransformation)
        px, py = 1920 - pimg.width() - 8, (1080 - pimg.height()) // 2
        p.fillRect(px - 5, py - 5, pimg.width() + 10, pimg.height() + 10,
                   QColor("#2b3b55"))
        p.drawImage(px, py, pimg)

    def combo_frame() -> QImage:
        """Workbench (left) + panel (right); any OPEN plot popup — a
        TOP-LEVEL window win.grab() never contains — is overlaid as
        the floating window it is."""
        canvas, p = _canvas()
        wimg = win.grab().toImage().scaledToWidth(
            1450, Qt.TransformationMode.SmoothTransformation)
        wy = (1080 - wimg.height()) // 2
        p.drawImage(0, wy, wimg)
        for dlg in visible_popups()[:1]:
            dimg = dlg.grab().toImage().scaledToWidth(
                760, Qt.TransformationMode.SmoothTransformation)
            dx, dy = 36, wy + wimg.height() - dimg.height() - 36
            p.fillRect(dx - 5, dy - 5, dimg.width() + 10,
                       dimg.height() + 10, QColor("#4b5f85"))
            p.drawImage(dx, dy, dimg)
        _draw_panel(p, 720)
        p.end()
        return canvas

    def sight_frame() -> QImage:
        """Panel (right) + the plot window LARGE (left) once it is
        open; the workbench until then."""
        canvas, p = _canvas()
        pops = visible_popups()
        if pops:
            dimg = pops[0].grab().toImage().scaledToWidth(
                1120, Qt.TransformationMode.SmoothTransformation)
            dx, dy = 24, (1080 - dimg.height()) // 2
            p.fillRect(dx - 5, dy - 5, dimg.width() + 10,
                       dimg.height() + 10, QColor("#4b5f85"))
            p.drawImage(dx, dy, dimg)
        else:
            wimg = win.grab().toImage().scaledToWidth(
                1140, Qt.TransformationMode.SmoothTransformation)
            p.drawImage(12, (1080 - wimg.height()) // 2, wimg)
        _draw_panel(p, 750)
        p.end()
        return canvas

    def orb_frame() -> QImage:
        img = panel.grab().toImage()
        r = QRect(panel._orb.geometry()).united(panel._lamp.geometry())
        h = r.height()
        w = int(min(r.width(), 2.4 * h))
        rect = QRect(r.center().x() - w // 2, r.top() - 6, w, h + 12) \
            .intersected(img.rect())
        crop = img.copy(rect)
        f = min(2.5, 1880 / crop.width(), 1040 / crop.height())
        big = crop.scaled(int(crop.width() * f), int(crop.height() * f),
                          Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)
        canvas, p = _canvas()
        p.drawImage((1920 - big.width()) // 2, (1080 - big.height()) // 2,
                    big)
        p.end()
        return canvas

    def ask(grab, text: str) -> None:
        rec.type_into(grab, panel._input, text)
        panel._send_btn.click()

    def wait_turn(grab, cap_s: float, stable_s: float = 2.0,
                  on_gate: str | None = "approve") -> bool:
        """Record until the turn is over (no worker running, no gate
        pending, transcript grew) and stayed so for stable_s.  An
        UNPLANNED gate is answered per on_gate (a real click, on
        camera) so a blocked approver can never wedge the build."""
        base = len(tr_text())
        t0 = _time.monotonic()
        while _time.monotonic() - t0 < cap_s:
            ok = rec.wait_until(
                grab, lambda: (not busy()) and (not gate_up())
                and len(tr_text()) > base,
                cap_s=max(1.0, cap_s - (_time.monotonic() - t0)),
                stable_s=stable_s)
            if ok:
                return True
            if gate_up() and on_gate:
                print(f"[warn] unplanned gate in {rec._st['dir'].name} "
                      f"-> {on_gate}")
                rec.hold(grab, 2.0)
                (panel._btn_approve if on_gate == "approve"
                 else panel._btn_deny).click()
                continue
            return False
        return False

    def wait_gate(grab, cap_s: float) -> bool:
        ok = rec.wait_until(grab, gate_up, cap_s)
        if not ok:
            print(f"[warn] confirmation gate never appeared "
                  f"({rec._st['dir'].name})")
        return ok

    def hud_crop() -> QImage:
        return crop_of(status_widgets)

    # ---- 025: orb state montage (real widget, real states) ----------
    rec.start("025_orb_states")
    for st_name in ("idle", "listening", "thinking", "responding",
                    "awaiting-confirm", "error", "off"):
        panel._set_state(st_name)
        rec.hold(orb_frame, 1.6)
    panel._set_state("idle")
    rec.hold(orb_frame, 0.6)
    rec.finish(scene("025_orb_states"))

    # ---- 030: backend tour -> connect claude_sdk -> LIVE turn -------
    rec.start("030_live")
    rec.hold(panel_frame, 1.0)
    for prov in ("anthropic", "openai", "claude_sdk"):
        idx = next(i for i in range(panel._provider.count())
                   if panel._provider.itemData(i) == prov)
        panel._provider.setCurrentIndex(idx)
        rec.hold(panel_frame, 2.6)
    for j in range(1, panel._model_edit.count()):     # the aliases
        panel._model_edit.setCurrentIndex(j)
        rec.hold(panel_frame, 0.9)
    panel._model_edit.setCurrentIndex(0)              # blank = default
    rec.hold(panel_frame, 1.2)
    panel._connect_btn.click()
    rec.hold(panel_frame, 4.0)
    ask(panel_frame, "What lattice is loaded right now? One short sentence.")
    wait_turn(panel_frame, cap_s=120)
    rec.hold(panel_frame, 3.0)
    rec.finish(scene("030_live"))
    hud_after_model = hud_crop()

    # ---- 033: the "What can you do" chip ----------------------------
    rec.start("033_capability_chip")
    rec.hold(panel_frame, 1.0)
    chip = next(b for b in panel._chip_btns if "What can you do" in b.text())
    chip.click()
    wait_turn(panel_frame, cap_s=120)
    rec.hold(panel_frame, 3.0)
    rec.finish(scene("033_capability_chip"))

    # ---- 040: instant commands (no model round trip) ---------------
    rec.start("040_instant")
    rec.hold(panel_frame, 1.0)
    ask(panel_frame, "status")
    wait_turn(panel_frame, cap_s=20, stable_s=1.5)
    hud_after_instant = hud_crop()
    rec.hold(panel_frame, 2.0)
    ask(panel_frame, "summarize the beam")
    wait_turn(panel_frame, cap_s=90, stable_s=1.5)
    rec.hold(panel_frame, 2.5)
    ask(panel_frame, "list plots")
    wait_turn(panel_frame, cap_s=20, stable_s=1.5)
    rec.hold(panel_frame, 2.5)
    rec.finish(scene("040_instant"))
    stack_card(s["042_hud"], [
        (hud_after_model, "status row after a model turn"),
        (hud_after_instant, "status row after an instant command"),
    ], max_scale=1.5)

    # ---- 045: navigation + plot window + subtab + pointing ---------
    rec.start("045_nav_pointing")
    rec.hold(combo_frame, 1.2)
    ask(combo_frame, "show the beam tab")
    wait_turn(combo_frame, cap_s=20, stable_s=1.2)
    rec.hold(combo_frame, 1.5)
    ask(combo_frame, "show the rms plot")
    wait_turn(combo_frame, cap_s=20, stable_s=1.2)
    rec.hold(combo_frame, 1.5)
    pop = getattr(win.results_tab, "_popups", {}).get("rms")
    if pop is not None and hasattr(pop, "_chk_ap"):
        pop._chk_ap.click()                       # aperture overlay off
        rec.hold(combo_frame, 2.0)
    ask(combo_frame, "Open the Lattice tab's Breakdown view.")
    wait_turn(combo_frame, cap_s=90)
    rec.hold(combo_frame, 2.0)
    ask(combo_frame, "Highlight the second quadrupole.")
    wait_turn(combo_frame, cap_s=90)
    rec.hold(combo_frame, 3.0)
    rec.finish(scene("045_nav_pointing"))
    print(f"[verify] selected after highlight: "
          f"{getattr(win.state.selected, 'name', None)}  "
          f"s_cursor={win.state.s_cursor}")
    if pop is not None:
        pop.hide()
    lattice_view()                                 # back to QUAD_001

    # ---- 050: mutate ask -> REAL confirmation gate -> approve ------
    rec.start("050_gate")
    rec.hold(combo_frame, 2.0)
    ask(combo_frame, "Set the gradient of QUAD_001 to 9 tesla per metre.")
    if wait_gate(combo_frame, cap_s=120):
        rec.hold(combo_frame, 3.5)          # let the viewer read the gate
        panel._btn_approve.click()
    wait_turn(combo_frame, cap_s=90)
    rec.hold(combo_frame, 3.0)
    rec.finish(scene("050_gate"))
    print(f"[verify] QUAD_001 gradient after approve: "
          f"{quad('QUAD_001').gradient}")

    # ---- 052: the same gate, Deny --------------------------------
    rec.start("052_deny")
    rec.hold(combo_frame, 1.5)
    ask(combo_frame, "Set the gradient of QUAD_001 back to 5 tesla per metre.")
    if wait_gate(combo_frame, cap_s=120):
        rec.hold(combo_frame, 3.0)
        panel._btn_deny.click()
    wait_turn(combo_frame, cap_s=90, on_gate="deny")
    rec.hold(combo_frame, 3.0)
    rec.finish(scene("052_deny"))
    print(f"[verify] QUAD_001 gradient after deny: "
          f"{quad('QUAD_001').gradient}")

    # ---- 054: compute gate (3 buttons) -> plain Approve -> GUI run --
    win.show_tab("Results")
    settle(10)
    rec.start("054_compute_gate")
    rec.hold(combo_frame, 1.5)
    ask(combo_frame, "Yes. Rerun the envelope in the GUI, the same way "
                     "the Run button does.")
    if wait_gate(combo_frame, cap_s=120):
        rec.hold(combo_frame, 3.5)
        panel._btn_approve.click()
    wait_turn(combo_frame, cap_s=150)
    rec.wait_until(combo_frame, lambda: not win.state.running, cap_s=60,
                   stable_s=1.0)
    rec.hold(combo_frame, 3.0)
    rec.finish(scene("054_compute_gate"))
    print(f"[verify] auto-approve checkbox after 054: "
          f"{panel._auto.isChecked()}")

    # ---- 056: diagnose (ledger-aware) + notebook -------------------
    rec.start("056_diagnose_notebook")
    rec.hold(panel_frame, 1.0)
    ask(panel_frame, "Diagnose the current results, and mention any "
                     "recent parameter changes.")
    wait_turn(panel_frame, cap_s=120)
    rec.hold(panel_frame, 2.5)
    ask(panel_frame, "Note down in the lab notebook that QUAD_001 was "
                     "raised to 9 T/m for the confirmation-gate demo.")
    wait_turn(panel_frame, cap_s=90)
    rec.hold(panel_frame, 2.0)
    ask(panel_frame, "What is the most recent entry in the lab notebook? "
                     "Quote it.")
    wait_turn(panel_frame, cap_s=90)
    rec.hold(panel_frame, 3.0)
    rec.finish(scene("056_diagnose_notebook"))

    # ---- 058: Python chip -> compute gate -> Approve (session) -----
    rec.start("058_python_sandbox")
    rec.hold(panel_frame, 1.0)
    chip = next(b for b in panel._chip_btns if "Python" in b.text())
    chip.click()
    if wait_gate(panel_frame, cap_s=120):
        rec.hold(panel_frame, 4.0)          # the echoed code is readable
        panel._btn_always.click()           # -> _auto ticks itself
        rec.hold(panel_frame, 1.5)
    wait_turn(panel_frame, cap_s=180)
    rec.hold(panel_frame, 4.0)
    rec.finish(scene("058_python_sandbox"))
    print(f"[verify] auto-approve checkbox after 058: "
          f"{panel._auto.isChecked()}")

    # ---- 060: compute with NO gate -> honest refusal ---------------
    rec.start("060_nogate_refusal")
    rec.hold(panel_frame, 1.0)
    ask(panel_frame, "Use grad_sensitivities to rank the quadrupole "
                     "gradients by their effect on the exit sigma y. If "
                     "it refuses, just tell me why in one sentence.")
    wait_turn(panel_frame, cap_s=120)
    rec.hold(panel_frame, 3.5)
    rec.finish(scene("060_nogate_refusal"))

    # ---- 062: sight — look_at_plot on the RMS window ---------------
    rec.start("062_sight")
    rec.hold(sight_frame, 1.0)
    ask(sight_frame, "Look at the RMS envelope plot. Does the beam look "
                     "matched? Answer in two sentences.")
    wait_turn(sight_frame, cap_s=150)
    rec.hold(sight_frame, 4.0)
    rec.finish(scene("062_sight"))
    caps = sorted((ROOT / "runs" / "assist_captures").glob("plot_*"))
    print(f"[verify] plot captures on disk: {[c.name for c in caps[-2:]]}")
    for dlg in visible_popups():
        dlg.hide()

    # ---- 064: manual search + the web_search refusal ---------------
    rec.start("064_manual_search")
    rec.hold(panel_frame, 1.0)
    ask(panel_frame, "Search the manual for barge-in and summarise it in "
                     "two sentences, citing the section title.")
    wait_turn(panel_frame, cap_s=120)
    rec.hold(panel_frame, 3.0)
    ask(panel_frame, "Search the web for kokoro voices.")
    wait_turn(panel_frame, cap_s=90)
    rec.hold(panel_frame, 3.0)
    rec.finish(scene("064_manual_search"))

    # ---- 070: guided tour — chip, instant next/next/stop -----------
    win.show_tab("Results")
    settle(8)
    rec.start("070_tour")
    rec.hold(combo_frame, 1.0)
    chip = next(b for b in panel._chip_btns if "Tour" in b.text())
    chip.click()
    wait_turn(combo_frame, cap_s=120)
    rec.hold(combo_frame, 2.5)
    ask(combo_frame, "next")
    wait_turn(combo_frame, cap_s=30, stable_s=1.5)
    rec.hold(combo_frame, 3.0)
    ask(combo_frame, "next")
    wait_turn(combo_frame, cap_s=30, stable_s=1.5)
    rec.hold(combo_frame, 3.0)
    ask(combo_frame, "stop the tour")
    wait_turn(combo_frame, cap_s=30, stable_s=1.5)
    rec.hold(combo_frame, 2.5)
    rec.finish(scene("070_tour"))

    # ---- 072: training drill — gate, hint, give up, restore --------
    snap = quad_gradients()
    lattice_view()
    rec.start("072_drill")
    rec.hold(combo_frame, 1.0)
    chip = next(b for b in panel._chip_btns if "Drill" in b.text())
    chip.click()
    if wait_gate(combo_frame, cap_s=120):
        rec.hold(combo_frame, 3.0)
        panel._btn_approve.click()
    wait_turn(combo_frame, cap_s=150)
    rec.hold(combo_frame, 2.5)
    ask(combo_frame, "Give me a hint.")
    wait_turn(combo_frame, cap_s=120)
    rec.hold(combo_frame, 2.5)
    ask(combo_frame, "I give up.")
    wait_turn(combo_frame, cap_s=120)
    rec.hold(combo_frame, 4.0)
    rec.finish(scene("072_drill"))
    after = quad_gradients()
    same = all(after.get(k) == v for k, v in snap.items()) \
        and set(after) == set(snap)
    print(f"[verify] drill restore bit-exact: {same}  "
          f"before={snap}  after={after}")
    meta["drill_restore_exact"] = bool(same)

    # ---- 074: queued inbox + Stop --------------------------------
    rec.start("074_stop_queue")
    rec.hold(panel_frame, 1.0)
    base = len(tr_text())
    ask(panel_frame, "Explain every element of this lattice in detail, "
                     "one by one, with a full paragraph for each of the "
                     "22 elements.")
    rec.wait_until(panel_frame,
                   lambda: bool(getattr(panel, "_streaming", False))
                   or len(tr_text()) > base + 400, cap_s=60)
    rec.hold(panel_frame, 1.5)
    ask(panel_frame, "What is the beam energy?")        # -> queued
    rec.hold(panel_frame, 2.5)
    panel._stop_btn.click()
    rec.wait_until(panel_frame, lambda: not busy(), cap_s=40, stable_s=1.0)
    rec.hold(panel_frame, 1.5)
    ask(panel_frame, "status")
    wait_turn(panel_frame, cap_s=30, stable_s=1.5)
    rec.hold(panel_frame, 3.0)
    rec.finish(scene("074_stop_queue"))

    # ---- 076: the options row (after the session approval) --------
    stack_card(s["076_watch_options"], [
        (crop_of([panel._auto, panel._font_plus]),
         "options row — as left by this recording"),
    ], max_scale=1.5)

    # ---- 078: voice-only view, close hides, reopen, backend… -------
    rec.start("078_voiceonly_close")
    rec.hold(panel_frame, 1.0)
    panel._text_chk.click()                       # voice-only view
    rec.hold(panel_frame, 3.0)
    panel._text_chk.click()                       # transcript back
    rec.hold(panel_frame, 2.0)
    rec.hold(combo_frame, 1.0)
    panel.close()                                 # closeEvent -> hide
    rec.hold(combo_frame, 2.5)
    win._open_assistant()                         # same panel, shown
    rec.hold(combo_frame, 2.5)
    panel._backend_btn.click()                    # provider row back
    rec.hold(combo_frame, 2.5)
    rec.finish(scene("078_voiceonly_close"))
    print(f"[verify] panel visible after reopen: {panel.isVisible()}  "
          f"provider row visible: {panel._settings_box.isVisible()}")

    # ---- the session's REAL ledger ---------------------------------
    led = None
    try:
        led = panel._session.ledger.path
    except Exception:                                       # noqa: BLE001
        cands = sorted((ROOT / "runs" / "assist_sessions").glob("*.jsonl"),
                       key=lambda p: p.stat().st_mtime)
        led = cands[-1] if cands else None
    meta["ledger"] = str(led) if led else None
    make_cards(s, meta["ledger"])

    for sc in SCENES:
        if sc.frames_dir:
            meta["clips"][sc.name] = sc.fps
        else:
            sc.image = s[sc.name]
    META.write_text(json.dumps(meta, indent=1))
    # the assistant window: hide (QDialog close), never win.close()
    try:
        panel.close()
        settle(20)
    except Exception as exc:                            # noqa: BLE001
        print(f"[warn] panel close: {exc}")
    return meta


_APP = None


def restore_visuals() -> dict:
    """EP13_REUSE=1: re-render from the last capture's clips/stills."""
    from PyQt6.QtWidgets import QApplication
    global _APP                       # keep the QApplication alive
    _APP = QApplication.instance() or QApplication([])
    meta = json.loads(META.read_text())
    s = {sc.name: str(SHOTS / f"{sc.name}.png") for sc in SCENES}
    make_cards(s, meta.get("ledger"))
    for sc in SCENES:
        if sc.name in meta["clips"]:
            sc.frames_dir = str(FRAMES / sc.name)
            sc.fps = float(meta["clips"][sc.name])
        else:
            sc.image = s[sc.name]
    print(f"[reuse] {len(meta['clips'])} clips + stills from {META}")
    return meta


def main() -> None:
    if os.environ.get("EP13_REUSE") == "1" and META.exists():
        restore_visuals()
    else:
        capture_visuals()
    info = build_video(SCENES, WORK, OUT)
    print(f"rendered {info['mp4']}  ({info['duration']:.1f} s)")
    print(f"captions {info['srt']}")
    sys.stdout.flush()
    os._exit(0)          # never hang on stray backend threads


if __name__ == "__main__":
    main()
