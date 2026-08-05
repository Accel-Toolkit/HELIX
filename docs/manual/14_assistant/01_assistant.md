# AI Assistant (optional)

HELIX ships an **optional** natural-language assistant that can run
simulations and matches, and answer questions about the
results you have generated — driving the *same* audited code you use
by hand.  It is entirely opt-in:

!!! warning "Completely optional — the app never depends on it"
    With **no API key and no network** HELIX behaves exactly as it
    always has.  The assistant subpackage (`linac_gen.assist`) is
    imported by nothing at startup; it uses only the Python standard
    library (a small `urllib` HTTP client — **no SDK dependency**);
    and it opens no network connection until you explicitly send it a
    message.  Deleting the `assist/` directory leaves the rest of
    HELIX fully working (there is a regression test that proves it).

## Backends — cloud, keyless subscription, or fully local

The assistant supports three backends, chosen from your configuration (or
the panel's **backend** selector):

* **Claude (subscription login — no API key)** — reasons through the
  **Claude Agent SDK**, which reuses your existing **Claude Code login**;
  no API key is read or stored. This is the in-GUI equivalent of the MCP
  route below: real Claude, on your subscription, right in the panel.
  Requires the optional `claude-agent-sdk` and a logged-in `claude` CLI
  (`pip install linac_gen[assist-sdk]`; install Claude Code and log in
  once). Needs network for reasoning; no local-model fallback.
* **Anthropic** Messages API (cloud) — set `HELIX_ASSIST_API_KEY`
  (or the standard `ANTHROPIC_API_KEY`). Pay-per-token; no SDK needed.
* **Any OpenAI-compatible endpoint** — set `HELIX_ASSIST_BASE_URL`.
  This covers a **local** model server such as
  [ollama](https://ollama.com) (`http://localhost:11434/v1`),
  `llama.cpp`, or vLLM, so the assistant can run **fully offline with
  no API key**, matching HELIX's offline ethos.

Pick by what you have: a **Claude subscription + Claude Code** → the first,
keyless option; an **API key** → the second; **no key and privacy/offline
needs** → a local server. All three drive the identical audited tools with
the identical tier-based confirmation.

```bash
# cloud
export ANTHROPIC_API_KEY=sk-...
python -m linac_gen assist examples/piplattice/fnalscl.lgproj

# fully local (ollama), no key
export HELIX_ASSIST_BASE_URL=http://localhost:11434/v1
export HELIX_ASSIST_MODEL=llama3.1
python -m linac_gen assist examples/pipii/btl/btl.lgproj
```

Nothing configured?  `python -m linac_gen assist` prints setup guidance
and exits — it never breaks.

## What it can do — and what it will refuse

The assistant's entire capability is a fixed registry of tools, each a
thin wrapper over an existing HELIX API.  **There is no shell, no
`eval`, and it can never write files.**  It *can* read local files —
read-only and windowed (`read_file`, defaulting to the loaded lattice
`.dat`) — so the raw element cards you see in the deck are exactly what
it sees.  It also understands them the way you do: `describe_lattice`
counts hardware by decoded field **channels** (an RF-channel FIELD_MAP
is a cavity even when parked at `ke=0`; a static-B map is a solenoid),
and `list_lattice_elements` shows each element's kind, decoded field
type, geom code and key parameters.  Tools are tiered:

| Tier | Examples | Confirmation |
|---|---|---|
| **read** | status, lattice info + hardware rollup, element table, raw-file window, query a result value, beam-parameter table, provenance, matched input Twiss | auto |
| **compute** | run envelope / multiparticle / match, parameter scan, compare to TraceWin | confirmed (session auto-approve optional) |
| **mutate** | load a lattice, set a beam field or element parameter, write results | **always confirmed** |

Every confirmation **echoes the exact resolved call with units** before
anything runs — important because HELIX's conventions are exacting
(mm, deg·MeV, and α_z = −TraceWin's sign).  Long simulations run in the
background and return a job you can poll or cancel.

When you ask for something unphysical or unrepresentable, the assistant
relays HELIX's own **loud refusal** (the same k2 / kaz-kbz / halo-CSR
guards the manual documents elsewhere) rather than guessing.

## Audit trail and replay

Every session writes a JSONL **ledger** under
`<calc-dir>/assist_sessions/` recording each tool call, its exact
parameters, tier, who approved it, status and duration — **never** API
keys.  Assistant-initiated result files get `assist_*` attributes in
their HDF5 provenance group, so any number the assistant produced is
traceable back to the exact ledger entry.

A ledger can be **replayed** with no LLM and no network, re-executing
its approved compute/mutate calls in order — deterministic audit:

```bash
python -m linac_gen assist --replay runs/assist_sessions/<id>.jsonl
```

## Using your Claude subscription (MCP server)

Anthropic's policy does **not** permit a third-party app to offer
claude.ai / Pro-Max **login** as an in-app option, so the built-in
panel above authenticates with an API key or a local model.  To use
your **own Claude subscription** instead, run HELIX as an **MCP
server** and drive it from Claude Code or Claude Desktop, which are
already logged in with your subscription:

```bash
# HELIX exposes its tools over stdio — no API key, no network from
# HELIX's side (auth lives entirely in the Claude client)
python -m linac_gen assist --mcp examples/pipii/btl/btl.lgproj
```

Register it with Claude Code (`.mcp.json` in your project, or the
global config):

```json
{
  "mcpServers": {
    "helix": {
      "command": "python",
      "args": ["-m", "linac_gen", "assist", "--mcp"]
    }
  }
}
```

Then, inside Claude Code, ask it to *"load the BTL lattice and run the
envelope"* — Claude calls the HELIX tools with your subscription, and
**Claude Code's own permission prompts** handle confirmation (HELIX
annotates read tools `readOnlyHint` and mutate tools `destructiveHint`
so reads auto-approve and writes/runs ask first).  In this mode HELIX
is a *tool provider*, not an auth broker — the sanctioned way to pair
a Claude subscription with the simulator.  Requires the optional
`mcp` package: `pip install linac_gen[assist-mcp]`.

## GUI

The desktop app exposes the assistant under **Tools → Assistant…** as a
non-modal panel.  On first open a short **capabilities card** lists what
it can do, and a row of **quick-action chips** sits above the input —
**🎓 Tour** (the 15-station guided walkthrough of HELIX), **🧩 Drill**
(a hidden-fault training exercise), **🐍 Python** (sandboxed analysis
of the current results), and **❓ What can you do** — each one sends
the corresponding request through the normal chat path, so nothing is
hidden behind knowing-what-to-ask.  While the brain and voice models
warm up in the background, anything you type is queued, not lost.

**Gradient sensitivities.**  On linear decks (e.g. the BTL) the
`grad_sensitivities` tool answers "which knob most affects σ_y at the
exit?" *exactly*: one torch autograd reverse pass over the
differentiable transfer-matrix path returns d(exit σ)/d(knob) for
every quad gradient, solenoid field and dipole angle at once, ranked
and signed.  It refuses decks the differentiable path cannot represent
faithfully (field maps, RF gaps, FREQ/energy cards) rather than
returning silently-wrong numbers — use `parameter_scan` there.

**Instant commands.**  Unambiguous read-only requests skip the model
entirely and execute in milliseconds: "status", "summarize the
results", "show the RMS plot", "switch to the Results tab", "list
runs" — and, during a guided tour, bare "next" / "back" / "stop the
tour".  The call is ledgered like any other (approved-by
`fast_path`), the model is quietly told about it at its next turn,
and anything nuanced still goes to the model — a missed shortcut
costs one model turn, a wrong one would hijack your question, so the
matcher insists on exact phrasings and read-tier tools only.  The
**Instant commands** checkbox turns the feature off.  A small **⏱
latency readout** in the status row shows time-to-first-token (or the
instant-command round trip) and speech-to-text time, so
responsiveness is a number, not a feeling.

The panel is a designed chat, not a log window: **rendered markdown** replies in role-styled cards (your turns accent-barred, tool traces and events as muted metadata), **syntax-highlighted code blocks**, and **inline thumbnails of every figure the assistant looks at** (click to open the saved capture).  Around it: a large **3-D state orb**
(idle · thinking · responding · listening · awaiting-confirm · error) —
a glossy sphere that pulses and changes colour with what the assistant
is doing, and whose equalizer ring **dances live with your voice the
moment you speak** in any listening state, so you always know it hears
you.  Below it: a live-streaming transcript (the **Text** checkbox
hides it for a voice-only view — the orb, voice controls and approval
strip stay, and the conversation keeps recording underneath), an input
box, a **■ Stop** button, an approve/deny strip that shows the echoed
call, a live progress readout for long runs, and a settings affordance
for the provider/model/key (stored in the app's settings, never in the
ledger).  The provider row hides once
connected — the **backend…** button next to the status line re-opens it
at any time, so you can switch between the keyless subscription login,
an Anthropic API key, or a local server and press **Connect**; the
running session is closed and replaced cleanly.  The **model** box is an
editable dropdown that picks the model for whichever backend you chose:
it suggests Anthropic model IDs for the API-key backend (blank =
`claude-sonnet-5`; typed gateway aliases like `azure/claude-sonnet-4-6`
also work), the `fable` / `opus` / `sonnet` / `haiku` aliases for the
subscription login, and — for a local backend with a known server URL —
it **asks the
server for its installed models** (ollama's `/v1/models`) and lists
them.  Free text always wins over the suggestions, and for a local
server you can still type `URL`, `model`, or `URL model` in one line
(e.g. `http://localhost:11434/v1 qwen2.5`; a bare model name reuses the
saved URL).  Actions that change
session state flow through the app's normal state signals, so the tabs
stay in sync.

**Hands-free voice.**  Beyond hold-to-talk (the 🎤 button or **holding
SPACE** with the input line unfocused), the **👂 HELIX** toggle keeps
the microphone open — fully locally — and wakes the assistant when you
say **"HELIX"** (a rising chime confirms; common mishearings are
accepted, and detection pauses while the assistant itself speaks so it
can never wake itself).  After it finishes *speaking* a reply to a
voice turn — and after **every station of an active guided tour**, no
matter how the tour was started — the mic reopens for ~10 s
(`HELIX_FOLLOWUP_S`, 0 disables) — just keep talking ("next", "back",
or a question), no wake word and no clicking needed.  You can also **talk over
it**: sustained speech while it talks cuts the audio and takes the
floor (`HELIX_BARGE_IN=0` disables).  While the wake mic is armed,
push-to-talk records through the *same* stream — one microphone owner,
always.

!!! tip "Better hearing with silero-VAD (optional)"
    Drop the ~2 MB `silero_vad.onnx` model into
    `~/.helix/assistant_models/` (next to the kokoro voice files) and
    every listening gate — wake detection, capture endpointing, the
    follow-up window, barge-in — upgrades from RMS loudness thresholds
    to a real speech probability: fans and typing stop triggering
    anything, and barge-in fires only on something that is loud AND
    speech (a dropped mug no longer cuts the assistant off; its own
    speaker bleed is still separated by the level calibration).
    With the model present, wake detection also becomes
    **event-driven**: Whisper runs the moment your utterance *ends*
    (VAD offset) instead of on a fixed 1-second cadence — saying
    "HELIX" is recognized noticeably faster, and continuous speech is
    still checked every ~2.5 s so the word is never missed
    mid-sentence.  Without the model, or with `HELIX_VAD=0`, the
    original RMS gates and polling cadence are used unchanged.

!!! note "Sleep/wake resilience"
    If macOS sleeps while a microphone is open, CoreAudio can leave
    that stream permanently wedged.  HELIX detects this on teardown and
    *parks* the wedged stream (it is never closed underneath an
    in-flight read — doing so crashes the process) and simply opens a
    fresh one on the next voice action.  If the stream instead *dies*
    (device change, unplugged headset, wake from sleep), the assistant
    notices, prints "microphone stream lost — reopening", and re-arms
    hands-free listening automatically; push-to-talk also falls back to
    its own fresh recording stream whenever the shared wake stream is
    not running.  If voice still seems unresponsive, toggle **👂
    HELIX** off and on again.

**Confirming by voice.**  When a compute/mutate confirmation appears
with voice active, the assistant speaks a short echo and listens: say
**"yes"/"confirm"/"go ahead"** or **"no"/"cancel"**.  The words are
matched *locally* (the model is never part of the decision), "no"
outranks "yes" in a mixed utterance, and anything unclear gets "please
say confirm, or cancel" — a misheard word can never approve an action.
Units are spoken as words (millimeters, M-e-V, pi millimeter
milliradian), and exact digits always stay on screen.

**Stopping a reply.**  The **■ Stop** button — or **Esc** — interrupts
the current turn: streaming halts, speech is cut, a pending
confirmation is denied, and on the keyless Claude backend the
server-side generation itself is interrupted.  The session stays
usable; just ask again.  (Esc used to close the panel; it now stops
instead.)  A message sent while a turn is still running is **queued,
not dropped** — it shows as `▷ … queued` and runs the moment the turn
ends.

**Closing the window.**  The window's close button **hides** the panel
— the assistant keeps running in the background (hands-free listening
included), and saying "HELIX" or reopening it from the toolbar brings
the same conversation back.  The assistant only shuts down with the
application.

**Run watching (proactive).**  With **Watch runs** ticked (the
default), *every* finished run — assistant-initiated **or one you
started yourself** — is inspected instantly by pure local numpy:
transmission drops vs the previous run (loss onset localized in s),
σ blow-ups, ≥2× emittance growth, and drift off the stored anomaly
baseline.  Alerts appear as `‣ [event]` lines (and are spoken when
narration is on): *"run watch: transmission fell 97.30 % → 95.00 %
(loss starts near s = 218.8 m)"*.  Each channel fires once and re-arms
on recovery or when the level stabilizes, with a rate limit so scans
can't spam; a baseline from a different lattice is announced once,
never silently ignored.  No tokens are spent until something fires.

**Machine events & narration.**  Job completions and run-watch alerts
appear in the transcript as `‣ [event]` lines the moment they happen.
With **Narrate events** ticked, an event arriving while the assistant is
idle gives it an unprompted turn to react — with **Speak replies** on,
it speaks up on its own ("the multiparticle run just finished, the
transmission held at 97.3 %").  Events are injected into the model as
clearly-labelled system notes, never as your words.

**Confirmation timeout.**  An unanswered confirmation auto-denies after
120 s (`AssistConfig.confirm_timeout_s`; 0 waits forever) — a forgotten
prompt can't leave a worker blocked all night.  The denial is reported
to the model like any other, and the strip closes with a note.

**Sight — the assistant can SEE figures.**  Ask *"look at the phase-space
plot — is the beam matched?"* and it captures the rendered figure
(`look_at_plot` / `look_at_screen`) and analyses the image itself —
matching, halo, filamentation, oscillation patterns.  Captures are saved
under `<calc-dir>/assist_captures/`.  (Images reach the model on the
keyless Claude backend; the other backends receive the saved-file path.)

**Press the Run buttons.**  Ask *"run the multiparticle simulation"* (or
envelope) and `run_in_gui` starts it **exactly as clicking Run does** —
the same Numerics-tab grid/steps/integrator/backend and recording
options, the same progress bar, results into the same tabs.  (The
`run_envelope` / `run_mp` tools still exist for headless/CLI use and for
one-off setting overrides, but in the GUI `run_in_gui` is preferred so
your on-screen settings are honoured.)

**Physicist tools.**  `parameter_scan` sweeps one element parameter over
N points (envelope per point, background job, cancellable) and returns a
value → exit-metrics table; `compare_runs` diffs two result sets (either
`.h5` files or `current`) with exit-KPI deltas and the max |Δσ| along s;
`search_manual` answers from this manual (offline, cited);
`generate_report` writes a markdown run report (KPIs, provenance and — in
the GUI — embedded figures) under `<calc-dir>/reports/`.

**Guided tour & training drills.**  Say *"give me a tour"* and the
assistant walks the interface station by station — 15 stops over the
real tabs, narrated aloud with a hands-on exercise at most stops; "next",
"back", or "jump to the Results one" all work, and the tour survives
across turns.  For training, *"start a drill"* injects a **hidden
fault** into the loaded lattice (a detuned quad, an offset cavity
phase) — hidden **from the assistant too**: the truth lives outside
its context until the debrief, so it can relay the three graded hints
without being able to spoil the answer.  Diagnose from the observables,
name the element (ambiguous names return candidates without
submitting), and get a 0–100 score (diagnosis + speed + hint economy).
The lattice is restored bit-exact at answer/give-up — or automatically
if the session closes mid-drill.

**Campaigns & verified tuning.**  `run_campaign` executes a multi-step
plan — parameter sets, envelope/multiparticle runs, anomaly checkpoints
— after **one** confirmation whose echo *is* the full numbered plan
("1. set QF.gradient = 5.5 … Any failure STOPS the chain; completed SET
steps are NOT reverted").  Every step is validated *before* you're
asked (an impossible plan never prompts), and the per-step gates are
deliberately bypassed once you approve the plan — the single ledger
record carries the complete step trail.  `tuning_plan` is the cautious
sibling: 1–8 knobs, a declared success window on an exit KPI, a
verification run — and if the window is missed **every knob is restored
bit-exact automatically**, stated in the echo up front.

**Analysis sandbox (`run_python`).**  The assistant can write and run
short Python analysis code — fits, FFTs, custom plots — in an
**isolated subprocess**: empty `PYTHONPATH` *and* `linac_gen` hard-
blocked (it cannot drive the simulator), no network, CPU and file-size
limits, 60 s cap.  It sees only what is explicitly passed: results
columns via `include=` (full-precision `data.npz`) and small JSON via
`data=`.  A saved PNG comes back as an image the assistant can *see*
and describe; captures land in `<calc-dir>/assist_captures/`.  Compute
tier — each call confirms unless session auto-approve is on.  It runs
real code on your machine: a machine-safety sandbox, not a security
jail.

**Diagnosis & health.**  `diagnose` runs an ordered differential over
the loaded results — transmission-loss onset mapped to the containing
element ("cause is upstream"), σ blow-up onset, emittance growth,
energy deviation, and any parameter changes made in the last 30 minutes
(from the session ledger) — and suggests the next tool.
`anomaly_baseline` records a known-good result as the healthy
fingerprint (exit KPIs + σ/transmission/energy profiles);
`anomaly_check` scores any later result against it with per-feature
z-scores and top offenders, and **refuses honestly** when the baseline
belongs to a different lattice.  `shift_data` gives a session briefing:
tools used, mutate/denied counts, result files written, current KPIs —
the "what did we do today" answer.

**Memory.**  A persistent lab notebook (`<calc-dir>/assist_notebook.md`)
records a mechanical summary of every session (lattice, tools run,
results written, last conclusion) plus anything you ask it to *"note
down"* (`notebook_note`).  Recent entries are loaded into the assistant's
context at start-up, so it remembers what was done and concluded in past
sessions (`read_notebook` reaches further back).

**Pointing.**  `highlight_element` selects and scrolls to an element in
the Lattice tab and moves the s-cursor; `set_cursor` points at any s
position; `get_gui_context` tells the assistant what you currently see
("what am I looking at?").

**Tab, subtab & plot control.**  The assistant can drive the app's own UI:

* *"show me the Results tab"* / *"open Matching"* → `show_tab`.
* *"show the Breakdown view under Lattice"* / *"Numerics → Plot"* →
  `show_tab` with a **subtab** (Lattice, Numerics and Error Study have
  nested subtabs; `list_tabs` reports them).
* *"open the phase-space plot"* / *"plot the RMS beam size"* → `open_plot`
  opens that window in the Results tab (switching there first);
  `list_plots` reports the ~40 available plots.

All are `read`-tier, GUI-only actions wired through **thread-safe queued
signals** into the main window and the Results tab — navigation and plot
windows are never touched from the agent's worker thread, and nothing
here goes near the simulation core.

## Voice (offline, push-to-talk)

The GUI panel has an optional **offline** voice mode — press-and-hold
the 🎤 button to speak, release to transcribe (local faster-whisper);
tick **Speak replies** to hear a concise spoken summary of each answer.
Install the audio stack with `pip install -e ".[assist-voice]"`;
speech synthesis uses macOS `say` out of the box, or install
`piper-tts` for a higher-quality offline voice.  With the stack absent
the mic/speak controls are simply disabled — the assistant stays
text-only.

**Natural voice.**  With the optional `kokoro-onnx` package and its two
model files in `~/.helix/assistant_models/` (`kokoro*.onnx` +
`voices*.bin`, one-time download), replies are spoken in a natural
neural voice — **sentence by sentence as the reply streams**, with
barge-in (pressing the mic cuts the speech).  Without kokoro the
speaker falls back to macOS `say`.  Install:
`pip install -e ".[assist-voice-kokoro]"`.  The mic level also
animates the orb's listening bars while you hold push-to-talk.

Two deliberate design choices:

* **Push-to-talk always works alongside the wake word** — hands-free
  listening (👂, on by default) is fully local and can be switched off;
  with it off the mic is live only while you hold the button (no
  continuous capture).
* **Precision stays on screen.**  The assistant *speaks* a summary with
  numbers coarsened ("σx at exit about 0.6 millimetres") while the exact
  digits and the echoed tool call remain in the text panel.  Spoken
  numbers are also rewritten for the TTS engine — "0.62" is voiced as
  "0 point 62", scientific notation as "times ten to the …", so the
  decimal point is never silently skipped.  You talk to
  it and hear the gist; you never rely on *hearing* a critical value —
  important because the beam-physics conventions (α_z sign, deg·MeV,
  mm-vs-m) are unforgiving.

Everything runs locally: no audio leaves the machine.
