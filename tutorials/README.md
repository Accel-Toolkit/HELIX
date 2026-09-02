# HELIX tutorial videos

Scripted, re-renderable tutorial videos, narrated by the HELIX
assistant's own voice (kokoro `af_sarah`, fully local).  Everything is
code: when the GUI changes, re-run the storyboard and the video
re-renders with fresh screenshots — no screen recording, no manual
editing.

## Layout

```
tutorials/
├── pipeline/            # the machinery (shared by all episodes)
│   ├── narrate.py       # kokoro text→WAV + .srt captions
│   ├── cards.py         # title / terminal / outro cards (Qt offscreen)
│   └── render.py        # scene → segment → concat (ffmpeg, 1080p H.264)
├── storyboards/         # one script per episode: narration + captures
│   └── ep01_install_first_run.py
└── rendered/            # output MP4 + SRT (gitignored)
```

## Build an episode

```bash
PYTHONPATH=.:gui python3 tutorials/storyboards/ep01_install_first_run.py
```

Requirements: `ffmpeg` on PATH (`brew install ffmpeg`), and the
assistant voice models in `~/.helix/assistant_models` (`kokoro-v*.onnx`
+ `voices-v*.bin` — the same files the in-app voice uses).

Narration WAVs are cached per scene (keyed on the text), so editing one
scene's narration re-synthesizes only that scene; screenshots are
re-captured on every build (they are cheap and must track the GUI).

## How a storyboard works

A storyboard is a plain Python file with two parts:

* `SCENES` — ordered `Scene(name, narration, min_s=...)` entries.
  Scene duration = narration length + a breathing tail (or `min_s`).
  The narration text is also the caption text (sidecar `.srt`).
* `capture_visuals()` — renders the cards and drives the **real GUI**
  offscreen (sandboxed `HELIX_QSETTINGS_DIR`, a true 1920×1080 virtual
  screen) to produce one PNG per scene: real dialogs, real runs, real
  plots.  Physics shown on screen is computed live — e.g. episode 1
  injects the matched Courant–Snyder beam from
  `linac_gen.matching.find_matched_input_twiss` so the FODO envelope
  actually breathes the way the narration says.

## Conventions

* 1920×1080 @ 30 fps, H.264 + AAC, fade-in/out; hard cuts between
  scenes (tutorial pacing, and it keeps concat trivial).
* Narration style: short sentences; spell out acronyms the first time
  ("R M S"); numbers become words automatically (the assistant's
  `speakable_numbers`).  Say only what the frame shows.
* Keep every claim on screen honest — if the physics is subtle, teach
  the subtlety; never overstate what the plot shows.
