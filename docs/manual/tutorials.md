# Video Tutorials

Thirteen narrated episodes plus a bonus tour of this manual, **about three
hours in total**, from installing HELIX to driving the workbench with its
voice assistant. Every episode is a maximum-detail deep dive: every panel,
control, option and algorithm of its subject is shown on screen or
demonstrated live. Every frame is captured from the real interface, every
simulation shown was actually run, and the assistant conversation in
episode 13 was recorded live.

Select any episode to play it on this page. Videos stream from the
[`tutorials-v1` release](https://github.com/Accel-Toolkit/HELIX/releases/tag/tutorials-v1)
of this repository, where you can also download them (plus `.srt`
captions) for offline viewing.

<style>
.tut-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 1.2rem; margin-top: 1.2rem; }
.tut-card { border: 1px solid var(--md-default-fg-color--lightest, #333); border-radius: 10px; overflow: hidden; background: var(--md-code-bg-color, rgba(0,0,0,.04)); }
.tut-card video { display: block; width: 100%; aspect-ratio: 16 / 9; background: #0b1220; }
.tut-card .tut-body { padding: .65rem .85rem .8rem; }
.tut-card h3 { margin: 0 0 .25rem; font-size: .95rem; }
.tut-card h3 .tut-ep { opacity: .55; font-weight: 400; margin-right: .4rem; }
.tut-card p { margin: 0; font-size: .8rem; opacity: .8; line-height: 1.45; }
.tut-card .tut-meta { margin-top: .45rem; font-size: .72rem; opacity: .6; }
.tut-card .tut-meta a { margin-left: .6rem; }
</style>

<div class="tut-grid" markdown="0">

<div class="tut-card"><video controls preload="none" poster="tutorials_media/ep01_install_first_run.jpg" src="videos/ep01_install_first_run.mp4"></video><div class="tut-body"><h3><span class="tut-ep">01</span>Install &amp; First Run</h3><p>Install HELIX, create a project with the wizard, and send a matched proton beam through a FODO cell — the full install and first-beam walkthrough, every menu and shortcut included.</p><div class="tut-meta">12:58 · <a href="videos/ep01_install_first_run.srt">captions</a></div></div></div>

<div class="tut-card"><video controls preload="none" poster="tutorials_media/ep02_gui_tour.jpg" src="videos/ep02_gui_tour.mp4"></video><div class="tut-body"><h3><span class="tut-ep">02</span>The Full GUI Tour</h3><p>Every tab of the workbench in one pass, while a drift-tube linac accelerates protons from 3 to 7&nbsp;MeV.</p><div class="tut-meta">12:38 · <a href="videos/ep02_gui_tour.srt">captions</a></div></div></div>

<div class="tut-card"><video controls preload="none" poster="tutorials_media/ep03_lattice_tab.jpg" src="videos/ep03_lattice_tab.mp4"></video><div class="tut-body"><h3><span class="tut-ep">03</span>The Lattice Tab</h3><p>Palette, outline, timeline, sequence and breakdown views, the inspector, editing with full undo, and the orbit tools — the machine editor in depth.</p><div class="tut-meta">12:35 · <a href="videos/ep03_lattice_tab.srt">captions</a></div></div></div>

<div class="tut-card"><video controls preload="none" poster="tutorials_media/ep04_beam_tab.jpg" src="videos/ep04_beam_tab.mp4"></video><div class="tut-body"><h3><span class="tut-ep">04</span>The Beam Tab</h3><p>Species, Twiss parameters, the six distributions, the live phase-space preview, .dst particle files, and continuous-beam mode for LEBT lines.</p><div class="tut-meta">11:38 · <a href="videos/ep04_beam_tab.srt">captions</a></div></div></div>

<div class="tut-card"><video controls preload="none" poster="tutorials_media/ep05_numerics_tab.jpg" src="videos/ep05_numerics_tab.mp4"></video><div class="tut-body"><h3><span class="tut-ep">05</span>The Numerics Tab</h3><p>Step density, the space-charge models and their grids, field-map integrators — and the built-in convergence scanner, run live on camera.</p><div class="tut-meta">13:05 · <a href="videos/ep05_numerics_tab.srt">captions</a></div></div></div>

<div class="tut-card"><video controls preload="none" poster="tutorials_media/ep06_matching_tab.jpg" src="videos/ep06_matching_tab.mp4"></video><div class="tut-body"><h3><span class="tut-ep">06</span>The Matching Tab</h3><p>All seven optimisation algorithms and when to use each, the periodic matcher, and a real least-squares match converging on camera.</p><div class="tut-meta">14:20 · <a href="videos/ep06_matching_tab.srt">captions</a></div></div></div>

<div class="tut-card"><video controls preload="none" poster="tutorials_media/ep07_results_tab.jpg" src="videos/ep07_results_tab.mp4"></video><div class="tut-body"><h3><span class="tut-ep">07</span>The Results Tab</h3><p>The tile wall and every major plot, told through two machines: a healthy FODO line and a DTL deliberately overloaded until 80% of the beam is lost.</p><div class="tut-meta">13:24 · <a href="videos/ep07_results_tab.srt">captions</a></div></div></div>

<div class="tut-card"><video controls preload="none" poster="tutorials_media/ep08_param_study.jpg" src="videos/ep08_param_study.mp4"></video><div class="tut-body"><h3><span class="tut-ep">08</span>The Param Study Manager</h3><p>Scan any parameter over a range, keep every run organised and resumable on disk, and read the response curves in the analysis views.</p><div class="tut-meta">12:32 · <a href="videos/ep08_param_study.srt">captions</a></div></div></div>

<div class="tut-card"><video controls preload="none" poster="tutorials_media/ep09_error_study.jpg" src="videos/ep09_error_study.mp4"></video><div class="tut-body"><h3><span class="tut-ep">09</span>The Error Study Tab</h3><p>Tolerance studies: statistical misalignments and field errors, seed ensembles, operator-style orbit correction, and the ensemble band plots.</p><div class="tut-meta">12:09 · <a href="videos/ep09_error_study.srt">captions</a></div></div></div>

<div class="tut-card"><video controls preload="none" poster="tutorials_media/ep10_failure_study.jpg" src="videos/ep10_failure_study.mp4"></video><div class="tut-body"><h3><span class="tut-ep">10</span>The Failure Study Tab</h3><p>Disable elements deliberately — single failures, pairs, custom sets — rank the resulting damage, and let the matcher attempt recovery.</p><div class="tut-meta">12:38 · <a href="videos/ep10_failure_study.srt">captions</a></div></div></div>

<div class="tut-card"><video controls preload="none" poster="tutorials_media/ep11_surrogates.jpg" src="videos/ep11_surrogates.mp4"></video><div class="tut-body"><h3><span class="tut-ep">11</span>The Surrogates Tab</h3><p>Neural stand-ins for expensive field maps: train them, score them on held-out data, verify against full physics, and swap them in.</p><div class="tut-meta">16:33 · <a href="videos/ep11_surrogates.srt">captions</a></div></div></div>

<div class="tut-card"><video controls preload="none" poster="tutorials_media/ep12_space_charge.jpg" src="videos/ep12_space_charge.mp4"></video><div class="tut-body"><h3><span class="tut-ep">12</span>Space Charge &amp; Real Beams</h3><p>The physics behind intense linacs: envelopes swelling with current, tune depression, the Hofmann stability chart, and a PIC-versus-envelope cross-check.</p><div class="tut-meta">14:30 · <a href="videos/ep12_space_charge.srt">captions</a></div></div></div>

<div class="tut-card"><video controls preload="none" poster="tutorials_media/ep13_assistant.jpg" src="videos/ep13_assistant.mp4"></video><div class="tut-body"><h3><span class="tut-ep">13</span>The Voice Assistant</h3><p>Recorded live: a model conversation grounded in application state, instant commands, the mutate-confirmation gate, and the session ledger.</p><div class="tut-meta">16:41 · <a href="videos/ep13_assistant.srt">captions</a></div></div></div>

<div class="tut-card"><video controls preload="none" poster="tutorials_media/ep14_manual_tour.jpg" src="videos/ep14_manual_tour.mp4"></video><div class="tut-body"><h3><span class="tut-ep">Bonus</span>A Tour of the Manual</h3><p>Where to look things up: the site structure, offline search, the per-element reference pages, validation benchmarks, matching recipes, the appendices, the assistant chapter, and how manual, tutorials and GitHub link together.</p><div class="tut-meta">9:51 · <a href="videos/ep14_manual_tour.srt">captions</a></div></div></div>

</div>

---

*The narration voice is the assistant's own (kokoro `af_sarah`, fully
local). Episodes are produced by scripted storyboards that drive the
real GUI offscreen and are re-rendered when the software changes, so
what you see is the interface as it actually behaves.*
