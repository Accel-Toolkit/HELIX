"""Bonus episode — Where to Look It Up: a tour of the HELIX user manual.

Build:  PYTHONPATH=.:gui:tutorials python3 tutorials/storyboards/ep14_manual_tour.py
Output: tutorials/rendered/ep14_manual_tour.mp4 (+ .srt)

Every frame is the REAL mkdocs site rendered offscreen in a
QtWebEngineView (1920x1080) and driven with JavaScript between
Recorder ticks — genuine scrolling, genuine tab and palette clicks,
genuine offline search (typed one character at a time, results
ranked by the built-in index, a real result clicked, the term
highlighted on the landing page).

Capture rig (see docs/mkdocs.yml):
  * the manual is rebuilt into a SCRATCH directory from a scratch copy
    of mkdocs.yml whose repo_url/repo_name carry the public
    Accel-Toolkit/HELIX identity (the header link is sticky in every
    frame); the repository's own docs/ and site/ are never written;
  * QtWebEngine has no CDN access offscreen, so MathJax never loads —
    the quadrupole tour jumps past the transfer-matrix block via real
    table-of-contents clicks and the narration never mentions equations;
  * the offscreen start palette is LIGHT (prefers-color-scheme: dark is
    false), so the dark slate palette is selected once, on the first
    page, through the same header toggle the viewer would click; the
    choice persists in the (off-the-record) profile's localStorage for
    the rest of the session.
"""
from __future__ import annotations

import os
import shutil
import subprocess
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
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu --no-sandbox"
if "QT_PLUGIN_PATH" not in os.environ:
    import PyQt6
    os.environ["QT_PLUGIN_PATH"] = os.path.join(
        os.path.dirname(PyQt6.__file__), "Qt6", "plugins")

from pipeline.render import Scene, build_video          # noqa: E402

WORK = ROOT / "tutorials" / "rendered" / "ep14_work"
SHOTS = WORK / "shots"
OUT = ROOT / "tutorials" / "rendered" / "ep14_manual_tour.mp4"
MKDOCS_YML = ROOT / "docs" / "mkdocs.yml"
PUBLIC_REPO = "Accel-Toolkit/HELIX"

SCENES = [
    Scene("010_title",
          "A bonus appendix to the HELIX tutorial series. The "
          "thirteen episodes showed you how: every tab, every run, "
          "every plot. This one shows you where to look things up. "
          "HELIX ships with a written manual, a complete reference "
          "site: every element, every configuration knob, every "
          "command and every diagnostic, with worked examples and "
          "validated benchmarks. A few minutes here will save you "
          "hours later, because the manual answers the questions "
          "the videos raise.",
          min_s=6.0),
    Scene("020_landing",
          "This is the front door. The manual is published on the "
          "project's GitHub Pages, and it builds locally from the "
          "repository with M K docs, which is exactly how these "
          "frames were captured, offline. Below the hero, eight "
          "tiles route to the most-used parts: Quick start, "
          "Elements, Space charge, Matching, the G U I workbench, "
          "the A I assistant, Validation, and Error studies. Under "
          "the tiles, two paragraphs say what HELIX is and what "
          "the manual promises: the reference for every element, "
          "knob, command and diagnostic."),
    Scene("030_three_tracks",
          "Scroll down and the manual asks who you are. Three "
          "tracks, side by side in one tabbed block. TraceWin "
          "users get the quick start, the migration checklist, the "
          "keyword cheatsheet and the parity page, in that order. "
          "Linac newcomers get the introduction, coordinates and "
          "units, a first F O D O lattice, and the glossary. HELIX "
          "developers get the data model, the Python A P I, the "
          "contributing guide and the physics references. One "
          "manual, three doors."),
    Scene("040_navigation",
          "Three navigation systems, always on screen. The top row "
          "of tabs is the parts of the book. Read it left to right "
          "once: Home, Tutorials, Getting Started, Concepts, "
          "Elements, Beam, Space charge, Running, Matching, Errors "
          "and tolerances, Diagnostics, G U I, Worked examples, "
          "Validation, M L Surrogates, A I Assistant, Multibunch, "
          "and Appendices. Click Elements and the left column "
          "becomes the chapters of that part, one per element. The "
          "right column is the table of contents of the open page, "
          "and its highlight follows you as you scroll. Scroll back "
          "up, even slightly, and a Back to top pill appears; one "
          "click returns you to the head of the page."),
    Scene("050_search",
          "Search is the fastest route to anything: element names, "
          "keywords, error directives, algorithm names. Type "
          "hofmann and the results rank as you type: six matching "
          "documents, the Diagnostics chapter on the Hofmann "
          "stability chart first, then the Results tab page that "
          "hosts it. The index is compiled into the site itself, "
          "so this works offline, straight off your disk. The "
          "share icon beside the query gives a link that reopens "
          "this exact search. Pick the top result, and every "
          "occurrence of the term is highlighted on the page you "
          "land on."),
    Scene("060_element_pages",
          "Every element in the lattice language has its own "
          "chapter, and the Elements overview is the catalogue: "
          "pick by use case, from field-free propagation to "
          "overlapping field maps, and each row links to its "
          "chapter. Open Quadrupole and you meet the anatomy every "
          "element chapter shares. First, a T L D R card for "
          "people who already know the physics: the TraceWin "
          "keyword beside the HELIX class, units side by side, and "
          "the sign conventions. Then a tutorial for newcomers, "
          "with a complete runnable example, here a single F O D O "
          "cell, and a copy button in the corner of the code "
          "block. Then the A P I reference for developers: the "
          "full constructor signature with every default, a "
          "parameter table, the properties, and the source file "
          "with its line number. And a See also list to the "
          "neighbouring chapters."),
    Scene("070_conventions",
          "The manual's callouts are a small language of their "
          "own, and the front page defines it. Boxes titled T L D "
          "R cards open every element and major-feature chapter. "
          "Tutorial boxes mark the narrative walkthroughs, and "
          "promise that every code block is runnable against the "
          "current install. A P I reference boxes hold the Python "
          "signatures and parameter tables. And the Caveats box is "
          "the important one: it marks known issues, deferred "
          "features, and convention differences against TraceWin, "
          "IMPACT-X and other codes.",
          min_s=8.0),
    Scene("075_caveat",
          "Here is a caveat in the wild, from the migration "
          "appendix. The longitudinal alpha flips sign relative to "
          "TraceWin, because HELIX's phase coordinate runs the "
          "opposite way. The box says exactly what to do, negate "
          "alpha z while beta z and the emittance carry over, gives "
          "the worked example, and describes the symptom you would "
          "see if you got it wrong. When the manual warns you, it "
          "names the exact difference.",
          min_s=8.0),
    Scene("080_recipes",
          "Beyond reference pages, the manual has cookbooks. The "
          "matching recipes are ordered from the smallest possible "
          "problem upward. Recipe one matches a single "
          "quadrupole. Two: emittance minimisation with C M A E S "
          "and a least-squares polish. Three: sequential scan with "
          "a seed-exit threshold. Four: the multi-particle cost "
          "solver for physics-accurate matching. Five: "
          "cancellation, where clicking Stop keeps the best "
          "solution found so far. Six: parallel C M A E S on a "
          "heavy lattice. Seven: guarding against beam loss with "
          "the min transmission card. Each recipe is written "
          "three times, as Python, as a command line, and as the "
          "G U I workflow the Matching tab episode showed. And "
          "recipe eight is a decision table: which optimiser fits "
          "which problem shape."),
    Scene("090_validation",
          "Before trusting any code, read its validation chapter. "
          "The TraceWin parity page opens with a coverage matrix, "
          "element by element: parser, tracker, and how closely "
          "each matched, from bit-exact linear matrices down to "
          "the R F Q cell rows at the bottom. Then the error "
          "directives, with what each one does and does not do. "
          "Then the validation residuals at five milliamps with "
          "space charge, section by section, and the end of "
          "machine values printed as numbers rather than "
          "adjectives: sigma x 3.69 millimetres in HELIX against "
          "3.55 in TraceWin, sigma y 4.71 against 4.94. And "
          "finally, what's known to differ: the sigma phi "
          "reporting convention, a factor of about four because "
          "HELIX reports at the local cavity frequency rather than "
          "the bunch frequency, and a P I C kernel calibration "
          "offset of about three percent. Each one explained, not "
          "hidden."),
    Scene("095_validation_chapters",
          "Parity is one of four validation chapters. PIP two "
          "validation is the full linac against TraceWin partran, "
          "the reference benchmark. Known limitations is the list "
          "of what HELIX deliberately does not do, what is parsed "
          "but not honoured, and the calibration biases. And the "
          "convergence checklist is a printable one-pager to run "
          "through before you report any result as converged."),
    Scene("100_appendices",
          "The appendices are the working toolbox. Appendix B is "
          "the page TraceWin users keep open: every keyword the "
          "parser recognises, grouped as header and control cards, "
          "element cards, diagnostics, error directives, matching "
          "directives, the recognised no-op markers, the HELIX "
          "extensions hidden in comments, and, just as important, "
          "the section headed Deferred, not honoured: the cards "
          "the parser reads but the physics ignores. Appendix E is "
          "the porting checklist: what just works, conventions and "
          "units, what is different, and what you must set outside "
          "the dot dat file. Appendix C is the glossary, every "
          "domain term defined, A to Z. And Appendix D is "
          "troubleshooting: install failures, sigmas that blow up "
          "at the L E B T entrance, the three percent and "
          "factor-of-four TraceWin disagreements, hangs, blank "
          "windows. Paste your error message into search, or scan "
          "this page."),
    Scene("110_assistant_chapter",
          "The assistant episode showed it live; this chapter is "
          "its written contract. Backends: cloud by A P I key, the "
          "keyless subscription path through the Claude agent S D "
          "K, or fully local. What it can do, and what it will "
          "refuse. The audit trail and replay. Using your Claude "
          "subscription through the M C P server. The G U I "
          "section, the longest on the page, documents the panel "
          "control by control. And the voice section: offline, "
          "push to talk, with the natural kokoro voice as an "
          "option. The voice reading this sentence is that same "
          "one."),
    Scene("120_tutorials_loop",
          "The loop closes on the manual's own Tutorials page. All "
          "thirteen episodes as cards, each with its poster frame, "
          "a one-paragraph summary, the running time, and a link "
          "to its caption file. The videos stream from the "
          "project's tutorials release on GitHub, where they and "
          "the captions can also be downloaded. Watch an episode "
          "for the tour, then read the matching chapter for the "
          "numbers."),
    Scene("130_dark_light",
          "Two palettes. The dark slate you have been looking at "
          "matches the workbench. One click on the header icon "
          "switches to light, and one more switches back. The "
          "choice is remembered by your browser, and it works "
          "offline like everything else here."),
    Scene("140_local_and_github",
          "Manual, videos and repository are one project, and each "
          "links to the other two. Every page carries the "
          "repository link in its header. At the foot of the front "
          "page, how to build the manual yourself: pip install the "
          "docs extras, then M K docs serve, and it appears on a "
          "local port. One more script regenerates every figure, "
          "and another re-runs every code snippet in the manual. "
          "If you change HELIX, this is how the manual keeps up."),
    Scene("150_outro",
          "That is the tour. Search first; the index knows every "
          "keyword. Trust the validation pages, and keep Appendix "
          "B open beside your dot dat files. The video series is "
          "complete, but the manual is where the project keeps "
          "talking: new features land there, and on the tutorials "
          "page, as they ship. Thank you for watching.",
          min_s=6.0),
]


def scene_by_name(name: str) -> Scene:
    return next(sc for sc in SCENES if sc.name == name)


# --------------------------------------------------------------------
# scratch site build (repo untouched; public header identity)
# --------------------------------------------------------------------
def build_site() -> Path:
    scratch = Path(tempfile.mkdtemp(prefix="helix_ep14_site_"))
    cfg_lines = []
    for ln in MKDOCS_YML.read_text(encoding="utf-8").splitlines():
        if ln.startswith("repo_url:"):
            ln = f"repo_url: https://github.com/{PUBLIC_REPO}"
        elif ln.startswith("repo_name:"):
            ln = f"repo_name: {PUBLIC_REPO}"
        elif ln.startswith("docs_dir:"):
            ln = f"docs_dir: {ROOT / 'docs' / 'manual'}"
        elif ln.startswith("site_dir:"):
            ln = f"site_dir: {scratch / 'site'}"
        cfg_lines.append(ln)
    cfg = scratch / "mkdocs.yml"
    cfg.write_text("\n".join(cfg_lines) + "\n", encoding="utf-8")
    # cwd = docs/ so pymdownx.snippets' relative base_path (../examples)
    # resolves exactly as in the documented `cd docs && mkdocs build`
    r = subprocess.run([sys.executable, "-m", "mkdocs", "build", "-f",
                        str(cfg)], cwd=str(ROOT / "docs"),
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"mkdocs build failed:\n{r.stderr[-2000:]}")
    site = scratch / "site"
    index = (site / "index.html").read_text(encoding="utf-8")
    assert PUBLIC_REPO in index, "public repo identity missing"
    assert "Abhishek-Pathak-90" not in index, "private identity leaked"
    print(f"[site] built {site}")
    return site


# --------------------------------------------------------------------
# offscreen browser
# --------------------------------------------------------------------
class Manual:
    """QWebEngineView driver: load / js / scroll / grab (offscreen)."""

    def __init__(self, site: Path):
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        self.site = site
        self.view = QWebEngineView()
        self.view.resize(1920, 1080)
        self.view.show()
        self._loaded = False
        self.view.loadFinished.connect(self._on_load)

    def _on_load(self, ok: bool) -> None:
        self._loaded = True

    # -- event loop ----------------------------------------------------
    @staticmethod
    def settle(n: int = 4) -> None:
        from PyQt6.QtCore import QEventLoop
        from PyQt6.QtWidgets import QApplication
        for _ in range(n):
            QApplication.processEvents(
                QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    def pump(self, seconds: float) -> None:
        t0 = _time.monotonic()
        while _time.monotonic() - t0 < seconds:
            self.settle(1)
            _time.sleep(0.005)

    # -- navigation ----------------------------------------------------
    def begin_load(self, rel: str) -> None:
        from PyQt6.QtCore import QUrl
        self._loaded = False
        self.view.load(QUrl.fromLocalFile(str(self.site / rel)))

    def load(self, rel: str, settle_s: float = 2.5) -> None:
        self.begin_load(rel)
        t0 = _time.monotonic()
        while not self._loaded and _time.monotonic() - t0 < 30:
            self.pump(0.05)
        assert self._loaded, f"page did not load: {rel}"
        self.pump(settle_s)                 # blank without this settle

    def js(self, code: str, timeout: float = 5.0):
        from PyQt6.QtCore import QEventLoop, QTimer
        box: dict = {}
        loop = QEventLoop()

        def _cb(res):
            box["r"] = res
            loop.quit()
        self.view.page().runJavaScript(code, _cb)
        QTimer.singleShot(int(timeout * 1000), loop.quit)
        loop.exec()
        return box.get("r")

    def scroll_to(self, y: float) -> None:
        self.js(f"window.scrollTo({{top:{int(y)},behavior:'instant'}})")

    def scroll_y(self) -> float:
        return float(self.js("window.scrollY") or 0.0)

    def max_scroll(self) -> float:
        h = float(self.js("document.documentElement.scrollHeight") or 0)
        return max(0.0, h - 1080.0)

    def y_of(self, selector: str) -> float:
        """Absolute page y of the first element matching ``selector``."""
        v = self.js("(function(){var e=document.querySelector(%r);"
                    "return e?e.getBoundingClientRect().top+window.scrollY"
                    ":null})()" % selector)
        assert v is not None, f"selector not found: {selector}"
        return float(v)

    def click(self, selector: str) -> None:
        ok = self.js("(function(){var e=document.querySelector(%r);"
                     "if(!e)return false;e.click();return true})()"
                     % selector)
        assert ok, f"click target not found: {selector}"

    def grab(self):
        return self.view.grab().toImage()


# --------------------------------------------------------------------
# recorded motions
# --------------------------------------------------------------------
def glide(rec, man: Manual, y1: float, px: float = 30.0,
          hold: float = 0.0) -> None:
    """Scroll from the current position to ``y1`` at ``px`` per frame."""
    y1 = max(0.0, min(float(y1), man.max_scroll()))
    y = man.scroll_y()
    step = px if y1 >= y else -px
    while abs(y1 - y) > abs(step):
        y += step
        man.scroll_to(y)
        rec.tick(man.grab)
    man.scroll_to(y1)
    rec.tick(man.grab)
    if hold:
        rec.hold(man.grab, hold)


def rec_click_and_load(rec, man: Manual, selector: str,
                       settle: float = 2.0) -> None:
    """Click a real link on camera and record through the page load."""
    man._loaded = False
    man.click(selector)
    ok = rec.wait_until(man.grab, lambda: man._loaded, cap_s=30)
    assert ok, f"load after click did not finish: {selector}"
    rec.hold(man.grab, settle)


def rec_load(rec, man: Manual, rel: str, settle: float = 2.0) -> None:
    man.begin_load(rel)
    ok = rec.wait_until(man.grab, lambda: man._loaded, cap_s=30)
    assert ok, f"load did not finish: {rel}"
    rec.hold(man.grab, settle)


def capture_visuals() -> None:
    from PyQt6 import QtWebEngineWidgets      # noqa: F401  (before QApp)
    from PyQt6.QtWidgets import QApplication

    from pipeline import cards
    from pipeline.record import Recorder

    SHOTS.mkdir(parents=True, exist_ok=True)
    s = {sc.name: str(SHOTS / f"{sc.name}.png") for sc in SCENES}

    app = QApplication.instance() or QApplication(["helix-manual-tour"])
    cards.title_card(s["010_title"], "Where to Look It Up",
                     "A tour of the HELIX user manual",
                     kicker="HELIX TUTORIALS · BONUS")
    cards.title_card(s["150_outro"], "Where to Look It Up",
                     "accel-toolkit.github.io/HELIX · search first · "
                     "keep Appendix B open",
                     kicker="HELIX TUTORIALS · BONUS · THANK YOU FOR "
                            "WATCHING")

    site = build_site()
    man = Manual(site)
    rec = Recorder(WORK / "frames", man.settle)
    grab = man.grab

    # ---- dark palette once, through the real header toggle ---------
    man.load("index.html", 3.0)
    man.click('label[for="__palette_0"]')
    man.pump(1.5)
    scheme = man.js("document.documentElement.getAttribute("
                    "'data-md-color-scheme')")
    assert scheme == "slate", f"palette not dark: {scheme}"
    repo = man.js("(function(){var e=document.querySelector("
                  "'.md-source__repository');return e?e.textContent"
                  ".trim():''})()")
    assert repo == PUBLIC_REPO, f"header repo text {repo!r}"

    # ---- 020 landing: hero + eight tiles + intro ---------------------
    man.scroll_to(0)
    man.pump(0.5)
    rec.start("020_landing")
    rec.hold(grab, 6.0)                     # hero + first tiles
    glide(rec, man, 1080, px=8, hold=3.0)   # tiles roll past as named
    rec.finish(scene_by_name("020_landing"))

    # ---- 030 three tracks: real tab clicks ---------------------------
    y_tabs = man.y_of(".tabbed-set")
    man.scroll_to(y_tabs - 260)             # "Choose your track" heading
    man.pump(0.5)
    rec.start("030_three_tracks")
    rec.hold(grab, 14.0)                    # TraceWin user track (default)
    man.click('label[for="__tabbed_1_2"]')
    rec.hold(grab, 7.0)                     # Linac newcomer
    man.click('label[for="__tabbed_1_3"]')
    rec.hold(grab, 8.0)                     # HELIX developer
    man.click('label[for="__tabbed_1_1"]')
    rec.hold(grab, 2.0)
    rec.finish(scene_by_name("030_three_tracks"))

    # ---- 040 navigation: top tab click, TOC follow, back-to-top -----
    man.scroll_to(0)
    man.pump(0.5)
    rec.start("040_navigation")
    rec.hold(grab, 22.0)                    # the tab row is read aloud
    rec_click_and_load(
        rec, man, 'a.md-tabs__link[href$="03_elements/00_overview.html"]',
        settle=3.0)
    glide(rec, man, 2400, px=40, hold=1.5)  # TOC highlight follows
    glide(rec, man, 1900, px=40, hold=2.5)  # back-to-top pill appears
    man.click(".md-top")                    # real pill click
    rec.hold(grab, 3.0)
    rec.finish(scene_by_name("040_navigation"))

    # ---- 050 search: typed live, result clicked, term highlighted ---
    man.load("index.html", 2.5)
    man.scroll_to(0)
    man.pump(0.3)
    rec.start("050_search")
    rec.hold(grab, 6.0)
    man.js("var t=document.getElementById('__search');t.checked=true;"
           "t.dispatchEvent(new Event('change',{bubbles:true}));")
    rec.hold(grab, 0.5)
    man.js("document.querySelector('.md-search__input').focus();")
    rec.hold(grab, 0.8)
    query = "hofmann"
    for k in range(1, len(query) + 1):
        man.js("var i=document.querySelector('.md-search__input');"
               f"i.value={query[:k]!r};"
               "i.dispatchEvent(new Event('input',{bubbles:true}));"
               "i.dispatchEvent(new KeyboardEvent('keyup',"
               f"{{bubbles:true,key:{query[k-1]!r}}}));")
        rec.tick(grab, 3)
    rec.hold(grab, 18.0)                    # results + share icon on screen
    n_hits = man.js("(function(){var e=document.querySelector("
                    "'.md-search-result__meta');return e?e.textContent"
                    ".trim():''})()")
    print(f"[search] {n_hits}")
    rec_click_and_load(rec, man, ".md-search-result__link", settle=3.0)
    glide(rec, man, 500, px=15, hold=2.0)
    rec.finish(scene_by_name("050_search"))

    # ---- 060 element pages: catalogue, then the quadrupole anatomy --
    man.load("03_elements/00_overview.html", 2.5)
    man.scroll_to(0)
    man.pump(0.3)
    rec.start("060_element_pages")
    rec.hold(grab, 2.0)
    glide(rec, man, 700, px=20, hold=4.5)   # catalogue table
    rec_load(rec, man, "03_elements/02_quadrupole.html", settle=1.0)
    glide(rec, man, 200, px=20, hold=12.0)  # TL;DR card (keyword table)
    man.click('.md-sidebar--secondary a[href="#tutorial-newcomers"]')
    rec.hold(grab, 1.5)                     # (MathJax never loads offscreen
                                            #  — pass the matrix quickly)
    man.click('.md-sidebar--secondary '
              'a[href="#example-a-single-fodo-cell"]')
    rec.hold(grab, 11.0)                    # FODO example + copy button
    man.click('.md-sidebar--secondary a[href="#api-reference-developers"]')
    rec.hold(grab, 9.0)                     # signature block
    glide(rec, man, man.y_of("#properties") - 520, px=30, hold=2.0)
    glide(rec, man, man.max_scroll(), px=30, hold=3.0)   # source, see also
    rec.finish(scene_by_name("060_element_pages"))

    # ---- 070 conventions (still) + 075 a real caveat (still) --------
    man.load("index.html", 2.5)
    man.scroll_to(man.y_of("#conventions-used-in-this-manual") - 130)
    man.pump(1.0)
    grab().save(s["070_conventions"])

    man.load("appendices/E_migrating_from_tw.html", 2.5)
    man.scroll_to(man.y_of(".md-content .admonition.warning") - 150)
    man.pump(1.0)
    grab().save(s["075_caveat"])

    # ---- 080 recipes cookbook ---------------------------------------
    man.load("07_matching/05_recipes.html", 2.5)
    man.scroll_to(0)
    man.pump(0.3)
    rec.start("080_recipes")
    rec.hold(grab, 8.0)
    for n in range(2, 9):
        h = man.js("(function(){var hs=Array.from(document.querySelectorAll"
                   f"('.md-content h2'));var h=hs.find(x=>x.textContent"
                   f".trim().startsWith('Recipe {n} '));return h?"
                   "h.getBoundingClientRect().top+window.scrollY:null})()")
        assert h is not None, f"Recipe {n} heading not found"
        glide(rec, man, float(h) - 120, px=90, hold=3.0)
    rec.hold(grab, 2.0)
    rec.finish(scene_by_name("080_recipes"))

    # ---- 090 validation: parity page ---------------------------------
    man.load("12_validation/01_tracewin_parity.html", 2.5)
    man.scroll_to(0)
    man.pump(0.3)
    rec.start("090_validation")
    rec.hold(grab, 14.0)                    # coverage matrix
    glide(rec, man, man.y_of("#error-directives") - 120, px=30, hold=5.0)
    glide(rec, man, man.y_of("#validation-residuals-5-ma-sc") - 120,
          px=30, hold=6.0)
    glide(rec, man, man.max_scroll(), px=30, hold=3.0)
    rec.finish(scene_by_name("090_validation"))

    # ---- 095 the other three validation chapters ---------------------
    man.load("12_validation/02_pipii_validation.html", 2.5)
    man.scroll_to(0)
    man.pump(0.3)
    rec.start("095_validation_chapters")
    rec.hold(grab, 9.0)
    rec_load(rec, man, "12_validation/03_known_limitations.html",
             settle=8.0)
    rec_load(rec, man, "12_validation/04_convergence_guide.html",
             settle=4.0)
    rec.finish(scene_by_name("095_validation_chapters"))

    # ---- 100 appendices: B, E, C, D -----------------------------------
    man.load("appendices/B_keyword_cheatsheet.html", 2.5)
    man.scroll_to(0)
    man.pump(0.3)
    rec.start("100_appendices")
    rec.hold(grab, 4.0)
    glide(rec, man, man.y_of("#element-cards") - 120, px=40, hold=2.0)
    glide(rec, man, man.y_of("#matching-directives") - 120, px=60,
          hold=1.5)
    glide(rec, man, man.y_of("#recognised-no-op-markers") - 120, px=60,
          hold=1.5)
    glide(rec, man, man.max_scroll(), px=60, hold=8.0)   # deferred
    rec_load(rec, man, "appendices/E_migrating_from_tw.html", settle=3.0)
    glide(rec, man, man.y_of("#whats-different") - 120, px=60, hold=2.5)
    rec_load(rec, man, "appendices/C_glossary.html", settle=2.5)
    glide(rec, man, 500, px=25, hold=1.0)
    rec_load(rec, man, "appendices/D_troubleshooting.html", settle=2.5)
    glide(rec, man, man.y_of("#tracking") - 120, px=80, hold=4.0)
    rec.finish(scene_by_name("100_appendices"))

    # ---- 110 assistant chapter ----------------------------------------
    man.load("14_assistant/01_assistant.html", 2.5)
    man.scroll_to(0)
    man.pump(0.3)
    rec.start("110_assistant_chapter")
    rec.hold(grab, 3.5)
    for anchor, hold in (("#backends-cloud-keyless-subscription-or-fully-"
                          "local", 6.0),
                         ("#what-it-can-do-and-what-it-will-refuse", 3.0),
                         ("#audit-trail-and-replay", 2.5),
                         ("#using-your-claude-subscription-mcp-server",
                          2.5),
                         ("#gui", 3.0)):
        glide(rec, man, man.y_of(anchor) - 120, px=60, hold=hold)
    glide(rec, man, man.max_scroll(), px=120, hold=4.0)   # voice section
    rec.finish(scene_by_name("110_assistant_chapter"))

    # ---- 120 tutorials page ------------------------------------------
    man.load("tutorials.html", 3.0)
    man.scroll_to(0)
    man.pump(0.3)
    rec.start("120_tutorials_loop")
    rec.hold(grab, 3.0)
    glide(rec, man, man.max_scroll(), px=20, hold=3.0)
    rec.finish(scene_by_name("120_tutorials_loop"))

    # ---- 130 palette toggle, live -------------------------------------
    man.load("index.html", 2.5)
    man.scroll_to(0)
    man.pump(0.3)
    rec.start("130_dark_light")
    rec.hold(grab, 4.5)
    man.click('label[for="__palette_1"]')   # -> light
    rec.hold(grab, 4.5)
    assert man.js("document.documentElement.getAttribute("
                  "'data-md-color-scheme')") == "default"
    man.click('label[for="__palette_0"]')   # -> dark again
    rec.hold(grab, 3.0)
    assert man.js("document.documentElement.getAttribute("
                  "'data-md-color-scheme')") == "slate"
    rec.finish(scene_by_name("130_dark_light"))

    # ---- 140 build locally + repository link ---------------------------
    man.scroll_to(man.y_of("#building-this-manual-locally") - 700)
    man.pump(0.5)
    rec.start("140_local_and_github")
    rec.hold(grab, 3.0)
    glide(rec, man, man.max_scroll(), px=20, hold=4.0)
    rec.finish(scene_by_name("140_local_and_github"))

    for sc in SCENES:
        if not sc.frames_dir:
            sc.image = s[sc.name]
    shutil.rmtree(site.parent, ignore_errors=True)


def main() -> None:
    capture_visuals()
    info = build_video(SCENES, WORK, OUT)
    print(f"rendered {info['mp4']}  ({info['duration']:.1f} s)")
    print(f"captions {info['srt']}")
    sys.stdout.flush()
    os._exit(0)          # never hang on QtWebEngine teardown


if __name__ == "__main__":
    main()
