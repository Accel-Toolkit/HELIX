/* HELIX landing — tiny progressive-enhancement layer (no dependencies) */
(function () {
  "use strict";

  /* --- mobile nav toggle --- */
  var toggle = document.querySelector(".nav-toggle");
  var links = document.querySelector(".nav-links");
  if (toggle && links) {
    toggle.addEventListener("click", function () {
      var open = links.classList.toggle("open");
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
    links.addEventListener("click", function (e) {
      if (e.target.tagName === "A") links.classList.remove("open");
    });
  }

  /* --- copy buttons on terminal/code blocks --- */
  document.querySelectorAll(".term .copy").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var pre = btn.closest(".term").querySelector("pre");
      if (!pre) return;
      var text = pre.innerText.replace(/^\s*[#$>]\s?/gm, "");
      navigator.clipboard.writeText(text).then(function () {
        var old = btn.textContent;
        btn.textContent = "copied ✓";
        setTimeout(function () { btn.textContent = old; }, 1400);
      }).catch(function () {});
    });
  });

  /* --- hero feature ticker (echoes the README typing SVG) --- */
  var ticker = document.querySelector("[data-ticker]");
  if (ticker) {
    var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var out = ticker.querySelector(".line");
    var lines = [
      "envelope Σ-matrix solver",
      "multi-particle 3-D PIC space charge",
      "TraceWin-compatible lattice language",
      "voice-driven AI copilot, fully offline",
      "PyQt6 workbench + scriptable batch CLI",
    ];
    if (!out) return;
    if (reduce) { out.textContent = lines[0]; return; }
    var li = 0, ci = 0, deleting = false;
    function tick() {
      var word = lines[li];
      out.textContent = word.slice(0, ci);
      if (!deleting && ci < word.length) { ci++; setTimeout(tick, 42); }
      else if (!deleting) { deleting = true; setTimeout(tick, 1500); }
      else if (ci > 0) { ci--; setTimeout(tick, 22); }
      else { deleting = false; li = (li + 1) % lines.length; setTimeout(tick, 220); }
    }
    tick();
  }
})();
