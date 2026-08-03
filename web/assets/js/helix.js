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

  var prefersReduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* --- hero canvas: macroparticles streaming along a beating FODO envelope --- */
  (function () {
    var canvas = document.querySelector(".hero-canvas");
    if (!canvas || prefersReduce) return;
    var ctx = canvas.getContext("2d");
    var W, H, DPR, raf, parts = [];
    var N = Math.min(170, Math.floor((window.innerWidth || 1200) / 9));

    function size() {
      DPR = Math.min(2, window.devicePixelRatio || 1);
      var r = canvas.getBoundingClientRect();
      W = canvas.width = Math.max(1, r.width * DPR);
      H = canvas.height = Math.max(1, r.height * DPR);
    }
    // envelope amplitude across the width — FODO-style beating (nodes/antinodes)
    function env(x) {
      var t = x / W;
      return 0.32 + 0.68 * Math.abs(Math.sin(t * Math.PI * 3.2));
    }
    function seed() {
      parts = [];
      for (var i = 0; i < N; i++) {
        parts.push({
          x: Math.random() * W,
          ph: Math.random() * Math.PI * 2,
          sp: 0.35 + Math.random() * 0.9,
          amp: 0.25 + Math.random() * 0.9,
          off: (Math.random() - 0.5) * 2,
          blue: Math.random() > 0.45
        });
      }
    }
    function frame() {
      ctx.clearRect(0, 0, W, H);
      var cY = H * 0.5;
      // faint envelope guide curves (±)
      ctx.lineWidth = DPR;
      for (var s = -1; s <= 1; s += 2) {
        ctx.beginPath();
        for (var x = 0; x <= W; x += 8 * DPR) {
          var y = cY + s * env(x) * H * 0.30;
          x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.strokeStyle = "rgba(42,212,238,0.10)";
        ctx.stroke();
      }
      // particles
      for (var i = 0; i < parts.length; i++) {
        var p = parts[i];
        p.x += p.sp * DPR * 1.15;
        if (p.x > W + 4) { p.x = -4; p.ph = Math.random() * Math.PI * 2; }
        var e = env(p.x);
        var wave = Math.sin(p.x / W * Math.PI * 6 + p.ph);
        var y = cY + wave * e * H * 0.28 * p.amp + p.off * e * H * 0.10;
        var a = 0.55 * e * (0.3 + 0.7 * p.amp);
        ctx.beginPath();
        ctx.arc(p.x, y, 1.5 * DPR, 0, 6.2832);
        ctx.fillStyle = (p.blue ? "rgba(90,150,255," : "rgba(42,212,238,") + a.toFixed(3) + ")";
        ctx.fill();
      }
      raf = requestAnimationFrame(frame);
    }
    function start() { if (!raf) raf = requestAnimationFrame(frame); }
    function stop() { if (raf) { cancelAnimationFrame(raf); raf = null; } }

    size(); seed(); start();
    var rz;
    window.addEventListener("resize", function () {
      clearTimeout(rz);
      rz = setTimeout(function () { size(); seed(); }, 180);
    });
    document.addEventListener("visibilitychange", function () {
      document.hidden ? stop() : start();
    });
  })();

  /* --- scroll reveal --- */
  (function () {
    if (prefersReduce || !("IntersectionObserver" in window)) return;
    var sel = ".section-head, .card, .solver, .shot, .term, .arch, .table-scroll, .callout, .iconlist, .cta-band, .endcap, .rail";
    var els = [].slice.call(document.querySelectorAll(sel)).filter(function (el) {
      return !el.closest(".hero"); // hero has its own staged load-in
    });
    if (!els.length) return;
    els.forEach(function (el) { el.classList.add("reveal"); });
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.06 });
    els.forEach(function (el) { io.observe(el); });
  })();

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
