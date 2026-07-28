/* HELIX manual — self-contained palette toggle (light/dark mode).
 *
 * This script does NOT depend on any Material-for-MkDocs internal JavaScript.
 * It works equally on http://, file://, with or without localStorage,
 * and survives Material module-loading failures.
 *
 * Mechanism:
 *   1. On load, find both palette radios + their labels.
 *   2. Apply the saved (or first) palette to <html> and <body>.
 *   3. Make ALL labels visible (Material renders them with `hidden`,
 *      relying on its own JS to un-hide; we don't need that).
 *   4. On label click, swap palettes and re-apply.
 */
(function () {
  "use strict";

  var STORAGE_KEY = "__helix_palette_idx";

  function applyPalette(input) {
    if (!input) return;
    var scheme  = input.getAttribute("data-md-color-scheme")  || "default";
    var primary = input.getAttribute("data-md-color-primary") || "indigo";
    var accent  = input.getAttribute("data-md-color-accent")  || "indigo";
    [document.body, document.documentElement].forEach(function (el) {
      if (!el) return;
      el.setAttribute("data-md-color-scheme",  scheme);
      el.setAttribute("data-md-color-primary", primary);
      el.setAttribute("data-md-color-accent",  accent);
    });
  }

  function showOnlyActiveLabel(inputs) {
    /* Material convention: each label's `for` attribute points to the OTHER
     * radio (so clicking the label switches to that other palette).  We
     * therefore want to show the label whose `for` is the *non-active*
     * palette's id — i.e. show the label that "lives next to" the active
     * radio.  In a 2-palette setup with the labels output in the same
     * order as the radios, that's: show label[i] iff inputs[i] is checked. */
    inputs.forEach(function (input) {
      var label = document.querySelector('label[for="' + input.id + '"]');
      if (!label) return;
      /* The label with for=input.id is the OTHER palette's label (it
       * switches TO this input).  We hide the one matching the
       * currently-active input (because clicking it would switch *away*
       * from the current state — which is what we want, the visible
       * button is "switch to the other state"). */
      if (input.checked) {
        label.setAttribute("hidden", "");
      } else {
        label.removeAttribute("hidden");
      }
    });
  }

  function pickInitial(inputs) {
    var saved = null;
    try { saved = localStorage.getItem(STORAGE_KEY); } catch (e) {}
    if (saved !== null) {
      var idx = parseInt(saved, 10);
      if (!isNaN(idx) && idx >= 0 && idx < inputs.length) return inputs[idx];
    }
    /* No saved choice — honour the OS preference if any palette has a
     * matching media query.  Otherwise pick the first. */
    if (window.matchMedia) {
      for (var i = 0; i < inputs.length; i++) {
        var media = inputs[i].getAttribute("data-md-color-media");
        if (media && window.matchMedia(media).matches) return inputs[i];
      }
    }
    return inputs[0];
  }

  function bind() {
    var inputs = Array.prototype.slice.call(
      document.querySelectorAll('input[name="__palette"]')
    );
    if (!inputs.length) return;

    function activate(input) {
      if (!input) return;
      inputs.forEach(function (i) { i.checked = (i === input); });
      applyPalette(input);
      showOnlyActiveLabel(inputs);
      try {
        var idx = inputs.indexOf(input);
        localStorage.setItem(STORAGE_KEY, String(idx));
      } catch (e) {}
    }

    /* Wire clicks on every palette label.  Each label's `for` attribute
     * points to the radio that label is meant to activate. */
    document.querySelectorAll('label[for^="__palette_"]').forEach(function (label) {
      label.addEventListener("click", function (ev) {
        ev.preventDefault();
        var targetId = label.getAttribute("for");
        var target = document.getElementById(targetId);
        activate(target);
      });
    });

    /* Also catch native radio change events as a defensive fallback. */
    inputs.forEach(function (input) {
      input.addEventListener("change", function () {
        if (input.checked) {
          applyPalette(input);
          showOnlyActiveLabel(inputs);
        }
      });
    });

    activate(pickInitial(inputs));
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bind);
  } else {
    bind();
  }
})();
