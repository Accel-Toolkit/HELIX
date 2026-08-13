#!/usr/bin/env bash
# Cut a public release of HELIX from this development checkout.
#
#   scripts/cut_public_release.sh v1.1 "one-line summary"          # dry run
#   scripts/cut_public_release.sh v1.1 "one-line summary" --push   # publish
#
# What it does, every time, identically:
#   1. exports the CURRENT COMMITTED tree (git archive — untracked files
#      can never leak) into ../HELIX_public
#   2. applies the standing exclusion list: PIP-II lattice content of any
#      kind, third-party field maps, dev-only notes/scripts
#   3. heals the manual (removes the PIP-II chapters from nav, unlinks
#      references) and hard-fails if `mkdocs build --strict` breaks
#   4. runs a remnant sweep and HARD-FAILS on any non-allowlisted match
#   5. grafts ONE new commit "HELIX <version> — <summary>" onto the public
#      repo's existing history (clean release ladder), and pushes only
#      with --push
set -euo pipefail

VERSION=${1:?usage: cut_public_release.sh vX.Y \"summary\" [--push]}
SUMMARY=${2:?usage: cut_public_release.sh vX.Y \"summary\" [--push]}
PUSH=${3:-}
PUBLIC_URL="https://github.com/Accel-Toolkit/HELIX.git"
DEV_ROOT=$(git rev-parse --show-toplevel)
STAGE="$DEV_ROOT/../HELIX_public"

echo "── 0/5 version check ──────────────────────────────────────────"
# The tree's version strings ARE the release version (the in-app update
# checker compares linac_gen.__version__ against the GitHub tag) — bump
# them in a commit FIRST, then cut.  A mismatched cut is refused.
VNUM="${VERSION#v}"
SRC_V=$(grep -Eo '__version__ = "[^"]+"' "$DEV_ROOT/linac_gen/__init__.py" | cut -d'"' -f2)
TOML_V=$(grep -Eo '^version = "[^"]+"' "$DEV_ROOT/pyproject.toml" | cut -d'"' -f2)
CFF_V=$(grep -Eo '^version: "[^"]+"' "$DEV_ROOT/CITATION.cff" | cut -d'"' -f2)
for v in "$SRC_V" "$TOML_V" "$CFF_V"; do
  if [ "$v" != "$VNUM" ]; then
    echo "REFUSED: VERSION $VERSION does not match the tree"
    echo "  linac_gen/__init__.py : $SRC_V"
    echo "  pyproject.toml        : $TOML_V"
    echo "  CITATION.cff          : $CFF_V"
    echo "Bump the version strings in a commit first, then cut."
    exit 1
  fi
done
echo "version strings match $VNUM"

# The landing pages carry human-facing version strings the update
# checker never sees — v1.7 shipped with v1.6 badges because nothing
# gated them.  Every vX.Y token in web/*.html must equal v$VNUM, and
# every version-context line in about.html (excluding the literal
# cff-version key) must carry $VNUM.
WEB_BAD=$(grep -RhoE "v[0-9]+\.[0-9]+(\.[0-9]+)?" "$DEV_ROOT"/web/*.html \
          | sort -u | grep -vx "v$VNUM" || true)
ABOUT_BAD=$(grep -iE "version" "$DEV_ROOT/web/about.html" \
            | grep -v "cff-version" \
            | grep -oE "[0-9]+\.[0-9]+(\.[0-9]+)?" \
            | sort -u | grep -vx "$VNUM" || true)
if [ -n "$WEB_BAD$ABOUT_BAD" ]; then
  echo "REFUSED: web/ landing pages carry stale version strings"
  echo "  badges: ${WEB_BAD:-ok}"
  echo "  about.html: ${ABOUT_BAD:-ok}"
  echo "Update web/ (brand badges + about.html card/CFF/BibTeX), then cut."
  exit 1
fi
echo "web/ version strings match $VNUM"

echo "── 1/5 export committed tree ──────────────────────────────────"
rm -rf "$STAGE"
mkdir -p "$STAGE"
git -C "$DEV_ROOT" archive HEAD | tar -x -C "$STAGE"

echo "── 2/5 exclusions ─────────────────────────────────────────────"
cd "$STAGE"
rm -rf examples/pipii examples/MEBT_To_Foil examples/pipii_tunable \
       examples/pip2_misalignment_study examples/pipii_hwr_ssr1_match \
       examples/piplattice examples/emittance_min examples/lebt_pxie \
       examples/lebt_plus_rfq examples/hebt_diag \
       examples/pipii_multibunch
rm -f  tests/analysis/test_scc_pxie_anchor.py
# Instability-paper machinery: HELD from public releases until the PRAB
# paper is submitted (user decision 2026-08-06) — remove these lines to
# ship the tools alongside the paper.
rm -rf examples/instability_atlas
rm -f  linac_gen/analysis/envelope_floquet.py \
       linac_gen/analysis/mode_growth.py \
       linac_gen/analysis/floquet_uq.py \
       linac_gen/analysis/smooth_baseline.py \
       tests/analysis/test_envelope_floquet.py \
       tests/analysis/test_mode_growth.py \
       tests/analysis/test_floquet_uq.py
rm -f  examples/mebt_plus_hwr*.dat examples/mebt_pipii.dat \
       examples/mebt_line.dat examples/mebt_pipii_demo.py \
       examples/test.lgproj mebt_pipii.dat \
       REVIEW.md LICENSING_PLAN.md mebt_hwr_plots.py \
       diag_lebt_3cases.py compare_mebt_hwr.py \
       diag_fnalscl_diag_matching.py
rm -f  tests/errors/test_pip2_study_defs.py \
       tests/errors/test_btl_alignment_tolerance.py
rm -f  docs/manual/11_examples/04_lebt_rfq.md \
       docs/manual/11_examples/05_pipii_mebt.md \
       docs/manual/11_examples/06_pipii_full.md \
       docs/manual/12_validation/02_pipii_validation.md \
       docs/manual/15_multibunch/02_pipii_pulse.md \
       docs/manual/_build/figures/fig_15_01_pip2_pulse_droop.png

echo "── 3/5 heal the manual ────────────────────────────────────────"
python3 - <<'PY'
import os
import re

s = open("docs/mkdocs.yml").read()
for line in ("      - LEBT + RFQ: 11_examples/04_lebt_rfq.md\n",
             "      - PIP-II MEBT: 11_examples/05_pipii_mebt.md\n",
             "      - Full PIP-II: 11_examples/06_pipii_full.md\n",
             "      - PIP-II validation: 12_validation/02_pipii_validation.md\n",
             "      - PIP-II pulse walkthrough: 15_multibunch/02_pipii_pulse.md\n"):
    s = s.replace(line, "")
open("docs/mkdocs.yml", "w").write(s)

targets = ("05_pipii_mebt.md", "04_lebt_rfq.md",
           "02_pipii_validation.md", "06_pipii_full.md")
for root, _, files in os.walk("docs/manual"):
    for f in files:
        if not f.endswith(".md"):
            continue
        p = os.path.join(root, f)
        t = open(p).read()
        orig = t
        for tgt in targets:
            t = re.sub(r"\[([^\]]+)\]\((?:\.\./)*"
                       r"(?:11_examples/|12_validation/)?"
                       + re.escape(tgt) + r"\)", r"\1", t)
        # drop navigation stubs that pointed at removed pages
        t = re.sub(r"^\* \[PIP-II validation\][^\n]*\n", "", t, flags=re.M)
        t = re.sub(r"← \[Full PIP-II\][^\n]*\n?", "", t)
        t = re.sub(r"\[Continue to PIP-II[^\]]*\][^\n]*", "", t)
        # multibunch page: drop the whole worked-example pointer section
        # (the walkthrough page + example dir are removed above)
        t = re.sub(r"## Worked example\n\n\[Continue to the PIP-II "
                   r"pulse walkthrough\][^\n]*\n?", "", t)
        if t != orig:
            open(p, "w").write(t)
print("manual healed")
PY
PYTHONPATH="$STAGE:$STAGE/gui" python3 docs/manual/_build/regen_plots.py > /dev/null
( cd docs && mkdocs build --strict > /dev/null )
echo "mkdocs --strict OK"
# build artifacts verified — drop them (CI regenerates; the sweep and
# the commit must only ever see distributable content)
rm -rf docs/site site
find docs/manual/_build/figures -maxdepth 1 -name "*.png" -delete 2>/dev/null || true

echo "── 4/5 remnant sweep ─────────────────────────────────────────"
BAD=$(find . -path ./.git -prune -o -type f -print \
      | grep -iE "pipii|pip2|fnalscl|mebt|hebt|SOL.-PXIE|Tracewin_code|TraceWIn_Tools" \
      | grep -vE "test_review|test_live_match_preview|test_engine_preview_hook" \
      || true)
if [ -n "$BAD" ]; then
    echo "REMNANTS FOUND — refusing to release:"; echo "$BAD"; exit 1
fi
echo "sweep clean"

echo "── 5/5 graft the release commit ──────────────────────────────"
git clone --quiet "$PUBLIC_URL" "$STAGE/.pub"
mv "$STAGE/.pub/.git" "$STAGE/.git"
rm -rf "$STAGE/.pub"
git config user.name  "Abhishek Pathak"
git config user.email "59810426+Abhishek-Pathak-90@users.noreply.github.com"
git add -A
if git diff --cached --quiet; then
    echo "nothing changed since the last release — aborting"; exit 0
fi
git commit -q -m "HELIX ${VERSION} — ${SUMMARY}"
git log --oneline -3
if [ "$PUSH" = "--push" ]; then
    git push origin main
    # a real GitHub Release (tag + notes) for the Releases sidebar
    gh release create "${VERSION}" --repo Accel-Toolkit/HELIX \
        --title "HELIX ${VERSION} — ${SUMMARY}" \
        --notes "${SUMMARY}

📖 Manual: https://accel-toolkit.github.io/HELIX/" \
        || echo "(release object failed — tag manually later)"
    echo "PUBLISHED ${VERSION}"
else
    echo
    echo "DRY RUN complete — inspect $STAGE, then publish with:"
    echo "  scripts/cut_public_release.sh ${VERSION} \"${SUMMARY}\" --push"
fi
