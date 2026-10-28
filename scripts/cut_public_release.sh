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

echo "── 0/5 commit and version check ───────────────────────────────"
# The stage is this checkout's HEAD (git archive below), so HEAD must be
# what the dev repo publishes: origin/master.  A checkout left on a
# feature branch once staged the wrong commit (2026-10-11).  Release from
# master, or from a worktree of it.
git -C "$DEV_ROOT" fetch -q origin master
HEAD_SHA=$(git -C "$DEV_ROOT" rev-parse HEAD)
ORIGIN_SHA=$(git -C "$DEV_ROOT" rev-parse origin/master)
if [ "$HEAD_SHA" != "$ORIGIN_SHA" ]; then
  echo "REFUSED: HEAD is not origin/master"
  echo "  HEAD          : $(git -C "$DEV_ROOT" log --oneline -1 HEAD)"
  echo "  origin/master : $(git -C "$DEV_ROOT" log --oneline -1 origin/master)"
  echo "Check out master (git switch master && git pull), or cut from a"
  echo "worktree of it: git worktree add <dir> origin/master"
  exit 1
fi
echo "HEAD is origin/master ($(git -C "$DEV_ROOT" log --oneline -1 HEAD | cut -c1-60))"
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
       examples/pipii_multibunch examples/commissioning \
       examples/ring_pipii benchmarks/ring/dev tests/ring/dev \
       scripts/dev_only
rm -f  tests/analysis/test_scc_pxie_anchor.py
# the release scanner's own tests read the dev-only token lists and name a
# dev-only ramp file (scripts/dev_only is removed above): dev only
rm -f  tests/scripts/test_release_scan.py
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
       docs/manual/12_validation/05_pipii_sc_methods.md \
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
             "      - Space-charge methods on the PIP-II MEBT: 12_validation/05_pipii_sc_methods.md\n",
             "      - PIP-II pulse walkthrough: 15_multibunch/02_pipii_pulse.md\n"):
    s = s.replace(line, "")
open("docs/mkdocs.yml", "w").write(s)

targets = ("05_pipii_mebt.md", "04_lebt_rfq.md",
           "02_pipii_validation.md", "06_pipii_full.md",
           "05_pipii_sc_methods.md")
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
        # the methods page's pointer to the PIP-II measurement page
        t = re.sub(r"^Measured on the PIP-II MEBT:[^\n]*\n\n?", "", t,
                   flags=re.M)
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
# Private lattice collections: their tokens live in the dev-only folder
# (removed from the stage in step 2, so this script ships without them).
# The lists are checked first (each pattern against its canary and against
# public text it must not match), then matched against file names and file
# contents; a grep that fails refuses the release instead of passing it.
TOKENS="$DEV_ROOT/scripts/dev_only"
python3 "$DEV_ROOT/scripts/release_scan.py" check-tokens "$TOKENS" \
    || { echo "REFUSED: fix the private token lists in scripts/dev_only/"; exit 1; }
set +e
BADN=$(find . -path ./.git -prune -o -type f -print \
       | grep -iEf "$TOKENS/private_name_tokens.txt"); RC_N=$?
BADC=$(grep -rail --exclude-dir=.git -Ef "$TOKENS/private_content_tokens.txt" .); RC_C=$?
set -e
if [ "$RC_N" -gt 1 ] || [ "$RC_C" -gt 1 ]; then
    echo "REFUSED: the private-token sweep failed (grep exit $RC_N / $RC_C)"; exit 1
fi
if [ -n "$BADN$BADC" ]; then
    echo "PRIVATE LATTICE REFERENCES FOUND — refusing to release:"
    echo "$BADN"; echo "$BADC"; exit 1
fi
# What a plain grep cannot read (compressed or re-encoded text: .npz, PNG
# text chunks, UTF-16), links that leave the stage, and byte-for-byte copies
# of the private lattice files.
python3 "$DEV_ROOT/scripts/release_scan.py" deep . "$TOKENS" \
    || { echo "PRIVATE CONTENT FOUND — refusing to release"; exit 1; }
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
    # Release TITLE stays short (the Releases sidebar truncates long ones
    # into unreadable paragraphs); the full SUMMARY always lands in the
    # notes body below, so nothing is lost.
    SHORT_TITLE="${SUMMARY%%:*}"          # drop anything after a colon
    SHORT_TITLE="${SHORT_TITLE%%.*}"      # ... and after the first sentence
    if [ "${#SHORT_TITLE}" -gt 60 ]; then
        SHORT_TITLE="$(printf '%.60s' "$SHORT_TITLE")…"
    fi
    gh release create "${VERSION}" --repo Accel-Toolkit/HELIX \
        --title "HELIX ${VERSION} — ${SHORT_TITLE}" \
        --notes "${SUMMARY}

Manual: https://accel-toolkit.github.io/HELIX/" \
        || echo "(release object failed — tag manually later)"
    echo "PUBLISHED ${VERSION}"
else
    echo
    echo "DRY RUN complete — inspect $STAGE, then publish with:"
    echo "  scripts/cut_public_release.sh ${VERSION} \"${SUMMARY}\" --push"
fi
