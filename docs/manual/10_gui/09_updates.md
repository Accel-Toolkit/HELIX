# Keeping HELIX up to date

HELIX checks for new releases of the official repository
([Accel-Toolkit/HELIX](https://github.com/Accel-Toolkit/HELIX)) a few
seconds after launch, and on demand via **Help → Check for Updates…**.
The launch check is silent unless a newer release actually exists: no
network, a rate-limited API, or an offline machine never produce any
message.  Disable it with **Help → Check for Updates at Startup**.

When a newer release is found you get a status-bar notice
(*⭱ HELIX v1.5 available — click to update*) and the Help entry is
emphasised.  Nothing updates until you ask.

## What "update" does

It depends on how your copy of HELIX was installed:

* **Clean `git clone` of the official repository (on `main`)** — HELIX
  offers to update itself: a fast-forward `git pull` with a progress
  window (the fetch is cancellable; applying the fast-forward is not,
  but it can never create merge conflicts on a clean clone).  On
  success you can restart HELIX in place.
* **Clone with local modifications, a different branch, or a zip
  download** — HELIX opens the release page in your browser instead.
  It will never overwrite local changes or guess at your setup.
* **Developer checkouts** are detected and left alone entirely.

## What it never does

* It never installs Python packages.  If a release changes
  dependencies, the release notes say so — compare with
  `pyproject.toml` if a launch after updating fails.
* It never updates without an explicit confirmation, and never
  restarts the app mid-session without asking.

## Restarting after an update

Choosing *Restart Now* closes HELIX and relaunches `run_gui.sh` for
you.  On Windows, close HELIX and start `run_gui.bat` again manually.
