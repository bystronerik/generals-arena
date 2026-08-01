#!/usr/bin/env bash
# Resolve this script's directory so `python main.py` works no matter where
# the caller invoked us from (matchup.py sets cwd to the script dir, but
# absolute paths stay correct if someone runs the script directly).
DIR="$(cd "$(dirname "$0")" && pwd)"
# yankee imports numpy (search.py). numpy is in the competition sandbox's own
# pins (competition/requirements.txt: numpy==2.4.6), so the submitted bot is
# fine — but a bare `python3` on a dev box is whatever PATH found first and
# usually has no numpy, which turns the AGENTS.md verification gate into a
# BrokenPipeError with no message. Fall back to the repo venv exactly as
# bots/_common/cm_run.sh already does for the benchmark wrappers; an explicit
# PYTHON still wins, so the arena's own launcher is unaffected.
REPO="$(cd "$DIR/../.." && pwd)"
if [ -z "${PYTHON:-}" ] && [ -x "$REPO/.venv/bin/python" ]; then
  PYTHON="$REPO/.venv/bin/python"
fi
exec "${PYTHON:-python3}" -u "$DIR/main.py"
