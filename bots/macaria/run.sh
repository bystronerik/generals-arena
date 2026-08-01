#!/usr/bin/env bash
# Resolve this script's directory so `python main.py` works no matter where
# the caller invoked us from (matchup.py sets cwd to the script dir, but
# absolute paths stay correct if someone runs the script directly).
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${PYTHON:-python3}" -u "$DIR/main.py"
