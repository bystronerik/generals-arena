# Shared launcher body for competition-module benchmark bots.
# Source from bots/cm_*/run.sh after setting DIR to that bot directory.
REPO="$(cd "$DIR/../.." && pwd)"
BOTS="$REPO/bots"
CM="$REPO/competition-module"
export PYTHONPATH="${BOTS}:${CM}${PYTHONPATH:+:$PYTHONPATH}"
if [ -z "${PYTHON:-}" ] && [ -x "$REPO/.venv/bin/python" ]; then
  PYTHON="$REPO/.venv/bin/python"
fi
exec "${PYTHON:-python3}" -u "$DIR/main.py"
