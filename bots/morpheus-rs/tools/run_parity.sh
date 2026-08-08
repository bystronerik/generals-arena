#!/usr/bin/env bash
# Full-corpus tier-1 parity, plus the mutation check that says whether a green
# result means anything. Required before an M5/M6 exit gate and before any
# registry step of morpheus-rs (rewrite-plan §5).
#
# Minutes, not seconds — hundreds of thousands of cases. The CI-sized slice is
# `pytest bots/morpheus-rs/tests/`, which runs the same code over the committed
# seven-frame fixture in about a second.
set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$DIR/../.." && pwd)"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
export PATH="$HOME/.cargo/bin:$PATH"

echo "[parity] building release binary"
cargo build --release --manifest-path "$DIR/Cargo.toml" >&2

echo "[parity] full corpus"
"$PYTHON" "$DIR/tests/parity_cases.py" "$@"

# A parity run proves the two implementations agree on the cases it ran; it
# cannot tell you whether those cases reach the behaviour under test. This
# breaks one behaviour at a time and checks the harness notices.
echo "[parity] mutation check (smoke scope)"
"$PYTHON" "$DIR/tools/mutation_check.py" \
  --output "$REPO/docs/research/measurements/morpheus-rs-mutation-check.json"
