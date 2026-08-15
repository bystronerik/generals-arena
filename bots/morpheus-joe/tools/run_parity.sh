#!/usr/bin/env bash
# Full-corpus tier-1 parity, plus the mutation check that says whether a green
# result means anything. Required before an N-phase gate and before any
# registry step of morpheus-joe (rewrite-plan §5, joe-net-plan §9).
#
# Twenty-eight surfaces since N1, all of them against Python morpheus and none
# of them a network: what the port deleted, it deleted from both sides. joe's
# forward is covered elsewhere and differently — see tests/morpheus_joe_parity_cases.py.
#
# Minutes, not seconds — hundreds of thousands of cases. The CI-sized slice is
# `pytest -m morpheus bots/morpheus-joe/tests/`, which runs the same code over
# the committed seven-frame fixture in about six seconds.
set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$DIR/../.." && pwd)"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
export PATH="$HOME/.cargo/bin:$PATH"

echo "[parity] building release binary"
cargo build --release --manifest-path "$DIR/Cargo.toml" >&2

echo "[parity] full corpus"
"$PYTHON" "$DIR/tests/morpheus_joe_parity_cases.py" "$@"

# A parity run proves the two implementations agree on the cases it ran; it
# cannot tell you whether those cases reach the behaviour under test. This
# breaks one behaviour at a time and checks the harness notices.
echo "[parity] mutation check (smoke scope)"
"$PYTHON" "$DIR/tools/mutation_check.py" \
  --output "$REPO/docs/research/measurements/morpheus-joe-mutation-check.json"
