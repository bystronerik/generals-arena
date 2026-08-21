#!/usr/bin/env bash
# Repo-harness launcher for joe-rs.
#
# Pin every math backend to one thread before the process starts. The binary
# is single-threaded by construction (no rayon at play time), but the pins
# are free and guard anything a future dependency drags in. One dedicated
# core is a competition constraint, not a tuning choice
# (docs/bots/joe-rs/port-plan.md §1).
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS=1

# Frozen S4 round arm: the trail penalty pinned on for the rated
# contrast. Do not edit while the round runs.
export JOE_RS_NOUNDO=4
export JOE_RS_NOUNDO_WINDOW=8

DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="$DIR/target/release/joe-rs"
STAMP="$DIR/target/.source-stamp"
export JOE_RS_ARTIFACT="$DIR/artifact"

# Build only when the sources moved: a match spawns this per seat per game,
# and `cargo build` on an up-to-date tree still costs ~100 ms of dependency
# scanning inside the first-move grace window. The stamp hashes everything
# that can change the binary; `target/` and `vendor/` are excluded from the
# bot's content hash for the same reason (fingerprint._SKIP_DIRS).
source_stamp() {
  {
    find "$DIR/src" -type f -name '*.rs' -exec shasum -a 256 {} +
    shasum -a 256 "$DIR/Cargo.toml" "$DIR/Cargo.lock" "$DIR/rust-toolchain.toml" 2>/dev/null
  } | LC_ALL=C sort | shasum -a 256 | cut -d' ' -f1
}

WANT="$(source_stamp)"
if [[ ! -x "$BIN" || ! -f "$STAMP" || "$(cat "$STAMP")" != "$WANT" ]]; then
  # Stderr, not stdout: stdout is the wire. A cold checkout pays this once,
  # outside any match.
  echo "[joe-rs] building (sources changed or no binary)" >&2
  # `cd` first: cargo finds .cargo/config.toml from the working directory,
  # and that file carries the Linux target-cpu.
  (cd "$DIR" && cargo build --release --manifest-path "$DIR/Cargo.toml") >&2
  mkdir -p "$(dirname "$STAMP")"
  printf '%s' "$WANT" > "$STAMP"
fi

exec "$BIN"
