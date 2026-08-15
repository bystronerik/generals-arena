#!/usr/bin/env bash
# Repo-harness launcher for unclejoe.
#
# Pin every math backend to one thread before the process starts. The binary
# is single-threaded by construction (no rayon at play time), but the pins
# are free and guard anything a future dependency drags in. One dedicated
# core is a competition constraint, not a tuning choice; the port plan under
# the joe-rs bot docs argues it.
#
# No comment in this file may spell a path. The rating identity is computed
# over a source closure that reads shell scripts for sibling references, so a
# path written in prose here can pull the file it names behind this bot's
# content hash. Name documents in words instead.
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS=1

DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="$DIR/target/release/unclejoe"
STAMP="$DIR/target/.source-stamp"
export UNCLEJOE_ARTIFACT="$DIR/artifact"

# Build only when the sources moved: a match spawns this per seat per game,
# and `cargo build` on an up-to-date tree still costs ~100 ms of dependency
# scanning inside the first-move grace window. The stamp hashes everything
# that can change the binary. Build output is excluded from the bot's content
# hash for the same reason.
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
  echo "[unclejoe] building (sources changed or no binary)" >&2
  # `cd` first: cargo finds its config from the working directory, and that
  # file carries the Linux target-cpu.
  (cd "$DIR" && cargo build --release --manifest-path "$DIR/Cargo.toml") >&2
  mkdir -p "$(dirname "$STAMP")"
  printf '%s' "$WANT" > "$STAMP"
fi

exec "$BIN"
