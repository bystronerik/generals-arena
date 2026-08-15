#!/usr/bin/env bash
# Repo-harness launcher for morpheus-joe. The submission zip carries a
# different, build-free run.sh — see tools/package_submission.py.
#
# This bot runs the network of a sibling bot, which this file must describe and
# must never name: `fingerprint._SHELL_REF_RE` scans shell sources for
# bot-relative paths and cannot tell a comment from a `source` line, so a path
# in prose here would pull that bot into this one's content hash and re-identify
# this one every time it changed. (That is not hypothetical — an early
# morpheus-rs run.sh did exactly this.) The weights and the copied source files
# are wired up in Python instead, where a path is only a path.
#
# Pin every math backend to one thread before the process starts, exactly as
# the Python sibling's launcher does. (Naming that file here would be a
# mistake: `fingerprint._SHELL_REF_RE` scans shell sources for bot-relative
# paths, so a path in a *comment* pulls that bot into this one's content hash,
# and editing Python morpheus would silently re-identify this bot.)
# This binary is single-threaded by construction and
# links no BLAS today, but the pins are free and they guard anything a future
# dependency drags in; RAYON_NUM_THREADS is here for the same reason. One
# dedicated core is a competition constraint, not a tuning choice
# (docs/bots/morpheus-rs/rewrite-plan.md §1).
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS=1

DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="$DIR/target/release/morpheus-joe"
STAMP="$DIR/target/.source-stamp"

# Build only when the sources moved. A match spawns this per seat per game, and
# `cargo build` on an up-to-date tree still costs ~100 ms of dependency
# scanning — small, but it lands in the first-move grace window where the real
# bot will want every millisecond. The stamp is a hash of everything that can
# change the binary; `target/` and `vendor/` are excluded from the bot's
# content hash for the same reason (fingerprint._SKIP_DIRS).
source_stamp() {
  {
    find "$DIR/crates" -type f -name '*.rs' -exec shasum -a 256 {} +
    shasum -a 256 "$DIR/Cargo.toml" "$DIR/Cargo.lock" "$DIR/rust-toolchain.toml" 2>/dev/null
  } | LC_ALL=C sort | shasum -a 256 | cut -d' ' -f1
}

WANT="$(source_stamp)"
if [[ ! -x "$BIN" || ! -f "$STAMP" || "$(cat "$STAMP")" != "$WANT" ]]; then
  # Stderr, not stdout: stdout is the wire. A cold checkout pays this once,
  # outside any match, at the caller's leisure.
  echo "[morpheus-joe] building (sources changed or no binary)" >&2
  # `cd` first: cargo finds .cargo/config.toml from the working directory,
  # not from --manifest-path, and that file carries the Linux target-cpu.
  # See tools/submission/build.sh for what building from the wrong cwd costs.
  (cd "$DIR" && cargo build --release --manifest-path "$DIR/Cargo.toml") >&2
  mkdir -p "$(dirname "$STAMP")"
  printf '%s' "$WANT" > "$STAMP"
fi

exec "$BIN"
