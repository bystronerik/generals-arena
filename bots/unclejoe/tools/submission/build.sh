#!/usr/bin/env bash
# Submission build step, copied verbatim into the zip by
# tools/package_submission.py. The judge runs it once at intake, with no
# network (RULES.md §08), then never again — matches only ever run run.sh.
#
# It lives here rather than beside run.sh because `matchup.py::build_agent`
# runs any build.sh it finds next to a run.sh, then crashes formatting the log
# line for a path outside the submodule (`build.relative_to(REPO_ROOT)`,
# unguarded where the identical call in `spawn_agent` is guarded) — so a
# build.sh in the bot directory breaks the repo's own verification gate.
#
# **The compile is one crate.** unclejoe has an empty [dependencies], the same
# shape joe-rs arrived at after qualification rejected its 93-crate vendored
# build, so intake compiles this bot's own sources and nothing else.
# `--offline` proves the no-network claim rather than asserting it; `vendor/`
# ships empty for the same reason morpheus's does.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

# `cd`, not just `--manifest-path`. Cargo discovers `.cargo/config.toml` by
# walking up from the **working directory**, not from the manifest, and that
# file is what sets `target-cpu=x86-64-v3`. Building from the wrong cwd
# produces a baseline x86-64 binary with no FMA instruction — where every
# `f32::mul_add` in src/nn/gemm.rs becomes a call to libm's `fmaf()`. Measured
# on morpheus-rs on a one-core x86 container: 277 ms per forward instead of
# 5.6 ms, a 49x pessimisation that fails silently, builds fine, and would
# blow the 150 ms deadline on every move of every game.
cd "$DIR"
cargo build --release --offline --locked --manifest-path "$DIR/Cargo.toml"
echo "[build] $(ls -l "$DIR/target/release/unclejoe")"

# Intake is the last moment where failing is cheaper than playing.
#
# Everything `selfcheck` looks at — the artifact, the manifest, the FMA the
# line above depends on — degrades *silently* at match time on purpose: the
# judge forfeits on an early exit but charges one fault out of fifty for a bad
# reply, so a seat that cannot start passes every turn instead of dying. That
# is the right trade during a game and the wrong one here. A rejected
# submission costs a resubmission; a degraded one costs every rated game it
# plays, with nothing in a match log to say why.
"$DIR/target/release/unclejoe" selfcheck
