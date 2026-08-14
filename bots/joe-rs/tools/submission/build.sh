#!/usr/bin/env bash
# Submission build step, copied verbatim into the vendored-source zip by
# tools/package_submission.py. The judge runs it once at intake, with no
# network (RULES.md §08), then never again — matches only ever run run.sh.
#
# It lives here rather than beside run.sh because `matchup.py::build_agent`
# runs any build.sh it finds next to a run.sh, then crashes formatting the log
# line for a path outside the submodule (`build.relative_to(REPO_ROOT)`,
# unguarded where the identical call in `spawn_agent` is guarded) — so a
# build.sh in the bot directory breaks the repo's own verification gate.
#
# `--offline` is not belt-and-braces, it is the point: without it cargo would
# try to reach crates.io, and a build that only works with a network is a
# build that fails at intake. Every dependency has to be under vendor/ with
# .cargo/config.toml redirecting the registry there. joe-rs has 93 of them —
# candle, gemm, half, libc and the rest — so unlike morpheus this flag is
# carrying real weight, and `--locked` refuses a resolve that would differ
# from the Cargo.lock in the zip.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

# `cd`, not just `--manifest-path`. Cargo discovers `.cargo/config.toml` by
# walking up from the **working directory**, not from the manifest, and that
# file is what sets `target-cpu=x86-64-v3`. Building from the wrong cwd
# produces a baseline x86-64 binary with no FMA instruction — where every
# `f32::mul_add` in the inference kernels becomes a call to libm's `fmaf()`.
# Measured on morpheus on a one-core x86 container: 277 ms per forward instead
# of 5.6 ms, a 49x pessimisation that fails silently, builds fine, and would
# blow the 150 ms deadline on every move of every game.
cd "$DIR"
cargo build --release --offline --locked --manifest-path "$DIR/Cargo.toml"
echo "[build] $(ls -l "$DIR/target/release/joe-rs")"

# Intake is the last moment where failing is cheaper than playing.
#
# joe-rs used to exit 1 when it could not load, which made a broken bundle
# answer nothing and let the packager's own smoke detect it. That was the wrong
# trade in a game: RULES.md §08 forfeits the match on an early exit but charges
# one fault out of fifty for a bad reply, so the seat now degrades to passing
# every turn — and a broken bundle answers well-formed actions, exactly like a
# healthy one. `selfcheck` is what that costs, and this line is where it is
# paid: same constructor, same parser, same `act`, plus the AVX2/FMA check that
# a macOS packaging host cannot make.
"$DIR/target/release/joe-rs" selfcheck
