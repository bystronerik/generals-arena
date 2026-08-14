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

# No `selfcheck` line here, unlike morpheus's build.sh, and the difference is
# deliberate rather than an omission (packaging.md §5.3). morpheus needs one
# because its seat degrades to passing every turn when it cannot load — so a
# broken bundle still answers well-formed actions. joe-rs constructs its seat
# with `Seat::new(...)?`, so a load failure leaves `main` with `exit(1)` and a
# broken bundle answers nothing at all, which the packager's own smoke already
# detects. What stays uncovered is the per-turn error path and a build that
# lost the target flag; closing those costs a new subcommand in main.rs, which
# is in the rated closure and would re-identify the bot.
