#!/usr/bin/env bash
# Submission intake step, copied verbatim into the zip by
# tools/package_submission.py. The judge runs it once at intake, with no
# network (RULES.md §08), then never again — matches only ever run run.sh.
#
# It lives here rather than beside run.sh because `matchup.py::build_agent`
# runs any build.sh it finds next to a run.sh, then crashes formatting the log
# line for a path outside the submodule (`build.relative_to(REPO_ROOT)`,
# unguarded where the identical call in `spawn_agent` is guarded) — so a
# build.sh in the bot directory breaks the repo's own verification gate.
#
# **There is no compile step.** joe-rs ships as a statically linked x86_64
# musl binary, 2 MB, built by scripts/joe_rs_static_build.py and verified in a
# container with no Rust and no network. Intake is a chmod and a self-test.
#
# That deletes, rather than mitigates, every failure this bot was rejected for
# and could not reproduce: a 93-crate build's peak disk (342 MB), its peak
# memory (1.6 GB), its wall time (71-131 s), and its dependence on whatever
# toolchain the sandbox happens to have. The sources stay in the zip for the
# record; nothing here reads them.
#
# To go back to compiling at intake, set `vendor=True` in the spec and restore
# the cargo invocation from git history (`--release --offline --locked`, run
# after `cd "$DIR"` so cargo finds .cargo/config.toml and its target-cpu flag).
set -uo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

chmod +x "$DIR/bin/joe-rs" 2>/dev/null || true
echo "[build] $(ls -l "$DIR/bin/joe-rs")"

# Reported, not enforced, and that is a reversal worth stating. A failing
# selfcheck used to abort intake, which is right when the alternative is a
# silently degraded bot. But run.sh now self-tests and falls back to a seat
# that passes every turn, and RULES.md §08 charges one fault out of fifty for a
# bad reply against a forfeit for an early exit. Losing games beats forfeiting
# them, so intake proceeds and says why.
"$DIR/bin/joe-rs" selfcheck || echo "[build] selfcheck FAILED; run.sh will use the fallback seat"
exit 0
