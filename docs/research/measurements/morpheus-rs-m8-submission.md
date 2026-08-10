# Morpheus-rs M8 — the submission, and the check it could not pass without

Milestone M8 of [the rewrite plan](../../bots/morpheus-rs/rewrite-plan.md):
zip builder, vendored crates, `build.sh`, size and file-count audit. Measured
2026-08-10. Layout and rules: [packaging.md](../../bots/morpheus-rs/packaging.md).
Offline x86 figures: [morpheus-rs-sandbox-smoke.md](morpheus-rs-sandbox-smoke.md).

**Exit gate: met.** `morpheus-rs-ef5a20484a38.zip` was submitted to
generals.bot by the account holder on 2026-08-10, the sandbox accepted and
built it, and the bot is playing rated games. That closes the one thing this
repo could never test — M0.5 rehearsed it, §15 said so, and the rehearsal was
right.

**The submission evaluation reported zero faults**, over the six games
generals.bot plays against a new bot. That is the first latency evidence this
project has from the *deployment host itself* rather than a proxy: M7's
qualification is a Modal container standing in for a machine nobody can
measure, and this is the machine. It is also six games, so it is a floor rather
than a distribution — RULES.md §08 forfeits at fifty faults in one game, and
M0's Python baseline was late on 13.8% of moves, so zero is a real signal at
this sample size and not a tautology.

**Fault counts are only reported for those six games.** Rated play does not
expose them, so no amount of live play will refine this number: a submission is
the only channel that ever yields fault evidence for a configuration. That is
worth knowing before designing any future latency check — the answer arrives
once, at intake, or not at all.

Two things are *not* recorded here: no intake log was captured, so the
sandbox's own build time and whether `selfcheck`'s output is visible to a
submitter are unknown; and no rated results — wins, losses or forfeits — have
been observed.

## The archive

One archive, `data/bundles/morpheus-rs-<content_hash>.zip`, built by
`tools/package_submission.py --force` and audited through
`arena.bundle.check_limits` — the same code that guards the Python bundles. The
name comes from `arena.bundle.default_output_path` too, so a Rust bundle and a
Python one are the same kind of object on disk and an archive says on its face
which rated program it holds.

| | zip | unpacked | files |
| --- | ---: | ---: | ---: |
| morpheus-rs-ef5a20484a38.zip | 1,042,430 B | 1,364,159 B | 43 |

Against limits of 50 MB, 512 MB and 10,000 files: **2.0% of the zip budget,
0.25% of the unpacked budget, 0.4% of the file count.** The file count was the
binding constraint the plan worried about (§1, R6) and it never came close,
because the crate has no dependencies — which is also why the zip's
`vendor/` is empty and why the offline build path is proved by a separate
probe carrying `sha2`'s transitive graph instead.

It is byte-reproducible: fixed member dates, sorted members, and a
`SUBMISSION.json` with no clock in it. Repackaging produces the same digest, so
"are the bytes I submitted the bytes I still have" is a checksum question.

The archive's `.rs` members are minified on the way in, the launchers
and `.cargo/config.toml` lose their whole-line comments, and the release profile
strips symbols — all added after the milestone's first pass, since the Python
bundler has minified since it existed and the Rust side was shipping every
comment in the crate. Measured: **669,589 → 345,746 bytes** of Rust,
**1,479,416 → 1,298,656 bytes** of binary, and 1,132,733 → 1,042,441 bytes of
zip against the same archive built `--no-minify`. Mechanics
and the reasons in
[packaging.md](../../bots/morpheus-rs/packaging.md#what-the-submission-does-not-carry).

## What M8 found

### A well-formed reply is not evidence of a working bot

Delete `artifact/` from the bundle and the packaging smoke test still passes —
byte-identically. Two well-formed actions, exit zero, green.

That is not a bug in the test. It is the shape of the bot. `Seat::new`
returning `Err` degrades the seat to passing every turn rather than exiting,
because the judge forfeits a game on an early exit but charges one fault out of
fifty for a bad reply (RULES.md §08). Keeping the seat alive is the right
behaviour during a game, and it makes every way the submission can be broken
look identical from outside:

| broken | at play time |
| --- | --- |
| no `artifact/` | legal skip every turn, one line on stderr |
| no `deployment.json` | plays the Part 07 placeholder knobs — 4× the particles, 2× the search depth, a 125 ms deadline against 140 — well, and as a different bot |
| built without a hardware FMA | correct moves, 277 ms per forward, late on every one |

The M3 report already called the third one "a build bug that would have
shipped" and fixed the two launchers that caused it. What it did not add was
anything that would *notice*. `morpheus-rs selfcheck` is that: it resolves
`deployment.json` explicitly instead of falling back, loads the weights against
their manifest digest, constructs the real playing seat with its warmup and
thread-count invariant, reports `gemm::HAS_HARDWARE_FMA`, and decides one
hand-built frame — a general on thirteen army with four empty neighbours, where
a skip is the only wrong answer available. `build.sh` runs it and aborts intake
on a non-zero exit, which is the cheaper failure by a wide margin:
a rejected submission costs a resubmission, a degraded one costs every rated
game it plays.

It earned itself immediately. An aborted packaging run — the negative test for
"does the packager refuse without weights" — left a **partial four-file zip**
in `data/bundles/`, and the next Modal smoke shipped it to an x86
container without noticing. `build.sh`'s selfcheck failed it there. The
packager now stages archives and moves them into place only on success, so the
partial can no longer exist; but the sequence is the milestone in miniature,
because the old smoke would have scored that zip green.

### The submitted bytes play a competition match — and nothing had asked them to

Every verification gate this project has run drove `bots/morpheus-rs/run.sh`:
the repo launcher, which builds from the working tree and finds the artifact by
walking up from `target/`. The zip has a different launcher, a different
directory shape and a build step, and none of it had ever been in front of a
competition match. A two-frame script cannot substitute — it never reaches a
belief update, an admission decision, a castle build or a deathtouch turn.

`package_submission.py --gate` extracts the archive, runs its own `build.sh`,
and plays the AGENTS.md gate with the result:

```
[matchup] turn 182: player 0 (extracted-morpheus-rs-vendored) built a castle at (13, 2)
[matchup] turn 277: player 0 (extracted-morpheus-rs-vendored) built a castle at (8, 0)
[matchup] turn 512: player 0 captured the enemy general
[matchup] castles built: 2 (extracted-morpheus-rs-vendored) vs 0 (cm_expander)
```

An earlier run of the same seed against the same opponent won at turn 592 with
one castle. Same bytes, same seed, different game — the controller is
deadline-driven, so how much search a turn affords depends on the host, which
is M6's lesson restated at the scale of a single match.

One deviation, stated rather than hidden: **`build.sh` is run and then
deleted** before `matchup.py` is invoked. That models the end of intake exactly
— the judge builds once and afterwards only ever spawns `run.sh` — and it is
also the only way the gate can run at all. `matchup.py::build_agent` executes
any `build.sh` beside a `run.sh` and formats its log line with
`build.relative_to(REPO_ROOT)`, unguarded, where `REPO_ROOT` is the
*submodule's* root; every path outside `competition-module/` raises. So a
submission-shaped directory crashes the repo's own gate before the first move,
wherever it is put. `packaging.md` has recorded that defect since M0.5 as a
reason to keep `build.sh` out of the bot directory; M8 is where it stopped
being a packaging inconvenience and started constraining what can be verified.
What goes untested is `build_agent`, which the judge does not have.

### R4's fallback played a different speed, so it was retired

The static musl variant existed so the plan's R4 had a tested escape from a
sandbox that cannot build the vendored tree, built and smoked on every
packaging run for exactly that reason. On one x86 core, five runs each:

| variant | warmup ms (26 forwards) | one decision, ms | artifact load, ms |
| --- | --- | --- | --- |
| vendored (glibc) | 107.8, 107.7, 108.1, 107.9, 107.7 | 4.7, 4.5, 4.7, 4.8, 4.7 | 7.3 – 9.5 |
| static (musl) | 109.3, 109.9, 111.4, 109.0, 108.8 | 9.9, 11.7, 10.0, 10.9, 10.3 | 14.4 – 17.2 |

Replicated: an earlier container on the same image read 5.4/5.4/4.9/4.9/5.4
against 8.0/9.0/8.9/9.0/9.6, so the ratio moves between runs and its direction
does not.

The inference is *identical* — 26 forwards land within 2% of each other, so the
compile target, the FMA and the kernels are the same in both. Everything that
allocates is not: a decision costs **1.75× to 2.2×** more under musl and the
artifact load ~1.9×, which is what musl's allocator against glibc's looks like.

Consequence: **the fallback never inherited M7's latency qualification.** That
qualification is `n8-s32-b4-d8` at p99.9 = 140 ms with zero moves over 150 on
one x86 core, measured with the gnu build. A doubling of the decision path is
not a rounding error against a 150 ms judge limit, so taking the fallback meant
shipping a bot nobody had measured — which is not the safety §9 wanted from it.

The packager therefore stops building it. One archive comes out, and if R4 ever
fires the musl variant must be rebuilt from git history and qualified before it
plays; an allocator would be the other route, and would be this crate's first
dependency ever. That is a worse position than §9 imagined and a more honest
one than a tested fallback nobody could responsibly use.

## Everything else, measured

- **Intake build, offline, one x86 core: 17–18 s.** Fat LTO and a single
  codegen unit over 17,600 lines. There is no documented intake timeout; this
  leaves room under any plausible one.
- **`block_network=True` is verified, not asserted:** a curl to crates.io in
  the same container exits 6, and the vendor probe (571 files, `sha2` →
  `digest` → `block-buffer` → `generic-array` → `typenum`, plus a build script)
  builds offline in 2.4 s.
- **Both variants and both hosts decide the same frame the same way**
  (`0 10 10 0 0`) — arm64 macOS, x86 glibc, x86 musl.
- **`cargo test --release`** passes; `pytest tests bots/morpheus-rs/tests` is
  green apart from `tests/test_bundle.py::test_bundle_runs_standalone`, which
  fails on the *Python* bundle path and is untouched by this milestone.

## Submitting

Only the account holder can do this part. Done once, on 2026-08-10, with
`ef5a20484a38`; the procedure below is what the next one follows.

```bash
python bots/morpheus-rs/tools/package_submission.py --force --gate
```

Then upload `data/bundles/morpheus-rs-<content_hash>.zip` to generals.bot. Its
`SUBMISSION.json` records the content hash of the program inside it, which is
the row to look for in `data/bot_versions/morpheus-rs.json` when a rated result
comes back — so **commit and register before packaging the zip you actually
upload**, or the hash inside it will name a closure that is nowhere in the
repo's history:

```bash
python -m arena.records.registry --register morpheus-rs --strict
```

`SUBMISSION.json` also carries `git_dirty`; a `true` there means the archive
was built from a working tree and its hash cannot be looked up.

If the sandbox rejects the build — R4's tripwire — there is no second archive
to fall back to since M8 retired it; the musl variant has to be rebuilt from
git history and qualified before it plays. **The tripwire did not fire:** the
vendored-source zip built inside the judge's own image on the first attempt,
which is the evidence the removal was waiting on and did not have when the
decision was taken.

Still to check on the other side: whether `build.sh`'s selfcheck output is
legible in whatever intake log exists. The fault question is answered as well
as it can be — the submission evaluation's six games are the only place
generals.bot reports faults at all.

**The shipped knobs are `n8-s16-b4-d8`, and until the submission nobody had
measured them.** A
`Configuration update` commit (`386d9ae`) landed on `deployment.json` after M8's
own commit, taking `search_depth` from 2 to 8 and replacing the note with the
s32 arm's — which claims `n8-s32-b4-d8` while leaving `target_simulations` at
16. The decision was to ship the file as it stands and correct only the note, so
the archive now says what it is. What it is:

| config | measured |
| --- | --- |
| `n8-s32-b4-d8` | M7's x86 qualification — p99.9 140 ms, max 141, zero moves over 150, twenty simulations; strength `unproven` at +18.15 ± 23.42 |
| `n8-s16-b4-d2` | the parity knobs — M6's rated arm and what the parity harness ran against |
| **`n8-s16-b4-d8`** | **neither — and, since the submission, zero faults over the judge's six evaluation games** |

M7's cost model says a simulation's cost is its leaf forward rather than its
tree walk, so depth 2 → 8 should be nearly free and this is very likely inside
the deadline. That was an inference from other points of the grid rather than a
measurement of this one — the shape of claim this project keeps recording as
its mistakes — and the six-game evaluation is the first thing to test it. It
agreed. Note what that does and does not settle: it is the deadline behaviour on
the real host, at a sample size of six games, and it says nothing about
strength. `deployment.json` still reads *no on any host*, and deliberately so —
editing it would mint a new content hash and re-identify the bot that is
currently playing.
