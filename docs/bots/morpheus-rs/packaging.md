# Morpheus-rs build and packaging

How the Rust bot builds in the repo, how it is packaged for generals.bot, and
which of its facts are load-bearing. Established at milestone M0.5 of the
[rewrite plan](rewrite-plan.md), whose whole purpose is to find out whether the
submission path works *before* any of the port is written.

## Layout

```
bots/morpheus-rs/
  Cargo.toml            workspace: crates/core (lib) + crates/bot (bin)
  Cargo.lock            pins the graph; the reason vendor/ need not be hashed
  rust-toolchain.toml   dev-side pin, deliberately NOT shipped (below)
  .cargo/config.toml    target-cpu for the Linux targets only
  crates/core/src/      the ported bot; wire, board, network, parity
  crates/bot/src/       main: read frames, decide, reply; `parity` and `bench`
  artifact/             safetensors weights + manifest; inside the content hash
  deployment.json       the coupled knobs; inside the content hash (M5)
  run.sh                repo launcher — builds when stale, then execs
  tools/                dev tooling; outside the content hash
    submission/build.sh the build.sh copied into the vendored zip
    package_submission.py
    vendor_probe.py     proves the offline build with real vendored crates
    dependency_budget.py what a crate would cost in the zip
    capture_morpheus.py the M0 capture module
    convert_artifact.py TorchScript -> safetensors, reproducibly
    bench_inference.py  the M3 shoot-out, local half
    spikes/             candle and tract crates kept as evidence, never linked
  tests/                pytest parity slice + the selfcheck guard; unhashed
  target/  vendor/      build output and vendored crates; gitignored, unhashed
```

Each zip additionally carries a generated `run.sh`, a generated `build.sh`, and
`SUBMISSION.json` — the bot id, content hash, git commit and weights digest of
the program inside it. The judge never reads that file; a human comparing a
rated result on generals.bot against a row in `data/bot_versions/morpheus-rs.json`
does, and without it "which program is playing up there" is answerable only by
rebuilding and hoping. It deliberately carries no timestamp, because the zips
are byte-reproducible from the sources and a clock would end that.

## Toolchain

`rustc`/`cargo` **1.97.1**, matching the sandbox's 1.97 stable. Installed with
rustup; `rust-toolchain.toml` pins the channel and carries the
`x86_64-unknown-linux-musl` target for the fallback variant.

## Content hash

`bots/morpheus-rs/` is rated on its sources: the `.rs` files, `Cargo.toml`,
`Cargo.lock`, `rust-toolchain.toml`, `.cargo/config.toml`, `run.sh`, and — from
M3 — `artifact/`. `fingerprint._SKIP_DIRS` gained `target`, `vendor`, and
`tools` for this — the first two because build output is what sources compile
*to* and vendored crates are a copy of crates.io the lock file already pins,
the third for the same reason `probe.py` is excluded: arena-owned tooling that
never plays must not fork a rating identity when it is edited.

**The artifact is inside the hash, so its bytes have to be reproducible.**
`artifact/model.safetensors` is a conversion of the frozen Python artifact, and
a bot that plays different weights is a different bot — so it belongs in the
identity. That makes byte-reproducibility a hash requirement rather than a
nicety, and it is the reason `tools/convert_artifact.py` writes the
safetensors container itself instead of calling `safetensors.torch.save_file`:
the library serializes its metadata block from a Rust `HashMap` whose order
varies per process, so two conversions of identical weights would mint two bot
identities. `convert_artifact.py --check` verifies the committed files
reproduce exactly.

**A shell comment can widen the closure.** `_SHELL_REF_RE` scans shell sources
for bot-relative paths and cannot tell a comment from a `source` line. An early
`run.sh` mentioned the Python bot's launcher in prose, which put
`bots/morpheus/run.sh` inside morpheus-rs's hash — so editing Python morpheus
would have silently re-identified the Rust bot. Refer to sibling bots by
description, not by path, in any `.sh` under `bots/`.

## Two variants, both built every time

```bash
python bots/morpheus-rs/tools/package_submission.py --force
```

**vendored** — sources, `Cargo.lock`, `vendor/`, and a `build.sh` running
`cargo build --release --offline --locked`. The intended submission: the judge
compiles it at intake. Note `vendor/` is currently **empty**, because the crate
has no dependencies — which is why the probe below exists.

**static** — a prebuilt `x86_64-unknown-linux-musl` binary with `build.sh`
reduced to a no-op plus the selfcheck below. This is R4's fallback (§9) for a
sandbox that cannot build the vendored tree. It is built and smoked on every
packaging run because a fallback nobody exercises is a fallback nobody can rely
on.

**The fallback plays a different speed.** Measured on one x86 core at M8: the
26 warmup forwards land within 2% of the vendored variant — same target, same
FMA, same kernels — while one decision costs **1.75× to 2.2×** more and the
artifact load ~1.9×, which is musl's allocator against glibc's. So the static
variant does *not* inherit M7's latency qualification (p99.9 = 140 ms, zero
moves over 150). If R4 is ever triggered, the fallback needs its own measured
knobs before it plays rated games, or an allocator — which would be this
crate's first dependency and should be argued from a measurement. Figures:
[morpheus-rs-m8-submission.md](../../research/measurements/morpheus-rs-m8-submission.md).

Both are built to a scratch path and moved into place only on success. Writing
straight to `data/bundles/` leaves a **partial archive** when a member check
fails mid-zip, and nothing downstream can tell one from a finished bundle: an
aborted static build left a four-file zip with no weights, which the Modal
smoke then shipped to an x86 container. The `selfcheck` below is what caught
it.

**Both variants carry `artifact/` and `deployment.json`.** That is a
correction, not a description: until M5 the static variant shipped only the
binary and its launchers, which was right for the M0.5 pass bot, quietly wrong
once M3 gave the bot weights to load, and wrong twice over once M5 gave it a
configuration to read. ~~A missing `deployment.json` is the nastier of the two
— the artifact's absence fails loudly at load~~ — **wrong, and M8 measured it.**
Neither fails loudly. The missing config falls back to the Part 07 placeholders
and plays a *different bot* (four times the particles, twice the search depth, a
125 ms deadline against 140); the missing artifact degrades the seat to a legal
skip on every turn. Both answer the protocol perfectly, and the difference is
invisible in a match log — see the next section. The packager now refuses to
build either variant without both, `try_load_deployment` says on stderr when it
falls back, and the lookup tries the binary's own directory before
`target/../..` so the static layout resolves.

Both are audited through `arena.bundle.check_limits` — the same code that
guards the Python bundles, so the two cannot drift — then extracted to a
scratch directory with no repo around them, built, and driven with a scripted
frame.

## A well-formed reply is not evidence of a working bot

The single most important fact about this bot's failure modes, found at M8:

**Delete `artifact/` and the smoke test still passes.** Byte-identically. Two
well-formed actions, exit zero, green.

That is not a bug in the test, it is the shape of the bot. `Seat::new`
returning `Err` degrades the seat to passing every turn rather than exiting,
because the judge forfeits a game on an early exit but charges one fault out of
fifty for a bad reply (RULES.md §08) — so the *right* behaviour during a game is
to keep answering. Every way the submission can be broken produces a bot that
speaks the protocol perfectly and loses every game:

| broken | what it looks like at play time |
| --- | --- |
| no `artifact/` | legal skip every turn, one line on stderr |
| no `deployment.json` | plays the Part 07 placeholder knobs — four times the particles, twice the search depth, a 125 ms deadline against 140 — well, and as a different bot |
| built without a hardware FMA | correct moves, 277 ms per forward, late on every one |

None of the three is visible in a match log, a reply stream, or a file count.

`morpheus-rs selfcheck` is the one place that refuses. It resolves
`deployment.json` explicitly instead of falling back, loads the weights against
their manifest digest, constructs the real playing seat with its warmup and
thread-count invariant, reports `gemm::HAS_HARDWARE_FMA`, and decides one
hand-built frame — a general on thirteen army with four empty neighbours, where
a skip is the only wrong answer available. Any failure exits non-zero.

**Both `build.sh` variants run it**, so a broken submission is rejected at
intake rather than rated. That is the whole trade: a rejected submission costs a
resubmission, a degraded one costs every rated game it plays. The packager
parses its output, `bots/morpheus-rs/tests/test_selfcheck.py` proves it still
notices a bot with its weights removed, and the Modal smoke requires it — a
checker nobody checks is exactly the thing it was written to prevent.

`warmup_ms` doubles as a per-forward probe: it is 26 forwards, so ~4.7 ms each
on the M3 Pro against M7's ~5.1 ms on one x86 core, and a build that lost its
FMA would read fifty times that.

## The offline build is proven by a probe, not by the shipping zips

The two shipping variants have no dependencies. That makes their
`cargo build --offline` cheap to pass and nearly meaningless: with nothing to
resolve, `--offline` cannot fail. It never exercises `.cargo/config.toml`
source replacement, a transitive graph, or a build script running at intake —
and M3's inference crate brings ninety crates of exactly that.

`tools/vendor_probe.py` closes the gap without putting a dependency in the
shipped binary. It builds a scratch crate that depends on `sha2` — chosen for
its shape: a transitive chain (`digest` → `block-buffer` → `generic-array` →
`typenum`) and a build script via the `version_check` build-dependency, which
is the failure mode most likely to be missed. It vendors, packages in exactly
the submission shape, builds offline locally, and rides the Modal smoke as a
third variant.

```bash
python bots/morpheus-rs/tools/vendor_probe.py
modal run scripts/morpheus_rs_modal_submission_smoke.py
```

Result: 571 files, 6.3 MB unpacked, builds in 2.5 s inside a `block_network`
container on the sandbox toolchain, and answers the protocol.

## What a dependency costs

```bash
python bots/morpheus-rs/tools/dependency_budget.py --crate tract-onnx=0.21
```

Measured, because the plan's §1 originally asserted that `cargo vendor` "blows
past 10k files easily" and that assertion had already been used to justify
hand-writing SHA-256. It is not true — `sha2` is 5.6% of the cap. Current
figures live in
[morpheus-rs-dependency-budget.md](../../research/measurements/morpheus-rs-dependency-budget.md);
the budget binds at M3's inference crate, not before.

## Things that would have broken the submission

**`rust-toolchain.toml` must not ship.** It pins an exact patch release. A
sandbox whose stable toolchain is any other patch would see the pin and try to
*download* the pinned one, over a network that does not exist at intake. The
pin is a development-side guarantee that nothing newer than 1.97 gets used;
shipping it converts that guarantee into a build failure.

**`build.sh` must not sit beside `run.sh` in the repo.** `matchup.py`'s
`build_agent` runs any `build.sh` it finds next to a `run.sh`, then crashes
formatting its log line — `build.relative_to(REPO_ROOT)` against the
submodule's root, unguarded, where the identical call in `spawn_agent` twenty
lines below *is* guarded. So a `build.sh` in the bot directory breaks the
repo's own verification gate. It lives in `tools/submission/` and is copied
into the zip. (A one-line submodule fix would work too; AGENTS.md prefers
wrapping, and here the plan's own layout already agrees.)

**Cross-linking musl from macOS needs `rust-lld`.** Apple's `ld` cannot emit
ELF, and the failure is a wall of "unknown options". The packager points rustc
at the `rust-lld` that ships in the toolchain, discovering the host triple from
`rustc -vV` — `platform.machine()` says `arm64` where Rust's triple says
`aarch64`, close enough to look right and wrong enough to miss the file. On an
x86 Linux host no override is used at all.

## Compile target

`x86-64-v3`, from the [M0 CPU probe](../../research/measurements/morpheus-rs-cpu-probe.md):
the floor across sampled one-core Linux containers, which included an AMD
generation without AVX-512. Set per-target in `.cargo/config.toml` so it
applies to the Linux builds and not to arm64 macOS development. Any
hand-written SIMD path must still detect features at runtime — the sandbox
cannot be probed, so the target is an informed guess about a machine nobody
has measured.

## Panic policy

`panic = "unwind"`, and the decision runs inside `catch_unwind`. The judge
forfeits a game on a crash or an early exit but charges one fault out of fifty
for a bad reply (RULES.md §08), so a panicking turn should cost a fault, not
the match. `panic = "abort"` would be the reflexive release-profile choice and
is the wrong trade here.

## Verification

```bash
cargo test --release --manifest-path bots/morpheus-rs/Cargo.toml
python bots/morpheus-rs/tools/package_submission.py --force --gate
modal run scripts/morpheus_rs_modal_submission_smoke.py
```

`--gate` extracts the archive, runs its own `build.sh`, and plays the AGENTS.md
verification gate with the *submitted* bytes rather than the repo launcher —
a different launcher, a different directory shape and a build step, none of
which a two-frame script reaches. It costs minutes, so it is opt-in, and it is
required before a submission.

It runs `build.sh` and then **deletes it** before invoking `matchup.py`. That
models the end of intake — the judge builds once and afterwards only spawns
`run.sh` — and it is also the only way the gate can run: `build_agent`'s
unguarded `build.relative_to(REPO_ROOT)` raises for every path outside
`competition-module/`, so a submission-shaped directory crashes the gate
wherever it is extracted. What goes untested is `build_agent`, which the judge
does not have.

The last command extracts both zips inside a one-core Linux x86 container with
`block_network=True` and the sandbox's own toolchain, builds them, runs their
selfchecks five times each, and speaks the protocol to each. Results:
[morpheus-rs-sandbox-smoke.md](../../research/measurements/morpheus-rs-sandbox-smoke.md).
It is a proxy, not the judge — necessary, not sufficient, which is the whole
reason the fallback variant exists.
