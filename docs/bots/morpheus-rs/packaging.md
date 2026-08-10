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
  crates/core/src/      the ported bot, one directory per layer:
    support/ io/          rng and sha256; the wire protocol and JSON
    board/                state, transition, action, observe, memory, hashing
    nn/                   the 49-plane contract, the graph, the kernels
    belief/               the particle filter and its recovery paths
    tactics/              the play mask, the planners, the shaping, the rules
    search/               the tree, the regret matrices, the evaluators
    runtime/              the deadline, plus deployment.json and telemetry
    parity/               the harness half; never plays
  crates/bot/src/       main: read frames, decide, reply; `parity` and `bench`
  artifact/             safetensors weights + manifest; inside the content hash
  deployment.json       the coupled knobs; inside the content hash (M5)
  run.sh                repo launcher — builds when stale, then execs
  tools/                dev tooling; outside the content hash
    submission/build.sh the build.sh copied into the submission zip
    package_submission.py
    minify/             dev-time crate: strips comments on the way into the zip
    vendor_probe.py     proves the offline build with real vendored crates
    dependency_budget.py what a crate would cost in the zip
    capture_morpheus.py the M0 capture module
    convert_artifact.py TorchScript -> safetensors, reproducibly
    bench_inference.py  the M3 shoot-out, local half
    spikes/             candle and tract crates kept as evidence, never linked
  tests/                pytest parity slice + the selfcheck guard; unhashed
  target/  vendor/      build output and vendored crates; gitignored, unhashed
```

The zip additionally carries a generated `run.sh`, a generated `build.sh`, and
`SUBMISSION.json` — the bot id, content hash, git commit and weights digest of
the program inside it. The judge never reads that file; a human comparing a
rated result on generals.bot against a row in `data/bot_versions/morpheus-rs.json`
does, and without it "which program is playing up there" is answerable only by
rebuilding and hoping. It deliberately carries no timestamp, because the zip
is byte-reproducible from the sources and a clock would end that.

## Toolchain

`rustc`/`cargo` **1.97.1**, matching the sandbox's 1.97 stable. Installed with
rustup; `rust-toolchain.toml` pins the channel and carries the
`x86_64-unknown-linux-musl` target, which the retired static fallback needed
and nothing builds today.

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

## One archive, named for the program in it

```bash
python bots/morpheus-rs/tools/package_submission.py --force
```

Out comes `data/bundles/morpheus-rs-<content_hash>.zip`: sources, `Cargo.lock`,
`vendor/`, and a `build.sh` running `cargo build --release --offline --locked`.
The judge compiles it at intake. `vendor/` is currently **empty**, because the
crate has no dependencies — which is why the probe below exists. The `.rs`
members are minified on the way in; see below.

The name comes from `arena.bundle.default_output_path`, the same call the
Python bundles use, so a Rust bundle and a Python one are the same kind of
object on disk and an archive says on its face which rated program it holds.
Two builds of different code cannot land on one path.

**The static-binary variant is gone (M8).** It was R4's fallback (§9) for a
sandbox that cannot build the vendored tree — a prebuilt
`x86_64-unknown-linux-musl` binary with `build.sh` reduced to a no-op — and the
plan asked for it to be built every run "so the fallback stays tested rather
than theoretical". Tested it was; qualified it never was. Measured on one x86
core at M8, its 26 warmup forwards land within 2% of the vendored build — same
target, same FMA, same kernels — while one decision costs **1.75× to 2.2×**
more and the artifact load ~1.9×, which is musl's allocator against glibc's.
It therefore never inherited M7's `p99.9 = 140 ms, zero moves over 150`, so
falling back to it meant shipping an unmeasured bot; and keeping it alive cost
a cross-linker path, a second `build.sh`, and a smoke test the packaging host
could not run. **If R4 ever fires, the fallback must be rebuilt from git
history and qualified before it plays** — that is a real cost of this decision,
recorded rather than smoothed over. It has not fired: the vendored-source
archive built inside the judge's own image on the first submission
(2026-08-10), which is the evidence the removal was waiting on. Figures:
[morpheus-rs-m8-submission.md](../../research/measurements/morpheus-rs-m8-submission.md).

The archive is built to a scratch path and moved into place only on success.
Writing straight to `data/bundles/` leaves a **partial archive** when a member
check fails mid-zip, and nothing downstream can tell one from a finished
bundle: an aborted build once left a four-file zip with no weights, which the
Modal smoke then shipped to an x86 container. The `selfcheck` below is what
caught it.

**The archive carries `artifact/` and `deployment.json`, both required.** That
is a correction, not a description: until M5 the packaging shipped only the
binary and its launchers, which was right for the M0.5 pass bot, quietly wrong
once M3 gave the bot weights to load, and wrong twice over once M5 gave it a
configuration to read. ~~A missing `deployment.json` is the nastier of the two
— the artifact's absence fails loudly at load~~ — **wrong, and M8 measured it.**
Neither fails loudly. The missing config falls back to the Part 07 placeholders
and plays a *different bot* (four times the particles, twice the search depth, a
125 ms deadline against 140); the missing artifact degrades the seat to a legal
skip on every turn. Both answer the protocol perfectly, and the difference is
invisible in a match log — see the next section. The packager now refuses to
build the archive without both, `try_load_deployment` says on stderr when it
falls back, and the lookup tries the binary's own directory before
`target/../..` so a flat layout would also resolve.

It is audited through `arena.bundle.check_limits` — the same code that guards
the Python bundles, so the two cannot drift — then extracted to a scratch
directory with no repo around it, built, and driven with a scripted frame.

## What the submission does not carry

The archive ships source, and this crate's comments are not
incidental: the FMA bug and its 277 ms measurement, the BLAS reduction the
oracle cannot pin, NumPy's unstable `argsort`, the reasoning behind every
tactical rule. `arena/bundle.py` already decided that question for the Python
bots — a submission carries logic, not strategy notes — and the Rust side now
follows the same rule, with the same discipline: **the rewrite happens on the
way into the zip only, and the repo files are never touched.**

`tools/minify/` is a dev-time crate over
[`rustminify`](https://docs.rs/rustminify) 0.2.0, pinned exactly, reading one
`.rs` on stdin and writing the stripped equivalent on stdout. It is a parse and
a re-print, not a text rewrite: ordinary `//` comments never reach the AST, so
they vanish for free, and doc comments survive as `#[doc]` attributes that
`remove_docs` takes. A regex over the same problem eats code the first time
`//` appears inside a string literal. The output is re-parsed before it is
returned, and a failure at any step is fatal — falling back to the original
text would leave the packager reporting a minified bundle it did not build.

Measured: **669,589 → 345,746 bytes** of Rust, and the smoke test compiles and
plays what came out, so "does the minified source still build offline" is
answered on every packaging run rather than at intake.

Two things it deliberately does not do. It does not rename anything: Python's
bundler renames locals because Python ships names at runtime, and a compiled
binary does not. And it does not touch string literals, so operational messages
— including the selfcheck's own explanation of the FMA trap — are still in
there, as they must be to be printed.

**The launchers get the same treatment, by a much simpler rule.** `run.sh`,
`build.sh` and `.cargo/config.toml` carry as much reasoning as the Rust does —
why `cd` before `cargo build`, what a missing FMA costs, why intake fails
loudly — so `strip_line_comments` drops whole-line `#` comments from all three
on the way into the zip. The shebang stays. A *trailing* `# …` after code is
deliberately left alone: a line-based rule cannot tell a comment from a `#`
inside a string or from `${var#prefix}`, and a launcher that stops working is a
forfeit, not a smaller zip. `tests/test_packaging.py` pins that boundary, and
the packaging smoke runs the stripped scripts for real.

The other half is the binary. `[profile.release]` sets `strip = "symbols"`,
because an unstripped executable carries `morpheus_core::tactics::…` path by
path. It also took **1,479,416 → 1,298,656
bytes** off it. Nothing at match time needs those names: panics are caught and
reported by message, and the judge does not hand back backtraces.

`--no-minify` ships the sources verbatim, for when a judge traceback has to be
read against real line numbers. `SUBMISSION.json` records which way the archive
was built.

Note the version pin is exact (`=0.2.0`) and the helper depends on **syn 1.x**,
not 2 — `rustminify::remove_docs` takes a 1.0 `syn::File`, and cargo will link
both majors into one graph without complaint, so the failure reads as a type
mismatch between two types spelled identically.

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

**`build.sh` runs it**, so a broken submission is rejected at
intake rather than rated. That is the whole trade: a rejected submission costs a
resubmission, a degraded one costs every rated game it plays. The packager
parses its output, `bots/morpheus-rs/tests/test_selfcheck.py` proves it still
notices a bot with its weights removed, and the Modal smoke requires it — a
checker nobody checks is exactly the thing it was written to prevent.

`warmup_ms` doubles as a per-forward probe: it is 26 forwards, so ~4.7 ms each
on the M3 Pro against M7's ~5.1 ms on one x86 core, and a build that lost its
FMA would read fifty times that.

## The offline build is proven by a probe, not by the shipping zip

The shipped archive has no dependencies. That makes its
`cargo build --offline` cheap to pass and nearly meaningless: with nothing to
resolve, `--offline` cannot fail. It never exercises `.cargo/config.toml`
source replacement, a transitive graph, or a build script running at intake —
and M3's inference crate brings ninety crates of exactly that.

`tools/vendor_probe.py` closes the gap without putting a dependency in the
shipped binary. It builds a scratch crate that depends on `sha2` — chosen for
its shape: a transitive chain (`digest` → `block-buffer` → `generic-array` →
`typenum`) and a build script via the `version_check` build-dependency, which
is the failure mode most likely to be missed. It vendors, packages in exactly
the submission shape, builds offline locally, and rides the Modal smoke
alongside the real archive.

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

The last command extracts the archive inside a one-core Linux x86 container
with `block_network=True` and the sandbox's own toolchain, builds it, runs its
selfcheck five times, and speaks the protocol to it. Results:
[morpheus-rs-sandbox-smoke.md](../../research/measurements/morpheus-rs-sandbox-smoke.md).
It is a proxy, not the judge — necessary, not sufficient, and since M8 there is
no built fallback standing behind it.
