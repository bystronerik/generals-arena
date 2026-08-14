# Joe-rs refactor — a flat module list into domain subdirectories

Status: **done (2026-08-15).** One commit. A **pure move**: files changed
place, code did not change meaning. The proof is that
`tools/mutation_check.py`'s nine `old`/`new` source snippets still match
byte-for-byte in their new files, and that the tally is 9/9 as before.

Modeled on [morpheus-rs's refactor plan](../morpheus-rs/refactor-plan.md),
which did the same thing to a crate five times the size. The conventions are
copied; the tree is not. What morpheus's plan did not predict and this one
hit anyway is recorded in [§6](#6-what-the-morpheus-plan-did-not-predict).

Scope: `bots/joe-rs/src`, 10 modules and 3,053 lines behind a flat `main.rs`.
Nothing here is large enough that "which file is this in" has stopped being a
useful question — joe's biggest file is `obs.rs` at 507 lines, where
morpheus's was `tactics.rs` at 2,560. **This refactor is about the layer
boundaries, not about file size**, which is why it splits nothing.

## 1. Proposed tree

**Decision: three module subdirectories, three files left at the crate root.**
The crate stays one binary crate; there is no workspace and no lib split.

```
bots/joe-rs/src/
  main.rs           487  crate root: mod list + the layering rule, Seat,
                         wire loop, bench, selfcheck, dispatch  (§4)
  xla_math.rs       135  the float-semantics floor
  parity.rs         319  the harness half; never plays

  io/                    the two things the process reads from outside itself
    mod.rs
    wire.rs         235
    json.rs         324

  board/                 frame -> planes -> action index; parity tolerance zero
    mod.rs
    obs.rs          507
    action.rs        97

  nn/                    weights in, masked logits and a value out
    mod.rs
    gemm.rs         204
    net.rs          496
    safetensors.rs  249
```

One line each for why the grouping is that grouping:

| group | why these files are one thing |
| --- | --- |
| `io/` | Format codecs with zero model and zero game knowledge — the judge's frames, and the two files on disk. Both ported from morpheus-rs, which wrote them for the same two inputs. |
| `board/` | Everything that speaks in board coordinates: the observation pipeline in, the action codec out. The layer whose parity tolerance is **zero**. |
| `nn/` | Everything defined by the trained artifact rather than by the frame: the graph, the kernel under it, the container the weights arrive in. The only non-zero parity tolerances live here. |

`board/` needs its `mod.rs` to say one thing out loud, and it does: **joe never
simulates the board.** There is no `transition.rs` here and there should not
be one — the engine runs the game and sends a fogged frame per turn. The name
is morpheus's, kept because these are sibling bots and a reader who knows one
tree should be able to read the other.

### Why three directories and not nine

morpheus core is ~18k lines in 67 files across 9 directories. joe-rs is 3k in
10. Copying the tree rather than the conventions would give `support/` and
`parity/` one file each — a `mod.rs` that organizes nothing, plus a level of
nesting. Rejected, along with:

- **`obs/` as a directory name.** `obs.rs` mirrors the Python sibling's
  `joe_obs.py` and is pinned by name six times in `mutation_check.py`. It
  cannot become `obs/mod.rs` without giving up the leaf name, which is a
  parity asset (§5). So the parent takes a different name.
- **`support/xla_math.rs`.** morpheus's `support/` earned its directory by
  holding two files that share a *reason* — the dependency budget declined
  `rand` and `sha2`. joe has exactly one such file.
- **`xla_math.rs` inside `board/`.** `nn/net.rs` and `parity.rs` both reach
  it, and it is f32 arithmetic, not a board concept.
- **Full uniformity — every module inside a directory.** morpheus is uniform
  because it has the mass for it. joe's three root files each have a reason a
  directory would hide: `main.rs` is the top, `xla_math.rs` the floor below
  every layer, `parity.rs` the harness beside them.
- **A crate split.** morpheus's §1 argues this at length and every argument
  transfers: `lto = "fat"` plus `codegen-units = 1` means a module move is
  codegen-neutral where a crate boundary is not, crate boundaries force `pub`
  where `pub(crate)` is honest, and the packager walks `src/` with `rglob` so
  either shape ships. joe adds one of its own: the whole point of the
  2026-08-15 dependency purge was that intake compiles **one crate**
  ([packaging.md §11](packaging.md#11-back-to-a-source-build-2026-08-15)), and
  a second crate is the wrong direction to walk on the day after.

### mod.rs, not `foo.rs` beside `foo/`

Both are legal in edition 2021; pick one and be uniform. This copies
morpheus's choice so that every path in `mutation_check.py`, the docs and the
content-hash file list is `src/<group>/<leaf>.rs` with no exceptions. A coin
flip; what matters is that it is not mixed.

### The layering rule

Stated in `main.rs` because there is no `lib.rs` to state it in:

```text
xla_math  io  ->  board  ->  nn  ->  parity   main
```

A module may name the ones above it and must not name the ones below it. This
is the true edge set, not an aspiration — there are only 13 edges in the whole
crate and every one of them respects it. Nothing enforces it but review.

## 2. Old → new mapping

All 10 modules. Every row is a `git mv` plus import fixups; **nothing is
split, renamed, or merged.**

| current | lines | destination | note |
| --- | ---: | --- | --- |
| `main.rs` | 487 | `main.rs` | stays; `mod` list rewritten, layering rule added |
| `wire.rs` | 235 | `io/wire.rs` | move |
| `json.rs` | 324 | `io/json.rs` | move |
| `obs.rs` | 507 | `board/obs.rs` | move |
| `action.rs` | 97 | `board/action.rs` | move |
| `gemm.rs` | 204 | `nn/gemm.rs` | move |
| `net.rs` | 496 | `nn/net.rs` | move |
| `safetensors.rs` | 249 | `nn/safetensors.rs` | move |
| `xla_math.rs` | 135 | `xla_math.rs` | stays flat |
| `parity.rs` | 319 | `parity.rs` | stays flat |

The import sweep in full — 13 `use` lines and 3 inline `crate::` paths:

- `board/action.rs` → `crate::board::obs`
- `board/obs.rs` → `crate::io::wire`; its `crate::xla_math` unchanged
- `nn/net.rs` → `crate::board::obs`, `crate::io::json`, `crate::nn::gemm`,
  `crate::nn::safetensors`; its inline `crate::xla_math::RECIP_50` unchanged
- `nn/safetensors.rs` → `crate::io::json`
- `parity.rs` → `crate::board::action`, `crate::board::obs`, `crate::io::wire`,
  `crate::nn::net`; its two inline `crate::xla_math` paths unchanged
- `main.rs` → the same four groups, plus `crate::io::json::parse`

Sibling modules inside one directory use the full `crate::<group>::<leaf>`
path rather than `use super::`, which is what morpheus does.

**No visibility widened, anywhere.** morpheus widened 73 items because it
split files, and a private helper that lands in a sibling file needs
`pub(super)`. Nothing here splits, so every item keeps the visibility it had.

## 3. What stays put, and why

**`parity.rs`, flat and whole at 319 lines.** morpheus made `parity/` a
directory so that "this never plays" is visible in the tree, and split it ten
ways because it was 1,755 lines with a `run()` match that needed a `Ctx`
struct threaded through it. joe's is one file with seven surfaces. A
`parity/mod.rs` holding all of it would rename the leaf for nothing.

**`xla_math.rs`, flat.** See §1. It is also the file most likely to grow if a
JAX upgrade changes another lowering, and a directory is one `mkdir` away on
the day that happens.

**Leaf names.** `obs.rs` mirrors `joe_obs.py`, `net.rs` the network module,
`wire.rs` the protocol. The parents changed; not one leaf did. morpheus's
plan allowed itself a single leaf rename (`particle_summary.rs` →
`belief/summary.rs`) and flagged it as droppable; this one has no candidate
worth the argument.

**Inline `#[cfg(test)] mod tests`.** They stay in the file whose private items
they reach. `mutation_check.py`'s `argmax-tie-break` kill depends on it — an
exact logit tie never occurs in a real frame, so only
`board::action::tests::argmax_is_first_max` can see the flip.

**`tools/` and `tests/`.** Both are outside the content hash by
`fingerprint._SKIP_DIRS`, and `tools/submission/build.sh` must not move beside
`run.sh` — `matchup.py::build_agent` runs any `build.sh` it finds there and
then crashes formatting its log line, which breaks the repo's own verification
gate.

**`Cargo.toml`'s profiles.** `lto = "fat"`, `codegen-units = 1`,
`strip = "symbols"`, `panic = "unwind"` and the `mutation` profile are
untouched. Only the comment naming source paths changed.

## 4. main.rs — flat and whole, 487 lines

The one place "pure move" could have turned into real surgery, so the decision
is stated rather than assumed. **It is not split.**

morpheus keeps its 498-line `main.rs` whole on the grounds that splitting a
composition root is how you get a composition root you have to search for. joe
reaches the same answer from a different premise: it is a **single binary
crate**, so `src/main.rs` is the crate root Cargo autodiscovers, it is where
the `mod` list and the layering rule have to live, and it cannot move — only
split. Three further reasons:

1. **`Seat` is the composition root itself.** It is the only thing in the
   crate that composes `board::obs` + `nn::net` + `board::action`. Moving it
   into a module means widening `Seat`, `Seat::new`, `Seat::act` and twelve
   fields from private to `pub(crate)` — visibility churn on the bot's hottest
   path, in a refactor whose entire claim is that nothing changed.
2. **The four bodies are four entry points, not four subjects.**
   `wire_main`, `bench_main`, `run_selfcheck` and `main` are each read
   alongside the dispatch match that reaches them.
3. **~130 of the 487 lines are doc comments** carrying the RULES.md §08
   forfeit policy and the reason `selfcheck` exists. The code is ~350 lines.

## 5. Blast radius outside `src/`

### `tools/mutation_check.py` — the largest single edit

Nine `file` fields, each a bare filename joined as `SRC / fname`. They become
paths: `"obs.rs"` ×6 → `"board/obs.rs"`, `"net.rs"` ×2 → `"nn/net.rs"`,
`"action.rs"` → `"board/action.rs"`. `SRC / "board/obs.rs"` resolves, so no
code changed.

**Every `old` and `new` source snippet stays byte-identical** — the
`net.rs` q_proj/k_proj pair, the five `obs.rs` patterns, the `action.rs`
comparison. That is the pure-move requirement cashing out: a move that
reordered arithmetic or tidied a line in passing would have broken a pattern,
and the tool would have said `pattern not found ... source drifted` and
exited non-zero. It did not. **9/9 killed**, the recorded tally.

morpheus's two rules for this file both apply and both were kept: surface sets
were not tightened, and pattern uniqueness needed no re-check because no two
files merged.

### The content hash, and the registry

**The rename changes the bot's content hash.**
`fingerprint.content_hash_for_dir` digests each source file's repo-relative
path followed by its content hash, so moving `src/obs.rs` to
`src/board/obs.rs` mints a new identity for a byte-identical program. There is
no way to avoid this and no reason to want one: a hash that ignored paths
would collide two different trees.

What follows:

1. **One lineage step**, registered after the parity corpus and the mutation
   check were green, with a note that this was a pure move.
2. **Existing rating data does not transfer.** Everything measured against
   `joe-rs@08e9f1118c22` — the A/B sanity round, the latency percentiles, the
   parity numbers — is attached to that hash. A round under
   `--strict-versions` after this compares a different one. The program is
   provably the same (the corpus and the 9/9 tally are the proof), so the
   honest handling is to say so here rather than to re-measure. **Do not
   straddle an open measurement with this refactor.**
3. **The submitted artifact is unaffected.** What generals.bot accepted is
   pinned by its registry entry and by `SUBMISSION.json` inside the uploaded
   zip. After this, `package_submission.py` on the working tree produces a
   differently named archive holding the same program; if that submission ever
   has to be rebuilt, it is rebuilt from the pinned commit.
4. **No other bot's hash moves.** Nothing outside `bots/joe-rs/` changed that
   the closure walk can see, and `_SHELL_REF_RE` matches only `.sh`/`.py`
   paths — so the `src/nn/gemm.rs` string in `tools/submission/build.sh` is
   prose, not an edge.

### Build and packaging — no code changes, four comments

- **`run.sh` needs no edit.** `find "$DIR/src" -type f -name '*.rs'` recurses,
  and `shasum` prints each path beside its digest, so the source stamp picks
  up subdirectories and moves correctly on the first build after the rename.
- **`tools/package_submission.py` assumes nothing about layout.**
  `arena/rust_bundle.py` walks `source_trees=("src",)` with `rglob("*")`;
  there is no file list and no flat glob. The archive goes from 10 sources to
  13. Two comments named `src/gemm.rs` / `src/net.rs` / `src/wire.rs`.
- **`tools/submission/build.sh`** — one comment, the FMA trap's
  `src/gemm.rs`. Its `cd`-before-`cargo` rule is untouched.
- **`scripts/joe_rs_modal_static_build.py` needs no edit.** It names the
  `src/` directory only, and `add_local_dir` is recursive.
- **`Cargo.toml`** — the `[dependencies]` comment names four source files.

### Tests and docs

- **`tests/parity_lib.py` needs no edit** — it names `src/parity.rs`, which
  stays flat. Same for `tools/capture_fixtures.py`'s `parity.rs::crc32`. The
  Python harness is coupled to the **integer stream and the surface names**,
  neither of which moved.
- **`tests/test_parity.py`** — one comment path. Its bare `gemm.rs` mentions
  beside the measured tolerances are leaf names in prose and stay.
- **`packaging.md`** — four line-anchored links, all re-anchored. Two were
  **already stale before this refactor**: `artifact_dir()` was anchored at
  `main.rs:43` when it lived at 61, and `main.rs:257` was said to dispatch the
  subcommands when 257 sat inside `bench_main`'s doc comment and the dispatch
  is `fn main`. `net.rs:142` pointed at the `Scratch` field list rather than
  at the schema check in `Net::load`. Plus three prose paths.
- **`export.md`** (`src/net.rs`), **`parity.md`** (`src/action.rs`) — one each.
- **`xla-semantics.md` needs no edit** — `src/xla_math.rs` did not move.
- **`port-plan.md` is not rewritten.** Its §2 layout block is a dated design
  record — it still lists candle for `net.rs`, which stopped being true on
  2026-08-15 — and rewriting paths inside it would falsify a history this repo
  keeps honest. It gets a pointer to this file instead.
- **`docs/research/measurements/joe-rs-mutation-check.json` is not rewritten.**
  It is the recorded output of a dated run and names the files as they were
  that day.

## 6. What the morpheus plan did not predict

- **`mod io;` collides with `use std::io::{self}` in a binary crate.** Both
  bind `io` in the crate root, which is E0255. morpheus never hit it: its `io`
  lives in `crates/core`, and its `main.rs` is in a different crate that does
  not declare the module. Fixed by aliasing the std import —
  `use std::io::{self as stdio, BufWriter, Write}` — and four call sites
  (`stdio::stdin`, `stdio::stdout`, `stdio::sink`). The alternative was
  renaming the directory to dodge a collision with the standard library, which
  is the wrong thing to optimize.
- **The `use` sweep was smaller than the doc-comment writing.** morpheus
  reports the sweeps as the whole cost of its move-only stages; at joe's size
  the 13 edges took minutes and the three `mod.rs` headers took longer. That
  is the right ratio — the headers are the part that makes the tree mean
  something.
- **Nothing needed `cargo fix`,** so morpheus's `--all-targets` trap (twice it
  pruned imports only a `#[cfg(test)]` module reached) never came up. The
  sweep was done by hand against a list of 16 known sites.

## 7. Risks and non-goals

This was a pure move. Out of scope and not done: no reordered floating-point
arithmetic, no `mul_add` touched, no visibility narrowing, no item renames, no
change to the parity surface names or the integer encodings, no new
dependencies, no threads, no crate split, no `Cargo.toml` profile edits.

Two risks worth naming, both retired by the gate rather than by argument:

1. **A move reorders arithmetic by accident** — the realistic vector is a
   function landing in a file whose `use` statements resolve a name
   differently. The corpus is bit-exact on every surface but `forward`, so a
   reordering shows up as a mismatch, not as drift. It did not.
2. **The optimizer notices anyway.** Module moves within one crate should be
   codegen-neutral, but morpheus's M3 has a track record of "should" here.
   `selfcheck` reports `decide_ms` for free; it read 22.8 ms after the move on
   the M3 Pro, against `bench` percentiles in [latency.md](latency.md).

Non-goals: this does not shrink the submission, speed anything up, change what
the bot plays, or touch the Python sibling `bots/joe/` in any way.
