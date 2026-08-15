# unclejoe fork plan

Plan for creating `bots/unclejoe/` as a faithful fork of `bots/joe-rs/` (the
dependency-free Rust port, compiled at intake — not the Python `bots/joe/`).
Status: **built, U1 proven** (2026-08-15) — see [§7](#7-what-shipped-2026-08-15)
for the results and the two places the code differs from this plan.

This document covers the fork itself: the copy, the rename, the bot's own
rating identity, and the proof that the fork is behavior-identical to joe-rs
(milestone U1). What the fork is *for* — the tactics layer that overrides the
NN move when a bounded exact search proves a kill or a defense — is the
second document, [tactics-plan.md](tactics-plan.md), which builds on a
finished U1.

Ground rules honored throughout: `bots/joe/`, `bots/joe-rs/`, and
`competition-module/` internals are never edited (the only joe-rs-adjacent
change is a docs checklist line in [export.md](../joe-rs/export.md)), and
the crate stays dependency-free so intake still compiles it from source.

## 1. Prerequisite: strategy spec first

A spec is required. The repo workflow (AGENTS.md subagent roles) has
bot-author work "from a spec under `docs/research/strategies/`", and every
prior bot follows it. Step 1 is therefore
[`docs/research/strategies/unclejoe.md`](../../research/strategies/unclejoe.md)
— written, 2026-08-15. Its content is the design in
[tactics-plan.md](tactics-plan.md) — concept, trigger definitions, search
model and caps, latency budget, evaluation plan, and non-goals — because the
tactics layer is the bot's reason to exist; the fork below is the substrate.

## 2. Copy strategy

Start from `cp -R bots/joe-rs bots/unclejoe` minus `target/`, then adapt:

| Item | Action |
| --- | --- |
| `src/` (io/board/nn/xla_math) | **Copy verbatim.** No edits except `main.rs` (below). The `mod io` / `use std::io as stdio` alias pattern carries over unchanged. |
| `src/main.rs` | **Adapt:** rename env var `JOE_RS_ARTIFACT` → `UNCLEJOE_ARTIFACT`, stderr tags `[joe-rs]` → `[unclejoe]`, drop the `parity` subcommand. Keep `selfcheck` and `bench`. (The tactics hook comes later — [tactics-plan.md](tactics-plan.md) §2.) |
| `src/parity.rs` | **Drop.** The parity harness (JAX-oracle `.npz` corpus, tier-2 pins, mutation kills) stays owned by joe-rs; duplicating it doubles the re-export checklist for zero information since `nn/` is byte-identical. unclejoe's sequence-level guard is the wire-replay test (§4). |
| `Cargo.toml` | **Adapt:** `name = "unclejoe"`. Keep `[dependencies]` empty (hard requirement: intake compiles one crate from source), keep the release profile; the `[profile.mutation]` section goes with `mutation_check.py`. |
| `Cargo.lock` | Copy (trivial with zero deps) — or regenerate; either is fine. |
| `rust-toolchain.toml`, `.cargo/config.toml` | **Copy verbatim** (1.97.1 pin, `target-cpu=x86-64-v3`). |
| `artifact/` | **Byte-copy** `model.safetensors` + `manifest.json` from joe-rs. Do not edit the manifest — provenance lives in docs and the version registry, and a byte-identical manifest keeps the `safetensors_sha256` pin verifiable against joe-rs's. |
| `run.sh` | **Adapt:** binary path `target/release/unclejoe`, `UNCLEJOE_ARTIFACT` export, same source-stamp rebuild logic. **Rewrite all comments to contain no path strings** — the closure scan pulls any path mentioned in run.sh prose into the rated closure (joe-rs's run.sh drags its port-plan doc in this way). Refer to docs by prose name only, never with path syntax. |

Per-tool decisions (`bots/joe-rs/tools/` — six files, all naming joe-rs
source paths):

| Tool | Decision |
| --- | --- |
| `package_submission.py` | **Adapt.** Change `bot_id`/`binary` to `unclejoe`, `BOT_DIR`, the generated bundle `run.sh` (env var + binary name), and the smoke-frame check (the fog-everywhere smoke position triggers no tactic, so the expected reply class is unchanged). It keeps using the shared `arena/rust_bundle.py`. |
| `convert_artifact.py` | **Drop.** unclejoe never converts from `ema.eqx`; it derives from joe-rs's already-converted artifact. |
| `capture_fixtures.py`, `make_synthetic_long.py`, `make_smoke_fixture.py` | **Drop.** Corpus production stays with joe-rs; unclejoe's wire-replay test reads the shared corpus at `data/joe/joe-rs-parity/` read-only. |
| `mutation_check.py` | **Drop.** Its mutation list pins exact `nn/net.rs` / `board/obs.rs` source lines and drives the joe-rs parity pytest; that harness proves the *shared* code once, in joe-rs. unclejoe's own code gets direct unit tests instead. |
| **New:** `tools/sync_artifact.py` | Small script: byte-copy `bots/joe-rs/artifact/*` into `bots/unclejoe/artifact/`, verify `safetensors_sha256` from the manifest against the copied file, refuse on mismatch. This is the single deliberate resync knob (§3). |

## 3. Identity and provenance

- **Own rating identity.** The content hash covers `src/`, `Cargo.*`,
  `run.sh`, and `artifact/model.safetensors` (with `target/` skipped), so
  unclejoe registers as a new lineage in `data/bot_versions/` at its first
  rated round. That is correct and intended — it is a different program.
- **Exporter fan-out.** unclejoe becomes a second derived target of the joe
  artifact, one hop further removed (joe → joe-rs → unclejoe). Policy:
  **unclejoe tracks joe-rs**, not joe directly — the joe-rs vs unclejoe
  contrast must isolate the tactics layer, which requires identical weights.
  Concretely: append one step to the "After a joe re-export" checklist in
  [export.md](../joe-rs/export.md) —
  `python bots/unclejoe/tools/sync_artifact.py && cargo build --release
  --manifest-path bots/unclejoe/Cargo.toml` — so a joe re-export cannot
  silently leave unclejoe on stale weights. `sync_artifact.py`'s sha
  verification makes a *partial* sync (manifest copied, safetensors not)
  fail loudly. A resync forks unclejoe's content hash, same as it forks
  joe-rs's — expected, and the registry records it.
- **run.sh closure hygiene** as in §2: no path strings in comments, so
  unclejoe's closure is exactly its code + artifact.

## 4. Tests

- **Rust (`cargo test`, zero cost to the Python budget):** the copied
  modules' unit tests come along with the copy and must pass unchanged.
- **Python (`bots/unclejoe/tests/`, outside the content hash):** one adapted
  wire-replay test (`test_unclejoe_wire_replay.py` — §7 says why the basename
  cannot repeat joe-rs's): pipe the shared `data/joe/joe-rs-parity/` corpus
  logs through `unclejoe` and require reply-for-reply equality with Python
  joe's recordings — proving the copied NN path is untouched. Mark it with a
  new `unclejoe` marker added to `pytest.ini`'s `markers` and to the
  `addopts` exclusion, exactly like `joe`. Since `testpaths` already
  includes `bots`, the default suite collects-and-deselects it at ~zero
  cost: **the 15 s budget is untouched**. (Once the tactics layer exists,
  this test runs with the `UNCLEJOE_TACTICS=0` kill-switch —
  [tactics-plan.md](tactics-plan.md) §5.)

## 5. Verification

The gate (AGENTS.md, with the known environment traps: absolute `PYTHON`,
cargo on PATH):

```bash
export PATH="$HOME/.cargo/bin:$PATH" && \
PYTHON=$PWD/.venv/bin/python python competition-module/competition/matchup.py \
  bots/unclejoe/run.sh bots/joe-rs/run.sh --mode competition --seed 0
```

Must reach a normal end; store the game under `data/games/<round>/` before
any refit. Latency sanity: `unclejoe bench` over the synthetic-long log must
land at joe-rs's measured numbers ([latency.md](../joe-rs/latency.md) —
p99 23.7 ms on the Modal 1-core x86-64-v3 proxy), since the fork changes no
per-move code.

## 6. Done criteria (milestone U1)

The fork is complete when, in one working tree:

1. `cargo test` passes in `bots/unclejoe/`.
2. The wire-replay test passes: reply-for-reply equality with the recorded
   corpus.
3. The matchup gate above finishes a competition match normally.
4. `unclejoe bench` ≈ joe-rs's numbers.

Only then does the tactics work start —
[tactics-plan.md](tactics-plan.md).

## 7. What shipped (2026-08-15)

All four done criteria met, in one tree, on dev arm64 macOS:

1. `cargo test` — 18 passed, the copied modules' tests unchanged.
2. Wire replay — 14 games, 7,502 turns, every reply equal to Python joe's
   recording.
3. Gate — `unclejoe` vs `joe-rs`, competition mode, seed 0: normal end,
   general captured on turn 385. No game was stored and no rating was fitted;
   storing and the contrast belong to
   [tactics-plan.md](tactics-plan.md) §6.
4. `bench` over `synthetic-long.in.log` (2,400 turns), same host, back to
   back: unclejoe p50 22.18 ms / p99 23.49 ms, joe-rs p50 22.35 ms /
   p99 23.41 ms — equal inside noise, and matching the dev-arm64 row of
   [latency.md](../joe-rs/latency.md). The x86 proxy re-run belongs to U4.

Two deviations from the plan above, both forced:

- **The test file is `tests/test_unclejoe_wire_replay.py`, not
  `test_wire_replay.py`.** pytest imports test modules by basename when there
  is no `__init__.py`, so a second `test_wire_replay.py` collides with
  joe-rs's and fails collection of the *whole* suite, marker or no marker.
  The unique basename is the fix that does not edit joe-rs.
- **`mod nn;` in `main.rs` carries `#[allow(dead_code)]`.** joe-rs's parity
  harness read `ForwardOut::value_bins`, and the harness stayed behind, so the
  field is now unused. The allow is scoped to the copied module — anything the
  fork adds keeps its warning — and it is what lets `nn/` stay byte-identical
  to joe-rs's.

Also shipped, and implied by the packager rather than listed in §2:
`tools/submission/build.sh`, adapted from joe-rs's (binary name only). The
shared packager reads it by path, so a fork without it cannot package.

Closure check: `bot_source_closure` returns 19 files for unclejoe — its own
sources, `Cargo.*`, `run.sh`, `rust-toolchain.toml`, `.cargo/config.toml`, and
the two artifact members. Nothing outside the bot directory, as intended.

Fork check: `diff -r bots/joe-rs/src bots/unclejoe/src` reports `main.rs` and
the dropped `parity.rs`, and nothing else. The weights are byte-identical —
both manifests pin `safetensors_sha256 919b593ce7c8`. The identities are
separate, as they must be: `joe-rs@490b968aba4b`, `unclejoe@c1841f68ca4b`.
Neither bot is registered in `data/bot_versions/` for this milestone; that
happens at the first rated round.

Packaging works from the adapted spec: `package_submission.py --force` builds
a 20-file, 31.8 MB zip, smokes the extracted bundle (two well-formed replies,
one of them a move), and `selfcheck` passes inside it — zero failures.
