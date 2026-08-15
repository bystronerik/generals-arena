# unclejoe fork plan

Plan for creating `bots/unclejoe/` as a faithful fork of `bots/joe-rs/` (the
dependency-free Rust port, compiled at intake — not the Python `bots/joe/`).
Status: **plan only** — no code exists yet.

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
`docs/research/strategies/unclejoe.md`. Its content is the design in
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
  `test_wire_replay.py`: pipe the shared `data/joe/joe-rs-parity/` corpus
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
