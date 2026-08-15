# morpheus-joe

Morpheus's search stack over **joe's frozen network** — forward pass and
weights, unretrained. The plan, the gates and the kill criteria are
[joe-net-plan.md](../morpheus-rs/joe-net-plan.md); this page is what exists on
disk and how to run it.

Status: **N1 done 2026-08-16. The net is loaded and not consulted.** N2 (the
observation bridge) has not started.

## What it is right now

`bots/morpheus-joe/` is a fork of `bots/morpheus-rs/`, which the port never
edits — the N6 round needs current morpheus-rs, this candidate, and joe-rs as
three arms in **one** rating round, and a baseline that exists only in git
history cannot be an arm. The fork doubles as the rollback point: undoing all
of this is `git rm -r bots/morpheus-joe/`.

At N1 the seat resolves its artifact, verifies the digest, schema-checks it,
loads 8,556,250 parameters, warms one forward — and then decides through the
network-free `ShapedUniformEvaluator`. It plays legal, finishing games and it
plays them badly. **Any rating from this build is meaningless**, and the bot is
deliberately not registered in `data/bot_versions/` for that reason; the first
registration belongs to the phase that first measures something.

Measured on the dev host (arm64 macOS): load 131 ms, warmup 15 ms, first
decision 0.3 ms, and a gate match against `cm_expander` that truncated at turn
1200 with two castles built.

## The crate layout, and the one thing that is unusual about it

```
crates/joenet/   joe's files, copied byte for byte and never edited
crates/core/     morpheus's search, belief, tactics, runtime, parity
crates/bot/      the stdio seat, the artifact loader, selfcheck, bench
```

`crates/joenet/src/` is a **byte-identical copy** of `bots/joe-rs/src/` —
eleven files: `nn/{net,gemm,safetensors,mod}.rs`, `board/{obs,action,mod}.rs`,
`io/{wire,json,mod}.rs`, and `xla_math.rs`. Only `lib.rs` is written here.

That identity is the fork's entire parity argument for the forward pass. joe-rs
already carries a JAX-oracle corpus proving its forward matches joe to pinned
relative bounds; equal files plus a green joe-rs corpus means this bot's
forward pass is joe's forward pass, with no second corpus to build.
`tests/test_joe_source_fanout.py` asserts the equality and names the file when
it breaks.

**So the copies may not be improved here.** A fix lands in joe-rs, is proved
against joe-rs's corpus, and is then re-copied down to both forks. Batching the
leaf forwards is the concrete case, and the plan scopes it as upstream work for
exactly this reason.

The reason it is a separate crate rather than files dropped into
`crates/core/src/` is that byte-identity includes the imports: `board/obs.rs`
opens with `use crate::io::wire::{Observation, ...}` and joe's `Observation`
carries `i32` grids where morpheus's carries `u8`. A crate boundary gives joe's
files joe's `crate::` root and leaves morpheus's alone. The cost is one
widening adapter at the seam, which N2 writes.

## The weights

They come from **joe-rs**, one hop down the chain `joe → joe-rs → {unclejoe,
morpheus-joe}`, never from joe's `.eqx` directly. The joe-rs / morpheus-joe
contrast only isolates the search stack if both arms run byte-identical
weights, not merely weights from the same checkpoint.

```bash
python scripts/joe_artifact_fanout.py --check
```

`--check` reports and writes nothing; without it, every stale downstream is
synced. `tests/test_joe_source_fanout.py` runs the same comparison in the
default suite, which is the part that actually fires. A sync forks this bot's
content hash and **voids any rating contrast that spans it**.

## What N1 deleted, and where it went

| gone from the fork | why | still lives in |
| --- | --- | --- |
| `nn/network.rs`, `nn/tensor.rs`, `nn/inference.rs`, `nn/{gemm,safetensors}.rs` | morpheus's 249,316-parameter CNN, its 49-plane input contract, and its loader | `bots/morpheus-rs/` |
| `belief/summary.rs` | `build_tensor` was its only consumer | `bots/morpheus-rs/` |
| the policy-proposal path in `belief/proposal.rs` | joe's net would need a hypothesized enemy `AugState` that nothing reconstructs, and the turn affords 3–5 forwards in total | — (structural) |
| `use_policy_proposal`, `max_proposal_batch` | the branch they selected is gone; a knob nobody can turn on is a knob nobody can re-qualify | — |
| parity surfaces `tensor`, `net`, `prior`, `summary` | the Torch oracle went with the CNN | `bots/morpheus-rs/` |
| parity surface `decide` | **loses its oracle outright** — no Python program plays this combination | — (accepted cost) |
| 29 mutations, `tools/{convert_artifact,bench_inference}.py`, `tools/spikes/` | they named the deleted files | `bots/morpheus-rs/` |

`legal_normalized_policy` survives verbatim in `nn/head.rs`: it is already
written against a 3,970-long `f32` logit vector, which is what N3's remap
produces, and it is oracle-checked code whose oracle is now retired.

Twenty-eight parity surfaces survive, all against Python morpheus, none
involving a network. They run in 6.3 s against the old 11.4 s — the three
`torch.jit.load` calls were most of the difference.

## Running it

```bash
cargo build --release --manifest-path bots/morpheus-joe/Cargo.toml
```

```bash
./bots/morpheus-joe/target/release/morpheus-joe selfcheck
```

```bash
PYTHON=$PWD/.venv/bin/python .venv/bin/python competition-module/competition/matchup.py bots/morpheus-joe/run.sh bots/cm_expander/run.sh --mode competition --seed 0
```

```bash
.venv/bin/python -m pytest -m morpheus bots/morpheus-joe/tests
```

The absolute `PYTHON` in the matchup is load-bearing: a relative path silently
`BrokenPipe`s the Python seat.

## What is knowingly wrong, waiting for a later phase

- **`deployment.json` still describes the CNN.** Every per-component cost in it
  is morpheus's, wrong for this binary by orders of magnitude. N4 rewrites it
  from N0's measurements, and K0 attaches four non-optional changes to that
  rewrite — `pending_leaf_batch` 4 → 1 above all, because at 4 the controller
  admitted **zero** simulations on every turn of 8,166.
- **`COST_COMPONENTS` still lists `belief_tensor` and `enemy_prior_batch`.**
  That list is a wire format shared with the `runtime` parity surface and the
  trace schema, so retiring a component is a change with its own gate.
- **The network has no mutation coverage** until N3 plants the four mutations
  for the remap, the pass collapse, the bin→scalar dot and the value sign.
- **`resident_memory_target_mb` is a leftover** (Q12), trivially safe against
  the 2 GB cap and still wrong.
