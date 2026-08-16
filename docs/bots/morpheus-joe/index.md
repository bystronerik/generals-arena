# morpheus-joe

Morpheus's search stack over **joe's frozen network** — forward pass and
weights, unretrained. The plan, the gates and the kill criteria are
[joe-net-plan.md](../morpheus-rs/joe-net-plan.md); this page is what exists on
disk and how to run it.

Status: **N2 done 2026-08-16. The net decides; the search does not.** N3 (the
channel remap, the value decode, and the evaluator that hands both to the
search) has not started.

## What it is right now

`bots/morpheus-joe/` is a fork of `bots/morpheus-rs/`, which the port never
edits — the N6 round needs current morpheus-rs, this candidate, and joe-rs as
three arms in **one** rating round, and a baseline that exists only in git
history cannot be an arm. The fork doubles as the rollback point: undoing all
of this is `git rm -r bots/morpheus-joe/`.

At N2 the seat resolves its artifact, verifies the digest, schema-checks it,
loads 8,556,250 parameters, warms one forward — and then plays **joe's argmax**
every turn through the [observation bridge](bridge.md). `RuntimeController` is
constructed and never called. That is the phase, not an oversight: with no
search in the way, the bot's replies must be byte-equal to joe's over a whole
recorded game, which localizes a bridge bug before N3 can hide one.

So a rating from this build would measure **joe-rs with a slower launcher**,
and the bot stays deliberately unregistered in `data/bot_versions/`; the first
registration belongs to the phase that first measures something.

Measured on the dev host (arm64 macOS) against joe's **step-23500**
checkpoint: load 127 ms, warmup 15 ms, a decision in 15.5 ms — which is the
forward and almost nothing else — and a gate match against `cm_expander` won on
turn 304 by capturing the general.

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
widening adapter at the seam, and N2 wrote it: it is a widening loop and
nothing else, in [`nn/bridge.rs`](bridge.md).

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

**Twenty-six** parity surfaces survive, all against Python morpheus, none
involving a network. They run in 6.3 s against the old 11.4 s — the three
`torch.jit.load` calls were most of the difference. (N1's page and code
comments said twenty-eight, down from thirty-three; N2 counted the dispatch
tables and it is twenty-six, down from thirty-one. Corrected in the code.)

## What N2 added

| new | what it is |
| --- | --- |
| `nn/bridge.rs` | morpheus's frame into joe's `AugState` — [bridge.md](bridge.md) |
| parity surface `sequence` | the twenty-seventh, and the first checked against **joe's** recorded corpus rather than Python morpheus |
| `tests/test_morpheus_joe_bridge.py` | that surface over three whole games: tensor digests, final state, temporal input, and Q10 |
| `tests/test_morpheus_joe_wire_replay.py` | the whole binary against joe's recorded replies |
| 3 mutations under `nn/bridge.rs` | 3/3 caught, in the fork's first full pass: 118/140, no unexplained survivor |

N2 also fixed two things N1 left. `morpheus_joe_parity_cases.py`'s CLI default
`--kinds` list still named the five retired kinds, so a bare run died on
`unknown kind 'tensor'` — and that is the run `tools/mutation_check.py` makes
for every mutation mapped to "all surfaces", so the mutation gate was red for
anything outside the narrow map. And `crates/core/Cargo.toml` gained the
`joenet` path dependency, which is where the bridge lives.

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
- **The network itself has no mutation coverage** until N3 plants the four
  mutations for the remap, the pass collapse, the bin→scalar dot and the value
  sign. The bridge that feeds it has three, all caught.
- **The search does not run.** The controller, the belief filter and the
  tactics layer are all constructed and none of them is consulted; the reply is
  joe's argmax. N3 is what turns them back on.
- **`resident_memory_target_mb` is a leftover** (Q12), trivially safe against
  the 2 GB cap and still wrong.
