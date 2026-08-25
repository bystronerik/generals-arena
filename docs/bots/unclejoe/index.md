# unclejoe — the X16 network at pure argmax

A Rust stdio bot, forked from `bots/joe-rs` on 2026-08-25: the same
dependency-free crate (in-house GEMM, safetensors/JSON readers, joe-net-v2
unpacker), playing a **different network at a different selection point**.

| | joe-rs | unclejoe |
| --- | --- | --- |
| network | M7F4 (depth 7, ff x4, 13.58M params) | **X16** (depth 16, ff x4, 29.55M params, 276 leaves) |
| weights source | derived from `bots/joe`'s artifact | **own export** from R2 run `joe-X16-gcp-20260825` |
| selection | Gumbel S1, T = 1 | **plain argmax, T = 0** |
| trail penalty (S4) | off (δ = 0) | off (δ = 0) |
| binary / bot_id | `joe-rs` | `unclejoe` |

The crate keeps joe-rs's Gumbel and trail code intact (minimal diff, future
syncs stay readable); only the parsed default temperature moved from 1 to 0
in `src/main.rs`. The `JOE_RS_*` env names are kept — `run.sh` sets them per
process, so two bots in one match cannot collide — and a positive
`JOE_RS_TEMPERATURE` re-enables the Gumbel draw for diagnostics.

## The argmax choice, stated plainly

Pure argmax at deploy is the **measured cause of joe's castle limit cycle**
([joe-argmax-limit-cycle](../../research/measurements/joe-argmax-limit-cycle.md)):
PPO sampled the policy during training, deployment replays its mode, and the
escape from a preference loop never had to be learned. joe-rs's Gumbel layer
(selection-plan S1) exists to fix exactly that. unclejoe ships the argmax
anyway, as a deliberate choice — the X16 net at its training distribution's
mode, with no temperature and no penalty compensating. If unclejoe circles a
castle, that is this choice showing; do not patch it with a knob — measure
it (`tools/osc_grid.py` is the diagnostic) and decide in a rated round.

## History of the name

A previous `bots/unclejoe` — a tactics fork of joe-rs carrying a **byte copy
of joe's weights** — was removed in `f22e252` on 2026-08-20 (its record:
[shadow.md](shadow.md)). This bot does not revive it: the new unclejoe has
its **own weights lineage**, so the artifact fan-out machinery
(`scripts/joe_artifact_fanout.py`, the staleness comparison) that a
weight-copying fork must restore does **not** apply here. Nothing copies
weights to or from this bot; its chain is its own export from R2
([export.md](export.md)). The bot_id is reused on purpose:
`data/bot_versions/unclejoe.json` appends a new content hash, ratings are
fitted one round at a time, and the old versions stay as history.

## Harness

Same shape as joe-rs's, with its own corpus and goldens:

- Parity corpus: `data/joe/unclejoe-parity/games/` (gitignored), built by
  `bots/unclejoe/tools/capture_fixtures.py` — the played seat is unclejoe
  itself, and the JAX oracle loads **unclejoe's artifact** through joe's
  imported network code. Because the bot argmaxes, the capture cross-check
  against the recorded `.out.log` grades the full Rust decision path against
  JAX on every turn (for joe-rs that check compares Python joe with itself
  — the two crates' caveats are opposites; do not copy them across).
- Committed smoke slice + self-goldens under `bots/unclejoe/tests/fixtures/`;
  drivers in `bots/unclejoe/tests/` (marker `joe`, outside the default
  suite); `tools/mutation_check.py` with the same 18 plants.
- The manifest-vs-bytes check runs in the default suite:
  `tests/test_joe_artifact_manifest.py` covers `unclejoe` alongside `joe-rs`.

Latency, measured over real recorded competition games (dev arm64, M-series):
forward ~57 ms per move — see [export.md](export.md) for the measured
percentiles against the 140 ms budget.
