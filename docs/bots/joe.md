# joe

`bots/joe/`. The deployed **Average Joe** agent: the EMA policy of the pure
self-play PPO run under competition rules
([plan](../research/strategies/averagejoe-competition-plan.md), Phase 5),
played greedily — no search, no heuristics, one forward pass per move.

Plan: [`averagejoe-competition-plan.md`](../research/strategies/averagejoe-competition-plan.md).
Arena entry report:
[`joe-phase5-arena.md`](../research/measurements/joe-phase5-arena.md).

## Layout

| File | What it is |
| --- | --- |
| `joe_obs.py` | copy of `training/joe/networks/common.py` + a port of the engine's move mask |
| `joe_net.py` | copy of `training/joe/networks/transformer.py` (`HistoryTransformer`) |
| `agent.py` | wire frame → 14-channel tensor → 39-channel augment → greedy argmax |
| `artifact/manifest.json` | architecture + checkpoint provenance (run, step, sha256, R2 key) |
| `artifact/ema.eqx` | the EMA weights — **gitignored**; re-fetch with the export script |
| `run.sh` | thread pinning (OMP/BLAS/XLA to 1) + `JAX_PLATFORMS=cpu` |

### Why the training code is vendored, not imported

The content hash (`arena/records/fingerprint.py`) covers only files under
`bots/`, and a submission bundle ships only the bot directory — an import of
`training.joe` would leave the program's actual decision code outside its
rating identity. The copies are pinned to the training path by
`bots/joe/tests/test_wire_fidelity.py` (marker `joe`): bit-identical tensors,
masks, and greedy actions on a live engine game, and identical network
outputs after a serialization round trip.

## Per-turn path

1. Parse the wire frame back into the engine's 14-channel observation tensor
   (`frame_to_raw` — exact inverse of `competition/protocol.py`).
2. Compute the per-cell build cost and the move/build masks from the tensor
   alone (own structures are always visible, so obs-side cost equals the
   state-side `build_cost_grid`).
3. `augment_obs` with the persistent `AugmentedObsState`: 39 channels, army
   history stacks, seen/structure memory, opponent temporal windows. True
   18–21 boards are padded to 21 with padding-as-mountains.
4. One float32 forward pass (`use_bf16=False` — CPU has no fast bf16 path),
   greedy argmax over the masked 10-channel head, decode to the wire action.

The obs state carries across turns and is never reset mid-game; the JIT
compile happens in `Agent.__init__` inside the first-move grace.

## Weights

`artifact/ema.eqx` is exported by `scripts/joe_export_bot.py` from the R2
checkpoint store (or a local run dir). The manifest pins run name, global
step, engine SHA, sha256, and R2 key, so any rated version's weights are
recoverable. Re-exporting changes the content hash — a new checkpoint is a
new rated entity by construction.

Latency (`scripts/joe_bot_latency.py`, full competition games over the real
wire): p50 5.0 ms, p99 5.4 ms per move on an M3 Pro core; first move 2.2 s.
The x86 budget evidence is
[`joe-phase2-cpu-latency.md`](../research/measurements/joe-phase2-cpu-latency.md)
(M tier p99 ≈ 19 ms on one hard-limited core vs the 150 ms limit).

## Rust sibling

`bots/joe-rs/` plays this same frozen network — identical greedy decisions,
proved turn-for-turn over recorded games — as a separate rated entity with
its own lineage. Port design: [joe-rs/port-plan.md](joe-rs/port-plan.md);
the parity evidence is [joe-rs/parity.md](joe-rs/parity.md). Editing
`bots/joe/` is still off-limits from the joe-rs side; the one shared object
is the exported artifact, converted read-only
([joe-rs/export.md](joe-rs/export.md)).

"Same network" is a claim someone has to maintain: it holds only while
joe-rs's artifact is a conversion of joe's *current* one. A joe re-export
breaks it silently — joe-rs keeps its own `model.safetensors` and goes on
playing the previous checkpoint — so every re-export must be followed by the
re-conversion and parity rebuild in
[joe-rs/export.md](joe-rs/export.md#after-a-joe-re-export). Both bots are on
**step 6000** as of 2026-08-14.
