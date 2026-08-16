# The observation bridge

Landed at N2, 2026-08-16. Morpheus's frame in, joe's 39-channel network input
out — `bots/morpheus-joe/crates/core/src/nn/bridge.rs`. The plan is
[joe-net-plan §4](../morpheus-rs/joe-net-plan.md); this page is what the code
does and how it is proved.

## Two pipelines, one authority each

The port keeps both observation pipelines and gives each exactly one job.

| pipeline | authoritative for |
| --- | --- |
| joe's `board/obs.rs` + `AugState` | the network's 39-channel input, and nothing else |
| morpheus's `board/memory.rs` + `VisibleMemory` | legality, build cost, tactics, node identity, belief filtering, symmetry |

Morpheus's own 49-plane `build_tensor` is deleted. It was the expensive half of
the old path and it described a network that no longer exists.

## What the bridge does per turn

Joe's order, unchanged, because a reordering here is silent:

```
widen the frame        u8 grids -> i32 grids; scalars already agree
frame_to_raw           -> (14, h, w)
build_cost_from_raw    -> (h, w) castle price
augment_obs            reads the old AugState, fully writes the new one
mem::swap              this turn's answer becomes next turn's input
copy the ring buffers  -> the (2, 512) temporal input
normalize_observations *= 1/50 on the army-valued channels
```

Everything after the widening is joe's code, byte for byte, in the `joenet`
crate. The bridge adds no arithmetic.

**The widening is a widening.** The two `Observation` structs carry the same
ten fields in the same order and differ only in element type; the `TYPE_*` and
`OWNER_*` constants are the same wire codes, and both parsers read them off the
same protocol line ([joe-net-plan §1.6](../morpheus-rs/joe-net-plan.md)).

**The penalties are zero and stay zero.** Joe's `-1e9` action mask is added to
the flat logits *after* unpatchify, in one elementwise pass, so the trunk never
sees it — and the port does not want joe's mask anyway, because morpheus masks
with `play_mask` at leaves and `legal_mask` elsewhere and those are strictly
different sets (§6.2, §7.2). N0 priced the mask build this removes at 0.007 ms,
so it is a correctness decision and not a saving.

**Two methods where joe-rs has one.** `advance` is the play path; `augment`
stops one step short, before `normalize_observations`, because that is where
joe's recorded corpus takes its digest. Splitting them is what makes the check
below possible.

**The bridge does not know where its frame came from**, and that is §6.1's
point rather than an accident. The 512-step temporal window is a ring buffer of
the wire scalars `opp_army` and `opp_land`, and `emit_observation` fills those
two fields from the true state totals at any node in the search tree — so
appending one value is O(1) and exact wherever the frame is from. At N2 the
frames are the engine's; at N3 they are the transition kernel's, and nothing
here changes.

## Cost, and why the history is not frozen

N0 measured `augment_obs` at 0.042 ms and the whole observation path at
0.067 ms, against a 20.81 ms forward. That is §6.3's top row by an order of
magnitude, so the 14 delta-history planes are advanced **per node** rather than
frozen at the root, and risks R3 and R8 retired with the measurement. At
`search_depth: 8` that is nine `AugState`s — 403 KB — and 0.38 ms of
`augment_obs` against a 23 ms leaf forward.

## How it is proved

Two gates, and they fail differently.

### The `sequence` parity surface

`morpheus-joe parity sequence` replays a whole recorded game and reports, per
turn, a CRC-32 of the augmented tensor and a count of build-cost
disagreements; then the final `AugState`, then the `(2, 512)` temporal buffer.
`bots/morpheus-joe/tests/test_morpheus_joe_bridge.py` drives it.

**Its oracle is not Python morpheus.** It is joe's own recorded corpus under
`data/joe/joe-rs-parity/games/`, which already stores `all_aug_hash` per turn
and `final_state_*` at the end — captured from the JAX pipeline, not from Rust.
That is what makes this cheap: the fork records nothing, and the thing it is
compared against was produced by joe itself. Q11 is answered — the fork needs
no fixtures of its own.

Four assertions, because they catch different bugs:

| assertion | catches |
| --- | --- |
| per-turn tensor digest | any wrong channel, on the turn it first goes wrong |
| final `AugState`, field by field | state fields no channel reads — `enemy_seen`, the two `last_enemy_army_seen_*` planes |
| the temporal buffer is the two ring buffers | §6.1's window copied stale or with its halves swapped, which the two above cannot see |
| build-cost disagreements are zero | Q10 (below) |

**Measured 2026-08-16 on the step-23500 corpus: every digest equal, every final
state bit-exact, zero disagreements**, over `aegis-seed0` (21×18, 491 turns),
`blitz-seed7` (21×19, 177) and `boom-seed2` (21×20, 299) — 967 turns and three
board widths, none of them square, so the pad path is exercised three ways.

This gate **survives a joe re-export unchanged**, which the step-29000 export
later the same day confirmed: `augment_obs` is a function of the frames alone,
so a rebuilt corpus brings new games and new digests and the bridge still
matches them. The weights never enter it.

### Byte-equal replies

`test_morpheus_joe_wire_replay.py` pipes each game's `.in.log` to the binary in
wire mode and compares stdout to Python joe's recorded replies.

At N2 the fork **is** joe wearing morpheus's I/O — it plays joe's argmax and
runs no search — so every remaining difference between the two bots is I/O and
a divergence is a bridge bug, full stop. Localizing one after N3 lands would
mean separating a bad tensor from a bad remap from a bad prior.

**Measured 2026-08-16 on the step-23500 corpus: 3 games, 967 turns, every reply
equal.**

Unlike the surface above, this gate **is** weight-dependent: it compares
decisions, so both bots must be on the same checkpoint. A joe re-export turns
it red until `scripts/joe_artifact_fanout.py` syncs this bot's weights, and
that is correct behaviour rather than a flake — it is the same staleness
`tests/test_joe_source_fanout.py` reports, seen from the other end. The
step-29000 export on 2026-08-16 did exactly this: the reply at
`aegis-seed0` turn 33 differed by one direction field, because the corpus was
step 29000 and the seat was still step 23500.

For this to be comparable at all, the argmax has to be over *masked* logits, as
joe's is. So N2 asks the network with zero penalties and adds joe's mask to the
logits it returns — bit-identical, because the forward's own use of `penalties`
is that same elementwise add. `ObsBridge::joe_action_penalties` is that
scaffolding and N3 deletes it, replacing the caller with the 4,410 → 3,970
remap and morpheus's `legal_mask`. **This bot stops replying like joe at N3, by
design**, and the gate retires rather than relaxing.

### Mutation coverage

Three mutations in `tools/mutation_check.py` under `nn/bridge.rs`: the
`mem::swap` deleted, the owner grid widened from the type grid, the temporal
input's halves swapped. **3/3 caught.** The third is caught only by the
temporal assertion, which is the argument for having added it.

`nn/bridge.rs` is the first row in that tool's `FILE_SURFACES` map whose
surface has a different oracle, so `_parity` now routes `sequence` to its
pytest driver and everything else to the Python-morpheus harness. N3's `prior`
surface wants the same treatment for a third oracle again.

The whole pass is **118/140 caught, 22 survived, every survivor with a recorded
note** (`docs/research/measurements/morpheus-joe-mutation-check.json`) — the
first full run the fork has had, because N1 left the harness unable to make it:
`morpheus_joe_parity_cases.py`'s CLI default `--kinds` still named the five
retired kinds, so the bare run the tool makes for every "all surfaces" mutation
died on `unknown kind 'tensor'`. Fixed at N2.

## Q10, closed

Joe's `build_cost_from_raw` and morpheus's `live_build_cost` must agree cell
for cell, because after §6.2 only morpheus's is consulted and a disagreement
would be silent — it would change which castles the bot believes it can afford
and say nothing.

They can differ in principle: joe prices the own structures the *frame* shows,
where morpheus folds in what `VisibleMemory` latched (`own_general`,
`known_castle`). Owning a cell implies seeing it, so the latch should never add
a structure the frame does not already carry.

Checked rather than argued, over 967 turns of three real games with the memory
advanced exactly as the runtime advances it: **zero disagreeing cells.** A
hand-built frame would not have reached a latched castle.

## What the search loses, and does not

The bridge does not restore the belief's write direction. Joe's patch embedding
is `PATCH_DIM = 39·3·3`, so belief planes cannot be appended without
retraining, and two leaves that differ only in hidden-state uncertainty will
receive identical priors and identical values (R1). That is priced by K1, not
fixed here.
