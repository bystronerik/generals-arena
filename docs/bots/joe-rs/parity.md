# joe-rs parity harness

How the Rust port is proved equal to the deployed Python joe, and how the
proof is itself checked. Tiers follow port-plan §6; both morpheus-rs rules
apply: *replay proves agreement, mutation proves the proof*, and *loose
float tolerances cannot see precision bugs*.

## What parity covers: the network, not the selection layer

**joe-rs ports joe's network path. Selection above the network is joe-rs's
own design, not a port.**

`bots/joe/agent.py` takes the plain argmax of the network's masked logits.
(It applied a **repetition penalty** first between 2026-08-16 and
2026-08-20, commit `1f550ad`; that came out again — see
[joe-argmax-limit-cycle](../../research/measurements/joe-argmax-limit-cycle.md).)
Since 2026-08-20 joe-rs answers the limit cycle the penalty was written
against with **deterministic Gumbel selection** (`src/board/select.rs`,
selection-plan S1): an exact sample of `softmax(logits / T)` whose noise is
hashed from `(board digest, turn, action index)`, so the binary stays a pure
function of the game. `JOE_RS_TEMPERATURE` sets T (default 1; 0 restores the
plain argmax). Neither bot's selection is a spec for the other; the
divergence is deliberate and documented in
[selection-plan.md](selection-plan.md).

So the two bots emit different moves, by design. The consequence for this
harness is specific and easy to trip over:

| reference | what it is | who is graded on it |
| --- | --- | --- |
| `.npz` surfaces, incl. `all_action` | the JAX oracle: network out, greedy argmax | joe-rs's network — every tier of `test_parity.py` |
| `<name>.joe-rs.log` | joe-rs's **own** recorded replies (Gumbel, default T) | joe-rs's full played path — `test_wire_replay.py` |
| `.out.log` | what **deployed joe** actually replied | nobody grades joe-rs on this |

`test_wire_replay.py` therefore compares the binary's whole stdout, byte for
byte, against its own self-golden (`capture_fixtures.py
--selection-golden`): selection is deterministic, so an unexplained change
anywhere in the played path still fails, while no sibling is ever the
reference. Grading joe-rs on the `.out.log` fails it for a difference it is
supposed to have — that is what happened on the step-29000 rebuild, at
`aegis-seed0` turn 57 — and since S1 shipped the same is true of the
oracle's `all_action` argmax, which the noise deliberately departs from on
near-ties (12.9% of corpus turns at T = 1).

The `.out.log` still earns its keep: `capture_fixtures.py` asserts the
recomputed reply against the recorded one, turn for turn, which keeps the
recorded games honest about the deployed path. Python joe plays the argmax,
so that check now compares the capture with itself. It has one operational
edge: **the logs must come from the current joe.** A corpus recorded before
joe's selection last changed — the penalty removal on 2026-08-20 is the most
recent such change — fails the check on every turn the old program differed.
Re-play those games; do not debug the capture.

## Corpus

`tools/capture_fixtures.py`, two phases:

- **Play**: real `--mode competition` matchup games with the deployed
  `bots/joe/run.sh` seat wrapped in `tee`, recording the exact wire text joe
  received (`.in.log`) and replied (`.out.log`). 14 games, mixed opponents
  and seeds (incl. three joe mirrors); longest natural game 801 turns. All
  14 are in `DEFAULT_GAMES`; four of them used to be passed by hand as
  `--game`, so a bare `--play` rebuilt only ten and dropped two joe mirrors.
- **Capture**: replay each `.in.log` through the *imported* joe functions
  (`frame_to_raw`, `joe_obs.*`, `net._forward`) under `eqx.filter_jit`, dump
  per-turn surfaces to `.npz`. Every recomputed reply is asserted equal to
  the recorded one, so the capture path is pinned to deployment.

A third phase, **self-golden** (`--selection-golden`), runs the joe-rs
release binary over every `.in.log` and records its replies as
`<name>.joe-rs.log` — the wire-replay reference since selection-plan S1.
`make_smoke_fixture.py` records one for the committed smoke slice too.

**The corpus is keyed to the network — and the goldens to the selection
layer as well.** The `.npz` surfaces are the JAX oracle's outputs for
specific weights, so a joe re-export invalidates them: re-run both phases
(and `make_smoke_fixture.py`, and `--selection-golden`) after
`convert_artifact.py`, or the drivers compare a new binary against an old
oracle. A deliberate selection change regenerates the goldens alone. The
corpus was rebuilt for step 13500 on 2026-08-14.

`synthetic-long` is the longest corpus game's frames played twice through
the state machine — at step 13500, castle_rush-seed3 doubled to **1,602
turns** (macaria-seed1 doubled to 1,572 at step 6000; metro-seed8 doubled to
1,320 at step 5000 — the source game changes with the net, so the rebuild
picks whatever the longest natural game is). It exercises the counters past
natural game length and both 512-window rollovers. The state update never
reads the bot's actions, so any frame stream is a valid state-machine
input, and there is deliberately no `.out.log` to cross-check against.
`tools/make_synthetic_long.py` builds it; it used to be a hand-run snippet,
which is why corpus rebuilds kept losing the fixture. Corpus
lives under `data/joe/joe-rs-parity/games/` (derived, gitignored); a 40-turn
13-frame smoke slice is committed under `bots/joe-rs/tests/fixtures/` and
the drivers fall back to it automatically.

## Mechanics

`joe-rs parity {raw,cost,mask,obs,forward,decide,sequence,log1p}` reads a
flat whitespace-separated integer stream on stdin (floats as f32 bit
patterns — nothing to argue about in formatting) and writes one in the same
shape. Drivers: `bots/joe-rs/tests/test_parity.py` (marker `joe`, outside
the default suite). The `sequence` surface replays a whole game and emits a
CRC-32 of every turn's augmented tensor plus the final state; localization
of a sequence failure is the per-frame `obs` surface with the recorded
input state.

## Tiers and pinned bounds

- **Tier 1, bit-exact**: wire parse, 14-channel raw, build cost, both
  masks, action codec, all 39 augmented channels including channel 21, the
  full state after every turn of every corpus game. Two oracle behaviors had
  to be mirrored rather than assumed (measured 2026-08-14, see
  [xla-semantics.md](xla-semantics.md)): XLA's constant-division-to-
  reciprocal rewrite, and XLA's own `log1p` polynomial with backend FMA
  contraction. Channel 21's pin is **0 ULP**, backed by an exhaustive sweep
  of the whole integer counter domain (0..16384) against live JAX.
- **Tier 2, tolerance** (candle forward, different GEMM summation orders;
  nominal ≤ 1e-4): achieved over 14 games / 731 sampled frames at step 6000 —
  max |Δlogit| 2.861e-5, max |Δbin| 5.722e-5, max |Δvalue| 2.205e-6.
  Enforced as **per-frame relative** error (each frame's max |Δ| over its own
  largest activation): 2.331e-6 logit, 2.244e-6 bin, pinned at 3.0e-6 with the
  absolute 1e-4 contract kept as a backstop. A refresh that exceeds these is
  still a finding.

  The gate was absolute until 2026-08-14, when the step-6000 corpus tripped
  the old `BIN_TOL_ACHIEVED` at 5.722e-5 > 4.4e-5. That was **not** a
  precision regression: the step-6000 value head is more confident, so its
  bin logits are 1.32× larger (35.18 → 46.54) and the absolute error grew
  1.30× with them, leaving the ratio flat (1.247e-6 → 1.230e-6). An absolute
  pin fails on every stronger export for a reason that is not a bug; the
  relative one is the invariant a real precision bug would move. Normalising
  per frame rather than per corpus matters too — the early-game smoke slice
  has ~half the activation magnitude, and a corpus-wide scale made it read
  1.7e-6 against a bound the full corpus passed.
- **Tier 3, decision**: greedy action equal on **731/731 sampled frames
  (100%)** (gate: ≥ 99.5% with tie-margin enumeration — none needed; the
  tie-margin bound is the relative tier-2 bound put back on the corpus's own
  logit scale). The `decide` surface is the network's greedy argmax and
  stays graded against the oracle; the *played* decision adds Gumbel noise
  on top (S1) and is graded by `test_wire_replay.py` against the self-golden
  instead: the real binary in wire mode over every recorded game, **15
  streams (14 games + synthetic-long), 8,985 replies, whole stdout
  byte-equal**, under the bot's own accumulated state.

## Synthetic fixtures

`tests/test_synthetic.py` builds the frames the corpus cannot isolate: an
18×19 board with an owned corner cell whose 3×3 visibility spills into the
pad region (the seen-pad-mountain rule, including its memory after the cell
is lost), and the enemy-visibility counterpart (pools into `enemy_seen`,
never becomes a mountain). Compared against the imported JAX oracle turn by
turn. Codec edge cases (half-move, build, pass decode) are Rust unit tests
in `src/board/action.rs`.

## Mutation pass

`tools/mutation_check.py`: 13 planted bugs — history-roll direction,
seen-accumulation OR, pad-mountain rule, a divide-by-50 site, the channel-21
counter, the R2 q/k-proj swap, softmax scale, argmax tie-break, pass-channel
mask, and four in the selection layer (turn-blind hash, noise on masked
entries, temperature ignored, noise sign) — each must make the harness fail.
Baseline must pass first, on the fast `mutation` profile (same float
semantics; Rust does not reassociate). **13/13 killed** (2026-08-20). The
tie-break and all four selection kills come from the crate's unit tests,
which the checker runs alongside the parity drivers: an exact logit tie
never occurs in real frames, and the corpus cannot see the selection plants
either — a turn-blind seed still varies with the board, a masked logit
already carries −1e9, and the noise's sign and scale only show in the
draw's distribution, which `select.rs`'s frequency test pins to softmax.

## Running it

```bash
cargo build --release --manifest-path bots/joe-rs/Cargo.toml
.venv/bin/pytest bots/joe-rs/tests/ -m joe          # full corpus if present
.venv/bin/python bots/joe-rs/tools/mutation_check.py
```

After a deliberate selection or network change, regenerate the wire-replay
goldens before the suite:

```bash
.venv/bin/python bots/joe-rs/tools/capture_fixtures.py --selection-golden
```

(and `make_smoke_fixture.py` for the committed smoke slice's golden).
