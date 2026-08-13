# joe-rs parity harness

How the Rust port is proved equal to the deployed Python joe, and how the
proof is itself checked. Tiers follow port-plan §6; both morpheus-rs rules
apply: *replay proves agreement, mutation proves the proof*, and *loose
float tolerances cannot see precision bugs*.

## Corpus

`tools/capture_fixtures.py`, two phases:

- **Play**: real `--mode competition` matchup games with the deployed
  `bots/joe/run.sh` seat wrapped in `tee`, recording the exact wire text joe
  received (`.in.log`) and replied (`.out.log`). 14 games, mixed opponents
  and seeds (incl. three joe mirrors); longest natural game 786 turns.
- **Capture**: replay each `.in.log` through the *imported* joe functions
  (`frame_to_raw`, `joe_obs.*`, `net._forward`) under `eqx.filter_jit`, dump
  per-turn surfaces to `.npz`. Every recomputed reply is asserted equal to
  the recorded one, so the capture path is pinned to deployment.

**The corpus is keyed to the network.** The `.npz` surfaces are the JAX
oracle's outputs for specific weights, so a joe re-export invalidates them:
re-run both phases (and `make_smoke_fixture.py`) after
`convert_artifact.py`, or the drivers compare a new binary against an old
oracle. The corpus was rebuilt for step 6000 on 2026-08-14.

`synthetic-long` is the longest corpus game's frames played twice through
the state machine — at step 6000, macaria-seed1 doubled to **1,572 turns**
(it was metro-seed8 doubled to 1,320 at step 5000; the source game changes
because a stronger net ends games sooner, so the rebuild picks whatever the
longest natural game is). It exercises the counters past natural game
length and both 512-window rollovers. The state update never reads the
bot's actions, so any frame stream is a valid state-machine input. Corpus
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
  logit scale), and `test_wire_replay.py` runs the real binary in wire mode
  over every recorded game: **14 games, 4,760 turns, every reply byte-equal**
  to Python joe's recorded replies, under the bot's own accumulated state.
  (The turn count fell from 5,693 at step 5000 because the stronger network
  wins sooner, not because coverage shrank — the game count is the same.)

## Synthetic fixtures

`tests/test_synthetic.py` builds the frames the corpus cannot isolate: an
18×19 board with an owned corner cell whose 3×3 visibility spills into the
pad region (the seen-pad-mountain rule, including its memory after the cell
is lost), and the enemy-visibility counterpart (pools into `enemy_seen`,
never becomes a mountain). Compared against the imported JAX oracle turn by
turn. Codec edge cases (half-move, build, pass decode) are Rust unit tests
in `src/action.rs`.

## Mutation pass

`tools/mutation_check.py`: 9 planted bugs — history-roll direction,
seen-accumulation OR, pad-mountain rule, a divide-by-50 site, the channel-21
counter, the R2 q/k-proj swap, softmax scale, argmax tie-break, pass-channel
mask — each must make the harness fail. Baseline must pass first, on the
fast `mutation` profile (same float semantics; Rust does not reassociate).
**9/9 killed** (2026-08-14, re-run against step 6000 and the relative tier-2
gate — the check is what proves the relative bound did not buy portability
by giving up detection power). The tie-break kill comes from the crate's unit
tests, which the checker runs alongside the parity drivers: an exact logit
tie never occurs in real frames, so no fixture can see that flip.

## Running it

```bash
cargo build --release --manifest-path bots/joe-rs/Cargo.toml
.venv/bin/pytest bots/joe-rs/tests/ -m joe          # full corpus if present
.venv/bin/python bots/joe-rs/tools/mutation_check.py
```
