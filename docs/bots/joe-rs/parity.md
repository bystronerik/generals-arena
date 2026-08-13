# joe-rs parity harness

How the Rust port is proved equal to the deployed Python joe, and how the
proof is itself checked. Tiers follow port-plan §6; both morpheus-rs rules
apply: *replay proves agreement, mutation proves the proof*, and *loose
float tolerances cannot see precision bugs*.

## Corpus

`tools/capture_fixtures.py`, two phases:

- **Play**: real `--mode competition` matchup games with the deployed
  `bots/joe/run.sh` seat wrapped in `tee`, recording the exact wire text joe
  received (`.in.log`) and replied (`.out.log`). 13 games, mixed opponents
  and seeds (incl. two joe mirrors); longest natural game 675 turns.
- **Capture**: replay each `.in.log` through the *imported* joe functions
  (`frame_to_raw`, `joe_obs.*`, `net._forward`) under `eqx.filter_jit`, dump
  per-turn surfaces to `.npz`. Every recomputed reply is asserted equal to
  the recorded one, so the capture path is pinned to deployment.

`synthetic-long` is metro-seed8's 660 frames played twice through the state
machine (1,320 turns): the 800+ regime, both 512-window rollovers, and
counters past natural game length. The state update never reads the bot's
actions, so any frame stream is a valid state-machine input. Corpus lives
under `data/joe/joe-rs-parity/games/` (derived, gitignored); a 40-turn
9-frame smoke slice is committed under `bots/joe-rs/tests/fixtures/` and the
drivers fall back to it automatically.

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
  nominal ≤ 1e-4): achieved over 13 games / 673 sampled frames —
  max |Δlogit| 3.052e-5, max |Δbin| 4.387e-5, max |Δvalue| 9.537e-7.
  Enforced at the achieved bounds (`LOGIT_TOL_ACHIEVED` etc. in
  `test_parity.py`); a corpus refresh that exceeds them is a finding.
- **Tier 3, decision**: greedy action equal on 100% of sampled frames
  (gate: ≥ 99.5% with tie-margin enumeration — none needed), and
  `test_wire_replay.py` runs the real binary in wire mode over every
  recorded game: **14 games, 5,693 turns, every reply byte-equal** to
  Python joe's recorded replies, under the bot's own accumulated state.

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
**9/9 killed** (2026-08-14). The tie-break kill comes from the crate's unit
tests, which the checker runs alongside the parity drivers: an exact logit
tie never occurs in real frames, so no fixture can see that flip.

## Running it

```bash
cargo build --release --manifest-path bots/joe-rs/Cargo.toml
.venv/bin/pytest bots/joe-rs/tests/ -m joe          # full corpus if present
.venv/bin/python bots/joe-rs/tools/mutation_check.py
```
