# Experiment protocol

Use this for every measurable bot or arena change.

## Steps

1. **Hypothesis** — one change, one claim. Write a short note under
   `docs/research/` when useful.
2. **Match grid** — default Rule C: `--games-per-pair N` (default 50) random
   map seeds per unordered pair; pin `--round-seed` for a reproducible round.
   Use `--seeds` only for fixed-seed A/B debug.
3. **Seat policy** — `--seat-policy random` (the default) for exploratory and
   regeneration rounds: seat is drawn per game from the seeded stream, so it
   balances in expectation at **zero extra games**. `--seat-policy alternate`
   for decision arms: every map seed is played in both orientations, giving an
   exactly 50/50 split and cancelling map difficulty within each matched pair.
4. **Run** — always competition mode. Store games under `data/games/<round>/`
   via `arena/tournaments/competition.py` or `scripts/measure_heuristics.py`.
   The runner registers each bot's content hash before the pool starts.
5. **Metrics** — winrate, draw rate, mean turns, sample size, and the pairwise
   rating contrast (`Δ ± SE`, `CI₉₅`, `P(B > A)`) from a refit.
6. **Decision** — apply
   [`docs/arena/decision-rule.md`](../arena/decision-rule.md). Do not merge
   strategy changes without a measured contrast in `data/`.

## Rules

- Prefer one change per experiment.
- Parallel batches store with ratings off inside workers; every round is refitted
  once after the round finishes. There is no incremental rating update, and no
  pooled fit — each round is fitted independently over its own games.
- A decision compares two **content hashes**, not two bot ids. The hash moves
  on its own when the bot's source closure changes; the registry records it.
- Both arms of a contrast must be measured **in the same round**, with matched
  seed lists and `--seat-policy alternate`. A baseline measured in earlier
  rounds is not a valid comparator: round-to-round drift alone has measured at
  +46 Elo between byte-identical programs (see
  [`docs/arena/decision-rule.md`](../arena/decision-rule.md)). Re-run the
  baseline alongside the candidate — keep a frozen pre-change copy of the bot
  under `bots/` for the duration of the comparison, then delete it. Since ratings
  became per-round this is enforced rather than advised: a cross-round contrast has
  no answer to return.
- Never quote a rank as a result — ranks restart at 1 in every round. Quote the
  contrast.
- Two engine eras never pool. If `excluded.engine_mismatch` is non-zero after a
  refit, the `competition-module` submodule moved and the leaderboard needs a
  regeneration round.
- Verification gate: matches must finish under competition mode (see root
  `AGENTS.md`).

## Sample size

At the observed 35% draw rate, `SE(Δ) ≈ 430.9/√n` per arm:

| Games per arm | `SE(Δ)` | What it buys |
| ---: | ---: | --- |
| 200 | ≈ 30 Elo | the floor for *any* verdict |
| 1150 | ≈ 12.75 Elo | a `±25` interval — "proven flat" becomes affordable |

`fit.games_to_resolve(a, b, target_se=...)` computes this from the model's own
Fisher information for the contrast, using the games already in hand.

## Related

- [`docs/arena/decision-rule.md`](../arena/decision-rule.md) — the thresholds
- [`docs/arena/ratings.md`](../arena/ratings.md) — the model
- [`docs/arena/bot-version-registry.md`](../arena/bot-version-registry.md)
