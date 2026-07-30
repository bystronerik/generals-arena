# Learned bot plan (Phase 4 scaffold)

Brief placeholder only. No training code lives here yet — this is a
pointer for the agent that picks up Phase 4, not an implementation.

## Scope (later)

- Train against the Phase 3 heuristic bots (`bots/expand_plus/`,
  `bots/castle_builder/`, `bots/general_hunter/`) plus `bots/smoke/` and
  `expander_python` as a fixed opponent pool.
- Reference the engine's vectorized env / experimental PPO under
  `competition-module/examples/_experimental/` for API shape only — do not
  edit submodule internals.
- Training data: accumulated `data/games/` records, plus richer per-turn
  trajectory dumps if/when the game-record schema grows to support them
  (see the open follow-up in
  [`docs/research/experiments/001-expand-plus-frontier-march.md`](experiments/001-expand-plus-frontier-march.md)
  and
  [`002-castle-builder-early-investment.md`](experiments/002-castle-builder-early-investment.md)
  about missing final land/army telemetry).
- Checkpoints become rated arena citizens through the same
  `arena/run_match.py` / `arena/tournament.py` + `arena/ratings.py` path
  used for heuristic bots — no separate scoring path.

## Open question worth resolving first

All 31 stored games so far (`data/games/`) between the current heuristic
bots are draws at the 1200-turn cap — none of them scout deep enough into
enemy territory to end a game early. A learned bot's reward signal needs
either denser per-turn observations/telemetry or an opponent pool that
produces decisive games, or early training will mostly see draws too.

## Not in scope here

- Any model code, training loop, or checkpoint format.
- Changing `arena/` schemas — propose that as its own experiment note if
  Phase 4 needs it.
