# Learned bot plan

Brief placeholder only. No training code lives here yet — this is a
pointer for the agent that picks up learned bots, not an implementation.

## Scope (later)

- Train against the heuristic bots (`bots/expand_plus/`,
  `bots/castle_builder/`, `bots/general_hunter/`) plus `bots/smoke/` and
  `expander_python` as a fixed opponent pool.
- Reference the engine's vectorized env / experimental PPO under
  `competition-module/examples/_experimental/` for API shape only — do not
  edit submodule internals.
- Training data: accumulated `data/games/` records, plus per-turn trajectories.
  The follow-up this plan left open is **resolved** — see
  [`docs/arena/trajectories.md`](../arena/trajectories.md). A recorded game
  stores the seed and the applied action sequence, and
  `python -m arena.records.trajectories <file> --materialize` replays it into
  dense per-turn states on demand, so states are regenerated rather than
  stored. Final land and army are now engine truth on **every** record,
  recorded or not, which closes the land/army telemetry gap that the early
  `expand_plus` and `castle_builder` work flagged. Shard layout is still an
  open decision for whenever this work starts.
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
- Changing `arena/` schemas — propose that as its own experiment note if a
  learned bot needs it.
