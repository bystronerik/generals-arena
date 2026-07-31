# Strategy spec — proteus

Bot: `bots/proteus/`. Migrated from the generals-bot repo (source name:
adaptive/Proteus). Grounded idea: **classify the opponent, switch between
the four migrated strategies with hysteresis.** Composes the cores of
[`blitz`](blitz.md), [`boom`](boom.md), [`metro`](metro.md), and
[`aegis`](aegis.md); baselines for comparison are those four pure bots and
`phase_switch` (the roster's clock-driven switcher).

## 1. Architecture

- **One shared `OpponentModel`** — `Agent.act` updates it once per turn;
  every core was constructed around the same instance, and each core's
  `observe()` is idempotent per turn, so warming never double-counts.
- **Classifier** (`classifier.py`) — pure function of the model + turn.
  Labels: rusher, boomer, citier, turtler, unknown. Ported verbatim; the
  only semantic rename is cities → castles (`enemy_castles_seen`), and the
  signal matters *more* here — the arena roster is full of castle-builders.
- **Switcher** (`switcher.py`) — debounced selection: same non-current
  proposal for `STREAK_NEEDED = 12` consecutive turns at confidence
  ≥ 0.45, at most every `COOLDOWN = 50` turns. Switching **into** aegis is
  cheap (`DEFENSE_STREAK = 5`, no cooldown — a rush that lands while we
  dither is fatal); leaving it is expensive (`LEAVE_DEFENSE_STREAK = 60` —
  rushers strike in waves).
- **Warm handover** — inactive cores' `observe(obs)` runs every turn so
  their beliefs, latches, and threat memories are current at switch time.

## 2. The counter map (source live calibration)

| Classified as | Play | Why |
| --- | --- | --- |
| rusher | blitz | Mirror the rush; nothing else survives one. |
| boomer | blitz | Punish greed before it scales. |
| citier | blitz | Kill the castle program before it compounds. |
| turtler | boom | Out-economy true passivity; don't feed a keep. |
| unknown | default (blitz) | Blitz is the spine, played from turn 0 — mid-game switches *into* blitz cold-start its opening and measurably lost games in the source. |

## 3. Classifier signals (all from exact aggregates + vision)

Rusher: early contact (< 120), **sticky** deep incursion (a ≥15 stack that
ever came within 8 of home), high concentration. Boomer: sustained land
rate > 0.30/turn, land lead > 1.3×, low concentration; a deep incursion
*subtracts* (a rusher's rebuild phase reads as economy). Citier:
`enemy_castles_seen` (1.5/castle) + production estimate > 0.65/turn
(general alone ≈ 0.5; the land-bonus filter `% 50` matches the arena
cadence). Turtler: stalled land rate, small territory, no contact, army
parked at home.

## 4. Arena-rule adaptations (vs the generals-bot source)

| Rule | Adaptation |
| --- | --- |
| Cities → castles | `enemy_cities_owned` → `enemy_castles_seen`; the "citier" label now detects castle-builders. |
| Everything else | Inherited from the four cores' own adaptations (see their specs). The source's rejected anti-rush Boom profile stays rejected. |

## 5. Diversity check

vs `phase_switch` (clock-driven phases): proteus switches on *opponent
evidence* with hysteresis, not on the turn number, and switches between
four full strategies rather than phases of one. vs the four pure cores:
proteus is exactly one of them at any instant — its value is picking which
one, so its measurement target is beating the *worst-case* pure matchups,
not the best-case ones.

## 6. Experiment hypothesis

**Hypothesis:** evidence-driven switching keeps blitz's strong matchups
while patching its weak ones (defensive fast-path vs rushers, boom vs
passive turtlers). Verification: beat `smoke` (capture 347 — classified
turtler, switched to boom, built a castle) and `expand_plus` (capture
344); lost the near-mirror to pure `blitz` (289). The composite machinery
(shared model, warm handover, switch telemetry) verified end-to-end.

Seed grid: opponents `smoke`, `blitz`, `garrison`, `castle_builder`;
seeds 0–2; `--mode competition`. Metrics: W-L-D, switches per game and
their labels (from telemetry), worst-case matchup vs the four pure cores,
faults.

See [`016-proteus-adaptive-switching.md`](../experiments/016-proteus-adaptive-switching.md).
