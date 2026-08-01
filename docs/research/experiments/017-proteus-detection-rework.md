# 017 — proteus: opponent detection rework

Bot: [`bots/proteus/`](../../../bots/proteus/). Follows
[016](016-proteus-adaptive-switching.md). Spec:
[`strategies/proteus.md`](../strategies/proteus.md).

Baseline `proteus@78bfa6234921` · candidate `proteus@7ae237e99d34`.

## Verdict

**`no change (proven flat)`**, per
[`docs/arena/decision-rule.md`](../../arena/decision-rule.md).

```
delta.value  = -3.37 Elo      (theta_B - theta_A)
delta.se     =  9.89
delta.ci     = [-22.76, +16.01]
p_stronger   =  0.367
comparable   =  True
```

`CI₉₅` lies entirely within ±25. Gates: both entities registered and
non-provisional; every game `mode == "competition"`; one `engine_version`;
`fit.comparable(A, B) = True`; ≥200 games per arm; ≥30 per (arm, opponent);
≥60 decisive per arm. 20,376 games in the fitted pool.

The rework is **equal in strength** to what it replaced. It ships on
simplicity and instrumentation, not on strength: two cores instead of four,
no dead code, and the evidence behind each switch recorded per turn.

## What the roster sweep established

A 32-game-per-cell grid of every pure core against every roster bot:

| core | pooled over 12 contested opponents | uniquely best against |
| --- | --- | --- |
| boom | 0.816 | aegis, army_convey, late_rush, metro, fog_scout |
| blitz | 0.788 | blitz, boom, cm_hunter |
| aegis | 0.603 | *nothing* |
| metro | 0.596 | *nothing* |

- The old proteus was **byte-identical to pure blitz in 6-8 games of 8**
  against every hard opponent. Its switching bought nothing (proteus 0.797,
  pure blitz 0.802, pure boom 0.839 over the same 12 opponents, same seeds).
- `metro` and `aegis` were unreachable, and *should* be: dominated, uniquely
  best nowhere, and aegis is 1-0-31 against `garrison` — two turtles running
  out the [`RULES.md`](../../../RULES.md) §07 draw.
- Dropping both is **behaviour-neutral**: verified byte-identical action
  streams in 10/10 games across five opponents.

## Two defects found and fixed

1. **The rusher rule ANDed facts that never co-occurred.**
   `OpponentModel` latches a running-min enemy distance and a running-max
   enemy stack independently, so `closest <= 8 and biggest >= 15` fired on a
   border nibble at distance 8 in turn 90 plus an unrelated stack seen across
   the map in turn 300. `HomePressure` latches the conjunction instead.
2. **Three of four labels scored against `my_tiles`.** Switching cores
   changed proteus's own land curve, which changed the label, which changed
   the core. The trace showed the loop as a stereotyped oscillation: against
   `aegis`, `[162 turtler→boom], [212 boomer→blitz]` on nearly every seed and
   both seats — reaching the core that wins 0.86 and leaving it 50 turns
   later.

## What did not work, and why it looked like it did

The first candidate left the blitz spine at turn ~162. A blitz opponent's
fist reaches our door at turn ~226, so proteus banked for sixty turns and met
the wave with an economy. Holding the spine until `DUEL_DEADLINE` fixed the
three matchups it cost.

**The bigger lesson is methodological.** Both step-3 iterations were tuned on
8-16 game cells, far below the noise floor. In-process runs showed
`army_convey` +0.44; under the formal protocol at 50 matched-seed games it is
**+0.02**. The exit-turn sweep that chose turn 275 over turn 162 was 16
games per cell — the two settings are indistinguishable. Tune after the
decision protocol, not before it.

## Method note: buying precision with a parity twin

proteus beats every roster bot 80-100% of the time, and a game at 80% carries
a fraction of the Fisher information of a game at parity. Routing the
contrast through the ordinary panel needed **~13,000 games per arm**; routing
it through `bots/proteus_base/` — a frozen, verified byte-copy of the
baseline policy — reached the target in **~4,600**, matching the ~1,150 the
decision rule quotes for a balanced design.

`SE(Δ)` across successive rounds: 60.5 → 35.4 → 19.8 → 14.2 → 12.5 → 10.7 →
**9.89**. `delta.value` converged −37 → −35 → −25 → −12 → −7 → −5 → −3.4.

The twin was deleted once the verdict was recorded, per its own README. To
recreate it: copy `bots/proteus/` at the baseline commit into
`bots/<name>/`, repoint the `from proteus.` imports, register it, and verify
byte-identical play before trusting a contrast that leans on it.

## Reproduction

```bash
python -m arena.tournaments.competition \
  bots/proteus/run.sh bots/<panel...>/run.sh \
  --round proteus-detect-<n> --games-per-pair 50 --round-seed 7 \
  --seat-policy alternate --strict-versions
```

Panels used, each declared before either arm ran: round 1 the four pure cores
plus `cm_expander`, `expand_plus`, `garrison`; round 2 added `army_convey`,
`late_rush`, `phase_switch`, `classic_duel`, `fog_scout` — the opponents
where the effect was predicted, and where it also failed to appear; rounds
3-7 the parity twin.
