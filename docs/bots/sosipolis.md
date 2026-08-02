# sosipolis

`bots/sosipolis/`. Research bot: **find the enemy general early, fund a thin
early castle programme, then strike.** Triple-mode MCTS with persistent fog
memory, mountain-pocket skip, and section priors.

Spec: [`../research/strategies/sosipolis.md`](../research/strategies/sosipolis.md).
Experiments:
[`../research/experiments/019-sosipolis-dual-mcts.md`](../research/experiments/019-sosipolis-dual-mcts.md),
[`../research/experiments/020-sosipolis-contact-castle.md`](../research/experiments/020-sosipolis-contact-castle.md),
[`../research/experiments/021-sosipolis-phase-defense.md`](../research/experiments/021-sosipolis-phase-defense.md).

## Layout

| File | Role |
| --- | --- |
| `run.sh` / `main.py` | entry |
| `stdio.py` | local wire protocol (no `bots/_common`) |
| `state.py` | persistent memory + priors + phase |
| `brain.py` | `Agent.act`: defense → economy → MCTS |
| `params.py` | every named threshold |
| `probe.py` | arena telemetry only |
| `components/` | map_memory, pockets, sections, search_mcts, contact_mcts, strike_mcts, economy, threat, army, clock |

## Behaviour

1. **search** — no enemy land known: SearchMCTS, wide front, pocket skip.
2. **contact** — enemy land remembered, general unknown: ContactMCTS, sector focus.
3. **strike** — general sighted: StrikeMCTS, gather and advance; no new castles.
4. From turn `CASTLE_START_TURN`, build up to `CASTLE_MAX` castles near the general.
5. Defense is scored inside each MCTS (perimeter + home bank). Brain only
   hard-overrides on kill shot or imminent loss.
6. Move budget: 100 ms hard cap. First move may use up to 9 s grace for precompute.

## Diversity

Owns the research axis **directed triple-MCTS general search**. Does not replace
`fog_scout`, `general_hunter`, or `castle_builder`.

## Verification

```bash
python competition-module/competition/matchup.py \
  bots/sosipolis/run.sh \
  bots/smoke/run.sh \
  --mode competition --seed 0
```
