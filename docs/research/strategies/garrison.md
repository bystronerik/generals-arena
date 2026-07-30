# Garrison strategy specification

## Goal

`garrison` protects its general before it expands or attacks. The bot keeps a
reserve on the general, reinforces the general when visible pressure increases,
and expands with armies that are not part of the reserve network.

The bot must use the competition observation and the five-integer stdio action.
The bot must not use hidden engine state. The bot must finish each decision
inside the 150 ms move limit.

## Strategy features

1. Find the own general from a cell with `owner == 1` and `type == 4`.
   Store this position because the general does not move.
2. Build a passable-cell graph from visible map data. Treat mountains
   (`type == 2`) and fogged structures (`type == 5`) as blocked.
3. Compute an own-territory distance field from the general on each turn.
   Reinforcement moves use only cells with `owner == 1`.
4. Estimate pressure from visible opponent armies near the general.
5. Keep a phase-based army floor on the general.
6. Move an own stack one step toward the general when the defense deficit is
   larger than zero.
7. Expand only when no urgent reinforcement move exists.
8. Remember the enemy general after a sighting. Use this memory only for threat
   context before turn 800. Do not weaken the own general for an early attack.
9. After turn 800, treat every passable route to the own general as a
   deathtouch route. Prefer route denial and source-cell capture over a larger
   passive garrison.

## State

The implementation must keep this state:

- `general_pos`: the fixed own general position.
- `enemy_general_pos`: the last confirmed enemy general position.
- `last_seen_enemy`: a map from cell to `(turn, army)` for recent visible enemy
  cells. Remove an entry after 20 turns or after the cell becomes visible and
  no longer belongs to the opponent.
- `last_defense_move_turn`: the last turn that moved an army toward the general.
- `last_expansion_target`: an optional target used only to make tie-breaking
  stable.

The implementation must not treat a stale enemy army count as current. A stale
entry can increase caution, but a visible entry must have more weight.

## Phases and thresholds

Use named constants so experiments can change one threshold at a time.

### Phase 1: establish, turns 1–199

- General reserve floor: `8`.
- Safe radius: own-territory distance `2` from the general.
- Threat scan radius: passable BFS distance `6`.
- Permit expansion from the general only when its army remains at or above the
  reserve floor after the move.
- Prefer nearby neutral land. Do not cross a visible opponent frontier unless
  the capture leaves a safe retreat path.

### Phase 2: fortify, turns 200–599

- General reserve floor: `16`.
- Safe radius: own-territory distance `3`.
- Threat scan radius: passable BFS distance `8`.
- Increase the reserve floor by visible pressure.
- Prefer reinforcement over expansion when the defense deficit is positive.
- Permit a local counterattack when it captures a visible enemy stack that can
  reach the general in two moves.

### Phase 3: prepare for deathtouch, turns 600–799

- General reserve floor: `24`.
- Safe radius: own-territory distance `4`.
- Threat scan radius: passable BFS distance `10`.
- Keep at least two owned layers around the general when the map permits this.
- Do not move the last movable army from a cell on a shortest visible route to
  the general.
- From turn 750, give a large score bonus to captures of visible enemy runner
  stacks inside the safe radius.

### Phase 4: deathtouch defense, turns 800–1200

- Army size on the general no longer stops a deathtouch move.
- Keep a nominal general reserve floor of `12` for normal combat.
- Define a critical runner as an enemy-owned cell with army at least `2` and
  passable BFS distance at most `2` from the general.
- If a critical runner is adjacent to the general, first search for a legal
  chase that captures the runner's source cell from a third own cell.
- If no chase exists, capture the runner before it can touch the general.
- Maintain movable stacks on cells at distance `1` and `2` from the general.
- Expand only when no visible or recent threat exists inside distance `6`.

## Threat model

For each visible opponent cell, compute a threat contribution:

`threat = max(0, enemy_army - distance_penalty) * proximity_weight`

Use these values:

- `distance_penalty = 2 * max(0, distance_to_general - 1)`.
- `proximity_weight = 8` at distance `1`.
- `proximity_weight = 5` at distance `2`.
- `proximity_weight = 3` at distance `3`.
- `proximity_weight = 1` at greater distances inside the phase scan radius.

For each recent stale enemy entry, use one quarter of the visible contribution.
The total visible pressure is the sum of the contributions, capped at `60`.

Before turn 800, calculate:

`required_general_army = phase_floor + ceil(total_visible_pressure / 4)`

After turn 800, use pressure to select intercept cells. Do not add all pressure
to the general reserve because army on the general does not stop deathtouch.

The defense deficit is:

`max(0, required_general_army - army_on_general)`

## Candidate move rules

Generate only legal-looking moves:

- Source belongs to the bot.
- Source army is at least `2`.
- Destination is inside the board.
- Destination is not a visible mountain or fogged structure.
- An attack must send strictly more army than the visible defender has.
- A reinforcement destination must belong to the bot.

The bot sends all but one army by default. The first implementation must not use
half moves. A later experiment can add half moves as one isolated change.

### Emergency defense score

Emergency defense applies when a visible enemy is within distance `2` of the
general or when the defense deficit is positive.

Score each candidate:

- `+100000` for a deathtouch chase that captures the source of an adjacent
  touch attempt.
- `+90000` for capturing an adjacent enemy runner after turn 800.
- `+50000` for capturing an enemy adjacent to the general before turn 800.
- `+5000 * distance_reduction` for a reinforcement move toward the general.
- `+100 * transferred_army` for the army that reaches the next defense cell.
- `-20000` if the move starts from the general and leaves less than the
  required general army.
- `-10000` if the move opens the only owned route between a threat and the
  general.

Select an emergency move before all expansion candidates.

### Reinforcement score

A reinforcement move must end on an own cell that is one own-territory BFS step
closer to the general.

Score each candidate:

`3000 * distance_reduction + 20 * transferred_army - 30 * source_distance`

Add these terms:

- `+2000` when the destination is inside the safe radius.
- `+1000` when the destination is on a shortest route from a visible threat to
  the general.
- `-5000` when the source is a frontier cell next to a stronger visible enemy.
- `-4000` when the move would leave the source with one army next to an enemy.

Use reinforcement when the defense deficit is positive. Also use reinforcement
when a visible threat is inside the safe radius.

### Careful expansion score

An expansion candidate captures a visible neutral or opponent cell.

Start with:

`score = 30 * captured_land_value + 5 * source_army - 4 * destination_army`

Use `captured_land_value = 1` for a neutral plain and `3` for an opponent cell.
Use these additions:

- `+1000` for a legal capture of the enemy general before turn 800.
- `+200` for an opponent castle.
- `+80` for a cell that increases the known frontier.
- `+40` for a cell that keeps own-territory distance to the general at most
  `6`.
- `+20` for a neutral plain with at least two own neighbors.
- `-500` when the source is inside the safe radius.
- `-1000` when the source is the general and the move breaks the reserve floor.
- `-200` when the destination has only one known passable neighbor.
- `-100` when the destination becomes adjacent to a stronger visible enemy.

The bot must reject an expansion move if the move creates a defense deficit.
The bot can expand while below the general reserve floor only from a source
outside the safe radius and only when no reinforcement path exists.

### Idle consolidation score

If no capture is safe, move an outer own stack toward the safe radius.
Use the same own-territory distance field as reinforcement. Do not move the
general. Prefer the largest stack and the greatest reduction in distance.

If no consolidation move exists, return pass.

## Pseudocode for `act()`

```text
act(observation):
    locate and store own general
    update enemy-general memory
    update recent visible-enemy memory

    build passable distance field from own general
    build own-territory distance field from own general
    identify visible threats and compute pressure
    choose phase from turn
    compute reserve floor, safe radius, and defense deficit

    emergency_moves = score emergency captures, chases, and blocks
    if emergency_moves is not empty:
        return highest-scoring emergency move

    if defense deficit is positive or threat is inside safe radius:
        reinforcement_moves = score moves one own step toward general
        if reinforcement_moves is not empty:
            return highest-scoring reinforcement move

    expansion_moves = score safe visible captures
    reject moves that break reserve or critical route rules
    if expansion_moves is not empty:
        return highest-scoring expansion move

    consolidation_moves = score outer stacks moving toward safe radius
    if consolidation_moves is not empty:
        return highest-scoring consolidation move

    return pass
```

Use deterministic tie-breaking in this order:

1. Higher score.
2. Larger source army.
3. Shorter destination distance to the general for defense.
4. Smaller source row, source column, and direction code.

## Edge cases

- If the own general is not present in the first frame, pass until the bot finds
  it. Do not guess the general position.
- If no own-territory reinforcement route exists, use a passable visible route
  only for threat distance. Do not send a reinforcement into neutral fog.
- Treat `type == 0` as fog. Do not call a fog cell neutral expansion.
- Treat `type == 5` as blocked because the hidden structure can be impassable
  to the bot protocol policy used by existing bots.
- A visible enemy with army `0` can still be a deathtouch runner after growth.
  From turn 800, use ownership and route distance as well as current army.
- Do not move from the general to attack an adjacent enemy after turn 800 when
  the move creates a head-on touch risk. Prefer a third-cell chase.
- If both an enemy-general capture and an own-general emergency exist before
  turn 800, choose the legal enemy-general capture because it ends the match.
- If the general has one army, reinforcement has priority over all normal
  expansion.
- If all legal-looking moves weaken defense, pass.
- Do not build castles in the first implementation. Castle economy changes the
  reserve model and requires a separate experiment.

## Expected behavior against existing bots

### Versus `smoke`

`smoke` expands to the first visible neutral and then takes the first legal
move. `garrison` should lose less often to accidental deep routes because
`garrison` keeps army near the general. `garrison` can own less land because
the reserve does not expand.

### Versus `expand_plus`

`expand_plus` sends large stacks toward capturable frontier cells. `garrison`
should detect these visible stacks and pull reinforcement inward. The main risk
is that `expand_plus` gains enough land that `garrison` cannot replace losses.

### Versus `castle_builder`

`castle_builder` spends early army on economy. `garrison` should take safe
frontier land during this window. `garrison` must not attack a castle with a
reserve stack unless the capture is clearly legal.

### Versus `general_hunter`

`general_hunter` attacks a known general after turn 800. `garrison` should keep
intercept stacks on the last two route cells and capture the runner source.
Army on the general alone is not sufficient.

## Parameter revision 1

Round 1 produced `0` wins, `3` losses, and `7` draws in `10` games. Both games
against `smoke` and the game against `expand_plus` reached the 1,200-turn
limit. The losses occurred against `army_convey` on turn 461, `late_rush` on
turn 477, and `phase_switch` on turn 1,089. These results show that the first
parameters preserve the general in passive games, but do not convert that
security into wins and do not stop all timed attacks.

Use these revisions for the next measurement:

- Keep the four defensive phases and the no-castle rule.
- Change the phase reserve floors from `8`, `16`, `24`, and `12` to `8`, `14`,
  `20`, and `10`.
- Keep the safe radii and threat scan radii unchanged.
- Change the visible-pressure divisor in `required_general_army` from `4` to
  `3`. This change gives visible pressure more weight while the lower base
  floors release army when no threat is visible.
- Keep recent enemy entries for `30` turns instead of `20` turns.
- Permit the local counterattack in phases 2 and 3 when a visible enemy stack
  can reach the general in `3` moves instead of `2` moves.
- Increase the neutral-plain base value from `30` to `45`.
- Increase the new-frontier bonus from `80` to `120`.
- Reduce the safe-radius source penalty from `-500` to `-300` only when no
  visible or recent threat exists inside the phase threat scan radius.
- Keep the reserve check, defense-deficit rejection, route denial, and
  deathtouch interception rules unchanged.

This revision must remain a defensive strategy with careful expansion. It must
not add a fixed commitment turn, a main rush stack, or an irreversible attack
phase.

## Experiment hypothesis

The primary hypothesis is:

> A phase-based general reserve plus route-aware reinforcement reduces losses
> against bots that reach the home area, without increasing the draw rate by
> more than 20 percentage points against passive expanders.

Use seeds `0–9` against `smoke`, `expand_plus`, `castle_builder`, and
`general_hunter`. Run both seat orders under `--mode competition`. Store every
game before rating updates.

Report:

- Win, loss, and draw counts by opponent.
- Mean match turns.
- General capture turn for every loss.
- Number of turns with a positive defense deficit.
- Final land and army if telemetry supports these fields.
- Elo change from the same fixed seed grid.

Keep the strategy if it has fewer losses than `expand_plus` against the active
opponents and does not reduce total wins. If every game draws, classify the
result as unproven and add a deliberate raider opponent before changing the
thresholds.
