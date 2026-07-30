# Late rush strategy specification

## Goal

`late_rush` expands early, combines army into one main stack, and commits that
stack toward the opponent between turns 650 and 800. The attack seeks a normal
general capture before turn 800 or a deathtouch contact from turn 800.

The bot must use only the competition observation and its own memory. The bot
must use the competition stdio action format. The bot must finish each decision
inside the 150 ms move limit.

## Strategy features

1. Expand quickly during the early game.
2. Remember every confirmed enemy cell and the enemy general position.
3. Track a frontier target even when fog hides the target later.
4. Select one rally cell and one main stack before the commitment phase.
5. Route large outer stacks toward the rally cell through own territory.
6. Stop low-value expansion before the commitment turn.
7. Start the rush early when the enemy general is known or visible enemy
   pressure makes waiting unsafe.
8. From turn 800, prioritize any legal move onto the known enemy general over
   normal combat calculations.
9. If the enemy general is unknown at turn 800, probe through opponent land and
   recently seen enemy routes instead of waiting.
10. Keep a small home reserve. Do not move the own general stack into the rush.

## State

The implementation must keep this state:

- `own_general_pos`: the fixed own general position.
- `enemy_general_pos`: the fixed enemy general position after a sighting.
- `enemy_memory`: the most recent turn, army, and type for each confirmed enemy
  cell.
- `rally_pos`: the current rally cell.
- `main_stack_pos`: the expected position of the rush stack after the last
  issued move.
- `commit_started`: whether the bot entered the commitment phase.
- `commit_turn`: the turn on which commitment started.
- `target_pos`: the current attack target.
- `target_kind`: one of `enemy_general`, `visible_enemy`, `remembered_enemy`,
  `frontier_probe`, or `unknown`.
- `stalled_turns`: consecutive turns in which the main stack did not reduce
  its route distance to the target.

The bot must validate stored positions against each new observation. If
`main_stack_pos` is no longer owned, select the largest suitable owned stack.
If a remembered enemy cell becomes visible and is no longer enemy-owned, remove
or update that memory.

## Phases and thresholds

Use named constants. The initial values below form the first experiment.

### Phase 1: expansion, turns 1–449

- Expand onto visible neutral land.
- Attack a visible opponent cell when the capture is legal and the source is
  not the own general.
- Keep `8` army on the own general when possible.
- Prefer captures that increase the frontier and keep the army moving away from
  the own general.
- Do not build castles.
- Do not select a permanent rally cell.

### Phase 2: rally selection, turns 450–599

- Continue high-value expansion.
- Select a rally cell every 25 turns or when the current rally cell is invalid.
- Prefer an owned cell near the enemy-facing frontier, but keep the rally cell
  at own-territory distance at least `4` from the own general.
- Move non-frontier stacks toward the rally cell when no strong capture exists.
- Keep `10` army on the own general when possible.
- Start to reject neutral captures that move away from the rally cell.

### Phase 3: accumulation, turns 600–699

- Stop normal expansion from the largest stack.
- Define the largest eligible stack as the main stack.
- Move supporting stacks toward the rally cell.
- Move the main stack toward the rally cell if the main stack is elsewhere.
- Permit only high-value expansion by secondary stacks.
- Keep `12` army on the own general when possible.
- Start commitment at turn `650` if the enemy general is known.
- Start commitment at turn `675` if a visible or remembered enemy route gives
  a stable target.
- Otherwise, start commitment at turn `700`.

### Phase 4: committed rush, turns 700–799

- Do not switch back to economy or broad expansion.
- Move the main stack one shortest-path step toward the selected target.
- Capture opponent cells on the route when legal.
- Use all-but-one moves.
- Secondary stacks can reinforce the route behind the main stack.
- Recompute the target when the route is blocked or when a better target
  appears.
- Attack the visible enemy general immediately when a normal capture is legal.
- Do not attack the visible enemy general before turn 800 when the sent army is
  not strictly greater than the visible garrison.

### Phase 5: deathtouch rush, turns 800–1200

- If the enemy general is known, make contact the highest priority.
- A source with at least `2` army can win by moving onto the enemy general.
- Ignore the visible general garrison size for the contact move.
- Prefer multiple small runners near the target over one delayed reinforcement
  when the main stack fragments.
- If the enemy general is unknown, target the closest visible opponent cell.
- If no opponent is visible, target the freshest remembered enemy cell.
- If no enemy memory exists, probe from the enemy-facing frontier.
- Do not retreat because the main stack is smaller than a non-general defender.
  Search for a route around that defender first.
- After 12 stalled turns, choose a new frontier probe target.

## Derived map data

The implementation must compute these fields when required:

- `passable_distance(target)`: BFS over cells that are not known mountains or
  fogged structures.
- `owned_distance(target)`: BFS over cells currently owned by the bot.
- `frontier_cells`: owned cells adjacent to a visible neutral, fog cell, or
  opponent cell.
- `enemy_facing_score`: a score that estimates how close a frontier cell is to
  confirmed enemy activity.
- `support_distance`: own-territory distance from a supporting stack to the
  rally cell or current main stack.

Unknown fog can be part of an attack route after commitment. The bot must not
call the fog cell a legal capture target before it confirms that the engine
accepts movement into that cell. The candidate still must not have `type == 5`.

## Rally-cell selection

Score each owned frontier cell:

`rally_score = 20 * enemy_facing_score + 4 * local_support - 3 * general_risk`

Definitions:

- `enemy_facing_score = 20 - min(20, distance_to_best_enemy_target)`.
- If no enemy target exists, use distance from the own general and prefer a
  frontier cell far from the own general.
- `local_support` is the total movable army on owned cells within owned-distance
  `4`, capped at `50`.
- `general_risk = max(0, 4 - owned_distance_to_general) * 20`.

Add:

- `+200` when the rally cell is on a route to the known enemy general.
- `+100` when the rally cell is next to a visible opponent cell.
- `-150` when the rally cell is a dead end with only one known passable
  neighbor.
- `-300` when the rally cell is the own general.

Use deterministic tie-breaking by larger local support, then smaller row and
column.

## Main-stack selection

An eligible main stack:

- Belongs to the bot.
- Has at least `12` army before turn 700.
- Is not the own general.
- Is not inside own-territory distance `2` from the own general.

Score each eligible stack:

`main_score = 10 * army - 5 * distance_to_rally - 2 * distance_to_target`

Add `+300` to the previously selected main stack when it is still valid. This
prevents the main-stack identity from changing for a small army difference.

If no stack has `12` army, use the largest eligible stack with at least `2`
army. If no eligible stack exists, delay commitment for at most 25 turns while
secondary stacks consolidate. At the end of the delay, select the largest
non-general stack.

## Target selection

Choose one target in this priority order:

1. Known enemy general.
2. Visible opponent general.
3. Visible opponent castle.
4. Visible opponent cell with the highest target score.
5. Freshest remembered opponent cell.
6. Frontier probe cell.

For a visible opponent cell:

`target_score = 8 * type_value + 4 * enemy_army + 3 * recency - 5 * route_distance`

Use:

- `type_value = 1000` for the enemy general.
- `type_value = 20` for an enemy castle.
- `type_value = 5` for another enemy cell.
- `recency = max(0, 20 - age_in_turns)`.

Before turn 800, reject a target when no route can bring enough visible army to
capture the target. This rule does not reject intermediate opponent cells that
the main stack can capture.

After turn 800, route distance to the known enemy general has more value than
all army terms.

If no enemy cell is known, select a frontier probe:

- Prefer the frontier cell with the greatest BFS distance from the own general.
- Prefer a frontier cell that is near the last visible opponent direction.
- Prefer a frontier cell with at least two passable exits.
- Reject a frontier cell inside distance `3` of the own general.

## Candidate move scoring

Generate legal-looking moves from owned sources with at least `2` army.
Reject off-board destinations, mountains, and fogged structures.

### Early expansion score

For turns before 600:

`score = 50 * capture_value + 8 * source_army - 6 * defender_army`

Use:

- `capture_value = 1` for a neutral plain.
- `capture_value = 4` for an opponent cell.
- `capture_value = 20` for an opponent castle.
- `capture_value = 10000` for an opponent general that is legally capturable.

Add:

- `+100` for a destination that has at least two new frontier neighbors.
- `+60` for movement toward the current best enemy target.
- `+40` for movement away from the own general during turns 1–449.
- `-1000` for a source that is the own general and would break the home reserve.
- `-150` for movement into a dead end.
- `-80` for movement away from the rally cell during turns 450–599.

### Consolidation score

For turns 450–699, a supporting move must move one own-territory BFS step
toward the rally cell or main stack.

`score = 2000 * distance_reduction + 20 * transferred_army`

Add:

- `+1000` when the destination is the rally cell.
- `+500` when the destination is the current main-stack cell.
- `-1200` when the source is an exposed frontier cell next to an opponent.
- `-5000` when the source is the own general and the move breaks the reserve.

The implementation must avoid two stacks that exchange positions on
alternating turns. The distance to the fixed rally cell must decrease.

### Rush score

After commitment:

`score = 10000 * target_distance_reduction + 20 * source_army`

Add:

- `+1000000` for a move onto the enemy general from turn 800.
- `+500000` for a legal normal capture of the enemy general before turn 800.
- `+50000` for a move by the current main stack.
- `+10000` for capture of an opponent cell on a shortest route.
- `+5000` for capture of an opponent castle on a shortest route.
- `+2000` for movement from a support stack toward the main stack.
- `-30000` for movement by the own general.
- `-20000` for a move that increases target distance.
- `-10000` for changing to a different main stack without invalidation.

Before turn 800, reject an attack when the sent army is not strictly greater
than the visible defender. From turn 800, keep this rejection for normal cells
but remove it for the known enemy general.

## Pseudocode for `act()`

```text
act(observation):
    locate and store own general
    update enemy-general position and enemy-cell memory
    validate rally, main-stack, and target state
    compute current phase from turn and commit state

    if turn >= 800 and enemy general is known:
        contact_moves = legal moves onto enemy general
        if contact_moves is not empty:
            return best contact move

    if not committed:
        if turn >= 450:
            select or refresh rally cell
        if turn >= 600:
            select or validate main stack
        if early-commit condition is true or turn >= 700:
            set commit_started and commit_turn

    if committed:
        select target by target priority
        compute passable distance field from target
        score moves by rush rules
        if a main-stack move reduces target distance:
            reset stalled turns
            return that move
        if a support move advances the rush:
            increment stalled turns when target distance did not change
            return that move
        increment stalled turns
        if stalled turns >= 12:
            select a new target and clear stalled count

    if turn is 450 through 699:
        score consolidation and high-value expansion
        prefer consolidation after turn 600
        if a candidate exists:
            return highest-scoring candidate

    score early expansion candidates
    if a candidate exists:
        return highest-scoring candidate

    score owned-cell movement toward rally or frontier
    if a candidate exists:
        return highest-scoring candidate

    return pass
```

Use deterministic tie-breaking:

1. Higher score.
2. Move by the current main stack.
3. Greater target-distance reduction.
4. Larger source army.
5. Smaller source row, source column, and direction code.

## Edge cases

- If the own general is not visible, pass until the bot finds it. Do not choose
  a main stack without excluding the own general.
- If no enemy information exists, use a frontier probe. Do not guess an exact
  enemy general position.
- If the remembered enemy general is under fog, keep the position. Generals do
  not move.
- If the remembered enemy general cell becomes visible as owned by the bot, the
  match should already end. Do not clear the target during the last action.
- If the main stack is destroyed, captured, or split by combat, select a new
  main stack on the next turn.
- If the shortest path enters a known mountain after fog changes, recompute the
  route immediately.
- If the route enters `type == 5`, treat that cell as blocked and find another
  route.
- If a visible normal defender is too large before turn 800, route around the
  defender or wait for support. Do not issue an invalid capture.
- From turn 800, a known enemy general with a large garrison is still a valid
  target for a source with army at least `2`.
- Do not repeatedly move support stacks into and out of the rally cell. A
  support move must reduce distance to a fixed destination.
- Do not build castles in the first implementation. A castle consumes the
  accumulation stack and delays commitment.
- If the own general is under immediate visible threat, permit one emergency
  capture or reinforcement. Resume commitment on the next turn. Do not clear
  `commit_started`.
- If both generals can be captured before turn 800, choose the immediate legal
  enemy-general capture.
- If no move can advance or reinforce the rush, make a safe frontier probe.
  Pass when every move weakens the own general reserve.

## Expected behavior against existing bots

### Versus `smoke`

`smoke` expands without a planned attack. `late_rush` should use the quiet
early game to accumulate one larger stack. The committed stack should cross the
shared frontier instead of stopping at neutral captures.

### Versus `expand_plus`

`expand_plus` sends its largest stacks toward nearby capturable cells.
`late_rush` can lose support stacks during accumulation. The rush must start by
turn 700 even when the ideal rally size is not available. A delayed perfect
stack has less value than a runner that reaches the enemy area near turn 800.

### Versus `castle_builder`

`castle_builder` removes army from movement to build castles. `late_rush`
should attack before the castle production repays the cost. Enemy castles are
useful route targets because they identify developed enemy territory.

### Versus `general_hunter`

Both bots use deathtouch. `general_hunter` waits for an enemy-general sighting
before its direct beeline. `late_rush` should gain the first sighting through
the committed probe. `late_rush` has more home risk because its main army moves
away from the own general.

## Parameter revision 1

Round 1 produced `8` wins, `2` losses, and no draws in `10` games. The strategy
beat `smoke` on turns 685 and 683, beat `expand_plus` on turn 554, and beat
`garrison`, `splitter`, `choke_control`, `phase_switch`, and `castle_rush` on
turns 477 through 809. The losses against `army_convey` on turn 480 and
`fog_scout` on turn 509 occurred before the default commitment window. These
results support the timed-rush identity, but they show that the first
accumulation schedule can wait too long against fast pressure.

Use these revisions for the next measurement:

- Start rally selection at turn `400` instead of turn `450`.
- Start accumulation at turn `550` instead of turn `600`.
- Start commitment at turn `625` when the enemy general is known.
- Start commitment at turn `650` when a visible or remembered enemy route gives
  a stable target.
- Start commitment at turn `675` in all other cases instead of turn `700`.
- Change the home reserve floors from `8`, `10`, and `12` to `10`, `12`, and
  `12` for expansion, rally selection, and accumulation.
- Change the minimum eligible main-stack army from `12` to `10`. Keep the
  25-turn maximum consolidation delay.
- Refresh the rally cell every `20` turns instead of every `25` turns.
- Change the stalled-target threshold from `12` turns to `8` turns after
  commitment.
- Permit the one-move emergency defense before or after commitment when a
  visible enemy can reach the own general in at most `3` moves. Resume the same
  commitment on the next turn.
- Keep all deathtouch contact priorities and the irreversible commitment rule
  unchanged.

This revision must remain a timed commitment strategy. It must not keep
reinforcement near the own general after the immediate emergency ends, and it
must not return to broad expansion after commitment.

## Experiment hypothesis

The primary hypothesis is:

> A fixed accumulation window followed by an irreversible turn-700 rush
> produces more decisive games and more wins than `expand_plus`, especially
> after deathtouch begins.

Use seeds `0–9` against `smoke`, `expand_plus`, `castle_builder`,
`general_hunter`, and `garrison` after `garrison` exists. Run both seat orders
under `--mode competition`. Store every game before rating updates.

Report:

- Win, loss, and draw counts by opponent.
- Mean match turns.
- Commitment turn.
- First enemy sighting turn.
- Enemy general sighting turn.
- Main-stack army at commitment.
- Minimum route distance to the enemy general after a sighting.
- Wins before turn 800 and wins from turn 800.
- Elo change from the same fixed seed grid.

Compare against `expand_plus` on the same seeds and seat orders. Keep the
strategy if it reduces draws and does not reduce wins. If the rush reaches
enemy territory but fails before turn 800, test the commitment threshold as a
separate experiment. If the rush never reaches enemy territory, test rally and
path scoring before changing deathtouch behavior.
