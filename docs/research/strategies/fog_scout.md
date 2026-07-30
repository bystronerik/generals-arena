# Strategy Specification: fog_scout

This document specifies the design for `bots/fog_scout/`.

## 1. Goal

The goal of `fog_scout` is to probe fog of war tiles systematically.
It reveals unknown map area and locates the enemy general early while expanding territory.

## 2. Observation Features Used

The bot uses the following observation fields from `Observation`:

- `type_grid`: Identifies cell types (0 = fog, 1 = plain, 2 = mountain, 3 = castle, 4 = general, 5 = structure in fog).
- `owner_grid`: Identifies ownership (0 = neutral/unknown, 1 = me, 2 = opponent).
- `army_grid`: Identifies army counts on visible cells.
- `turn`: Current game turn number.
- `H`, `W`: Grid height and width.

The bot maintains state across turns:

- `ever_seen_grid`: A 2D boolean grid tracking cells revealed at any point during the match.

## 3. Scoring and Priority Rules

Actions follow a strict priority hierarchy each turn:

1. **Immediate Enemy General Attack**:
   - Execute an attack onto a visible opponent general if owned army exceeds defender army + 1.

2. **Fog-Probing Expansion**:
   - Score adjacent moves from owned cells with `army > 1` onto unowned or fogged cells.
   - Base score: `army * 10.0`.
   - Apply `FOG_BONUS` (multiplier 3.0) if destination cell is currently fogged (`type == 0`) or never seen (`ever_seen == False`).
   - Apply `OPPONENT_BONUS` (multiplier 2.0) if destination cell is owned by opponent (`owner == 2`).

3. **Visible Neutral Expansion**:
   - Capture adjacent visible neutral plain cells if owned army exceeds cell army + 1.

4. **Fog Frontier March**:
   - If no adjacent capture is available, run multi-source BFS from all unrevealed fog cells.
   - Move the largest owned stack one step closer to the nearest fog cell along the shortest path.

5. **Fallback**:
   - Issue first valid legal move or `PASS`.

## 4. Pseudocode for act()

```text
FUNCTION act(obs):
    UPDATE ever_seen_grid WITH currently visible cells in obs

    // Priority 1: Enemy General Attack
    IF visible_enemy_general IS NOT NULL AND can_capture(visible_enemy_general):
        RETURN move_to(visible_enemy_general)

    // Priority 2 & 3: Direct Expansion (Fog + Neutral + Opponent)
    best_move = NULL
    best_score = -1.0

    FOR EACH cell (r, c) OWNED BY me WITH army > 1:
        FOR EACH neighbor (nr, nc) OF (r, c):
            IF NOT passable(nr, nc): CONTINUE
            IF army <= army_at(nr, nc) + 1: CONTINUE
            
            is_opp = (owner_at(nr, nc) == 2)
            is_fog = (type_at(nr, nc) == 0 OR NOT ever_seen_grid[nr][nc])
            is_neutral = (owner_at(nr, nc) == 0 AND type_at(nr, nc) NOT IN (0, 5))
            
            IF NOT (is_opp OR is_fog OR is_neutral): CONTINUE
            
            score = float(army) * 10.0
            IF is_fog: score *= 3.0
            IF is_opp: score *= 2.0
            
            IF score > best_score:
                best_score = score
                best_move = move(r, c -> nr, nc)

    IF best_move IS NOT NULL:
        RETURN best_move

    // Priority 4: Fog Frontier March via BFS
    q = DEQUE OF ALL (r, c) WHERE NOT ever_seen_grid[r][c] AND passable(r, c)
    dist = COMPUTED_BFS_DISTANCE_FIELD(q)
    
    march_move = FIND_BEST_STEP_TOWARD_MIN_DIST(obs, dist)
    IF march_move IS NOT NULL:
        RETURN march_move

    // Priority 5: Fallback
    RETURN first_valid_move(obs) OR PASS
```

## 5. Edge Cases

- **Complete Map Visibility**:
  - If all passable tiles are revealed (`ever_seen_grid` full), fog march distance calculation returns no targets.
  - The bot falls back smoothly to visible neutral expansion and standard frontier marching.

- **Mountain Traps**:
  - Fog tiles behind mountain walls are unreachable.
  - BFS pathfinding ignores impassable cells (`type == 2`) to prevent marching into dead ends.

- **Insufficient Army for Fog Entry**:
  - Stepping into fog requires at least 2 army units (leaving 1 unit behind).
  - If a fog cell contains a hidden neutral or opponent stack higher than attacker - 1, combat rules apply normally.

## 6. Differences from smoke and expand_plus

- `smoke` and `expand_plus` filter candidate destinations using `dest_owner == 0` and visible cell types, ignoring fogged cells (`type == 0`).
- `fog_scout` actively incentivizes moves into fogged cells with a score multiplier.
- `fog_scout` tracks persistent map vision history across turns to guide long-range scouting marches toward unrevealed terrain.

## 7. Experiment Hypothesis

Directing idle stacks toward fog frontiers reveals the enemy general earlier in the match compared to `expand_plus`, providing actionable targeting data before turn 800 deathtouch.
