# Strategy Specification: army_convey

This document specifies the design for `bots/army_convey/`.

## 1. Goal

The goal of `army_convey` is to funnel idle interior armies forward toward the expansion frontier.
It concentrates army force at frontier tips to enable sustained expansion breakthroughs.

## 2. Observation Features Used

The bot uses the following observation fields from `Observation`:

- `type_grid`: Identifies cell types (0 = fog, 1 = plain, 2 = mountain, 3 = castle, 4 = general, 5 = structure in fog).
- `owner_grid`: Identifies ownership (0 = neutral/unknown, 1 = me, 2 = opponent).
- `army_grid`: Identifies army counts on cells.
- `turn`: Current game turn number.
- `H`, `W`: Grid height and width.

The bot calculates derived spatial metrics each turn:

- `frontier_mask`: Owned cells adjacent to at least one passable unowned cell (neutral or opponent).
- `convey_distance_field`: Shortest path distance from every owned cell to the nearest frontier cell.

## 3. Scoring and Priority Rules

Actions follow a strict priority hierarchy each turn:

1. **Frontier Capture Execution**:
   - Evaluate adjacent capture moves originating from frontier cells.
   - Score capture moves using `army * 10.0` (with 2.0x multiplier for opponent cells).
   - Execute the highest scoring valid capture.

2. **Interior Army Conveyance**:
   - If no frontier capture is valid, identify interior owned cells (`owner == 1`, `army > 1`, not on frontier).
   - Calculate candidate convey moves from interior cells to adjacent owned cells that reduce `convey_distance_field`.
   - Score convey candidates by `army_size / (distance_to_frontier + 1)`.
   - Execute move from the highest scored interior stack.

3. **Frontier Gathering**:
   - Move interior armies neighboring frontier cells onto frontier cells to consolidate force.

4. **Fallback**:
   - Issue first valid legal move or `PASS`.

## 4. Pseudocode for act()

```text
FUNCTION act(obs):
    frontier_cells = FIND_FRONTIER_CELLS(obs)
    
    // Priority 1: Direct Frontier Capture
    best_capture = NULL
    best_capture_score = -1.0
    
    FOR EACH (r, c) IN frontier_cells:
        IF army_at(r, c) <= 1: CONTINUE
        FOR EACH neighbor (nr, nc) OF (r, c):
            IF NOT passable(nr, nc): CONTINUE
            IF owner_at(nr, nc) == 1: CONTINUE
            IF army_at(r, c) <= army_at(nr, nc) + 1: CONTINUE
            
            score = float(army_at(r, c)) * 10.0
            IF owner_at(nr, nc) == 2: score *= 2.0
            
            IF score > best_capture_score:
                best_capture_score = score
                best_capture = move(r, c -> nr, nc)
                
    IF best_capture IS NOT NULL:
        RETURN best_capture

    // Priority 2: Interior Army Conveyance
    q = DEQUE OF ALL frontier_cells
    dist = COMPUTED_BFS_DISTANCE_FIELD(obs, q, ONLY_THROUGH_OWNED_CELLS)
    
    best_convey = NULL
    best_convey_score = -1.0
    
    FOR EACH cell (r, c) OWNED BY me WITH army > 1:
        IF (r, c) IN frontier_cells: CONTINUE
        here_dist = dist[r][c]
        IF here_dist <= 0: CONTINUE
        
        FOR EACH neighbor (nr, nc) OF (r, c) OWNED BY me:
            IF dist[nr][nc] < here_dist:
                // Convey move reduces distance to frontier
                score = float(army_at(r, c)) / float(here_dist)
                IF score > best_convey_score:
                    best_convey_score = score
                    best_convey = move(r, c -> nr, nc)
                    
    IF best_convey IS NOT NULL:
        RETURN best_convey

    // Priority 3 & 4: Fallback
    RETURN first_valid_move(obs) OR PASS
```

## 5. Edge Cases

- **No Active Frontier**:
  - If the bot owns all passable map tiles or has no adjacent unowned tiles, `frontier_cells` is empty.
  - Distance field generation terminates cleanly, falling back to basic legal moves.

- **Disconnected Territory Islands**:
  - If owned territory is split into isolated clusters, BFS distance field restricts convey moves within each connected component.
  - Armies will not attempt impossible moves across unowned or mountain barriers.

- **Infinite Loops**:
  - Distance reduction requirement (`dist[neighbor] < dist[current]`) guarantees acyclic movement toward the frontier.

## 6. Differences from smoke and expand_plus

- `smoke` and `expand_plus` allow interior armies to sit idle when no adjacent unowned cell is capturable.
- `expand_plus` uses a global BFS toward unowned cells for the single largest stack only when no capture exists anywhere.
- `army_convey` continuously funnels all interior stacks along distance-gradient pipelines directly to active frontier tips.

## 7. Experiment Hypothesis

Funneling interior armies along convey paths to the frontier increases total army force at expansion tips, yielding higher total captured land and higher winrates against static expanders.
