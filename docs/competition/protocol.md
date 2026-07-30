# Competition stdio protocol

Source of truth: `competition-module/competition/protocol.py`.

This protocol is **not** documented on DeepWiki. Use the submodule `competition/` tree.

## Handshake (engine → agent, once)

```text
<player_id> <H> <W>
```

## Per turn (engine → agent)

```text
<turn> <my_land> <my_army> <opp_land> <opp_army>
H lines of W ints   # type
H lines of W ints   # owner
H lines of W ints   # army
```

### Type codes

| Code | Meaning |
| --- | --- |
| 0 | fog |
| 1 | plain |
| 2 | mountain |
| 3 | castle |
| 4 | general |
| 5 | structure-in-fog |

### Owner codes (perspective-relative)

| Code | Meaning |
| --- | --- |
| 0 | neutral / unknown |
| 1 | me |
| 2 | opponent |

Army is `0` when not visible.

## Per turn (agent → engine)

```text
<pass> <row> <col> <dir> <split>
```

| Field | Values |
| --- | --- |
| pass | `0` move, `1` skip, `2` build castle at (row, col) |
| dir | `0` up, `1` down, `2` left, `3` right (ignored for pass/build) |
| split | `0` all-but-one, `1` half (ignored for pass/build) |

## Game end

The engine closes the agent's stdin. Treat EOF as game over and exit cleanly. No explicit end frame.

## Runner

Local matches: [`local-matchup.md`](../engine/local-matchup.md).
