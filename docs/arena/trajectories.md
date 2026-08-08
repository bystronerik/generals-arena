# Per-turn trajectories

Recording what happened *inside* a game, beside the game record rather than in
it. Off by default; nothing in the rating path knows trajectories exist.

Two channels answer two different questions:

| Channel | Written by | Answers |
| --- | --- | --- |
| **engine trajectory** | the match loop | what actually happened — actions, land/army per turn, the outcome |
| **probe trace** | `arena.instrument.runner` | what a bot believed — phase, sightings, counters |

The engine channel is canonical: it alone carries the replay guarantee. A trace
is an attachment to it. Neither substitutes for the other — engine truth cannot
say what a bot believed, and a bot's belief is not ground truth (its view of the
opponent is fog-limited).

## Files

```
data/trajectories/<round>/<game_id>.traj.jsonl.gz      engine, canonical
data/trajectories/<round>/<game_id>.trace.a.jsonl.gz   probe, seat A
data/trajectories/<round>/<game_id>.trace.b.jsonl.gz   probe, seat B
```

Keyed by `game_id`, so a trajectory joins to its `GameRecord` by name and
nothing else. Gitignored — derived data, like `data/games/`.

Traces appear only for seats whose bot has a probe. `cm_*` bots never have one.

## Engine trajectory format

Line-oriented; `v` guards the shape.

```jsonl
{"v":1,"game_id":"…","seed":7,"mode":"competition","round":"…","engine_version":"…","bot_a":"…","bot_b":"…","H":19,"W":21}
{"t":1,"a":[0,3,4,1,0],"b":[1,0,0,0,0],"land":[2,1],"army":[12,11]}
{"t":100,"digest":"sha256:…"}
{"end":{"winner":"a","turns":412,"terminated":true,"truncated":false}}
```

- `a` / `b` are the decoded actions the engine **applied**, invalid moves
  included. Resolving those is the engine's job; a trajectory that dropped them
  would not replay the same game.
- `land` / `army` are the engine's own per-turn totals, both seats, fog-free.
- `digest` every 100 turns and at the end: sha256 over `armies`, `ownership`,
  and `castles`. It turns "the replay diverged" into "the replay diverged in
  this century".
- Trace lines are `{"t": N, …probe dict}`, one per turn.

**States are not stored.** `(seed, engine_version, actions)` determines every
state the game ever had, so storing them would be caching, not recording — and
it is the difference between ~12 KB and ~3 MB a game. States come back by
replay; either seat's fog view comes back from a state through
`get_observation`.

Writes are atomic (`.tmp` then `os.replace`) and each worker writes only its own
game's files, so the process pool needs no lock and a killed worker leaves no
half-readable trajectory.

## Probes

A probe is `bots/<name>/probe.py`:

```python
def extras(agent) -> dict:
    return {"phase": agent._core.memory.phase}
```

Rules that make this safe:

- **Passive.** It reads attributes and returns them. It must never mutate the
  agent or influence play.
- **Outside the closure.** `arena.records.fingerprint` excludes the fixed name
  `probe.py`, so adding or editing a probe moves no hash and reaches no
  submission bundle. Only `arena.instrument.runner` ever loads it.
- **Never imported by the agent.** `fingerprint` raises `ProbeInClosureError`
  if any module in the closure imports `probe` — unhashed code must be
  unreachable from the hashed program.
- **Declared keys only.** Every key needs an entry in
  `arena/records/telemetry_schema.py` (kind, meaning, reducers). An undeclared
  key raises and fails the recorded match; that is intended, since probe and
  schema live in one repo.
- **Fails loudly.** A probe that reads an attribute an agent refactor renamed
  aborts the recorded match rather than logging a partial trace that looks like
  evidence.

## Captures

A **capture module** is the heavy sibling of a probe: instead of a handful of
declared scalars it writes whole arrays per turn — observations, belief
particles, network tensors, RNG draw logs — to
`<game_id>.capture.<seat>.jsonl.gz` beside the trajectory. Nothing it writes
reaches `GameRecord.metrics` or the rating fit; a capture is a side file for
one analysis. See `arena/instrument/capture.py`, and
[the morpheus-rs parity corpus](../bots/morpheus-rs/parity-corpus.md) for the
case it was built for.

The same safety rules apply — loaded by path under a private name, outside
every bot's closure, errors never swallowed — with three differences that
follow from the size and the purpose:

- **Untyped.** Its consumer is a specific analysis, not the telemetry schema,
  so there is no key registry to declare against.
- **Opt-in per run.** Armed by exporting `ARENA_CAPTURE_MODULE`; unset, a
  recorded match behaves exactly as before.
- **May wrap, not only read.** A capture may proxy agent internals — the
  morpheus one wraps the RNG to log draws — strictly to observe them. Frames
  are written as they arrive rather than buffered to exit, because a whole
  game's tensors do not serialize inside the three seconds the matchup harness
  gives a closing agent.

## From series to metrics

On a recorded game, each key's declared reducers collapse its series into
`GameRecord.metrics`:

| Reducer | Metric key | Meaning |
| --- | --- | --- |
| `final` | `<key>_a` | last frame |
| `mean` | `<key>_mean_a` | arithmetic mean over recorded turns |
| `max` | `<key>_max_a` | maximum |
| `argmax_turn` | `<key>_argmax_turn_a` | 1-based turn of the first maximum |
| `auc` | `<key>_auc_a` | sum over turns |
| `first_turn_true` | `<key>_first_turn_a` | first turn the predicate held; **absent** if it never did |

Truthiness for `first_turn_true` comes from the key's kind, not from Python:
`BOOL01` is `== 1`, `INT` is `> 0` (so a negative `land_margin` means behind,
not "true"), `TOKEN` is non-empty.

Unrecorded games carry engine finals and no probe metrics at all — missing
means not measured, never zero.

## Retention

Manual. A round's trajectory directory lives and dies with the experiment that
recorded it: delete it once the experiment note is published, keep it while a
tuning effort or the RL work references it. No auto-GC; revisit when
`data/trajectories/` first crosses ~2 GB.

Game records under `data/games/` are kept regardless — deleting a trajectory
never costs a rated game.

## Scope

Only the competition path records. `arena/matches/classic.py` and everything
under `arena/remote/` construct no recorder and always spawn `run.sh`, the same
fencing that keeps their games out of `data/games/`. Classic traces, if ever
wanted, would be a separate effort with its own `data/classic_trajectories/`
root.

## Related

- [game-record-schema.md](game-record-schema.md)
- [match-runner.md](match-runner.md)
- [tournament.md](tournament.md)
- [`docs/bots/adding-a-bot.md`](../bots/adding-a-bot.md)
