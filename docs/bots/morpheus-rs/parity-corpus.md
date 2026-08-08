# Morpheus-rs parity corpus

The recorded frames a Rust port is checked against. Produced at milestone M0 of
the [rewrite plan](rewrite-plan.md); consumed from M1 onward by the parity
subcommands the plan's §5 describes.

A frame answers one question: *given exactly what the Python bot saw, and
exactly the random numbers it drew, does the Rust bot do the same thing?*
Everything in the format exists to make that question answerable without
running Python.

## Producing one

```bash
python scripts/morpheus_rs_baseline.py capture --games 20 --control-games 4
```

That plays instrumented competition games with the capture armed, then a few
uncaptured control games over the same seeds. Output is derived data under
`data/morpheus/morpheus-rs/<round>/`, one `<game_id>.capture.<seat>.jsonl.gz`
per game plus a `manifest.json` naming the oracle hash, host, and schedule.

```bash
python scripts/morpheus_rs_baseline.py report     # the measurement report
python scripts/morpheus_rs_baseline.py fixtures   # the committed smoke slice
```

The full corpus is tens of megabytes and never enters git. What ships is
`bots/morpheus-rs/tests/fixtures/parity-smoke.jsonl.gz` — a couple of frames
per stratum, enough for CI to catch a port that broke a whole surface, not
enough to prove parity. Proving parity is what re-running the full corpus
before an M5/M6 gate is for.

## How capture attaches

Nothing in `bots/morpheus/` is edited for this. The capture module
`bots/morpheus-rs/tools/capture_morpheus.py` is loaded by
[`arena.instrument.runner`](../../arena/trajectories.md), which already stands
in for `run.sh` on recorded matches, and which loads capture modules by file
path under a private name — so unhashed code stays unreachable from the hashed
program, exactly as with `probe.py`.

The module wraps two things and reads the rest:

- **The RNG.** `RecordingGenerator` delegates every draw to the real
  `numpy.random.Generator` and logs the result. The controller and the search
  share one generator, so both holders are pointed at the *same* proxy and the
  log is one interleaved stream.
- **The tensor builder.** `NetworkEvaluator._tensor` is wrapped to keep a
  reference to what it returned. Rebuilding the tensor after the move would be
  easier and wrong: `previous_action` has already advanced, so the rebuilt
  tensor is not the one the network saw.

Everything else — timings, priors, belief, chosen action — is read after the
reply is flushed, from state the turn left behind. Frames are written straight
to disk as they arrive, flushed per frame, because the matchup harness allows a
closing agent three seconds before SIGKILL.

## Why the draws are recorded rather than reproduced

Reimplementing NumPy's bit generator in Rust is possible, fragile, and useless
at play time. Instead the Rust bot's RNG is an injected trait: `Replay` over a
recorded stream in parity mode, an ordinary PRNG in play.

The consequence is deliberate. **Draw-site order becomes part of the ported
contract.** A Rust port that samples in a different order consumes the stream
differently and diverges visibly on the next decision — which is precisely how
parity mode finds control-flow bugs that array comparisons miss.

`rng_unrecorded` counts calls to generator methods the proxy does not record.
It must be empty. A non-empty count means the bot has started drawing through a
path the replay stream does not carry, and the corpus is no longer replayable.

## Strata

Every turn produces a frame. The heavy payload — belief particles, root
tensor, shaping scores, memory planes — rides every stride-th turn *within each
stratum* (default stride 8, `MORPHEUS_CAPTURE_STRIDE` to change it).

| stratum | what it covers |
| --- | --- |
| `first_move` | the grace-window turn: load, warmup, belief init |
| `first_contact` | the turn the enemy becomes visible |
| `pre_contact` | expansion phase, no enemy in sight |
| `post_contact` | enemy visible |
| `late_game` | turn ≥ 800, the deathtouch regime |
| `*+recovery` | any of the above with the belief update deferred |

Phase and recovery are **crossed, not ranked**. On a host where the deployment
is not qualified the belief update is deferred on most turns, so a rule that
ranked recovery first would file nearly every post-contact turn as recovery and
empty the post-contact stratum. Striding per stratum rather than over the game
is the same argument: a flat `turn % stride` samples each stratum in proportion
to how common it happens to be on the capture host, which is the bias the
stratification exists to remove.

## Frame format

Gzipped JSON lines, one object per turn. Arrays are encoded as
`{"__ndarray__": <base64 of the raw little-endian buffer>, "dtype": ..., "shape": [...]}`
— buffers, not nested lists, so they round-trip bit-exactly and a 49×21×21
float32 tensor costs 86 KB instead of half a megabyte. Read them with
`arena.instrument.capture.read_frames`.

### Every turn

| field | meaning |
| --- | --- |
| `t` | 1-based turn index in this seat's trajectory |
| `seat` | 0 or 1 — morpheus's player index |
| `stratum`, `heavy` | stratum label; whether the heavy payload is present |
| `contact` | enemy visible this turn |
| `obs` | the wire observation: `H`, `W`, `turn`, land/army scalars, and the `type`/`owner`/`army` grids |
| `action` | the five ints replied: `p r c d s` |
| `prev_action`, `recent_actions` | pre-move values the hard-rule layer saw. Mirrored by the capture, not read back: the controller overwrites both before the capture runs |
| `timing` | `move_ms`, the per-component `component_ms` and `component_calls`, and the `cost_*_ms` roll-ups |
| `counters` | completed simulations, forward equivalents, ESS, recovery flag, tree size, degradation level, belief size |
| `rng_draws` | every draw this turn: `{"m": method, "a": args, "r": result}`, in order |
| `rng_unrecorded` | generator calls that were *not* recorded; must be empty |
| `root` | `has_result`, the shaped `prior`, the `unshaped_prior`, and `tensors_built` |

### Heavy turns only

| field | meaning |
| --- | --- |
| `root.tensor` | the 1×49×21×21 float32 tensor the root evaluation ran on |
| `memory` | the `VisibleMemory` planes |
| `play_mask` | the legal mask after Morpheus's play rules |
| `shaping_scores` | `heuristic_action_scores` over that mask |
| `belief` | particles at full fidelity: state arrays, weights, enemy memory, enemy previous action, plus `ess` and `ess_fraction` |

Particle **histories** are recorded as their action pairs plus the oldest
recorded state, not as a state per frame: eight particles with an eight-deep
history is two thirds of a megabyte per turn. Replay reconstructs the
intermediate states through the transition kernel — which is the property M4's
rejuvenation gate is checking anyway.

## What the corpus does not settle

- **Timings are host-local.** `move_ms` and `component_ms` come from the
  agent's own clock on the capture machine. They are the M0 baseline, not a
  claim about the judge's host.
- **Captured play is not free.** The capture costs a few milliseconds per turn,
  which is why `capture` also plays uncaptured control games over the same
  seeds and the report quotes the gap.
- **Games are not reproducible from a seed.** Morpheus is deadline-driven, so
  the same seed and the same opponent give different games on a differently
  loaded machine. The corpus is a record of what happened, not a script that
  can be re-run to happen again — which is exactly why the frames are stored
  rather than regenerated.
