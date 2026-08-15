# Morpheus-rs telemetry

How the Rust bot reports what a turn cost, and why it has to report it itself.

Added at milestone M6 of the [rewrite plan](rewrite-plan.md), which needs it:
success criteria 2 and 4 are a latency comparison and a first-move budget, and
neither is measurable without a per-turn record from inside the bot.

## Why not the arena's probe

The Python sibling is traced from outside. `arena.instrument.runner` stands in
for `bots/morpheus/run.sh`, constructs `Agent` **in-process**, and samples
`bots/morpheus/probe.py` after every move. None of that reaches a subprocess
binary: there is no object to read attributes off, and the wire carries five
integers.

So this bot carries its own probe. Plan §7 said `morpheus-rs` "gets its own
arena-owned probe later"; M6 is later.

## Shape

```bash
MORPHEUS_RS_TRACE=/tmp/seat.jsonl bash bots/morpheus-rs/run.sh
MORPHEUS_RS_TRACE=/tmp/traces/seat-%p.jsonl python -m arena.tournaments.competition ...
```

One JSON object per turn, in `crates/core/src/runtime/telemetry.rs`. Unset — which is
every rated game — the whole mechanism is one `Option` check per turn and no
disk at all.

Three properties are copied from the arena's path rather than invented:

- **The same keys.** Every field is a field of `probe.py`'s `extras`, so one
  reducer reads both bots and a ratio between them is a ratio of the same
  quantity. Two additions ride along, ignorable by anything that does not know
  them: `components` (the per-component milliseconds `deployment.json`'s
  `offline_p99_ms` is fitted from), `calls` (how many times each of the ten
  ran), `forward_by_consumer`, and, on the first line only, `load_ms`,
  `warmup_ms` and `init_ms`.
- **`t` is the turn index, not `obs.turn`.** The observation is numbered before
  the step, so using it would offset every trace by one against the engine
  trajectory a reducer aligns it with.
- **Buffered to the end.** The runner writes its trace in a `finally` because a
  per-turn flush would put file IO on the move path — which is exactly what
  must not perturb the game being measured. Telemetry that changes the
  measurement is worse than none.

`%p` in the path becomes the process id. A tournament runs many seats from one
environment, and without it they would all write one file — which looks like a
trace and is a race.

## What the numbers mean, and one trap

`components` is a per-turn **total**, not a per-call cost, and the two are not
the same comparison. The search components run once per simulation, and this
bot completes more simulations per turn than the Python does, so a
total-versus-total ratio charges it for the extra work it managed to fit.
`scripts/morpheus_rs_m6.py` normalizes the three whose call count the trace
carries — `selection`, `leaf_batch`, `enemy_prior_batch` — and the difference
is not cosmetic: `enemy_prior_batch` reads 0.9× per turn and 1.6× per call.

`forward_equivalents` counts forwards; `forward_by_consumer` says who spent
them, across the four names in `runtime/config.rs`'s `FORWARD_CONSUMERS` —
`belief_proposal`, `root`, `enemy_prior`, `leaf_batch`. The controller has
tracked the breakdown since the port and only the total was ever written out.
It matters whenever a change removes a consumer rather than making one faster:
[N0](../../research/measurements/joe-net-n0.md) needed to know that around a
quarter of the shipped bot's forwards go to enemy priors before it could say
what a four-times-slower network leaves for the search.

The trap is the other direction. M0's aggregation walks the Python's
`component_ms` **dict**, which only holds a key for a component that was
actually timed, so its p99 for anything that does not run every turn is taken
over the turns it ran. The Rust side carries all ten slots on every turn.
Counting those zeros compares two different questions — `belief_tensor` is a
first-move cost in both bots, and against 10,974 zeros its p99 is 0 while
against its own 20 values it is 0.13 ms. The M6 report filters to non-zero
turns for that reason; a future reducer should do the same or emit a call count
for all ten components.

## Asking what the controller would do at costs it cannot produce

`deployment.json` has two more keys, both absent from every rated file:

| key | effect |
| --- | --- |
| `fixed_forecasts_ms` | per-component forecasts that override the estimators |
| `charge_fixed_forecasts` | advance a virtual clock by the forecast instead of timing the real work |

Together they run the **real** admission controller — the real `can_admit`, the
real widening freeze, the real degrade bands — over injected component costs.
`RuntimeController` has carried both fields since the port and only its tests
could reach them; the parse makes the shipped binary answer the question, which
is the difference between measuring the controller and modelling it.

Three things to know before reading a spike's numbers:

- **`move_ms` is virtual.** Under a charged clock it is the sum of the
  forecasts, not wall time. Nothing in a spike's trace bounds what the process
  actually spent.
- **A component forecast at `0.0` is modelled as deleted, not as free.** It
  always admits and never advances the clock. The work still happens, so the
  tree it builds is the one the real component builds — which is the right
  model for "this consumer goes away" and the wrong one for "this consumer got
  faster".
- **It announces itself on stderr, and the trace's first-line `config` carries
  `spike` and `charged`.** A bot playing forecast costs instead of measured ones
  is a different bot, and it must never be one that looks the same.

Used by [N0](../../research/measurements/joe-net-n0.md), which had to know how
many simulations survive a 21 ms forward before porting a line of the network
that costs it.

## The thread-count invariant

Plan §10 asks for the Rust analog of
`test_play_and_calibration_pin_the_same_thread_count`. `telemetry::thread_count`
reads `/proc/self/status`, and `main.rs` checks it **after warmup** — the point
where a dependency's lazily-started pool would exist.

Three deliberate choices:

- **Linux only.** macOS has no `/proc` and reading its thread count needs
  `libc`, a dependency this zero-dependency crate will not take for a check
  that does not run where it matters. `None` means "not asked", never "passed".
- **Refusal degrades to passing, not to exiting.** The judge forfeits a game on
  an early exit but charges one fault out of fifty for a bad reply, so a
  refusal that costs the match is a worse answer than one that costs the game
  and says why on stderr. Same trade the artifact loader already makes.
- **After warmup, not before.** Before warmup the check would pass on a bot
  whose pool starts on the first forward.

## Related

- [parity-harness.md](parity-harness.md) — how the port is proved equal
- [`bots/morpheus/probe.py`](../../../bots/morpheus/probe.py) — the key set this mirrors
- [morpheus-rs-m6-latency.md](../../research/measurements/morpheus-rs-m6-latency.md) — what it measured
