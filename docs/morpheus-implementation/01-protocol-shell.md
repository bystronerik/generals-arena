# Part 01: Protocol shell

## Deliverable

Create `bots/morpheus/` with `agent.py`, `main.py`, and `run.sh`. The initial
agent returns the protocol-safe pass action.

This is the earliest runnable Morpheus-shaped process. It retires startup,
frame parsing, reply serialization, and EOF risk before training uses any
compute.

**Touches**

- `bots/`: add the Morpheus shell and bot-local tests.
- `arena/`: use the existing bundle tool without changing it.
- `scripts/`: none.
- `data/bot_versions/`: none. Do not rate or register the stub.

## Prerequisites

None.

## Source specifications

- [Competition protocol](../competition/protocol.md)
- [Runtime degradation](../bots/morpheus/runtime.md#degradation-path)
- [Competition gate](../bots/morpheus/evaluation.md#competition-verification-gate)
- [Unified bot API](../engine/unified-bot-api.md)

## Defaults and replacement measurement

This part has no open-question default. Pass is always legal under the
[action-space specification](../bots/morpheus/action-space.md#pass-and-normalization).

The shell does not claim to implement the Morpheus strategy. Later parts
replace the pass-only body.

## Implementation boundary

Copy the process shape from `bots/smoke/`. Keep `main.py` as a thin call to
`bots/_common/wire.run_stdio`.

The shell must exit with status 0 after stdin closes. It must write exactly one
five-integer reply for each complete observation and no protocol data to
stderr.

Do not add a model, transition, belief, search, training code, or telemetry to
the bot closure in this part.

## Isolated test

```bash
python -m pytest bots/morpheus/tests/test_protocol_shell.py \
  -m morpheus -q
```

```bash
python competition-module/competition/matchup.py \
  bots/morpheus/run.sh bots/smoke/run.sh \
  --mode competition --seed 0
```

```bash
python -m arena.bundle morpheus --force
```

## Specification gaps

The design requires an RL-trained policy, a WDL value, particles, and
simultaneous search. The shell has none of these properties. It is a degraded
protocol fixture only.

## Exit criterion

Answer `yes` if the unit test passes, the bundle smoke passes, and the
competition match ends normally by win, loss, draw, or truncation.

Answer `no` for any startup, parsing, reply, crash, EOF, or bundle failure.
