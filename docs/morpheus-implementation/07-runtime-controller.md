# Part 07: Runtime controller

## Deliverable

Implement first-move setup, normal-turn work order, monotonic admission
control, deterministic degradation, bounded tree memory, and passive runtime
metrics.

The controller must store a valid fallback before it starts optional work.

**Touches**

- `bots/`: add the controller and runtime counters.
- `arena/`: declare passive probe keys and reducers.
- `scripts/`: none.
- `data/bot_versions/`: none.

## Prerequisites

- [Part 01: Protocol shell](01-protocol-shell.md)
- [Part 06: Simultaneous search](06-simultaneous-search.md)

## Source specifications

- [Runtime](../bots/morpheus/runtime.md)
- [Network inference budget](../bots/morpheus/network.md#inference-budget)
- [Evaluation secondary checks](../bots/morpheus/evaluation.md#promotion-decision)

## Defaults and replacement measurement

Start with these runtime guesses:

- 125 ms normal internal deadline and 25 ms reserve;
- 8.5 s first-move internal limit;
- 32 target simulations and 8 minimum;
- search leaf batch up to 4;
- 113 forward-equivalents;
- 10 ms admission guard;
- 4096 tree nodes;
- resident-memory target below 256 MB.

Part 09 replaces the coupled configuration. No component can be tuned alone.

## Implementation boundary

Use a monotonic clock. Check admission before selection, before inference, and
before each new simulation. Do not assume that an inference call can be
cancelled.

The fallback order is pass, then highest-prior legal policy action, then the
best available completed-search result. A partial simulation cannot affect the
choice.

Expose component timing, completed simulations, forward-equivalents, belief
ESS, recovery, tree size, and fallback level through `probe.py`. The probe
must remain outside the content hash and must not affect play.

## Isolated test

```bash
python -m pytest \
  bots/morpheus/tests/test_runtime_controller.py \
  bots/morpheus/tests/test_deadline_degradation.py \
  -m morpheus -q
```

The tests use an injected clock and fixed cost forecasts for every degradation
level.

```bash
python competition-module/competition/matchup.py \
  bots/morpheus/run.sh bots/smoke/run.sh \
  --mode competition --seed 0
```

## Specification gaps

The specs do not define the moving p99 estimator, its warm-up behavior, sample
window, or what forecast to use before enough samples exist.

The boundary between `0 completed simulations` and `no root result` needs an
executable definition. The first-move warm-up batch set also depends on the
runtime selected in Part 09.

## Exit criterion

Answer `yes` if every injected-cost case returns the specified legal fallback,
no partial simulation changes statistics, all storage bounds hold, probe
metrics are passive, and the competition gate finishes.

Answer `no` for a missed fallback, non-monotonic timing decision, work admitted
past its forecast, or unbounded memory.
