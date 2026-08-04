# Part 09b: Defer the normal latency gate

## Decision

2026-08-05: pause hard optimization against the ~150 ms normal reply budget.

Competition judge discussion: Morpheus may take longer than 150 ms per normal
move. The practical cost is a slower reaction than opponents that reply inside
the soft budget. That risk is accepted for now.

## Working rule

1. Prefer a correct, complete belief-plus-search bot over a Part 09 latency
   survivor.
2. Keep the competition match gate (`--mode competition`) and search-quality
   floors (including the 8-simulation minimum when search runs).
3. Do not block progress on trunk redesign or further micro-opts solely to
   clear local normal p99 ≤ 125–150 ms.
4. Measure how much reply delay matters in real competition games, then decide
   whether to tighten deadlines again.

## Still record

Local online-runtime measurements, component telemetry, and submission-harness
timeouts remain useful diagnostics. They are not the current go / no-go for
shipping a playable Morpheus.

## Related

- [Part 09: Online qualification](09-online-qualification.md)
- [Part 09a: Complete-turn cost reduction](09a-complete-turn-cost.md)
- Measurement:
  [`morpheus-complete-turn-cost.md`](../research/measurements/morpheus-complete-turn-cost.md)
