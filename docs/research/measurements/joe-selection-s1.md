# joe-rs S1 Gumbel selection vs the argmax baseline

**Claim tested:** deterministic Gumbel selection at T = 1 (selection-plan
S1, shipped 2026-08-20) plays no worse than the plain argmax of the same
network. Since 2026-08-20 both bots hold the same f16-rounded
`joe-M7F4-vast-20260819-0207` step-10000 checkpoint and Python joe is the
plain argmax again (commit `b54dc23`), so the joe vs joe-rs head-to-head
isolates the selection layer to within the measured tier-2 parity bound
(≤ 3.0e-6 relative logit error). At T = 1 the played move departs from the
argmax on 12.9% of corpus turns.

## Setup

- Arms: baseline `joe@3a2bd858631c` (Python, plain argmax), candidate
  `joe-rs@bcd7c980aa5f` (Rust, Gumbel T = 1). Both arms in the same round
  per [decision-rule.md](../../arena/decision-rule.md).
- Panel: aegis, macaria, boom, cm_hunter, cm_expander (anchor) — the J4
  panel, the same shape as the
  [f16 contrast](joe-rs-f16-quantization.md): panel games satisfy the
  per-opponent gate and connectivity; the contrast information comes from
  the head-to-head pair.
- `--seat-policy alternate --strict-versions --mode competition --timeout
  600` throughout; engine as recorded in the game files.
- Round `s1-gumbel-r1` (2026-08-20/21): panel invocations with round-seeds
  202, 212, 222 (8 + 8 + 16 games per pair — `--games-per-pair` counts
  games, and `alternate` splits them across orientations rather than
  doubling them, which is why a third panel invocation was needed to reach
  the ≥ 30 per-opponent gate); H2H invocations with round-seeds 101, 111,
  121, 131, 141 (200 + 200 + 150 + 400 + 150 games, 6 jobs; panel
  invocations at 8 jobs). 1,772 games, 1,292 per arm, 1,132 head-to-head.
- Host: the dev arm64 Mac (11 cores, 18 GB), otherwise idle for the
  duration (no builds, benchmarks, or remote jobs). Neither arm is
  deadline-driven — both compute one fixed forward per turn — so host load
  could not steer decisions, only latency.

## Gates

All pass: 1,772 games all `mode=competition`, one engine (`9e3b9d1`),
both arms registered, same round, one connected component with the global
anchor (`cm_expander@3097ee53a533` at 1500), no excluded games;
1,292 ≥ 200 games per arm; 32 ≥ 30 games per (arm, opponent);
1,184/1,181 ≥ 60 decisive games per arm.

## Results

| Round | Δ(Gumbel − argmax) | CI₉₅ | P(B > A) | H2H (rs–joe–draw) | Verdict |
| --- | --- | --- | --- | --- | --- |
| r1 | **+3.78 ± 10.69** | (−17.2, +24.7) | 0.638 | 520–504–108 | **no change (proven flat)** |

`games_to_resolve(target_se=12.75)` = 0 — the round met the SE target.
Mean H2H game length 593 turns; H2H draw rate 9.5%.

## Reading

S1's Gumbel noise at T = 1 costs nothing measurable against the same
network's argmax: the CI sits entirely inside the ±25 flat window, with
the point estimate mildly positive. That is the outcome S1 was designed
for — the mechanism exists to break the argmax limit cycle
([joe-argmax-limit-cycle](joe-argmax-limit-cycle.md)), a pathology too
rare on random maps to carry Elo of its own, and the risk the round
priced was the noise picking genuinely worse moves on near-ties. It
does not, at panel scale. joe-rs keeps Gumbel T = 1 as its deployed
selection.

Replication caveat (decision-rule): one round is not evidence; an `r2`,
separately scheduled, is the required follow-up before this verdict is
final. Neither arm is deadline-driven, so the M6 fragility binds least
here, but the rule stands.
