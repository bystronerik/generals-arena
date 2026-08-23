# Joe-M checkpoint selection plan

Status: draft 2026-08-21. Find the strongest EMA checkpoint of
`joe-M-vast-20260813-0213` (depth 5, embed 384, ff×3, 8.56 M params) out of
the 98 sets R2 still holds. The run is finished at step 50000; nothing above
step 10000 has ever been measured for strength.

The instrument is a two-tier funnel: a **batched GPU screen** narrows 98
checkpoints to a shortlist, and a **rated arena round** decides. The screen
never produces a verdict, and its games never enter `data/games/` or
`data/ratings/` ([AGENTS.md](../../../AGENTS.md#what-is-never-stored-or-rated)).

---

## 0. Answer (2026-08-22)

**Step 50000**, the run's last checkpoint — tied with 49500 at 1.2 ± 3.0 Elo,
with 48500 the next candidate 5.5 Elo back. The strength curve is monotone
across the whole run at both 2500- and 500-step granularity, and the
improvement rate decayed to +1.9 Elo per 1k iterations at the end: the run
converged rather than being cut short or degrading.

Reached on ~2.8 H100-hours of screening (396,288 games) and **no arena time**,
because the top of the run turned out to sit below what a rated round can
resolve — see §4b. Evidence:
[calibration](../measurements/joe-ckpt-screen-calibration.md) ·
[stage 1](../measurements/joe-ckpt-s1-screen.md) ·
[stage 2](../measurements/joe-ckpt-s2-screen.md).

## 1. What "strongest" means here

The target is **arena Elo against the standing panel**, read as a pairwise
contrast inside one round per
[decision-rule.md](../../arena/decision-rule.md). It is *not*
checkpoint-vs-checkpoint self-play winrate, which is what the cheap screen
measures. The two can disagree — a late checkpoint can drift into a style its
own neighbours punish while the heuristic panel does not, or the reverse — so
the screen is explicitly a **narrowing instrument** and stage 3 is the answer.

**Scope: the joe-M lineage alone.** Decided 2026-08-21. Every arm in every
stage is a checkpoint of `joe-M-vast-20260813-0213`. The deployed
`joe-M7F4-vast-20260819-0207` step-16000 artifact is *not* an arm — "is M7F4
better than M?" is a different question, needs its own round, and mixing it
in here would leave neither question answered.

Pre-committed non-failure outcome: if the plateau turns out to be genuinely
flat, the plan's answer is *"no checkpoint in the band is distinguishable;
take step 50000"*. That is a result, not a wasted round.

## 2. What is already known (measured, this repo)

**Inventory (R2, verified 2026-08-21).** 98 complete checkpoint sets under
`joe/joe-M-vast-20260813-0213/`: EMA steps 500 → 50000 every 500, **missing
41500 and 47000**, plus a `_ema_final.eqx`. Each EMA blob is 34,237,800 bytes
(f32). 98 matching `state/state-<step>.json`.

**Arena history — the run stops being measured at step 10000:**

| Step | Round | Result |
| ---: | --- | --- |
| 3000 | [joe-phase5-arena](../measurements/joe-phase5-arena.md) | entry |
| 3500 | [joe-r2](../measurements/joe-r2-step3500.md) | +104.5 over 3000 |
| 5000, 6000 | [joe-r3](../measurements/joe-r3-step6000.md) | +62.7 (5000→6000) |
| 10000 | [joe-r4](../measurements/joe-r4-step10000.md) | **+284.8 ± 29.0** over 6000 |
| 13500 | sandbox smoke | not a strength measurement |
| 23500 | joe-net port plan | not a strength measurement |
| 15000–50000 | — | **never measured** |

r4 also established the improvement rate stopped decelerating: ~71 Elo per
1000 iterations across 6000 → 10000. Extrapolating that to 50000 is not
evidence; it is the hypothesis this plan tests.

**Training diagnostics plateau from ~20000** (means per 2500-step bin, from
the run's own `logs/metrics.jsonl`, 48,046 rows, steps 2501–50000):

| step | ep_len | draw | entropy | castles | EV | approx_kl | lr |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10000 | 497.9 | 0.021 | 0.551 | 0.95 | 0.974 | 0.0175 | 1.8e-05 |
| 20000 | 501.2 | 0.033 | 0.414 | 1.31 | 0.973 | 0.0141 | 8.7e-06 |
| 30000 | 535.7 | 0.042 | 0.396 | 1.54 | 0.976 | 0.0132 | 5.7e-06 |
| 40000 | 521.3 | 0.034 | 0.384 | 1.44 | 0.975 | 0.0125 | 5.0e-06 |
| 50000 | 496.0 | 0.010 | 0.381 | 1.32 | 0.972 | 0.0117 | 5.0e-06 |

No collapse, no divergence — and no ranking signal either. The LR sits on its
5e-6 floor from ~35000. **The in-training eval cannot rank these
checkpoints**: it plays vs `sample_valid_action` and reads 0.94–0.99, noise
around saturation, in every 5k bin. There is no free strength curve; games
must be bought.

## 3. Stage 1 — coarse GPU screen (Modal H100)

**New script: `scripts/joe_modal_ckpt_sweep.py`** — Modal, `gpu="H100"`, decided
2026-08-21. The `scripts/*_modal_*.py` name is what
[modal-jobs.md](../../engine/modal-jobs.md) keys its discipline to. It wraps the primitive that already exists:
`training/joe/train/evaluations.py::evaluate_vs_ref` — a `lax.scan` over
`truncation` steps that plays `n_maps` games with net-A on seat 0 and again on
seat 1, both sides greedy, returning (wins, losses, draws, finished). Seat
balance is by construction, maps are shared between the arms, and the outcome
is deterministic per map.

Fidelity settings, all matching the arena rather than the training default:

- `make_competition_env(min_generals_distance=17)` — the final curriculum
  stage, which *is* the competition preset (`training/joe/env.py`).
- `truncation = 1200`, the competition cap ([RULES.md](../../../RULES.md)
  line 148). No divergence to reconcile here.
- Opponent legs against `generals.agents.{Expander,Hunter,Harvester}Agent` —
  JAX-native and **byte-identical to the arena's `cm_expander`, `cm_hunter`,
  `cm_harvester`**, which are four-line wrappers over them. `cm_expander` is
  the global rating anchor.

**Grid:** every 2500 from 2500 → 50000 (20 nets), forced to include 3000,
6000 and 10000 so the screen overlaps the rated rounds. Full round-robin, 1024
games per pair (512 per seat) = 190 pairs, plus 3 heuristic legs per net.

**Cost — measured, not estimated.** The pilot ran on 2026-08-21:
**~42 games/s on one H100**, flat in batch size from n_maps 256 upward, so the
card is already saturated and batching harder buys nothing
([joe-ckpt-screen-calibration](../measurements/joe-ckpt-screen-calibration.md)).
Stage 1 at 1024 games/pair is 194,560 games ≈ **77 min** of H100, ~$5. The
pre-measurement estimate in this plan's first draft was ~35 min; it came from
extrapolating the phase-1 rollout figure and was optimistic by 2.2×.

**Getting the weights onto the GPU.** The checkpoints live in R2; Modal
containers do not have the R2 credentials, and no `modal.Secret` exists in this
account today. Decided: **stage the blobs into a Modal Volume**, not a new
secret. `joe-M-ckpts` (`modal volume create`), populated once from the local
machine — `store.download_file` per step, then `modal volume put`. All 98 EMA
blobs are 3.4 GB; stage 1 alone needs 20 (685 MB), but pushing the whole set
once is simpler than deciding twice. The alternative — a Modal Secret carrying
`R2_*` so the container pulls directly — is faster but copies a token that can
*write* the bucket holding every joe training run into a third-party service,
to save a one-time upload. Not worth it. `.env` stays the only home for those
values either way.

**Container discipline** ([modal-jobs.md](../../engine/modal-jobs.md)): guard
every repo import with `modal.is_local()`, because Modal re-imports the script
inside the container with only that file mounted. Redirect the run to a file —
never pipe it through `tail`. **Check `modal app logs <app-id>` a few minutes
after launch and confirm the job got past startup**, before doing anything
else. A job that dies at import looks exactly like one that is working.

**The calibration gate — passed 2026-08-21.** The screen had to reproduce the
contrasts the arena already rated, or stage 2 would not start. On 6,144 games
over steps 5000/6000/10000, all three land inside the published intervals, and
the widest span matches to 7.7 Elo:

| Contrast | Screen | Arena (r4) | Arena CI₉₅ |
| --- | ---: | ---: | --- |
| 5000 → 6000 | +49.2 | +69.9 ± 29.1 | [+12.9, +126.8] |
| 6000 → 10000 | +313.2 | +284.8 ± 29.0 | [+228.0, +341.7] |
| 5000 → 10000 | +362.4 | +354.7 ± 33.7 | [+288.7, +420.7] |

Full setup and the throughput numbers:
[joe-ckpt-screen-calibration](../measurements/joe-ckpt-screen-calibration.md).

**Ranking is a joint Bradley-Terry fit over the pair matrix, never chained
per-pair Elo.** The pilot measured the difference: per-pair score rates gave
+43.0 and +269.6 across two hops against a direct +402.3, a 73-Elo
transitivity gap that would compound at every step of a 20-checkpoint ladder.
`bradley_terry()` in the sweep script does the joint fit.

Expect the heuristic legs to saturate near 100%; they are a smoke test, not a
ranker.

Output: `docs/research/measurements/joe-M-ckpt-screen.json` + a short `.md`.
Nothing under `data/games/` or `data/ratings/`.

## 3b. Stage 1 result (2026-08-21): step 50000, monotone

Ran: 22 checkpoints, 231 pairs, 236,544 games, 98 min.
[joe-ckpt-s1-screen](../measurements/joe-ckpt-s1-screen.md).

**Step 50000 is the strongest, and the curve is monotone at all 21 steps** —
no degradation, no local peak at 2500 granularity. Only 47500 is
indistinguishable from it (−4.0 ± 4.1); 45000 is already 3σ behind. The
improvement rate decayed from +145 Elo per 1k (step 3000–5000) to **+1.9 per
1k** over the last 2,500 iterations: the run converged.

This lands on §1's pre-committed outcome by a different route than expected.
The band is not flat — it is monotone and decelerating — but the top of it is
**below the arena's resolution**: 47500 → 50000 is 4.6 Elo against a decision
rule whose affordable floor is ±25. So the plan's answer is **step 50000**, and
stages 2–3 need re-justifying rather than executing on momentum.

## 4. Stage 2 — fine GPU screen

Take the band stage 1 flags (the plateau shape predicts something in
20000–50000, but the data decides). Run **every** 500-step checkpoint in that
band against a fixed 5-net reference panel drawn from stage 1's spread, 2048
games each, then a full round-robin at 4096 games among the top 8. At the
measured 42 games/s that is ~2.0 h + ~46 min ≈ **2.8 h** of H100, ~$11 —
so the whole screen, both stages, costs about four H100-hours.

Deliverable: a ranked shortlist with intervals, and an explicit statement of
whether any separation exceeds the screen's own noise floor. If it does not,
stop here and report the flat result — do not spend arena hours resolving a
difference the cheap instrument says is not there.

**Post-stage-1 scoping (2026-08-21).** The band is 44000–50000 at 500
granularity — 12 checkpoints, since 47000 is not retained — round-robin at
2048 games/pair, 66 pairs, ~56 min. This is now a **cheap completeness check,
not a search**: stage 1 found no local peak anywhere on the curve, so the
prior on one hiding inside the last 6,000 iterations is low, and anything it
did find would be worth a handful of Elo. Run it to close the question at the
granularity the run actually saved, or skip it and take 50000 — but do not
run it expecting to change the answer.

## 4c. Stage 2 result (2026-08-22): no local peak — the answer is step 50000

Ran: 13 checkpoints (44000–50000 at 500 granularity, 47000 not retained, plus
42500 as a control), 78 pairs, 159,744 games, 64 min.
[joe-ckpt-s2-screen](../measurements/joe-ckpt-s2-screen.md).

**49500 (+15.6 ± 2.0) and 50000 (+14.4 ± 2.0) tie at 1.2 ± 3.0 Elo**; 48500 is
next at −5.5. No local peak, no inversion above noise, nothing the 2500-step
grid had hidden. Six pairs overlapping stage 1 repeat to a mean 5.5 Elo.

**This closes the plan's question. The strongest checkpoint of
`joe-M-vast-20260813-0213` is step 50000** — tied with 49500, and the run's
last checkpoint either way, so it costs nothing to adopt.

## 4b. Stage 3 is not currently justified

Stage 3 exists to confirm a screen winner in the arena. The screen's winner is
step 50000 — which is also the run's last checkpoint, i.e. the default. There
is no candidate to promote over it and no decision waiting on a rated round:
the nearest rival is 1.2 Elo away, six times finer than the ±25 the arena can
buy in 4–6 h.

An arena round here would answer a question worth asking only if it were
reframed — for example, *"how much did the 40,000 iterations past step 10000
actually buy against the panel?"*, which the screen cannot answer because it
measures joe against joe and the panel may saturate. That is a real question,
it is not this plan's question, and it needs its own round.

## 5. Stage 3 — the rated arena round (the verdict)

The top 2 checkpoints from stage 2, plus **step 50000 as the baseline arm**.

Step 50000 is the null hypothesis: "just take the last checkpoint" is what you
do without this plan, so it is the thing a winner has to beat. If 50000 is
already in stage 2's top 2, the round runs two arms instead of three.

1. Export each arm with `scripts/joe_export_bot.py --run-name
   joe-M-vast-20260813-0213 --step <s>` and freeze it as its own bot directory
   (`bots/joe_c<step>/`), the `joe_base` / `joe_prev` pattern r3 and r4 used.
   Frozen copies exist only for the round's duration. Every arm, baseline
   included, is measured **inside this one round** — cross-round baselines are
   not comparators, measured at +46 Elo between byte-identical programs.
2. One round, all arms inside it, J4 panel (aegis, macaria, boom, cm_hunter,
   cm_expander — the panel the f16 and S1 rounds used):

```bash
python -m arena.tournaments.competition bots/joe_c50000/run.sh bots/joe_c<top1>/run.sh bots/joe_c<top2>/run.sh bots/aegis/run.sh bots/macaria/run.sh bots/boom/run.sh bots/cm_hunter/run.sh bots/cm_expander/run.sh --round joem-sel-r1 --games-per-pair 50 --round-seed 7 --seat-policy alternate --strict-versions --mode competition
```

3. Sample size from the decision rule: ≥200 games/arm is the floor for any
   verdict; ~1150/arm buys the ±25 "proven flat" verdict. **Honest cost:**
   past rounds on the dev Mac ran 560 games in 33 min (joe-r4) and 4,276 in
   308 min (s4-trail-r1) — call it 14–18 games/min. Three arms at 1150 is
   roughly 4–6 h, and the host must be otherwise idle for the duration.
4. Record host state by hand. A round whose host state is unknown is
   unpublishable.
5. Read the verdict from `fits["joem-sel-r1"].delta(A, B)`, quote it per the
   decision rule, and **schedule an r2 replication before publishing**. One
   round is not evidence; the M6 contrast passed every gate and was wrong by
   300 Elo.

## 6. Stage 4 — ship, only on `improvement`

Follow the artifact fan-out, in order: `joe_export_bot.py` → f16
`quantize_artifact.py` → joe-rs `convert_artifact.py` → regenerate the parity
corpus, the committed fixture, and the joe-rs self-goldens. A joe re-export
silently staleens every one of those. Then the version registry, the
measurement doc, and the `docs/index.md` line.

## 7. Risks, named

| Risk | Handling |
| --- | --- |
| Screen criterion ≠ arena criterion (non-transitivity) | Stage-1 calibration against r3/r4; stage 3 is authoritative |
| Screen plays argmax; joe-rs deploys Gumbel T=1 | S1 measured Gumbel vs argmax **proven flat** (+3.8 ± 10.7); acceptable, noted |
| Screen is f32; deployment is f16 | The f16 contrast measured non-regressive |
| Argmax limit cycle inflates screen draws | Draw rate is reported per pair; a pair above the run's ~3% training draw rate is flagged, not averaged away |
| Heuristic legs saturate | Declared a smoke test up front; ranking comes from checkpoint-vs-checkpoint |
| Plateau is genuinely flat | Pre-committed answer in §1 — report flat, take step 50000 |
| Screen results leak into ratings | Written only to `docs/research/measurements/`; §3 states the prohibition |

## 8. Decided

Both questions this plan opened were answered on 2026-08-21.

| Question | Decision |
| --- | --- |
| Which GPU path runs the screen? | **Modal**, `gpu="H100"`. No vast lease to hold, no interruption to resume around, and the screen is ~35 min of work — the interruptible machinery exists for 20k-iteration training runs, not for this. |
| joe-M alone, or a contrast against the deployed M7F4 artifact? | **joe-M alone.** Every arm is a joe-M checkpoint; step 50000 is the baseline. A cross-lineage contrast is a separate round with a separate question. |

Consequences already written into the plan above: the sweep script is
`scripts/joe_modal_ckpt_sweep.py` (§3), weights reach the GPU through a
`joe-M-ckpts` Modal Volume rather than a new R2 secret (§3), and the stage-3
round is 2–3 joe-M arms with step 50000 as the null (§5).

## 9. Follow-on, not in scope

If stage 3 finds a joe-M checkpoint that beats step 50000, the obvious next
question is whether it also beats the deployed M7F4 step-16000 artifact. That
is a second round, `joem-vs-m7f4-r1`, with both lineages as arms in it. It is
named here only so it does not get smuggled into this plan's round.
