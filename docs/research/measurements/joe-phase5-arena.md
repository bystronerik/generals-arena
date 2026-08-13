# Joe Phase 5 — deployment and arena entry (2026-08-13)

Phase 5 of
[`averagejoe-competition-plan.md`](../strategies/averagejoe-competition-plan.md):
the trained net deployed as `bots/joe/` and rated in the arena. The
weights are an **interim snapshot of the still-running Phase 4 training**
— step 3000 of `joe-M-vast-20260813-0213` (curriculum stage 4 of 4,
in-training eval 97.7% vs random), exported with
`scripts/joe_export_bot.py`. Later exports fork new rated entities by
construction (the weights file is in the content hash).

## Verdict

**Joe enters the pool at 2483.5 ± 58.8 — the highest-rated entity — with
140W/9L/1D against a 5-bot panel spanning the whole rating range.** It won
28 of 30 games against `morpheus-rs@5456f5532cc2`, the previous best bot.
The pairwise contrast vs morpheus-rs is **+115.8 ± 59.1 Elo,
CI₉₅ [+0.3, +231.6], P(joe stronger) = 0.975** — quoted per
[decision-rule.md](../../arena/decision-rule.md) as **unproven**: the CI
low fails the +10 floor, the arm is 151 games against the 200-game floor,
and no replication round has run. The contrasts against every other panel
member clear the improvement thresholds by hundreds of Elo. Joe plays
greedy argmax with no search; morpheus-rs is deadline-driven, so the
decision rule's host-state caveat binds on the baseline side of this
contrast, not on joe's.

## Provenance

- Bot: `joe@04974d0e206b` (rated hash, this round). A comment-only
  `run.sh` reword afterwards — removing a literal path that had dragged
  the morpheus launcher into the closure via the shell-reference scan —
  forked the hash to `63cc71f5710e`; behavior is byte-identical (the
  weights, network, and observation code did not change), and the
  competition gate was re-run on the new hash (win vs morpheus-rs,
  turn 338, seed 0).
- Checkpoint: R2 `joe/joe-M-vast-20260813-0213/checkpoints/`
  `joe-M-vast-20260813-0213_ema_3000.eqx`, sha256 pinned in
  `bots/joe/artifact/manifest.json`.
- Engine: `9e3b9d13cca5` (competition-engine-2026-15), both the training
  run and every game below.
- Host: M3 Pro (11 physical cores), 11 tournament jobs, no other load.
  Joe is deterministic per position; morpheus-rs thinks to a deadline.

## Competition gate

`matchup.py bots/joe/run.sh bots/morpheus-rs/run.sh --mode competition
--seed 0` (with an **absolute** `PYTHON` — matchup sets cwd to the bot
dir, so the usual relative `.venv/bin/python` does not resolve): joe won
by general capture at turn 304 on the rated hash, turn 338 on the
re-gated comment-fix hash.

## Latency (deployed bot, full games, one local core)

`scripts/joe_bot_latency.py --games 3` — the real `run.sh` subprocess
over the real wire, competition-preset env, random opponent: 1,923 timed
moves, **p50 5.0 ms, p90 5.2 ms, p99 5.4 ms, max 23.3 ms**; first move
(process start + weight load + JIT) 2.2 s against the 10 s grace. Local
Apple-Silicon numbers; the x86 budget evidence remains
[`joe-phase2-cpu-latency.md`](joe-phase2-cpu-latency.md) (M tier true
p99 ≈ 19 ms on one hard-limited x86 core vs the 150 ms limit).

## Round `joe-entry-r1`

Five pairwise invocations of `arena.tournaments.competition`, one shared
round: `--games-per-pair 30 --round-seed 7 --seat-policy alternate
--strict-versions`, 150 games, all `mode=competition`, one engine
version, stored under `data/games/joe-entry-r1/`.

| Opponent | Rating (pre-round) | Joe W–L–D | Contrast (opp → joe) | P(joe) |
| --- | ---: | :---: | --- | ---: |
| `morpheus-rs@5456f5532cc2` | 2383 | 28–2–0 | +115.8 ± 59.1, CI [+0, +232] | 0.975 |
| `macaria@5760b30723e4` | 2269 | 28–2–0 | +233.7 ± 56.9, CI [+122, +345] | 1.000 |
| `sosipolis@e8d618ef7dc6` | 2100 | 28–2–0 | +398.6 ± 57.2, CI [+286, +511] | 1.000 |
| `metro@b2ab1e936cb1` | 1810 | 27–3–0 | +680.9 ± 58.0, CI [+567, +795] | 1.000 |
| `cm_expander@3097ee53a533` (anchor) | 1500 | 29–0–1 | +983.5 ± 58.8, CI [+868, +1099] | 1.000 |

Fit after the full-pool refit (7,309 rated games): joe 2483.5 ± 58.8,
151 games (the 151st is the seed-100 scout game vs morpheus-rs stored
under round `joe-entry-scout`). 150 of 151 games decisive; the single
draw was a 1200-turn truncation as seat B vs the anchor.

Behavioral note for later phases: joe built **zero castles** in all 151
games while morpheus-rs built up to 2 per game — the policy either
learned to spend armies elsewhere or under-uses the build action; worth a
look when the final Phase 4 checkpoint lands.

## What this is and is not

This is the Phase 5 arena entry of an interim checkpoint: games stored,
ratings refitted full-pool, contrast quoted with its interval. It is
**not** a published joe-beats-morpheus-rs claim — that needs a
replication round (separately scheduled, host state noted) and ≥ 200
games per arm per the decision rule, and it will be run against the
**final** Phase 4 checkpoint rather than spent on step 3000.
