# Evaluation and artifact identity

## Decision

A Morpheus checkpoint becomes an arena citizen only as a frozen bot artifact.
It uses the same competition match store, version registry, rating fit, and
decision rule as every other rated bot.

There is no neural-model leaderboard or separate checkpoint score.

## Frozen artifact

The future `bots/morpheus/` closure must contain:

- inference and search code;
- the exact quantized weight file;
- a model manifest;
- `main.py`, `run.sh`, and any deterministic build input.

The manifest identifies the tensor schema, action schema, architecture,
quantization, training run, and full weight SHA-256.

Every file that affects play must exist under the bot closure at hash time.
The existing fingerprint includes binary assets in the bot directory. A
one-byte weight change therefore creates a new content-hash rated entity.

A mutable external path, download, or "latest checkpoint" symlink is forbidden.
The competition build has no network and must not select different weights.

## Competition verification gate

After weights are frozen, the artifact must finish this required shape of
match:

```bash
python competition-module/competition/matchup.py \
  bots/morpheus/run.sh \
  bots/smoke/run.sh \
  --mode competition --seed 0
```

The match must end normally by win, loss, draw, or turn-1200 truncation. A
classic match does not count.

This local gate proves that the final artifact starts, loads its weights,
exchanges parsable actions, and completes a competition-mode game. The local
runner does not enforce the judge's move timer, fault budget, or memory limit.
It also cannot prove a clean voluntary EOF exit when it can terminate a process.

Checkpoint acceptance therefore also needs a resource-enforcing,
submission-shaped check. That check measures first response within 10 seconds,
normal responses within 150 ms, peak memory below 2 GB, malformed or missing
replies, crashes, and voluntary exit after stdin closes.

No such harness exists in the repository. Its owner, path, and enforcement
method are an explicit [open question](open-questions.md). Morpheus cannot
promote until that question is resolved and the check exists.

Any timing, format, missing-reply, memory, crash, or EOF failure rejects the
artifact.

## Arena entry

Evaluation stores each game before any rating refit:

```bash
python -m arena.matches.run_match \
  bots/morpheus/run.sh \
  bots/<opponent>/run.sh \
  --mode competition --seed <seed> --round <round>
```

The parent process registers Morpheus's content hash in
`data/bot_versions/morpheus.json`. Tournament workers must find that exact
registered hash.

A checkpoint panel uses held-out seeds and both seat orientations through the
competition tournament's alternate-seat policy. The panel includes the rating
anchor, strong heuristic bots, research bots, and the prior promoted Morpheus
checkpoint.

## Promotion decision

The batch Bradley-Terry-Davidson fit rates `morpheus@content_hash`. A new
checkpoint remains provisional until it has the repository's required game
count.

Promotion compares the candidate and current checkpoint with the pairwise
rating contrast and confidence interval in
[the decision rule](../../arena/decision-rule.md). Leaderboard rank is not a
decision.

Required secondary checks are:

- zero protocol faults;
- p99 reply time below the internal deadline;
- completed simulations per move;
- belief effective sample size and recovery rate;
- decisive, draw, and truncation rates;
- performance by board size and seat.

These checks can reject an unsafe checkpoint. They cannot promote a weaker one
against the rating decision.

## ResBot separation

ResBot leaderboard replays remain observational. They do not receive bot
versions, become arena games, connect the rating graph, or affect a rating fit.

Similarity to ResBot behavior is a diagnostic only. A checkpoint can promote
with different expansion, castle, or sight timing when arena evidence supports
it.

## Alternatives rejected

Rating a bare weight filename is rejected because the result also depends on
tensor, search, runtime, and code. Rating checkpoints outside the content-hash
registry is rejected because it can pool different programs under one name.

## Failure mode

A checkpoint can be strong in warm in-process evaluation and fail as a cold
stdio submission. The raw competition gate must therefore run the final bundle
shape after every artifact change.
