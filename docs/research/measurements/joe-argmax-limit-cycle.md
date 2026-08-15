# Joe — why it oscillates instead of hunting the general (2026-08-16)

Diagnostic, not a measurement round. No games were stored into a rated round
directory and no ratings were refitted. The seven games this investigation
produced were written to `data/games/adhoc/` by `run_match.py` and deleted
afterwards; the traces under `data/trajectories/joe-osc*` are what the numbers
below come from.

## Verdict in one line

Joe's deployment path takes the **argmax** of a policy that PPO trained by
**sampling**. Where the policy is confident about pulling army *onto* one of its
own structures and near-indifferent about which way to push it back *off*, the
argmax turns that indifference into a fixed answer, and joe walks a stack around
its own castle forever. The training-time sampler broke exactly this state for
free, so the gradient never had to learn an escape.

## How specific the failure is

`bots/macaria/run.sh` vs `bots/joe/run.sh`, `--mode competition`, seeds 0-4,
both seat orders. Ten games:

| Seed | joe as seat 1 | joe as seat 0 |
| ---: | --- | --- |
| 0 | **win** t=139 | win t=288 |
| 1 | **LOSS** t=641 | win t=449 |
| 2 | **win** t=210 | win t=285 |
| 3 | **win** t=428 | win t=681 |
| 4 | **win** t=299 | win t=280 |

Joe wins **9 of 10**. The reported failure is **one game** — seed 1, seat 1 —
and it is neither seat-specific nor seed-specific in the way the phrasing
suggests: joe as seat 0 on seed 1 wins, and joe as seat 1 on every other seed
wins. "Joe never locates the enemy general" is true of that one game only.

That single fact reframes the rest. This is not a broken bot. It is a **latent
attractor present in most games** that only closes into a trapping cycle when
the board around it goes quiet, and the two slowest wins (t=641 and t=681) are
the same defect surviving rather than a different one.

## What the cycle is

Instrumented with `bots/joe/probe.py`, read back with
`scripts/inspect_joe_oscillation.py`, `..._confinement.py`, `..._window.py`,
`..._castle_pull.py` and `..._frontier.py`.

Joe builds a castle at **(7,18) on turn 158**. From roughly turn 200 it walks
one stack up and down a three-cell corridor — the castle and its two vertical
neighbours — for about 250 turns. The period-4 loop is exact:

```
(6,18) --down--> (7,18) --down--> (8,18) --up--> (7,18) --up--> (6,18) --> ...
```

Confirmed turn by turn from t=255 to t=290: the stack grows +1 every two turns
from castle production (26 → 44 army) and never leaves.

**Onset, period, duration, and whether it breaks.** The strict detector (period
≤ 4, three full repetitions) finds 11 spans covering 90 of 641 turns (14.0%),
periods 1, 2 and 4, the two longest being 17 turns each (t=310-326, t=343-359).
That undercounts the real symptom, which is looser confinement. Per 50-turn
band, counting how many distinct cells joe acted from:

| Turns | Distinct cells | Top cell | Its share | Value head |
| --- | ---: | --- | ---: | ---: |
| 51-100 | 44 | (5,13) | 4% | +0.003 |
| 101-150 | 45 | (6,18) | 4% | +0.393 |
| **151-200** | **14** | (7,18) | 38% | +0.329 |
| **201-250** | **6** | (7,18) | 38% | +0.038 |
| **251-300** | **4** | (7,18) | 50% | −0.225 |
| **301-350** | **13** | (7,18) | 42% | −0.707 |
| **351-400** | **16** | (7,18) | 36% | −0.813 |
| 401-450 | 45 | (14,1) | 4% | −0.786 |

The cycle **starts within a turn or two of the castle build**, runs ~250 turns,
and **does break on its own around turn 400** — by which point macaria holds 144
land to joe's 48 and the game is already decided. Joe's land is *frozen* at
41-61 from turn 200 to the end while macaria climbs 71 → 212. The enemy general
never enters joe's view in the whole 641 turns.

## Which of the three candidate faults it is

The probe was built to separate them. Two are cleanly refuted.

**Not stale `AugmentedObsState`.** The net is deterministic, so identical logits
mean an identical input tensor. Over 641 turns there are **641 distinct logit
hashes — zero repeats**. Every turn's observation is genuinely different (the
army counts and timestep channels alone guarantee it). The persistent obs state
is not laundering two board states into one.

**Not a near-tie flipping under argmax.** Median top1−top2 margin is **2.213
logits in-cycle** against **2.330 out-of-cycle** — indistinguishable. Only 5.3%
of turns have a margin under 0.1 logit. The policy is not confused; the cycle is
not argmax noise between two tied actions.

**It is a confident cycle — but confident asymmetrically.** Splitting the chosen
move by what it does, inside the confinement window t=200-400:

| Move | n | Median margin | Share with margin > 5.0 |
| --- | ---: | ---: | ---: |
| **onto** an own structure | 84 | **11.313** | **76%** |
| **off** an own structure | 85 | **1.276** | **6%** |
| touching no structure | 32 | 1.206 | 22% |

A **9× median gap**. On the turns that pull the stack back onto the castle the
policy is maximally certain — entropy 0.000, p(top-1) = **1.0000**. On the turns
that push it off, it is nearly indifferent: p10 of p(top-1) is **0.522**, and
the top three candidates are the same cell in three different directions
separated by 0.01-1.4 logits.

That asymmetry is the whole mechanism, and it is why the failure needs no bug to
explain it. Each of the two preferences is individually reasonable. Composed
under a deterministic argmax they form a limit cycle: leave the structure in
whatever direction the near-tie happens to pick, get pulled straight back,
and — because the board barely changes while joe shuffles — resolve the same
near-tie the same way next turn, forever.

## Joe is not trapped; it declines to spend

The obvious alternative explanation is that joe is walled in. It is not:

| Turn | Land | Free neutral cells adjacent | Cheapest adjacent enemy tile | Joe's top stack |
| ---: | ---: | ---: | ---: | ---: |
| 200 | 50 | 18 | 1 | 51 |
| 250 | 45 | 16 | 2 | 77 |
| 300 | 41 | 15 | 2 | 103 |
| 350 | 44 | 15 | 2 | 102 |
| 400 | 48 | 14 | 2 | 103 |

Throughout the cycle joe has **14-18 neutral cells it could take for free**, the
cheapest enemy tile on its border costs **2 army**, and it is sitting on a stack
of **103** — about a third of its entire army, parked on its own general. It
loses land (59 → 41) while holding that reserve idle. The value head tracks the
decline honestly all the way to **−0.925**: joe's critic knows it is losing. The
policy simply does not act on it.

## Root cause: the sampling/argmax mismatch

`training/joe/train/rollout_selfplay.py` collects PPO rollouts through
`HistoryTransformer.__call__`, which draws `jax.random.categorical(key, logits)`.
`bots/joe/agent.py` deploys `jnp.argmax(logits)`. The curriculum gate
(`train/evaluations.py`) is greedy, but the **data the policy learned from was
sampled**.

Measured on the deployed distribution, per window on seed 1:

| Turns | Median p(top-1) | Expected turns for a sampler to deviate |
| --- | ---: | ---: |
| 1-150 | 0.933 | 15 |
| 151-200 | 0.972 | 36 |
| 201-300 | 0.909 | 11 |
| 301-400 | 0.975 | 40 |
| 501-641 | 0.714 | 3.5 |

Inside the confinement window the sampler deviated every ~21 turns on the
median, and on the *departure* turns — p10 = 0.522 — nearly every other turn. In
training, the loop dissolved on its own within a handful of turns every single
time it formed. The gradient therefore never received a state where escaping
mattered, and never learned to escape. Argmax deployment removes the only thing
that was breaking the loop.

**Direct check.** A throwaway copy of joe (scratchpad only; `bots/joe` unchanged)
that samples at temperature 1 instead of taking the argmax, on the same board and
opponent, **wins seed 1 at turn 356 instead of losing at 641**. Recording it
shows the attractor is *still there* — it forms the same four-action loop around
its own castle at (4,18) — but the longest span drops from 17 turns to 12, and
joe escapes at ~t=300, sights the general at t=355, and wins.

So the precise claim is: **sampling does not remove the attractor, it bounds the
time spent inside it.** That distinction matters for choosing a remedy.

## Candidate remedies, ranked

Cost and risk are for the mechanism, not for a particular file. None of these
were measured — a verdict needs both arms inside one rating round.

**1. Restore stochasticity at decision time.** Draw from the masked policy
instead of taking its argmax, so the departure near-tie stops resolving
identically every turn. This is the mechanism that trained the weights, so it is
the smallest possible distribution shift — it makes deployment match rollout.
*Cost:* trivial, one draw per turn, no latency change, and a seeded key keeps
games reproducible. *Risk:* it also randomises the turns where the policy is
correctly confident, which costs decisive speed. Exploratory 5-seed runs (seat 1
only) gave 5/5 wins at T=1.0, 5/5 at T=0.25, and **3/5 at T=0.5** — a
non-monotone result on a grid far too small to rank temperatures, and it should
not be read as one. What it does show is that the effect is real and that
temperature is a parameter needing its own measured round.

**2. Sample only where the policy is indifferent.** Keep the argmax whenever the
top-1/top-2 margin is decisive, and draw only when it is not. This targets the
exact site of the defect: the departure turns are near-ties (p10 = 0.522) while
the structure-pull turns are p(top-1) = 1.0000, so a margin threshold cleanly
separates the two and leaves every confident decision untouched. *Cost:* still
one draw, plus one threshold that has to be fitted. *Risk:* the threshold is a
new tunable with no principled value, and it is a tiebreak rule — which the
current design deliberately excludes. Ranked second on that ground, not on
expected effect; on expected effect it is the best of the list.

**3. Break the tie with the decision history rather than with noise.** Make the
decision depend on how recently joe acted on a cell, so a state it has just
visited stops looking identical to a state it has not. The cycle survives only
because a period-2 board yields a period-2 policy; any input that distinguishes
the second visit from the first destroys the fixed point deterministically.
*Cost:* moderate — it needs per-cell recency to reach the decision, which the
observation does not currently carry in a form the net reads. *Risk:* highest of
the list. It changes what the network is fed, so the weights are being asked to
generalise to a channel they were never trained on, and it is a heuristic
bolted onto a pure policy.

**4. Fix it in training instead of at deployment.** Continue the run with the
loop states in the data — e.g. by evaluating greedily during rollout collection
often enough that argmax-reachable cycles enter the replay distribution, so the
gradient sees a state that sampling used to hide. This is the only remedy that
removes the attractor rather than escaping it. *Cost:* highest by a wide
margin — a training run, a re-export, and the artifact fan-out that follows one
(joe-rs weights, unclejoe's copy, the parity corpus, the committed fixture).
*Risk:* an open-ended training change with no guarantee the behaviour is
reachable in self-play, where both sides camp symmetrically and camping is not
punished.

**Not recommended: suppressing castle building.** The castle is where the
attractor anchors, and every confined game has one. But seed 4 built a castle at
turn 190 and never confined at all (40-50 distinct cells per band, value rising
to +0.815), and seed 0 built two and won at turn 288. The castle is the
*occasion*, not the cause; removing it would forfeit real value and leave the
argmax/sampling mismatch in place to resurface elsewhere.

## Instrumentation added

- `bots/joe/probe.py` — top-5 decoded logits, top1−top2 margin, softmax entropy,
  p(top-1), value head, logit-vector hash, move/build mask sizes, cell revisits,
  detected cycle period.
- `bots/joe/agent.py` — the jitted `step` closure now returns the logits, value,
  and both masks alongside the action so the probe has something to read. The
  action is the same argmax of the same logits; `pytest -m joe
  bots/joe/tests/test_wire_fidelity.py` passes, including
  `test_agent_matches_training_eval_path`, which pins the greedy action to the
  training path. **This forks joe's content hash and therefore its rating
  identity**, as accepted in the task.
- `arena/records/telemetry_schema.py` — ten `joe_*` keys declared.
- `scripts/inspect_joe_{oscillation,confinement,window,castle_pull,frontier}.py`
  — the read side.

Two pre-existing, unrelated test failures were observed and left alone. Both
were confirmed to reproduce with the changes above stashed.

- `pytest -m joe` over all bots at once fails
  `bots/joe-rs/tests/test_synthetic.py`: the bare `agent` module resolves to
  whichever bot reached `sys.path` first (it picked up
  `bots/sosipolis/agent.py`). `bots/joe/tests/` and `bots/joe-rs/tests/` each
  pass in isolation.
- The default `pytest -q` aborts during collection with three import-file
  mismatches between `bots/morpheus-joe/tests/` and `bots/morpheus-rs/tests/`,
  which share test basenames. Because collection aborts, the 15 s suite budget
  could not be checked on this tree.
