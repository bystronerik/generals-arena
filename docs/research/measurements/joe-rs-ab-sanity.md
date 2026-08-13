# joe-rs A/B sanity check (J4)

**Claim tested:** at true parity, an A/B of joe vs joe-rs reads `no change`
(port-plan §7). This is a port-equivalence sanity check, not a strength
claim — nothing here feeds a keep/revert decision, so the decision-rule
replication requirement does not bind (and would be trivially satisfiable:
the arms are deterministic).

## Setup

- Round `joe-rs-ab-r1`, 2026-08-14, mode competition, engine at the
  registered pin; both arms in the same round per
  [decision-rule.md](../../arena/decision-rule.md).
- Arms: `joe@5309199f8fb9` and `joe-rs@0995d03a59c8`.
- Panel: aegis, macaria, boom, cm_hunter, cm_expander (anchor).
- Seeds 0–9 per pair, `--seat-policy alternate` → 20 games per pair per
  arm, 200 games total, stored under `data/games/joe-rs-ab-r1/`.
- Host note (decision-rule: host state is a variable): the joe arm ran
  while a Rust build was occasionally active; the joe-rs arm on an
  otherwise idle machine. Irrelevant here — no bot is deadline-driven, and
  the arms are compared transcript-first.

## Result, transcript level (primary)

Both bots are deterministic and the engine is seed-deterministic, so at
true parity the two arms must produce **identical games**. They do: over
all **100 matched (opponent, seed, seat) pairs**, winner and turn count are
equal pair by pair. Records: joe 99–1, joe-rs 99–1 (the one loss is the
same map both times: macaria, one seed, one seat).

## Result, rating contrast (secondary)

```
delta(joe -> joe-rs) = -33.34 ± 107.79   CI95 (-244.6, +177.9)   P(B>A) = 0.379
```

The interval is wide by construction — 99–1 arms carry almost no rating
information — and straddles zero: **verdict `no change`**, exactly the
reading port-plan §7 requires. The load-bearing equivalence evidence is not
this contrast but the parity gates ([joe-rs/parity.md](../../bots/joe-rs/parity.md)):
bit-exact tier 1 over every turn of every corpus game, pinned tier-2
bounds, and 5,693 recorded turns replayed through the shipped binary with
every reply equal to Python joe's.
