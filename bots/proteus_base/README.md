# proteus_base — measurement twin, not a roster bot

A frozen copy of `bots/proteus/` at commit e68f2a0, kept only so the two
proteus versions have a **near-parity opponent** while the classifier rework
is being decided.

Why it exists: proteus beats every roster bot 80-100% of the time, and a game
at 80% carries a fraction of the Fisher information of a game at 50%. Routing
the A/B contrast through the ordinary panel needed ~13,000 games to reach
`SE(delta) = 12.75`; routing it through a twin at parity needs ~1,150, which
is the number `docs/arena/decision-rule.md` quotes for a balanced design.

It is **not** a roster member: it claims no axis in
`docs/research/strategies/diversity-constraints.md` §3.1, it is a byte copy of
another bot's decision code, and it must be deleted once the proteus verdict
is recorded. Do not tune it, do not build on it, do not cite its rating.
