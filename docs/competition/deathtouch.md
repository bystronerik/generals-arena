# Deathtouch

From turn **800**, a move that **executes** onto the enemy general's tile wins instantly. Army counts on the general do not matter. One unit is enough.

Code: `competition-module/generals/modifiers/deathtouch.py`. Default threshold `DEATHTOUCH_TURN = 800`; competition preset passes `deathtouch_turn=800`.

## Defense

- A **chase** from a third tile onto the attacker's **source** can stop the touch if it captures/strips the source so the touch never executes.
- Counter-attacking from the general itself is a head-on clash — the attacker typically wins that exchange.

## Draws

- Both players touching on the same turn → draw (`winner = -1`, episode done).
- Mutual general capture is a draw at **every** turn, not only after the threshold.

## Composition

Builds rewrite to passes before the deathtouch step. Only a real move (`pass=0`) can touch.

See also [`RULES.md`](../../RULES.md) section 07 and [`vs-classic.md`](vs-classic.md).
