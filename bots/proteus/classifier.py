"""Classify the opponent from what is observable through fog.

Labels — the set is deliberately small, because a label that cannot change
which core we play is not a classification, it is a comment:

- ``"aggressor"`` — has walked a sized stack at our general. Answer: blitz.
- ``"economy"``   — anything else, once there is evidence. Answer: boom.
- ``"unknown"``   — not enough evidence yet. Answer: the default spine.

The five source labels (rusher / boomer / citier / turtler) were measured
against the arena roster and collapsed. Two reasons, both recorded in
docs/research/strategies/proteus.md §3:

1. **Three of the four mapped to the same core.** rusher, boomer and citier
   all mapped to blitz, so three quarters of the classifier could not change
   a single move.
2. **The distinctions that survived do not change the answer.** A 32-game
   per-cell grid of every pure core against every roster bot put blitz and
   boom within noise of each other on the slow-expander and turtle clusters
   (both ≈1.00), so separating those buys nothing measurable.

What *does* change the answer is one question: will this opponent bring a
fist to our general, or will it spend its army on land? Blitz answers the
first, boom answers the second, and the gap is large in both directions
(blitz-against-blitz 0.50 versus boom-against-blitz 0.36; boom-against-aegis
0.86 versus blitz-against-aegis 0.75).

Pure function of an ``OpponentModel``, a :class:`~proteus.signals.HomePressure`
and the current turn, so it unit-tests offline with no engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from _common.oppmodel import OpponentModel
from proteus.signals import HomePressure

AGGRESSOR = "aggressor"
ECONOMY = "economy"
UNKNOWN = "unknown"

#: Smallest stack at our door that counts as a committed attack rather than
#: an expansion that happened to arrive. Below ~15 an opponent cannot take a
#: defended general, so the stack is a raid, not a strike.
DUEL_ARMY = 15

#: Latest turn a fist at our door may *first* arrive and still mean "this is
#: a rusher". Measured on the roster sweep: a genuine aggressor's first fist
#: lands at median turn 169 (blitz) or 181 (cm_hunter), whereas the identical
#: measurement against aegis (411), metro (378) and late_rush (278) is their
#: counterattack or their timed commit — a different animal that boom, not
#: blitz, is the right answer to. Without this cut aegis eventually latches
#: as an aggressor in 8 games of 12 and costs 0.11 of score.
DUEL_DEADLINE = 250

#: Turn from which the aggregates carry enough history to say anything. The
#: earliest committed contact on the roster is turn ~72; before turn 60 every
#: bot looks alike because every bot is still chain-expanding.
EVIDENCE_TURN = 60

#: Turns over which the economy claim saturates to full confidence.
ECONOMY_RAMP = 140

#: Structures (general + castles) above which the opponent is running a
#: castle programme. Validated against ground truth on the roster sweep:
#: bots holding no castles estimate 0-1 and bots holding 2-3 estimate 2-3.
#: Reported, not acted on — see `structure_estimate`.
CASTLE_STRUCTURES = 2


@dataclass
class Classification:
    label: str
    confidence: float  # 0..1
    scores: dict[str, float] = field(default_factory=dict)


def classify(
    model: OpponentModel,
    turn: int,
    pressure: HomePressure | None = None,
) -> Classification:
    """Which of the two answers this opponent needs, and how sure we are."""
    scores = {AGGRESSOR: 0.0, ECONOMY: 0.0}
    if turn < EVIDENCE_TURN or not model.opp_land:
        return Classification(UNKNOWN, 0.0, scores)

    # --- aggressor: a real fist has stood at our door ----------------------
    # Latched, not current: an opponent that walked one stack at us will walk
    # the next one too, and between waves it looks exactly like an economy.
    near = pressure.max_stack_near if pressure is not None else 0
    in_time = (
        pressure is not None
        and pressure.duel_turn is not None
        and pressure.duel_turn <= DUEL_DEADLINE
    )
    if near >= DUEL_ARMY and in_time:
        # Confidence grows with how far past the bar the fist was: a
        # 15-stack is ambiguous, a 40-stack is not.
        scores[AGGRESSOR] = min(1.0, 0.45 + (near - DUEL_ARMY) / 50.0)

    # --- economy: everything else, once the aggregates have spoken ---------
    # Deliberately not a positive test for economic play. "Has not brought a
    # fist" is the whole claim, and the grid says boom is the right answer to
    # all of it — castle programmes, slow expanders, turtles and the timed
    # committer alike.
    scores[ECONOMY] = min(1.0, (turn - EVIDENCE_TURN) / ECONOMY_RAMP)

    # Evidence of an attack outranks the absence of one, at any turn: the
    # economy score saturates at 1.0 by turn 200 and would otherwise drown a
    # fist that arrived later.
    if scores[AGGRESSOR] > 0.0:
        return Classification(AGGRESSOR, scores[AGGRESSOR], scores)
    if scores[ECONOMY] <= 0.0:
        return Classification(UNKNOWN, 0.0, scores)
    return Classification(ECONOMY, scores[ECONOMY], scores)


def structure_estimate(model: OpponentModel, window: int = 60) -> int | None:
    """Structures the opponent owns — general plus castles (RULES.md §04).

    Production is the *only* thing that raises an opponent's total army:
    moving onto neutral plain conserves it (§02 always leaves one behind) and
    combat only lowers it. §04 fires one army per structure every *other*
    turn, so in a combat-free stretch the per-turn ``opp_army`` delta
    alternates between 0 and the structure count. Dropping the every-50
    land-bonus turns and taking the 75th percentile lands on a production
    turn, which reads the count straight off — and discards the combat dips
    without having to detect combat at all.

    Fog-proof by construction: it counts castles that were never seen.
    Reported by the probe rather than consumed by :func:`classify`. It splits
    castle programmes cleanly on the roster sweep (no castles estimates 0-1,
    two or three castles estimates 2-3), but *both sides of that split want
    boom*, so acting on it would be motion without effect. It is recorded so
    that claim stays falsifiable.
    """
    if len(model.turns) < 16:
        return None
    lo = max(1, len(model.turns) - window)
    deltas = []
    for i in range(lo, len(model.turns)):
        t0, t1 = model.turns[i - 1], model.turns[i]
        if t1 % 50 == 0 or t0 % 50 == 0:
            continue  # land bonus pollutes the delta
        deltas.append(model.opp_army[i] - model.opp_army[i - 1])
    if len(deltas) < 12:
        return None
    deltas.sort()
    return max(0, deltas[int(0.75 * (len(deltas) - 1))])
