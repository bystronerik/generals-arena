"""Classify the opponent's strategy from observable signals.

Labels: "rusher" (early aggression), "boomer" (fast expand), "citier"
(castle-focused production), "turtler" (compact defense), "unknown".

Verbatim port of the generals-bot adaptive classifier; the only semantic
rename is cities → castles (`enemy_castles_seen`), and the signal matters
*more* here — the arena roster is full of castle-builders. Pure function of
an OpponentModel plus the current turn, so it unit-tests offline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from _common.oppmodel import OpponentModel


@dataclass
class Classification:
    label: str
    confidence: float  # 0..1
    scores: dict[str, float] = field(default_factory=dict)


def classify(model: OpponentModel, turn: int) -> Classification:
    """Score each opponent archetype from the model's signals."""
    scores = {"rusher": 0.0, "boomer": 0.0, "citier": 0.0, "turtler": 0.0}
    if turn < 40 or not model.opp_land:
        return Classification("unknown", 0.0, scores)

    tiles = model.opp_land[-1]
    total = model.opp_army[-1]
    my_tiles = model.my_land[-1] or 1
    tile_rate = model.opp_tile_rate(window=min(100, len(model.turns) - 1))
    concentration = model.biggest_enemy_stack / max(total, 1)

    # --- rusher: early contact, deep approach, concentrated army ----------
    # A deep incursion by a real stack is *sticky* evidence: an opponent who
    # has ever pushed a big stack to our doorstep can do it again, however
    # economy-shaped they look between waves.
    deep_incursion = model.closest_enemy_dist <= 8 and model.biggest_enemy_stack >= 15
    if model.first_contact_turn is not None and model.first_contact_turn < 120:
        scores["rusher"] += 1.0
    if deep_incursion:
        scores["rusher"] += 2.5
    elif model.closest_enemy_dist <= 12:
        scores["rusher"] += 0.5
    if concentration > 0.45 and model.biggest_enemy_stack > 15:
        scores["rusher"] += 1.0
    if turn < 250 and model.under_attack(threshold_dist=10):
        scores["rusher"] += 1.0

    # --- boomer: high sustained tile growth, few big stacks ---------------
    if tile_rate > 0.30:
        scores["boomer"] += 1.5
    elif tile_rate > 0.20:
        scores["boomer"] += 0.75
    if tiles > 1.3 * my_tiles:
        scores["boomer"] += 1.0
    if concentration < 0.25:
        scores["boomer"] += 0.5
    if model.first_contact_turn is None and turn > 150:
        scores["boomer"] += 0.5
    if deep_incursion:
        # Fast expansion plus a doorstep stack is a rusher's rebuild phase,
        # not an economy plan.
        scores["boomer"] = max(0.0, scores["boomer"] - 1.5)

    # --- citier: owns castles / production above land-implied -------------
    if model.enemy_castles_seen >= 1:
        scores["citier"] += 1.5 * model.enemy_castles_seen
    prod = _production_estimate(model)
    if prod is not None and prod > 0.65:  # per turn; general alone ≈ 0.5
        scores["citier"] += 1.0
    if model.enemy_castles_seen >= 1 and tiles < my_tiles:
        scores["citier"] += 0.5

    # --- turtler: small compact territory, army parked at home ------------
    if tile_rate < 0.12 and turn > 100:
        scores["turtler"] += 1.0
    if tiles < 0.7 * my_tiles and turn > 150:
        scores["turtler"] += 1.0
    if (
        model.first_contact_turn is None
        or model.closest_enemy_dist > 12
    ) and turn > 150:
        scores["turtler"] += 0.5
    if total > 1.2 * tiles * 1.5 and concentration > 0.5 and not model.under_attack():
        scores["turtler"] += 0.5

    best = max(scores, key=lambda k: scores[k])
    ranked = sorted(scores.values(), reverse=True)
    margin = ranked[0] - ranked[1]
    if ranked[0] < 1.0:
        return Classification("unknown", 0.0, scores)
    confidence = min(1.0, ranked[0] / 4.0 + margin / 4.0)
    return Classification(best, confidence, scores)


def _production_estimate(model: OpponentModel, window: int = 48) -> float | None:
    """Opponent army production per turn over the last window, combat-free.

    Uses total-army growth between turns, ignoring land-bonus turns and any
    window where total dropped (combat). General-only ≈ 0.5/turn; each
    castle adds 0.5/turn (RULES.md §04 — the cadence matches the source).
    """
    if len(model.turns) < 8:
        return None
    n = min(window, len(model.turns) - 1)
    gains = []
    for i in range(-n, -1):
        t0, t1 = model.turns[i], model.turns[i + 1]
        if t1 % 50 == 0 or t0 % 50 == 0:
            continue  # land-bonus turn pollutes the estimate
        d_total = model.opp_army[i + 1] - model.opp_army[i]
        d_tiles = model.opp_land[i + 1] - model.opp_land[i]
        if d_total < 0 or d_tiles != 0:
            continue  # combat or capture in progress
        dt = t1 - t0
        if dt > 0:
            gains.append(d_total / dt)
    if len(gains) < 6:
        return None
    return sum(gains) / len(gains)
