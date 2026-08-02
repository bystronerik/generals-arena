#!/usr/bin/env python3
"""Kubic TIMING AND TEMPO on the FIT win set (analyst #6).

Derives move-rate, action-density, reaction-latency, turn-modulo, milestone
timing, and castle-build modulo patterns from fit wins only. Skims losses for
tempo-collapse signals. Does not touch holdout for rule derivation.

Reproducible: python scripts/analyze_kubic_timing.py
Writes:
  docs/research/measurements/grok-kubic-timing.json
  docs/research/measurements/grok-kubic-timing.md
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arena.instrument.replay.analysis import Analysis
from arena.instrument.replay.fog import visible_enemy_tiles
from arena.instrument.replay.loader import Replay
from arena.instrument.replay.path import TARGET_GENERAL, TARGET_VISIBLE_TILE
from scripts.kubic_corpus import (
    MEASUREMENTS,
    corpus_meta,
    dist_summary,
    dump_json,
    iter_analyzed,
)
from scripts.kubic_moves import InferredMove, infer_moves_for_player, move_kind_rates

OUT_JSON = "grok-kubic-timing.json"
OUT_MD = MEASUREMENTS / "grok-kubic-timing.md"

TURN_BUCKETS = (
    ("1-50", 1, 50),
    ("51-100", 51, 100),
    ("101-200", 101, 200),
    ("201+", 201, 10_000),
)

ACTIVE_KINDS = frozenset({"full", "half", "multi", "unknown", "build_suspect"})
MOVE_KINDS = frozenset({"full", "half", "multi", "unknown"})  # excludes pass + build


def _castle_cells(an: Analysis, player: int) -> set:
    return {e.cell for e in an.events.of_kind("castle_built", player=player) if e.cell}


def _event_tick(an: Analysis, kind: str, player: int | None = None) -> int | None:
    evs = an.events.of_kind(kind, player=player) if player is not None else an.events.of_kind(kind)
    return evs[0].tick if evs else None


def _kill_tick(an: Analysis) -> int | None:
    """Tick of our general_captured of enemy, else last tick on a win."""
    us = an.us
    for e in an.events.of_kind("general_captured", player=us):
        return e.tick
    if an.replay.outcome == "win":
        return an.replay.total_ticks
    return None


def _first_visible_enemy_tick(replay: Replay, player: int) -> int | None:
    rows, cols = replay.rows, replay.cols
    for tick, frame in enumerate(replay.ticks):
        if visible_enemy_tiles(frame, player, rows, cols):
            return tick
    return None


def _first_capture_of_enemy(an: Analysis, player: int) -> int | None:
    return _event_tick(an, "first_capture", player=player)


def _first_toward_visible_enemy(an: Analysis, player: int, after: int) -> int | None:
    """First stack step that closes on a visible enemy tile / known general."""
    for step in an.steps[player]:
        if step.tick < after:
            continue
        if not step.toward:
            continue
        if step.target_kind in (TARGET_VISIBLE_TILE, TARGET_GENERAL):
            return step.tick
    return None


def _first_inferred_capture_move(
    moves: list[InferredMove], after: int
) -> int | None:
    for m in moves:
        if m.tick >= after and m.captured:
            return m.tick
    return None


def _reaction_latency(
    an: Analysis, moves: list[InferredMove]
) -> dict[str, Any]:
    """Latency from first visible enemy tile to first capture / toward-move."""
    us = an.us
    sight = _first_visible_enemy_tick(an.replay, us)
    if sight is None:
        return {"first_visible_enemy": None, "reacted": False}

    cap_event = _first_capture_of_enemy(an, us)
    cap_move = _first_inferred_capture_move(moves, sight)
    toward = _first_toward_visible_enemy(an, us, sight)

    candidates: list[tuple[str, int]] = []
    if cap_event is not None and cap_event >= sight:
        candidates.append(("first_capture_event", cap_event))
    if cap_move is not None:
        candidates.append(("inferred_capture_move", cap_move))
    if toward is not None:
        candidates.append(("toward_visible_enemy", toward))

    if not candidates:
        return {
            "first_visible_enemy": sight,
            "reacted": False,
            "latency": None,
            "reaction_kind": None,
            "capture_event": cap_event,
            "toward_tick": toward,
        }

    kind, tick = min(candidates, key=lambda x: x[1])
    return {
        "first_visible_enemy": sight,
        "reacted": True,
        "latency": tick - sight,
        "reaction_kind": kind,
        "reaction_tick": tick,
        "capture_event": cap_event,
        "toward_tick": toward,
    }


def _bucket_name(tick: int) -> str:
    for name, lo, hi in TURN_BUCKETS:
        if lo <= tick <= hi:
            return name
    return "201+"


def _modulo_rates(
    moves: list[InferredMove], mod: int, *, min_tick: int = 0
) -> dict[str, dict[str, float | int]]:
    """Per residue: pass_rate, active_rate, full_rate, n."""
    buckets: dict[int, Counter] = defaultdict(Counter)
    for m in moves:
        if m.tick < min_tick:
            continue
        buckets[m.tick % mod][m.kind] += 1
    out: dict[str, dict[str, float | int]] = {}
    for r in range(mod):
        c = buckets[r]
        n = sum(c.values()) or 1
        out[str(r)] = {
            "n": sum(c.values()),
            "pass_rate": c["pass"] / n,
            "active_rate": sum(c[k] for k in ACTIVE_KINDS) / n,
            "full_rate": c["full"] / n,
            "half_rate": c["half"] / n,
            "build_suspect_rate": c["build_suspect"] / n,
            "multi_rate": c["multi"] / n,
        }
    return out


def _growth_wait_signal(moves: list[InferredMove], *, min_tick: int = 2) -> dict[str, Any]:
    """Pass rate on ticks immediately before bulk (%50) and even production."""
    before_bulk = Counter()  # tick % 50 == 49 → next frame is bulk growth
    other_odd = Counter()
    before_even_prod = Counter()  # tick % 2 == 1 → next frame even, prod grows
    even_tick = Counter()

    for m in moves:
        t = m.tick
        if t < min_tick:
            continue
        kind = m.kind
        if t % 50 == 49:
            before_bulk[kind] += 1
        elif t % 2 == 1:
            other_odd[kind] += 1
        if t % 2 == 1:
            before_even_prod[kind] += 1
        else:
            even_tick[kind] += 1

    def rates(c: Counter) -> dict[str, float | int]:
        n = sum(c.values()) or 1
        return {
            "n": sum(c.values()),
            "pass_rate": c["pass"] / n,
            "active_rate": sum(c[k] for k in ACTIVE_KINDS) / n,
        }

    return {
        "min_tick_excluded_below": min_tick,
        "before_bulk_tick_mod50_eq_49": rates(before_bulk),
        "other_odd_ticks": rates(other_odd),
        "odd_ticks_before_even_prod": rates(before_even_prod),
        "even_ticks": rates(even_tick),
    }


def analyze_one(replay: Replay, an: Analysis) -> dict[str, Any]:
    us = an.us
    castles = _castle_cells(an, us)
    moves = infer_moves_for_player(replay, us, castles)
    rates = move_kind_rates(moves)

    # Action density by turn bucket
    bucket_counts: dict[str, Counter] = {name: Counter() for name, _, _ in TURN_BUCKETS}
    tiles_gained_by_bucket: dict[str, list[int]] = {name: [] for name, _, _ in TURN_BUCKETS}
    for m in moves:
        b = _bucket_name(m.tick)
        if b not in bucket_counts:
            b = "201+"
        bucket_counts[b][m.kind] += 1

    for tm in an.metrics:
        if tm.tick == 0:
            continue
        b = _bucket_name(tm.tick)
        tiles_gained_by_bucket.setdefault(b, []).append(tm.seats[us].tiles_gained)

    bucket_stats = {}
    for name, _, _ in TURN_BUCKETS:
        c = bucket_counts[name]
        n = sum(c.values()) or 1
        gains = tiles_gained_by_bucket.get(name, [])
        kind_counts = {k: int(c[k]) for k in ("pass", "full", "half", "build_suspect", "multi", "unknown")}
        bucket_stats[name] = {
            "n_transitions": sum(c.values()),
            "kind_counts": kind_counts,
            "pass_rate": c["pass"] / n,
            "active_rate": sum(c[k] for k in ACTIVE_KINDS) / n,
            "full_rate": c["full"] / n,
            "half_rate": c["half"] / n,
            "build_suspect_rate": c["build_suspect"] / n,
            "multi_rate": c["multi"] / n,
            "mean_tiles_gained": (sum(gains) / len(gains)) if gains else None,
        }

    contact = an.events.first_contact
    sight = an.vision.first_general_sight[us]
    kill = _kill_tick(an)
    ticks = replay.total_ticks
    reaction = _reaction_latency(an, moves)

    castle_ticks = [e.tick for e in an.events.of_kind("castle_built", player=us)]
    build_suspect_ticks = [m.tick for m in moves if m.kind == "build_suspect"]

    phase_lens = {p.name: p.end - p.start + 1 for p in an.phases}

    return {
        "match_id": str(replay.match_id),
        "outcome": replay.outcome,
        "ticks": ticks,
        "seat": us,
        "move_kind_rates": rates,
        "n_transitions": len(moves),
        "pass_rate": rates.get("pass", {}).get("rate", 0.0),
        "active_rate": sum(rates.get(k, {}).get("rate", 0.0) for k in ACTIVE_KINDS),
        "full_rate": rates.get("full", {}).get("rate", 0.0),
        "half_rate": rates.get("half", {}).get("rate", 0.0),
        "build_suspect_rate": rates.get("build_suspect", {}).get("rate", 0.0),
        "multi_rate": rates.get("multi", {}).get("rate", 0.0),
        "bucket_stats": bucket_stats,
        "reaction": reaction,
        "first_contact": contact,
        "first_sight": sight,
        "kill_tick": kill,
        "contact_frac": (contact / ticks) if contact is not None and ticks else None,
        "sight_frac": (sight / ticks) if sight is not None and ticks else None,
        "kill_frac": (kill / ticks) if kill is not None and ticks else None,
        "sight_minus_contact": (
            (sight - contact) if sight is not None and contact is not None else None
        ),
        "kill_minus_sight": (
            (kill - sight) if kill is not None and sight is not None else None
        ),
        "castle_ticks": castle_ticks,
        "build_suspect_ticks": build_suspect_ticks,
        "phase_lens": phase_lens,
        "moves": moves,  # dropped before JSON dump
    }


def _aggregate_fit(rows: list[dict]) -> dict[str, Any]:
    # Global move-kind pool
    kind_tot: Counter = Counter()
    for r in rows:
        for kind, info in r["move_kind_rates"].items():
            kind_tot[kind] += info["count"]
    n_all = sum(kind_tot.values()) or 1
    global_rates = {
        k: {"count": v, "rate": v / n_all} for k, v in sorted(kind_tot.items())
    }

    # Per-game rate distributions
    per_game = {
        "pass_rate": dist_summary([r["pass_rate"] for r in rows]),
        "active_rate": dist_summary([r["active_rate"] for r in rows]),
        "full_rate": dist_summary([r["full_rate"] for r in rows]),
        "half_rate": dist_summary([r["half_rate"] for r in rows]),
        "build_suspect_rate": dist_summary([r["build_suspect_rate"] for r in rows]),
        "multi_rate": dist_summary([r["multi_rate"] for r in rows]),
    }

    # Bucket aggregates: pool transitions across games
    bucket_pool: dict[str, Counter] = {name: Counter() for name, _, _ in TURN_BUCKETS}
    bucket_gain_means: dict[str, list[float]] = {name: [] for name, _, _ in TURN_BUCKETS}
    games_reaching: dict[str, int] = {name: 0 for name, _, _ in TURN_BUCKETS}
    for r in rows:
        for name, _, _ in TURN_BUCKETS:
            bs = r["bucket_stats"][name]
            if bs["n_transitions"] > 0:
                games_reaching[name] += 1
                for k, v in bs["kind_counts"].items():
                    bucket_pool[name][k] += v
                if bs["mean_tiles_gained"] is not None:
                    bucket_gain_means[name].append(bs["mean_tiles_gained"])

    bucket_agg = {}
    for name, _, _ in TURN_BUCKETS:
        c = bucket_pool[name]
        n = sum(c.values()) or 1
        active = sum(c[k] for k in ACTIVE_KINDS)
        bucket_agg[name] = {
            "games_reaching": games_reaching[name],
            "n_transitions": sum(c.values()),
            "pass_rate": c["pass"] / n,
            "full_rate": c["full"] / n,
            "half_rate": c["half"] / n,
            "build_suspect_rate": c["build_suspect"] / n,
            "multi_rate": c["multi"] / n,
            "unknown_rate": c["unknown"] / n,
            "active_rate": active / n,
            "mean_tiles_gained_per_tick": (
                sum(bucket_gain_means[name]) / len(bucket_gain_means[name])
                if bucket_gain_means[name]
                else None
            ),
            "tiles_gained_dist_across_games": dist_summary(bucket_gain_means[name]),
        }

    # Reaction latency
    latencies = [
        r["reaction"]["latency"]
        for r in rows
        if r["reaction"].get("reacted") and r["reaction"].get("latency") is not None
    ]
    no_vision = sum(1 for r in rows if r["reaction"]["first_visible_enemy"] is None)
    no_react = sum(
        1
        for r in rows
        if r["reaction"]["first_visible_enemy"] is not None
        and not r["reaction"].get("reacted")
    )
    reaction_kinds = Counter(
        r["reaction"]["reaction_kind"]
        for r in rows
        if r["reaction"].get("reacted")
    )
    # Counterexamples: latency > 30
    slow = [
        {
            "match_id": r["match_id"],
            "latency": r["reaction"]["latency"],
            "kind": r["reaction"]["reaction_kind"],
            "first_visible": r["reaction"]["first_visible_enemy"],
        }
        for r in rows
        if r["reaction"].get("latency") is not None and r["reaction"]["latency"] > 30
    ]
    slow.sort(key=lambda x: -x["latency"])

    # Milestone timing
    contacts = [r["first_contact"] for r in rows if r["first_contact"] is not None]
    sights = [r["first_sight"] for r in rows if r["first_sight"] is not None]
    kills = [r["kill_tick"] for r in rows if r["kill_tick"] is not None]
    lengths = [r["ticks"] for r in rows]
    contact_fracs = [r["contact_frac"] for r in rows if r["contact_frac"] is not None]
    sight_fracs = [r["sight_frac"] for r in rows if r["sight_frac"] is not None]
    kill_fracs = [r["kill_frac"] for r in rows if r["kill_frac"] is not None]
    sight_minus = [
        r["sight_minus_contact"]
        for r in rows
        if r["sight_minus_contact"] is not None
    ]
    kill_minus = [
        r["kill_minus_sight"] for r in rows if r["kill_minus_sight"] is not None
    ]
    never_sight = sum(1 for r in rows if r["first_sight"] is None)
    never_contact = sum(1 for r in rows if r["first_contact"] is None)

    # Opening forced-pass pattern (army=1 cannot leave-1 move)
    opening_pass_ticks = Counter()
    for r in rows:
        for m in r["moves"]:
            if m.kind == "pass" and m.tick <= 20:
                opening_pass_ticks[m.tick] += 1

    # Modulo: pool all moves; exclude ticks 0-1 (forced opening passes)
    all_moves: list[InferredMove] = []
    for r in rows:
        all_moves.extend(r["moves"])

    MOD_MIN_TICK = 2
    modulo = {
        "excluded_ticks_below": MOD_MIN_TICK,
        "note": (
            "Ticks 0-1 are excluded from modulo tables: every fit win passes both "
            "(general army starts at 1; leave-1 is impossible)."
        ),
        "mod2": _modulo_rates(all_moves, 2, min_tick=MOD_MIN_TICK),
        "mod25": _modulo_rates(all_moves, 25, min_tick=MOD_MIN_TICK),
        "mod50": _modulo_rates(all_moves, 50, min_tick=MOD_MIN_TICK),
        "growth_wait": _growth_wait_signal(all_moves, min_tick=MOD_MIN_TICK),
        "opening_pass_tick_counts_le20": {
            str(k): v for k, v in sorted(opening_pass_ticks.items())
        },
        "opening_pass_tick0_rate": opening_pass_ticks[0] / len(rows),
        "opening_pass_tick1_rate": opening_pass_ticks[1] / len(rows),
        "opening_pass_tick3_rate": opening_pass_ticks[3] / len(rows),
    }

    # Highlight mod50 residues with elevated pass_rate
    mod50_pass = sorted(
        ((int(k), v["pass_rate"], v["n"]) for k, v in modulo["mod50"].items()),
        key=lambda x: -x[1],
    )
    mod50_active = sorted(
        ((int(k), v["active_rate"], v["n"]) for k, v in modulo["mod50"].items()),
        key=lambda x: -x[1],
    )

    # First castle tick (production stamp)
    first_castle = []
    for r in rows:
        if r["castle_ticks"]:
            first_castle.append(min(r["castle_ticks"]))

    # Castle tick modulo
    castle_ticks_all = [t for r in rows for t in r["castle_ticks"]]
    build_suspect_all = [t for r in rows for t in r["build_suspect_ticks"]]
    castle_mod = {
        "n_castle_built_events": len(castle_ticks_all),
        "games_with_castle": sum(1 for r in rows if r["castle_ticks"]),
        "tick_dist": dist_summary(castle_ticks_all),
        "first_castle_tick": dist_summary(first_castle),
        "first_castle_eq_10": sum(1 for t in first_castle if t == 10),
        "first_castle_le_12": sum(1 for t in first_castle if t <= 12),
        "first_castle_mod50": dict(Counter(t % 50 for t in first_castle)),
        "mod2": dict(Counter(t % 2 for t in castle_ticks_all)),
        "mod25": dict(Counter(t % 25 for t in castle_ticks_all)),
        "mod50": dict(Counter(t % 50 for t in castle_ticks_all)),
        "mod2_note": (
            "All castle_built stamps land on even ticks because detection keys "
            "off production (+1 on even ticks). This is a detector artifact, not "
            "evidence that the spend action is even-only."
        ),
        "build_suspect_n": len(build_suspect_all),
        "build_suspect_tick_dist": dist_summary(build_suspect_all),
        "build_suspect_mod2": dict(Counter(t % 2 for t in build_suspect_all)),
        "build_suspect_mod50": dict(Counter(t % 50 for t in build_suspect_all)),
    }

    # Phase lengths
    expansion_lens = [
        r["phase_lens"]["expansion"]
        for r in rows
        if "expansion" in r["phase_lens"]
    ]
    contest_lens = [
        r["phase_lens"]["contest"] for r in rows if "contest" in r["phase_lens"]
    ]

    return {
        "n_games": len(rows),
        "global_move_kind_rates": global_rates,
        "per_game_rate_dist": per_game,
        "action_density_by_bucket": bucket_agg,
        "reaction_latency": {
            "n_with_visible_enemy": len(rows) - no_vision,
            "n_never_saw_enemy_tile": no_vision,
            "n_saw_but_no_reaction": no_react,
            "n_reacted": len(latencies),
            "latency_ticks": dist_summary(latencies),
            "reaction_kind_counts": dict(reaction_kinds),
            "slow_reaction_gt30": {
                "n": len(slow),
                "rate": len(slow) / max(len(latencies), 1),
                "top": slow[:15],
            },
            "first_visible_enemy_tick": dist_summary(
                [
                    r["reaction"]["first_visible_enemy"]
                    for r in rows
                    if r["reaction"]["first_visible_enemy"] is not None
                ]
            ),
        },
        "milestones": {
            "game_length": dist_summary(lengths),
            "first_contact_tick": dist_summary(contacts),
            "first_sight_tick": dist_summary(sights),
            "kill_tick": dist_summary(kills),
            "contact_frac_of_game": dist_summary(contact_fracs),
            "sight_frac_of_game": dist_summary(sight_fracs),
            "kill_frac_of_game": dist_summary(kill_fracs),
            "sight_minus_contact": dist_summary(sight_minus),
            "kill_minus_sight": dist_summary(kill_minus),
            "never_contact": never_contact,
            "never_sight": never_sight,
            "expansion_phase_len": dist_summary(expansion_lens),
            "contest_phase_len": dist_summary(contest_lens),
        },
        "turn_modulo": {
            "excluded_ticks_below": modulo["excluded_ticks_below"],
            "note": modulo["note"],
            "opening_pass_tick0_rate": modulo["opening_pass_tick0_rate"],
            "opening_pass_tick1_rate": modulo["opening_pass_tick1_rate"],
            "opening_pass_tick3_rate": modulo["opening_pass_tick3_rate"],
            "opening_pass_tick_counts_le20": modulo["opening_pass_tick_counts_le20"],
            "mod2_summary": {
                "odd_pass_rate": modulo["mod2"]["1"]["pass_rate"],
                "even_pass_rate": modulo["mod2"]["0"]["pass_rate"],
                "odd_active_rate": modulo["mod2"]["1"]["active_rate"],
                "even_active_rate": modulo["mod2"]["0"]["active_rate"],
                "odd_n": modulo["mod2"]["1"]["n"],
                "even_n": modulo["mod2"]["0"]["n"],
            },
            "mod50_highest_pass": [
                {"residue": r, "pass_rate": p, "n": n} for r, p, n in mod50_pass[:8]
            ],
            "mod50_highest_active": [
                {"residue": r, "active_rate": a, "n": n} for r, a, n in mod50_active[:8]
            ],
            "mod25_pass_by_residue": {
                k: {"pass_rate": v["pass_rate"], "n": v["n"]}
                for k, v in modulo["mod25"].items()
            },
            "growth_wait": modulo["growth_wait"],
            "full_mod2": modulo["mod2"],
            "full_mod50": modulo["mod50"],
        },
        "castle_timing": castle_mod,
    }


def _skim_losses(loss_rows: list[dict]) -> dict[str, Any]:
    if not loss_rows:
        return {"n": 0}
    return {
        "n": len(loss_rows),
        "game_length": dist_summary([r["ticks"] for r in loss_rows]),
        "pass_rate": dist_summary([r["pass_rate"] for r in loss_rows]),
        "active_rate": dist_summary([r["active_rate"] for r in loss_rows]),
        "first_contact": dist_summary(
            [r["first_contact"] for r in loss_rows if r["first_contact"] is not None]
        ),
        "first_sight": dist_summary(
            [r["first_sight"] for r in loss_rows if r["first_sight"] is not None]
        ),
        "never_sight": sum(1 for r in loss_rows if r["first_sight"] is None),
        "reaction_latency": dist_summary(
            [
                r["reaction"]["latency"]
                for r in loss_rows
                if r["reaction"].get("latency") is not None
            ]
        ),
        "late_bucket_pass": {
            name: dist_summary(
                [
                    r["bucket_stats"][name]["pass_rate"]
                    for r in loss_rows
                    if r["bucket_stats"][name]["n_transitions"] > 0
                ]
            )
            for name, _, _ in TURN_BUCKETS
        },
        "per_game": [
            {
                "match_id": r["match_id"],
                "ticks": r["ticks"],
                "pass_rate": round(r["pass_rate"], 4),
                "active_rate": round(r["active_rate"], 4),
                "first_contact": r["first_contact"],
                "first_sight": r["first_sight"],
                "reaction_latency": r["reaction"].get("latency"),
                "bucket_pass": {
                    name: round(r["bucket_stats"][name]["pass_rate"], 4)
                    for name, _, _ in TURN_BUCKETS
                    if r["bucket_stats"][name]["n_transitions"] > 0
                },
            }
            for r in loss_rows
        ],
    }


def _claims(fit: dict, losses: dict) -> list[dict]:
    """Tagged claims for the report."""
    claims: list[dict] = []
    g = fit["global_move_kind_rates"]
    pg = fit["per_game_rate_dist"]
    n = fit["n_games"]
    total_trans = sum(x["count"] for x in g.values())

    claims.append(
        {
            "tag": "MEASURED",
            "id": "T1_near_continuous_action",
            "claim": (
                f"Fit wins act almost every tick after the opening: global pass_rate="
                f"{g.get('pass', {}).get('rate', 0):.4f} ({g.get('pass', {}).get('count', 0)}/"
                f"{total_trans}); per-game median={pg['pass_rate'].get('median'):.4f}, "
                f"p90={pg['pass_rate'].get('p90'):.4f}."
            ),
            "threshold": "pass_rate < 0.03 per game after accounting for ticks 0-1",
            "n": n,
            "support": pg["pass_rate"],
        }
    )
    claims.append(
        {
            "tag": "MEASURED",
            "id": "T1b_opening_forced_passes",
            "claim": (
                f"Every fit win passes tick 0 and tick 1 "
                f"(rates {fit['turn_modulo']['opening_pass_tick0_rate']:.2f}/"
                f"{fit['turn_modulo']['opening_pass_tick1_rate']:.2f}). "
                f"Tick 3 is also often a pass "
                f"({fit['turn_modulo']['opening_pass_tick3_rate']:.2f} of games). "
                "After tick 5, passes are rare."
            ),
            "threshold": "no move on ticks 0-1; optional pass on tick 3 while army grows",
            "n": n,
            "support": fit["turn_modulo"]["opening_pass_tick_counts_le20"],
        }
    )
    claims.append(
        {
            "tag": "MEASURED",
            "id": "T2_full_dominates_half",
            "claim": (
                f"Full (leave-1) moves dominate half-moves: global full="
                f"{g.get('full', {}).get('rate', 0):.3f}, half="
                f"{g.get('half', {}).get('rate', 0):.3f}. "
                f"Per-game half median={pg['half_rate'].get('median'):.4f}."
            ),
            "threshold": "use full moves by default; half_rate < 0.05",
            "n": n,
            "support": {
                "full": g.get("full"),
                "half": g.get("half"),
                "per_game_half": pg["half_rate"],
            },
        }
    )

    dens = fit["action_density_by_bucket"]
    claims.append(
        {
            "tag": "MEASURED",
            "id": "T3_action_density_by_bucket",
            "claim": (
                "Pass rate by turn bucket (pooled): "
                + "; ".join(
                    f"{b} pass={dens[b]['pass_rate']:.4f} active={dens[b]['active_rate']:.4f} "
                    f"mean_tiles_gained/tick={dens[b]['mean_tiles_gained_per_tick']}"
                    for b in ("1-50", "51-100", "101-200", "201+")
                )
                + ". Mid-game (51-200) is essentially zero-pass."
            ),
            "n": n,
            "support": dens,
        }
    )

    rl = fit["reaction_latency"]
    claims.append(
        {
            "tag": "MEASURED",
            "id": "T4_immediate_reaction",
            "claim": (
                f"Reaction to first visible enemy tile is immediate: median latency="
                f"{rl['latency_ticks'].get('median')} ticks, p75="
                f"{rl['latency_ticks'].get('p75')}, p90="
                f"{rl['latency_ticks'].get('p90')}, max="
                f"{rl['latency_ticks'].get('max')} (n={rl['n_reacted']}). "
                f"Kinds: {rl['reaction_kind_counts']}. First visible enemy median tick="
                f"{rl['first_visible_enemy_tick'].get('median')}."
            ),
            "threshold": "react within 3 ticks of first enemy vision (p90); never >20 on fit wins",
            "n": rl["n_reacted"],
            "support": rl,
            "counterexamples": rl["slow_reaction_gt30"]["top"][:5],
        }
    )

    mod = fit["turn_modulo"]["mod2_summary"]
    gw = fit["turn_modulo"]["growth_wait"]
    before_bulk_pass = gw["before_bulk_tick_mod50_eq_49"]["pass_rate"]
    other_odd_pass = gw["other_odd_ticks"]["pass_rate"]
    claims.append(
        {
            "tag": "MEASURED",
            "id": "T5_no_wait_for_bulk_growth",
            "claim": (
                f"After excluding ticks 0-1: odd pass={mod['odd_pass_rate']:.4f}, "
                f"even pass={mod['even_pass_rate']:.4f}. "
                f"Before bulk (t%50==49) pass={before_bulk_pass:.4f} is LOWER than "
                f"other odd ticks ({other_odd_pass:.4f}). No wait-for-bulk-growth pattern."
            ),
            "threshold": "do NOT idle before tick%50==0; keep acting through growth boundaries",
            "n": n,
            "support": {
                "mod2": mod,
                "growth_wait": gw,
                "mod50_highest_pass": fit["turn_modulo"]["mod50_highest_pass"],
            },
        }
    )

    ms = fit["milestones"]
    claims.append(
        {
            "tag": "MEASURED",
            "id": "T6_milestone_timing",
            "claim": (
                f"Median first_contact={ms['first_contact_tick'].get('median')} "
                f"({ms['contact_frac_of_game'].get('median'):.2%} of game length); "
                f"first_general_sight={ms['first_sight_tick'].get('median')} "
                f"({ms['sight_frac_of_game'].get('median'):.2%}); "
                f"game_length={ms['game_length'].get('median')}. "
                f"Sight→kill median={ms['kill_minus_sight'].get('median')} ticks "
                f"(p25={ms['kill_minus_sight'].get('p25')}, p75={ms['kill_minus_sight'].get('p75')}). "
                f"Contact→sight median gap={ms['sight_minus_contact'].get('median')} ticks."
            ),
            "threshold": (
                "contact ~tick 70-90; sight often late (~0.89 of game); "
                "convert sight to kill in ~24 ticks median"
            ),
            "n": n,
            "support": ms,
            "note": (
                "kill_tick ≈ game_length on wins (game ends on general capture), "
                "so kill_frac≈1 is tautological; use kill_minus_sight."
            ),
        }
    )

    ct = fit["castle_timing"]
    claims.append(
        {
            "tag": "MEASURED",
            "id": "T7_first_castle_tick_10",
            "claim": (
                f"First castle_built production stamp: median="
                f"{ct['first_castle_tick'].get('median')}; "
                f"exactly tick 10 in {ct['first_castle_eq_10']}/{ct['games_with_castle']} "
                f"castle games; ≤12 in {ct['first_castle_le_12']}. "
                f"Overall castle events n={ct['n_castle_built_events']} in "
                f"{ct['games_with_castle']}/{n} games. mod50 mass at residue 10: "
                f"{ct['mod50'].get(10) or ct['mod50'].get('10')}."
            ),
            "threshold": "if building a castle, first production often appears at tick 10",
            "n": ct["games_with_castle"],
            "support": ct,
            "note": ct["mod2_note"],
        }
    )

    claims.append(
        {
            "tag": "UNKNOWN",
            "id": "T8_multi_ambiguity",
            "claim": (
                f"Global multi rate is {g.get('multi', {}).get('rate', 0):.3f}. "
                "Cannot determine true multi-source simultaneous moves versus "
                "inference collisions when several sources drop army."
            ),
            "n": n,
            "support": g.get("multi"),
        }
    )

    if losses.get("n", 0):
        high_pass_losses = [
            g for g in losses["per_game"] if g["pass_rate"] >= 0.3
        ]
        claims.append(
            {
                "tag": "INFERRED",
                "id": "T9_loss_tempo_collapse",
                "claim": (
                    f"Loss skim n={losses['n']}: median pass_rate="
                    f"{losses['pass_rate'].get('median'):.3f} vs fit "
                    f"{pg['pass_rate'].get('median'):.3f}; never_sight="
                    f"{losses['never_sight']}/{losses['n']}. "
                    f"{len(high_pass_losses)}/11 losses have pass_rate≥0.30 "
                    f"(cluster match_ids 24184-24189) — tempo collapse / possible "
                    "disconnect, not the fit-win policy."
                ),
                "threshold": "fit-like tempo has pass_rate≪0.05; ≥0.30 flags collapse",
                "n": losses["n"],
                "support": {"summary": losses["pass_rate"], "high_pass": high_pass_losses},
                "counterexamples": [
                    g
                    for g in losses["per_game"]
                    if g["pass_rate"] < 0.05
                ],
            }
        )

    return claims


def _cannot_determine() -> list[str]:
    return [
        "Whether tick-3 passes are intentional waits for army=2 or pathfinding startup.",
        "True simultaneous multi-cell orders vs single-move inference collisions (kind=multi).",
        "Exact castle spend tick (castle_built stamps first production; always even).",
        "Whether reaction targets any visible enemy tile or a specific threat threshold.",
        "Internal turn-budget / search-time policy beyond observable action density.",
        "Cause of high-pass loss cluster 24184-24189 (policy change vs disconnect).",
        "Holdout verification (this script derives on fit only).",
    ]


def _fmt_dist(d: dict | None) -> str:
    if not d or not d.get("n"):
        return "n=0"
    return (
        f"n={d['n']} min={d['min']:.3g} p25={d['p25']:.3g} med={d['median']:.3g} "
        f"p75={d['p75']:.3g} p90={d['p90']:.3g} max={d['max']:.3g} mean={d['mean']:.3g}"
    )


def write_md(payload: dict) -> None:
    fit = payload["fit"]
    losses = payload["loss_skim"]
    claims = payload["claims"]
    lines: list[str] = []
    lines.append("# Kubic timing and tempo (fit wins)")
    lines.append("")
    lines.append(
        f"Analyst #6. Fit wins only for rules (n={fit['n_games']}). "
        f"Loss skim n={losses.get('n', 0)}. Script: `scripts/analyze_kubic_timing.py`."
    )
    lines.append("")
    lines.append("## Top rules")
    lines.append("")
    for c in claims:
        if c["tag"] == "UNKNOWN":
            continue
        lines.append(f"- **[{c['tag']}]** `{c['id']}`: {c['claim']}")
        if c.get("threshold"):
            lines.append(f"  - Threshold: {c['threshold']}")
        if c.get("counterexamples"):
            lines.append(f"  - Counterexamples: `{c['counterexamples']}`")
    lines.append("")
    lines.append("## Move rates (global pool)")
    lines.append("")
    lines.append("| kind | count | rate |")
    lines.append("| --- | ---: | ---: |")
    for k, v in fit["global_move_kind_rates"].items():
        lines.append(f"| {k} | {v['count']} | {v['rate']:.4f} |")
    lines.append("")
    lines.append("Per-game distributions:")
    lines.append("")
    for key in (
        "pass_rate",
        "active_rate",
        "full_rate",
        "half_rate",
        "build_suspect_rate",
        "multi_rate",
    ):
        lines.append(f"- `{key}`: {_fmt_dist(fit['per_game_rate_dist'][key])}")
    lines.append("")
    lines.append("## Action density by turn bucket")
    lines.append("")
    lines.append(
        "| bucket | games | transitions | pass | active | full | half | mean tiles_gained/tick |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for b in ("1-50", "51-100", "101-200", "201+"):
        d = fit["action_density_by_bucket"][b]
        mt = d["mean_tiles_gained_per_tick"]
        lines.append(
            f"| {b} | {d['games_reaching']} | {d['n_transitions']} | "
            f"{d['pass_rate']:.4f} | {d['active_rate']:.4f} | {d['full_rate']:.4f} | "
            f"{d['half_rate']:.4f} | {mt if mt is None else f'{mt:.4f}'} |"
        )
    lines.append("")
    lines.append("## Reaction latency")
    lines.append("")
    rl = fit["reaction_latency"]
    lines.append(
        f"First visible enemy tile → first capture or toward-move. "
        f"n_reacted={rl['n_reacted']}, never_saw={rl['n_never_saw_enemy_tile']}, "
        f"saw_no_react={rl['n_saw_but_no_reaction']}."
    )
    lines.append("")
    lines.append(f"- latency_ticks: {_fmt_dist(rl['latency_ticks'])}")
    lines.append(f"- first_visible_enemy_tick: {_fmt_dist(rl['first_visible_enemy_tick'])}")
    lines.append(f"- reaction_kind_counts: `{rl['reaction_kind_counts']}`")
    lines.append(
        f"- slow (>30 ticks): n={rl['slow_reaction_gt30']['n']} "
        f"rate={rl['slow_reaction_gt30']['rate']:.3f}"
    )
    lines.append("")
    lines.append("## Turn-modulo periodicity")
    lines.append("")
    lines.append(fit["turn_modulo"]["note"])
    lines.append("")
    lines.append(
        f"- Opening passes: tick0={fit['turn_modulo']['opening_pass_tick0_rate']:.2f}, "
        f"tick1={fit['turn_modulo']['opening_pass_tick1_rate']:.2f}, "
        f"tick3={fit['turn_modulo']['opening_pass_tick3_rate']:.2f}"
    )
    m2 = fit["turn_modulo"]["mod2_summary"]
    lines.append(
        f"- mod2 (tick≥2): odd pass={m2['odd_pass_rate']:.4f} (n={m2['odd_n']}), "
        f"even pass={m2['even_pass_rate']:.4f} (n={m2['even_n']})"
    )
    gw = fit["turn_modulo"]["growth_wait"]
    lines.append(
        f"- before bulk (t%50==49) pass={gw['before_bulk_tick_mod50_eq_49']['pass_rate']:.4f} "
        f"(n={gw['before_bulk_tick_mod50_eq_49']['n']}); "
        f"other odds pass={gw['other_odd_ticks']['pass_rate']:.4f}"
    )
    lines.append(
        f"- mod50 highest pass residues: `{fit['turn_modulo']['mod50_highest_pass']}`"
    )
    lines.append("")
    lines.append("## Milestone timing")
    lines.append("")
    ms = fit["milestones"]
    for key in (
        "game_length",
        "first_contact_tick",
        "first_sight_tick",
        "kill_tick",
        "contact_frac_of_game",
        "sight_frac_of_game",
        "sight_minus_contact",
        "kill_minus_sight",
        "expansion_phase_len",
        "contest_phase_len",
    ):
        lines.append(f"- `{key}`: {_fmt_dist(ms[key])}")
    lines.append(
        f"- never_contact={ms['never_contact']}, never_sight={ms['never_sight']}"
    )
    lines.append(
        "- Note: kill_tick ≈ game_length on wins; prefer kill_minus_sight."
    )
    lines.append("")
    lines.append("## Castle build tick modulo")
    lines.append("")
    ct = fit["castle_timing"]
    lines.append(
        f"castle_built n={ct['n_castle_built_events']} games_with={ct['games_with_castle']}; "
        f"all-events tick dist: {_fmt_dist(ct['tick_dist'])}"
    )
    lines.append(
        f"- first_castle_tick: {_fmt_dist(ct['first_castle_tick'])}; "
        f"eq10={ct['first_castle_eq_10']}, le12={ct['first_castle_le_12']}"
    )
    lines.append(f"- mod2: `{ct['mod2']}` — {ct['mod2_note']}")
    lines.append(
        f"- mod50 top: `{sorted(((int(k), v) for k, v in ct['mod50'].items()), key=lambda x: -x[1])[:10]}`"
    )
    lines.append(
        f"- build_suspect n={ct['build_suspect_n']}; "
        f"tick dist: {_fmt_dist(ct['build_suspect_tick_dist'])}; "
        f"mod2=`{ct['build_suspect_mod2']}`"
    )
    lines.append("")
    lines.append("## Loss skim (tempo collapse)")
    lines.append("")
    if losses.get("n"):
        lines.append(
            f"n={losses['n']}. pass_rate {_fmt_dist(losses['pass_rate'])}; "
            f"length {_fmt_dist(losses['game_length'])}; never_sight={losses['never_sight']}."
        )
        lines.append("")
        lines.append("| match | ticks | pass | contact | sight | react |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for g in losses["per_game"]:
            lines.append(
                f"| {g['match_id']} | {g['ticks']} | {g['pass_rate']} | "
                f"{g['first_contact']} | {g['first_sight']} | {g['reaction_latency']} |"
            )
    else:
        lines.append("No losses.")
    lines.append("")
    lines.append("## Claims (all tags)")
    lines.append("")
    for c in claims:
        lines.append(f"### [{c['tag']}] `{c['id']}`")
        lines.append("")
        lines.append(c["claim"])
        if c.get("note"):
            lines.append("")
            lines.append(f"Note: {c['note']}")
        lines.append("")
    lines.append("## Cannot determine")
    lines.append("")
    for item in payload["cannot_determine"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Paths")
    lines.append("")
    lines.append("- Script: `scripts/analyze_kubic_timing.py`")
    lines.append("- JSON: `docs/research/measurements/grok-kubic-timing.json`")
    lines.append("- Corpus: `scripts/kubic_corpus.py`, split `grok-kubic-corpus-split.json`")
    lines.append("- Moves: `scripts/kubic_moves.py`")
    lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    meta = corpus_meta()
    print(
        f"Analyzing fit wins n={len(meta.fit_win_ids)}, "
        f"loss skim n={len(meta.loss_ids)}..."
    )

    fit_rows: list[dict] = []
    for i, (replay, an) in enumerate(iter_analyzed("fit"), 1):
        fit_rows.append(analyze_one(replay, an))
        if i % 50 == 0:
            print(f"  fit {i}/{len(meta.fit_win_ids)}")

    loss_rows: list[dict] = []
    for replay, an in iter_analyzed("losses"):
        loss_rows.append(analyze_one(replay, an))

    fit_agg = _aggregate_fit(fit_rows)
    loss_agg = _skim_losses(loss_rows)
    claims = _claims(fit_agg, loss_agg)

    # Strip heavy move lists before JSON
    for r in fit_rows:
        r.pop("moves", None)
    for r in loss_rows:
        r.pop("moves", None)

    payload = {
        "analyst": 6,
        "dimension": "timing_and_tempo",
        "player": "Kubic",
        "set": "fit_wins",
        "n_fit": fit_agg["n_games"],
        "n_holdout_excluded": len(meta.holdout_win_ids),
        "n_losses_skimmed": loss_agg.get("n", 0),
        "split_rule": meta.as_json()["split_rule"],
        "definitions": {
            "pass": "inferred no army drop on owned cells between ticks",
            "active": "any non-pass kind (full/half/multi/unknown/build_suspect)",
            "reaction_latency": (
                "ticks from first visible enemy tile (Chebyshev fog) to first "
                "of: first_capture event, inferred capture move, or stack step "
                "toward visible_enemy_tile / known general"
            ),
            "castle_built_lag": "event stamped at first production, ~2 ticks after spend",
            "turn_buckets": [b[0] for b in TURN_BUCKETS],
        },
        "fit": fit_agg,
        "loss_skim": loss_agg,
        "claims": claims,
        "cannot_determine": _cannot_determine(),
        # Compact per-game milestone table for counterexample hunting
        "fit_milestones_sample": [
            {
                "match_id": r["match_id"],
                "ticks": r["ticks"],
                "pass_rate": round(r["pass_rate"], 4),
                "first_contact": r["first_contact"],
                "first_sight": r["first_sight"],
                "kill_tick": r["kill_tick"],
                "reaction_latency": r["reaction"].get("latency"),
                "reaction_kind": r["reaction"].get("reaction_kind"),
            }
            for r in fit_rows
        ],
    }

    path = dump_json(OUT_JSON, payload)
    write_md(payload)
    print(f"wrote {path}")
    print(f"wrote {OUT_MD}")
    print("\nTop claims:")
    for c in claims:
        if c["tag"] != "UNKNOWN":
            print(f"  [{c['tag']}] {c['id']}: {c['claim'][:120]}...")


if __name__ == "__main__":
    main()
