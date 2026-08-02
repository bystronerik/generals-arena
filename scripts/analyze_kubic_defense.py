#!/usr/bin/env python3
"""Kubic defense / reaction measurement on fit wins + losses.

Observational only: never writes to data/games/, data/ratings/, or
data/remote_games/. Emits docs/research/measurements/grok-kubic-defense.{json,md}.

Definitions (all Manhattan unless noted):
  enemy_home_dist[t] = them.nearest_tile_dist = closest enemy tile to OUR general
  stack_home_dist[t] = manhattan(our max_stack_pos, our general)
  incursion[t]       = us.tiles_lost > 0
  proximity threat   = first tick enemy_home_dist <= D for D in {1,2,3,4,5,6,8}
  recall             = within LATENCY_WINDOW after threat, stack_home_dist falls
                       by >= 1 via a move/jump (not already on home)
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arena.instrument.replay.analysis import Analysis
from arena.instrument.replay.fog import sees_cell
from arena.instrument.replay.loader import Cell, Replay
from arena.instrument.replay.metrics import manhattan
from scripts.kubic_corpus import (
    MEASUREMENTS,
    corpus_meta,
    dist_summary,
    dump_json,
    iter_analyzed,
    percentile,
)

PLAYER = "Kubic"
LATENCY_WINDOW = 30
RECALL_DROP = 1  # home-dist must fall by at least this
HOME_NEAR = 2  # stack already "home" if dist <= this
PROX_THRESHOLDS = (1, 2, 3, 4, 5, 6, 8)
PRESSURE_WINDOW = 20  # ticks before/after threat for abandon metrics
INCURSION_MIN_LOST = 1
# Treat a multi-tile hit as a stronger signal for redirect measurement.
HEAVY_INCURSION = 3


def _stack_home_series(an: Analysis, us: int) -> list[int | None]:
    home = an.replay.generals[us]
    out: list[int | None] = []
    for entry in an.metrics:
        pos = entry.seats[us].max_stack_pos
        out.append(manhattan(pos, home) if pos else None)
    return out


def _enemy_home_series(an: Analysis, them: int) -> list[int | None]:
    return [e.seats[them].nearest_tile_dist for e in an.metrics]


def _first_reach(series: list[int | None], d: int) -> int | None:
    for t, v in enumerate(series):
        if v is not None and v <= d:
            return t
    return None


def _min_before_capture(
    series: list[int | None], an: Analysis, them: int
) -> int | None:
    """Min enemy_home_dist while our general still owned by us."""
    us = an.us
    home = an.replay.generals[us]
    best: int | None = None
    for t, v in enumerate(series):
        frame = an.replay.ticks[t]
        if frame.owners[home[0]][home[1]] != us:
            break
        if v is None:
            continue
        if best is None or v < best:
            best = v
    return best


def _recall_latency(
    stack_home: list[int | None],
    threat_tick: int,
    window: int = LATENCY_WINDOW,
) -> dict[str, Any] | None:
    """Return latency until stack_home_dist drops by RECALL_DROP, or None."""
    if threat_tick >= len(stack_home):
        return None
    base = stack_home[threat_tick]
    if base is None:
        return None
    already_home = base <= HOME_NEAR
    end = min(len(stack_home) - 1, threat_tick + window)
    for t in range(threat_tick + 1, end + 1):
        cur = stack_home[t]
        if cur is None:
            continue
        if cur <= base - RECALL_DROP:
            return {
                "latency": t - threat_tick,
                "stack_home_at_threat": base,
                "stack_home_at_recall": cur,
                "already_home": already_home,
                "recalled": True,
            }
    return {
        "latency": None,
        "stack_home_at_threat": base,
        "stack_home_at_recall": stack_home[end],
        "already_home": already_home,
        "recalled": False,
    }


def _visible_enemy_near_home(
    replay: Replay, us: int, tick: int, home: Cell, radius: int
) -> bool:
    """True if any enemy tile within Manhattan radius of home is in our vision."""
    frame = replay.ticks[tick]
    them = 1 - us
    rows, cols = replay.rows, replay.cols
    hr, hc = home
    for r in range(max(0, hr - radius), min(rows, hr + radius + 1)):
        for c in range(max(0, hc - radius), min(cols, hc + radius + 1)):
            if abs(r - hr) + abs(c - hc) > radius:
                continue
            if frame.owners[r][c] == them and sees_cell(frame, us, (r, c), rows, cols):
                return True
    return False


def _rate(n: int, d: int) -> float | None:
    return (n / d) if d else None


def _abandon_window(
    an: Analysis, us: int, them: int, t0: int, window: int = PRESSURE_WINDOW
) -> dict[str, Any]:
    """Compare expansion / castles / attack before vs after a threat tick."""
    metrics = an.metrics
    n = len(metrics)
    pre_lo = max(0, t0 - window)
    post_hi = min(n - 1, t0 + window)

    def gains(a: int, b: int) -> int:
        return sum(metrics[t].seats[us].tiles_gained for t in range(a, b + 1))

    def losses(a: int, b: int) -> int:
        return sum(metrics[t].seats[us].tiles_lost for t in range(a, b + 1))

    castles = an.events.of_kind("castle_built", player=us)
    pre_castles = sum(1 for e in castles if pre_lo <= e.tick < t0)
    post_castles = sum(1 for e in castles if t0 < e.tick <= post_hi)

    # Attack continuation: moves whose path target is enemy_general after t0.
    steps = an.steps[us]
    post_moves = [s for s in steps if t0 < s.tick <= post_hi and s.moved]
    toward_eg = sum(
        1 for s in post_moves if s.target_kind == "enemy_general" and s.toward
    )
    away_homeish = 0
    home = an.replay.generals[us]
    for s in post_moves:
        if s.pos is None:
            continue
        # previous step pos for delta-to-home
        prev = next((x for x in steps if x.tick == s.tick - 1), None)
        if prev is None or prev.pos is None:
            continue
        if manhattan(s.pos, home) < manhattan(prev.pos, home):
            away_homeish += 1  # actually toward home

    gen_before = metrics[t0].seats[us].general_army
    gen_after_t = min(n - 1, t0 + 10)
    gen_after = metrics[gen_after_t].seats[us].general_army
    army_before = metrics[t0].seats[us].army

    return {
        "pre_tiles_gained": gains(pre_lo, max(pre_lo, t0 - 1)),
        "post_tiles_gained": gains(t0 + 1, post_hi),
        "pre_tiles_lost": losses(pre_lo, max(pre_lo, t0 - 1)),
        "post_tiles_lost": losses(t0 + 1, post_hi),
        "pre_castles": pre_castles,
        "post_castles": post_castles,
        "post_moves": len(post_moves),
        "post_toward_enemy_general": toward_eg,
        "post_toward_home_moves": away_homeish,
        "general_army_at_threat": gen_before,
        "general_army_plus_10": gen_after,
        "general_army_delta_10": gen_after - gen_before,
        "total_army_at_threat": army_before,
        "home_reserve_frac": (gen_before / army_before) if army_before else None,
    }


def analyze_game(replay: Replay, an: Analysis) -> dict[str, Any]:
    us, them = an.us, an.them
    home = replay.generals[us]
    stack_home = _stack_home_series(an, us)
    enemy_home = _enemy_home_series(an, them)
    min_enemy = _min_before_capture(enemy_home, an, them)

    reach: dict[str, int | None] = {}
    for d in PROX_THRESHOLDS:
        reach[str(d)] = _first_reach(enemy_home, d)

    # Incursion episodes: contiguous runs of tiles_lost > 0, take start tick.
    incursion_starts: list[int] = []
    in_run = False
    for t, m in enumerate(an.metrics):
        lost = m.seats[us].tiles_lost
        if lost >= INCURSION_MIN_LOST:
            if not in_run:
                incursion_starts.append(t)
                in_run = True
        else:
            in_run = False

    heavy_starts: list[int] = []
    for t, m in enumerate(an.metrics):
        if m.seats[us].tiles_lost >= HEAVY_INCURSION:
            heavy_starts.append(t)

    # Sample first 5 incursions + all heavy ones (dedupe).
    sample_ticks = sorted(set(incursion_starts[:5] + heavy_starts[:5]))
    incursion_responses: list[dict[str, Any]] = []
    for t0 in sample_ticks:
        lost = an.metrics[t0].seats[us].tiles_lost
        eh = enemy_home[t0]
        recall = _recall_latency(stack_home, t0)
        visible = _visible_enemy_near_home(replay, us, t0, home, radius=8)
        row = {
            "tick": t0,
            "tiles_lost": lost,
            "enemy_home_dist": eh,
            "visible_threat": visible,
            "general_army": an.metrics[t0].seats[us].general_army,
            **(recall or {}),
        }
        if recall and recall.get("recalled") is False and not recall.get("already_home"):
            row["abandon"] = _abandon_window(an, us, them, t0)
        elif recall and recall.get("recalled"):
            row["abandon"] = _abandon_window(an, us, them, t0)
        incursion_responses.append(row)

    # Proximity-triggered recall at each D.
    prox_recall: dict[str, Any] = {}
    for d in PROX_THRESHOLDS:
        t0 = reach[str(d)]
        if t0 is None:
            prox_recall[str(d)] = None
            continue
        # Skip if general already gone.
        if replay.ticks[t0].owners[home[0]][home[1]] != us:
            prox_recall[str(d)] = {"tick": t0, "general_already_lost": True}
            continue
        recall = _recall_latency(stack_home, t0)
        abandon = _abandon_window(an, us, them, t0)
        visible = _visible_enemy_near_home(replay, us, t0, home, radius=d + 1)
        prox_recall[str(d)] = {
            "tick": t0,
            "visible": visible,
            "general_army": an.metrics[t0].seats[us].general_army,
            "enemy_home_dist": enemy_home[t0],
            **(recall or {}),
            "abandon": abandon,
        }

    streaks = [
        {
            "start": e.tick,
            "end": e.data.get("end"),
            "tiles_lost": e.data.get("tiles_lost"),
        }
        for e in an.events.of_kind("tile_loss_streak", player=us)
    ]

    # Peak home reserve and reserve under threat.
    gen_armies = [m.seats[us].general_army for m in an.metrics]
    peak_gen = max(gen_armies) if gen_armies else 0
    # At closest approach (pre-capture).
    closest_tick = None
    if min_enemy is not None:
        for t, v in enumerate(enemy_home):
            if v == min_enemy:
                if replay.ticks[t].owners[home[0]][home[1]] == us:
                    closest_tick = t
                    break

    gen_at_closest = (
        an.metrics[closest_tick].seats[us].general_army if closest_tick is not None else None
    )
    stack_at_closest = (
        stack_home[closest_tick] if closest_tick is not None else None
    )

    # Kill tick (general captured by them).
    kill_tick = None
    for t, frame in enumerate(replay.ticks):
        if frame.owners[home[0]][home[1]] == them:
            kill_tick = t
            break

    return {
        "match_id": str(replay.match_id),
        "outcome": replay.outcome,
        "opponent": replay.name(them),
        "ticks": replay.total_ticks,
        "min_enemy_home_dist_pre_capture": min_enemy,
        "first_reach": reach,
        "reached_le": {
            str(d): reach[str(d)] is not None for d in PROX_THRESHOLDS
        },
        "n_incursion_episodes": len(incursion_starts),
        "n_heavy_incursions": len(heavy_starts),
        "incursion_responses": incursion_responses,
        "prox_recall": prox_recall,
        "tile_loss_streaks": streaks,
        "peak_general_army": peak_gen,
        "general_army_at_closest": gen_at_closest,
        "stack_home_at_closest": stack_at_closest,
        "closest_tick": closest_tick,
        "kill_tick": kill_tick,
        "n_castles": len(an.events.of_kind("castle_built", player=us)),
        "first_contact": (
            an.events.of_kind("first_contact")[0].tick
            if an.events.of_kind("first_contact")
            else None
        ),
    }


def _collect(which: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for replay, an in iter_analyzed(which):
        rows.append(analyze_game(replay, an))
        if len(rows) % 50 == 0:
            print(f"  {which}: {len(rows)} games…", flush=True)
    return rows


def _agg_reach_rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    out: dict[str, Any] = {"n_games": n}
    for d in PROX_THRESHOLDS:
        hits = sum(1 for r in rows if r["reached_le"][str(d)])
        out[f"frac_reached_<={d}"] = {
            "count": hits,
            "n": n,
            "rate": _rate(hits, n),
            "tag": "MEASURED",
        }
    mins = [
        r["min_enemy_home_dist_pre_capture"]
        for r in rows
        if r["min_enemy_home_dist_pre_capture"] is not None
    ]
    out["min_enemy_home_dist"] = {**dist_summary(mins), "tag": "MEASURED"}
    return out


def _agg_incursion_redirect(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies: list[float] = []
    stack_at: list[float] = []
    recalled = 0
    already_home = 0
    no_recall = 0
    total = 0
    visible = 0
    by_enemy_dist: dict[str, Counter] = defaultdict(Counter)

    for r in rows:
        for resp in r["incursion_responses"]:
            total += 1
            if resp.get("visible_threat"):
                visible += 1
            sh = resp.get("stack_home_at_threat")
            if sh is not None:
                stack_at.append(float(sh))
            if resp.get("already_home"):
                already_home += 1
                bucket = "already_home"
            elif resp.get("recalled"):
                recalled += 1
                bucket = "recalled"
                if resp.get("latency") is not None:
                    latencies.append(float(resp["latency"]))
            else:
                no_recall += 1
                bucket = "no_recall"
            ed = resp.get("enemy_home_dist")
            key = str(ed) if ed is not None else "none"
            by_enemy_dist[key][bucket] += 1

    return {
        "n_incursion_samples": total,
        "visible_threat_frac": {
            "count": visible,
            "n": total,
            "rate": _rate(visible, total),
            "tag": "MEASURED",
        },
        "already_home_frac": {
            "count": already_home,
            "n": total,
            "rate": _rate(already_home, total),
            "tag": "MEASURED",
        },
        "recall_frac_among_away": {
            "count": recalled,
            "n": recalled + no_recall,
            "rate": _rate(recalled, recalled + no_recall),
            "tag": "MEASURED",
            "note": (
                f"Recall = stack_home_dist drops by >={RECALL_DROP} within "
                f"{LATENCY_WINDOW} ticks; excludes already_home (dist<={HOME_NEAR})."
            ),
        },
        "no_recall_count": no_recall,
        "latency_ticks": {**dist_summary(latencies), "tag": "MEASURED"},
        "stack_home_dist_at_threat": {**dist_summary(stack_at), "tag": "MEASURED"},
        "by_enemy_home_dist_at_incursion": {
            k: dict(v) for k, v in sorted(by_enemy_dist.items(), key=lambda x: (x[0] == "none", int(x[0]) if x[0].isdigit() else 999))
        },
    }


def _agg_prox_recall(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for d in PROX_THRESHOLDS:
        latencies: list[float] = []
        stack_at: list[float] = []
        gen_at: list[float] = []
        recalled = already = none = visible = 0
        n_reach = 0
        post_toward_eg: list[float] = []
        post_toward_home: list[float] = []
        post_gain: list[float] = []
        pre_gain: list[float] = []
        post_castles: list[float] = []
        pre_castles: list[float] = []
        gen_delta: list[float] = []
        reserve_frac: list[float] = []

        for r in rows:
            pr = r["prox_recall"].get(str(d))
            if pr is None or pr.get("general_already_lost"):
                continue
            n_reach += 1
            if pr.get("visible"):
                visible += 1
            sh = pr.get("stack_home_at_threat")
            if sh is not None:
                stack_at.append(float(sh))
            ga = pr.get("general_army")
            if ga is not None:
                gen_at.append(float(ga))
            if pr.get("already_home"):
                already += 1
            elif pr.get("recalled"):
                recalled += 1
                if pr.get("latency") is not None:
                    latencies.append(float(pr["latency"]))
            else:
                none += 1
            ab = pr.get("abandon") or {}
            if ab.get("post_moves"):
                post_toward_eg.append(
                    ab["post_toward_enemy_general"] / ab["post_moves"]
                )
                post_toward_home.append(
                    ab["post_toward_home_moves"] / ab["post_moves"]
                )
            if "post_tiles_gained" in ab:
                post_gain.append(float(ab["post_tiles_gained"]))
                pre_gain.append(float(ab["pre_tiles_gained"]))
            if "post_castles" in ab:
                post_castles.append(float(ab["post_castles"]))
                pre_castles.append(float(ab["pre_castles"]))
            if ab.get("general_army_delta_10") is not None:
                gen_delta.append(float(ab["general_army_delta_10"]))
            if ab.get("home_reserve_frac") is not None:
                reserve_frac.append(float(ab["home_reserve_frac"]))

        out[str(d)] = {
            "n_games_reached": n_reach,
            "visible_frac": {
                "count": visible,
                "n": n_reach,
                "rate": _rate(visible, n_reach),
                "tag": "MEASURED",
            },
            "already_home_frac": {
                "count": already,
                "n": n_reach,
                "rate": _rate(already, n_reach),
                "tag": "MEASURED",
            },
            "recall_frac_among_away": {
                "count": recalled,
                "n": recalled + none,
                "rate": _rate(recalled, recalled + none),
                "tag": "MEASURED",
            },
            "no_recall_count": none,
            "latency_ticks": {**dist_summary(latencies), "tag": "MEASURED"},
            "stack_home_dist_at_threat": {
                **dist_summary(stack_at),
                "tag": "MEASURED",
            },
            "general_army_at_threat": {**dist_summary(gen_at), "tag": "MEASURED"},
            "home_reserve_frac_at_threat": {
                **dist_summary(reserve_frac),
                "tag": "MEASURED",
            },
            "general_army_delta_10": {
                **dist_summary(gen_delta),
                "tag": "MEASURED",
            },
            "abandon_under_pressure": {
                "pre_tiles_gained_in_window": {
                    **dist_summary(pre_gain),
                    "tag": "MEASURED",
                },
                "post_tiles_gained_in_window": {
                    **dist_summary(post_gain),
                    "tag": "MEASURED",
                },
                "pre_castles_in_window": {
                    **dist_summary(pre_castles),
                    "tag": "MEASURED",
                },
                "post_castles_in_window": {
                    **dist_summary(post_castles),
                    "tag": "MEASURED",
                },
                "frac_post_moves_toward_enemy_general": {
                    **dist_summary(post_toward_eg),
                    "tag": "MEASURED",
                },
                "frac_post_moves_toward_home": {
                    **dist_summary(post_toward_home),
                    "tag": "MEASURED",
                },
                "window_ticks": PRESSURE_WINDOW,
            },
        }
    return out


def _agg_streaks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_with = sum(1 for r in rows if r["tile_loss_streaks"])
    depths = [
        float(s["tiles_lost"])
        for r in rows
        for s in r["tile_loss_streaks"]
        if s.get("tiles_lost") is not None
    ]
    return {
        "games_with_streak": {
            "count": n_with,
            "n": len(rows),
            "rate": _rate(n_with, len(rows)),
            "tag": "MEASURED",
        },
        "tiles_lost_in_streak": {**dist_summary(depths), "tag": "MEASURED"},
    }


def _agg_reserve(rows: list[dict[str, Any]]) -> dict[str, Any]:
    peaks = [float(r["peak_general_army"]) for r in rows]
    at_closest = [
        float(r["general_army_at_closest"])
        for r in rows
        if r["general_army_at_closest"] is not None
    ]
    stack_closest = [
        float(r["stack_home_at_closest"])
        for r in rows
        if r["stack_home_at_closest"] is not None
    ]
    return {
        "peak_general_army": {**dist_summary(peaks), "tag": "MEASURED"},
        "general_army_at_closest_enemy": {
            **dist_summary(at_closest),
            "tag": "MEASURED",
        },
        "stack_home_dist_at_closest_enemy": {
            **dist_summary(stack_closest),
            "tag": "MEASURED",
        },
    }


def _counterexamples(fit: list[dict], losses: list[dict]) -> dict[str, Any]:
    """Pick concrete match ids that stress each claim."""
    # Wins where enemy got very close (dist <= 2).
    close_wins = [
        r
        for r in fit
        if r["min_enemy_home_dist_pre_capture"] is not None
        and r["min_enemy_home_dist_pre_capture"] <= 2
    ]
    close_wins.sort(key=lambda r: r["min_enemy_home_dist_pre_capture"])

    # Wins with no recall on a heavy/prox threat while stack away.
    no_recall_wins = []
    for r in fit:
        for d in (3, 2, 1):
            pr = r["prox_recall"].get(str(d))
            if (
                pr
                and not pr.get("already_home")
                and pr.get("recalled") is False
                and not pr.get("general_already_lost")
            ):
                no_recall_wins.append(
                    {
                        "match_id": r["match_id"],
                        "d": d,
                        "tick": pr["tick"],
                        "stack_home": pr.get("stack_home_at_threat"),
                    }
                )
                break

    # Losses: how close before death, and whether recall fired at d=3.
    loss_skim = []
    for r in losses:
        pr3 = r["prox_recall"].get("3") or {}
        loss_skim.append(
            {
                "match_id": r["match_id"],
                "opponent": r["opponent"],
                "ticks": r["ticks"],
                "min_enemy_home_dist": r["min_enemy_home_dist_pre_capture"],
                "kill_tick": r["kill_tick"],
                "reached_le3": r["reached_le"]["3"],
                "reached_le2": r["reached_le"]["2"],
                "reached_le1": r["reached_le"]["1"],
                "prox3_recalled": pr3.get("recalled"),
                "prox3_already_home": pr3.get("already_home"),
                "prox3_latency": pr3.get("latency"),
                "prox3_stack_home": pr3.get("stack_home_at_threat"),
                "prox3_gen_army": pr3.get("general_army"),
                "n_streaks": len(r["tile_loss_streaks"]),
                "peak_general_army": r["peak_general_army"],
                "general_at_closest": r["general_army_at_closest"],
            }
        )

    return {
        "close_wins_enemy_le2": [
            {
                "match_id": r["match_id"],
                "min_dist": r["min_enemy_home_dist_pre_capture"],
                "closest_tick": r["closest_tick"],
                "gen_at_closest": r["general_army_at_closest"],
                "stack_home_at_closest": r["stack_home_at_closest"],
            }
            for r in close_wins[:15]
        ],
        "wins_no_recall_at_prox_le3": no_recall_wins[:15],
        "all_losses": loss_skim,
    }


def _infer_rules(fit_agg: dict, loss_agg: dict, cex: dict) -> list[dict[str, Any]]:
    """Tag candidate defense rules from aggregates."""
    rules: list[dict[str, Any]] = []
    fr = fit_agg["reach_rates"]
    lr = loss_agg["reach_rates"]

    rules.append(
        {
            "id": "D1_proximity_asymmetry",
            "claim": (
                "Enemy rarely reaches Manhattan <=3 of Kubic's general in fit wins; "
                "losses almost always do."
            ),
            "tag": "MEASURED",
            "fit": {
                k: fr[k]
                for k in (
                    "frac_reached_<=1",
                    "frac_reached_<=2",
                    "frac_reached_<=3",
                    "min_enemy_home_dist",
                )
            },
            "losses": {
                k: lr[k]
                for k in (
                    "frac_reached_<=1",
                    "frac_reached_<=2",
                    "frac_reached_<=3",
                    "min_enemy_home_dist",
                )
            },
        }
    )

    # Prefer threat-conditioned D<=3 (and <=5) over raw max-rate D.
    d3 = fit_agg["prox_recall"]["3"]["recall_frac_among_away"]
    d5 = fit_agg["prox_recall"]["5"]["recall_frac_among_away"]
    l3 = loss_agg["prox_recall"]["3"]["recall_frac_among_away"]
    best_d = 3
    best_rate = d3.get("rate")
    rules.append(
        {
            "id": "D2_recall_distance",
            "claim": (
                f"Fit wins that admit enemy_home_dist<=3 recall away stacks at "
                f"{(d3.get('rate') or 0):.0%} ({d3.get('count')}/{d3.get('n')}); "
                f"at <=5 the rate is {(d5.get('rate') or 0):.0%} "
                f"({d5.get('count')}/{d5.get('n')}). "
                f"Losses at <=3 recall only {(l3.get('rate') or 0):.0%} "
                f"({l3.get('count')}/{l3.get('n')}). "
                "Exact trigger D remains under-sampled in wins."
            ),
            "tag": "MEASURED",
            "best_D": best_d,
            "best_recall_rate": best_rate,
            "note": (
                "Larger D can show higher raw recall rates from mid-map maneuver; "
                "prefer D<=3/<=5 as threat-conditioned."
            ),
            "per_D_fit": {
                str(d): {
                    "n_reached": fit_agg["prox_recall"][str(d)]["n_games_reached"],
                    "recall_frac_away": fit_agg["prox_recall"][str(d)][
                        "recall_frac_among_away"
                    ],
                    "latency": fit_agg["prox_recall"][str(d)]["latency_ticks"],
                }
                for d in PROX_THRESHOLDS
            },
            "per_D_losses": {
                str(d): {
                    "n_reached": loss_agg["prox_recall"][str(d)]["n_games_reached"],
                    "recall_frac_away": loss_agg["prox_recall"][str(d)][
                        "recall_frac_among_away"
                    ],
                    "latency": loss_agg["prox_recall"][str(d)]["latency_ticks"],
                }
                for d in PROX_THRESHOLDS
            },
        }
    )

    inc = fit_agg["incursion_redirect"]
    rules.append(
        {
            "id": "D3_incursion_redirect",
            "claim": (
                "On sampled tile-loss episodes in fit wins, away max-stacks "
                f"redirect home in "
                f"{(inc['recall_frac_among_away'].get('rate') or 0):.0%} "
                f"of cases; median latency "
                f"{(inc['latency_ticks'].get('median'))} ticks; "
                f"median stack_home_dist at threat "
                f"{(inc['stack_home_dist_at_threat'].get('median'))}."
            ),
            "tag": "MEASURED",
            "fit": inc,
            "losses": loss_agg["incursion_redirect"],
        }
    )

    # Abandonment at D=3 if available else D=5.
    abandon_d = 3 if fit_agg["prox_recall"]["3"]["n_games_reached"] >= 5 else 5
    ab = fit_agg["prox_recall"][str(abandon_d)]["abandon_under_pressure"]
    rules.append(
        {
            "id": "D4_abandon_under_pressure",
            "claim": (
                f"After first enemy_home_dist<={abandon_d} in fit wins: "
                f"median pre/post tile gains "
                f"{ab['pre_tiles_gained_in_window'].get('median')}/"
                f"{ab['post_tiles_gained_in_window'].get('median')}; "
                f"median frac post moves toward enemy general "
                f"{ab['frac_post_moves_toward_enemy_general'].get('median')}; "
                f"toward home "
                f"{ab['frac_post_moves_toward_home'].get('median')}."
            ),
            "tag": "MEASURED",
            "D": abandon_d,
            "fit": ab,
            "losses": loss_agg["prox_recall"][str(abandon_d)]["abandon_under_pressure"],
        }
    )

    rules.append(
        {
            "id": "D5_home_reserve",
            "claim": (
                "Fit wins: peak general_army median "
                f"{fit_agg['reserve']['peak_general_army'].get('median')}, "
                "at closest enemy approach median "
                f"{fit_agg['reserve']['general_army_at_closest_enemy'].get('median')} "
                "(closest usually far). Losses: peak median "
                f"{loss_agg['reserve']['peak_general_army'].get('median')}, "
                "at closest (dist=1) median "
                f"{loss_agg['reserve']['general_army_at_closest_enemy'].get('median')} "
                "but stack_home_dist at closest median "
                f"{loss_agg['reserve']['stack_home_dist_at_closest_enemy'].get('median')} "
                f"vs fit {fit_agg['reserve']['stack_home_dist_at_closest_enemy'].get('median')} "
                "(stack not home — not an empty general)."
            ),
            "tag": "MEASURED",
            "fit": fit_agg["reserve"],
            "losses": loss_agg["reserve"],
        }
    )

    rules.append(
        {
            "id": "D6_loss_failure_modes",
            "claim": (
                "Primary defense failures appear in losses: enemy reaches <=1 "
                f"in {lr['frac_reached_<=1']['rate']:.0%} of losses vs "
                f"{fr['frac_reached_<=1']['rate']:.0%} of fit wins. "
                "See all_losses counterexamples."
            ),
            "tag": "MEASURED",
            "n_losses": len(cex["all_losses"]),
            "loss_ids": [x["match_id"] for x in cex["all_losses"]],
        }
    )

    rules.append(
        {
            "id": "D7_vision_gated_recall",
            "claim": (
                "Whether recall requires vision of the threatening tile is only "
                "partially measured (visible_threat on incursions / prox events). "
                "Causal vision→recall link remains uncertain without action logs."
            ),
            "tag": "INFERRED",
            "note": "Replay frames do not record intended policy; vision is reconstructed.",
        }
    )

    rules.append(
        {
            "id": "D8_exact_recall_threshold",
            "claim": (
                "Exact integer recall radius (policy threshold) is UNKNOWN: fit wins "
                "rarely admit enemy near home, so the trigger D is under-sampled; "
                "losses confound failed defense with late proximity."
            ),
            "tag": "UNKNOWN",
        }
    )

    return rules


def _write_md(payload: dict, path: Path) -> None:
    fit = payload["fit_wins"]
    loss = payload["losses"]
    rules = payload["rules"]
    cex = payload["counterexamples"]
    lines: list[str] = []
    lines.append("# Kubic defense / reaction (analyst #5)")
    lines.append("")
    lines.append(
        "Corpus: fit wins for rules; all seat-resolved losses for failure modes. "
        "Observational leaderboard replays only."
    )
    lines.append("")
    lines.append("## Corpus")
    lines.append("")
    lines.append(f"- Player: `{payload['player']}`")
    lines.append(f"- Fit wins analysed: **{fit['n_games']}**")
    lines.append(f"- Losses analysed: **{loss['n_games']}**")
    lines.append(
        f"- Split: `{payload['split_rule']}`"
    )
    lines.append(
        f"- Definitions: recall = `stack_home_dist` drops by ≥{RECALL_DROP} "
        f"within {LATENCY_WINDOW} ticks; already_home if dist ≤{HOME_NEAR}; "
        f"enemy_home_dist = `them.nearest_tile_dist`."
    )
    lines.append("")
    lines.append("## Top rules")
    lines.append("")
    for rule in rules:
        lines.append(f"### `{rule['id']}` — **{rule['tag']}**")
        lines.append("")
        lines.append(rule["claim"])
        lines.append("")

    lines.append("## Proximity rates (enemy to Kubic general)")
    lines.append("")
    lines.append("| D | fit wins rate | losses rate | tag |")
    lines.append("| --- | --- | --- | --- |")
    for d in (1, 2, 3):
        fk = f"frac_reached_<={d}"
        lines.append(
            f"| ≤{d} | {fit['reach_rates'][fk]['rate']:.3f} "
            f"({fit['reach_rates'][fk]['count']}/{fit['reach_rates'][fk]['n']}) | "
            f"{loss['reach_rates'][fk]['rate']:.3f} "
            f"({loss['reach_rates'][fk]['count']}/{loss['reach_rates'][fk]['n']}) | "
            f"MEASURED |"
        )
    lines.append("")
    lines.append(
        f"Min enemy_home_dist (pre-capture): fit "
        f"median={fit['reach_rates']['min_enemy_home_dist'].get('median')}, "
        f"p10={fit['reach_rates']['min_enemy_home_dist'].get('p10')}; "
        f"losses median={loss['reach_rates']['min_enemy_home_dist'].get('median')}, "
        f"p10={loss['reach_rates']['min_enemy_home_dist'].get('p10')}."
    )
    lines.append("")

    lines.append("## Recall by proximity threshold (fit)")
    lines.append("")
    lines.append(
        "| D | n reached | recall frac (away) | median latency | median stack_home | "
        "median gen_army |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for d in PROX_THRESHOLDS:
        b = fit["prox_recall"][str(d)]
        rf = b["recall_frac_among_away"]
        lines.append(
            f"| ≤{d} | {b['n_games_reached']} | "
            f"{(rf.get('rate') if rf.get('rate') is not None else 'n/a')} "
            f"({rf.get('count')}/{rf.get('n')}) | "
            f"{b['latency_ticks'].get('median')} | "
            f"{b['stack_home_dist_at_threat'].get('median')} | "
            f"{b['general_army_at_threat'].get('median')} |"
        )
    lines.append("")

    lines.append("## Recall by proximity threshold (losses)")
    lines.append("")
    lines.append(
        "| D | n reached | recall frac (away) | median latency | median stack_home | "
        "median gen_army |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for d in PROX_THRESHOLDS:
        b = loss["prox_recall"][str(d)]
        rf = b["recall_frac_among_away"]
        lines.append(
            f"| ≤{d} | {b['n_games_reached']} | "
            f"{(rf.get('rate') if rf.get('rate') is not None else 'n/a')} "
            f"({rf.get('count')}/{rf.get('n')}) | "
            f"{b['latency_ticks'].get('median')} | "
            f"{b['stack_home_dist_at_threat'].get('median')} | "
            f"{b['general_army_at_threat'].get('median')} |"
        )
    lines.append("")

    lines.append("## Incursion redirect")
    lines.append("")
    for label, block in (("fit wins", fit["incursion_redirect"]), ("losses", loss["incursion_redirect"])):
        lines.append(f"### {label}")
        lines.append("")
        lines.append(
            f"- Samples: {block['n_incursion_samples']}; "
            f"visible_threat rate={block['visible_threat_frac'].get('rate')}; "
            f"already_home={block['already_home_frac'].get('rate')}; "
            f"recall among away={block['recall_frac_among_away'].get('rate')} "
            f"({block['recall_frac_among_away'].get('count')}/"
            f"{block['recall_frac_among_away'].get('n')})."
        )
        lines.append(
            f"- Latency ticks: median={block['latency_ticks'].get('median')}, "
            f"p25={block['latency_ticks'].get('p25')}, "
            f"p75={block['latency_ticks'].get('p75')}."
        )
        lines.append(
            f"- Stack home dist at threat: "
            f"median={block['stack_home_dist_at_threat'].get('median')}, "
            f"p90={block['stack_home_dist_at_threat'].get('p90')}."
        )
        lines.append("")

    lines.append("## Abandoned under pressure (fit, first reach ≤3 or ≤5)")
    lines.append("")
    for d in (3, 5):
        ab = fit["prox_recall"][str(d)]["abandon_under_pressure"]
        if fit["prox_recall"][str(d)]["n_games_reached"] == 0:
            continue
        lines.append(f"### First enemy_home_dist ≤{d} (fit)")
        lines.append("")
        lines.append(
            f"- Tile gains window±{PRESSURE_WINDOW}: pre median="
            f"{ab['pre_tiles_gained_in_window'].get('median')}, post median="
            f"{ab['post_tiles_gained_in_window'].get('median')}."
        )
        lines.append(
            f"- Castles in window: pre median="
            f"{ab['pre_castles_in_window'].get('median')}, post median="
            f"{ab['post_castles_in_window'].get('median')}."
        )
        lines.append(
            f"- Frac post moves toward enemy general: median="
            f"{ab['frac_post_moves_toward_enemy_general'].get('median')}; "
            f"toward home: median="
            f"{ab['frac_post_moves_toward_home'].get('median')}."
        )
        lines.append("")

    lines.append("## Home army reserve")
    lines.append("")
    lines.append(
        f"- Fit peak general_army: median="
        f"{fit['reserve']['peak_general_army'].get('median')}, "
        f"p10={fit['reserve']['peak_general_army'].get('p10')}."
    )
    lines.append(
        f"- Fit general_army at closest enemy: median="
        f"{fit['reserve']['general_army_at_closest_enemy'].get('median')}."
    )
    lines.append(
        f"- Loss peak general_army: median="
        f"{loss['reserve']['peak_general_army'].get('median')}."
    )
    lines.append(
        f"- Loss general_army at closest enemy: median="
        f"{loss['reserve']['general_army_at_closest_enemy'].get('median')}."
    )
    lines.append("")

    lines.append("## Tile-loss streaks")
    lines.append("")
    lines.append(
        f"- Fit games with streak: "
        f"{fit['streaks']['games_with_streak'].get('rate')} "
        f"({fit['streaks']['games_with_streak'].get('count')}/"
        f"{fit['streaks']['games_with_streak'].get('n')})."
    )
    lines.append(
        f"- Loss games with streak: "
        f"{loss['streaks']['games_with_streak'].get('rate')} "
        f"({loss['streaks']['games_with_streak'].get('count')}/"
        f"{loss['streaks']['games_with_streak'].get('n')})."
    )
    lines.append("")

    lines.append("## Losses (failure-mode skim)")
    lines.append("")
    lines.append(
        "| match | opponent | ticks | min_d | ≤3/≤2/≤1 | prox3 recall | "
        "latency | stack_home | gen@threat | gen@closest |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for x in cex["all_losses"]:
        lines.append(
            f"| {x['match_id']} | {x['opponent']} | {x['ticks']} | "
            f"{x['min_enemy_home_dist']} | "
            f"{int(x['reached_le3'])}/{int(x['reached_le2'])}/{int(x['reached_le1'])} | "
            f"{x['prox3_recalled']} (home={x['prox3_already_home']}) | "
            f"{x['prox3_latency']} | {x['prox3_stack_home']} | "
            f"{x['prox3_gen_army']} | {x['general_at_closest']} |"
        )
    lines.append("")

    lines.append("## Counterexamples")
    lines.append("")
    lines.append("### Fit wins with enemy ≤2 of home")
    lines.append("")
    if not cex["close_wins_enemy_le2"]:
        lines.append("None.")
    else:
        for x in cex["close_wins_enemy_le2"]:
            lines.append(
                f"- `{x['match_id']}` min_dist={x['min_dist']} "
                f"tick={x['closest_tick']} gen={x['gen_at_closest']} "
                f"stack_home={x['stack_home_at_closest']}"
            )
    lines.append("")
    lines.append("### Fit wins with no recall at prox ≤3 (stack away)")
    lines.append("")
    if not cex["wins_no_recall_at_prox_le3"]:
        lines.append("None in sample.")
    else:
        for x in cex["wins_no_recall_at_prox_le3"][:12]:
            lines.append(
                f"- `{x['match_id']}` D≤{x['d']} tick={x['tick']} "
                f"stack_home={x['stack_home']}"
            )
    lines.append("")

    lines.append("## Unknowns")
    lines.append("")
    lines.append(
        "- Exact policy recall radius (integer D) — **UNKNOWN** (under-sampled in wins)."
    )
    lines.append(
        "- Whether recall is vision-gated vs omniscient proximity — **INFERRED** only."
    )
    lines.append(
        "- Half-moves / leave-1 vs full pulls on defense — not decoded here "
        "(see `kubic_moves.py` if needed)."
    )
    lines.append(
        "- Intentional abandon of castle builds vs coincidence of phase — "
        "**INFERRED** from pre/post counts only."
    )
    lines.append("")
    lines.append("## Paths")
    lines.append("")
    lines.append(f"- Script: `scripts/analyze_kubic_defense.py`")
    lines.append(f"- JSON: `docs/research/measurements/grok-kubic-defense.json`")
    lines.append(f"- This report: `docs/research/measurements/grok-kubic-defense.md`")
    lines.append(f"- Corpus split: `docs/research/measurements/grok-kubic-corpus-split.json`")
    lines.append("")

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    meta = corpus_meta()
    print(f"Analysing fit wins ({len(meta.fit_win_ids)})…", flush=True)
    fit_rows = _collect("fit")
    print(f"Analysing losses ({len(meta.loss_ids)})…", flush=True)
    loss_rows = _collect("losses")

    fit_agg = {
        "n_games": len(fit_rows),
        "reach_rates": _agg_reach_rates(fit_rows),
        "incursion_redirect": _agg_incursion_redirect(fit_rows),
        "prox_recall": _agg_prox_recall(fit_rows),
        "streaks": _agg_streaks(fit_rows),
        "reserve": _agg_reserve(fit_rows),
    }
    loss_agg = {
        "n_games": len(loss_rows),
        "reach_rates": _agg_reach_rates(loss_rows),
        "incursion_redirect": _agg_incursion_redirect(loss_rows),
        "prox_recall": _agg_prox_recall(loss_rows),
        "streaks": _agg_streaks(loss_rows),
        "reserve": _agg_reserve(loss_rows),
    }
    cex = _counterexamples(fit_rows, loss_rows)
    rules = _infer_rules(fit_agg, loss_agg, cex)

    payload = {
        "player": PLAYER,
        "analyst": 5,
        "topic": "defense_and_reaction",
        "split_rule": meta.as_json()["split_rule"],
        "definitions": {
            "enemy_home_dist": "them.nearest_tile_dist (Manhattan to our general)",
            "stack_home_dist": "manhattan(our max_stack_pos, our general)",
            "recall": (
                f"stack_home_dist drops by >={RECALL_DROP} within "
                f"{LATENCY_WINDOW} ticks of threat"
            ),
            "already_home": f"stack_home_dist <= {HOME_NEAR} at threat",
            "incursion": f"tiles_lost >= {INCURSION_MIN_LOST}",
            "heavy_incursion": f"tiles_lost >= {HEAVY_INCURSION}",
            "pressure_window": PRESSURE_WINDOW,
            "min_enemy_home_dist": "pre-capture only (general still owned)",
        },
        "fit_wins": fit_agg,
        "losses": loss_agg,
        "rules": rules,
        "counterexamples": cex,
        "per_game_losses": loss_rows,
        # Keep fit per-game compact: ids + key fields only (full responses in aggregates).
        "fit_win_summaries": [
            {
                "match_id": r["match_id"],
                "min_enemy_home_dist_pre_capture": r[
                    "min_enemy_home_dist_pre_capture"
                ],
                "reached_le": r["reached_le"],
                "n_incursion_episodes": r["n_incursion_episodes"],
                "n_streaks": len(r["tile_loss_streaks"]),
                "peak_general_army": r["peak_general_army"],
                "general_army_at_closest": r["general_army_at_closest"],
            }
            for r in fit_rows
        ],
    }

    json_path = dump_json("grok-kubic-defense.json", payload)
    md_path = MEASUREMENTS / "grok-kubic-defense.md"
    _write_md(payload, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")

    # Console digest for the parent agent.
    print("\n=== DIGEST ===")
    for rule in rules:
        print(f"[{rule['tag']}] {rule['id']}: {rule['claim'][:160]}")


if __name__ == "__main__":
    main()
