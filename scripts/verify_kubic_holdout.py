#!/usr/bin/env python3
"""Holdout verification of fit-derived Kubic behavioral rules.

Derive rules on fit only (see dimension scripts). This script checks falsifiable
predictions on holdout wins (every 10th sorted win id).

Reproduce:
  python3 scripts/verify_kubic_holdout.py

Writes:
  docs/research/measurements/grok-kubic-holdout-verification.json
  docs/research/measurements/grok-kubic-holdout-verification.md
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arena.instrument.replay.metrics import manhattan
from scripts.kubic_corpus import MEASUREMENTS, corpus_meta, dump_json, iter_analyzed
from scripts.kubic_moves import infer_moves_for_player


def components(replay, tick: int, player: int) -> int:
    owners = replay.ticks[tick].owners
    seen = set()
    n = 0
    rows, cols = replay.rows, replay.cols

    def neighbors(r, c):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols and owners[rr][cc] == player:
                yield rr, cc

    for r in range(rows):
        for c in range(cols):
            if owners[r][c] != player or (r, c) in seen:
                continue
            n += 1
            stack = [(r, c)]
            seen.add((r, c))
            while stack:
                cr, cc = stack.pop()
                for nr, nc in neighbors(cr, cc):
                    if (nr, nc) not in seen:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
    return n


def army_drop_before(replay, us, cell, stamp_tick, lookback=8):
    r, c = cell
    best = None
    for t in range(max(1, stamp_tick - lookback), stamp_tick + 1):
        prev, cur = replay.ticks[t - 1], replay.ticks[t]
        if prev.owners[r][c] != us or cur.owners[r][c] != us:
            continue
        drop = prev.armies[r][c] - cur.armies[r][c]
        if best is None or drop > best[0]:
            best = (drop, t)
    return best


def check_game(replay, an) -> dict:
    us, them = an.us, an.them
    gen = replay.generals[us]
    metrics = an.metrics
    events = an.events
    steps = an.steps[us]
    castles = {e.cell for e in events.of_kind("castle_built", player=us) if e.cell}
    moves = infer_moves_for_player(replay, us, castles)

    contact = events.first_contact
    sight_ev = events.of_kind("first_general_sight", player=us)
    sight = sight_ev[0].tick if sight_ev else None
    waves = events.of_kind("gather_wave", player=us)

    # pass 0/1
    pass01 = all(
        m.kind == "pass" for m in moves if m.tick in (0, 1)
    ) if len(moves) > 1 else False

    tiles50 = metrics[50].seats[us].tiles if len(metrics) > 50 else None
    army50 = metrics[50].seats[us].army if len(metrics) > 50 else None

    # contiguity at expansion end
    exp_end = (contact - 1) if contact else (len(replay.ticks) - 1)
    exp_end = max(0, min(exp_end, len(replay.ticks) - 1))
    comps = components(replay, exp_end, us)

    # army ratio at contact
    army_ratio = None
    if contact is not None and contact < len(metrics):
        ua = metrics[contact].seats[us].army
        ta = metrics[contact].seats[them].army
        army_ratio = ua / ta if ta > 0 else None

    # toward after sight
    toward_frac = None
    if sight is not None:
        directed = toward = 0
        for s in steps:
            if s.tick < sight or not s.moved or s.delta is None:
                continue
            directed += 1
            if s.delta < 0:
                toward += 1
        toward_frac = toward / directed if directed else None

    sight_to_kill = None
    if sight is not None:
        sight_to_kill = replay.total_ticks - sight

    tip_at_sight = None
    if sight is not None and sight < len(metrics):
        tip_at_sight = metrics[sight].seats[us].max_stack

    # pass rate
    pass_rate = sum(1 for m in moves if m.kind == "pass") / max(1, len(moves))
    actionable = [m for m in moves if m.kind in ("full", "half")]
    full_frac = (
        sum(1 for m in actionable if m.kind == "full") / len(actionable)
        if actionable
        else None
    )

    # enemy min dist to home
    min_d = 999
    for t in range(len(replay.ticks)):
        d = metrics[t].seats[them].nearest_tile_dist
        if d is not None and d < min_d:
            min_d = d
    if min_d == 999:
        min_d = None

    # reaction latency
    from arena.instrument.replay.fog import visible_enemy_tiles

    first_vis = None
    for t in range(len(replay.ticks)):
        vis = visible_enemy_tiles(replay.ticks[t], us, replay.rows, replay.cols)
        if vis:
            first_vis = t
            break
    react_lat = None
    if first_vis is not None:
        react = None
        for t in range(first_vis, len(replay.ticks)):
            if t == 0:
                continue
            prev, cur = replay.ticks[t - 1], replay.ticks[t]
            found = False
            for r in range(replay.rows):
                for c in range(replay.cols):
                    if prev.owners[r][c] == them and cur.owners[r][c] == us:
                        found = True
                        break
                if found:
                    break
            if found:
                react = t
                break
            if t < len(steps):
                s = steps[t]
                if s.moved and s.target is not None and s.delta is not None and s.delta < 0:
                    react = t
                    break
        if react is not None:
            react_lat = react - first_vis

    # opening pulse / flood
    def gained(t: int) -> bool:
        if t < 1 or t >= len(metrics):
            return False
        return metrics[t].seats[us].tiles > metrics[t - 1].seats[us].tiles

    pulse3 = gained(3)
    # flood start: first tick >=20 with gain after a quiet stretch, near 27
    flood = None
    for t in range(20, min(35, len(metrics))):
        if gained(t):
            flood = t
            break
    flood_near_27 = flood is not None and abs(flood - 27) <= 1

    # real castles: EventLog stamps with spend drop >= 30
    real_early = 0
    real_ticks = []
    for e in events.of_kind("castle_built", player=us):
        if not e.cell:
            continue
        drop = army_drop_before(replay, us, e.cell, e.tick)
        if drop and drop[0] >= 30:
            real_ticks.append(e.tick)
            if e.tick <= 50:
                real_early += 1
    no_castle_by_50 = real_early == 0

    # legacy false-positive rule (kept marked refuted)
    castle_ev = events.of_kind("castle_built", player=us)
    first_castle = castle_ev[0].tick if castle_ev else None
    early_event = first_castle is not None and first_castle <= 20
    early_event_is_10 = first_castle == 10

    return {
        "match_id": str(replay.match_id),
        "ticks": replay.total_ticks,
        "pass01": pass01,
        "tiles50": tiles50,
        "army50": army50,
        "tiles50_in_band": tiles50 is not None and 20 <= tiles50 <= 25,
        "comps_exp_end": comps,
        "contiguous": comps == 1,
        "contact": contact,
        "army_ratio_contact": army_ratio,
        "army_ratio_ge_1": army_ratio is not None and army_ratio >= 1.0,
        "sight": sight,
        "has_sight": sight is not None,
        "toward_after_sight": toward_frac,
        "toward_ge_0_7": toward_frac is not None and toward_frac >= 0.7,
        "sight_to_kill": sight_to_kill,
        "sight_to_kill_le_120": sight_to_kill is not None and sight_to_kill <= 120,
        "tip_at_sight": tip_at_sight,
        "tip_at_sight_ge_10": tip_at_sight is not None and tip_at_sight >= 10,
        "n_gather_waves": len(waves),
        "has_gather": len(waves) >= 1,
        "pass_rate": pass_rate,
        "pass_rate_lt_05": pass_rate < 0.05,
        "full_frac": full_frac,
        "full_frac_ge_90": full_frac is not None and full_frac >= 0.90,
        "first_castle_event": first_castle,
        "early_castle": early_event,
        "early_castle_is_10": early_event_is_10,
        "no_real_castle_by_50": no_castle_by_50,
        "pulse_tick3": pulse3,
        "flood_near_27": flood_near_27,
        "flood_start": flood,
        "min_enemy_home_dist": min_d,
        "enemy_not_adj": min_d is not None and min_d > 1,
        "react_latency": react_lat,
        "react_le_3": react_lat is not None and react_lat <= 3,
    }


# Predictions: (rule_id, field_bool, description, fit_support_note)
RULES = [
    ("H1_pass_ticks_0_1", "pass01", "Passes on ticks 0 and 1", "fit: 100%"),
    ("H2_tiles50_band", "tiles50_in_band", "tiles@50 in [20,25]", "fit median 24, tight band"),
    ("H3_contiguous_expansion", "contiguous", "One connected component at expansion end", "fit max=1"),
    ("H4_army_ratio_contact_ge1", "army_ratio_ge_1", "army_ratio at contact >= 1.0", "fit 97%"),
    ("H5_has_general_sight", "has_sight", "Sees enemy general before win", "fit 100%"),
    ("H6_toward_after_sight", "toward_ge_0_7", "Post-sight toward_fraction >= 0.7", "fit p10~0.73"),
    ("H7_gather_wave", "has_gather", "At least one gather_wave", "fit 100%"),
    ("H8_low_pass_rate", "pass_rate_lt_05", "pass_rate < 0.05", "fit median 0.011"),
    ("H9_full_over_half", "full_frac_ge_90", "Among full|half, full >= 90%", "fit ~98.5%"),
    ("H10_react_le_3", "react_le_3", "React to first visible enemy within 3 ticks", "fit p90=3"),
    ("H11_tip_at_sight_ge10", "tip_at_sight_ge_10", "Tip stack at sight >= 10", "fit p10=12"),
    ("H12_sight_to_kill_le120", "sight_to_kill_le_120", "Sight→kill <= 120 ticks", "fit p90=101"),
    ("H13_enemy_not_adjacent_home", "enemy_not_adj", "Enemy never reaches Manhattan<=1 of home", "fit 96.2%"),
    (
        "H14_legacy_event_castle_tick10",
        "early_castle_is_10",
        "LEGACY/REFUTED policy: EventLog first castle stamp==10 if stamp<=20",
        "detector FP; kept only as artifact check",
    ),
    ("H14b_no_real_castle_by_50", "no_real_castle_by_50", "No real castle spend (drop>=30) by tick 50", "opening 341/341"),
    ("H15_pulse_tick3", "pulse_tick3", "Tile gain on tick 3", "opening p=0.985"),
    ("H16_flood_near_27", "flood_near_27", "Flood start within 27±1", "opening 94.4%"),
]


def main() -> None:
    meta = corpus_meta()
    rows = []
    for replay, an in iter_analyzed("holdout"):
        rows.append(check_game(replay, an))

    results = []
    for rule_id, field, desc, fit_note in RULES:
        if rule_id == "H14_legacy_event_castle_tick10":
            subset = [r for r in rows if r["early_castle"]]
            n = len(subset)
            hits = sum(1 for r in subset if r["early_castle_is_10"])
            fails = [r["match_id"] for r in subset if not r["early_castle_is_10"]]
            rate = hits / n if n else None
            # Policy is refuted even if the FP detector is consistent
            status = "refuted_policy_artifact"
        else:
            applicable = rows
            n = len(applicable)
            hits = sum(1 for r in applicable if r.get(field) is True)
            fails = [r["match_id"] for r in applicable if r.get(field) is not True]
            rate = hits / n if n else None
            status = "confirmed" if rate is not None and rate >= 0.80 else (
                "weak" if rate is not None and rate >= 0.60 else "refuted"
            )
        results.append({
            "rule_id": rule_id,
            "description": desc,
            "fit_note": fit_note,
            "n": n,
            "hits": hits,
            "agreement_rate": rate,
            "status": status,
            "fail_match_ids": fails[:20],
            "n_fails": len(fails),
        })

    payload = {
        "meta": {
            "n_holdout": len(rows),
            "holdout_ids": list(meta.holdout_win_ids),
            "split_rule": meta.as_json()["split_rule"],
            "agreement_thresholds": {
                "confirmed": ">=0.80",
                "weak": ">=0.60 and <0.80",
                "refuted": "<0.60",
            },
        },
        "per_game": rows,
        "rules": results,
    }
    dump_json("grok-kubic-holdout-verification.json", payload)

    lines = [
        "# Kubic holdout verification",
        "",
        f"Holdout wins n={len(rows)} (every 10th sorted win id). "
        "Rules derived on fit only; this page only scores holdout.",
        "",
        "Script: `scripts/verify_kubic_holdout.py`",
        "",
        "## Per-rule agreement",
        "",
        "| Rule | n | hits | rate | status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for r in results:
        rate_s = "n/a" if r["agreement_rate"] is None else f"{r['agreement_rate']:.3f}"
        lines.append(
            f"| `{r['rule_id']}` | {r['n']} | {r['hits']} | {rate_s} | **{r['status']}** |"
        )
    lines += ["", "## Failures / counterexamples", ""]
    for r in results:
        if r["n_fails"] == 0:
            continue
        lines.append(
            f"- `{r['rule_id']}` ({r['status']}): {r['description']} — "
            f"fails {r['n_fails']}: `{r['fail_match_ids']}`"
        )
    lines += [
        "",
        "## Status summary",
        "",
        f"- confirmed: {sum(1 for r in results if r['status']=='confirmed')}",
        f"- weak: {sum(1 for r in results if r['status']=='weak')}",
        f"- refuted: {sum(1 for r in results if r['status']=='refuted')}",
        f"- no_sample: {sum(1 for r in results if r['status']=='no_sample')}",
        "",
        "Refuted rules stay in the behavioral spec marked **refuted**, with these "
        "counterexamples. They are not dropped.",
        "",
    ]
    path = MEASUREMENTS / "grok-kubic-holdout-verification.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"holdout n={len(rows)}")
    for r in results:
        print(f"  {r['rule_id']}: {r['agreement_rate']} [{r['status']}] fails={r['n_fails']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
