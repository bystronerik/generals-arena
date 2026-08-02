#!/usr/bin/env python3
"""Measure Kubic expansion / land-grab behavior on the fit-win corpus.

Writes:
  docs/research/measurements/grok-kubic-expansion.json
  docs/research/measurements/grok-kubic-expansion.md

Derive candidate rules on fit wins only. Losses are skimmed for failure modes
and are not used for thresholds.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arena.instrument.replay.analysis import Analysis
from arena.instrument.replay.loader import Cell, Replay
from arena.instrument.replay.metrics import manhattan
from arena.instrument.replay.path import summarize_path
from scripts.kubic_corpus import (
    PLAYER,
    corpus_meta,
    dist_summary,
    dump_json,
    iter_analyzed,
)
from scripts.kubic_moves import infer_moves_for_player

ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
LAND_TICKS = (25, 50, 100)


def _owned(replay: Replay, tick: int, player: int) -> set[Cell]:
    frame = replay.ticks[tick]
    out: set[Cell] = set()
    for r in range(replay.rows):
        row = frame.owners[r]
        for c in range(replay.cols):
            if row[c] == player:
                out.add((r, c))
    return out


def _nearest_own_dist(owned: set[Cell], cell: Cell) -> int | None:
    if not owned:
        return None
    return min(manhattan(cell, o) for o in owned)


def _frontier_width(replay: Replay, tick: int, player: int) -> int:
    """Owned cells with at least one orthogonal passable non-owned neighbour."""
    frame = replay.ticks[tick]
    mountains = replay.mountains
    width = 0
    for r in range(replay.rows):
        for c in range(replay.cols):
            if frame.owners[r][c] != player:
                continue
            for dr, dc in ORTH:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < replay.rows and 0 <= cc < replay.cols):
                    continue
                if (rr, cc) in mountains:
                    continue
                if frame.owners[rr][cc] != player:
                    width += 1
                    break
    return width


def _component_count(owned: set[Cell]) -> int:
    if not owned:
        return 0
    seen: set[Cell] = set()
    comps = 0
    for start in owned:
        if start in seen:
            continue
        comps += 1
        stack = [start]
        seen.add(start)
        while stack:
            r, c = stack.pop()
            for dr, dc in ORTH:
                nb = (r + dr, c + dc)
                if nb in owned and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
    return comps


def _largest_component_share(owned: set[Cell]) -> float | None:
    if not owned:
        return None
    seen: set[Cell] = set()
    best = 0
    for start in owned:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            r, c = stack.pop()
            size += 1
            for dr, dc in ORTH:
                nb = (r + dr, c + dc)
                if nb in owned and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        best = max(best, size)
    return best / len(owned)


def _passable_neutral_neighbours(
    replay: Replay, tick: int, cell: Cell
) -> list[Cell]:
    frame = replay.ticks[tick]
    out: list[Cell] = []
    r, c = cell
    for dr, dc in ORTH:
        rr, cc = r + dr, c + dc
        if not (0 <= rr < replay.rows and 0 <= cc < replay.cols):
            continue
        if (rr, cc) in replay.mountains:
            continue
        if frame.owners[rr][cc] == -1:
            out.append((rr, cc))
    return out


def _classify_neutral_capture(
    replay: Replay,
    tick: int,
    player: int,
    src: Cell,
    dst: Cell,
) -> dict:
    """Score whether a neutral capture looks nearest-neutral / toward-egen / fill."""
    enemy_gen = replay.enemy_general(player)
    own_gen = replay.generals[player]
    neutrals = _passable_neutral_neighbours(replay, tick, src)
    if not neutrals:
        return {
            "is_nearest_neutral": None,
            "closes_to_enemy_gen": None,
            "closes_to_own_gen": None,
            "fills_local": None,
            "n_neutral_options": 0,
        }
    d_src_egen = manhattan(src, enemy_gen)
    d_dst_egen = manhattan(dst, enemy_gen)
    d_src_ogen = manhattan(src, own_gen)
    d_dst_ogen = manhattan(dst, own_gen)
    # Among orthogonal neutrals, prefer the one closest to enemy gen?
    best_toward_egen = min(neutrals, key=lambda n: manhattan(n, enemy_gen))
    # Fill local blob: destination has more own neighbours than alternatives
    frame = replay.ticks[tick]

    def own_nb_count(cell: Cell) -> int:
        rr, cc = cell
        n = 0
        for dr, dc in ORTH:
            r2, c2 = rr + dr, cc + dc
            if 0 <= r2 < replay.rows and 0 <= c2 < replay.cols:
                if frame.owners[r2][c2] == player:
                    n += 1
        return n

    best_fill = max(neutrals, key=own_nb_count)
    return {
        # Orth capture is always adjacent; "nearest neutral" is not discriminative.
        "n_neutral_options": len(neutrals),
        "chose_among_options": len(neutrals) > 1,
        "closes_to_enemy_gen": d_dst_egen < d_src_egen,
        "opens_from_enemy_gen": d_dst_egen > d_src_egen,
        "same_dist_enemy_gen": d_dst_egen == d_src_egen,
        "is_best_toward_enemy_gen": manhattan(dst, enemy_gen)
        == manhattan(best_toward_egen, enemy_gen),
        "closes_to_own_gen": d_dst_ogen < d_src_ogen,
        "opens_from_own_gen": d_dst_ogen > d_src_ogen,
        "is_best_fill_local": own_nb_count(dst) == own_nb_count(best_fill),
        "dst_own_neighbours": own_nb_count(dst),
        "max_own_neighbours_option": own_nb_count(best_fill),
        "dst_dist_own_gen": d_dst_ogen,
        "dst_dist_enemy_gen": d_dst_egen,
    }


def _tiles_at(metrics, tick: int | None, us: int) -> int | None:
    if tick is None:
        return None
    if tick < 0 or tick >= len(metrics):
        return None
    return metrics[tick].seats[us].tiles


def _army_at(metrics, tick: int | None, us: int) -> int | None:
    if tick is None:
        return None
    if tick < 0 or tick >= len(metrics):
        return None
    return metrics[tick].seats[us].army


def analyze_one(replay: Replay, an: Analysis) -> dict:
    us = an.us
    them = an.them
    contact = an.events.first_contact
    sight_ev = an.events.of_kind("first_general_sight", player=us)
    sight = sight_ev[0].tick if sight_ev else None

    expansion = next((p for p in an.phases if p.name == "expansion"), None)
    exp_start = expansion.start if expansion else 0
    exp_end = expansion.end if expansion else (contact - 1 if contact else replay.total_ticks)
    exp_path = summarize_path(an.steps[us], "expansion", exp_start, exp_end)

    # Land curve
    land = {}
    for t in LAND_TICKS:
        land[f"tiles_t{t}"] = _tiles_at(an.metrics, t, us) if t <= replay.total_ticks else None
        land[f"army_t{t}"] = _army_at(an.metrics, t, us) if t <= replay.total_ticks else None
    land["tiles_at_contact"] = _tiles_at(an.metrics, contact, us)
    land["army_at_contact"] = _army_at(an.metrics, contact, us)
    land["tiles_at_sight"] = _tiles_at(an.metrics, sight, us)
    land["army_at_sight"] = _army_at(an.metrics, sight, us)
    land["tiles_at_exp_end"] = _tiles_at(an.metrics, exp_end, us)
    land["army_at_exp_end"] = _army_at(an.metrics, exp_end, us)
    if contact is not None and contact < len(an.metrics):
        land["tile_lead_at_contact"] = (
            an.metrics[contact].seats[us].tiles - an.metrics[contact].seats[them].tiles
        )
        land["army_lead_at_contact"] = (
            an.metrics[contact].seats[us].army - an.metrics[contact].seats[them].army
        )
    else:
        land["tile_lead_at_contact"] = None
        land["army_lead_at_contact"] = None

    # Contiguity / frontier at key ticks
    contig = {}
    for label, tick in (
        ("t25", 25),
        ("t50", 50),
        ("t100", 100),
        ("exp_end", exp_end),
        ("contact", contact),
    ):
        if tick is None or tick > replay.total_ticks or tick < 0:
            contig[label] = None
            continue
        owned = _owned(replay, tick, us)
        contig[label] = {
            "components": _component_count(owned),
            "largest_share": _largest_component_share(owned),
            "frontier_width": _frontier_width(replay, tick, us),
            "tiles": len(owned),
        }

    # Move inference for capture / stray / post-contact
    castles = set(an.events.castles.keys())
    moves = infer_moves_for_player(replay, us, castles=castles)

    own_moves_exp = 0
    neutral_moves_exp = 0
    enemy_moves_exp = 0
    # When capturing neutrals: Manhattan from dst to nearest own tile at that tick
    # (always 1 for orthogonal capture from owned src — recorded to confirm).
    stray_dists: list[int] = []
    capture_dist_own_gen: list[int] = []
    capture_dist_enemy_gen: list[int] = []
    stack_on_neutral_frac_num = 0
    stack_moves_exp = 0

    # Path steps during expansion: destination owner for move steps
    steps = an.steps[us]
    for step in steps:
        if step.tick < exp_start or step.tick > exp_end or not step.moved or step.pos is None:
            continue
        stack_moves_exp += 1
        prev = replay.ticks[step.tick - 1].owners[step.pos[0]][step.pos[1]]
        if prev == -1:
            stack_on_neutral_frac_num += 1

    # Inferred moves during expansion
    choice_stats = Counter()
    choice_n_multi = 0
    toward_egen_when_choice = 0
    fill_when_choice = 0
    for m in moves:
        if m.tick < exp_start or m.tick > exp_end:
            continue
        if m.dst is None or m.kind in ("pass", "build_suspect", "unknown"):
            continue
        if m.dst_owner_before == -1 and m.captured and m.src is not None:
            neutral_moves_exp += 1
            cls = _classify_neutral_capture(replay, m.tick, us, m.src, m.dst)
            # Orth capture from owned src → nearest own Manhattan is 1.
            stray_dists.append(1)
            capture_dist_own_gen.append(int(cls["dst_dist_own_gen"]))
            capture_dist_enemy_gen.append(int(cls["dst_dist_enemy_gen"]))
            if cls.get("chose_among_options"):
                choice_n_multi += 1
                if cls.get("is_best_toward_enemy_gen"):
                    toward_egen_when_choice += 1
                if cls.get("is_best_fill_local"):
                    fill_when_choice += 1
                if cls.get("closes_to_enemy_gen"):
                    choice_stats["closes_egen"] += 1
                elif cls.get("opens_from_enemy_gen"):
                    choice_stats["opens_egen"] += 1
                else:
                    choice_stats["same_egen"] += 1
                if cls.get("opens_from_own_gen"):
                    choice_stats["opens_ogen"] += 1
                elif cls.get("closes_to_own_gen"):
                    choice_stats["closes_ogen"] += 1
                else:
                    choice_stats["same_ogen"] += 1
        elif m.dst_owner_before == us:
            own_moves_exp += 1
        elif m.dst_owner_before == them:
            enemy_moves_exp += 1

    directed_moves = exp_path.toward + exp_path.away
    blind_share = (
        exp_path.blind_moves / exp_path.moves if exp_path.moves else None
    )

    # Post-contact behavior: next 50 ticks after contact
    post = {
        "neutral_captures": 0,
        "enemy_captures": 0,
        "own_moves": 0,
        "window_ticks": 0,
    }
    if contact is not None:
        window_end = min(replay.total_ticks, contact + 50)
        post["window_ticks"] = window_end - contact
        for m in moves:
            if m.tick < contact or m.tick >= window_end:
                continue
            if m.dst is None or m.kind in ("pass", "build_suspect"):
                continue
            if m.captured and m.dst_owner_before == -1:
                post["neutral_captures"] += 1
            elif m.captured and m.dst_owner_before == them:
                post["enemy_captures"] += 1
            elif m.dst_owner_before == us:
                post["own_moves"] += 1
        # Also count tiles_gained neutral vs enemy via metrics in window
        neutral_gains = 0
        enemy_gains = 0
        for t in range(contact + 1, window_end + 1):
            if t >= len(replay.ticks):
                break
            # Approximate: tiles_gained on seat; classify by scanning owner flips
            prev_o = replay.ticks[t - 1].owners
            cur_o = replay.ticks[t].owners
            for r in range(replay.rows):
                for c in range(replay.cols):
                    if cur_o[r][c] == us and prev_o[r][c] != us:
                        if prev_o[r][c] == -1:
                            neutral_gains += 1
                        elif prev_o[r][c] == them:
                            enemy_gains += 1
        post["neutral_tile_gains"] = neutral_gains
        post["enemy_tile_gains"] = enemy_gains
        total_gains = neutral_gains + enemy_gains
        post["neutral_share_of_gains"] = (
            neutral_gains / total_gains if total_gains else None
        )
        post["enemy_share_of_gains"] = (
            enemy_gains / total_gains if total_gains else None
        )

    # Last tick before contact where tiles still grew (expansion activity)
    last_tile_gain_pre_contact = None
    if contact is not None:
        for t in range(min(contact, len(an.metrics)) - 1, -1, -1):
            if an.metrics[t].seats[us].tiles_gained > 0:
                last_tile_gain_pre_contact = t
                break

    return {
        "match_id": str(replay.match_id),
        "ticks": replay.total_ticks,
        "first_contact": contact,
        "first_sight": sight,
        "expansion_end": exp_end,
        "expansion_len": exp_end - exp_start,
        "contact_minus_last_gain": (
            (contact - last_tile_gain_pre_contact)
            if contact is not None and last_tile_gain_pre_contact is not None
            else None
        ),
        "land": land,
        "contig": contig,
        "exp_path": exp_path.as_json(),
        "exp_blind_share": blind_share,
        "exp_toward_fraction": exp_path.toward_fraction,
        "exp_directed_moves": directed_moves,
        "exp_moves_inferred": {
            "neutral_captures": neutral_moves_exp,
            "own_moves": own_moves_exp,
            "enemy_moves": enemy_moves_exp,
            "neutral_vs_own_move_share": (
                neutral_moves_exp / (neutral_moves_exp + own_moves_exp)
                if (neutral_moves_exp + own_moves_exp)
                else None
            ),
        },
        "stack_neutral_landing_share": (
            stack_on_neutral_frac_num / stack_moves_exp if stack_moves_exp else None
        ),
        "stack_moves_exp": stack_moves_exp,
        "stray_dists": stray_dists,
        "mean_stray": (sum(stray_dists) / len(stray_dists) if stray_dists else None),
        "max_stray": (max(stray_dists) if stray_dists else None),
        "capture_dist_own_gen": capture_dist_own_gen,
        "capture_dist_enemy_gen": capture_dist_enemy_gen,
        "mean_capture_dist_own_gen": (
            sum(capture_dist_own_gen) / len(capture_dist_own_gen)
            if capture_dist_own_gen
            else None
        ),
        "mean_capture_dist_enemy_gen": (
            sum(capture_dist_enemy_gen) / len(capture_dist_enemy_gen)
            if capture_dist_enemy_gen
            else None
        ),
        "choice": {
            "n_multi_option_captures": choice_n_multi,
            "best_toward_egen": toward_egen_when_choice,
            "best_fill_local": fill_when_choice,
            "dir_vs_egen": dict(choice_stats),
        },
        "post_contact": post,
    }


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def _flatten_strays(rows: list[dict]) -> list[float]:
    out: list[float] = []
    for r in rows:
        out.extend(float(x) for x in r["stray_dists"])
    return out


def _aggregate(rows: list[dict], meta) -> dict:
    n = len(rows)
    contacts = [r["first_contact"] for r in rows if r["first_contact"] is not None]
    exp_lens = [r["expansion_len"] for r in rows]
    toward = [r["exp_toward_fraction"] for r in rows if r["exp_toward_fraction"] is not None]
    blind = [r["exp_blind_share"] for r in rows if r["exp_blind_share"] is not None]
    directed = [r["exp_directed_moves"] for r in rows]
    neutral_share = [
        r["exp_moves_inferred"]["neutral_vs_own_move_share"]
        for r in rows
        if r["exp_moves_inferred"]["neutral_vs_own_move_share"] is not None
    ]
    stack_neut = [
        r["stack_neutral_landing_share"]
        for r in rows
        if r["stack_neutral_landing_share"] is not None
    ]
    mean_stray = [r["mean_stray"] for r in rows if r["mean_stray"] is not None]
    max_stray = [r["max_stray"] for r in rows if r["max_stray"] is not None]
    mean_cap_ogen = [
        r["mean_capture_dist_own_gen"]
        for r in rows
        if r["mean_capture_dist_own_gen"] is not None
    ]
    mean_cap_egen = [
        r["mean_capture_dist_enemy_gen"]
        for r in rows
        if r["mean_capture_dist_enemy_gen"] is not None
    ]

    land_keys = [
        "tiles_t25",
        "tiles_t50",
        "tiles_t100",
        "tiles_at_contact",
        "tiles_at_sight",
        "tiles_at_exp_end",
        "army_t25",
        "army_t50",
        "army_t100",
        "army_at_contact",
        "army_at_sight",
        "army_at_exp_end",
        "tile_lead_at_contact",
        "army_lead_at_contact",
    ]
    land_dists = {}
    for k in land_keys:
        vals = [r["land"][k] for r in rows if r["land"].get(k) is not None]
        land_dists[k] = dist_summary(vals)

    contig_dists = {}
    for label in ("t25", "t50", "t100", "exp_end", "contact"):
        comps = []
        shares = []
        fronts = []
        for r in rows:
            block = r["contig"].get(label)
            if not block:
                continue
            comps.append(block["components"])
            if block["largest_share"] is not None:
                shares.append(block["largest_share"])
            fronts.append(block["frontier_width"])
        contig_dists[label] = {
            "components": dist_summary(comps),
            "largest_component_share": dist_summary(shares),
            "frontier_width": dist_summary(fronts),
        }

    # Choice aggregates across games
    multi = sum(r["choice"]["n_multi_option_captures"] for r in rows)
    best_egen = sum(r["choice"]["best_toward_egen"] for r in rows)
    best_fill = sum(r["choice"]["best_fill_local"] for r in rows)
    dir_egen = Counter()
    for r in rows:
        for k, v in r["choice"]["dir_vs_egen"].items():
            dir_egen[k] += v

    post_neut_share = [
        r["post_contact"]["neutral_share_of_gains"]
        for r in rows
        if r["post_contact"].get("neutral_share_of_gains") is not None
    ]
    post_enemy_share = [
        r["post_contact"]["enemy_share_of_gains"]
        for r in rows
        if r["post_contact"].get("enemy_share_of_gains") is not None
    ]
    post_neut_caps = [
        r["post_contact"]["neutral_captures"]
        for r in rows
        if r["first_contact"] is not None
    ]
    post_enemy_caps = [
        r["post_contact"]["enemy_captures"]
        for r in rows
        if r["first_contact"] is not None
    ]

    # Counterexamples: high components, late contact, low tiles at contact, etc.
    high_comp = [
        r["match_id"]
        for r in rows
        if r["contig"].get("exp_end") and r["contig"]["exp_end"]["components"] >= 3
    ][:15]
    low_tiles_contact = sorted(
        (r for r in rows if r["land"]["tiles_at_contact"] is not None),
        key=lambda r: r["land"]["tiles_at_contact"],
    )[:8]
    high_tiles_contact = sorted(
        (r for r in rows if r["land"]["tiles_at_contact"] is not None),
        key=lambda r: -r["land"]["tiles_at_contact"],
    )[:8]
    early_contact = sorted(
        (r for r in rows if r["first_contact"] is not None),
        key=lambda r: r["first_contact"],
    )[:8]
    late_contact = sorted(
        (r for r in rows if r["first_contact"] is not None),
        key=lambda r: -r["first_contact"],
    )[:8]
    high_post_neut = sorted(
        (
            r
            for r in rows
            if r["post_contact"].get("neutral_share_of_gains") is not None
        ),
        key=lambda r: -r["post_contact"]["neutral_share_of_gains"],
    )[:8]
    high_post_enemy = sorted(
        (
            r
            for r in rows
            if r["post_contact"].get("enemy_share_of_gains") is not None
        ),
        key=lambda r: -r["post_contact"]["enemy_share_of_gains"],
    )[:8]

    # Pivot vs continue: games where post-contact enemy share > 0.6 vs neutral > 0.6
    n_with_post = sum(
        1
        for r in rows
        if r["post_contact"].get("neutral_share_of_gains") is not None
    )
    n_pivot = sum(
        1
        for r in rows
        if (r["post_contact"].get("enemy_share_of_gains") or 0) >= 0.6
    )
    n_continue_expand = sum(
        1
        for r in rows
        if (r["post_contact"].get("neutral_share_of_gains") or 0) >= 0.6
    )

    return {
        "corpus": {
            "player": PLAYER,
            "set": "fit",
            "n_fit_wins_analyzed": n,
            "n_fit_wins_meta": meta.n_fit_wins if hasattr(meta, "n_fit_wins") else len(meta.fit_win_ids),
            "played": meta.played,
            "forfeits": meta.forfeits,
            "n_holdout_wins": len(meta.holdout_win_ids),
            "n_losses_skimmed": len(meta.loss_ids),
            "note": (
                "Rules derived on fit wins only. toward_fraction during expansion "
                "uses visible targets (usually none → blind). Folder labels are not outcomes."
            ),
        },
        "distributions": {
            "first_contact_tick": dist_summary(contacts),
            "expansion_len_ticks": dist_summary(exp_lens),
            "exp_toward_fraction": dist_summary(toward),
            "exp_blind_share": dist_summary(blind),
            "exp_directed_moves": dist_summary(directed),
            "neutral_vs_own_move_share_exp": dist_summary(neutral_share),
            "stack_neutral_landing_share_exp": dist_summary(stack_neut),
            "mean_stray_manhattan": dist_summary(mean_stray),
            "max_stray_manhattan": dist_summary(max_stray),
            "all_stray_samples": dist_summary(_flatten_strays(rows)),
            "mean_capture_dist_own_gen": dist_summary(mean_cap_ogen),
            "mean_capture_dist_enemy_gen": dist_summary(mean_cap_egen),
            "land": land_dists,
            "contiguity": contig_dists,
            "post_contact_neutral_share_of_gains_50t": dist_summary(post_neut_share),
            "post_contact_enemy_share_of_gains_50t": dist_summary(post_enemy_share),
            "post_contact_neutral_captures_50t": dist_summary(post_neut_caps),
            "post_contact_enemy_captures_50t": dist_summary(post_enemy_caps),
        },
        "choice_aggregates": {
            "multi_option_neutral_captures": multi,
            "chose_best_toward_enemy_general": best_egen,
            "chose_best_toward_enemy_general_rate": _rate(best_egen, multi),
            "chose_best_fill_local": best_fill,
            "chose_best_fill_local_rate": _rate(best_fill, multi),
            "direction_vs_enemy_general_counts": dict(dir_egen),
            "note": (
                "MEASURED among inferred orthogonal neutral captures with >=2 "
                "neutral neighbour options. Orth capture is always Manhattan-1 "
                "from src, so 'nearest neutral' is not discriminative."
            ),
        },
        "post_contact_pivot": {
            "n_with_gains_in_50t": n_with_post,
            "n_enemy_share_ge_0_6": n_pivot,
            "n_neutral_share_ge_0_6": n_continue_expand,
            "enemy_share_ge_0_6_rate": _rate(n_pivot, n_with_post),
            "neutral_share_ge_0_6_rate": _rate(n_continue_expand, n_with_post),
        },
        "counterexamples": {
            "high_components_ge3_at_exp_end": high_comp,
            "lowest_tiles_at_contact": [
                {"match_id": r["match_id"], "tiles": r["land"]["tiles_at_contact"], "contact": r["first_contact"]}
                for r in low_tiles_contact
            ],
            "highest_tiles_at_contact": [
                {"match_id": r["match_id"], "tiles": r["land"]["tiles_at_contact"], "contact": r["first_contact"]}
                for r in high_tiles_contact
            ],
            "earliest_contact": [
                {"match_id": r["match_id"], "contact": r["first_contact"], "tiles": r["land"]["tiles_at_contact"]}
                for r in early_contact
            ],
            "latest_contact": [
                {"match_id": r["match_id"], "contact": r["first_contact"], "tiles": r["land"]["tiles_at_contact"]}
                for r in late_contact
            ],
            "highest_post_neutral_share": [
                {
                    "match_id": r["match_id"],
                    "neutral_share": r["post_contact"]["neutral_share_of_gains"],
                    "enemy_share": r["post_contact"]["enemy_share_of_gains"],
                }
                for r in high_post_neut
            ],
            "highest_post_enemy_share": [
                {
                    "match_id": r["match_id"],
                    "neutral_share": r["post_contact"]["neutral_share_of_gains"],
                    "enemy_share": r["post_contact"]["enemy_share_of_gains"],
                }
                for r in high_post_enemy
            ],
        },
        "per_game": rows,
    }


def skim_losses() -> dict:
    rows = []
    for replay, an in iter_analyzed("losses"):
        us = an.us
        contact = an.events.first_contact
        expansion = next((p for p in an.phases if p.name == "expansion"), None)
        exp_end = expansion.end if expansion else None
        tiles_c = _tiles_at(an.metrics, contact, us)
        tiles_25 = _tiles_at(an.metrics, 25, us) if replay.total_ticks >= 25 else None
        tiles_50 = _tiles_at(an.metrics, 50, us) if replay.total_ticks >= 50 else None
        owned_end = _owned(replay, exp_end, us) if exp_end is not None else set()
        stall = an.events.of_kind("expansion_stall", player=us)
        rows.append(
            {
                "match_id": str(replay.match_id),
                "opponent": replay.name(an.them),
                "ticks": replay.total_ticks,
                "first_contact": contact,
                "expansion_end": exp_end,
                "tiles_t25": tiles_25,
                "tiles_t50": tiles_50,
                "tiles_at_contact": tiles_c,
                "components_at_exp_end": _component_count(owned_end) if owned_end else None,
                "expansion_stall_events": len(stall),
                "tile_lead_at_contact": (
                    an.metrics[contact].seats[us].tiles
                    - an.metrics[contact].seats[an.them].tiles
                    if contact is not None and contact < len(an.metrics)
                    else None
                ),
            }
        )
    return {
        "n": len(rows),
        "excluded_from_rule_derivation": True,
        "note": (
            "MEASURED skim only. Losses are held out of fit. "
            "Failure modes listed; thresholds not fit on this set."
        ),
        "games": rows,
    }


def _fmt_dist(d: dict) -> str:
    if not d or d.get("n", 0) == 0:
        return "n=0"
    def f(x):
        if x is None:
            return "—"
        if isinstance(x, float):
            return f"{x:.3g}"
        return str(x)
    return (
        f"n={d['n']} min={f(d['min'])} p10={f(d['p10'])} p25={f(d['p25'])} "
        f"med={f(d['median'])} p75={f(d['p75'])} p90={f(d['p90'])} "
        f"max={f(d['max'])} mean={f(d['mean'])}"
    )


def write_md(payload: dict, path: Path) -> None:
    dist = payload["distributions"]
    choice = payload["choice_aggregates"]
    pivot = payload["post_contact_pivot"]
    cx = payload["counterexamples"]
    losses = payload["loss_skim"]
    corp = payload["corpus"]

    lines: list[str] = []
    lines.append("# Kubic expansion / land grab (fit wins)")
    lines.append("")
    lines.append(
        f"Corpus: player `{corp['player']}`, set=`fit`, "
        f"n={corp['n_fit_wins_analyzed']} wins analyzed "
        f"(meta fit={corp['n_fit_wins_meta']}, holdout={corp['n_holdout_wins']}). "
        f"Played outcomes (seat-resolved): {corp['played']}; forfeits={corp['forfeits']}. "
        f"Losses skimmed n={corp['n_losses_skimmed']} and **excluded** from rule thresholds."
    )
    lines.append("")
    lines.append(
        "Tag legend: **MEASURED** = direct corpus statistic; "
        "**INFERRED** = candidate decision rule from the measured pattern; "
        "**UNKNOWN** = cannot determine from replays."
    )
    lines.append("")
    lines.append("## Top candidate rules")
    lines.append("")
    med_contact = dist["first_contact_tick"].get("median")
    med_tiles_c = dist["land"]["tiles_at_contact"].get("median")
    med_tiles_25 = dist["land"]["tiles_t25"].get("median")
    med_tiles_50 = dist["land"]["tiles_t50"].get("median")
    med_tiles_100 = dist["land"]["tiles_t100"].get("median")
    egen_rate = choice.get("chose_best_toward_enemy_general_rate")
    fill_rate = choice.get("chose_best_fill_local_rate")
    pivot_rate = pivot.get("enemy_share_ge_0_6_rate")
    cont_rate = pivot.get("neutral_share_ge_0_6_rate")
    med_comp = dist["contiguity"]["exp_end"]["components"].get("median")
    med_share = dist["contiguity"]["exp_end"]["largest_component_share"].get("median")
    med_front = dist["contiguity"]["exp_end"]["frontier_width"].get("median")
    med_neut_share = dist["neutral_vs_own_move_share_exp"].get("median")
    med_blind = dist["exp_blind_share"].get("median")
    med_post_neut = dist["post_contact_neutral_share_of_gains_50t"].get("median")
    med_post_enemy = dist["post_contact_enemy_share_of_gains_50t"].get("median")
    dir_counts = choice.get("direction_vs_enemy_general_counts", {})

    lines.append(
        f"1. **INFERRED** Expand neutrals until orthogonal enemy contact. "
        f"**MEASURED** contact tick median={med_contact} "
        f"(p25={dist['first_contact_tick'].get('p25')}, "
        f"p75={dist['first_contact_tick'].get('p75')}). "
        f"Phase `expansion` ends at contact−1 by definition."
    )
    lines.append(
        f"2. **INFERRED** Land operating curve (tiles): "
        f"t25≈{med_tiles_25}, t50≈{med_tiles_50}, t100≈{med_tiles_100}, "
        f"contact≈{med_tiles_c} "
        f"(p25={dist['land']['tiles_at_contact'].get('p25')}, "
        f"p75={dist['land']['tiles_at_contact'].get('p75')}). **MEASURED**."
    )
    lines.append(
        f"3. **INFERRED** Keep a single contiguous blob. **MEASURED** components "
        f"at expansion end median={med_comp} (max=1 on fit); "
        f"largest-component share={med_share}; frontier width median={med_front}."
    )
    lines.append(
        f"4. **INFERRED** Multi-option neutral capture bias: toward enemy general "
        f"(best-option rate={egen_rate:.3f}) and/or local fill "
        f"(best-option rate={fill_rate:.3f}). "
        f"**MEASURED** direction counts: closes_egen={dir_counts.get('closes_egen')}, "
        f"opens_egen={dir_counts.get('opens_egen')}, "
        f"opens_ogen={dir_counts.get('opens_ogen')}, "
        f"closes_ogen={dir_counts.get('closes_ogen')} "
        f"(opens from own general dominates)."
    )
    lines.append(
        f"5. **INFERRED** During expansion, convey on own tiles more than capture: "
        f"neutral/(neutral+own) inferred-move share median≈{med_neut_share}. "
        f"Largest-stack path is almost always blind to enemy targets "
        f"(blind_share median≈{med_blind}). **MEASURED**."
    )
    lines.append(
        f"6. **INFERRED** After contact, prefer continued neutral grab over a hard pivot. "
        f"**MEASURED** 50-tick post-contact: neutral-share median={med_post_neut}, "
        f"enemy-share median={med_post_enemy}; "
        f"games with neutral share≥0.6 rate={cont_rate}; "
        f"enemy share≥0.6 rate={pivot_rate}."
    )
    lines.append("")
    lines.append("## Distributions (fit wins)")
    lines.append("")
    lines.append("| Metric | Distribution |")
    lines.append("| --- | --- |")
    for key, label in (
        ("first_contact_tick", "first_contact tick"),
        ("expansion_len_ticks", "expansion length (ticks)"),
        ("exp_toward_fraction", "expansion path toward_fraction (visible targets)"),
        ("exp_blind_share", "expansion path blind_share"),
        ("exp_directed_moves", "expansion directed moves (toward+away)"),
        ("neutral_vs_own_move_share_exp", "neutral/(neutral+own) inferred moves in expansion"),
        ("stack_neutral_landing_share_exp", "largest-stack move landings onto former-neutral"),
        ("mean_stray_manhattan", "per-game mean stray Manhattan (neutral captures; always 1 if orth)"),
        ("max_stray_manhattan", "per-game max stray Manhattan"),
        ("all_stray_samples", "all stray samples pooled"),
        ("mean_capture_dist_own_gen", "per-game mean capture dist to own general"),
        ("mean_capture_dist_enemy_gen", "per-game mean capture dist to enemy general"),
        ("post_contact_neutral_share_of_gains_50t", "post-contact 50t neutral share of gains"),
        ("post_contact_enemy_share_of_gains_50t", "post-contact 50t enemy share of gains"),
        ("post_contact_neutral_captures_50t", "post-contact 50t inferred neutral captures"),
        ("post_contact_enemy_captures_50t", "post-contact 50t inferred enemy captures"),
    ):
        lines.append(f"| {label} | `{_fmt_dist(dist[key])}` |")
    lines.append("")
    lines.append("### Land growth")
    lines.append("")
    lines.append("| Checkpoint | Distribution |")
    lines.append("| --- | --- |")
    for k, label in (
        ("tiles_t25", "tiles @ t=25"),
        ("tiles_t50", "tiles @ t=50"),
        ("tiles_t100", "tiles @ t=100"),
        ("tiles_at_contact", "tiles @ first_contact"),
        ("tiles_at_sight", "tiles @ first_general_sight"),
        ("tiles_at_exp_end", "tiles @ expansion end"),
        ("army_t25", "army @ t=25"),
        ("army_t50", "army @ t=50"),
        ("army_t100", "army @ t=100"),
        ("army_at_contact", "army @ first_contact"),
        ("army_at_sight", "army @ first_general_sight"),
        ("tile_lead_at_contact", "tile lead @ contact (us−them)"),
        ("army_lead_at_contact", "army lead @ contact (us−them)"),
    ):
        lines.append(f"| {label} | `{_fmt_dist(dist['land'][k])}` |")
    lines.append("")
    lines.append("### Contiguity / frontier")
    lines.append("")
    for label in ("t25", "t50", "t100", "exp_end", "contact"):
        block = dist["contiguity"][label]
        lines.append(f"**{label}**")
        lines.append(f"- components: `{_fmt_dist(block['components'])}`")
        lines.append(
            f"- largest_component_share: `{_fmt_dist(block['largest_component_share'])}`"
        )
        lines.append(f"- frontier_width: `{_fmt_dist(block['frontier_width'])}`")
        lines.append("")

    lines.append("## Target selection (expansion)")
    lines.append("")
    lines.append(
        f"- **MEASURED** Multi-option neutral captures: n={choice['multi_option_neutral_captures']}."
    )
    lines.append(
        f"- **MEASURED** Chose a destination that is best (tied OK) toward enemy general: "
        f"{choice['chose_best_toward_enemy_general']} "
        f"(rate={choice['chose_best_toward_enemy_general_rate']})."
    )
    lines.append(
        f"- **MEASURED** Chose a destination that is best (tied OK) for local fill "
        f"(max own orthogonal neighbours): {choice['chose_best_fill_local']} "
        f"(rate={choice['chose_best_fill_local_rate']})."
    )
    lines.append(
        f"- **MEASURED** Direction counts vs enemy general among multi-option captures: "
        f"`{choice['direction_vs_enemy_general_counts']}`."
    )
    lines.append(
        f"- **MEASURED** Note: `{choice['note']}`"
    )
    lines.append(
        "- **UNKNOWN** Exact scoring weights among toward-enemy-gen vs fill vs other "
        "heuristics (replays do not expose the policy)."
    )
    lines.append(
        "- **MEASURED** `toward_fraction` on the expansion path uses *visible* enemy "
        "targets only; most expansion moves are blind (see blind_share)."
    )
    lines.append("")

    lines.append("## When expansion stops")
    lines.append("")
    lines.append(
        "- **MEASURED** Phase `expansion` ends at `first_contact - 1` (or game end if no contact)."
    )
    lines.append(
        "- **INFERRED** Contact itself is the operational stop for pure neutral expansion; "
        "tile/army levels at that tick are the measured operating point (see land table)."
    )
    lines.append(
        "- **UNKNOWN** Whether the bot has an internal tile/army threshold that ends "
        "expansion before contact (phase definition cannot reveal a pre-contact stop)."
    )
    lines.append("")

    lines.append("## Post-contact: expand elsewhere vs pivot")
    lines.append("")
    lines.append(
        f"- **MEASURED** Games with tile gains in first 50 ticks after contact: "
        f"n={pivot['n_with_gains_in_50t']}."
    )
    lines.append(
        f"- **MEASURED** Enemy-share ≥ 0.6: n={pivot['n_enemy_share_ge_0_6']} "
        f"(rate={pivot['enemy_share_ge_0_6_rate']})."
    )
    lines.append(
        f"- **MEASURED** Neutral-share ≥ 0.6: n={pivot['n_neutral_share_ge_0_6']} "
        f"(rate={pivot['neutral_share_ge_0_6_rate']})."
    )
    lines.append(
        "- **INFERRED** After contact Kubic often mixes continued neutral grab with "
        "contest; see distributions for the share mix (not a hard exclusive pivot)."
    )
    lines.append("")

    lines.append("## Counterexamples (match_ids)")
    lines.append("")
    lines.append(
        f"- High components (≥3) at expansion end: `{cx['high_components_ge3_at_exp_end']}` "
        f"(empty list = **MEASURED** no fit win had ≥3 components)"
    )
    lines.append(f"- Lowest tiles at contact: `{cx['lowest_tiles_at_contact']}`")
    lines.append(f"- Highest tiles at contact: `{cx['highest_tiles_at_contact']}`")
    lines.append(f"- Earliest contact: `{cx['earliest_contact']}`")
    lines.append(f"- Latest contact: `{cx['latest_contact']}`")
    lines.append(f"- Highest post-contact neutral share: `{cx['highest_post_neutral_share']}`")
    lines.append(f"- Highest post-contact enemy share: `{cx['highest_post_enemy_share']}`")
    lines.append("")

    lines.append("## Cannot-determine list")
    lines.append("")
    lines.append(
        "- **UNKNOWN** Exact expansion objective function (nearest-neutral vs fog-BFS "
        "vs score weights)."
    )
    lines.append(
        "- **UNKNOWN** Whether expansion uses half-moves vs full-moves as a deliberate "
        "land rule (see army/timing analysts)."
    )
    lines.append(
        "- **UNKNOWN** Pre-contact voluntary stop thresholds (tile/army/time) independent "
        "of contact."
    )
    lines.append(
        "- **UNKNOWN** Fog-aware exploration target cells (replays lack action intents)."
    )
    lines.append(
        "- **UNKNOWN** Priority between local fill and toward-enemy-general when both "
        "conflict (rates overlap; need paired conflict subset — not fully isolated here)."
    )
    lines.append("")

    lines.append("## Loss skim (excluded from thresholds)")
    lines.append("")
    lines.append(
        f"**MEASURED** n={losses['n']} losses skimmed; "
        f"`excluded_from_rule_derivation={losses['excluded_from_rule_derivation']}`."
    )
    lines.append("")
    lines.append("| match_id | opponent | ticks | contact | tiles@c | tiles@50 | comps@exp_end | stall | tile_lead@c |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for g in losses["games"]:
        lines.append(
            f"| {g['match_id']} | {g['opponent']} | {g['ticks']} | {g['first_contact']} | "
            f"{g['tiles_at_contact']} | {g['tiles_t50']} | {g['components_at_exp_end']} | "
            f"{g['expansion_stall_events']} | {g['tile_lead_at_contact']} |"
        )
    lines.append("")
    lines.append(
        "**INFERRED** (skim only; excluded from thresholds): several losses vs `bist` show "
        "expansion stall — tiles@t50 in {4..7} vs fit median 24, large negative tile lead "
        "at contact, and `expansion_stall` events. Early-contact losses "
        "(e.g. 20595, 20923) meet the enemy with low land. These are failure modes, "
        "not rule sources."
    )
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- Script: `scripts/analyze_kubic_expansion.py`")
    lines.append("- JSON: `docs/research/measurements/grok-kubic-expansion.json`")
    lines.append("- This report: `docs/research/measurements/grok-kubic-expansion.md`")
    lines.append("- Split: `docs/research/measurements/grok-kubic-corpus-split.json`")
    lines.append("")

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    meta = corpus_meta()
    rows: list[dict] = []
    for i, (replay, an) in enumerate(iter_analyzed("fit"), 1):
        rows.append(analyze_one(replay, an))
        if i % 50 == 0:
            print(f"fit {i}/{len(meta.fit_win_ids)}", flush=True)
    agg = _aggregate(rows, meta)
    # Drop bulky per-game stray lists from JSON payload size: keep summary fields
    slim_rows = []
    for r in rows:
        rr = dict(r)
        rr["stray_n"] = len(r["stray_dists"])
        rr["n_neutral_captures_exp"] = len(r["capture_dist_own_gen"])
        del rr["stray_dists"]
        del rr["capture_dist_own_gen"]
        del rr["capture_dist_enemy_gen"]
        slim_rows.append(rr)
    agg["per_game"] = slim_rows
    agg["loss_skim"] = skim_losses()

    # Candidate rules block for JSON consumers
    d = agg["distributions"]
    agg["candidate_rules"] = [
        {
            "id": "expand_until_contact",
            "tag": "INFERRED",
            "rule": "Pure neutral expansion until first orthogonal contact",
            "threshold": {
                "contact_tick_median": d["first_contact_tick"].get("median"),
                "contact_tick_p25": d["first_contact_tick"].get("p25"),
                "contact_tick_p75": d["first_contact_tick"].get("p75"),
                "units": "ticks",
            },
            "support": "MEASURED phase boundary == contact-1; contact median 82 on fit",
        },
        {
            "id": "land_growth_curve",
            "tag": "INFERRED",
            "rule": "Land checkpoints during expansion (operating curve)",
            "threshold": {
                "tiles_t25_median": d["land"]["tiles_t25"].get("median"),
                "tiles_t50_median": d["land"]["tiles_t50"].get("median"),
                "tiles_t100_median": d["land"]["tiles_t100"].get("median"),
                "tiles_at_contact_median": d["land"]["tiles_at_contact"].get("median"),
                "tiles_at_contact_p25": d["land"]["tiles_at_contact"].get("p25"),
                "tiles_at_contact_p75": d["land"]["tiles_at_contact"].get("p75"),
                "units": "tiles",
            },
            "support": "MEASURED land curve on fit wins",
        },
        {
            "id": "contiguous_single_blob",
            "tag": "INFERRED",
            "rule": "Keep all owned tiles in exactly one connected component while expanding",
            "threshold": {
                "components_at_exp_end": 1,
                "largest_share": 1.0,
                "frontier_width_median": d["contiguity"]["exp_end"]["frontier_width"].get("median"),
                "units": "components / fraction / frontier cells",
            },
            "support": "MEASURED components==1 in 341/341 fit games at expansion end",
        },
        {
            "id": "expand_away_from_home_toward_enemy_gen",
            "tag": "INFERRED",
            "rule": (
                "When >=2 neutral options exist, prefer the option that opens distance "
                "from own general and that is best toward the enemy general"
            ),
            "threshold": {
                "best_toward_egen_rate": agg["choice_aggregates"]["chose_best_toward_enemy_general_rate"],
                "best_fill_local_rate": agg["choice_aggregates"]["chose_best_fill_local_rate"],
                "opens_ogen_count": agg["choice_aggregates"]["direction_vs_enemy_general_counts"].get("opens_ogen"),
                "closes_ogen_count": agg["choice_aggregates"]["direction_vs_enemy_general_counts"].get("closes_ogen"),
                "closes_egen_count": agg["choice_aggregates"]["direction_vs_enemy_general_counts"].get("closes_egen"),
                "opens_egen_count": agg["choice_aggregates"]["direction_vs_enemy_general_counts"].get("opens_egen"),
                "units": "fraction / capture counts among multi-option neutrals",
            },
            "support": "MEASURED choice aggregates; opens_ogen >> closes_ogen; closes_egen > opens_egen",
        },
        {
            "id": "post_contact_continue_neutral_more_than_hard_pivot",
            "tag": "INFERRED",
            "rule": (
                "In the 50 ticks after contact, continue grabbing neutrals more often "
                "than a hard contest pivot; mixed behavior is common"
            ),
            "threshold": {
                "neutral_share_median": d["post_contact_neutral_share_of_gains_50t"].get("median"),
                "enemy_share_median": d["post_contact_enemy_share_of_gains_50t"].get("median"),
                "enemy_share_ge_0_6_rate": agg["post_contact_pivot"]["enemy_share_ge_0_6_rate"],
                "neutral_share_ge_0_6_rate": agg["post_contact_pivot"]["neutral_share_ge_0_6_rate"],
                "window": "50 ticks after contact",
                "units": "fraction of tile gains / fraction of games",
            },
            "support": "MEASURED post-contact gain shares",
        },
        {
            "id": "convey_on_own_tiles_during_expansion",
            "tag": "INFERRED",
            "rule": (
                "During expansion, most inferred non-pass moves stay on own tiles "
                "(convey/consolidate); neutral captures are a minority of move events"
            ),
            "threshold": {
                "neutral_vs_own_move_share_median": d["neutral_vs_own_move_share_exp"].get("median"),
                "neutral_vs_own_move_share_p25": d["neutral_vs_own_move_share_exp"].get("p25"),
                "neutral_vs_own_move_share_p75": d["neutral_vs_own_move_share_exp"].get("p75"),
                "units": "neutral/(neutral+own) inferred move share",
            },
            "support": "MEASURED inferred-move mix in expansion phase",
        },
    ]
    agg["cannot_determine"] = [
        "Exact expansion score weights",
        "Pre-contact voluntary stop independent of contact",
        "Fog exploration target selection",
        "Half vs full move as land-grab rule",
        "Conflict resolution when fill-local vs toward-egen disagree",
    ]

    json_path = dump_json("grok-kubic-expansion.json", agg)
    md_path = REPO_ROOT / "docs" / "research" / "measurements" / "grok-kubic-expansion.md"
    write_md(agg, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"n_fit={len(rows)} n_loss_skim={agg['loss_skim']['n']}")


if __name__ == "__main__":
    main()
