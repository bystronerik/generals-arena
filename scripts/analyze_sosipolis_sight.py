"""Analyze Sosipolis sight timing on a recorded round."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from arena.records.trajectories import read_jsonl_gz, trace_path


def analyze(round_name: str) -> None:
    games_dir = Path(f"data/games/{round_name}")
    traj_dir = Path(f"data/trajectories/{round_name}")
    game_files = sorted(
        f for f in games_dir.glob("*.json") if f.name != "manifest.json"
    )

    records = []
    for gf in game_files:
        with open(gf) as f:
            data = json.load(f)

        game_id = data["game_id"]
        bot_a = data["bot_a"]
        bot_b = data["bot_b"]
        seed = data.get("seed")
        if seed is None:
            m = re.search(r"_s(\d+)_", game_id)
            seed = int(m.group(1)) if m else -1
        winner = data["winner"]
        turns = data["turns"]

        seat = "a" if bot_a == "sosipolis" else "b"

        trace_file = trace_path(game_id, seat, traj_dir)
        t_contact = None
        t_sight = None
        phase_counts = {"search": 0, "contact": 0, "strike": 0}
        distinct_waypoints: set[str] = set()
        max_switches = 0
        commit_ages: list[int] = []
        macro_kinds: Counter[str] = Counter()
        cand_at_contact = None
        cand_at_sight = None
        cand_at_end = None

        if trace_file.exists():
            for line in read_jsonl_gz(trace_file):
                t = line.get("t")
                phase = line.get("phase")
                sighted = line.get("enemy_general_sighted")
                wp = line.get("contact_waypoint")
                switches = line.get("contact_switches")
                commit_turn = line.get("contact_commit_turn")
                macro = line.get("contact_macro")
                cands = line.get("candidate_count")

                if phase in phase_counts:
                    phase_counts[phase] += 1

                if t_contact is None and phase in ("contact", "strike"):
                    t_contact = t
                    if cands is not None:
                        cand_at_contact = cands

                if t_sight is None and (
                    sighted is True or sighted == 1 or phase == "strike"
                ):
                    t_sight = t
                    if cands is not None:
                        cand_at_sight = cands

                if phase == "contact":
                    if isinstance(wp, str) and wp not in ("none", ""):
                        distinct_waypoints.add(wp)
                    if isinstance(switches, int):
                        max_switches = max(max_switches, switches)
                    if (
                        isinstance(commit_turn, int)
                        and commit_turn >= 0
                        and isinstance(t, int)
                    ):
                        commit_ages.append(t - commit_turn)
                    if isinstance(macro, str) and macro not in ("none", ""):
                        macro_kinds[macro] += 1
                if cands is not None:
                    cand_at_end = cands

        if winner == "a":
            winning_bot = bot_a
        elif winner == "b":
            winning_bot = bot_b
        else:
            winning_bot = winner

        records.append(
            {
                "game_id": game_id,
                "seed": seed,
                "seat": seat,
                "bot_a": bot_a,
                "bot_b": bot_b,
                "winner_seat": winner,
                "winner_bot": winning_bot,
                "turns": turns,
                "t_contact": t_contact,
                "t_sight": t_sight,
                "contact_to_sight": (
                    (t_sight - t_contact)
                    if (t_sight is not None and t_contact is not None)
                    else None
                ),
                "phases": phase_counts,
                "distinct_waypoints": len(distinct_waypoints),
                "max_switches": max_switches,
                "median_commit_age": (
                    float(np.median(commit_ages)) if commit_ages else None
                ),
                "macro_kinds": dict(macro_kinds),
                "cand_at_contact": cand_at_contact,
                "cand_at_sight": cand_at_sight,
                "cand_at_end": cand_at_end,
            }
        )

    records.sort(key=lambda r: (r["seed"], r["seat"]))

    print(f"Round: {round_name}")
    print(
        f"{'Seed':<5} {'Seat':<5} {'Winner':<10} {'Turns':<6} "
        f"{'Contact':<8} {'Sight':<8} {'Contact->Sight':<16} {'Phases (S/C/St)':<15}"
    )
    print("-" * 75)

    sighted_count = 0
    sight_turns = []
    contact_turns = []
    c2s_deltas = []
    wins = 0
    distinct_hunts = []
    switch_counts = []
    commit_age_medians = []
    all_macros: Counter[str] = Counter()

    for r in records:
        win_str = r["winner_bot"]
        s_str = str(r["t_sight"]) if r["t_sight"] is not None else "None"
        c_str = str(r["t_contact"]) if r["t_contact"] is not None else "None"
        c2s_str = (
            str(r["contact_to_sight"]) if r["contact_to_sight"] is not None else "N/A"
        )
        p_str = (
            f"{r['phases']['search']}/{r['phases']['contact']}/{r['phases']['strike']}"
        )

        if r["winner_bot"] == "sosipolis":
            wins += 1

        if r["t_sight"] is not None:
            sighted_count += 1
            sight_turns.append(r["t_sight"])
            if r["contact_to_sight"] is not None:
                c2s_deltas.append(r["contact_to_sight"])

        if r["t_contact"] is not None:
            contact_turns.append(r["t_contact"])
            distinct_hunts.append(r["distinct_waypoints"])
            switch_counts.append(r["max_switches"])
            if r["median_commit_age"] is not None:
                commit_age_medians.append(r["median_commit_age"])
            for k, v in r["macro_kinds"].items():
                all_macros[k] += v

        print(
            f"{r['seed']:<5} {r['seat']:<5} {win_str:<10} {r['turns']:<6} "
            f"{c_str:<8} {s_str:<8} {c2s_str:<16} {p_str:<15}"
        )

    print("-" * 75)
    print(f"Total Games: {len(records)}")
    print(f"Sosipolis W-L: {wins}-{len(records) - wins}")
    print(
        f"Sight Rate: {sighted_count}/{len(records)} "
        f"({sighted_count / len(records) * 100:.1f}%)"
    )
    if sight_turns:
        print(
            f"Median Sight Turn (when sighted): {np.median(sight_turns):.1f} "
            f"(min={min(sight_turns)}, max={max(sight_turns)})"
        )
    if contact_turns:
        print(
            f"Median Contact Turn: {np.median(contact_turns):.1f} "
            f"(min={min(contact_turns)}, max={max(contact_turns)})"
        )
    else:
        print("Median Contact Turn: n/a (no contact)")
    if c2s_deltas:
        print(
            f"Median Contact->Sight Delta: {np.median(c2s_deltas):.1f} "
            f"(min={min(c2s_deltas)}, max={max(c2s_deltas)})"
        )
    if distinct_hunts:
        print(
            f"Median Distinct Contact Waypoints: {np.median(distinct_hunts):.1f} "
            f"(min={min(distinct_hunts)}, max={max(distinct_hunts)})"
        )
    if switch_counts:
        print(
            f"Median Contact Switches: {np.median(switch_counts):.1f} "
            f"(min={min(switch_counts)}, max={max(switch_counts)})"
        )
    if commit_age_medians:
        print(
            f"Median Commitment Age (per-game median): "
            f"{np.median(commit_age_medians):.1f}"
        )
    if all_macros:
        print(f"Macro Kind Counts: {dict(all_macros)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--round",
        default="sosipolis-sight3",
        help="Round name under data/games and data/trajectories",
    )
    args = parser.parse_args()
    analyze(args.round)
