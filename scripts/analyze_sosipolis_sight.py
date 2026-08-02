"""Analyze contact→sight (and sight→kill) on a recorded round.

Uses the shared sticky probe `enemy_land_visible` when present. Falls back to
sosipolis phase markers for older traces.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from arena.records.trajectories import read_jsonl_gz, trace_path


def _truthy(value) -> bool:
    return value is True or value == 1


def _median(xs: list[float]) -> float | None:
    return float(statistics.median(xs)) if xs else None


def analyze(round_name: str, bot_name: str) -> None:
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
        truncated = bool(data.get("truncated"))

        seats: list[str] = []
        if bot_a == bot_name:
            seats.append("a")
        if bot_b == bot_name:
            seats.append("b")
        if not seats:
            continue

        for seat in seats:
            opponent = bot_b if seat == "a" else bot_a
            trace_file = trace_path(game_id, seat, traj_dir)
            t_contact = None
            t_sight = None
            used_land_flag = False
            phase_counts = {"search": 0, "contact": 0, "strike": 0}
            distinct_waypoints: set[str] = set()
            max_switches = 0
            commit_ages: list[int] = []
            macro_kinds: Counter[str] = Counter()

            if trace_file.exists():
                for line in read_jsonl_gz(trace_file):
                    t = line.get("t")
                    phase = line.get("phase")
                    sighted = line.get("enemy_general_sighted")
                    land_vis = line.get("enemy_land_visible")
                    wp = line.get("contact_waypoint")
                    switches = line.get("contact_switches")
                    commit_turn = line.get("contact_commit_turn")
                    macro = line.get("contact_macro")

                    if phase in phase_counts:
                        phase_counts[phase] += 1

                    if t_contact is None:
                        if land_vis is not None:
                            used_land_flag = True
                            if _truthy(land_vis):
                                t_contact = t
                        elif phase in ("contact", "strike"):
                            t_contact = t

                    if t_sight is None and (
                        _truthy(sighted) or (not used_land_flag and phase == "strike")
                    ):
                        t_sight = t

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

            if winner == "a":
                winning_bot = bot_a
            elif winner == "b":
                winning_bot = bot_b
            else:
                winning_bot = winner

            won = winning_bot == bot_name and not truncated
            sight_to_kill = None
            if t_sight is not None and won:
                sight_to_kill = turns - t_sight

            records.append(
                {
                    "game_id": game_id,
                    "seed": seed,
                    "seat": seat,
                    "bot": bot_name,
                    "opponent": opponent,
                    "winner_bot": winning_bot,
                    "won": won,
                    "turns": turns,
                    "t_contact": t_contact,
                    "t_sight": t_sight,
                    "contact_to_sight": (
                        (t_sight - t_contact)
                        if (t_sight is not None and t_contact is not None)
                        else None
                    ),
                    "sight_to_kill": sight_to_kill,
                    "contact_source": (
                        "enemy_land_visible" if used_land_flag else "phase_fallback"
                    ),
                    "phases": phase_counts,
                    "distinct_waypoints": len(distinct_waypoints),
                    "max_switches": max_switches,
                    "median_commit_age": (
                        float(np.median(commit_ages)) if commit_ages else None
                    ),
                    "macro_kinds": dict(macro_kinds),
                }
            )

    records.sort(key=lambda r: (r["seed"], r["seat"]))

    print(f"Round: {round_name}  bot={bot_name}")
    print(
        f"{'Seed':<6} {'Seat':<5} {'Opp':<12} {'Winner':<12} {'Turns':<6} "
        f"{'Contact':<8} {'Sight':<8} {'C→S':<8} {'S→K':<8}"
    )
    print("-" * 90)

    contact_turns = []
    sight_turns = []
    c2s = []
    s2k = []
    wins = 0
    contact_sources: Counter[str] = Counter()

    for r in records:
        if r["won"]:
            wins += 1
        if r["t_contact"] is not None:
            contact_turns.append(r["t_contact"])
        if r["t_sight"] is not None:
            sight_turns.append(r["t_sight"])
        if r["contact_to_sight"] is not None:
            c2s.append(r["contact_to_sight"])
        if r["sight_to_kill"] is not None:
            s2k.append(r["sight_to_kill"])
        contact_sources[r["contact_source"]] += 1

        print(
            f"{r['seed']:<6} {r['seat']:<5} {r['opponent']:<12} "
            f"{r['winner_bot']:<12} {r['turns']:<6} "
            f"{str(r['t_contact']):<8} {str(r['t_sight']):<8} "
            f"{str(r['contact_to_sight']):<8} {str(r['sight_to_kill']):<8}"
        )

    n = len(records)
    print("-" * 90)
    print(f"Seats: {n}")
    print(f"{bot_name} W-L: {wins}-{n - wins}")
    print(
        f"Sight rate: {len(sight_turns)}/{n} "
        f"({(len(sight_turns) / n * 100) if n else 0:.1f}%)"
    )
    print(
        f"Contact rate: {len(contact_turns)}/{n} "
        f"({(len(contact_turns) / n * 100) if n else 0:.1f}%)"
    )
    print(f"Contact source counts: {dict(contact_sources)}")
    print(f"Median contact: {_median(contact_turns)}")
    print(f"Median sight (when sighted): {_median(sight_turns)}")
    print(f"Median contact→sight: {_median(c2s)} (n={len(c2s)})")
    print(f"Median sight→kill (wins): {_median(s2k)} (n={len(s2k)})")
    if bot_name == "sosipolis":
        print("Kubic targets: contact≤82, contact→sight≤95.5, sight→kill~20–24")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--round",
        default="sosipolis-sight3",
        help="Round name under data/games and data/trajectories",
    )
    parser.add_argument(
        "--bot",
        default="sosipolis",
        help="Bot id whose seats to analyze (sosipolis or macaria)",
    )
    args = parser.parse_args()
    analyze(args.round, args.bot)
