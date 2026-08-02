"""Analyze Sosipolis sight timing on recorded sosipolis-sight1 round."""
from __future__ import annotations

import json
from pathlib import Path
from arena.records.trajectories import read_jsonl_gz, trace_path

GAMES_DIR = Path("data/games/sosipolis-sight1")
TRAJ_DIR = Path("data/trajectories/sosipolis-sight1")

def analyze():
    game_files = sorted(GAMES_DIR.glob("*.json"))
    game_files = [f for f in game_files if f.name != "manifest.json"]

    records = []
    for gf in game_files:
        with open(gf) as f:
            data = json.load(f)

        game_id = data["game_id"]
        bot_a = data["bot_a"]
        bot_b = data["bot_b"]
        seed = data.get("seed")
        if seed is None:
            import re
            m = re.search(r"_s(\d+)_", game_id)
            seed = int(m.group(1)) if m else -1
        winner = data["winner"]
        turns = data["turns"]

        if bot_a == "sosipolis":
            seat = "a"
            opp_seat = "b"
        else:
            seat = "b"
            opp_seat = "a"

        trace_file = trace_path(game_id, seat, TRAJ_DIR)
        t_contact = None
        t_sight = None
        phase_counts = {"search": 0, "contact": 0, "strike": 0}

        if trace_file.exists():
            for line in read_jsonl_gz(trace_file):
                t = line.get("t")
                phase = line.get("phase")
                sighted = line.get("enemy_general_sighted")

                if phase in phase_counts:
                    phase_counts[phase] += 1

                if t_contact is None and phase in ("contact", "strike"):
                    t_contact = t

                if t_sight is None and (sighted == 1 or phase == "strike"):
                    t_sight = t

        if winner == "a":
            winning_bot = bot_a
        elif winner == "b":
            winning_bot = bot_b
        else:
            winning_bot = winner

        records.append({
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
            "contact_to_sight": (t_sight - t_contact) if (t_sight is not None and t_contact is not None) else None,
            "phases": phase_counts,
        })

    records.sort(key=lambda r: (r["seed"], r["seat"]))

    print(f"{'Seed':<5} {'Seat':<5} {'Winner':<10} {'Turns':<6} {'Contact':<8} {'Sight':<8} {'Contact->Sight':<16} {'Phases (S/C/St)':<15}")
    print("-" * 75)

    sighted_count = 0
    sight_turns = []
    contact_turns = []
    c2s_deltas = []

    for r in records:
        win_str = r["winner_bot"]
        s_str = str(r["t_sight"]) if r["t_sight"] is not None else "None"
        c_str = str(r["t_contact"]) if r["t_contact"] is not None else "None"
        c2s_str = str(r["contact_to_sight"]) if r["contact_to_sight"] is not None else "N/A"
        p_str = f"{r['phases']['search']}/{r['phases']['contact']}/{r['phases']['strike']}"

        if r["t_sight"] is not None:
            sighted_count += 1
            sight_turns.append(r["t_sight"])
            if r["contact_to_sight"] is not None:
                c2s_deltas.append(r["contact_to_sight"])

        if r["t_contact"] is not None:
            contact_turns.append(r["t_contact"])

        print(f"{r['seed']:<5} {r['seat']:<5} {win_str:<10} {r['turns']:<6} {c_str:<8} {s_str:<8} {c2s_str:<16} {p_str:<15}")

    print("-" * 75)
    print(f"Total Games: {len(records)}")
    print(f"Sight Rate: {sighted_count}/{len(records)} ({sighted_count/len(records)*100:.1f}%)")
    import numpy as np
    if sight_turns:
        print(f"Median Sight Turn (when sighted): {np.median(sight_turns):.1f} (min={min(sight_turns)}, max={max(sight_turns)})")
    if contact_turns:
        print(f"Median Contact Turn: {np.median(contact_turns):.1f} (min={min(contact_turns)}, max={max(contact_turns)})")
    if c2s_deltas:
        print(f"Median Contact->Sight Delta: {np.median(c2s_deltas):.1f} (min={min(c2s_deltas)}, max={max(c2s_deltas)})")

if __name__ == "__main__":
    analyze()
