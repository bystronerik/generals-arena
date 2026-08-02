"""Inspect detailed trace metrics for Sosipolis."""
from __future__ import annotations

import json
from pathlib import Path
from arena.records.trajectories import read_jsonl_gz, trace_path, read_trajectory, trajectory_path

GAMES_DIR = Path("data/games/sosipolis-sight1")
TRAJ_DIR = Path("data/trajectories/sosipolis-sight1")

def inspect_game(game_id: str, seat: str):
    print(f"=== Inspecting {game_id} (Sosipolis = seat {seat}) ===")
    t_file = trace_path(game_id, seat, TRAJ_DIR)
    lines = list(read_jsonl_gz(t_file))
    print(f"Total recorded turns in trace: {len(lines)}")

    phases = [l.get("phase") for l in lines]
    cands = [l.get("candidate_count") for l in lines]
    searched = [l.get("searched") for l in lines]
    castles = [l.get("castles_built_probe") for l in lines]

    # Find turns where searched is False in contact phase
    contact_turns = [l for l in lines if l.get("phase") == "contact"]
    contact_unsearched = [l for l in contact_turns if l.get("searched") == False]
    print(f"Contact phase total turns: {len(contact_turns)}")
    print(f"Contact phase UNSEARCHED turns (tip feed / kill / defense / castle): {len(contact_unsearched)} ({len(contact_unsearched)/max(1, len(contact_turns))*100:.1f}%)")

    # Sample candidate counts over time
    step = max(1, len(lines) // 10)
    print("Sample turns (t, phase, cands, searched, castles, sighted):")
    for l in lines[::step]:
        print(f"  t={l.get('t'):<4} phase={l.get('phase'):<8} cands={l.get('candidate_count'):<4} searched={l.get('searched')} castles={l.get('castles_built_probe')} sighted={l.get('enemy_general_sighted')}")

    # Final turn
    if lines:
        l = lines[-1]
        print(f"  FINAL: t={l.get('t'):<4} phase={l.get('phase'):<8} cands={l.get('candidate_count'):<4} searched={l.get('searched')} castles={l.get('castles_built_probe')} sighted={l.get('enemy_general_sighted')}")

if __name__ == "__main__":
    inspect_game("20260802T132018Z_sosipolis_vs_macaria_s0_e727fb1f", "a")
    print()
    inspect_game("20260802T132018Z_macaria_vs_sosipolis_s1_41f439cd", "b")
    print()
    inspect_game("20260802T132018Z_sosipolis_vs_macaria_s4_14cdf58f", "a")
