"""Check turn-by-turn stack army and search status during contact phase."""
from __future__ import annotations

import json
from pathlib import Path
from arena.records.trajectories import read_jsonl_gz, trace_path

TRAJ_DIR = Path("data/trajectories/sosipolis-sight1")

def inspect_contact_turns(game_id: str, seat: str):
    t_file = trace_path(game_id, seat, TRAJ_DIR)
    lines = list(read_jsonl_gz(t_file))
    print(f"Turn-by-turn contact phase for {game_id} seat {seat}:")
    for l in lines:
        if l.get("phase") == "contact":
            print(f"t={l.get('t'):<4} searched={str(l.get('searched')):<5} cands={l.get('candidate_count'):<4} iters={l.get('search_iters'):<3}")

if __name__ == "__main__":
    inspect_contact_turns("20260802T132018Z_sosipolis_vs_macaria_s0_e727fb1f", "a")
