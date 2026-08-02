#!/usr/bin/env python3
"""Detailed behavioral analysis script for Kubic replays.

Legacy whole-corpus sketch. For the fit/holdout behavioral spec, prefer:

  python3 scripts/grok-analyze/kubic_corpus.py
  python3 scripts/grok-analyze/analyze_kubic_opening.py
  python3 scripts/grok-analyze/analyze_kubic_expansion.py
  python3 scripts/grok-analyze/analyze_kubic_army.py
  python3 scripts/grok-analyze/analyze_kubic_attack.py
  python3 scripts/grok-analyze/analyze_kubic_defense.py
  python3 scripts/grok-analyze/analyze_kubic_timing.py
  python3 scripts/grok-analyze/verify_kubic_holdout.py

Spec: docs/research/strategies/grok-kubic-behavior-spec.md
"""

import sys
from pathlib import Path
import json
from statistics import median, mean

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arena.instrument.replay.loader import iter_replay_paths, load_replay
from arena.instrument.replay.analysis import analyze, ForfeitReplay
from arena.instrument.replay.batch import iter_games

def run_analysis():
    player = "Kubic"
    root = REPO_ROOT / "competition-replays"
    
    wins = []
    losses = []
    draws = []
    forfeits = 0

    print("Iterating through all games for Kubic...")
    
    # We will gather both batch GameLines and full Analysis for detailed events
    games_data = []
    
    for folder, path in iter_replay_paths(player, root=root):
        replay = load_replay(path, queried_player=player, folder=folder)
        if replay.is_forfeit:
            forfeits += 1
            continue
            
        try:
            an = analyze(replay)
        except ForfeitReplay:
            forfeits += 1
            continue
        except Exception as e:
            print(f"Error analyzing {path}: {e}")
            continue

        us = an.us
        them = an.them
        outcome = replay.outcome # 'win', 'lose', 'draw'
        
        # Events summary
        events = an.events
        
        # Castle stats
        castles_built = events.of_kind("castle_built", player=us)
        castle_count = len(castles_built)
        first_castle_tick = castles_built[0].tick if castle_count > 0 else None
        
        # Sight & contact
        contact_events = events.of_kind("first_contact")
        first_contact = contact_events[0].tick if contact_events else None
        
        sight_events = events.of_kind("first_general_sight", player=us)
        first_sight = sight_events[0].tick if sight_events else None
        
        # General captured
        gen_captured = events.of_kind("general_captured")
        kill_turn = replay.total_ticks if outcome == "win" else None
        
        # Phases
        expansion_len = None
        contest_len = None
        strike_len = None
        for p in an.phases:
            if p.name == "expansion":
                expansion_len = p.end - p.start
            elif p.name == "contest":
                contest_len = p.end - p.start
                
        if first_sight is not None:
            strike_len = replay.total_ticks - first_sight
            
        # Path & Stack stats
        path_summary = an.path_summary(us)
        toward_frac = path_summary.toward_fraction
        steps = an.steps[us]
        peak_stack = max(s.army for s in steps) if steps else 0
        
        # Stack mass near sight / kill
        # Look at steps around sight and end
        steps = an.steps[us]
        stack_at_sight = None
        if first_sight is not None:
            for step in steps:
                if step.tick >= first_sight:
                    stack_at_sight = step.army
                    break
                    
        # Peak stack within 20 ticks before kill or during strike
        stack_near_kill = None
        if outcome == "win":
            strike_steps = [s for s in steps if s.tick >= (first_sight if first_sight else 0)]
            if strike_steps:
                stack_near_kill = max(s.army for s in strike_steps)

        # Home bank / general army reserve
        gen_cell = replay.generals[us]
        gen_armies = [tick.armies[gen_cell[0]][gen_cell[1]] for tick in replay.ticks]
        max_gen_army = max(gen_armies) if gen_armies else 0
        gen_army_at_contact = gen_armies[first_contact] if (first_contact and first_contact < len(gen_armies)) else None
        gen_army_at_sight = gen_armies[first_sight] if (first_sight and first_sight < len(gen_armies)) else None

        # Enemy near home check
        # Check if enemy ever reached Manhattan distance <= 1 or 2 from Kubic general
        r0, c0 = gen_cell
        enemy_min_dist_to_home = 999
        for tick in replay.ticks:
            for r in range(replay.rows):
                for c in range(replay.cols):
                    if tick.owners[r][c] == them and tick.armies[r][c] > 0:
                        d = abs(r - r0) + abs(c - c0)
                        if d < enemy_min_dist_to_home:
                            enemy_min_dist_to_home = d

        # Gather waves
        gather_waves = events.of_kind("gather_wave", player=us)
        gw_count = len(gather_waves)
        gw_pre_sight = len([e for e in gather_waves if first_sight is None or e.tick < first_sight])
        gw_post_sight = len([e for e in gather_waves if first_sight is not None and e.tick >= first_sight])
        
        item = {
            "match_id": replay.match_id,
            "outcome": outcome,
            "opponent": replay.name(them),
            "ticks": replay.total_ticks,
            "seat": us,
            "castle_count": castle_count,
            "first_castle_tick": first_castle_tick,
            "first_contact": first_contact,
            "first_sight": first_sight,
            "kill_turn": kill_turn,
            "expansion_len": expansion_len,
            "contest_len": contest_len,
            "strike_len": strike_len,
            "toward_frac": toward_frac,
            "peak_stack": peak_stack,
            "stack_at_sight": stack_at_sight,
            "stack_near_kill": stack_near_kill,
            "max_gen_army": max_gen_army,
            "gen_army_at_contact": gen_army_at_contact,
            "gen_army_at_sight": gen_army_at_sight,
            "enemy_min_dist_to_home": enemy_min_dist_to_home,
            "gather_waves_total": gw_count,
            "gather_waves_pre_sight": gw_pre_sight,
            "gather_waves_post_sight": gw_post_sight,
        }
        
        if outcome == "win":
            wins.append(item)
        elif outcome == "lose":
            losses.append(item)
        else:
            draws.append(item)

    print(f"\nCompleted processing!")
    print(f"Forfeits skipped: {forfeits}")
    print(f"Wins: {len(wins)}, Losses: {len(losses)}, Draws: {len(draws)}")
    
    def get_stats(arr, key):
        vals = [x[key] for x in arr if x[key] is not None]
        if not vals:
            return "N/A"
        return {
            "n": len(vals),
            "median": round(float(median(vals)), 2),
            "mean": round(float(mean(vals)), 2),
            "min": round(float(min(vals)), 2),
            "max": round(float(max(vals)), 2)
        }

    print("\n--- WIN AGGREGATES ---")
    for key in ["ticks", "first_contact", "first_sight", "kill_turn", "first_castle_tick", "castle_count",
                "expansion_len", "contest_len", "strike_len", "toward_frac", "peak_stack",
                "stack_at_sight", "stack_near_kill", "max_gen_army", "gen_army_at_contact", "gen_army_at_sight",
                "gather_waves_total", "gather_waves_pre_sight", "gather_waves_post_sight"]:
        print(f"Wins {key}: {get_stats(wins, key)}")

    print("\n--- LOSS AGGREGATES ---")
    for key in ["ticks", "first_contact", "first_sight", "first_castle_tick", "castle_count",
                "expansion_len", "contest_len", "strike_len", "toward_frac", "peak_stack",
                "stack_at_sight", "max_gen_army", "gen_army_at_contact", "gen_army_at_sight",
                "enemy_min_dist_to_home", "gather_waves_total"]:
        print(f"Losses {key}: {get_stats(losses, key)}")

    # Percentage with >= 1 castle
    win_castles = [w["castle_count"] for w in wins]
    loss_castles = [l["castle_count"] for l in losses]
    print(f"\nWin % >= 1 castle: {sum(1 for c in win_castles if c >= 1) / len(win_castles):.1%}")
    if loss_castles:
        print(f"Loss % >= 1 castle: {sum(1 for c in loss_castles if c >= 1) / len(loss_castles):.1%}")

    # Enemy near home in wins
    enemy_near_win = [w["enemy_min_dist_to_home"] for w in wins]
    print(f"Win % enemy dist <= 1 to home: {sum(1 for d in enemy_near_win if d <= 1) / len(enemy_near_win):.1%}")
    print(f"Win % enemy dist <= 2 to home: {sum(1 for d in enemy_near_win if d <= 2) / len(enemy_near_win):.1%}")

    # Save processed json for deep inspection
    out_file = REPO_ROOT / "docs/research/measurements/kubic_full_analysis.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({"wins": wins, "losses": losses, "draws": draws, "forfeits": forfeits}, f, indent=2)
    print(f"Saved JSON report to {out_file}")

if __name__ == "__main__":
    run_analysis()
