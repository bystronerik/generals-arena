#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path

path = Path("docs/research/measurements/kubic_full_analysis.json")
data = json.loads(path.read_text())

wins = data["wins"]
losses = data["losses"]
draws = data["draws"]

print("=== OPPONENT MIX IN WINS ===")
win_opps = Counter(w["opponent"] for w in wins)
for opp, cnt in win_opps.most_common():
    print(f"  {opp}: {cnt}")

print("\n=== OPPONENT MIX IN LOSSES ===")
loss_opps = Counter(l["opponent"] for l in losses)
for opp, cnt in loss_opps.most_common():
    print(f"  {opp}: {cnt}")

print("\n=== LOSS GAMES DETAILS ===")
for l in losses:
    print(f"ID: {l['match_id']}, Opp: {l['opponent']}, Ticks: {l['ticks']}, Contact: {l['first_contact']}, Sight: {l['first_sight']}, Castles: {l['castle_count']}, FirstCastle: {l['first_castle_tick']}, PeakStack: {l['peak_stack']}, Waves: {l['gather_waves_total']}")

print("\n=== SAMPLE WIN GAMES DETAILS ===")
# sample fast wins, median wins, long wins
wins_sorted = sorted(wins, key=lambda x: x["ticks"])
print("Shortest win:", wins_sorted[0]["match_id"], "ticks:", wins_sorted[0]["ticks"])
print("Median win:", wins_sorted[len(wins_sorted)//2]["match_id"], "ticks:", wins_sorted[len(wins_sorted)//2]["ticks"])
print("Long win:", wins_sorted[-1]["match_id"], "ticks:", wins_sorted[-1]["ticks"])

