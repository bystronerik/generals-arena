"""Oscillation diagnostic grid for joe-rs selection knobs (S4 shortlist).

Runs the limit-cycle diagnostic setup — macaria vs joe-rs, seeds 0-4, both
seat orders — once per knob arm, records joe-rs's wire traffic through a
tee, and reports the oscillation metrics that motivated S4
(docs/research/strategies/joe-rs-noundo.md):

  - outcome and end turn;
  - undo rate: moves that exactly invert the previous move;
  - strict cycle coverage (period <= 4, >= 3 repetitions), the
    joe-argmax-limit-cycle detector;
  - worst confinement: minimum distinct source cells over any trailing
    50-move window.

These are **adhoc, unrated games**: nothing is stored under data/games/ and
no rating is touched — the grid shortlists a knob value, the rated round
prices it (experiment-protocol.md). Wire logs land under
data/joe/osc-grid/<arm>/ (derived, gitignored) and are overwritten per run.

Usage:
    .venv/bin/python bots/joe-rs/tools/osc_grid.py [--deltas 0,2,4]
        [--seeds 0-4] [--jobs 5] [--json OUT]
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
OUT = REPO / "data" / "joe" / "osc-grid"
DIRS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def run_game(delta: float, seed: int, joers_seat: int, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"s{seed}-seat{joers_seat}"
    wrapper = out_dir / f"_tee-{tag}.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'tee "$JOE_TEE_IN" | bash "$JOE_RUN_SH" | tee "$JOE_TEE_OUT"\n')
    wrapper.chmod(0o755)
    env = dict(os.environ)
    env["PYTHON"] = str(REPO / ".venv" / "bin" / "python")
    env["JOE_RUN_SH"] = str(REPO / "bots" / "joe-rs" / "run.sh")
    env["JOE_TEE_IN"] = str(out_dir / f"{tag}.in.log")
    env["JOE_TEE_OUT"] = str(out_dir / f"{tag}.out.log")
    env["JOE_RS_NOUNDO"] = str(delta)
    macaria = str(REPO / "bots" / "macaria" / "run.sh")
    bots = [macaria, str(wrapper)] if joers_seat == 1 else [str(wrapper), macaria]
    proc = subprocess.run(
        [env["PYTHON"], str(REPO / "competition-module" / "competition" / "matchup.py"),
         *bots, "--mode", "competition", "--seed", str(seed)],
        env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"matchup failed ({tag}, delta {delta}):\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
    outcome, turn = "truncated", None
    for line in proc.stdout.splitlines():
        if "captured the enemy general" in line:
            winner = int(line.split("player ")[1].split()[0])
            outcome = "win" if winner == joers_seat else "loss"
            turn = int(line.split("turn ")[1].split(":")[0])
    row = analyze(out_dir / f"{tag}.in.log", out_dir / f"{tag}.out.log")
    row.update(delta=delta, seed=seed, seat=joers_seat, outcome=outcome,
               end_turn=turn if turn is not None else row["turns"])
    return row


def analyze(in_log: Path, out_log: Path) -> dict:
    lines = in_log.read_text().splitlines()
    H = int(lines[0].split()[1])
    frame_len = 1 + 3 * H
    turns = [int(lines[p].split()[0])
             for p in range(1, len(lines) - frame_len + 1, frame_len)]
    replies = [[int(x) for x in l.split()] for l in out_log.read_text().splitlines()]
    T = min(len(turns), len(replies))

    moves = []
    for t in range(T):
        p, r, c, d, s = replies[t]
        if p == 0:
            dr, dc = DIRS[d]
            moves.append(((r, c), (r + dr, c + dc)))
        else:
            moves.append(None)
    n_moves = sum(1 for m in moves if m)
    undos = sum(1 for i in range(1, T)
                if moves[i - 1] and moves[i]
                and moves[i - 1][0] == moves[i][1] and moves[i - 1][1] == moves[i][0])

    # Trail revisits: a move landing on one of the last 8 move-source
    # cells — the any-period circuit symptom S4's trail penalty taxes.
    revisits = 0
    sources = []
    for m in moves:
        if not m:
            continue
        if m[1] in sources[-8:]:
            revisits += 1
        sources.append(m[0])

    # Strict cycles over move actions only: the opening pass run is a
    # period-1 "cycle" in every game and would flatten the metric.
    acts = [tuple(r) for r in replies[:T]]
    cycle_turns = 0
    i = 0
    while i < T:
        best = 0
        for p in (1, 2, 3, 4):
            reps = 0
            while i + (reps + 1) * p <= T and acts[i + reps * p:i + (reps + 1) * p] == acts[i:i + p]:
                reps += 1
            if reps >= 3 and any(a[0] == 0 for a in acts[i:i + p]):
                best = max(best, reps * p)
        if best:
            cycle_turns += best
            i += best
        else:
            i += 1

    window, min_sources = 50, None
    for i in range(window, T):
        w = [m for m in moves[i - window:i] if m]
        if len(w) >= 40:
            srcs = len({m[0] for m in w})
            min_sources = srcs if min_sources is None else min(min_sources, srcs)

    return {"turns": T, "moves": n_moves, "undos": undos,
            "undo_rate": round(undos / max(1, n_moves), 4),
            "revisits": revisits,
            "revisit_rate": round(revisits / max(1, n_moves), 4),
            "cycle_turns": cycle_turns,
            "cycle_frac": round(cycle_turns / max(1, T), 4),
            "min_sources_w50": min_sources}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deltas", default="0,2,4")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    deltas = [float(x) for x in args.deltas.split(",")]
    if "-" in args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(x) for x in args.seeds.split(",")]

    jobs = [(d, s, seat, OUT / f"delta{d:g}")
            for d in deltas for s in seeds for seat in (0, 1)]
    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_game, *j) for j in jobs]
        for f in futures:
            row = f.result()
            rows.append(row)
            print(f"[grid] delta {row['delta']:g} seed {row['seed']} seat {row['seat']}: "
                  f"{row['outcome']} t={row['end_turn']}, undo {100*row['undo_rate']:.1f}%, "
                  f"revisit {100*row['revisit_rate']:.1f}%, "
                  f"cycles {100*row['cycle_frac']:.1f}%, min_src {row['min_sources_w50']}",
                  flush=True)

    print(f"\n{'delta':>6} {'wins':>5} {'mean_end':>9} {'undo%':>7} {'revisit%':>9} {'cycle%':>7} {'min_src':>8}")
    for d in deltas:
        rr = [r for r in rows if r["delta"] == d]
        wins = sum(r["outcome"] == "win" for r in rr)
        me = sum(r["end_turn"] for r in rr) / len(rr)
        undo = 100 * sum(r["undos"] for r in rr) / max(1, sum(r["moves"] for r in rr))
        rev = 100 * sum(r["revisits"] for r in rr) / max(1, sum(r["moves"] for r in rr))
        cyc = 100 * sum(r["cycle_turns"] for r in rr) / sum(r["turns"] for r in rr)
        msrc = min(r["min_sources_w50"] for r in rr if r["min_sources_w50"] is not None)
        print(f"{d:>6g} {wins:>4}/{len(rr)} {me:>9.0f} {undo:>7.1f} {rev:>9.1f} {cyc:>7.1f} {msrc:>8}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1) + "\n")
        print(f"[grid] wrote {args.json}")


if __name__ == "__main__":
    main()
