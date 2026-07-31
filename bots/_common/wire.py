"""
Competition stdio wire loop shared by every ``bots/<name>/main.py``.

Parses the matchup line protocol and drives ``Agent.act``. Nothing else: this
module sits in every bot's source closure, so it ships in every submission
bundle and is part of every rating identity. Per-turn introspection lives
outside ``bots/`` entirely — ``arena.instrument.runner`` drives this same
protocol from these same helpers and reads ``bots/<name>/probe.py`` alongside
it (docs/arena/trajectories.md).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Type


@dataclass
class Observation:
    H: int
    W: int
    turn: int
    my_land: int
    my_army: int
    opp_land: int
    opp_army: int
    type_grid: List[List[int]]
    owner_grid: List[List[int]]
    army_grid: List[List[int]]


def _read_grid(stdin, H: int) -> List[List[int]]:
    return [[int(x) for x in stdin.readline().split()] for _ in range(H)]


def _read_observation(stdin, H: int, W: int, scalars_line: str) -> Observation:
    turn, my_land, my_army, opp_land, opp_army = (int(x) for x in scalars_line.split())
    type_grid = _read_grid(stdin, H)
    owner_grid = _read_grid(stdin, H)
    army_grid = _read_grid(stdin, H)
    return Observation(
        H=H,
        W=W,
        turn=turn,
        my_land=my_land,
        my_army=my_army,
        opp_land=opp_land,
        opp_army=opp_army,
        type_grid=type_grid,
        owner_grid=owner_grid,
        army_grid=army_grid,
    )


def run_stdio(agent_class: Type) -> None:
    """Run the competition stdin/stdout loop for ``agent_class``."""
    stdin = sys.stdin
    stdout = sys.stdout

    handshake = stdin.readline()
    if not handshake:
        return
    player_id, H, W = (int(x) for x in handshake.split())

    agent = agent_class(player_id=player_id, H=H, W=W)

    while True:
        first = stdin.readline()
        if not first:
            return

        obs = _read_observation(stdin, H, W, first)
        p, r, c, d, s = agent.act(obs)
        stdout.write(f"{p} {r} {c} {d} {s}\n")
        stdout.flush()
