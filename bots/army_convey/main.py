"""
Python starter — game-loop scaffolding.

You shouldn't need to edit this file. Implement your strategy in `agent.py`.

This file does three things:
  1. Defines the `Observation` dataclass that holds one turn's parsed view.
  2. Parses the wire-format frame from stdin into an `Observation`.
  3. Drives the per-turn loop: read frame, call `Agent.act`, write action.

Protocol summary (see competition/protocol.py in the engine repo for the spec):
    Handshake (engine -> agent, once):
        <player_id> <H> <W>
    Per turn (engine -> agent):
        <turn> <my_land> <my_army> <opp_land> <opp_army>
        H lines of W ints   (type grid)
        H lines of W ints   (owner grid)
        H lines of W ints   (army grid)
    Per turn (agent -> engine):
        <pass> <row> <col> <dir> <split>
    Game end: the engine closes our stdin. We exit on EOF.
"""
import sys
from dataclasses import dataclass
from typing import List

from agent import Agent


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


def _read_grid(stdin, H):
    return [[int(x) for x in stdin.readline().split()] for _ in range(H)]


def _read_observation(stdin, H, W, scalars_line):
    turn, my_land, my_army, opp_land, opp_army = (int(x) for x in scalars_line.split())
    type_grid = _read_grid(stdin, H)
    owner_grid = _read_grid(stdin, H)
    army_grid = _read_grid(stdin, H)
    return Observation(
        H=H, W=W, turn=turn,
        my_land=my_land, my_army=my_army,
        opp_land=opp_land, opp_army=opp_army,
        type_grid=type_grid,
        owner_grid=owner_grid,
        army_grid=army_grid,
    )


def main():
    stdin = sys.stdin
    stdout = sys.stdout

    handshake = stdin.readline()
    if not handshake:
        return
    player_id, H, W = (int(x) for x in handshake.split())

    agent = Agent(player_id=player_id, H=H, W=W)

    while True:
        first = stdin.readline()
        if not first:
            return

        obs = _read_observation(stdin, H, W, first)
        p, r, c, d, s = agent.act(obs)

        stdout.write(f"{p} {r} {c} {d} {s}\n")
        stdout.flush()


if __name__ == "__main__":
    main()
