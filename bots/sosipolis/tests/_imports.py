"""Import sosipolis as top-level modules, then scrub sys.modules.

Bot code uses bare imports (`from params import ...`, `from components...`).
Each test that imports bot code must use `sosipolis_imports()` so one bot's
modules do not leak into another bot's test in the same pytest session.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from boards import BoardFixture

BOT_DIR = Path(__file__).resolve().parents[1]

_BOT_MODULES = (
    "brain",
    "params",
    "state",
    "agent",
    "probe",
    "stdio",
    "mcts_diag",
)


@contextmanager
def sosipolis_imports():
    """Import this bot as top-level modules, then scrub sys.modules."""
    path = str(BOT_DIR)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        if path in sys.path:
            sys.path.remove(path)
        for name in list(sys.modules):
            if name in _BOT_MODULES or name.startswith("components"):
                sys.modules.pop(name, None)


def seed_memory(agent, fixture: BoardFixture) -> None:
    mem = agent.state.memory
    if fixture.own_general is not None:
        mem.own_general = fixture.own_general
    if fixture.first_contact is not None:
        mem.first_contact = fixture.first_contact
        mem.first_contact_turn = fixture.first_contact_turn
        mem.enemy_seen.add(fixture.first_contact)
        mem.known_owner[fixture.first_contact[0]][fixture.first_contact[1]] = 2
    for cell in fixture.primary_path:
        mem.primary_path.add(cell)
        mem.enemy_seen.add(cell)
    if fixture.own_general is not None:
        mem._seeded = True
    if fixture.clear_enemy_general:
        mem.enemy_general = None


def agent_on(fixture: BoardFixture):
    from brain import Agent

    agent = Agent(player_id=0, H=fixture.H, W=fixture.W)
    seed_memory(agent, fixture)
    return agent
