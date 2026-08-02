"""Hand-built board states for sosipolis unit tests.

These fixtures recreate decision scenes (contact latch, adjacent kill) without
a live match. Army is placed where the test needs it; nothing checks
conservation across turns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from arena.bot_api import UnifiedObservation

Cell = tuple[int, int]

T_FOG = 0
T_PLAIN = 1
T_MOUNTAIN = 2
T_CASTLE = 3
T_GENERAL = 4

# Wire directions: N S W E — must match bots/sosipolis/params.DIRECTIONS.
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def grid(H: int, W: int, fill: int = 0) -> list[list[int]]:
    return [[fill] * W for _ in range(H)]


def clone_grid(g: list[list[int]]) -> list[list[int]]:
    return [row[:] for row in g]


def plain(H: int, W: int) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """All-plain empty boards: types=PLAIN, owner=0, army=0."""
    return grid(H, W, T_PLAIN), grid(H, W), grid(H, W)


def corridor(
    H: int, W: int, *, row: int | None = None
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Mountain walls with one passable row (default mid-row). Unique BFS path."""
    if row is None:
        row = H // 2
    types = grid(H, W, T_MOUNTAIN)
    for c in range(W):
        types[row][c] = T_PLAIN
    return types, grid(H, W), grid(H, W)


def make_obs(
    types: list[list[int]],
    owner: list[list[int]],
    army: list[list[int]],
    turn: int,
    **overrides: Any,
):
    """Build UnifiedObservation from three grids; stats scan unless overridden."""
    H, W = len(types), len(types[0])
    fixture = BoardFixture(
        label="ad_hoc",
        turn=turn,
        types=types,
        owner=owner,
        army=army,
    )
    return fixture.obs(**overrides)


def probe_macro(
    *,
    kind: str,
    waypoint: Cell,
    evidence_anchor: Cell | None = None,
    candidate_cells: frozenset[Cell] | None = None,
    score: float = 1.0,
):
    """Build a ProbeMacro; import only inside sosipolis_imports()."""
    from components.contact_mcts import ProbeMacro

    return ProbeMacro(
        kind=kind,
        waypoint=waypoint,
        evidence_anchor=evidence_anchor or waypoint,
        candidate_cells=candidate_cells or frozenset({waypoint}),
        score=score,
    )


def seeded_contact_mcts(
    *,
    belief: dict[Cell, float] | None = None,
    tip_bfs: dict[Cell, int] | None = None,
    footprint_bfs: dict[Cell, int] | None = None,
    waypoint_bfs: dict[Cell, int] | None = None,
    reveal_count: dict[Cell, int] | None = None,
):
    """ContactMCTS with cache fields pre-seeded; skip _refresh_cache in tests."""
    from components.contact_mcts import ContactMCTS
    from params import PARAMS

    mcts = ContactMCTS(PARAMS)
    cache = mcts._cache
    if belief is not None:
        cache.belief = dict(belief)
        cache.belief_sum = sum(belief.values())
    if tip_bfs is not None:
        cache.tip_bfs = dict(tip_bfs)
    if footprint_bfs is not None:
        cache.footprint_bfs = dict(footprint_bfs)
    if waypoint_bfs is not None:
        cache.waypoint_bfs = dict(waypoint_bfs)
    if reveal_count is not None:
        cache.reveal_count = dict(reveal_count)
    return mcts


@dataclass
class BoardFixture:
    """One observation plus optional MapMemory seed applied before update."""

    label: str
    turn: int
    types: list[list[int]]
    owner: list[list[int]]
    army: list[list[int]]
    own_general: Cell | None = None
    first_contact: Cell | None = None
    first_contact_turn: int = -1
    primary_path: tuple[Cell, ...] = ()
    # When True, seed leaves enemy_general unset so update/act must latch it.
    clear_enemy_general: bool = True
    notes: str = ""

    @property
    def H(self) -> int:
        return len(self.types)

    @property
    def W(self) -> int:
        return len(self.types[0])

    def obs(self, **overrides: Any) -> UnifiedObservation:
        types = clone_grid(self.types)
        owner = clone_grid(self.owner)
        army = clone_grid(self.army)
        stats = {"my_land": 0, "my_army": 0, "opp_land": 0, "opp_army": 0}
        for r in range(self.H):
            for c in range(self.W):
                if owner[r][c] == 1:
                    stats["my_land"] += 1
                    stats["my_army"] += army[r][c]
                elif owner[r][c] == 2:
                    stats["opp_land"] += 1
                    stats["opp_army"] += army[r][c]
        stats.update(overrides)
        return UnifiedObservation(
            H=self.H,
            W=self.W,
            turn=self.turn,
            type_grid=types,
            owner_grid=owner,
            army_grid=army,
            **stats,
        )


def action_dst(action: tuple[int, int, int, int, int]) -> Cell | None:
    """Destination cell of a leave-1 move, or None for pass/build."""
    if action[0] != 0:
        return None
    _, r, c, d, _ = action
    if not (0 <= d < len(DIRECTIONS)):
        return None
    dr, dc = DIRECTIONS[d]
    return r + dr, c + dc


def action_src(action: tuple[int, int, int, int, int]) -> Cell | None:
    if action[0] != 0:
        return None
    return action[1], action[2]


# ---------------------------------------------------------------------------
# Named scenes
# ---------------------------------------------------------------------------


def contact_then_off_axis_general() -> BoardFixture:
    """East first-contact, then off-axis enemy general north (turn-572 latch).

    Reproduces the MapMemory `t`-shadow defect: without the fix,
    `enemy_general` stays None after this observation.
    """
    H, W = 8, 8
    types = grid(H, W, T_PLAIN)
    owner = grid(H, W)
    army = grid(H, W)
    types[7][0] = T_GENERAL
    owner[7][0] = 1
    army[7][0] = 5
    # Keep the early contact tile visible so memory stays coherent.
    owner[7][3] = 2
    army[7][3] = 3
    types[1][0] = T_GENERAL
    owner[1][0] = 2
    army[1][0] = 8
    return BoardFixture(
        label="off_axis_general_sight",
        turn=572,
        types=types,
        owner=owner,
        army=army,
        own_general=(7, 0),
        first_contact=(7, 3),
        first_contact_turn=40,
        primary_path=((7, 3),),
        notes="GUI miss class: general off first-contact axis, late sight.",
    )


def contact_seed_only() -> BoardFixture:
    """Turn-40 east contact used to prime MapMemory before off-axis sight."""
    H, W = 8, 8
    types = grid(H, W, T_PLAIN)
    owner = grid(H, W)
    army = grid(H, W)
    types[7][0] = T_GENERAL
    owner[7][0] = 1
    army[7][0] = 5
    owner[7][3] = 2
    army[7][3] = 3
    return BoardFixture(
        label="east_contact_prime",
        turn=40,
        types=types,
        owner=owner,
        army=army,
    )


def adjacent_kill(
    *,
    stack_army: int,
    general_army: int,
    turn: int = 572,
    label: str | None = None,
) -> BoardFixture:
    """Own stack west of enemy general; home general in the corner.

    Default sizes match the reported miss: 12 vs 8 on a gather-clock turn
    (572 % 50 == 22 ∈ [10, 27]).
    """
    H, W = 5, 5
    types = grid(H, W, T_PLAIN)
    owner = grid(H, W)
    army = grid(H, W)
    types[0][0] = T_GENERAL
    owner[0][0] = 1
    army[0][0] = 3
    types[2][2] = T_GENERAL
    owner[2][2] = 2
    army[2][2] = general_army
    owner[2][1] = 1
    army[2][1] = stack_army
    return BoardFixture(
        label=label or f"adjacent_{stack_army}v{general_army}_t{turn}",
        turn=turn,
        types=types,
        owner=owner,
        army=army,
        own_general=(0, 0),
        first_contact=(4, 4),
        first_contact_turn=40,
        primary_path=((4, 4),),
        notes="Stack at (2,1) must capture general at (2,2) when legal.",
    )


def adjacent_kill_reported() -> BoardFixture:
    """Canonical GUI report: turn 572, 12 army next to general with 8."""
    return adjacent_kill(stack_army=12, general_army=8, turn=572, label="kill_12v8_t572")


def same_turn_on_axis_general() -> BoardFixture:
    """First sight of enemy general on the contact axis in one update.

    Old bug: projection overwrote `t` on the same cell before the latch
    check, so even on-axis generals missed the first turn.
    """
    H, W = 6, 6
    types = grid(H, W, T_PLAIN)
    owner = grid(H, W)
    army = grid(H, W)
    types[5][0] = T_GENERAL
    owner[5][0] = 1
    army[5][0] = 4
    # Prior contact one step east of home.
    owner[5][2] = 2
    army[5][2] = 2
    # Enemy general further east on the same row (on-axis).
    types[5][4] = T_GENERAL
    owner[5][4] = 2
    army[5][4] = 6
    return BoardFixture(
        label="same_turn_on_axis_general",
        turn=80,
        types=types,
        owner=owner,
        army=army,
        own_general=(5, 0),
        first_contact=(5, 2),
        first_contact_turn=50,
        primary_path=((5, 2),),
    )


FIXTURES: dict[str, BoardFixture] = {
    f.label: f
    for f in (
        contact_seed_only(),
        contact_then_off_axis_general(),
        adjacent_kill_reported(),
        same_turn_on_axis_general(),
        adjacent_kill(stack_army=10, general_army=8, turn=572, label="kill_exact_10v8"),
        adjacent_kill(stack_army=9, general_army=8, turn=572, label="refuse_9v8"),
        adjacent_kill(stack_army=2, general_army=99, turn=800, label="deathtouch_2v99"),
    )
}
