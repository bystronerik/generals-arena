"""Game state: persistent memory, pockets, section priors, economy tracking."""
from __future__ import annotations

from components.clock import Deadline
from components.map_memory import MapMemory
from components.pockets import compute_dead_pockets
from components.sections import SectionPrior
from params import Params


class GameState:
    def __init__(self, H: int, W: int, params: Params):
        self.H = H
        self.W = W
        self.params = params
        self.memory = MapMemory(H, W, params.MIN_GENERAL_DISTANCE)
        self.sections = SectionPrior(H, W, params.SECTION_ROWS, params.SECTION_COLS)
        self.dead_pockets: set[tuple[int, int]] = set()
        self.phase = "search"
        self.precomputed = False
        self.pocket_skip_count = 0
        self._contact_reweighted = False
        self.build_site: tuple[int, int] | None = None
        self.castles_owned = 0
        self.strike_tip: tuple[int, int] | None = None
        self.strike_tip_turn: int = -10_000

    def enemy_land_known(self) -> bool:
        if self.memory.first_contact is not None:
            return True
        return any(
            self.memory.known_owner[r][c] == 2
            for r in range(self.H)
            for c in range(self.W)
        )

    def enemy_footprint(self) -> list[tuple[int, int]]:
        cells: list[tuple[int, int]] = []
        for r in range(self.H):
            for c in range(self.W):
                if self.memory.known_owner[r][c] == 2:
                    cells.append((r, c))
        return cells

    def update(self, obs) -> None:
        self.memory.update(obs)
        if self.memory.enemy_general is not None:
            self.phase = "strike"
        elif self.enemy_land_known():
            self.phase = "contact"
            self.strike_tip = None
        else:
            self.phase = "search"
            self.strike_tip = None

        if (
            self.memory.first_contact is not None
            and not self._contact_reweighted
            and self.memory.enemy_general is None
        ):
            self.sections.reweight_contact(
                self.memory.first_contact, self.params.CONTACT_REWEIGHT
            )
            self._contact_reweighted = True

        if self.memory.candidates:
            self.sections.prune_to_candidates(self.memory.candidates)

    def precompute(self, deadline: Deadline) -> None:
        """First-move grace work: pockets + section seed."""
        if self.precomputed or deadline.expired():
            return
        self.dead_pockets = compute_dead_pockets(
            self.H,
            self.W,
            self.memory.is_passable_belief,
            self.params.POCKET_MAX_CELLS,
        )
        self.pocket_skip_count = len(self.dead_pockets)
        if deadline.expired():
            self.precomputed = True
            return
        if self.memory.own_general is not None:
            self.sections.seed_from_candidates(
                self.memory.candidates, self.memory.own_general
            )
        self.precomputed = True

    def refresh_pockets_light(self) -> None:
        """Cheap pocket refresh after more mountains are known."""
        self.dead_pockets = compute_dead_pockets(
            self.H,
            self.W,
            self.memory.is_passable_belief,
            self.params.POCKET_MAX_CELLS,
        )
        self.pocket_skip_count = len(self.dead_pockets)
