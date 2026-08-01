"""Proteus-owned observation latches, beside the shared OpponentModel.

`OpponentModel` keeps a running *minimum* enemy distance and a running
*maximum* enemy stack. Both are latched over the whole game, independently —
so the shipped classifier's `deep_incursion` test
(``closest_enemy_dist <= 8 and biggest_enemy_stack >= 15``) fires on a pair of
facts that need never have been true at the same time: a border cell nibbled
at distance 8 on turn 90, and an unrelated 20-stack seen across the map on
turn 300, together read as "a big stack came to our door".

`HomePressure` latches the **conjunction** instead: the largest stack ever
seen *while that stack was near our general*. That is the mechanical
difference between an opponent whose expansion happened to reach us and one
that walked a sized fist at our general — the only distinction that changes
which core we should be playing (docs/research/strategies/proteus.md §3).

Lives here rather than in `_common/oppmodel.py` on purpose: that module is in
the source closure of blitz, boom, metro and aegis as well, and editing it
would fork four other bots' content hashes for a signal only proteus reads
(docs/arena/decision-rule.md, "connectivity").
"""
from __future__ import annotations

from dataclasses import dataclass, field

from _common.strategy_common import locate_own_general
from _common.tactics import multi_bfs, visible_enemy_tiles


@dataclass
class HomePressure:
    """How much force the opponent has actually brought to our door."""

    #: BFS steps from our general within which an enemy stack counts as "at
    #: the door". Tuned on the roster sweep: 8 keeps the false-positive rate
    #: on pure expanders at 4% of games, 12 triples it.
    radius: int = 8

    #: Largest enemy army ever seen on a cell within `radius` — latched,
    #: because an opponent that once walked a fist at us can do it again.
    max_stack_near: int = 0
    #: Same, but for the current turn only.
    stack_near_now: int = 0
    #: Turns on which any enemy cell was inside `radius`.
    turns_near: int = 0
    #: First turn any enemy stack at all stood inside `radius`.
    first_near_turn: int | None = None
    #: Turn `max_stack_near` first reached `duel_army`. *When* the fist
    #: arrived is as diagnostic as whether it did: on the roster sweep a
    #: genuine rusher's first fist lands at median turn 169-181, while the
    #: same measurement on a turtle or a castle bot is its counterattack and
    #: lands at 378-411. `classifier.DUEL_DEADLINE` is the cut.
    duel_turn: int | None = None

    #: Stack size that counts as a fist. Kept here so `update` can stamp
    #: `duel_turn` as it happens; `classifier.DUEL_ARMY` is the same number
    #: and the tests pin them together.
    duel_army: int = 15

    my_general: tuple[int, int] | None = field(default=None, repr=False)
    _last_turn: int = field(default=-1, repr=False)

    def update(self, obs) -> None:
        """Fold this turn's observation in. Idempotent within a turn."""
        if obs.turn == self._last_turn:
            return
        self._last_turn = obs.turn

        if self.my_general is None:
            self.my_general = locate_own_general(obs)

        self.stack_near_now = 0
        enemy_cells = visible_enemy_tiles(obs)
        if not enemy_cells or self.my_general is None:
            return

        dist = multi_bfs(obs, [self.my_general])
        near = [
            obs.army_grid[r][c]
            for r, c in enemy_cells
            if dist[r][c] <= self.radius
        ]
        if not near:
            return

        self.turns_near += 1
        if self.first_near_turn is None:
            self.first_near_turn = obs.turn
        self.stack_near_now = max(near)
        if self.stack_near_now > self.max_stack_near:
            self.max_stack_near = self.stack_near_now
            if self.duel_turn is None and self.max_stack_near >= self.duel_army:
                self.duel_turn = obs.turn
