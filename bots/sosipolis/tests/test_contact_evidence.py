"""Fused contact evidence: later contacts must be able to move the hunt.

The scenes here are the shape that broke the single-anchor hunt: our general
in one corner, a first contact out toward the middle, and a second, thicker
contact region somewhere else. Nothing calls `state.update()` — memory is
seeded by hand so each field is under test on its own.
"""
from __future__ import annotations

from _imports import sosipolis_imports
from boards import seeded_contact_mcts


def _memory(H, W, *, home, first_contact=None, enemy=(), unseen=()):
    """MapMemory with hand-placed contact history and a scouted board."""
    from components.map_memory import MapMemory
    from params import PARAMS

    mem = MapMemory(H, W, PARAMS.MIN_GENERAL_DISTANCE)
    mem.own_general = home
    mem._seeded = True
    # Everything counts as scouted unless the scene says otherwise, so the
    # openness field only lights up where a test asks for it.
    for r in range(H):
        for c in range(W):
            mem.ever_seen[r][c] = True
    for cell in unseen:
        mem.ever_seen[cell[0]][cell[1]] = False
    if first_contact is not None:
        mem.first_contact = first_contact
        mem.first_contact_turn = 40
        mem.enemy_seen.add(first_contact)
        mem.primary_path.add(first_contact)
        mem.known_owner[first_contact[0]][first_contact[1]] = 2
    for cell in enemy:
        mem.enemy_seen.add(cell)
        mem.known_owner[cell[0]][cell[1]] = 2
    mem.enemy_obs_epoch += 1
    return mem


def _state_with(mem):
    from params import PARAMS
    from state import GameState

    st = GameState(mem.H, mem.W, PARAMS)
    st.memory = mem
    return st


def _block(r0, c0, r1, c1):
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


# ---------------------------------------------------------------------------
# window field
# ---------------------------------------------------------------------------


def test_window_field_counts_inside_the_square_and_clips_at_the_edge():
    with sosipolis_imports():
        from components.contact_evidence import _WindowField

        hot = {(0, 0), (1, 1), (4, 4)}
        field = _WindowField(6, 6, lambda r, c: (r, c) in hot)

        # Radius 1 around (1,1) covers rows/cols 0..2: two hot cells, 9 looked at.
        assert field.window(1, 1, 1) == (2, 9)
        # At the corner the window is clipped, not wrapped.
        assert field.window(0, 0, 1) == (2, 4)
        # Nothing hot out here.
        assert field.window(5, 0, 1) == (0, 4)


# ---------------------------------------------------------------------------
# density / openness
# ---------------------------------------------------------------------------


def test_density_is_higher_beside_thick_enemy_land_than_beside_a_snake():
    """A base region is thick; a raiding snake is one tile wide."""
    with sosipolis_imports():
        from components.contact_evidence import ContactEvidence
        from params import PARAMS

        mem = _memory(
            20, 20,
            home=(19, 0),
            first_contact=(8, 8),
            enemy=_block(8, 4, 8, 12) + _block(2, 15, 6, 19),
        )
        ev = ContactEvidence(PARAMS)
        ev.refresh(mem)

        thick = ev.density((4, 17))
        snake = ev.density((8, 8))
        assert thick > snake


def test_openness_marks_the_unscouted_neighbourhood():
    with sosipolis_imports():
        from components.contact_evidence import ContactEvidence
        from params import PARAMS

        mem = _memory(20, 20, home=(19, 0), unseen=_block(0, 0, 5, 5))
        ev = ContactEvidence(PARAMS)
        ev.refresh(mem)

        assert ev.openness((2, 2)) > ev.openness((15, 15))
        assert ev.openness((15, 15)) == 0.0


def test_refresh_rebuilds_only_when_what_we_know_changed():
    with sosipolis_imports():
        from components.contact_evidence import ContactEvidence
        from params import PARAMS

        mem = _memory(12, 12, home=(11, 0), enemy=[(4, 4)])
        ev = ContactEvidence(PARAMS)
        ev.refresh(mem)
        first = ev._enemy

        ev.refresh(mem)
        assert ev._enemy is first  # same epochs, no rebuild

        mem.enemy_seen.add((4, 5))
        mem.enemy_obs_epoch += 1
        ev.refresh(mem)
        assert ev._enemy is not first


# ---------------------------------------------------------------------------
# the anchor the whole hunt axis hangs off
# ---------------------------------------------------------------------------


def test_anchor_moves_to_a_later_thicker_contact_region():
    """The reported defect: contact 2 must be able to take over from contact 1.

    Home bottom-left, first contact out in the middle, then a thick region
    found later along the bottom-right. The anchor has to end up in the second
    region — that is the whole point of fusing contacts.
    """
    with sosipolis_imports():
        from components.contact_evidence import ContactEvidence
        from params import PARAMS

        mem = _memory(
            20, 20,
            home=(19, 0),
            first_contact=(8, 9),
            enemy=_block(14, 15, 18, 19),
        )
        ev = ContactEvidence(PARAMS)
        ev.refresh(mem)
        depth = mem._bfs_multi([mem.own_general])

        anchor = ev.anchor(mem, depth)
        assert anchor is not None
        assert anchor != mem.first_contact
        assert anchor[0] >= 14 and anchor[1] >= 15


def test_anchor_is_none_without_any_contact():
    with sosipolis_imports():
        from components.contact_evidence import ContactEvidence
        from params import PARAMS

        mem = _memory(12, 12, home=(11, 0))
        ev = ContactEvidence(PARAMS)
        ev.refresh(mem)

        assert ev.anchor(mem, mem._bfs_multi([(11, 0)])) is None


def test_hunt_anchor_falls_back_to_first_contact_alone():
    """One contact and nothing else: the fused anchor is that contact."""
    with sosipolis_imports():
        mem = _memory(20, 20, home=(19, 0), first_contact=(8, 9))
        state = _state_with(mem)
        mcts = seeded_contact_mcts()

        assert mcts._hunt_anchor(state) == (8, 9)


def test_axis_admits_the_later_contact_direction():
    """A bearing the first-contact axis rejects is on-axis once fused.

    Home `(19,0)`, first contact `(8,9)` — that line runs up and to the right,
    and `(18,18)` is far off it. The later contact region along the bottom
    right is what makes `(18,18)` a live direction.
    """
    with sosipolis_imports():
        target = (18, 18)

        lone = _state_with(_memory(20, 20, home=(19, 0), first_contact=(8, 9)))
        fused = _state_with(
            _memory(
                20, 20,
                home=(19, 0),
                first_contact=(8, 9),
                enemy=_block(14, 15, 18, 19),
            )
        )
        mcts = seeded_contact_mcts()

        assert mcts._on_hunt_axis(lone, target) is False
        assert mcts._far_off_axis(lone, target) is True

        mcts = seeded_contact_mcts()
        assert mcts._on_hunt_axis(fused, target) is True
        assert mcts._far_off_axis(fused, target) is False
