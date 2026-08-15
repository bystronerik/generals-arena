//! Per-game memory: the facts that hold for a whole game.
//!
//! **What lives here is decided by permanence, not by availability.** A fact
//! that a rule makes permanent is established once and read afterwards; it is
//! not re-derived from each frame, even when the frame happens to state it.
//! Both generals' cells are the clearest case — a general never relocates,
//! and the only event that changes its cell's owner is a capture, which ends
//! the game (RULES.md §05, §07) — so a general's position is settled the
//! first time it is seen and cannot go stale while there is still a turn to
//! play. Mountains are permanent by §01, first contact is an event that
//! cannot un-happen, and the last-seen garrison is the one *aging* fact here,
//! carried with the turn it was taken on so a bound can be aged honestly.
//!
//! What is deliberately **not** here: everything a turn can change — current
//! ownership, current armies, this turn's visibility, the opponent's army
//! total. Those are frame reads, and caching one would mean inventing an
//! invalidation rule. A stale value inside a proof search is not a slow path,
//! it is a wrong answer, so the bar for adding a field is that a rule in
//! RULES.md keeps it true.
//!
//! `AugState` (`board::obs`) already carries a mountain plane and a
//! last-seen-army plane for the network. This is a separate copy on purpose.
//! That one is padded to 21×21, folds in the pad-region rule (`augment_obs`
//! reads visible padding as mountain), and is pinned bit-for-bit against the
//! Python/XLA oracle — reading it here would tie a tactic to a tensor the
//! parity corpus owns, and any tactic-driven change to it would break that
//! corpus. This copy is board-shaped, integer, and answers one question per
//! field.

use crate::io::wire::{
    Observation, OWNER_ME, OWNER_OPP, TYPE_FOG, TYPE_GENERAL, TYPE_MOUNTAIN,
    TYPE_STRUCTURE_IN_FOG,
};

/// Is this cell in view this turn?
///
/// The frame encodes visibility in the type grid rather than in a mask: the
/// engine writes `TYPE_FOG` for an unseen cell and `TYPE_STRUCTURE_IN_FOG`
/// for an unseen mountain-or-castle, and masks every other plane by the same
/// visibility (`get_observation` in the engine's `game.py`). So a concrete
/// type — plain, mountain, castle, general — is exactly "we can see it", and
/// the two fog codes are exactly "we cannot".
pub fn is_visible(cell_type: i32) -> bool {
    cell_type != TYPE_FOG && cell_type != TYPE_STRUCTURE_IN_FOG
}

/// Everything the tactics layer remembers about one game. Board-shaped
/// (`h × w`, row-major), allocated once at the handshake.
#[derive(Debug, Clone)]
pub struct Memory {
    pub h: usize,
    pub w: usize,
    /// Our own general's cell, settled on the first frame. We own it and
    /// vision is a 3×3 pool around owned cells, so the first frame always
    /// carries it and it is never in doubt afterwards. `None` only before the
    /// first update, or on a frame so malformed that our own general was not
    /// in it — and in that second case the remembered cell is the right
    /// answer, not the missing one.
    pub own_general: Option<usize>,
    /// The enemy general's cell, once seen. Set on the first sighting and
    /// never moved afterwards, for the reason given in the module docs.
    /// `None` until that sighting, which is also the honest state: no kill is
    /// provable against an unlocated general.
    pub enemy_general: Option<usize>,
    /// Cells seen to be mountains at least once. Mountains are impassable and
    /// permanent (RULES.md §01), so a sighting is knowledge that never
    /// expires, and this is the only set of cells a tactic may soundly treat
    /// as blocked.
    pub mountains: Vec<bool>,
    /// The turn we first saw any enemy cell at all — first contact. `None`
    /// until then, and latched afterwards: it records that we have *met* the
    /// opponent, not that we can see them now. Losing sight of them again
    /// does not unmake the meeting.
    pub first_contact_turn: Option<i32>,
    /// Was the enemy general in view on the frame just processed?
    pub general_visible_now: bool,
    /// The general's army at the last sighting — a lower bound on its army
    /// now, since a general only ever grows between sightings (RULES.md §04)
    /// unless the opponent spends it. Meaningless while `enemy_general` is
    /// `None`.
    pub last_seen_general_army: i32,
    /// `obs.turn` of that sighting, so a bound can be aged. `-1` before the
    /// first one.
    pub last_seen_turn: i32,
}

impl Memory {
    pub fn new(h: usize, w: usize) -> Self {
        Self {
            h,
            w,
            own_general: None,
            enemy_general: None,
            mountains: vec![false; h * w],
            first_contact_turn: None,
            general_visible_now: false,
            last_seen_general_army: 0,
            last_seen_turn: -1,
        }
    }

    /// Fold one frame in. Called every turn, unconditionally, before anything
    /// decides anything — per-game state advances regardless of who chooses
    /// the move, exactly like `AugState`.
    pub fn update(&mut self, obs: &Observation) {
        debug_assert_eq!((self.h, self.w), (obs.h, obs.w));
        let mut saw_enemy = false;
        for i in 0..self.h * self.w {
            let cell_type = obs.type_grid[i];
            // `owner_grid` is 2 only where the opponent is *visible* — the
            // engine masks the opponent plane by visibility — so this is
            // exactly "we can see them right now".
            saw_enemy |= obs.owner_grid[i] == OWNER_OPP;
            // Only a *visible* mountain proves a mountain.
            // `TYPE_STRUCTURE_IN_FOG` means mountain-or-castle and cannot be
            // told apart (`structures_in_fog = invisible & (mountains |
            // castles)`), and a castle is passable and capturable — recording
            // one as blocked would prune real lines from a search that is
            // supposed to be exact.
            if cell_type == TYPE_MOUNTAIN {
                self.mountains[i] = true;
            }
            if cell_type == TYPE_GENERAL {
                match obs.owner_grid[i] {
                    OWNER_ME if self.own_general.is_none() => self.own_general = Some(i),
                    OWNER_OPP if self.enemy_general.is_none() => self.enemy_general = Some(i),
                    _ => {}
                }
            }
        }

        if saw_enemy && self.first_contact_turn.is_none() {
            self.first_contact_turn = Some(obs.turn);
        }

        self.general_visible_now = false;
        if let Some(cell) = self.enemy_general {
            if is_visible(obs.type_grid[cell]) {
                self.general_visible_now = true;
                self.last_seen_general_army = obs.army_grid[cell];
                self.last_seen_turn = obs.turn;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::wire::{OWNER_ME, TYPE_CASTLE, TYPE_PLAIN};

    /// An all-plain, all-neutral, all-visible frame.
    fn frame(h: usize, w: usize, turn: i32) -> Observation {
        let mut obs = Observation::with_dims(h, w);
        obs.turn = turn;
        obs.type_grid.iter_mut().for_each(|t| *t = TYPE_PLAIN);
        obs
    }

    fn at(obs: &Observation, r: usize, c: usize) -> usize {
        r * obs.w + c
    }

    fn put(obs: &mut Observation, r: usize, c: usize, cell_type: i32, owner: i32, army: i32) {
        let i = at(obs, r, c);
        obs.type_grid[i] = cell_type;
        obs.owner_grid[i] = owner;
        obs.army_grid[i] = army;
    }

    #[test]
    fn enemy_general_locks_in_on_the_first_sighting() {
        let mut mem = Memory::new(5, 5);
        let mut seen = frame(5, 5, 10);
        put(&mut seen, 1, 1, TYPE_GENERAL, OWNER_OPP, 20);
        mem.update(&seen);
        assert_eq!(mem.enemy_general, Some(at(&seen, 1, 1)));
        assert!(mem.general_visible_now);
        assert_eq!((mem.last_seen_general_army, mem.last_seen_turn), (20, 10));

        // It fogs over: the cell is remembered, the army bound freezes.
        let mut lost = frame(5, 5, 11);
        put(&mut lost, 1, 1, TYPE_FOG, 0, 0);
        mem.update(&lost);
        assert_eq!(mem.enemy_general, Some(at(&lost, 1, 1)));
        assert!(!mem.general_visible_now);
        assert_eq!((mem.last_seen_general_army, mem.last_seen_turn), (20, 10));

        // Seen again: the bound refreshes.
        let mut again = frame(5, 5, 12);
        put(&mut again, 1, 1, TYPE_GENERAL, OWNER_OPP, 25);
        mem.update(&again);
        assert!(mem.general_visible_now);
        assert_eq!((mem.last_seen_general_army, mem.last_seen_turn), (25, 12));
    }

    #[test]
    fn the_two_generals_never_change_places() {
        let mut mem = Memory::new(4, 4);
        let mut obs = frame(4, 4, 3);
        put(&mut obs, 0, 0, TYPE_GENERAL, OWNER_ME, 30);
        mem.update(&obs);
        assert_eq!(mem.own_general, Some(at(&obs, 0, 0)));
        assert_eq!(mem.enemy_general, None);
        assert!(!mem.general_visible_now);

        // Theirs turns up later and does not disturb ours.
        let mut both = frame(4, 4, 4);
        put(&mut both, 0, 0, TYPE_GENERAL, OWNER_ME, 31);
        put(&mut both, 3, 3, TYPE_GENERAL, OWNER_OPP, 12);
        mem.update(&both);
        assert_eq!(mem.own_general, Some(at(&both, 0, 0)));
        assert_eq!(mem.enemy_general, Some(at(&both, 3, 3)));
    }

    #[test]
    fn our_general_survives_a_frame_that_omits_it() {
        // Settled once, read afterwards: a frame we could not parse into a
        // general does not unlearn where ours is.
        let mut mem = Memory::new(4, 4);
        let mut obs = frame(4, 4, 1);
        put(&mut obs, 2, 1, TYPE_GENERAL, OWNER_ME, 8);
        mem.update(&obs);
        assert_eq!(mem.own_general, Some(at(&obs, 2, 1)));

        mem.update(&frame(4, 4, 2)); // no general anywhere on this frame
        assert_eq!(mem.own_general, Some(at(&obs, 2, 1)));
    }

    #[test]
    fn first_contact_latches_on_the_first_enemy_cell() {
        let mut mem = Memory::new(4, 4);
        let empty = frame(4, 4, 5);
        mem.update(&empty);
        assert_eq!(mem.first_contact_turn, None);

        let mut met = frame(4, 4, 6);
        put(&mut met, 2, 2, TYPE_PLAIN, OWNER_OPP, 4);
        mem.update(&met);
        assert_eq!(mem.first_contact_turn, Some(6));

        // Out of sight again is not out of memory, and a later sighting does
        // not move the recorded turn.
        let gone = frame(4, 4, 7);
        mem.update(&gone);
        assert_eq!(mem.first_contact_turn, Some(6));
        let mut again = frame(4, 4, 8);
        put(&mut again, 0, 0, TYPE_PLAIN, OWNER_OPP, 9);
        mem.update(&again);
        assert_eq!(mem.first_contact_turn, Some(6));
    }

    #[test]
    fn mountains_accumulate_and_never_expire() {
        let mut mem = Memory::new(3, 3);
        let mut obs = frame(3, 3, 1);
        put(&mut obs, 0, 2, TYPE_MOUNTAIN, 0, 0);
        mem.update(&obs);
        assert!(mem.mountains[2]);

        // The same cell in fog still reads as a mountain to us.
        let mut fogged = frame(3, 3, 2);
        put(&mut fogged, 0, 2, TYPE_STRUCTURE_IN_FOG, 0, 0);
        mem.update(&fogged);
        assert!(mem.mountains[2]);
        // And nothing else ever became one.
        assert_eq!(mem.mountains.iter().filter(|m| **m).count(), 1);
    }

    #[test]
    fn a_structure_in_fog_is_not_a_remembered_mountain() {
        // Type 5 is "mountain or castle, in fog" and the two cannot be told
        // apart. A castle is passable, so this cell stays open.
        let mut mem = Memory::new(3, 3);
        let mut obs = frame(3, 3, 1);
        put(&mut obs, 1, 1, TYPE_STRUCTURE_IN_FOG, 0, 0);
        mem.update(&obs);
        assert!(!mem.mountains[at(&obs, 1, 1)]);

        // A castle that comes into view confirms it, and still blocks nothing.
        let mut lit = frame(3, 3, 2);
        put(&mut lit, 1, 1, TYPE_CASTLE, OWNER_OPP, 7);
        mem.update(&lit);
        assert!(!mem.mountains[at(&lit, 1, 1)]);
    }
}
