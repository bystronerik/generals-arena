//! The local forward model — the engine's own resolution, run on our own
//! board.
//!
//! This is the one module in the crate that simulates the game. `board::obs`
//! says joe never does, and that stays true of joe's pipeline; a proof needs a
//! transition function, so the fork adds one here and keeps it out of the
//! network's way. Everything it implements is transcribed from the engine
//! rather than reasoned about: `core/game.py` for moves, combat, order and
//! growth, `modifiers/deathtouch.py` for the endgame and for the mutual
//! capture that is a draw at every turn. Where the transcription had a
//! choice, the comment says which line it came from.
//!
//! # What "exact" covers, and what it cannot
//!
//! Exact: the §02 priority ladder including its seat-order tiebreak, §05
//! combat with the leave-one-behind rule and the tie that keeps the defender,
//! §04 growth on both clocks with the engine's phase, §07 deathtouch with the
//! chase defense, and the draw a simultaneous decapitation makes.
//!
//! Not exact, and named because a proof that hides an assumption is worth
//! less than a decline:
//!
//! * **Fog.** Cells we cannot see get [`Fog::Pessimistic`] — enemy-owned,
//!   holding the whole hidden-army bound, every one of them independently.
//!   That is a superset of the real board, so a goal proven against it holds
//!   on the real one; it is also why deep proofs stop landing, which is the
//!   knowledge limit the plan already expects.
//! * **Opponent builds** are modeled as a pass. A build is a pass that also
//!   spends 35+ army off one of their own cells (RULES.md §03), so it is
//!   dominated by a pass in every line these searches care about, with one
//!   exception this model drops: the new castle's own production, at most one
//!   army inside a three-ply horizon.
//! * **Structures under fog.** A fogged cell could be a castle producing for
//!   them. Rather than guess, the caller inflates the hidden bound
//!   ([`Config::hidden_bound`]) by the most that production can add over the
//!   horizon.
//!
//! # Passability is asymmetric on purpose
//!
//! `TYPE_FOG` is *known* to be neither mountain nor castle — the engine
//! writes `fog_cells = invisible & ~(mountains | castles)` — so it is a plain
//! cell we cannot see into. `TYPE_STRUCTURE_IN_FOG` is the one that cannot be
//! told apart: mountain or castle, blocked or walkable. So it is closed to us
//! (a move the engine would void must not appear inside a proof) and open to
//! them (a route we cannot rule out must not be pruned from their replies).
//!
//! # Make/unmake, not clone
//!
//! One tick touches four cells plus whatever grows, so the search advances one
//! `Sim` through an undo journal instead of copying a board per node. A
//! 21×21 clone per node would cost more than the search it serves.

use crate::board::memory::{is_visible, Memory};
use crate::board::obs::DIRECTIONS;
use crate::io::wire::{
    Observation, OWNER_ME, OWNER_OPP, TYPE_CASTLE, TYPE_FOG, TYPE_GENERAL, TYPE_MOUNTAIN,
};

/// Cell owners. The two player codes are the wire's own (`OWNER_ME`,
/// `OWNER_OPP`), so a visible cell's owner copies straight across.
pub const NEUTRAL: u8 = 0;
pub const ME: u8 = OWNER_ME as u8;
pub const OPP: u8 = OWNER_OPP as u8;

/// One player's move: a source cell, a direction into
/// [`DIRECTIONS`](crate::board::obs::DIRECTIONS), and the split flag (half the
/// army, rather than all but one).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Move {
    pub from: usize,
    pub dir: u8,
    pub half: bool,
}

/// What a player does with a tick. `None` is a pass — and also what an
/// opponent build looks like to this model (see the module docs).
pub type Choice = Option<Move>;

/// How a position stands. `Draw` is a mutual decapitation, which the engine
/// treats as a draw on every turn and not only past the deathtouch threshold;
/// both searches count it as failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Ongoing,
    MeWins,
    OppWins,
    Draw,
}

/// What to do with the cells we cannot see.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fog {
    /// Every fogged cell is enemy-owned and holds the whole hidden bound. The
    /// proof device: it can only make a goal harder to prove.
    Pessimistic,
    /// Fogged cells are empty ground. Optimistic, and therefore useless for
    /// proving anything — its one job is
    /// [`refutes`](crate::search::minimax::refutes), which needs the opposite
    /// guarantee: every reply it finds is a reply the opponent really has.
    VisibleOnly,
}

/// What the frame alone does not say. The caller owns these numbers because
/// the tactics layer owns the rules that produce them.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub fog: Fog,
    /// Upper bound on the enemy army that may sit on any one fogged cell.
    pub hidden_bound: i32,
    /// The turn deathtouch goes live. Passed in rather than read from a
    /// constant here: the tactics layer owns the constants block, and this
    /// module sits below it.
    pub deathtouch_turn: i32,
    /// Are we seat 0? The §02 ladder's last tiebreak is the seat index, so an
    /// exactly equal clash resolves differently for the two seats.
    pub i_am_p0: bool,
}

/// One undo frame: where the journal stood before a tick, plus the scalars
/// that tick could change.
#[derive(Debug, Clone, Copy)]
struct Frame {
    journal: usize,
    turn: i32,
    outcome: Outcome,
    winner: Option<u8>,
}

/// A board that can be advanced and taken back.
#[derive(Debug, Clone)]
pub struct Sim {
    pub h: usize,
    pub w: usize,
    pub turn: i32,
    pub deathtouch_turn: i32,
    pub i_am_p0: bool,
    /// Our general's cell, from memory: known for the whole game.
    pub own_general: usize,
    /// Theirs, once seen. `None` keeps every kill unprovable, which is the
    /// honest answer.
    pub enemy_general: Option<usize>,
    pub(super) owner: Vec<u8>,
    pub(super) army: Vec<i32>,
    /// Generals and castles: the cells that produce on the §04 two-turn clock.
    /// Ownership is read live, so a castle that changes hands starts producing
    /// for its captor.
    pub(super) structures: Vec<usize>,
    pub(super) passable_me: Vec<bool>,
    pub(super) passable_opp: Vec<bool>,
    outcome: Outcome,
    /// The engine's own `state.winner`, which a second capture in the same
    /// tick overwrites. [`Outcome`] is what the search reads; this is kept
    /// because the deathtouch modifier's draw test compares the winner before
    /// and after the second move.
    winner: Option<u8>,
    journal: Vec<(usize, u8, i32)>,
    frames: Vec<Frame>,
}

impl Sim {
    /// An empty board: nothing owned, nothing passable, no general, and
    /// deathtouch off. Every caller fills in what it knows — `from_frame`
    /// from a frame, the test fixtures cell by cell.
    pub(super) fn blank(h: usize, w: usize) -> Sim {
        let cells = h * w;
        Sim {
            h,
            w,
            turn: 0,
            // Never, until a caller says when. The threshold is the tactics
            // layer's constant, and this module sits below it.
            deathtouch_turn: i32::MAX,
            i_am_p0: true,
            own_general: usize::MAX,
            enemy_general: None,
            owner: vec![NEUTRAL; cells],
            army: vec![0; cells],
            structures: Vec::new(),
            passable_me: vec![false; cells],
            passable_opp: vec![false; cells],
            outcome: Outcome::Ongoing,
            winner: None,
            journal: Vec::new(),
            frames: Vec::new(),
        }
    }

    /// Build the model from one frame plus what we remember. `None` when our
    /// own general is not known yet, which is only ever the very first frame.
    pub fn from_frame(obs: &Observation, mem: &Memory, cfg: &Config) -> Option<Sim> {
        let own_general = mem.own_general?;
        let cells = obs.h * obs.w;
        let mut sim = Sim {
            turn: obs.turn,
            deathtouch_turn: cfg.deathtouch_turn,
            i_am_p0: cfg.i_am_p0,
            own_general,
            enemy_general: mem.enemy_general,
            ..Sim::blank(obs.h, obs.w)
        };

        for cell in 0..cells {
            let cell_type = obs.type_grid[cell];
            // A wall stays a wall for both sides: `mem.mountains` only records
            // cells seen to be mountains, so this never blocks a castle.
            if cell_type == TYPE_MOUNTAIN || mem.mountains[cell] {
                continue;
            }
            if is_visible(cell_type) {
                sim.passable_me[cell] = true;
                sim.passable_opp[cell] = true;
                sim.owner[cell] = obs.owner_grid[cell] as u8;
                sim.army[cell] = obs.army_grid[cell];
                if cell_type == TYPE_GENERAL || cell_type == TYPE_CASTLE {
                    sim.structures.push(cell);
                }
            } else {
                sim.passable_me[cell] = cell_type == TYPE_FOG;
                sim.passable_opp[cell] = true;
                if cfg.fog == Fog::Pessimistic {
                    sim.owner[cell] = OPP;
                    sim.army[cell] = cfg.hidden_bound;
                }
            }
        }

        // A general in fog is still a general: it produces, and it is the cell
        // a kill has to land on. Under `VisibleOnly` it is left as plain
        // ground, because that model exists to enumerate the opponent's real
        // replies and never to decide whether we won.
        if cfg.fog == Fog::Pessimistic {
            if let Some(enemy) = mem.enemy_general {
                if !is_visible(obs.type_grid[enemy]) {
                    sim.structures.push(enemy);
                }
            }
        }
        Some(sim)
    }

    #[inline]
    pub fn owner_at(&self, cell: usize) -> u8 {
        self.owner[cell]
    }

    #[inline]
    pub fn army_at(&self, cell: usize) -> i32 {
        self.army[cell]
    }

    #[inline]
    pub fn outcome(&self) -> Outcome {
        self.outcome
    }

    /// Has every advance been taken back? One board serves both searches on a
    /// turn, and that is only safe while this holds.
    #[inline]
    pub fn is_root(&self) -> bool {
        self.frames.is_empty()
    }

    /// Where a move from `cell` in `dir` lands, or `None` off the board.
    #[inline]
    pub fn dest(&self, cell: usize, dir: u8) -> Option<usize> {
        let (dr, dc) = DIRECTIONS[dir as usize];
        let row = (cell / self.w) as i32 + dr;
        let col = (cell % self.w) as i32 + dc;
        if row < 0 || col < 0 || row >= self.h as i32 || col >= self.w as i32 {
            return None;
        }
        Some(row as usize * self.w + col as usize)
    }

    /// May `player` walk onto this cell? Asymmetric — see the module docs.
    #[inline]
    pub fn passable_for(&self, player: u8, cell: usize) -> bool {
        if player == ME {
            self.passable_me[cell]
        } else {
            self.passable_opp[cell]
        }
    }

    /// Advance one tick with both players' actions, recording an undo frame.
    pub fn push(&mut self, mine: Choice, theirs: Choice) {
        self.frames.push(Frame {
            journal: self.journal.len(),
            turn: self.turn,
            outcome: self.outcome,
            winner: self.winner,
        });
        self.resolve(mine, theirs);
    }

    /// Take the last tick back, exactly.
    pub fn pop(&mut self) {
        let frame = self.frames.pop().expect("pop without a matching push");
        while self.journal.len() > frame.journal {
            let (cell, owner, army) = self.journal.pop().expect("journal shorter than its frame");
            self.owner[cell] = owner;
            self.army[cell] = army;
        }
        self.turn = frame.turn;
        self.outcome = frame.outcome;
        self.winner = frame.winner;
    }

    /// One tick: builds are already passes here, then the two moves in the
    /// engine's order, then the clock, then growth. Mirrors
    /// `deathtouch.step`, which wraps `game.step`.
    fn resolve(&mut self, mine: Choice, theirs: Choice) {
        debug_assert_eq!(self.outcome, Outcome::Ongoing);
        // `deathtouch.step` reads the threshold against the state *before* the
        // tick, which is the same clock `obs.turn` carries.
        let deathtouch = self.turn >= self.deathtouch_turn;

        let opp_first = self.opp_moves_first(mine, theirs);
        let (first, first_choice, second, second_choice) = if opp_first {
            (OPP, theirs, ME, mine)
        } else {
            (ME, mine, OPP, theirs)
        };

        // The touch test runs against the state each move actually sees: the
        // first mover against the position as it stands, the second against
        // the position the first one left. That is what gives the chase its
        // teeth — strip the source and the touch never executes.
        let touch_first = self.executes_onto_general(first, first_choice);
        self.apply(first, first_choice);
        let winner_after_first = self.winner;
        let touch_second = self.executes_onto_general(second, second_choice);
        self.apply(second, second_choice);

        // `_apply_move` is the only writer of `winner` and only a capture
        // fires it, so a winner that changed under the second move means both
        // generals fell in this tick.
        let both_captured = winner_after_first.is_some() && self.winner != winner_after_first;
        let (touch_me, touch_opp) = if first == ME {
            (touch_first && deathtouch, touch_second && deathtouch)
        } else {
            (touch_second && deathtouch, touch_first && deathtouch)
        };

        self.outcome = if (touch_me && touch_opp) || both_captured {
            Outcome::Draw
        } else if touch_me != touch_opp {
            // A lone touch can only agree with a winner the base step already
            // set, or settle one it could not.
            if touch_me {
                Outcome::MeWins
            } else {
                Outcome::OppWins
            }
        } else {
            match self.winner {
                Some(ME) => Outcome::MeWins,
                Some(_) => Outcome::OppWins,
                None => Outcome::Ongoing,
            }
        };

        self.turn += 1;
        // `game.step` grows the board only while nobody has won; a finished
        // game transfers the spoils instead, and no search reads them.
        if self.outcome == Outcome::Ongoing {
            self.grow();
        }
    }

    /// RULES.md §04 on the engine's phase: the clock has already ticked, every
    /// owned cell gains one on a 50, and every owned structure gains one on an
    /// even turn (`global_update`).
    fn grow(&mut self) {
        if self.turn % 50 == 0 {
            for cell in 0..self.owner.len() {
                if self.owner[cell] != NEUTRAL {
                    self.record(cell);
                    self.army[cell] += 1;
                }
            }
        }
        if self.turn % 2 == 0 {
            for idx in 0..self.structures.len() {
                let cell = self.structures[idx];
                if self.owner[cell] != NEUTRAL {
                    self.record(cell);
                    self.army[cell] += 1;
                }
            }
        }
    }

    /// Which action resolves first (RULES.md §02): chasing, then reinforcing,
    /// then the smaller army, then the seat index.
    ///
    /// When either player passes the question is empty — a pass changes
    /// nothing, so the two orders leave the same board and neither can touch a
    /// general. The engine still computes an order there, from the coordinates
    /// a pass carries; reproducing that arithmetic would prove nothing.
    fn opp_moves_first(&self, mine: Choice, theirs: Choice) -> bool {
        let (Some(my_move), Some(their_move)) = (mine, theirs) else {
            return false;
        };
        let my_dest = self.dest(my_move.from, my_move.dir);
        let their_dest = self.dest(their_move.from, their_move.dir);

        let my_chase = my_dest == Some(their_move.from);
        let their_chase = their_dest == Some(my_move.from);
        if my_chase != their_chase {
            return their_chase;
        }

        let my_reinforce = my_dest.is_some_and(|cell| self.owner[cell] == ME);
        let their_reinforce = their_dest.is_some_and(|cell| self.owner[cell] == OPP);
        if my_reinforce != their_reinforce {
            return their_reinforce;
        }

        let my_army = self.army[my_move.from];
        let their_army = self.army[their_move.from];
        if my_army != their_army {
            return their_army < my_army;
        }
        // Dead level: `_determine_move_order` leaves player 0 first.
        !self.i_am_p0
    }

    /// Is this a valid move whose destination is the other player's general
    /// tile (`deathtouch._executes_onto_general`)? A pass never touches, and
    /// neither does a build — both arrive here as `None`.
    fn executes_onto_general(&self, player: u8, choice: Choice) -> bool {
        let Some(mv) = choice else {
            return false;
        };
        let Some((_, dest)) = self.legal(player, mv) else {
            return false;
        };
        let their_general = if player == ME {
            self.enemy_general
        } else {
            Some(self.own_general)
        };
        their_general == Some(dest)
    }

    /// The engine's validity test and the army it would send
    /// (`game._execute_move`), or `None` when the move is a silent pass.
    fn legal(&self, player: u8, mv: Move) -> Option<(i32, usize)> {
        if self.owner[mv.from] != player {
            return None;
        }
        let dest = self.dest(mv.from, mv.dir)?;
        if !self.passable_for(player, dest) {
            return None;
        }
        let source_army = self.army[mv.from];
        let sending = if mv.half {
            source_army / 2
        } else {
            source_army - 1
        };
        let sending = sending.clamp(0, (source_army - 1).max(0));
        if sending <= 0 {
            return None;
        }
        Some((sending, dest))
    }

    /// Execute one player's action (`game._apply_move`). Runs even when the
    /// other player has just taken a general — the engine has no such guard,
    /// and that omission is what makes a mutual capture possible.
    fn apply(&mut self, player: u8, choice: Choice) {
        let Some(mv) = choice else {
            return;
        };
        let Some((sending, dest)) = self.legal(player, mv) else {
            return;
        };

        self.record(mv.from);
        self.army[mv.from] -= sending;

        self.record(dest);
        if self.owner[dest] == player {
            // Own cell: armies merge, no combat.
            self.army[dest] += sending;
            return;
        }

        let defending = self.army[dest];
        let captured = sending > defending;
        self.army[dest] = (defending - sending).abs();
        if captured {
            self.owner[dest] = player;
            // The generals mask is static in the engine, so the cell of a
            // fallen general is still a general cell.
            let is_general = dest == self.own_general || Some(dest) == self.enemy_general;
            if is_general {
                self.winner = Some(player);
            }
        }
    }

    /// Journal a cell before it changes.
    #[inline]
    fn record(&mut self, cell: usize) {
        if !self.frames.is_empty() {
            self.journal.push((cell, self.owner[cell], self.army[cell]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::wire::{TYPE_PLAIN, TYPE_STRUCTURE_IN_FOG};
    use crate::search::fixtures::{at, board, Setup};

    /// An all-plain, all-visible, all-neutral frame.
    fn wire_frame(h: usize, w: usize, turn: i32) -> Observation {
        let mut obs = Observation::with_dims(h, w);
        obs.turn = turn;
        obs.type_grid.iter_mut().for_each(|t| *t = TYPE_PLAIN);
        obs
    }

    fn set(obs: &mut Observation, row: usize, col: usize, cell_type: i32, owner: i32, army: i32) {
        let cell = row * obs.w + col;
        obs.type_grid[cell] = cell_type;
        obs.owner_grid[cell] = owner;
        obs.army_grid[cell] = army;
    }

    fn remembering(obs: &Observation) -> Memory {
        let mut mem = Memory::new(obs.h, obs.w);
        mem.update(obs);
        mem
    }

    const CFG: Config =
        Config { fog: Fog::Pessimistic, hidden_bound: 17, deathtouch_turn: 800, i_am_p0: true };

    /// The pessimistic board: every cell we cannot see is theirs and holds the
    /// whole bound, and the two passability masks part company on the one cell
    /// type the frame cannot resolve.
    #[test]
    fn fog_materializes_as_the_worst_enemy_it_could_hold() {
        let (h, w) = (4usize, 4usize);
        let mut obs = wire_frame(h, w, 20);
        set(&mut obs, 3, 3, TYPE_GENERAL, OWNER_ME, 9);
        set(&mut obs, 0, 0, TYPE_FOG, 0, 0);
        set(&mut obs, 0, 1, TYPE_STRUCTURE_IN_FOG, 0, 0);
        set(&mut obs, 1, 1, TYPE_MOUNTAIN, 0, 0);
        set(&mut obs, 2, 2, TYPE_PLAIN, OWNER_OPP, 4);
        let mem = remembering(&obs);

        let sim = Sim::from_frame(&obs, &mem, &CFG).expect("our general is on the frame");
        assert_eq!(sim.own_general, at(w, 3, 3));
        assert_eq!((sim.owner_at(at(w, 0, 0)), sim.army_at(at(w, 0, 0))), (OPP, 17));
        assert!(sim.passable_for(ME, at(w, 0, 0)), "plain fog is known to be plain");
        assert!(!sim.passable_for(ME, at(w, 0, 1)), "mountain or castle: we do not risk it");
        assert!(sim.passable_for(OPP, at(w, 0, 1)), "and they may walk it");
        assert!(!sim.passable_for(ME, at(w, 1, 1)) && !sim.passable_for(OPP, at(w, 1, 1)));
        assert_eq!((sim.owner_at(at(w, 2, 2)), sim.army_at(at(w, 2, 2))), (OPP, 4));

        // The other model leaves the same fog as empty ground.
        let seen = Sim::from_frame(&obs, &mem, &Config { fog: Fog::VisibleOnly, ..CFG }).unwrap();
        assert_eq!((seen.owner_at(at(w, 0, 0)), seen.army_at(at(w, 0, 0))), (NEUTRAL, 0));
        assert_eq!((seen.owner_at(at(w, 2, 2)), seen.army_at(at(w, 2, 2))), (OPP, 4));
    }

    /// A general we have seen and lost sight of is still a general: it keeps
    /// producing on the §04 clock, on top of the hidden bound it is credited.
    #[test]
    fn a_remembered_general_in_fog_still_produces() {
        let (h, w) = (4usize, 4usize);
        let mut seen = wire_frame(h, w, 40);
        set(&mut seen, 3, 3, TYPE_GENERAL, OWNER_ME, 9);
        set(&mut seen, 0, 0, TYPE_GENERAL, OWNER_OPP, 12);
        let mut mem = remembering(&seen);

        let mut hidden = wire_frame(h, w, 41);
        set(&mut hidden, 3, 3, TYPE_GENERAL, OWNER_ME, 9);
        set(&mut hidden, 0, 0, TYPE_FOG, 0, 0);
        mem.update(&hidden);

        let mut sim = Sim::from_frame(&hidden, &mem, &CFG).unwrap();
        assert_eq!(sim.enemy_general, Some(at(w, 0, 0)));
        assert_eq!(sim.army_at(at(w, 0, 0)), 17, "the bound, not the last sighting");
        sim.push(None, None); // the tick lands on 42
        assert_eq!(sim.army_at(at(w, 0, 0)), 18);
    }

    /// Before our own general is on any frame there is nothing to reason
    /// about, and the model says so instead of guessing.
    #[test]
    fn without_our_general_there_is_no_model() {
        let obs = wire_frame(4, 4, 0);
        assert!(Sim::from_frame(&obs, &remembering(&obs), &CFG).is_none());
    }

    /// A full move from a cell with `army` sends `army - 1`; the source keeps
    /// one behind (RULES.md §02).
    #[test]
    fn a_move_leaves_one_behind_and_a_half_move_splits() {
        let mut sim = board(3, 3, |b: &mut Setup| {
            b.mine(1, 1, 9);
            b.own_general(2, 2, 5);
        });
        sim.push(Some(Move { from: at(3, 1, 1), dir: 3, half: false }), None);
        assert_eq!(sim.army_at(at(3, 1, 1)), 1);
        assert_eq!(sim.army_at(at(3, 1, 2)), 8);
        assert_eq!(sim.owner_at(at(3, 1, 2)), ME);
        sim.pop();

        sim.push(Some(Move { from: at(3, 1, 1), dir: 3, half: true }), None);
        assert_eq!(sim.army_at(at(3, 1, 1)), 5);
        assert_eq!(sim.army_at(at(3, 1, 2)), 4);
    }

    #[test]
    fn undo_restores_the_board_exactly() {
        let mut sim = board(4, 4, |b: &mut Setup| {
            b.mine(1, 1, 9);
            b.opp(1, 2, 4);
            b.own_general(3, 3, 5);
            b.enemy_general(0, 0, 5);
        });
        let before = (sim.owner.clone(), sim.army.clone(), sim.turn, sim.outcome());
        sim.push(Some(Move { from: at(4, 1, 1), dir: 3, half: false }), None);
        assert_ne!(sim.army, before.1);
        sim.pop();
        assert_eq!((sim.owner.clone(), sim.army.clone(), sim.turn, sim.outcome()), before);
    }

    /// §05: strictly more army takes the cell; an exact tie keeps the
    /// defender, on zero army.
    #[test]
    fn combat_needs_strictly_more_and_a_tie_keeps_the_defender() {
        let mut sim = board(3, 3, |b: &mut Setup| {
            b.mine(0, 0, 6);
            b.opp(0, 1, 5);
            b.own_general(2, 2, 5);
        });
        // Sends 5 against 5: the defender holds the cell with nothing on it.
        sim.push(Some(Move { from: at(3, 0, 0), dir: 3, half: false }), None);
        assert_eq!(sim.owner_at(at(3, 0, 1)), OPP);
        assert_eq!(sim.army_at(at(3, 0, 1)), 0);
        sim.pop();

        sim.army[at(3, 0, 0)] = 7; // sends 6 against 5
        sim.push(Some(Move { from: at(3, 0, 0), dir: 3, half: false }), None);
        assert_eq!(sim.owner_at(at(3, 0, 1)), ME);
        assert_eq!(sim.army_at(at(3, 0, 1)), 1);
    }

    #[test]
    fn an_invalid_move_is_a_silent_pass() {
        let mut sim = board(3, 3, |b: &mut Setup| {
            b.mine(1, 1, 1); // one army cannot move
            b.mine(0, 0, 5);
            b.mountain(0, 1);
            b.own_general(2, 2, 5);
        });
        for mv in [
            Move { from: at(3, 1, 1), dir: 3, half: false }, // nothing to send
            Move { from: at(3, 0, 0), dir: 3, half: false }, // into a mountain
            Move { from: at(3, 0, 0), dir: 0, half: false }, // off the board
            Move { from: at(3, 2, 0), dir: 1, half: false }, // not our cell
        ] {
            let before = sim.army.clone();
            sim.push(Some(mv), None);
            assert_eq!(sim.army, before, "{mv:?} changed the board");
            sim.pop();
        }
    }

    /// §02, rung 1: chasing goes first. Their stack of 9 attacks our general;
    /// we chase its source from a third cell and take it, so their move finds
    /// nothing to send.
    #[test]
    fn chasing_resolves_first() {
        let mut sim = board(3, 3, |b: &mut Setup| {
            b.own_general(1, 1, 2);
            b.opp(1, 2, 9);
            b.mine(0, 2, 12);
        });
        let chase = Move { from: at(3, 0, 2), dir: 1, half: false }; // onto their source
        let attack = Move { from: at(3, 1, 2), dir: 2, half: false }; // onto our general
        sim.push(Some(chase), Some(attack));
        assert_eq!(sim.outcome(), Outcome::Ongoing);
        assert_eq!(sim.owner_at(at(3, 1, 1)), ME, "our general held");
        assert_eq!(sim.owner_at(at(3, 1, 2)), ME, "we took their source");
    }

    /// §02, rung 2: with neither move a chase, the one landing on its own
    /// cell resolves first. Ours reinforces, so our 5 is on the contested
    /// cell before their 4 arrives.
    #[test]
    fn reinforcing_resolves_before_a_plain_move() {
        let mut sim = board(3, 3, |b: &mut Setup| {
            b.own_general(0, 0, 6);
            b.mine(0, 1, 1);
            b.opp(1, 1, 5);
            b.enemy_general(2, 2, 5);
        });
        let reinforce = Move { from: at(3, 0, 0), dir: 3, half: false }; // 5 onto our own (0,1)
        let attack = Move { from: at(3, 1, 1), dir: 0, half: false }; // 4 onto (0,1)
        sim.push(Some(reinforce), Some(attack));
        assert_eq!(sim.owner_at(at(3, 0, 1)), ME);
        assert_eq!(sim.army_at(at(3, 0, 1)), 2, "6 held, 4 spent");
    }

    /// §02, rung 3: otherwise the smaller army moves first — so the bigger
    /// force resolves last and ends up holding the contested cell.
    #[test]
    fn the_smaller_army_resolves_first() {
        let mut sim = board(3, 3, |b: &mut Setup| {
            b.own_general(0, 0, 5);
            b.mine(2, 0, 11);
            b.opp(2, 2, 4);
            b.enemy_general(0, 2, 5);
        });
        let ours = Move { from: at(3, 2, 0), dir: 3, half: false }; // 10 onto (2,1)
        let theirs = Move { from: at(3, 2, 2), dir: 2, half: false }; // 3 onto (2,1)
        sim.push(Some(ours), Some(theirs));
        assert_eq!(sim.owner_at(at(3, 2, 1)), ME);
        assert_eq!(sim.army_at(at(3, 2, 1)), 7, "10 arrived after their 3");
    }

    /// The last rung is the seat index, and it is the only place the two seats
    /// differ: equal armies, no chase, no reinforce, player 0 first.
    #[test]
    fn an_exact_tie_falls_to_the_seat_order() {
        let contested = |i_am_p0: bool| {
            let mut sim = board(3, 3, |b: &mut Setup| {
                b.own_general(0, 0, 5);
                b.mine(2, 0, 6);
                b.opp(2, 2, 6);
                b.enemy_general(0, 2, 5);
            });
            sim.i_am_p0 = i_am_p0;
            sim.push(
                Some(Move { from: at(3, 2, 0), dir: 3, half: false }),
                Some(Move { from: at(3, 2, 2), dir: 2, half: false }),
            );
            sim.owner_at(at(3, 2, 1))
        };
        // Whoever moves first holds the cell: their 5 lands on it, and the 5
        // that follows ties, which keeps the defender (§05).
        assert_eq!(contested(true), ME);
        assert_eq!(contested(false), OPP);
    }

    /// §04: structures grow on even turns, every cell on a 50, and the clock
    /// has already ticked when growth applies.
    #[test]
    fn growth_follows_the_turn_parity_after_the_tick() {
        let mut sim = board(3, 3, |b: &mut Setup| {
            b.own_general(0, 0, 5);
            b.mine(2, 2, 3);
            b.enemy_general(0, 2, 7);
        });
        sim.turn = 41; // the tick lands on 42
        sim.push(None, None);
        assert_eq!(sim.army_at(at(3, 0, 0)), 6, "our general on an even turn");
        assert_eq!(sim.army_at(at(3, 0, 2)), 8, "theirs too");
        assert_eq!(sim.army_at(at(3, 2, 2)), 3, "a plain cell does not");
        sim.pop();

        sim.turn = 42; // the tick lands on 43
        sim.push(None, None);
        assert_eq!(sim.army_at(at(3, 0, 0)), 5, "odd turn, no production");
        sim.pop();

        sim.turn = 49; // the tick lands on 50: both clocks
        sim.push(None, None);
        assert_eq!(sim.army_at(at(3, 0, 0)), 7, "general: 1 for the 50, 1 for the parity");
        assert_eq!(sim.army_at(at(3, 2, 2)), 4, "a plain cell gets the 50");
    }

    /// A captured castle produces for its captor from then on.
    #[test]
    fn a_captured_castle_produces_for_its_captor() {
        let mut sim = board(3, 3, |b: &mut Setup| {
            b.own_general(0, 0, 5);
            b.mine(1, 0, 9);
            b.castle(1, 1, OPP, 3);
            b.enemy_general(2, 2, 5);
        });
        sim.turn = 41;
        sim.push(Some(Move { from: at(3, 1, 0), dir: 3, half: false }), None);
        assert_eq!(sim.owner_at(at(3, 1, 1)), ME);
        assert_eq!(sim.army_at(at(3, 1, 1)), 6, "5 held after taking 3, then +1");
    }

    /// §07: from the threshold, any move that executes onto the general wins,
    /// however large the garrison.
    #[test]
    fn deathtouch_beats_any_garrison_from_its_turn() {
        let position = |turn: i32| {
            let mut sim = board(3, 3, |b: &mut Setup| {
                b.own_general(0, 0, 5);
                b.mine(2, 1, 2);
                b.enemy_general(2, 2, 400);
            });
            sim.turn = turn;
            sim.push(Some(Move { from: at(3, 2, 1), dir: 3, half: false }), None);
            sim.outcome()
        };
        assert_eq!(position(799), Outcome::Ongoing);
        assert_eq!(position(800), Outcome::MeWins);
    }

    /// §07's defense: capture the touch's source from a third tile and the
    /// touch never executes. Leave the source able to move and it does.
    #[test]
    fn the_chase_defense_stops_a_touch_only_by_taking_the_source() {
        let defense = |our_army: i32| {
            let mut sim = board(3, 3, |b: &mut Setup| {
                b.own_general(0, 0, 5);
                b.mine(2, 0, our_army);
                b.opp(1, 0, 4);
                b.enemy_general(2, 2, 5);
            });
            sim.turn = 800;
            sim.push(
                Some(Move { from: at(3, 2, 0), dir: 0, half: false }), // chase their source
                Some(Move { from: at(3, 1, 0), dir: 0, half: false }), // touch our general
            );
            sim.outcome()
        };
        assert_eq!(defense(6), Outcome::Ongoing, "5 takes their 4: no source left");
        assert_eq!(defense(3), Outcome::OppWins, "2 fails, and their 2 still moves");
    }

    /// Both generals in one tick is a draw, and it is a draw at every turn:
    /// the engine detects the army-based double capture without a threshold.
    #[test]
    fn a_mutual_capture_is_a_draw_at_any_turn() {
        let mut sim = board(3, 3, |b: &mut Setup| {
            b.own_general(0, 0, 1);
            b.mine(0, 2, 9);
            b.opp(1, 0, 9);
            b.enemy_general(0, 1, 1);
        });
        sim.turn = 10;
        sim.push(
            Some(Move { from: at(3, 0, 2), dir: 2, half: false }), // onto their general
            Some(Move { from: at(3, 1, 0), dir: 0, half: false }), // onto ours
        );
        assert_eq!(sim.outcome(), Outcome::Draw);
    }

    /// Two touches in one tick is the same answer, by the other code path.
    #[test]
    fn a_mutual_touch_is_a_draw() {
        let mut sim = board(3, 3, |b: &mut Setup| {
            b.own_general(0, 0, 300);
            b.mine(0, 2, 2);
            b.opp(1, 0, 2);
            b.enemy_general(0, 1, 300);
        });
        sim.turn = 900;
        sim.push(
            Some(Move { from: at(3, 0, 2), dir: 2, half: false }),
            Some(Move { from: at(3, 1, 0), dir: 0, half: false }),
        );
        assert_eq!(sim.outcome(), Outcome::Draw);
    }
}
