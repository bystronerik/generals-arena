//! The afterstate: the position a candidate leaves behind, rendered back into
//! the frame the network eats.
//!
//! Every other file in `search` runs from a frame *inwards*, to an answer
//! about the rules. This one runs back out: candidate → [`Sim`] advance under
//! opponent-pass → a hypothetical wire frame → the 39-channel tensor. That is
//! what one-step policy improvement needs, because joe's value head reads an
//! augmented observation and nothing else. `Seat::act` runs the forward on
//! what this module produces; nothing here names the network, and the pipeline
//! it drives is the same `board::obs` functions the live path drives, called
//! in the same order.
//!
//! # The board it advances on is the visible one
//!
//! [`Fog::VisibleOnly`], never [`Fog::Pessimistic`]. The pessimistic board is
//! a proof device: it puts the opponent's entire unaccounted army on every
//! fogged cell at once, which is a superset of reality and not a position at
//! all. Asking the value head what it thinks of a board like that would return
//! a number about a game nobody is playing. The re-rank scores lines on the
//! board as the frame states it.
//!
//! # What one ply of opponent-pass can and cannot know
//!
//! * **The destination is always visible.** Vision is the 3×3 pool around
//!   owned cells (RULES.md §06), so every neighbour of a cell we own is lit.
//!   A move's source is ours, so its destination is a cell the frame states
//!   exactly — there is no fog gamble anywhere in a one-ply advance, and the
//!   combat it resolves is the combat the engine would resolve.
//! * **Vision only grows.** With the opponent passing we cannot lose a cell,
//!   so the owned set only gains the destination, so the §06 pool only gains
//!   cells. The ones it gains are cells the base frame could not see into and
//!   this module has nothing to put in — so they stay dark. That is the one
//!   place the render deviates from what the engine would send, and it
//!   deviates by *withholding* knowledge rather than by inventing it.
//! * **The opponent's totals are carried, not recomputed.** `opp_land` and
//!   `opp_army` are global truths the frame states and the board cannot
//!   reproduce, because most of their army is in fog. So the render carries
//!   the frame's numbers and applies the deltas it can see: what our move took
//!   off a visible enemy cell, the growth the sim gave their visible cells,
//!   and — on a 50-turn tick — one army for each of their cells we cannot see,
//!   which the frame's land count states exactly. What it misses is production
//!   from enemy castles under fog, at most one army per even tick.
//!
//! That last miss is worth being precise about, because it is the only
//! approximation here that a value could ride on. It is **the same for every
//! candidate on a turn**, and this module exists to *rank* candidates against
//! each other on one turn. An error every candidate carries equally cancels
//! out of the comparison; what has to be exact is what differs between them,
//! and that is our own move, which is exact.

use crate::board::memory::{is_visible, Memory};
use crate::board::obs::{
    augment_obs, build_cost_from_raw, compute_build_mask_from_raw, compute_valid_move_mask,
    frame_to_raw, normalize_observations, prepare_action_mask, AugScratch, AugState, CELLS,
    N_ACTION_CHANNELS, N_CHANNELS, TEMPORAL_WINDOW,
};
use crate::io::wire::{
    Observation, OWNER_ME, OWNER_OPP, TYPE_CASTLE, TYPE_FOG, TYPE_MOUNTAIN, TYPE_STRUCTURE_IN_FOG,
};

use super::sim::{Config, Move, Outcome, Sim, ME, OPP};

/// What we might do with the turn, as the afterstate models it.
///
/// The wire's three action kinds, minus the ones no position can hold: an
/// action in the padded region is a [`Play::Pass`], because the engine voids
/// it and a pass is what voiding leaves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Play {
    /// A move, exactly as [`Sim`] resolves it.
    Act(Move),
    /// A castle at this cell, at the price the base frame's cost grid states.
    Build(usize),
    Pass,
}

/// What advancing a candidate produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Advance {
    /// A network input is ready in [`Afterstate::aug`], `penalties` and
    /// `temporal`.
    Ready,
    /// The candidate takes their general while they stand still. There is no
    /// next frame to render, and nothing a value head could say would rank
    /// above it.
    Wins,
}

/// One turn's afterstate machine: point it at a frame with [`reset`], then
/// [`advance`] and [`undo`] one candidate at a time.
///
/// Every buffer is allocated once at the handshake, like the live pipeline's,
/// because this runs inside the same 150 ms.
///
/// No `Debug`: `AugScratch` has none, and giving it one would be a change to a
/// copied file for the sake of a derive nothing prints.
///
/// [`reset`]: Afterstate::reset
/// [`advance`]: Afterstate::advance
/// [`undo`]: Afterstate::undo
pub struct Afterstate {
    h: usize,
    w: usize,
    sim: Sim,
    /// The base frame's type per cell, and what that cell renders as when it
    /// is dark. Both settled by [`Afterstate::reset`].
    base_type: Vec<i32>,
    dark_type: Vec<i32>,
    /// Which cells the base frame could see into. Nothing outside this set is
    /// ever rendered, however much vision a candidate gains.
    visible: Vec<bool>,
    /// §06 vision, recomputed from the ownership a candidate leaves behind.
    lit: Vec<bool>,
    /// The §03 price per cell on the base frame — what a build candidate pays.
    price: Vec<i32>,
    /// The opponent's totals as the frame states them, and the part of those
    /// totals sitting on cells we can see. The difference is their army in
    /// fog, which the render carries forward untouched.
    opp_land: i32,
    opp_army: i32,
    seen_opp_land: i32,
    seen_opp_army: i32,
    /// The cell a build candidate turned into a castle, while it is advanced.
    built: Option<usize>,
    /// The rendered frame, and the pipeline over it: the same `board::obs`
    /// functions `Seat::act` calls, in the same order.
    frame: Observation,
    raw: Vec<f32>,
    cost: Vec<i32>,
    move_mask: Vec<bool>,
    build_mask: Vec<bool>,
    /// The scratch `augment_obs` writes its next state into. The live state is
    /// only ever read — `augment_obs` takes it by shared reference — so the
    /// afterstate cannot disturb the game's own accumulation.
    next: AugState,
    scratch: AugScratch,
    /// The forward's three inputs, valid after an [`Advance::Ready`].
    pub aug: Vec<f32>,
    pub penalties: Vec<f32>,
    pub temporal: Vec<f32>,
}

impl Afterstate {
    pub fn new(h: usize, w: usize) -> Self {
        let cells = h * w;
        Self {
            h,
            w,
            sim: Sim::blank(h, w),
            base_type: vec![TYPE_FOG; cells],
            dark_type: vec![TYPE_FOG; cells],
            visible: vec![false; cells],
            lit: vec![false; cells],
            price: vec![0; cells],
            opp_land: 0,
            opp_army: 0,
            seen_opp_land: 0,
            seen_opp_army: 0,
            built: None,
            frame: Observation::with_dims(h, w),
            raw: Vec::new(),
            cost: Vec::new(),
            move_mask: Vec::new(),
            build_mask: Vec::new(),
            next: AugState::zeros(),
            scratch: AugScratch::new(),
            aug: vec![0.0; N_CHANNELS * CELLS],
            penalties: vec![0.0; N_ACTION_CHANNELS * CELLS],
            temporal: vec![0.0; 2 * TEMPORAL_WINDOW],
        }
    }

    /// Point the machine at this turn's frame. `cost` is the live pipeline's
    /// own build-cost grid, so a build candidate pays the price the engine
    /// would charge and not a second reading of §03.
    ///
    /// `false` when there is no model — only ever before our own general has
    /// been on a frame.
    pub fn reset(&mut self, obs: &Observation, mem: &Memory, cost: &[i32], cfg: &Config) -> bool {
        debug_assert_eq!((self.h, self.w), (obs.h, obs.w));
        let Some(sim) = Sim::from_frame(obs, mem, cfg) else {
            return false;
        };
        self.sim = sim;
        self.built = None;
        self.opp_land = obs.opp_land;
        self.opp_army = obs.opp_army;
        self.seen_opp_land = 0;
        self.seen_opp_army = 0;
        self.price.copy_from_slice(cost);

        for cell in 0..self.h * self.w {
            let cell_type = obs.type_grid[cell];
            self.base_type[cell] = cell_type;
            self.visible[cell] = is_visible(cell_type);
            // The engine splits fog by whether the cell is a mountain or a
            // castle (`fog_cells = invisible & ~(mountains | castles)`). The
            // cells we can classify are the ones we have seen: a remembered
            // mountain, one the frame already shows as unresolved, and a
            // castle in view now. Everything else is plain fog — which
            // includes a general in fog, since a general is neither.
            self.dark_type[cell] = if mem.mountains[cell]
                || matches!(cell_type, TYPE_MOUNTAIN | TYPE_CASTLE | TYPE_STRUCTURE_IN_FOG)
            {
                TYPE_STRUCTURE_IN_FOG
            } else {
                TYPE_FOG
            };
            if self.visible[cell] && obs.owner_grid[cell] == OWNER_OPP {
                self.seen_opp_land += 1;
                self.seen_opp_army += obs.army_grid[cell];
            }
        }
        true
    }

    /// Advance one candidate and render what it leaves. Pair every call with
    /// [`Afterstate::undo`] — including after an [`Advance::Wins`], which
    /// advanced the board just the same.
    pub fn advance(&mut self, play: Play, state: &AugState) -> Advance {
        match play {
            Play::Act(mv) => self.sim.push(Some(mv), None),
            Play::Pass => self.sim.push(None, None),
            // `push_build` says whether the tick actually built. An invalid
            // build is a silent pass (§03), and the render must show the pass
            // rather than a castle that was never paid for.
            Play::Build(cell) => {
                self.built = self.sim.push_build(cell, self.price[cell]).then_some(cell);
            }
        }
        // With the opponent passing, `MeWins` is the only end this can reach:
        // our own move cannot take our own general, and theirs never happens.
        if self.sim.outcome() == Outcome::MeWins {
            return Advance::Wins;
        }
        self.render();
        self.pipeline(state);
        Advance::Ready
    }

    /// Take the candidate back. The board returns to the frame `reset` was
    /// given, so the next candidate starts where this one did.
    pub fn undo(&mut self) {
        self.sim.pop();
        self.built = None;
    }

    /// The position the sim now holds, written back into a wire frame.
    fn render(&mut self) {
        self.vision();
        let cells = self.h * self.w;
        let (mut my_land, mut my_army) = (0i32, 0i32);
        let (mut opp_land, mut opp_army) = (0i32, 0i32);

        for cell in 0..cells {
            let owner = self.sim.owner_at(cell);
            let army = self.sim.army_at(cell);
            if owner == ME {
                // Every cell we own is inside its own 3×3 pool, so our totals
                // are exact — the frame's own numbers, recomputed.
                my_land += 1;
                my_army += army;
            }
            if owner == OPP && self.visible[cell] {
                opp_land += 1;
                opp_army += army;
            }
            if self.lit[cell] {
                self.frame.type_grid[cell] = if self.built == Some(cell) {
                    TYPE_CASTLE
                } else {
                    self.base_type[cell]
                };
                self.frame.owner_grid[cell] = match owner {
                    ME => OWNER_ME,
                    OPP => OWNER_OPP,
                    _ => 0,
                };
                self.frame.army_grid[cell] = army;
            } else {
                self.frame.type_grid[cell] = self.dark_type[cell];
                self.frame.owner_grid[cell] = 0;
                self.frame.army_grid[cell] = 0;
            }
        }

        self.frame.turn = self.sim.turn;
        self.frame.my_land = my_land;
        self.frame.my_army = my_army;
        // Their army in fog is what the frame's total does not account for.
        // It only moves on a 50-turn tick, where every owned cell gains one
        // and the frame states how many of theirs we cannot see.
        let hidden_land = (self.opp_land - self.seen_opp_land).max(0);
        let hidden_growth = i32::from(self.sim.turn % 50 == 0) * hidden_land;
        self.frame.opp_land = self.opp_land + (opp_land - self.seen_opp_land);
        self.frame.opp_army =
            self.opp_army + (opp_army - self.seen_opp_army) + hidden_growth;
    }

    /// RULES.md §06 on the ownership the candidate left: we see a cell when we
    /// own one of the nine in its 3×3 neighbourhood (the engine's
    /// `get_visibility`, which pools the owner mask including the cell
    /// itself), intersected with what the base frame could see into.
    ///
    /// The intersection is what keeps the render honest. At one ply against a
    /// passing opponent the pool can only *grow*, so it never darkens a cell
    /// the frame lit; every cell it adds is one we have no contents for, and
    /// leaving those dark is the alternative to inventing them.
    fn vision(&mut self) {
        for row in 0..self.h {
            for col in 0..self.w {
                let mut pooled = false;
                'pool: for dr in -1i32..=1 {
                    for dc in -1i32..=1 {
                        let (r, c) = (row as i32 + dr, col as i32 + dc);
                        if r < 0 || c < 0 || r >= self.h as i32 || c >= self.w as i32 {
                            continue;
                        }
                        if self.sim.owner_at(r as usize * self.w + c as usize) == ME {
                            pooled = true;
                            break 'pool;
                        }
                    }
                }
                let cell = row * self.w + col;
                self.lit[cell] = pooled && self.visible[cell];
            }
        }
    }

    /// The rendered frame through joe's own observation pipeline — the same
    /// calls in the same order as `Seat::act`, on this module's buffers.
    fn pipeline(&mut self, state: &AugState) {
        frame_to_raw(&self.frame, &mut self.raw);
        build_cost_from_raw(&self.raw, self.h, self.w, &mut self.cost);
        compute_valid_move_mask(&self.raw, self.h, self.w, &mut self.move_mask);
        compute_build_mask_from_raw(&self.raw, self.h, self.w, &self.cost, &mut self.build_mask);
        augment_obs(
            &self.raw,
            self.h,
            self.w,
            &self.cost,
            state,
            &mut self.next,
            &mut self.scratch,
            &mut self.aug,
        );
        self.temporal[..TEMPORAL_WINDOW].copy_from_slice(&self.next.opponent_army_history);
        self.temporal[TEMPORAL_WINDOW..].copy_from_slice(&self.next.opponent_land_history);
        normalize_observations(&mut self.aug);
        prepare_action_mask(
            &self.move_mask,
            &self.build_mask,
            self.h,
            self.w,
            &mut self.penalties,
        );
    }

    /// The frame the last [`Advance::Ready`] rendered. Test-only: play reads
    /// the tensors, not the frame that produced them.
    #[cfg(test)]
    pub fn frame(&self) -> &Observation {
        &self.frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::obs::HISTORY;
    use crate::io::wire::{TYPE_GENERAL, TYPE_PLAIN};
    use crate::search::fixtures::{apply_fog, at, remembering, set, wire_frame};
    use crate::search::sim::Fog;

    const H: usize = 6;
    const W: usize = 6;

    const CFG: Config =
        Config { fog: Fog::VisibleOnly, hidden_bound: 0, deathtouch_turn: 800, i_am_p0: true };

    /// A machine pointed at one frame, with the frame's own build prices.
    fn machine(obs: &Observation) -> Afterstate {
        let mem = remembering(&[obs]);
        let mut raw = Vec::new();
        let mut cost = Vec::new();
        frame_to_raw(obs, &mut raw);
        build_cost_from_raw(&raw, obs.h, obs.w, &mut cost);
        let mut after = Afterstate::new(obs.h, obs.w);
        assert!(after.reset(obs, &mem, &cost, &CFG));
        after
    }

    fn mv(row: usize, col: usize, dir: u8) -> Play {
        Play::Act(Move { from: at(W, row, col), dir, half: false })
    }

    /// Every plane of an `AugState` in one vector, so a test can say
    /// "unchanged" in a single assert.
    fn digest(state: &AugState) -> Vec<f64> {
        let mut out: Vec<f64> = Vec::new();
        for plane in [
            &state.army_stack,
            &state.enemy_stack,
            &state.last_army,
            &state.last_enemy_army,
            &state.last_enemy_army_seen_value,
            &state.last_enemy_army_seen_timestep,
            &state.opponent_army_history,
            &state.opponent_land_history,
        ] {
            out.extend(plane.iter().map(|v| f64::from(*v)));
        }
        for plane in [
            &state.castles,
            &state.generals,
            &state.mountains,
            &state.seen,
            &state.enemy_seen,
        ] {
            out.extend(plane.iter().map(|v| f64::from(i32::from(*v))));
        }
        out.push(f64::from(state.temporal_step));
        out
    }

    /// The frame a move leaves, cell for cell: the source keeps one behind,
    /// the destination holds the rest, and nothing else on the board moved.
    #[test]
    fn a_move_renders_the_army_it_actually_moved() {
        let mut obs = wire_frame(H, W, 41);
        set(&mut obs, 3, 3, TYPE_GENERAL, OWNER_ME, 4);
        set(&mut obs, 3, 2, TYPE_PLAIN, OWNER_ME, 9);
        set(&mut obs, 3, 1, TYPE_PLAIN, OWNER_OPP, 3);
        obs.my_land = 2;
        obs.my_army = 13;
        obs.opp_land = 5;
        obs.opp_army = 20;
        let mut after = machine(&obs);

        // (3,2) sends 8 left into their 3: it takes the cell with 5 left.
        assert_eq!(after.advance(mv(3, 2, 2), &AugState::zeros()), Advance::Ready);
        let frame = after.frame();
        assert_eq!(frame.turn, 42);
        assert_eq!(frame.army_grid[at(W, 3, 2)], 1, "one stays behind");
        assert_eq!(frame.army_grid[at(W, 3, 1)], 5, "8 arrived against 3");
        assert_eq!(frame.owner_grid[at(W, 3, 1)], OWNER_ME);
        assert_eq!(frame.type_grid[at(W, 3, 1)], TYPE_PLAIN);
        // Turn 42 is even, so our general produced (§04).
        assert_eq!(frame.army_grid[at(W, 3, 3)], 5);
        assert_eq!(frame.owner_grid[at(W, 3, 3)], OWNER_ME);
        assert_eq!(frame.type_grid[at(W, 3, 3)], TYPE_GENERAL);

        // Our totals are exact; theirs carry the fog they came with. They held
        // 20 over 5 cells with 3 in sight, and we took that cell.
        assert_eq!((frame.my_land, frame.my_army), (3, 1 + 5 + 5));
        assert_eq!((frame.opp_land, frame.opp_army), (4, 17));
    }

    /// Vision that a candidate gains buys the render nothing. The frame this
    /// starts from is shaped the way the engine shapes one: we own (0, 2), so
    /// the 3×3 pool lights out to column 3 and the rest is dark. Stepping onto
    /// (0, 3) pulls (0, 4) into the pool — and it stays dark, because the base
    /// frame never said what is on it.
    #[test]
    fn vision_grows_but_the_render_never_invents_a_cell() {
        let mut obs = wire_frame(H, W, 10);
        set(&mut obs, 0, 0, TYPE_GENERAL, OWNER_ME, 9);
        set(&mut obs, 0, 2, TYPE_PLAIN, OWNER_ME, 6);
        apply_fog(&mut obs);
        let mut after = machine(&obs);

        assert_eq!(after.advance(mv(0, 2, 3), &AugState::zeros()), Advance::Ready);
        let frame = after.frame();
        assert_eq!(frame.owner_grid[at(W, 0, 3)], OWNER_ME, "we took the lit cell");
        assert_eq!(frame.army_grid[at(W, 0, 3)], 5);
        assert_eq!(
            frame.type_grid[at(W, 0, 4)],
            TYPE_FOG,
            "newly pooled, but the base frame never saw it"
        );
        assert_eq!(frame.army_grid[at(W, 0, 4)], 0);
        assert_eq!(frame.owner_grid[at(W, 0, 4)], 0);
    }

    /// Nothing the base frame lit ever goes dark. With the opponent passing we
    /// cannot lose a cell, so the §06 pool only grows — which is the other
    /// half of why the intersection is safe.
    #[test]
    fn a_candidate_never_darkens_a_cell_the_frame_lit() {
        let mut obs = wire_frame(H, W, 10);
        set(&mut obs, 0, 0, TYPE_GENERAL, OWNER_ME, 9);
        set(&mut obs, 2, 2, TYPE_PLAIN, OWNER_ME, 8);
        apply_fog(&mut obs);
        let lit: Vec<usize> =
            (0..H * W).filter(|cell| is_visible(obs.type_grid[*cell])).collect();
        let mut after = machine(&obs);

        // Walk the stack off (2,2) in every direction it has.
        for dir in 0..4u8 {
            after.advance(Play::Act(Move { from: at(W, 2, 2), dir, half: false }), &AugState::zeros());
            for cell in &lit {
                assert!(
                    is_visible(after.frame().type_grid[*cell]),
                    "dir {dir} put cell {cell} back into fog"
                );
            }
            after.undo();
        }
    }

    /// A remembered mountain reads as an unresolved structure once it is back
    /// in fog — the engine's `structures_in_fog`, and the one dark cell that
    /// is not plain fog.
    #[test]
    fn a_dark_cell_renders_as_the_kind_of_dark_it_is() {
        let mut seen = wire_frame(H, W, 9);
        set(&mut seen, 0, 0, TYPE_GENERAL, OWNER_ME, 9);
        set(&mut seen, 5, 5, TYPE_MOUNTAIN, 0, 0);
        let mut now = wire_frame(H, W, 10);
        set(&mut now, 0, 0, TYPE_GENERAL, OWNER_ME, 9);
        set(&mut now, 5, 5, TYPE_STRUCTURE_IN_FOG, 0, 0);
        set(&mut now, 5, 4, TYPE_FOG, 0, 0);

        let mem = remembering(&[&seen, &now]);
        let mut raw = Vec::new();
        let mut cost = Vec::new();
        frame_to_raw(&now, &mut raw);
        build_cost_from_raw(&raw, H, W, &mut cost);
        let mut after = Afterstate::new(H, W);
        assert!(after.reset(&now, &mem, &cost, &CFG));

        assert_eq!(after.advance(Play::Pass, &AugState::zeros()), Advance::Ready);
        assert_eq!(after.frame().type_grid[at(W, 5, 5)], TYPE_STRUCTURE_IN_FOG);
        assert_eq!(after.frame().type_grid[at(W, 5, 4)], TYPE_FOG);
    }

    /// A build spends the price off the cell, turns it into a castle, and the
    /// castle produces on the very tick it was built when that tick is even
    /// (RULES.md §03, §04).
    #[test]
    fn a_build_pays_the_price_and_produces_at_once() {
        let mut obs = wire_frame(H, W, 41);
        set(&mut obs, 0, 0, TYPE_GENERAL, OWNER_ME, 5);
        set(&mut obs, 5, 5, TYPE_PLAIN, OWNER_ME, 50);
        obs.my_land = 2;
        obs.my_army = 55;
        let mut after = machine(&obs);

        // Our only structure is the general, ten manhattan steps away, so the
        // price is the bare 35 (§03: nothing at d >= 7 adds a surcharge).
        assert_eq!(after.advance(Play::Build(at(W, 5, 5)), &AugState::zeros()), Advance::Ready);
        let frame = after.frame();
        assert_eq!(frame.type_grid[at(W, 5, 5)], TYPE_CASTLE);
        assert_eq!(frame.owner_grid[at(W, 5, 5)], OWNER_ME);
        assert_eq!(frame.army_grid[at(W, 5, 5)], 50 - 35 + 1, "remainder, then production");
        assert_eq!(frame.my_land, 2, "a build takes no ground");

        // And it is gone again once the candidate is taken back.
        after.undo();
        assert_eq!(after.advance(Play::Pass, &AugState::zeros()), Advance::Ready);
        assert_eq!(after.frame().type_grid[at(W, 5, 5)], TYPE_PLAIN);
        assert_eq!(after.frame().army_grid[at(W, 5, 5)], 50);
    }

    /// A build the cell cannot afford is a silent pass, as the engine treats
    /// it — the army stays and no castle appears.
    #[test]
    fn a_build_the_cell_cannot_afford_is_a_pass() {
        let mut obs = wire_frame(H, W, 41);
        set(&mut obs, 0, 0, TYPE_GENERAL, OWNER_ME, 5);
        set(&mut obs, 5, 5, TYPE_PLAIN, OWNER_ME, 34);
        let mut after = machine(&obs);

        assert_eq!(after.advance(Play::Build(at(W, 5, 5)), &AugState::zeros()), Advance::Ready);
        assert_eq!(after.frame().type_grid[at(W, 5, 5)], TYPE_PLAIN);
        assert_eq!(after.frame().army_grid[at(W, 5, 5)], 34);
    }

    /// A candidate that takes their general has no next frame, and says so
    /// instead of rendering one.
    #[test]
    fn a_winning_candidate_has_no_frame_to_render() {
        let mut obs = wire_frame(H, W, 41);
        set(&mut obs, 5, 5, TYPE_GENERAL, OWNER_ME, 5);
        set(&mut obs, 0, 0, TYPE_GENERAL, OWNER_OPP, 3);
        set(&mut obs, 0, 1, TYPE_PLAIN, OWNER_ME, 9);
        let mut after = machine(&obs);

        assert_eq!(after.advance(mv(0, 1, 2), &AugState::zeros()), Advance::Wins);
        after.undo();
        // A weaker stack in the same place only takes the cell if it can.
        after.sim.army[at(W, 0, 1)] = 3;
        assert_eq!(after.advance(mv(0, 1, 2), &AugState::zeros()), Advance::Ready);
    }

    /// Every candidate starts from the frame, not from the one before it.
    #[test]
    fn undo_returns_the_board_to_the_frame() {
        let mut obs = wire_frame(H, W, 41);
        set(&mut obs, 3, 3, TYPE_GENERAL, OWNER_ME, 4);
        set(&mut obs, 3, 2, TYPE_PLAIN, OWNER_ME, 40);
        let mut after = machine(&obs);

        let root = (after.sim.owner.clone(), after.sim.army.clone(), after.sim.turn);
        for play in [mv(3, 2, 2), Play::Build(at(W, 3, 2)), Play::Pass, mv(3, 2, 0)] {
            after.advance(play, &AugState::zeros());
            after.undo();
            assert_eq!(
                (after.sim.owner.clone(), after.sim.army.clone(), after.sim.turn),
                root,
                "{play:?} left something behind"
            );
            assert_eq!(after.sim.structures.len(), 1, "{play:?} left a structure behind");
        }
    }

    /// The live augmentation state is read and never written. `augment_obs`
    /// takes it by shared reference, so this is the type system's guarantee —
    /// the test is here because the *wiring* is what could get it wrong, and a
    /// state quietly advanced by a hypothetical would corrupt every later turn.
    #[test]
    fn the_live_state_is_only_ever_read() {
        let mut obs = wire_frame(H, W, 41);
        set(&mut obs, 3, 3, TYPE_GENERAL, OWNER_ME, 4);
        set(&mut obs, 3, 2, TYPE_PLAIN, OWNER_ME, 40);
        let mut after = machine(&obs);

        let mut live = AugState::zeros();
        // A state with something in it, so an accidental write would show.
        live.last_army[7] = 3.0;
        live.seen[9] = true;
        live.temporal_step = 12;
        let before = digest(&live);

        for play in [mv(3, 2, 2), Play::Build(at(W, 3, 2)), Play::Pass] {
            after.advance(play, &live);
            after.undo();
        }
        assert_eq!(digest(&live), before);
        // And the scratch it wrote instead did move.
        assert_ne!(digest(&after.next), before);
        assert_eq!(after.next.temporal_step, 13, "the afterstate is one turn on");
    }

    /// The tensor the forward eats has the shape the forward asserts, and the
    /// history stacks came from the state the caller passed rather than from
    /// nothing.
    #[test]
    fn the_rendered_tensor_is_the_shape_the_forward_wants() {
        let mut obs = wire_frame(H, W, 41);
        set(&mut obs, 3, 3, TYPE_GENERAL, OWNER_ME, 4);
        set(&mut obs, 3, 2, TYPE_PLAIN, OWNER_ME, 9);
        let mut after = machine(&obs);
        after.advance(mv(3, 2, 2), &AugState::zeros());

        assert_eq!(after.aug.len(), N_CHANNELS * CELLS);
        assert_eq!(after.penalties.len(), N_ACTION_CHANNELS * CELLS);
        assert_eq!(after.temporal.len(), 2 * TEMPORAL_WINDOW);
        // The oldest history plane is the zeros the empty state carried in.
        let oldest = 25 + HISTORY - 1;
        assert!(after.aug[oldest * CELLS..(oldest + 1) * CELLS].iter().all(|v| *v == 0.0));
        // The pass channel is legal everywhere, as `prepare_action_mask` has
        // it — the cheapest proof that the mask ran on the rendered frame.
        assert!(after.penalties[8 * CELLS..9 * CELLS].iter().all(|v| *v == 0.0));
    }
}
