//! Morpheus's frame in, joe's network input out.
//!
//! The seam between the two observation pipelines §4 of the plan keeps side by
//! side. Joe's `AugState` is authoritative for **the network's 39-channel
//! input and nothing else**; morpheus's `VisibleMemory` stays authoritative for
//! legality, build cost, tactics, node identity, belief filtering and
//! symmetry. Neither is deleted and neither is re-derived — the 49-plane
//! `build_tensor` that used to sit here is what went (joe-net-plan §4).
//!
//! Everything downstream of [`ObsBridge::widen`] is joe's code, byte for byte,
//! in the `joenet` crate. This file adds three things and no arithmetic:
//!
//! 1. **The widening adapter.** The two `Observation` structs carry the same
//!    ten fields in the same order and differ only in element type — `u8`
//!    grids here, `i32` grids there (§1.6). So the adapter is a widening loop
//!    and nothing else, and the constants it widens *into* are joe's own.
//! 2. **The turn's ownership.** `AugState` is 44.8 KB of history that advances
//!    exactly once per real turn, by `mem::swap`, in joe's order. Getting that
//!    order wrong is silent: the tensor stays in range and the net stays
//!    calibrated-looking. `sequence` in the parity harness is what proves it.
//! 3. **A zero penalties tensor** (§6.2), built once and never written.
//!
//! # Why the penalties are zero, and stay zero
//!
//! Joe's `-1e9` penalties are added to the flat logits *after* unpatchify
//! (`joenet::nn::net`, one elementwise add). The trunk never sees them, so
//! passing zeros changes nothing the network computes — and the port does not
//! want joe's mask anyway, because morpheus masks with `play_mask` at leaves
//! and `legal_mask` elsewhere and those are strictly different sets (§6.2,
//! §7.2). N0 priced the mask build it removes at 0.007 ms, so this is a
//! correctness decision rather than a saving.
//!
//! # What it costs per turn
//!
//! N0 measured the whole pipeline this drives — parse, raw, cost,
//! `augment_obs`, normalize — at 0.067 ms against a 20.81 ms forward, with
//! `augment_obs` itself at 0.042 ms. That measurement is why §6.3 took its top
//! row: the history is advanced per node rather than frozen at the root, and
//! R3 and R8 retired with it. At `search_depth: 8` that is nine of these
//! objects, 403 KB, and 0.38 ms of `augment_obs` against a leaf forward of
//! 23 ms.
//!
//! # Two methods where joe-rs has one
//!
//! [`ObsBridge::advance`] is the play path and does what joe-rs's `act` does.
//! [`ObsBridge::augment`] stops one step short, before
//! `normalize_observations`, because that is where joe-rs's `sequence` surface
//! takes its digest — the recorded corpus hashes the **unnormalized** tensor.
//! Splitting the two is what lets this bridge be checked against that corpus
//! turn for turn instead of against a second one nobody has captured (Q11).

use joenet::board::obs::{
    augment_obs, build_cost_from_raw, compute_build_mask_from_raw, compute_valid_move_mask,
    frame_to_raw, normalize_observations, prepare_action_mask, AugScratch, AugState, CELLS,
    N_ACTION_CHANNELS, N_CHANNELS, PAD, TEMPORAL_WINDOW,
};
use joenet::io::wire::Observation as JoeObservation;

use crate::io::wire::Observation;

/// Joe's observation pipeline, driven from morpheus's frames.
///
/// One per state that needs a network evaluation: the seat holds one for the
/// root, and N3's evaluator holds one per search-path depth. State is carried
/// per *path step* rather than per node — the search re-walks from the root
/// each simulation — so `search_depth: 8` needs nine of these, not
/// `max_tree_nodes` of them (§6.3).
pub struct ObsBridge {
    h: usize,
    w: usize,
    /// The widened frame. Joe's pipeline reads this, never morpheus's.
    frame: JoeObservation,
    state: AugState,
    next: AugState,
    scratch: AugScratch,
    raw: Vec<f32>,
    cost: Vec<i32>,
    aug: Vec<f32>,
    temporal: Vec<f32>,
    /// All zeros, for the length of the process (§6.2).
    penalties: Vec<f32>,
    /// N2 scaffolding, written only by [`Self::joe_action_penalties`].
    move_mask: Vec<bool>,
    build_mask: Vec<bool>,
    joe_penalties: Vec<f32>,
}

impl ObsBridge {
    /// Allocate every per-turn buffer once, for a board the handshake fixed.
    ///
    /// `Err` rather than a panic on an oversized board: the seat turns a
    /// construction failure into passing every turn with the reason on stderr,
    /// which costs the game, where an early exit costs the match
    /// (RULES.md §08).
    pub fn new(h: usize, w: usize) -> Result<Self, String> {
        if h > PAD || w > PAD {
            return Err(format!("board {h}x{w} exceeds the net's pad_to {PAD}"));
        }
        Ok(Self {
            h,
            w,
            frame: JoeObservation::with_dims(h, w),
            state: AugState::zeros(),
            next: AugState::zeros(),
            scratch: AugScratch::new(),
            raw: Vec::new(),
            cost: Vec::new(),
            aug: vec![0.0; N_CHANNELS * CELLS],
            temporal: vec![0.0; 2 * TEMPORAL_WINDOW],
            penalties: vec![0.0; N_ACTION_CHANNELS * CELLS],
            move_mask: Vec::new(),
            build_mask: Vec::new(),
            joe_penalties: vec![0.0; N_ACTION_CHANNELS * CELLS],
        })
    }

    /// Morpheus's frame into joe's, widening the two `u8` grids (§1.6).
    ///
    /// The scalars are already `i32` on both sides and the field order is the
    /// same, so nothing here reinterprets a value: `TYPE_*` and `OWNER_*` are
    /// the same wire codes with the same meanings, and both parsers read them
    /// off the same protocol line.
    fn widen(&mut self, obs: &Observation) {
        debug_assert_eq!((obs.h, obs.w), (self.h, self.w));
        self.frame.turn = obs.turn;
        self.frame.my_land = obs.my_land;
        self.frame.my_army = obs.my_army;
        self.frame.opp_land = obs.opp_land;
        self.frame.opp_army = obs.opp_army;
        let n = self.h * self.w;
        for i in 0..n {
            self.frame.type_grid[i] = obs.type_grid[i] as i32;
            self.frame.owner_grid[i] = obs.owner_grid[i] as i32;
            self.frame.army_grid[i] = obs.army_grid[i];
        }
    }

    /// Advance the history by one turn and leave the **unnormalized** tensor
    /// in [`Self::aug`].
    ///
    /// joe-rs's order, unchanged and load-bearing: raw, then cost, then
    /// `augment_obs` reading the old state into the new one, then the swap,
    /// then the two ring buffers into the temporal input. `augment_obs` reads
    /// `state` and fully overwrites `next`, so the swap is what makes this
    /// turn's answer next turn's input.
    ///
    /// The masks joe-rs builds between cost and augment are **not** built:
    /// they only ever fed `prepare_action_mask`, and the penalties are zero.
    pub fn augment(&mut self, obs: &Observation) {
        self.widen(obs);
        frame_to_raw(&self.frame, &mut self.raw);
        build_cost_from_raw(&self.raw, self.h, self.w, &mut self.cost);
        augment_obs(
            &self.raw,
            self.h,
            self.w,
            &self.cost,
            &self.state,
            &mut self.next,
            &mut self.scratch,
            &mut self.aug,
        );
        std::mem::swap(&mut self.state, &mut self.next);
        self.temporal[..TEMPORAL_WINDOW].copy_from_slice(&self.state.opponent_army_history);
        self.temporal[TEMPORAL_WINDOW..].copy_from_slice(&self.state.opponent_land_history);
    }

    /// Scale the army-valued channels by 1/50, in place. Separate from
    /// [`Self::augment`] because the corpus digests the tensor before it.
    pub fn normalize(&mut self) {
        normalize_observations(&mut self.aug);
    }

    /// The play path: one real turn in, a network-ready input out.
    pub fn advance(&mut self, obs: &Observation) {
        self.augment(obs);
        self.normalize();
    }

    /// The `(39, 21, 21)` tensor — normalized after [`Self::advance`], raw
    /// after [`Self::augment`] alone.
    pub fn aug(&self) -> &[f32] {
        &self.aug
    }

    /// The `(2, 512)` opponent army/land windows, raw counts: the `/50` lives
    /// inside joe's temporal encoder.
    pub fn temporal(&self) -> &[f32] {
        &self.temporal
    }

    /// The `(10, 21, 21)` zeros joe's forward adds to its flat logits (§6.2).
    pub fn penalties(&self) -> &[f32] {
        &self.penalties
    }

    /// Joe's own `(h, w)` castle price per cell, at stride `w`.
    ///
    /// Exposed for Q10 only. After §6.2 the play path consults morpheus's
    /// `live_build_cost` and never this, so a disagreement between the two
    /// would be silent — the `sequence` parity surface counts the cells where
    /// they differ, per turn, over whole games.
    pub fn build_cost(&self) -> &[i32] {
        &self.cost
    }

    /// The accumulated history, for the parity harness's final-state check.
    pub fn state(&self) -> &AugState {
        &self.state
    }

    /// **N2 scaffolding.** Joe's own `-1e9` action mask, built from the frame
    /// [`Self::augment`] last read.
    ///
    /// The shipped bot never calls this and N3 deletes it. It exists for one
    /// phase and one gate: N2 asks for the fork's replies to be **byte-equal
    /// to joe-rs's** over a whole recorded game, which localizes a bridge bug
    /// before the search can complicate it — and joe-rs's argmax is over
    /// *masked* logits, so an unmasked argmax would diverge wherever joe's
    /// best raw action is illegal and the comparison would prove nothing.
    ///
    /// So the mask is real here and the *network input* is still zeros: joe's
    /// forward adds `penalties` to its flat logits as one elementwise add
    /// after unpatchify, so adding them afterwards instead gives bit-identical
    /// logits (`x + 0.0 + p` and `x + p` agree for every finite `x`, including
    /// `x = -0.0`, where both give `p`). §6.2's decision — pass zeros, never
    /// consult joe's mask on the play path — is therefore intact, and N3
    /// replaces the caller with the 4,410 -> 3,970 remap and morpheus's own
    /// `legal_mask`, which is a strictly different set.
    ///
    /// Costs 0.007 ms per call (N0), which is why this was never a saving.
    pub fn joe_action_penalties(&mut self) -> &[f32] {
        compute_valid_move_mask(&self.raw, self.h, self.w, &mut self.move_mask);
        compute_build_mask_from_raw(
            &self.raw,
            self.h,
            self.w,
            &self.cost,
            &mut self.build_mask,
        );
        prepare_action_mask(
            &self.move_mask,
            &self.build_mask,
            self.h,
            self.w,
            &mut self.joe_penalties,
        );
        &self.joe_penalties
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::wire::{OWNER_ME, OWNER_OPP, TYPE_GENERAL, TYPE_MOUNTAIN, TYPE_PLAIN};

    fn frame(turn: i32) -> Observation {
        let mut obs = Observation::with_dims(5, 5);
        obs.turn = turn;
        obs.my_land = 2;
        obs.my_army = 9;
        obs.opp_land = 3;
        obs.opp_army = 11;
        obs.type_grid.fill(TYPE_PLAIN);
        let general = obs.idx(2, 2);
        obs.type_grid[general] = TYPE_GENERAL;
        obs.owner_grid[general] = OWNER_ME;
        obs.army_grid[general] = 9 + turn;
        let enemy = obs.idx(0, 4);
        obs.owner_grid[enemy] = OWNER_OPP;
        obs.army_grid[enemy] = 11;
        let mountain = obs.idx(4, 0);
        obs.type_grid[mountain] = TYPE_MOUNTAIN;
        obs
    }

    /// The widening is a widening: every code survives it unchanged, and the
    /// two crates' constants are the same wire codes.
    #[test]
    fn widening_preserves_every_wire_code() {
        let obs = frame(3);
        let mut bridge = ObsBridge::new(5, 5).unwrap();
        bridge.widen(&obs);
        assert_eq!(bridge.frame.turn, 3);
        assert_eq!(bridge.frame.opp_army, 11);
        for i in 0..25 {
            assert_eq!(bridge.frame.type_grid[i], obs.type_grid[i] as i32);
            assert_eq!(bridge.frame.owner_grid[i], obs.owner_grid[i] as i32);
            assert_eq!(bridge.frame.army_grid[i], obs.army_grid[i]);
        }
        assert_eq!(
            joenet::io::wire::TYPE_GENERAL,
            i32::from(crate::io::wire::TYPE_GENERAL)
        );
        assert_eq!(
            joenet::io::wire::OWNER_OPP,
            i32::from(crate::io::wire::OWNER_OPP)
        );
    }

    /// The swap is the whole state machine. Without it every turn would read
    /// an all-zero history and the tensor would still look plausible.
    #[test]
    fn the_history_advances_once_per_turn() {
        let mut bridge = ObsBridge::new(5, 5).unwrap();
        assert_eq!(bridge.state().temporal_step, 0);
        for turn in 1..=4 {
            bridge.advance(&frame(turn));
            assert_eq!(bridge.state().temporal_step, turn);
        }
        // The ring buffer keeps the last four `opp_army` values at its end.
        let history = &bridge.temporal()[..TEMPORAL_WINDOW];
        assert_eq!(history[TEMPORAL_WINDOW - 1], 11.0);
        assert_eq!(history[TEMPORAL_WINDOW - 5], 0.0);
    }

    /// `augment` leaves the tensor unnormalized and `normalize` scales it, in
    /// that order — the split the corpus digest depends on.
    #[test]
    fn augment_stops_before_the_scaling_and_advance_does_not() {
        let mut raw_only = ObsBridge::new(5, 5).unwrap();
        raw_only.augment(&frame(1));
        let unnormalized = raw_only.aug()[0];
        raw_only.normalize();
        assert_eq!(raw_only.aug()[0], unnormalized / 50.0);

        let mut played = ObsBridge::new(5, 5).unwrap();
        played.advance(&frame(1));
        assert_eq!(played.aug()[0], raw_only.aug()[0]);
    }

    /// The penalties are zero and nothing writes them.
    #[test]
    fn the_penalties_are_zero() {
        let mut bridge = ObsBridge::new(5, 5).unwrap();
        bridge.advance(&frame(1));
        assert_eq!(bridge.penalties().len(), N_ACTION_CHANNELS * CELLS);
        assert!(bridge.penalties().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn a_board_past_the_pad_is_refused_rather_than_asserted() {
        assert!(ObsBridge::new(PAD + 1, PAD).is_err());
    }
}
