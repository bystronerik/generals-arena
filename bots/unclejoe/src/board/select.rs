//! Deterministic Gumbel selection over the masked logits — selection-plan S1
//! (docs/bots/joe-rs/selection-plan.md).
//!
//! The Gumbel-max trick: `argmax_i(logit_i + T · G_i)` with i.i.d. standard
//! Gumbel noise `G_i = −ln(−ln u_i)` is an **exact** sample from
//! `softmax(logits / T)`. PPO collected joe's training data by sampling the
//! policy; deployment's plain argmax is the measured cause of the castle
//! limit cycle, and T = 1 restores the very distribution the weights were
//! trained under. The `u_i` are not random: each is hashed from
//! `(board digest, turn, action index)`, so the draw is exact temperature
//! sampling while the bot stays a **pure function of the game** — replays
//! reproduce byte for byte, and the turn in the seed makes every revisit a
//! fresh draw, so escape from any loop is geometric in time.
//!
//! **unclejoe ships T = 0** — the plain first-max argmax over the legal
//! entries, deliberately accepting the limit-cycle risk the noise exists to
//! fix (docs/bots/unclejoe/index.md). The Gumbel path below is kept intact
//! and unit-tested; a positive `JOE_RS_TEMPERATURE` re-enables it for
//! diagnostics. joe-rs, which this file is forked from, ships T = 1.
//!
//! Two properties are load-bearing and each has a mutation plant in
//! `tools/mutation_check.py`, killed by the unit tests below:
//! * **masked entries never receive noise** — legality is the mask's call,
//!   not the logit's, so an illegal action cannot be lifted;
//! * **the seed folds in the turn** — a turn-blind hash repeats the same
//!   draw on a revisited position, which is the limit cycle back again.
//!
//! All float work here is integer hashing plus f64 `+ * /` and the module's
//! own `ln` polynomial, so the draw is bit-identical on every host — the
//! same reason `nn` carries its own `exp` and `xla_math` its own `log1p`.

use crate::board::action::argmax;
use crate::board::obs::{CELLS, CH_MOUNTAINS, CH_OWNED, PAD};
use crate::io::wire::Observation;

/// The wire directions, in the head's channel order: up, down, left, right.
const MOVE_DIRS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

/// The moving stack's recent departure cells, newest last — the memory the
/// S4 trail penalty reads (docs/research/strategies/joe-rs-noundo.md).
///
/// A ring of the last `cap` move *sources*. A pass or a build pushes
/// nothing and clears nothing: a pause does not forgive a circle, and a
/// circuit resumed after a build is still a circuit. `cap` 0 disables the
/// memory entirely.
pub struct Trail {
    cells: Vec<(usize, usize)>,
    cap: usize,
}

impl Trail {
    pub fn new(cap: usize) -> Self {
        Self { cells: Vec::with_capacity(cap), cap }
    }

    pub fn push(&mut self, cell: (usize, usize)) {
        if self.cap == 0 {
            return;
        }
        self.cells.push(cell);
        if self.cells.len() > self.cap {
            self.cells.remove(0);
        }
    }

    pub fn cells(&self) -> &[(usize, usize)] {
        &self.cells
    }

    pub fn cap(&self) -> usize {
        self.cap
    }
}

/// The S4 soft anti-circuit penalty: subtract `delta` from every move that
/// would LAND on a cell the stack recently departed **and still owns** —
/// all four approach directions, full and half channels — for as long as
/// the cell stays in the trail window.
///
/// Why this shape: the measured oscillation is the argmax itself preferring
/// the return leg at a ~1.9-logit median margin, and taxing only the exact
/// inverse arc displaced the walk into period-4 circles (observed on the
/// seed-1 diagnostic). Any circuit of period ≤ window must land on its own
/// trail, while a forward march never does — so the tax detects a forming
/// circle at its closing move, with zero detector lag, and leaves
/// non-revisiting play untouched.
///
/// Two guards are load-bearing, each with a unit test and a plant:
///
/// * **Ownership** — departing always leaves army behind, so a trail cell
///   stays ours unless the enemy takes it, and **retaking a lost cell is
///   combat, not shuffling**. The tax lifts the moment `raw`'s owned
///   channel drops, so the retake fights at full logit.
/// * **Encirclement** — a source whose every other passable exit is also
///   taxed trail (mountains, the board edge, or trail on all remaining
///   sides) has only the step back. **The only way out is never taxed**,
///   so a stack boxed in by hills retreats at full logit instead of
///   dithering against a flat tax for a whole window.
///
/// A twice-departed cell is taxed once, not stacked — stacking is the
/// counter semantics the removed Python penalty was criticised for. Masked
/// entries may be taxed too — harmless, legality is the mask's call and
/// −1e9 stays −1e9 for every practical `delta`.
pub fn apply_trail_penalty(
    logits: &mut [f32],
    trail: &[(usize, usize)],
    delta: f32,
    raw: &[f32],
    h: usize,
    w: usize,
) {
    if delta <= 0.0 {
        return;
    }
    // The taxed set: still-owned trail cells, deduped.
    let mut taxed: Vec<(usize, usize)> = Vec::with_capacity(trail.len());
    for &(r, c) in trail {
        if r < h && c < w && raw[CH_OWNED * h * w + r * w + c] != 0.0 && !taxed.contains(&(r, c)) {
            taxed.push((r, c));
        }
    }
    for &(r, c) in &taxed {
        for (d, &(dr, dc)) in MOVE_DIRS.iter().enumerate() {
            let (sr, sc) = (r as i32 - dr, c as i32 - dc);
            // A source outside the board can never legally move; skip it.
            if sr < 0 || sr >= h as i32 || sc < 0 || sc >= w as i32 {
                continue;
            }
            let mut has_free_exit = false;
            for &(er, ec) in &MOVE_DIRS {
                let (nr, nc) = (sr + er, sc + ec);
                if nr < 0 || nr >= h as i32 || nc < 0 || nc >= w as i32 {
                    continue;
                }
                let (nru, ncu) = (nr as usize, nc as usize);
                if raw[CH_MOUNTAINS * h * w + nru * w + ncu] != 0.0 {
                    continue;
                }
                if taxed.contains(&(nru, ncu)) {
                    continue;
                }
                has_free_exit = true;
                break;
            }
            if !has_free_exit {
                continue;
            }
            let src = sr as usize * PAD + sc as usize;
            logits[d * CELLS + src] -= delta;
            logits[(d + 4) * CELLS + src] -= delta;
        }
    }
}

/// splitmix64's golden-ratio increment, reused to fold the turn into the
/// per-turn seed.
const TURN_SALT: u64 = 0x9E3779B97F4A7C15;
/// Arbitrary odd 64-bit constant that spreads the flat action index across
/// the word before the finalizer mixes it.
const INDEX_SALT: u64 = 0xA24BAED4963EE407;

/// splitmix64's output finalizer (Steele et al.), the crate's one hash. Its
/// job is decorrelating near-identical inputs, which is exactly what the
/// digest fold, the turn fold, and the per-index draw all need.
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// One u64 for the frame's board content — everything the engine sent
/// except the turn counter. The turn stays out on purpose: it enters the
/// draw exactly once, in [`select_action`]'s seed, so a revisit's fresh draw
/// is a property a unit test can hold the board still for.
pub fn board_digest(obs: &Observation) -> u64 {
    let mut d = splitmix64(((obs.h as u64) << 32) | obs.w as u64);
    for &s in &[obs.my_land, obs.my_army, obs.opp_land, obs.opp_army] {
        d = splitmix64(d ^ s as u32 as u64);
    }
    for grid in [&obs.type_grid, &obs.owner_grid, &obs.army_grid] {
        for &v in grid {
            d = splitmix64(d ^ v as u32 as u64);
        }
    }
    d
}

/// Natural log of a finite positive normal f64, from the crate's own
/// polynomial: decompose `x = m · 2^e` with `m ∈ [√½, √2)`, then
/// `ln m = 2 atanh(t)` for `t = (m−1)/(m+1)`, `|t| ≤ 0.172`. Seven series
/// terms leave ~1e-12 relative error — noise needs the distribution, not
/// the last ULP, but it does need the same bits on every host, which libm
/// does not promise.
fn ln_pos(x: f64) -> f64 {
    debug_assert!(x > 0.0 && x.is_finite());
    let bits = x.to_bits();
    let mut e = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | (1023u64 << 52));
    if m > core::f64::consts::SQRT_2 {
        m *= 0.5;
        e += 1;
    }
    let t = (m - 1.0) / (m + 1.0);
    let s = t * t;
    let p = 2.0
        + s * (2.0 / 3.0
            + s * (2.0 / 5.0
                + s * (2.0 / 7.0
                    + s * (2.0 / 9.0 + s * (2.0 / 11.0 + s * (2.0 / 13.0))))));
    t * p + e as f64 * core::f64::consts::LN_2
}

/// The standard Gumbel draw for one action index: `−ln(−ln u)` with `u`
/// hashed from the per-turn seed and the flat index.
fn gumbel(seed: u64, index: usize) -> f64 {
    let h = splitmix64(seed ^ (index as u64).wrapping_mul(INDEX_SALT));
    // 52 hashed bits centred in (0, 1): u ∈ [2⁻⁵³, 1 − 2⁻⁵³], every value
    // exactly representable and never 0 or 1 — either endpoint would send a
    // `ln` to ±∞. The noise therefore spans [−3.61, 36.74].
    let u = ((h >> 12) as f64 + 0.5) * (1.0 / 4503599627370496.0);
    -ln_pos(-ln_pos(u))
}

/// The played selection: first-max `argmax_i(logit_i + T · G_i)` over the
/// legal entries — an exact sample of `softmax(logits / T)` restricted to
/// the mask. `temperature ≤ 0` is the plain argmax (the pre-S1 bot, since a
/// masked logit carries −1e9 and can never beat the always-legal pass).
///
/// `penalties` is `prepare_action_mask`'s plane: 0 for legal, −1e9 for not.
/// Legality is read from it and never from the logits, which merely happen
/// to have the same penalties added inside the forward.
pub fn select_action(
    logits: &[f32],
    penalties: &[f32],
    turn: i32,
    digest: u64,
    temperature: f32,
) -> usize {
    debug_assert_eq!(logits.len(), penalties.len());
    let t = f64::from(temperature.max(0.0));
    let seed = splitmix64(digest ^ (turn as u32 as u64).wrapping_mul(TURN_SALT));
    let mut best = usize::MAX;
    let mut best_score = f64::NEG_INFINITY;
    for (i, (&logit, &penalty)) in logits.iter().zip(penalties).enumerate() {
        if penalty != 0.0 {
            continue;
        }
        let mut score = f64::from(logit);
        if t > 0.0 {
            score += t * gumbel(seed, i);
        }
        if score > best_score {
            best_score = score;
            best = i;
        }
    }
    // Unreachable while `prepare_action_mask` keeps the pass channel open,
    // but the selection layer must not be the reason a seat panics mid-game.
    if best == usize::MAX {
        return argmax(logits);
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open(n: usize) -> Vec<f32> {
        vec![0.0; n]
    }

    /// A full-pad board raw where every cell is ours (h = w = PAD, so board
    /// and pad coordinates coincide).
    fn owned_raw() -> Vec<f32> {
        let mut raw = vec![0.0f32; 14 * CELLS];
        raw[CH_OWNED * CELLS..(CH_OWNED + 1) * CELLS].fill(1.0);
        raw
    }

    #[test]
    fn select_is_deterministic() {
        let logits: Vec<f32> = (0..64).map(|i| (i % 7) as f32 * 0.3).collect();
        let mask = open(64);
        let a = select_action(&logits, &mask, 17, 0xDEAD_BEEF, 1.0);
        let b = select_action(&logits, &mask, 17, 0xDEAD_BEEF, 1.0);
        assert_eq!(a, b);
    }

    /// Kills the `select-turn-blind` plant: same board, equal logits — only
    /// the turn moves, and the pick must move with it. A turn-blind seed
    /// repeats one draw forever, which is the limit cycle restored.
    #[test]
    fn a_new_turn_is_a_fresh_draw() {
        let logits = open(100);
        let mask = open(100);
        let picks: Vec<usize> = (0..12)
            .map(|turn| select_action(&logits, &mask, turn, 42, 1.0))
            .collect();
        assert!(
            picks.iter().any(|&p| p != picks[0]),
            "12 turns, one pick ({}) — the draw ignored the turn",
            picks[0]
        );
    }

    /// Kills the `select-masked-noise` plant. The logits here deliberately
    /// do NOT carry the −1e9 the forward bakes in: legality must be the
    /// mask's call alone, so a masked entry stays unselectable even when its
    /// logit towers over every legal one.
    #[test]
    fn a_masked_action_is_never_selected() {
        let mut logits = open(32);
        logits[0] = 100.0;
        let mut mask = open(32);
        mask[0] = -1e9;
        for turn in 0..200 {
            for &t in &[1.0f32, 5.0] {
                let pick = select_action(&logits, &mask, turn, 7, t);
                assert_ne!(pick, 0, "turn {turn} T {t}: noise lifted a masked action");
            }
        }
    }

    #[test]
    fn zero_temperature_is_the_plain_argmax() {
        let logits: Vec<f32> = (0..441).map(|i| ((i * 37) % 100) as f32 * 0.01).collect();
        let mask = open(441);
        for turn in 0..8 {
            assert_eq!(select_action(&logits, &mask, turn, 5, 0.0), argmax(&logits));
        }
    }

    /// Kills the `select-temperature-scale` plant. At T = 0.01 the noise
    /// spans [−0.036, +0.37], so a margin of 1.0 cannot flip — a bound, not
    /// luck. Unscaled noise spans [−3.61, +36.74] and flips a 1.0 margin on
    /// roughly a quarter of turns, so 256 turns cannot all hold.
    #[test]
    fn a_tiny_temperature_cannot_flip_a_decisive_margin() {
        let mut logits = open(16);
        logits[3] = 1.0;
        let mask = open(16);
        for turn in 0..256 {
            assert_eq!(select_action(&logits, &mask, turn, 99, 0.01), 3);
        }
    }

    /// The Gumbel-max property itself, and the kill for the
    /// `select-noise-sign` plant: at T = 1 the pick frequencies over logits
    /// `(ln 6, ln 3, 0)` must track softmax = (0.6, 0.3, 0.1). Subtracted
    /// noise samples a demonstrably different distribution — (0.635, 0.306,
    /// 0.060) by inclusion–exclusion — and 8000 deterministic draws put its
    /// third-action count near 476, far outside the band around 800.
    #[test]
    fn draw_frequencies_match_the_softmax() {
        let logits = [6.0f32.ln(), 3.0f32.ln(), 0.0];
        let mask = open(3);
        let mut counts = [0usize; 3];
        for turn in 0..8000 {
            counts[select_action(&logits, &mask, turn, 0xC0FFEE, 1.0)] += 1;
        }
        assert!(
            (4560..=5040).contains(&counts[0]) && (680..=920).contains(&counts[2]),
            "counts {counts:?} do not match softmax (4800, 2400, 800) of 8000"
        );
    }

    /// The digest reads the board and the scalars, never the turn — the
    /// turn's one entry point is `select_action`'s seed.
    #[test]
    fn the_digest_reads_the_board_not_the_turn() {
        let mut a = Observation::with_dims(3, 3);
        a.my_army = 12;
        a.army_grid[4] = 7;
        let mut b = Observation::with_dims(3, 3);
        b.my_army = 12;
        b.army_grid[4] = 7;
        b.turn = 500;
        assert_eq!(board_digest(&a), board_digest(&b));
        b.army_grid[4] = 8;
        assert_ne!(board_digest(&a), board_digest(&b));
    }

    /// Kills the `trail-approach` and `trail-sign` plants: with one trail
    /// cell, the penalty must land on exactly the eight entries that move
    /// INTO it (four approach sources × full/half), with a minus sign, and
    /// nowhere else.
    #[test]
    fn trail_taxes_exactly_the_landing_arcs() {
        let mut logits = vec![0.0f32; 10 * CELLS];
        apply_trail_penalty(&mut logits, &[(5, 6)], 2.0, &owned_raw(), PAD, PAD);
        let mut expected = std::collections::HashSet::new();
        for (d, (dr, dc)) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)].iter().enumerate() {
            let src = (5 - dr) as usize * PAD + (6 - dc) as usize;
            expected.insert(d * CELLS + src);
            expected.insert((d + 4) * CELLS + src);
        }
        for (i, &v) in logits.iter().enumerate() {
            if expected.contains(&i) {
                assert_eq!(v, -2.0, "index {i} must carry the penalty");
            } else {
                assert_eq!(v, 0.0, "index {i} must stay untouched");
            }
        }
    }

    /// A twice-departed cell is taxed once — stacking would be the counter
    /// semantics the removed Python penalty was criticised for.
    #[test]
    fn a_twice_departed_cell_is_taxed_once() {
        let mut logits = vec![0.0f32; 10 * CELLS];
        apply_trail_penalty(&mut logits, &[(5, 6), (7, 7), (5, 6)], 2.0, &owned_raw(), PAD, PAD);
        assert_eq!(logits[0 * CELLS + 6 * PAD + 6], -2.0); // UP into (5,6)
    }

    /// Kills the `trail-ownership` plant, and pins the retake rule: a trail
    /// cell the enemy has captured is combat ground, not shuffle ground —
    /// landing on it must fight at full logit while a still-owned trail
    /// cell stays taxed.
    #[test]
    fn a_lost_trail_cell_is_free_to_retake() {
        let mut raw = owned_raw();
        raw[CH_OWNED * CELLS + 5 * PAD + 6] = 0.0; // the enemy took (5,6)
        let mut logits = vec![0.0f32; 10 * CELLS];
        apply_trail_penalty(&mut logits, &[(5, 6), (7, 7)], 2.0, &raw, PAD, PAD);
        assert_eq!(logits[0 * CELLS + 6 * PAD + 6], 0.0, "retake of (5,6) must be free");
        assert_eq!(logits[0 * CELLS + 8 * PAD + 7], -2.0, "owned (7,7) stays taxed");
    }

    /// Kills the `trail-exemption` plant, and pins the encirclement rule:
    /// a source boxed in by mountains, with the trail cell as its only
    /// passable exit, steps back at full logit — the only way out is never
    /// taxed. An open source approaching the same trail cell stays taxed.
    #[test]
    fn an_encircled_step_back_stays_free() {
        let mut raw = owned_raw();
        // Source (5,5): mountains above, below, and to the left; the only
        // exit is RIGHT into the trail cell (5,6).
        for (mr, mc) in [(4, 5), (6, 5), (5, 4)] {
            raw[CH_MOUNTAINS * CELLS + mr * PAD + mc] = 1.0;
        }
        let mut logits = vec![0.0f32; 10 * CELLS];
        apply_trail_penalty(&mut logits, &[(5, 6)], 2.0, &raw, PAD, PAD);
        let boxed_src = 5 * PAD + 5;
        assert_eq!(logits[3 * CELLS + boxed_src], 0.0, "the only way out must be free");
        assert_eq!(logits[7 * CELLS + boxed_src], 0.0, "the half variant too");
        let open_src = 6 * PAD + 6; // approaches (5,6) from below, exits free
        assert_eq!(logits[0 * CELLS + open_src], -2.0, "an open approach stays taxed");
    }

    /// Kills the `trail-window` plant: the ring must hold every recent
    /// source, or a period-4 circle's closing move goes untaxed. Walk the
    /// square (5,5)→(5,6)→(6,6)→(6,5) and check the close back onto (5,5)
    /// is taxed while a fresh outward move is not.
    #[test]
    fn a_four_step_circle_is_taxed_at_the_close() {
        let mut trail = Trail::new(8);
        for cell in [(5, 5), (5, 6), (6, 6), (6, 5)] {
            trail.push(cell);
        }
        assert_eq!(trail.cells().len(), 4);
        let mut logits = vec![0.0f32; 10 * CELLS];
        apply_trail_penalty(&mut logits, trail.cells(), 2.0, &owned_raw(), PAD, PAD);
        // Closing move: UP (dir 0) from (6,5) into (5,5) — taxed.
        assert_eq!(logits[0 * CELLS + 6 * PAD + 5], -2.0);
        // Fresh outward move: DOWN (dir 1) from (6,5) into (7,5) — free.
        assert_eq!(logits[1 * CELLS + 6 * PAD + 5], 0.0);
    }

    #[test]
    fn the_ring_forgets_only_the_oldest() {
        let mut trail = Trail::new(2);
        trail.push((1, 1));
        trail.push((2, 2));
        trail.push((3, 3));
        assert_eq!(trail.cells(), &[(2, 2), (3, 3)]);
        assert_eq!(Trail::new(0).cells().len(), 0);
        let mut off = Trail::new(0);
        off.push((1, 1));
        assert_eq!(off.cells().len(), 0);
    }

    /// The sizing property end to end: a taxed return leg loses a margin
    /// smaller than delta (T = 0 makes it exact), delta 0 restores it.
    #[test]
    fn a_taxed_return_loses_the_near_margin() {
        let ret = 2 * CELLS + 5 * PAD + 6; // LEFT from (5,6) into (5,5)
        let alt = 3 * CELLS + 5 * PAD + 6; // RIGHT from (5,6), fresh ground
        let mask = open(10 * CELLS);
        for (delta, want) in [(2.0f32, alt), (0.0, ret)] {
            let mut logits = vec![-5.0f32; 10 * CELLS];
            logits[ret] = 0.5;
            logits[alt] = 0.0;
            apply_trail_penalty(&mut logits, &[(5, 5)], delta, &owned_raw(), PAD, PAD);
            assert_eq!(select_action(&logits, &mask, 7, 9, 0.0), want);
        }
    }

    #[test]
    fn ln_pos_matches_libm() {
        let mut x = 2f64.powi(-53);
        while x < 40.0 {
            let err = (ln_pos(x) - x.ln()).abs();
            let tol = x.ln().abs().max(1.0) * 1e-12;
            assert!(err <= tol, "ln_pos({x}) = {} vs libm {}", ln_pos(x), x.ln());
            x *= 1.37;
        }
    }
}
