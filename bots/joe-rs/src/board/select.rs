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
use crate::io::wire::Observation;

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
