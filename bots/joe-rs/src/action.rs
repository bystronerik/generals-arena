//! Action codec over the 10-channel head — mirror of `decode_action` /
//! `encode_action` in the Python sibling. Channels 0–3 full move, 4–7 half
//! move, 8 pass, 9 build; flat index is `channel * 441 + row * 21 + col`.

use crate::obs::{CELLS, PAD};

/// Engine action tuple `[pass_field, row, col, direction, is_half]` where
/// `pass_field` is 0=move, 1=pass, 2=build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Action5 {
    pub pass_field: i32,
    pub row: i32,
    pub col: i32,
    pub dir: i32,
    pub is_half: i32,
}

/// Flat logit index -> engine action array (`decode_action`).
pub fn decode_action(idx: usize) -> Action5 {
    let d = idx / CELLS;
    let pos = idx % CELLS;
    let (r, c) = (pos / PAD, pos % PAD);
    let is_pass = d == 8;
    let is_build = d == 9;
    let is_half = (4..8).contains(&d);
    let pf = if is_build { 2 } else if is_pass { 1 } else { 0 };
    let ad = if is_half { d - 4 } else if d < 4 { d } else { 0 };
    Action5 {
        pass_field: pf,
        row: r as i32,
        col: c as i32,
        dir: ad as i32,
        is_half: is_half as i32,
    }
}

/// Engine action array -> flat logit index (`encode_action`). Play never
/// encodes (greedy decode only); this is the codec's other half, kept for
/// the round-trip test.
#[cfg_attr(not(test), allow(dead_code))]
pub fn encode_action(a: Action5) -> usize {
    let ed = if a.pass_field == 2 {
        9
    } else if a.pass_field == 1 {
        8
    } else if a.is_half > 0 {
        (a.dir + 4) as usize
    } else {
        a.dir as usize
    };
    ed * CELLS + a.row as usize * PAD + a.col as usize
}

/// First-max argmax over the flat logits — matches `jnp.argmax`, which
/// returns the first occurrence of the maximum.
pub fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_round_trip() {
        for idx in 0..(10 * CELLS) {
            let a = decode_action(idx);
            assert_eq!(encode_action(a), idx, "idx {idx} decoded to {a:?}");
        }
    }

    #[test]
    fn decode_examples() {
        // Build at (2, 3).
        let a = decode_action(9 * CELLS + 2 * PAD + 3);
        assert_eq!(a, Action5 { pass_field: 2, row: 2, col: 3, dir: 0, is_half: 0 });
        // Half move DOWN (dir 1) at (0, 5) — channel 5.
        let a = decode_action(5 * CELLS + 5);
        assert_eq!(a, Action5 { pass_field: 0, row: 0, col: 5, dir: 1, is_half: 1 });
        // Pass.
        let a = decode_action(8 * CELLS);
        assert_eq!(a.pass_field, 1);
    }

    #[test]
    fn argmax_is_first_max() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
    }
}
