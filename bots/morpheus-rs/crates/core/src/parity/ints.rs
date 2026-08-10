//! The integer stream, and the one number that leaves it.
//!
//! **Format: a flat stream of integers, not JSON.** The plan says "canonical
//! JSON", and this is a deliberate, narrow deviation. Rust's standard library
//! has no JSON, so the choice was a dependency in the *shipped* binary or two
//! hundred lines of hand-rolled parser — for a machine-to-machine channel
//! whose entire payload is integers. The 10,000-file unpacked limit is the
//! binding sandbox constraint (rewrite-plan §1), the crate is zero-dependency
//! to protect it, and integers compare bit-exactly with no float formatting to
//! argue about. Reading is the same `parse_ints` the wire protocol uses.
//!
//! Floats travel as raw bit patterns, never as decimal text: a float that
//! round-trips through formatting is a float whose last bits depend on two
//! languages agreeing about printing, which is exactly the argument tier-2
//! parity is trying not to have.

use std::io::BufRead;

/// Positional reader over the whitespace-separated integer stream.
pub struct Ints {
    values: Vec<i64>,
    at: usize,
}

impl Ints {
    pub fn read_all<R: BufRead>(reader: &mut R) -> Result<Self, String> {
        let mut text = String::new();
        reader
            .read_to_string(&mut text)
            .map_err(|e| format!("reading cases: {e}"))?;
        let mut values = Vec::new();
        for token in text.split_ascii_whitespace() {
            values.push(
                token
                    .parse::<i64>()
                    .map_err(|_| format!("not an integer: {token:?}"))?,
            );
        }
        Ok(Self { values, at: 0 })
    }

    pub fn next(&mut self) -> Result<i64, String> {
        let value = *self
            .values
            .get(self.at)
            .ok_or_else(|| format!("case stream ended after {} integers", self.at))?;
        self.at += 1;
        Ok(value)
    }

    pub(in crate::parity) fn n(&mut self) -> Result<usize, String> {
        Ok(self.next()? as usize)
    }

    pub(in crate::parity) fn ints(&mut self, count: usize) -> Result<Vec<i32>, String> {
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(self.next()? as i32);
        }
        Ok(out)
    }

    /// Floats travel as their raw `f32` bit patterns, as integers.
    ///
    /// Not as decimal text: a float that round-trips through formatting is a
    /// float whose last bits depend on two languages agreeing about printing,
    /// which is exactly the argument tier-2 parity is trying not to have. Bit
    /// patterns make the channel lossless and leave the tolerance decision
    /// entirely to the comparison.
    pub(in crate::parity) fn floats(&mut self, count: usize) -> Result<Vec<f32>, String> {
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(f32::from_bits(self.next()? as u32));
        }
        Ok(out)
    }

    /// f64 the same way `floats` reads f32: raw bit patterns, as integers.
    pub(in crate::parity) fn f64s(&mut self, count: usize) -> Result<Vec<f64>, String> {
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(f64::from_bits(self.next()? as u64));
        }
        Ok(out)
    }

    pub(in crate::parity) fn action(&mut self) -> Result<[i32; 5], String> {
        Ok([
            self.next()? as i32,
            self.next()? as i32,
            self.next()? as i32,
            self.next()? as i32,
            self.next()? as i32,
        ])
    }
}

/// `i64` to text without pulling in a formatting dependency.
pub(in crate::parity) fn itoa(value: i64) -> String {
    value.to_string()
}
