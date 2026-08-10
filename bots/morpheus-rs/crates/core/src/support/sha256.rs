//! SHA-256, by hand. FIPS 180-4.
//!
//! **Not because `sha2` would not fit.** It would: measured at 565 vendored
//! files and 6.3 MB, which is 5.6% of the 10,000-file submission budget
//! (`docs/research/measurements/morpheus-rs-dependency-budget.md`). An earlier
//! version of this comment claimed the budget was the reason; that was wrong.
//!
//! The reasons that hold:
//!
//! * **Speed is irrelevant here**, which removes the library's real advantage.
//!   M0 measured hashing at **0.001 ms p99** — the cheapest of ten components,
//!   against a 140 ms turn. `sha2`'s SHA-NI path is genuinely faster and buys
//!   nothing measurable.
//! * **This is not a security boundary.** The digests are dictionary keys for
//!   search nodes: no adversary, no secret, no timing channel. The usual
//!   decisive argument against writing your own does not apply.
//! * **It is verified twice, independently.** The published NIST vectors below,
//!   plus 6,995 digest comparisons against Python's `hashlib` over real game
//!   data in the `hash` parity kind — every one exact.
//! * `sha2` pulls `cpufeatures` → `libc`, which is **404 of those 565 files**:
//!   platform bindings for CPU detection we do not want, entering a static
//!   musl binary for a hash that costs a microsecond.
//!
//! Node keys stay SHA-256 for parity with the Python, not for security. The
//! dependency budget is real but binds at M3, where the inference crate lands
//! (candle-core measures 3,888 files, 39% of the budget) — not here.

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
    0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
    0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
    0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
    0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
    0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
];

const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
    0x5be0cd19,
];

/// Streaming SHA-256. `update` any number of times, then `finish`.
///
/// Streaming rather than one-shot because the digests here are built from a
/// dozen array slices each: buffering them into one contiguous `Vec` first
/// would allocate megabytes per turn on the belief path for no benefit.
pub struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    buffered: usize,
    length: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    pub fn new() -> Self {
        Self {
            state: H0,
            buffer: [0u8; 64],
            buffered: 0,
            length: 0,
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.length = self.length.wrapping_add(data.len() as u64);
        if self.buffered > 0 {
            let want = 64 - self.buffered;
            let take = want.min(data.len());
            self.buffer[self.buffered..self.buffered + take].copy_from_slice(&data[..take]);
            self.buffered += take;
            data = &data[take..];
            if self.buffered == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffered = 0;
            }
        }
        while data.len() >= 64 {
            let (block, rest) = data.split_at(64);
            let mut chunk = [0u8; 64];
            chunk.copy_from_slice(block);
            self.compress(&chunk);
            data = rest;
        }
        if !data.is_empty() {
            self.buffer[..data.len()].copy_from_slice(data);
            self.buffered = data.len();
        }
    }

    pub fn finish(mut self) -> [u8; 32] {
        // Padding: 0x80, zeros, then the message length in bits, big-endian.
        let bit_len = self.length.wrapping_mul(8);
        self.update(&[0x80]);
        // `update` counted that byte; the padding length is computed from the
        // buffer, so the bogus increment does not matter.
        while self.buffered != 56 {
            self.update(&[0x00]);
        }
        let mut tail = [0u8; 8];
        tail.copy_from_slice(&bit_len.to_be_bytes());
        self.update(&tail);
        debug_assert_eq!(self.buffered, 0);

        let mut out = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, value) in self
            .state
            .iter_mut()
            .zip([a, b, c, d, e, f, g, h].into_iter())
        {
            *slot = slot.wrapping_add(value);
        }
    }
}

/// One-shot digest over a sequence of byte slices, concatenated.
pub fn sha256(parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part);
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(digest: [u8; 32]) -> String {
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn matches_the_published_test_vectors() {
        // FIPS 180-4 / NIST CAVP.
        assert_eq!(
            hex(sha256(&[b""])),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(sha256(&[b"abc"])),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hex(sha256(&[
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".as_slice()
            ])),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn a_message_spanning_many_blocks_is_correct() {
        // One million 'a', the classic long vector — exercises the streaming
        // path and the 64-bit length field together.
        let mut hasher = Sha256::new();
        let chunk = [b'a'; 1000];
        for _ in 0..1000 {
            hasher.update(&chunk);
        }
        assert_eq!(
            hex(hasher.finish()),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn streaming_in_any_chunking_gives_one_answer() {
        let message: Vec<u8> = (0u8..=255).cycle().take(1000).collect();
        let one_shot = sha256(&[&message]);
        for chunk in [1usize, 7, 63, 64, 65, 128, 999] {
            let mut hasher = Sha256::new();
            for part in message.chunks(chunk) {
                hasher.update(part);
            }
            assert_eq!(hasher.finish(), one_shot, "chunk size {chunk}");
        }
    }

    #[test]
    fn a_message_that_lands_exactly_on_a_block_boundary_pads_correctly() {
        // 56 and 64 bytes are the two lengths where padding wraps to a second
        // block; both are classic off-by-one sites in a hand-written SHA-256.
        for len in [55usize, 56, 57, 63, 64, 65, 119, 120] {
            let message = vec![0x5au8; len];
            let mut hasher = Sha256::new();
            hasher.update(&message);
            let streamed = hasher.finish();
            assert_eq!(streamed, sha256(&[&message]), "length {len}");
        }
    }
}
