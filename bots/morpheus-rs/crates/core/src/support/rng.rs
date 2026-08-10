//! The injected random source, and the two arithmetic details NumPy hides.
//!
//! Port of the draw sites in `bots/morpheus/belief.py`, `proposal.py`,
//! `recovery.py` and `reservoir.py`. Nothing here reimplements NumPy's bit
//! generator: rewrite-plan §5 decided that the parity harness **records** the
//! oracle's draws and replays them, so what has to match is the *sequence of
//! calls*, not the numbers. [`Replay`] enforces exactly that — it checks the
//! method and every argument of each draw before handing back the recorded
//! result — which turns draw-site order into a checked part of the port rather
//! than a hope. A Rust path that samples one extra time, or samples from a
//! population of a different size, fails on the spot instead of quietly
//! diverging three decisions later.
//!
//! [`SmallRng`] is what plays. It only has to be a decent PRNG; it is never
//! compared against NumPy.
//!
//! Two NumPy behaviours *are* reproduced here, because they change results:
//!
//! * [`npsum`] — `np.sum` is not a left-to-right loop. It is pairwise, and
//!   the difference is visible at the particle counts this bot uses: over
//!   eight `f64` weights the two answers differ in the last bits, and the ESS
//!   they feed decides whether a resample happens at all.
//! * [`argsort_desc_numpy`] — `np.argsort` is an *unstable* introsort, and
//!   `top_legal_actions` sorts a uniform distribution where every legal action
//!   ties. Which candidates recovery tries, and in which order, is therefore
//!   decided entirely by the sort's internals. See that function's note for
//!   the part of this that is not portable.

// ---------------------------------------------------------------- the trait

/// Which NumPy generator method a recorded draw came from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Method {
    Integers,
    Choice,
    Random,
}

impl Method {
    pub fn from_code(code: i64) -> Option<Self> {
        match code {
            0 => Some(Method::Integers),
            1 => Some(Method::Choice),
            2 => Some(Method::Random),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Method::Integers => "integers",
            Method::Choice => "choice",
            Method::Random => "random",
        }
    }
}

/// The generator surface the belief filter actually uses.
///
/// One method per NumPy call site, with NumPy's own signature shape —
/// `size: None` means "a scalar", which is a different recorded draw from
/// `size: Some(1)` even though both yield one number. Keeping the distinction
/// is what lets [`Replay`] tell a faithful port from a rearranged one.
pub trait Rng {
    fn integers(&mut self, low: i64, high: i64, size: Option<usize>) -> Vec<i64>;
    fn choice(&mut self, n: usize, size: Option<usize>, replace: bool, p: Option<&[f64]>)
        -> Vec<i64>;
    fn random(&mut self, size: Option<usize>) -> Vec<f64>;

    /// `rng.integers(low, high)` — the scalar form.
    fn integer(&mut self, low: i64, high: i64) -> i64 {
        self.integers(low, high, None)[0]
    }

    /// `rng.choice(n, p=...)` — the scalar form.
    fn choice_one(&mut self, n: usize, p: Option<&[f64]>) -> usize {
        self.choice(n, None, true, p)[0] as usize
    }

    /// `rng.random()` — the scalar form.
    fn random_one(&mut self) -> f64 {
        self.random(None)[0]
    }

    /// How many recorded draws have been taken. Zero for a real PRNG.
    ///
    /// The parity surfaces emit it, because a port that consumed a *different
    /// number* of draws and still produced the same answers has agreed by
    /// accident. Behind the trait rather than on `Replay` so a surface holding
    /// a `SharedRng` can still ask.
    fn consumed(&self) -> usize {
        0
    }
}

// ------------------------------------------------------------------- replay

/// One recorded draw: what was asked for, and what NumPy answered.
#[derive(Clone, Debug)]
pub struct RecordedDraw {
    pub method: Method,
    /// `integers`: low/high. `choice`: population size in `a`.
    pub a: i64,
    pub b: i64,
    pub size: Option<usize>,
    pub replace: bool,
    pub weighted: bool,
    pub ints: Vec<i64>,
    pub floats: Vec<f64>,
}

/// Replays a recorded stream, refusing any call that does not match it.
///
/// The refusal is a panic, deliberately. This runs under the parity
/// subcommand, never in a match: a mismatch means the port asked for a
/// different random number than the oracle did, which is a porting bug the
/// harness must not be able to absorb into a "close enough" comparison. The
/// message names the draw index and both signatures, which is enough to find
/// the site without a debugger.
pub struct Replay {
    draws: Vec<RecordedDraw>,
    at: usize,
}

impl Replay {
    pub fn new(draws: Vec<RecordedDraw>) -> Self {
        Self { draws, at: 0 }
    }

    pub fn consumed(&self) -> usize {
        self.at
    }

    pub fn remaining(&self) -> usize {
        self.draws.len() - self.at
    }

    fn next(&mut self, method: Method, want: String) -> RecordedDraw {
        if self.at >= self.draws.len() {
            panic!(
                "replay exhausted after {} draw(s): the port asked for {want} and the \
                 oracle drew nothing more",
                self.draws.len()
            );
        }
        let draw = self.draws[self.at].clone();
        self.at += 1;
        if draw.method != method {
            panic!(
                "draw {}: the port asked for {want} but the oracle drew {}",
                self.at - 1,
                draw.method.name()
            );
        }
        draw
    }
}

fn size_of(size: Option<usize>) -> String {
    match size {
        None => "None".to_string(),
        Some(n) => n.to_string(),
    }
}

impl Rng for Replay {
    fn integers(&mut self, low: i64, high: i64, size: Option<usize>) -> Vec<i64> {
        let want = format!("integers(low={low}, high={high}, size={})", size_of(size));
        let draw = self.next(Method::Integers, want.clone());
        if draw.a != low || draw.b != high || draw.size != size {
            panic!(
                "draw {}: the port asked for {want}, the oracle drew integers(low={}, \
                 high={}, size={})",
                self.at - 1,
                draw.a,
                draw.b,
                size_of(draw.size)
            );
        }
        draw.ints
    }

    fn choice(
        &mut self,
        n: usize,
        size: Option<usize>,
        replace: bool,
        p: Option<&[f64]>,
    ) -> Vec<i64> {
        let weighted = p.is_some();
        let want = format!(
            "choice(n={n}, size={}, replace={replace}, weighted={weighted})",
            size_of(size)
        );
        let draw = self.next(Method::Choice, want.clone());
        if draw.a != n as i64
            || draw.size != size
            || draw.replace != replace
            || draw.weighted != weighted
        {
            panic!(
                "draw {}: the port asked for {want}, the oracle drew choice(n={}, \
                 size={}, replace={}, weighted={})",
                self.at - 1,
                draw.a,
                size_of(draw.size),
                draw.replace,
                draw.weighted
            );
        }
        draw.ints
    }

    fn consumed(&self) -> usize {
        self.at
    }

    fn random(&mut self, size: Option<usize>) -> Vec<f64> {
        let want = format!("random(size={})", size_of(size));
        let draw = self.next(Method::Random, want.clone());
        if draw.size != size {
            panic!(
                "draw {}: the port asked for {want}, the oracle drew random(size={})",
                self.at - 1,
                size_of(draw.size)
            );
        }
        draw.floats
    }
}

// --------------------------------------------------------------- play PRNG

/// xoshiro256++ seeded through SplitMix64 — what the bot plays with.
///
/// Deliberately not a NumPy clone. Play-time draws are never compared against
/// the oracle (rewrite-plan §5: "record and replay, do not re-implement
/// NumPy"), so the only requirements are that it is fast, has no state larger
/// than four words, and does not correlate across the streams the filter pulls
/// from it.
pub struct SmallRng {
    s: [u64; 4],
}

impl SmallRng {
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Self {
            s: [next(), next(), next(), next()],
        }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, bound)` without modulo bias (Lemire).
    fn below(&mut self, bound: u64) -> u64 {
        if bound == 0 {
            return 0;
        }
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next_u64();
            let product = (value as u128) * (bound as u128);
            if (product as u64) >= threshold {
                return (product >> 64) as u64;
            }
        }
    }

    fn unit(&mut self) -> f64 {
        // 53 bits of mantissa, the same construction NumPy uses.
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

impl Rng for SmallRng {
    fn integers(&mut self, low: i64, high: i64, size: Option<usize>) -> Vec<i64> {
        let span = (high - low).max(0) as u64;
        let count = size.unwrap_or(1);
        (0..count).map(|_| low + self.below(span) as i64).collect()
    }

    fn choice(
        &mut self,
        n: usize,
        size: Option<usize>,
        replace: bool,
        p: Option<&[f64]>,
    ) -> Vec<i64> {
        let count = size.unwrap_or(1);
        match p {
            // Weighted: NumPy's own recipe — normalized cumulative sum, then a
            // right-side search per uniform draw.
            Some(probs) => {
                let mut cdf = Vec::with_capacity(probs.len());
                let mut running = 0.0f64;
                for &value in probs {
                    running += value;
                    cdf.push(running);
                }
                let total = *cdf.last().unwrap_or(&0.0);
                if total > 0.0 {
                    for value in cdf.iter_mut() {
                        *value /= total;
                    }
                }
                (0..count)
                    .map(|_| {
                        let u = self.unit();
                        let mut index = cdf.partition_point(|&c| c <= u);
                        if index >= n {
                            index = n.saturating_sub(1);
                        }
                        index as i64
                    })
                    .collect()
            }
            None if replace => (0..count).map(|_| self.below(n as u64) as i64).collect(),
            // Without replacement: a partial Fisher-Yates over a scratch
            // permutation, which is O(count) draws rather than a reject loop.
            None => {
                let mut pool: Vec<i64> = (0..n as i64).collect();
                let take = count.min(n);
                for i in 0..take {
                    let j = i + self.below((n - i) as u64) as usize;
                    pool.swap(i, j);
                }
                pool.truncate(take);
                pool
            }
        }
    }

    fn random(&mut self, size: Option<usize>) -> Vec<f64> {
        (0..size.unwrap_or(1)).map(|_| self.unit()).collect()
    }
}

// --------------------------------------------------------------- sharing it

/// One generator, two owners.
///
/// `runtime.py` builds a single `np.random.Generator` and hands the *same
/// object* to `SearchController`, so the belief's draws and the search's draws
/// interleave in one stream. That is not incidental: the replay harness checks
/// draw order across the whole turn, and two independent generators would
/// produce a different sequence even from the same seed. Rust cannot lend one
/// `&mut` to two structs, so the sharing is explicit — a refcounted cell whose
/// borrow lasts exactly one call, which is safe here because the crate is
/// single-threaded by construction and no draw site re-enters another.
pub struct SharedRng(std::rc::Rc<std::cell::RefCell<Box<dyn Rng>>>);

impl SharedRng {
    pub fn new(inner: Box<dyn Rng>) -> Self {
        Self(std::rc::Rc::new(std::cell::RefCell::new(inner)))
    }

    pub fn handle(&self) -> Self {
        Self(std::rc::Rc::clone(&self.0))
    }
}

impl Clone for SharedRng {
    fn clone(&self) -> Self {
        self.handle()
    }
}

impl Rng for SharedRng {
    fn integers(&mut self, low: i64, high: i64, size: Option<usize>) -> Vec<i64> {
        self.0.borrow_mut().integers(low, high, size)
    }

    fn choice(
        &mut self,
        n: usize,
        size: Option<usize>,
        replace: bool,
        p: Option<&[f64]>,
    ) -> Vec<i64> {
        self.0.borrow_mut().choice(n, size, replace, p)
    }

    fn random(&mut self, size: Option<usize>) -> Vec<f64> {
        self.0.borrow_mut().random(size)
    }

    fn consumed(&self) -> usize {
        self.0.borrow().consumed()
    }
}

// ------------------------------------------------------------ numpy details

const PW_BLOCKSIZE: usize = 128;

/// `np.sum` over `f64`: pairwise, not sequential.
///
/// NumPy's reduction accumulates in eight interleaved lanes for arrays of 8 to
/// 128 elements and recurses in blocks above that, so a straight `iter().sum()`
/// disagrees in the last bits. That is not academic here: `ess()` divides two
/// of these sums and compares the result against a threshold, so a
/// last-bit difference can flip whether the filter resamples — and a resample
/// consumes a draw, which the replay stream would then reject. Getting this
/// exactly right is what keeps the two implementations on the same branch.
///
/// Transcribed from `pairwise_sum_@TYPE@` in NumPy's `loops_utils`, with the
/// same block boundaries.
pub fn npsum(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 8 {
        let mut res = 0.0;
        for &value in values {
            res += value;
        }
        return res;
    }
    if n <= PW_BLOCKSIZE {
        let mut r = [0.0f64; 8];
        r[..8].copy_from_slice(&values[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for (lane, slot) in r.iter_mut().enumerate() {
                *slot += values[i + lane];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += values[i];
            i += 1;
        }
        return res;
    }
    let mut half = n / 2;
    half -= half % 8;
    npsum(&values[..half]) + npsum(&values[half..])
}

/// `np.argsort(-scores)` — NumPy's introsort, transcribed.
///
/// `top_legal_actions` ranks the enemy proposal by descending probability. On
/// the deployed configuration that distribution is **uniform**, so every legal
/// action carries the same float and the entire ordering is decided by how the
/// sort breaks ties. NumPy's default `kind="quicksort"` is an unstable
/// introsort — median-of-three quicksort, insertion sort under sixteen
/// elements, heapsort past a depth limit of `2·floor(log2 n)` — and its tie
/// order matches neither a stable sort nor anything else one would write from
/// scratch. Reproducing it is the only way the recovery paths pick the same
/// candidates as the oracle.
///
/// **This is host-conditional, and that is a property of the oracle, not of
/// the port.** NumPy ≥ 2.0 dispatches `argsort` on 64-bit dtypes to
/// `x86-simd-sort` when the CPU has AVX-512-SKX, which is a different
/// (also unstable) algorithm with a different tie order. The M0 CPU probe
/// found Modal hosts both with and without AVX-512, so on some x86 hosts the
/// *Python bot itself* would order these candidates differently. The parity
/// harness has an `argsort` surface that compares this function against
/// `np.argsort` directly, so a host that dispatches elsewhere fails loudly
/// with a named cause instead of surfacing as a recovery mismatch.
pub fn argsort_desc_numpy(scores: &[f64]) -> Vec<usize> {
    // NumPy sorts `-scores` ascending. Negating and sorting ascending is the
    // same comparison tree as sorting `scores` descending only if the
    // comparison is negated everywhere *and* nothing depends on the sign of a
    // zero — it does not, `<` treats -0.0 and 0.0 as equal. Negate the values
    // instead of the comparator so the transcription stays literal.
    let negated: Vec<f64> = scores.iter().map(|&v| -v).collect();
    let mut order: Vec<usize> = (0..negated.len()).collect();
    aquicksort(&negated, &mut order);
    order
}

/// `a < b`, with NumPy's NaN ordering (NaN sorts last).
#[inline]
fn lt(a: f64, b: f64) -> bool {
    a < b || (b.is_nan() && !a.is_nan())
}

const SMALL_QUICKSORT: usize = 15;

fn aquicksort(v: &[f64], tosort: &mut [usize]) {
    let num = tosort.len();
    if num == 0 {
        return;
    }
    // `npy_get_msb`: floor(log2(num)).
    let mut msb = 0i32;
    let mut u = num;
    while {
        u >>= 1;
        u
    } > 0
    {
        msb += 1;
    }
    let mut cdepth = msb * 2;

    // Index-based transcription of NumPy's pointer walk. `pl`/`pr` are the
    // inclusive bounds of the partition being worked on.
    let mut pl: usize = 0;
    let mut pr: usize = num - 1;
    let mut stack: Vec<(usize, usize, i32)> = Vec::new();

    loop {
        if cdepth < 0 {
            aheapsort(v, &mut tosort[pl..=pr]);
        } else {
            while pr > pl && pr - pl > SMALL_QUICKSORT {
                let pm = pl + ((pr - pl) >> 1);
                if lt(v[tosort[pm]], v[tosort[pl]]) {
                    tosort.swap(pm, pl);
                }
                if lt(v[tosort[pr]], v[tosort[pm]]) {
                    tosort.swap(pr, pm);
                }
                if lt(v[tosort[pm]], v[tosort[pl]]) {
                    tosort.swap(pm, pl);
                }
                let vp = v[tosort[pm]];
                let mut pi = pl;
                let mut pj = pr - 1;
                tosort.swap(pm, pj);
                loop {
                    loop {
                        pi += 1;
                        if !lt(v[tosort[pi]], vp) {
                            break;
                        }
                    }
                    loop {
                        pj -= 1;
                        if !lt(vp, v[tosort[pj]]) {
                            break;
                        }
                    }
                    if pi >= pj {
                        break;
                    }
                    tosort.swap(pi, pj);
                }
                tosort.swap(pi, pr - 1);
                cdepth -= 1;
                // Push the larger side, recurse into the smaller — the bound
                // on stack depth, and on which half is finished first.
                if pi - pl < pr - pi {
                    stack.push((pi + 1, pr, cdepth));
                    pr = pi - 1;
                } else {
                    stack.push((pl, pi - 1, cdepth));
                    pl = pi + 1;
                }
            }

            // Insertion sort finishes every partition.
            let mut pi = pl + 1;
            while pi <= pr {
                let vi = tosort[pi];
                let vp = v[vi];
                let mut pj = pi;
                while pj > pl && lt(vp, v[tosort[pj - 1]]) {
                    tosort[pj] = tosort[pj - 1];
                    pj -= 1;
                }
                tosort[pj] = vi;
                pi += 1;
            }
        }

        match stack.pop() {
            Some((l, r, depth)) => {
                pl = l;
                pr = r;
                cdepth = depth;
            }
            None => break,
        }
    }
}

/// NumPy's `aheapsort`, the introsort depth-limit fallback.
fn aheapsort(v: &[f64], tosort: &mut [usize]) {
    let mut n = tosort.len();
    if n < 2 {
        return;
    }
    // NumPy offsets the array by one for 1-based heap indexing; `at` does the
    // same without an out-of-range pointer.
    let at = |a: &[usize], i: usize| a[i - 1];

    let mut l = n >> 1;
    while l > 0 {
        let tmp = at(tosort, l);
        let mut i = l;
        let mut j = l << 1;
        while j <= n {
            if j < n && lt(v[at(tosort, j)], v[at(tosort, j + 1)]) {
                j += 1;
            }
            if lt(v[tmp], v[at(tosort, j)]) {
                tosort[i - 1] = at(tosort, j);
                i = j;
                j += j;
            } else {
                break;
            }
        }
        tosort[i - 1] = tmp;
        l -= 1;
    }

    while n > 1 {
        let tmp = at(tosort, n);
        tosort[n - 1] = at(tosort, 1);
        n -= 1;
        let mut i = 1;
        let mut j = 2;
        while j <= n {
            if j < n && lt(v[at(tosort, j)], v[at(tosort, j + 1)]) {
                j += 1;
            }
            if lt(v[tmp], v[at(tosort, j)]) {
                tosort[i - 1] = at(tosort, j);
                i = j;
                j += j;
            } else {
                break;
            }
        }
        tosort[i - 1] = tmp;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_summation_differs_from_a_plain_loop_at_eight() {
        // The smallest size where NumPy switches to eight lanes. If these two
        // ever agree for every input, this function has stopped earning its
        // existence.
        let values: Vec<f64> = (0..8).map(|i| 1.0 / (i as f64 + 3.0)).collect();
        let sequential: f64 = values.iter().sum();
        assert_ne!(npsum(&values), sequential);
    }

    #[test]
    fn pairwise_summation_is_exact_on_representable_inputs() {
        let values = vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0];
        assert_eq!(npsum(&values), 511.0);
        assert_eq!(npsum(&[]), 0.0);
    }

    #[test]
    fn argsort_orders_by_descending_score() {
        let order = argsort_desc_numpy(&[0.1, 0.9, 0.5, -1.0]);
        assert_eq!(order[0], 1);
        assert_eq!(order[1], 2);
        assert_eq!(order[3], 3);
    }

    #[test]
    fn argsort_is_a_permutation_at_every_size_that_crosses_a_branch() {
        // 15 is the insertion-sort ceiling, 16 the first quicksort partition,
        // and 3970 is the real action space.
        for n in [0usize, 1, 2, 15, 16, 17, 129, 3970] {
            let scores: Vec<f64> = (0..n).map(|i| ((i * 37) % 11) as f64).collect();
            let order = argsort_desc_numpy(&scores);
            assert_eq!(order.len(), n);
            let mut seen = vec![false; n];
            for &index in &order {
                assert!(!seen[index], "index {index} appears twice at n={n}");
                seen[index] = true;
            }
            for window in order.windows(2) {
                assert!(scores[window[0]] >= scores[window[1]], "n={n}");
            }
        }
    }

    #[test]
    fn a_uniform_distribution_still_sorts_to_a_permutation() {
        // The case that matters: every value ties, so the order is entirely
        // the algorithm's. Correctness here is only "it is a permutation";
        // agreeing with NumPy is the parity harness's `argsort` surface.
        let scores = vec![0.25f64; 3970];
        let order = argsort_desc_numpy(&scores);
        let mut seen = vec![false; scores.len()];
        for &index in &order {
            assert!(!seen[index]);
            seen[index] = true;
        }
    }

    #[test]
    fn replay_refuses_a_draw_the_oracle_did_not_make() {
        let mut rng = Replay::new(vec![RecordedDraw {
            method: Method::Integers,
            a: 0,
            b: 4,
            size: Some(2),
            replace: true,
            weighted: false,
            ints: vec![1, 3],
            floats: Vec::new(),
        }]);
        assert_eq!(rng.integers(0, 4, Some(2)), vec![1, 3]);
        assert_eq!(rng.remaining(), 0);
    }

    #[test]
    #[should_panic(expected = "the port asked for")]
    fn replay_refuses_a_different_population() {
        let mut rng = Replay::new(vec![RecordedDraw {
            method: Method::Choice,
            a: 8,
            b: 0,
            size: None,
            replace: true,
            weighted: true,
            ints: vec![3],
            floats: Vec::new(),
        }]);
        rng.choice(9, None, true, Some(&[1.0]));
    }

    #[test]
    fn the_play_generator_stays_inside_its_bounds() {
        let mut rng = SmallRng::seed_from_u64(7);
        for value in rng.integers(3, 9, Some(500)) {
            assert!((3..9).contains(&value));
        }
        for value in rng.random(Some(500)) {
            assert!((0.0..1.0).contains(&value));
        }
        let picked = rng.choice(10, Some(10), false, None);
        let mut sorted = picked.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), picked.len(), "sampling without replacement repeated");
    }

    #[test]
    fn weighted_choice_respects_a_degenerate_distribution() {
        let mut rng = SmallRng::seed_from_u64(11);
        let mut probs = vec![0.0f64; 6];
        probs[4] = 1.0;
        for value in rng.choice(6, Some(200), true, Some(&probs)) {
            assert_eq!(value, 4);
        }
    }
}
