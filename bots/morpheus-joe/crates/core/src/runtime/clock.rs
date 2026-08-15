//! Time, behind a trait, so the deadline logic can be tested without one.
//!
//! [`MonotonicClock`] is what plays. [`FakeClock`] is what lets a test drive
//! the controller past its deadline in a microsecond, and `is_fake` exists
//! because a few paths must know the difference.

use std::time::Instant;

// ---------------------------------------------------------------- the clock

/// Seconds, monotonic. `advance` exists so a fake clock can be charged the
/// forecast instead of the wall time, which is what makes an admission
/// sequence reproducible across two languages.
pub trait Clock {
    fn now(&self) -> f64;
    fn advance(&mut self, _ms: f64) {}
    fn is_fake(&self) -> bool {
        false
    }
}

pub struct MonotonicClock {
    start: Instant,
}

impl Default for MonotonicClock {
    fn default() -> Self {
        Self {
            start: Instant::now(),
        }
    }
}

impl Clock for MonotonicClock {
    fn now(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
}

/// Injectable monotonic clock for deadline tests, in seconds.
#[derive(Default)]
pub struct FakeClock {
    pub t: f64,
}

impl Clock for FakeClock {
    fn now(&self) -> f64 {
        self.t
    }

    fn advance(&mut self, ms: f64) {
        self.t += ms / 1000.0;
    }

    fn is_fake(&self) -> bool {
        true
    }
}
