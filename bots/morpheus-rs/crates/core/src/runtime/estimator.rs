//! Forecasting a component's cost, and naming how far play degraded.
//!
//! Admission control needs a number before the work runs, so every component
//! carries a rolling nearest-rank p99 with a deterministic warm-up: until `W`
//! local samples exist the forecast is the maximum of the offline
//! qualification p99 and everything seen locally, which is the conservative
//! direction. [`FallbackLevel`] is the other half — the band the committed
//! action actually came from.

/// Which degradation band produced the committed action.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum FallbackLevel {
    /// No root result.
    Pass,
    /// Zero completed simulations.
    Policy,
    /// Retired band, kept for old-trace schema compatibility. Three selectors
    /// flipping by simulation count measured 220 flips in a 573-turn game.
    Visit,
    /// One or more completed simulations.
    Average,
}

impl FallbackLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            FallbackLevel::Pass => "pass",
            FallbackLevel::Policy => "policy",
            FallbackLevel::Visit => "visit",
            FallbackLevel::Average => "average",
        }
    }
}

/// Nearest-rank empirical 99th percentile, one-indexed rank `ceil(0.99 n)`.
pub fn nearest_rank_p99(samples: &[f64]) -> Result<f64, String> {
    if samples.is_empty() {
        return Err("nearest-rank p99 needs at least one sample".to_string());
    }
    let mut ordered = samples.to_vec();
    ordered.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = (0.99 * samples.len() as f64).ceil() as usize;
    Ok(ordered[rank - 1])
}

/// Rolling nearest-rank p99 with a deterministic warm-up.
///
/// Before `W` local samples exist the forecast is the maximum of the offline
/// qualification p99 and every observed local sample; after that it is the
/// nearest-rank empirical p99 of the bounded window.
pub struct NearestRankP99Estimator {
    pub window: usize,
    pub offline_p99_ms: f64,
    samples: std::collections::VecDeque<f64>,
}

impl NearestRankP99Estimator {
    pub fn new(window: usize, offline_p99_ms: f64) -> Result<Self, String> {
        if window < 1 {
            return Err("p99 window must be >= 1".to_string());
        }
        if !offline_p99_ms.is_finite() {
            // A non-finite seed poisons every forecast and silently locks the
            // component out of admission forever.
            return Err("offline_p99_ms seed must be finite".to_string());
        }
        Ok(Self {
            window,
            offline_p99_ms,
            samples: std::collections::VecDeque::with_capacity(window),
        })
    }

    pub fn n_samples(&self) -> usize {
        self.samples.len()
    }

    pub fn warmed_up(&self) -> bool {
        self.samples.len() >= self.window
    }

    pub fn observe(&mut self, ms: f64) {
        if self.samples.len() == self.window {
            self.samples.pop_front();
        }
        self.samples.push_back(ms);
    }

    pub fn forecast(&self) -> f64 {
        if self.samples.is_empty() {
            return self.offline_p99_ms;
        }
        if !self.warmed_up() {
            let local = self
                .samples
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            return self.offline_p99_ms.max(local);
        }
        let values: Vec<f64> = self.samples.iter().copied().collect();
        nearest_rank_p99(&values).expect("a warmed-up window is non-empty")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_nearest_rank_p99_is_the_ceil_rank_not_an_interpolation() {
        let samples: Vec<f64> = (1..=100).map(|v| v as f64).collect();
        assert_eq!(nearest_rank_p99(&samples).unwrap(), 99.0);
        assert_eq!(nearest_rank_p99(&[5.0]).unwrap(), 5.0);
        assert!(nearest_rank_p99(&[]).is_err());
    }

    #[test]
    fn a_cold_estimator_forecasts_its_offline_seed() {
        let est = NearestRankP99Estimator::new(4, 12.0).unwrap();
        assert_eq!(est.forecast(), 12.0);
    }

    #[test]
    fn a_warming_estimator_takes_the_max_of_the_seed_and_the_samples() {
        let mut est = NearestRankP99Estimator::new(4, 12.0).unwrap();
        est.observe(3.0);
        assert_eq!(est.forecast(), 12.0);
        est.observe(40.0);
        assert_eq!(est.forecast(), 40.0);
        assert!(!est.warmed_up());
    }

    #[test]
    fn a_warmed_estimator_uses_the_window_and_forgets_the_seed() {
        let mut est = NearestRankP99Estimator::new(4, 1000.0).unwrap();
        for value in [1.0, 2.0, 3.0, 4.0] {
            est.observe(value);
        }
        assert!(est.warmed_up());
        assert_eq!(est.forecast(), 4.0);
        // The window is bounded: the oldest sample falls out.
        est.observe(0.5);
        assert_eq!(est.forecast(), 4.0);
    }

    #[test]
    fn a_non_finite_seed_is_refused_rather_than_forecast() {
        assert!(NearestRankP99Estimator::new(4, f64::INFINITY).is_err());
        assert!(NearestRankP99Estimator::new(0, 1.0).is_err());
    }
}
