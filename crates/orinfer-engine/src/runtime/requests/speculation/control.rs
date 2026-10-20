//! Runtime decisions only: kernels and legal row geometries remain package-owned.
use crate::scheduler::Estimate;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Default)]
pub(in crate::runtime) struct Acceptance {
    // Conditional acceptance at each actually tested prefix position. A rejected
    // prefix does not turn its untested suffix into independent rejections.
    probability: [f64; 7],
    samples: [usize; 7],
}
impl Acceptance {
    pub(in crate::runtime) fn observe(&mut self, proposed: usize, accepted: usize) {
        for j in 0..proposed.min(accepted.saturating_add(1)).min(7) {
            let value = f64::from(j < accepted);
            self.probability[j] = if self.samples[j] == 0 {
                value
            } else {
                self.probability[j] * 0.8 + value * 0.2
            };
            self.samples[j] += 1;
        }
    }
    pub(in crate::runtime) fn expected(&self, drafts: usize) -> f64 {
        let mut prefix = 1.;
        let mut total = 1.;
        for j in 0..drafts.min(7) {
            // Shrink sparse request observations toward a neutral model prior.
            let confidence = self.samples[j] as f64 / (self.samples[j] as f64 + 4.);
            prefix *= self.probability[j] * confidence + 0.7 * (1. - confidence);
            total += prefix;
        }
        total
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::runtime) struct Key {
    pub batch: usize,
    pub context: u32,
    pub sampling: bool,
    pub prefill: usize,
}
#[derive(Default)]
struct Arm {
    time: Estimate,
    tokens: f64,
    last_trial: usize,
}
#[derive(Default)]
struct Cohort {
    arms: BTreeMap<usize, Arm>,
    calls: usize,
    active: usize,
    probe_credit: f64,
}
#[derive(Default)]
pub(in crate::runtime) struct Controller {
    cohorts: BTreeMap<Key, Cohort>,
    #[cfg(test)]
    pub(in crate::runtime) fixed_width: Option<usize>,
}
#[derive(Clone, Copy)]
pub(in crate::runtime) struct Candidate {
    pub width: usize,
    pub expected_tokens: f64,
    pub prior_seconds: f64,
}
impl Controller {
    fn retain_key(&mut self, key: Key) {
        if self.cohorts.len() >= 256 && !self.cohorts.contains_key(&key) {
            let oldest = self.cohorts.keys().next().copied().unwrap();
            self.cohorts.remove(&oldest);
        }
    }
    pub(in crate::runtime) fn observe(
        &mut self,
        key: Key,
        width: usize,
        seconds: f64,
        tokens: usize,
    ) {
        if !seconds.is_finite() || seconds <= 0. || tokens == 0 {
            return;
        }
        self.retain_key(key);
        let cohort = self.cohorts.entry(key).or_default();
        let arm = cohort.arms.entry(width).or_default();
        arm.time.observe(seconds);
        arm.tokens = if arm.time.samples == 1 {
            tokens as f64
        } else {
            arm.tokens * 0.8 + tokens as f64 * 0.2
        };
        arm.last_trial = cohort.calls;
    }
    pub(in crate::runtime) fn choose(
        &mut self,
        key: Key,
        candidates: &[Candidate],
        normal_seconds: f64,
        slack: f64,
    ) -> Option<(usize, bool)> {
        #[cfg(test)]
        if let Some(width) = self.fixed_width {
            return candidates
                .iter()
                .any(|c| c.width == width)
                .then_some((width, false));
        }
        self.retain_key(key);
        let cohort = self.cohorts.entry(key).or_default();
        cohort.calls += 1;
        let normal = cohort.arms.get(&1)?;
        if normal.time.samples < 3 {
            return None;
        }
        let normal_seconds = normal
            .time
            .mean
            .max(normal_seconds.min(normal.time.upper()));
        let baseline = normal_seconds / key.batch.max(1) as f64;
        let credit_cap = cohort
            .arms
            .values()
            .map(|a| a.time.upper())
            .fold(normal_seconds * 4., f64::max);
        cohort.probe_credit = (cohort.probe_credit + normal_seconds * 0.02).min(credit_cap);
        let mut best = None;
        let mut probe = None;
        for candidate in candidates {
            if candidate.width < 2 || candidate.expected_tokens <= 0. {
                continue;
            }
            let arm = cohort.arms.get(&candidate.width);
            let seconds = arm.map_or(candidate.prior_seconds, |a| a.time.upper());
            if !seconds.is_finite() || seconds > slack {
                continue;
            }
            let samples = arm.map_or(0, |a| a.time.samples);
            let score = arm.map_or(f64::INFINITY, |a| {
                a.time.mean / candidate.expected_tokens.min(a.tokens).max(1.)
            });
            // One bounded initial trial, then collect at least three samples
            // for plausible winners. Expensive losers are periodically revisited
            // using accumulated time credit, including while ordinary decode wins.
            let initial = samples == 0;
            let promising = samples < 3 && score < baseline * 1.1;
            let stale = arm.is_some_and(|a| cohort.calls.saturating_sub(a.last_trial) >= 32);
            let retry = stale && cohort.probe_credit >= seconds;
            if (initial || promising || retry) && probe.is_none() {
                probe = Some((candidate.width, if retry { seconds } else { 0. }));
            }
            let margin = arm.map_or(1., |a| {
                ((a.time.upper() - a.time.mean)
                    / a.time.mean.max(1e-9)
                    / (samples.max(1) as f64).sqrt())
                .clamp(0.01, 0.1)
            });
            let threshold = if cohort.active == candidate.width {
                1.
            } else {
                1. - margin
            };
            if samples >= 3
                && score < baseline * threshold
                && best.is_none_or(|(_, cost)| score < cost)
            {
                best = Some((candidate.width, score));
            }
        }
        if let Some((width, charged)) = probe {
            cohort.probe_credit -= charged;
            return Some((width, true));
        }
        cohort.active = best.map_or(0, |(width, _)| width);
        best.map(|(width, _)| (width, false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn key(batch: usize) -> Key {
        Key {
            batch,
            context: 8,
            sampling: false,
            prefill: 0,
        }
    }
    fn normal(c: &mut Controller, k: Key) {
        for _ in 0..3 {
            c.observe(k, 1, 0.02, k.batch);
        }
    }
    #[test]
    fn acceptance_tracks_tested_prefix_and_never_invents_suffix_samples() {
        let mut a = Acceptance::default();
        a.observe(3, 1);
        assert_eq!(a.samples, [1, 1, 0, 0, 0, 0, 0]);
        assert!(a.expected(2) > 1. && a.expected(2) < 3.);
        a.observe(1, 1);
        assert_eq!(a.samples[1], 1);
    }
    #[test]
    fn selects_goodput_and_retains_ordinary_as_a_real_candidate() {
        let mut c = Controller::default();
        let k = key(4);
        normal(&mut c, k);
        let plans = [
            Candidate {
                width: 2,
                expected_tokens: 7.,
                prior_seconds: 0.03,
            },
            Candidate {
                width: 3,
                expected_tokens: 9.,
                prior_seconds: 0.04,
            },
        ];
        assert_eq!(c.choose(k, &plans, 0.02, 0.1), Some((2, true)));
        for _ in 0..3 {
            c.observe(k, 2, 0.03, 7);
            c.observe(k, 3, 0.05, 9);
        }
        assert_eq!(c.choose(k, &plans, 0.02, 0.1), Some((2, false)));
        let low = [Candidate {
            expected_tokens: 4.2,
            ..plans[0]
        }];
        assert_eq!(c.choose(k, &low, 0.02, 0.1), None);
        assert_eq!(c.choose(k, &plans, 0.02, 0.001), None);
    }
    #[test]
    fn disabled_plans_reprobe_and_batch_changes_have_independent_costs() {
        let mut c = Controller::default();
        let k = key(1);
        normal(&mut c, k);
        let plans = [Candidate {
            width: 2,
            expected_tokens: 1.5,
            prior_seconds: 0.04,
        }];
        c.observe(k, 2, 0.04, 1);
        assert!((0..150).any(|_| c.choose(k, &plans, 0.02, 0.1) == Some((2, true))));
        assert_eq!(c.choose(key(128), &plans, 0.2, 1.), None);
        c.observe(k, 2, f64::NAN, 1);
        assert_eq!(c.cohorts[&k].arms[&2].time.samples, 1);
    }
    #[test]
    fn slow_rejected_arm_can_accumulate_enough_credit_for_a_retry() {
        let mut c = Controller::default();
        let k = key(1);
        normal(&mut c, k);
        let plans = [Candidate {
            width: 2,
            expected_tokens: 1.5,
            prior_seconds: 0.2,
        }];
        c.observe(k, 2, 0.2, 1);
        assert!((0..1000).any(|_| c.choose(k, &plans, 0.02, 1.) == Some((2, true))));
    }
}
