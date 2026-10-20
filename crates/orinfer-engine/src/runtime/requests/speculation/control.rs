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

/// Spend a fixed verification-row budget on request prefixes. The estimates
/// come exclusively from earlier committed rounds, never the current drafts.
/// Every increment is a legal package width, including packages with holes.
pub(in crate::runtime) fn allocate(
    choices: &[Vec<(usize, f64)>],
    budget: usize,
) -> Option<Vec<usize>> {
    let mut indices = vec![0; choices.len()];
    let mut used = choices
        .iter()
        .try_fold(0usize, |n, c| n.checked_add(c.first()?.0))?;
    if used > budget {
        return None;
    }
    while used < budget {
        let next = choices
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let current = c[indices[i]];
                let next = *c.get(indices[i] + 1)?;
                let rows = next.0.checked_sub(current.0)?;
                (rows > 0 && rows <= budget - used).then_some((
                    i,
                    rows,
                    (next.1 - current.1) / rows as f64,
                ))
            })
            // Lower request index wins ties; comparisons use frozen estimates.
            .max_by(|a, b| a.2.total_cmp(&b.2).then_with(|| b.0.cmp(&a.0)))?;
        used += next.1;
        indices[next.0] += 1;
    }
    Some(choices.iter().zip(indices).map(|(c, i)| c[i].0).collect())
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
            .chain(
                candidates
                    .iter()
                    .map(|c| c.prior_seconds)
                    .filter(|s| s.is_finite() && *s > 0. && *s <= slack),
            )
            .fold(normal_seconds * 4., f64::max);
        cohort.probe_credit = (cohort.probe_credit + normal_seconds * 0.02).min(credit_cap);
        let bootstrap = candidates
            .iter()
            .filter(|c| c.width >= 2)
            .map(|c| c.width)
            .min();
        let cold_pending = candidates.iter().any(|c| {
            c.width >= 2
                && c.expected_tokens > 0.
                && c.prior_seconds.is_finite()
                && c.prior_seconds > 0.
                && c.prior_seconds <= slack
                && cohort
                    .arms
                    .get(&c.width)
                    .is_none_or(|a| a.time.samples == 0)
        });
        let mut best = None;
        let mut probe = None;
        for candidate in candidates {
            if candidate.width < 2 || candidate.expected_tokens <= 0. {
                continue;
            }
            let arm = cohort.arms.get(&candidate.width);
            let seconds = arm.map_or(candidate.prior_seconds, |a| a.time.upper());
            if !seconds.is_finite() || seconds <= 0. || seconds > slack {
                continue;
            }
            let samples = arm.map_or(0, |a| a.time.samples);
            let score = arm.map_or(f64::INFINITY, |a| {
                a.time.mean / candidate.expected_tokens.min(a.tokens).max(1.)
            });
            // Bootstrap the cheapest depth even with an uncertain cold prior.
            // Other cold depths must either predict a win or spend accumulated
            // time credit; a short cohort cannot afford a free sweep of all arms.
            // Include cold priors in credit_cap so pessimistic estimates never
            // permanently prevent discovering a profitable depth.
            let initial = samples == 0
                && (Some(candidate.width) == bootstrap
                    || candidate.prior_seconds / candidate.expected_tokens < baseline);
            let promising = samples < 3 && score < baseline * 1.1;
            let stale = arm.is_some_and(|a| cohort.calls.saturating_sub(a.last_trial) >= 32);
            // Cheap stale losers must not repeatedly consume all credit before
            // a more expensive, still unseen depth gets its first measurement.
            let retry = (stale || samples == 0)
                && (samples == 0 || !cold_pending)
                && cohort.probe_credit >= seconds;
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
    #[test]
    fn row_budget_favors_accepting_prefixes_and_respects_caps_and_holes() {
        let choices = vec![
            vec![(2, 1.9), (3, 2.7), (4, 3.4)],
            vec![(2, 1.2), (3, 1.22), (4, 1.221)],
        ];
        assert_eq!(allocate(&choices, 6), Some(vec![4, 2]));
        assert_eq!(allocate(&choices, 8), Some(vec![4, 4]));
        assert_eq!(allocate(&choices, 9), None);
        assert_eq!(allocate(&choices, 3), None);
        assert_eq!(allocate(&[vec![(2, 1.8), (4, 2.9)]], 3), None);
        assert_eq!(allocate(&[vec![(2, 1.8), (4, 2.9)]], 4), Some(vec![4]));
    }
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
    #[test]
    fn cold_expensive_depths_wait_for_credit_without_becoming_unreachable() {
        let mut c = Controller::default();
        let k = key(1);
        normal(&mut c, k);
        let plans = [
            Candidate {
                width: 2,
                expected_tokens: 1.5,
                prior_seconds: 0.04,
            },
            Candidate {
                width: 3,
                expected_tokens: 1.6,
                prior_seconds: 0.2,
            },
        ];
        assert_eq!(c.choose(k, &plans, 0.02, 1.), Some((2, true)));
        c.observe(k, 2, 0.04, 1);
        for _ in 0..20 {
            assert_eq!(c.choose(k, &plans, 0.02, 1.), None);
        }
        let mut discovered = false;
        for _ in 0..1000 {
            match c.choose(k, &plans, 0.02, 1.) {
                Some((2, _)) => c.observe(k, 2, 0.04, 1),
                Some((3, _)) => {
                    discovered = true;
                    break;
                }
                _ => (),
            }
        }
        assert!(
            discovered,
            "Cold prior must not cap credit below its trial cost"
        );
    }
}
