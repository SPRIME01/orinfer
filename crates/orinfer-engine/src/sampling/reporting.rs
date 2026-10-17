//! Full target softmax reporting needs normalization and a small top-N, not a
//! sorted sampling CDF. Keep this independent of top-k/nucleus and MTP draws.
use super::{Options, TokenLogprob};

pub(super) struct Report {
    masses: Vec<f64>,
    total: f64,
    top: Vec<u32>,
}

impl Report {
    pub fn new(scores: &mut [(u32, f64)], vocab: usize, options: &Options) -> Self {
        let maximum = scores.iter().map(|s| s.1).max_by(f64::total_cmp).unwrap();
        let temperature = if options.temperature == 0.0 {
            1.0
        } else {
            options.temperature
        };
        let mut masses = vec![0.0; vocab];
        let mut total = 0.0;
        for &(id, score) in scores.iter() {
            let mass = ((score - maximum) / temperature).exp();
            masses[id as usize] = mass;
            total += mass;
        }
        let count = options.top_logprobs.unwrap_or(0).min(scores.len());
        let compare =
            |a: &(u32, f64), b: &(u32, f64)| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0));
        if count > 0 && count < scores.len() {
            scores.select_nth_unstable_by(count, compare);
        }
        let top = &mut scores[..count];
        top.sort_unstable_by(compare);
        Self {
            masses,
            total,
            top: top.iter().map(|s| s.0).collect(),
        }
    }

    pub fn record(self, token: u32) -> TokenLogprob {
        let log = |id: u32| {
            (self.masses.get(id as usize).copied().unwrap_or(0.0) / self.total)
                .ln()
                .max(-9999.0)
        };
        TokenLogprob {
            token,
            logprob: log(token),
            top: self.top.iter().map(|&id| (id, log(id))).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::{Decoder, Distribution, EVALUATION_SEED, counter_uniform};

    fn build_report(
        logits: &[f32],
        history: &[u32],
        options: &Options,
        mask: Option<&[u32]>,
    ) -> Result<Report, String> {
        let mut scores = crate::sampling::processed_scores(logits, history, options, mask)?;
        Ok(Report::new(&mut scores, logits.len(), options))
    }

    #[test]
    fn reporting_matches_full_target_law_without_changing_seeded_draws() {
        let logits: Vec<_> = (0..1027)
            .map(|i| ((i * 17 % 101) as f32 - 50.0) / 7.0)
            .collect();
        let history = [0, 0, 20, 1026, u32::MAX];
        let mask: Vec<_> = (0..logits.len().div_ceil(32))
            .map(|_| 0x5555_5555)
            .collect();
        for temperature in [0.0, 0.7, 2.0] {
            for top in [0, 1, 3, 20] {
                let options = Options {
                    temperature,
                    top_logprobs: Some(top),
                    top_k: 20,
                    top_p: 0.8,
                    repetition_penalty: 1.1,
                    presence_penalty: 0.2,
                    frequency_penalty: 0.3,
                    logit_bias: [(0, 2.0), (20, -3.0)].into(),
                    ..Default::default()
                };
                for mask in [None, Some(mask.as_slice())] {
                    let reference = Distribution::from_masked(
                        &logits,
                        &history,
                        &Options {
                            temperature: if temperature == 0.0 { 1.0 } else { temperature },
                            top_k: 0,
                            top_p: 1.0,
                            ..options.clone()
                        },
                        mask,
                    )
                    .unwrap();
                    for token in [0, 20, 1026] {
                        let report = build_report(&logits, &history, &options, mask)
                            .unwrap()
                            .record(token);
                        assert!(
                            (report.logprob - reference.probability(token).ln().max(-9999.0)).abs()
                                < 1e-11
                        );
                        let expected: Vec<_> = reference.ordered.iter().take(top).collect();
                        assert_eq!(report.top.len(), expected.len());
                        for ((id, logp), &&(other, p)) in report.top.iter().zip(&expected) {
                            assert_eq!(*id, other);
                            assert!((logp - p.ln()).abs() < 1e-11);
                        }
                    }
                }
                let mut decoder = Decoder::default();
                for k in [0, 1, 20, 1024] {
                    let options = Options {
                        top_k: k,
                        ..options.clone()
                    };
                    let law = decoder.law(&logits, &history, &options).unwrap();
                    let reference = Distribution::from_logits(&logits, &history, &options).unwrap();
                    assert_eq!(law.distribution.ordered, reference.ordered);
                    for step in 0..32 {
                        let u = counter_uniform(EVALUATION_SEED, 0, step);
                        assert_eq!(law.distribution.draw(u), reference.draw(u));
                    }
                }
            }
        }
    }

    #[test]
    fn reports_tail_probabilities_and_masked_vocabulary_ties() {
        let options = Options {
            top_logprobs: Some(20),
            ..Default::default()
        };
        let report = build_report(&[0.0, -50.0, -1000.0], &[], &options, None)
            .unwrap()
            .record(1);
        assert_eq!(report.logprob, -50.0);
        assert_eq!(report.top, [(0, 0.0), (1, -50.0), (2, -9999.0)]);
        let report = build_report(&[1.0; 33], &[], &options, Some(&[0b101, 1]))
            .unwrap()
            .record(1);
        assert_eq!(report.logprob, -9999.0);
        assert_eq!(
            report.top.iter().map(|x| x.0).collect::<Vec<_>>(),
            [0, 2, 32]
        );
        assert!(build_report(&[f32::NAN], &[], &options, None).is_err());
        assert!(build_report(&[0.0], &[], &options, Some(&[0])).is_err());
    }
}
