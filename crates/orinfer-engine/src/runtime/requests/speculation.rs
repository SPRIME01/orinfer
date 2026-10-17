//! Speculation advances one bounded verification round. Catch-up is lazy while
//! batched target decode is profitable; stale ring entries are never consumed.
use super::*;

#[derive(Default)]
pub(in crate::runtime) struct Profile {
    time: scheduler::Estimate,
    tokens: f64,
    trials: usize,
    last_progress: usize,
}
impl Profile {
    pub fn upper(&self) -> f64 {
        self.time.upper()
    }
    pub fn observe(&mut self, seconds: f64, tokens: usize, progress: usize) {
        if self.trials == 1 {
            // The first use may capture a graph or load a lazy CUDA module.
            // Keep it as a trial, but estimate steady execution from later uses.
            self.time = Default::default();
            self.tokens = 0.;
        }
        self.time.observe(seconds);
        self.tokens = if self.time.samples == 1 {
            tokens as f64
        } else {
            self.tokens * 0.8 + tokens as f64 * 0.2
        };
        self.trials += 1;
        self.last_progress = progress;
    }
    pub fn observe_steady(&mut self, seconds: f64, tokens: usize, progress: usize) {
        if !seconds.is_finite() || seconds <= 0. || tokens == 0 {
            return;
        }
        // Batched callers subtract module/capture time themselves. Retaining
        // the first measured steady iteration bounds expensive exploration.
        self.time.observe(seconds);
        self.tokens = if self.time.samples == 1 {
            tokens as f64
        } else {
            self.tokens * 0.8 + tokens as f64 * 0.2
        };
        self.trials += 1;
        self.last_progress = progress;
    }
    fn batch_score(&self) -> Option<f64> {
        (self.trials >= 1).then_some(self.time.upper() / self.tokens.max(1.))
    }
    pub fn score(&self) -> Option<f64> {
        (self.trials >= 3).then_some(self.time.mean / self.tokens.max(1.))
    }
    pub fn trials(&self) -> usize {
        self.trials
    }
}

pub(super) fn select_profile(
    plans: &[usize],
    costs: &BTreeMap<usize, Profile>,
    progress: usize,
    last_probe: usize,
    slack: f64,
    fallback_per_token: f64,
) -> Option<(usize, bool)> {
    let valid: Vec<_> = plans
        .iter()
        .copied()
        .filter(|n| {
            costs
                .get(n)
                .map_or(fallback_per_token * *n as f64, |p| p.time.upper())
                <= slack
        })
        .collect();
    if let Some(rows) = valid
        .iter()
        .copied()
        .filter(|n| costs.get(n).is_none_or(|p| p.score().is_none()))
        .max()
    {
        return Some((rows, true));
    }
    // Count committed tokens, including ordinary decode: otherwise switching
    // away from speculation freezes the clock and can prevent recovery when
    // reasoning turns into code. Peer deadlines still bound every probe.
    if progress.saturating_sub(last_probe) >= 64 {
        return valid
            .into_iter()
            .min_by_key(|n| costs[n].last_progress)
            .map(|n| (n, true));
    }
    valid
        .into_iter()
        .min_by(|a, b| {
            costs[a]
                .score()
                .unwrap()
                .total_cmp(&costs[b].score().unwrap())
        })
        .map(|n| (n, false))
}

pub(super) enum Preparation {
    Finished(Vec<u32>),
    Round(PreparedRound),
}
pub(super) struct PreparedRound {
    pub(super) plan: crate::mtp::VerificationPlan,
    position: usize,
    remaining: usize,
    vocab: usize,
    host_sampling: bool,
    is_lookup: bool,
    drafts: Vec<u32>,
    proposals: Vec<crate::mtp::Proposal>,
    pub(super) verification: Vec<u32>,
}

struct CommittedRound {
    tokens: Vec<u32>,
    plan: crate::mtp::VerificationPlan,
    position: usize,
    is_lookup: bool,
}

enum DraftPreparation {
    Finished(Vec<u32>),
    Draft(Box<DraftRound>),
}
struct DraftRound {
    round: PreparedRound,
    decoder: crate::sampling::Decoder,
    history: Vec<u32>,
    done: bool,
    failed: bool,
    started: Instant,
}
pub(super) struct BatchChoice {
    pub(super) selected: Vec<(usize, usize)>,
    pub(super) exploring: bool,
}

impl ModelRuntime {
    pub(super) fn select_mixed_speculation(
        &self,
        requests: &[&mut RequestState],
        joint: &[(usize, usize)],
        options: &scheduler::Options,
        slack: f64,
    ) -> Option<Vec<(usize, usize)>> {
        let spec = self
            .manifest
            .mtp
            .as_ref()
            .filter(|s| !s.mixed_verification_tokens.is_empty())?;
        let decoders = joint
            .iter()
            .filter(|(i, _)| !requests[*i].prefilling)
            .count();
        if !(1..=8).contains(&decoders) || !joint.iter().any(|(i, _)| requests[*i].prefilling) {
            return None;
        }
        let cap = self
            .manifest
            .prefill_batch_profiles
            .iter()
            .map(|p| p.tokens)
            .max()?;
        let baseline = self.predict_iteration(requests, joint) / decoders as f64;
        for width in spec
            .mixed_verification_tokens
            .iter()
            .copied()
            .rev()
            .filter(|&n| {
                n <= spec.default_verification_tokens && n * decoders <= options.max_batch_tokens
            })
        {
            let selected: Vec<_> = joint
                .iter()
                .map(|&(i, n)| (i, if requests[i].prefilling { n } else { width }))
                .collect();
            if selected.iter().map(|(_, n)| n).sum::<usize>() > cap
                || selected
                    .iter()
                    .map(|&(i, n)| if requests[i].prefilling { 1 } else { n })
                    .sum::<usize>()
                    > *self.manifest.batch_profiles.iter().max().unwrap_or(&1)
                || selected.iter().any(|&(i, n)| {
                    !requests[i].prefilling && n > requests[i].limit - requests[i].generated
                })
            {
                continue;
            }
            let key = Self::iteration_key(requests, &selected);
            let measured = self.mixed_speculation_costs.get(&key);
            // A cold trial includes an explicit drafter allowance; measured
            // full rounds include proposal, verification, commit and warm.
            let upper = measured.map_or_else(
                || {
                    self.predict_iteration(requests, &selected)
                        + self.mtp_seconds_per_token * (width - 1) as f64 * decoders as f64
                },
                Profile::upper,
            );
            if upper > slack
                || measured
                    .and_then(Profile::batch_score)
                    .is_some_and(|s| s >= baseline)
            {
                continue;
            }
            return Some(selected);
        }
        None
    }

    pub(super) fn speculative_mixed_step(
        &mut self,
        requests: &mut [&mut RequestState],
        selected: &[(usize, usize)],
        output: &mut Vec<StepOutput>,
    ) -> Result<()> {
        let (prefills, decoders): (Vec<_>, Vec<_>) = selected
            .iter()
            .copied()
            .partition(|&(i, _)| requests[i].prefilling);
        if prefills.is_empty()
            || decoders.is_empty()
            || prefills
                .iter()
                .any(|&(i, _)| requests[i].prefill_frame.is_some())
        {
            return Err("Invalid mixed speculative work".into());
        }
        self.speculative_packet_step(requests, &decoders, &prefills, output)
    }

    pub(super) fn select_speculative_batch(
        &self,
        requests: &[&mut RequestState],
        decode: &[usize],
        options: &scheduler::Options,
        normal: f64,
        normal_mean: f64,
    ) -> Option<BatchChoice> {
        let spec = self.manifest.mtp.as_ref()?;
        let limit = (options.max_batch_tokens / decode.len()).min(spec.default_verification_tokens);
        let slack = decode
            .iter()
            .filter_map(|&i| requests[i].decode_slack_s(options.target_tpot_ms))
            .fold(f64::INFINITY, f64::min);
        let progress: usize = decode.iter().map(|&i| requests[i].generated).sum();
        let last_probe: usize = decode
            .iter()
            .map(|&i| requests[i].speculation_probe_token)
            .sum();
        let probe_due = progress.saturating_sub(last_probe) >= 64 * decode.len();
        let mut best = None;
        let mut probe = None;
        let mut seen = BTreeSet::new();
        for width in spec
            .batch_verification_tokens
            .iter()
            .copied()
            .filter(|&n| n <= limit)
            .rev()
        {
            let selected: Vec<_> = decode
                .iter()
                .map(|&i| {
                    spec.batch_verification_tokens
                        .iter()
                        .copied()
                        .filter(|&n| {
                            !requests[i].mtp_blocked
                                && n <= width
                                && n <= requests[i].limit - requests[i].generated
                                && n <= self.manifest.max_context
                                    - requests[i].history.len().saturating_sub(1)
                        })
                        .max()
                        .map(|n| (i, n))
                })
                .collect::<Option<_>>()?;
            let key = Self::batch_speculation_key(requests, &selected);
            if !seen.insert(key) {
                continue;
            }
            let profile = self.batch_speculation_costs.get(&key);
            // Crossing a context bucket does not erase a measured rejection.
            // Revisit it on the shared probe clock, rather than restarting all
            // lengths at every power-of-two boundary.
            if profile.is_none()
                && !probe_due
                && self
                    .batch_speculation_costs
                    .range((key.0, key.1, 0, 0)..=(key.0, key.1, key.2, usize::MAX))
                    .any(|(k, p)| {
                        k.3 == key.3
                            && p.batch_score().is_some_and(|score| {
                                score >= normal_mean / decode.len() as f64 * 0.95
                            })
                    })
            {
                continue;
            }
            let prediction = profile.map_or(normal * 1.5 + key.1 as f64 * 0.005, Profile::upper);
            if prediction > slack {
                continue;
            }
            if profile.is_none_or(|p| p.trials() < 1) {
                if probe.is_none() {
                    probe = Some(selected);
                }
            } else if probe_due && probe.is_none() {
                probe = Some(selected);
            } else if let Some(score) = profile.and_then(Profile::batch_score)
                && score < normal_mean / decode.len() as f64 * 0.95
                && best.as_ref().is_none_or(|(cost, _)| score < *cost)
            {
                best = Some((score, selected));
            }
        }
        probe
            .map(|selected| BatchChoice {
                selected,
                exploring: true,
            })
            .or_else(|| {
                best.map(|(_, selected)| BatchChoice {
                    selected,
                    exploring: false,
                })
            })
    }
    pub(super) fn speculative_batch_step(
        &mut self,
        requests: &mut [&mut RequestState],
        selected: &[(usize, usize)],
        output: &mut Vec<StepOutput>,
    ) -> Result<()> {
        self.speculative_packet_step(requests, selected, &[], output)
    }

    fn speculative_packet_step(
        &mut self,
        requests: &mut [&mut RequestState],
        selected: &[(usize, usize)],
        prefills: &[(usize, usize)],
        output: &mut Vec<StepOutput>,
    ) -> Result<()> {
        let mut rounds = vec![];
        let mut drafts = vec![];
        let grouped_draft = self
            .manifest
            .mtp
            .as_ref()
            .is_some_and(|s| !s.batch_draft_profiles.is_empty());
        for &(i, width) in selected {
            if grouped_draft {
                match self.with_request(requests[i], |model, req| {
                    model.start_speculative_round(req, width)
                }) {
                    Ok(DraftPreparation::Finished(tokens)) => {
                        self.scheduler_statistics.decode_tokens += tokens.len();
                        output.push(StepOutput { request: i, tokens });
                    }
                    Ok(DraftPreparation::Draft(build)) => drafts.push((i, *build)),
                    Err(_) if requests[i].failure().is_some() => output.push(StepOutput {
                        request: i,
                        tokens: vec![],
                    }),
                    Err(error) => return Err(error),
                }
                continue;
            }
            match self.with_request(requests[i], |model, req| {
                model.prepare_speculative_round(req, width)
            }) {
                Ok(Preparation::Finished(tokens)) => {
                    self.scheduler_statistics.decode_tokens += tokens.len();
                    output.push(StepOutput { request: i, tokens });
                }
                Ok(Preparation::Round(round)) => rounds.push((i, round)),
                Err(_) if requests[i].failure().is_some() => output.push(StepOutput {
                    request: i,
                    tokens: vec![],
                }),
                Err(error) => return Err(error),
            }
        }
        if grouped_draft {
            self.generate_batch_drafts(requests, &mut drafts)?;
            for (i, build) in drafts {
                if build.failed {
                    output.push(StepOutput {
                        request: i,
                        tokens: vec![],
                    });
                    continue;
                }
                rounds.push((i, Self::finish_drafts(requests[i], build)?));
            }
        }
        if rounds.len() == 1 && prefills.is_empty() {
            let (i, round) = rounds.pop().unwrap();
            let result = self.with_request(requests[i], |model, req| {
                model.verify_single_round(req, &round)?;
                model.commit_verified_round(req, round)
            });
            let tokens = match result {
                Ok(tokens) => tokens,
                Err(_) if requests[i].failure().is_some() => vec![],
                Err(error) => return Err(error),
            };
            self.scheduler_statistics.decode_tokens += tokens.len();
            output.push(StepOutput { request: i, tokens });
            return Ok(());
        }
        if rounds.is_empty() {
            if !prefills.is_empty() {
                self.execute_selected(requests, prefills, output)?;
            }
            return Ok(());
        }
        let at = Instant::now();
        let mut segments = if prefills.is_empty() {
            vec![]
        } else {
            self.prepare_batch_inputs(requests, prefills)?
        };
        for (i, round) in &rounds {
            let req = &mut requests[*i];
            self.execution.activate_sequence(req.slot)?;
            // Shared verification entries map KV using the live width, not
            // the capacity or the preceding prefill's workspace control.
            self.upload_segment_controls(round.plan.tokens)?;
            self.execution
                .ensure_sequence_program(req.slot, &round.plan.program)?;
            self.upload_ids(&self.manifest.input, &round.verification)?;
            self.prepare_verification_inputs(
                &round.plan.program,
                &round.verification,
                &req.history[..req.history.len() - 1],
            )?;
            self.upload_segment_controls(round.plan.tokens)?;
            segments.push(BatchSegment {
                slot: req.slot,
                tokens: round.plan.tokens,
            });
        }
        let operation = if prefills.is_empty() {
            orinfer_model_sdk::abi::BatchOperation::Verification
        } else {
            orinfer_model_sdk::abi::BatchOperation::MixedRows {
                prefills: prefills.len() as u8,
            }
        };
        let (plan, bindings) = self.model_package.cached_batch(&segments, operation)?;
        if !bindings.is_empty() {
            return Err("Unexpected verification state table".into());
        }
        self.execution.execute_batch_operation(
            operation,
            segments.iter().map(|s| (s.slot, s.tokens)).collect(),
            &plan,
            prefills.is_empty(),
            rounds
                .iter()
                .all(|(i, _)| requests[*i].limit - requests[*i].generated >= 2),
        )?;
        let seconds = at.elapsed().as_secs_f64();
        if !prefills.is_empty() {
            self.scheduler_statistics.mixed_iterations += 1;
            self.scheduler_statistics.mixed_speculative_iterations += 1;
            *self
                .scheduler_statistics
                .prefill_batch_histogram
                .entry(prefills.len())
                .or_default() += 1;
            self.commit_batch(requests, prefills, output)?;
        }
        let total_rows: usize = segments.iter().map(|s| s.tokens).sum();
        let grouped_commit = self.manifest.mtp.as_ref().is_some_and(|s| s.batch_commit);
        let mut committed = vec![];
        for (i, round) in rounds {
            requests[i].mtp.verification_s +=
                seconds * round.plan.tokens as f64 / total_rows as f64;
            if grouped_commit {
                let result = self.with_request(requests[i], |model, req| {
                    model.sample_verified_round(req, round)
                });
                match result {
                    Ok(round) => committed.push((i, round)),
                    Err(_) if requests[i].failure().is_some() => output.push(StepOutput {
                        request: i,
                        tokens: vec![],
                    }),
                    Err(error) => return Err(error),
                }
                continue;
            }
            let result = self.with_request(requests[i], |model, req| {
                model.commit_verified_round(req, round)
            });
            let tokens = match result {
                Ok(tokens) => tokens,
                Err(_) if requests[i].failure().is_some() => vec![],
                Err(error) => return Err(error),
            };
            self.scheduler_statistics.decode_tokens += tokens.len();
            output.push(StepOutput { request: i, tokens });
        }
        if !committed.is_empty() {
            let at = Instant::now();
            let spec = self.manifest.mtp.as_ref().ok_or("Missing MTP adapter")?;
            let mut segments = vec![];
            for (i, round) in &committed {
                self.execution.activate_sequence(requests[*i].slot)?;
                self.upload_ids(&spec.accepted_inputs, &[round.tokens.len() as u32])?;
                segments.push(BatchSegment {
                    slot: requests[*i].slot,
                    tokens: round.plan.tokens,
                });
            }
            if segments.len() >= 2 {
                let (plan, bindings) = self
                    .model_package
                    .cached_batch(&segments, orinfer_model_sdk::abi::BatchOperation::Commit)?;
                if !bindings.is_empty() {
                    return Err("Unexpected commit state table".into());
                }
                self.execution.execute_batch_operation(
                    orinfer_model_sdk::abi::BatchOperation::Commit,
                    segments.iter().map(|s| (s.slot, s.tokens)).collect(),
                    &plan,
                    true,
                    true,
                )?;
            } else {
                let round = &committed[0].1;
                self.launch_rows(
                    &round.plan.restore_program,
                    round.plan.tokens,
                    ExecutionPhase::Decode,
                )?;
            }
            let restore_s = at.elapsed().as_secs_f64() / committed.len() as f64;
            let grouped_warm = self.manifest.mtp.as_ref().is_some_and(|s| s.batch_warm);
            if grouped_warm {
                for (i, round) in &committed {
                    requests[*i].mtp.restore_s += restore_s;
                    self.with_request(requests[*i], |model, req| {
                        model.restore_verified_cursor(req, round, true)
                    })?;
                }
                self.warm_committed_batch(requests, &committed)?;
            }
            for (i, round) in committed {
                let tokens = if grouped_warm {
                    self.with_request(requests[i], |model, req| {
                        model.finish_warmed_commit(req, round)
                    })?
                } else {
                    requests[i].mtp.restore_s += restore_s;
                    self.with_request(requests[i], |model, req| {
                        model.finish_verified_round(req, round, true)
                    })?
                };
                self.scheduler_statistics.decode_tokens += tokens.len();
                output.push(StepOutput { request: i, tokens });
            }
        }
        Ok(())
    }
    fn execute_spec_packet(
        &mut self,
        segments: &[BatchSegment],
        phase: orinfer_model_sdk::abi::BatchOperation,
    ) -> Result<()> {
        let (plan, bindings) = self.model_package.cached_batch(segments, phase)?;
        if !bindings.is_empty() {
            return Err("Unexpected MTP packet state table".into());
        }
        self.execution.execute_batch_operation(
            phase,
            segments.iter().map(|s| (s.slot, s.tokens)).collect(),
            &plan,
            true,
            true,
        )
    }
    fn warm_committed_batch(
        &mut self,
        requests: &mut [&mut RequestState],
        committed: &[(usize, CommittedRound)],
    ) -> Result<()> {
        use orinfer_model_sdk::abi::BatchOperation;
        let spec = self.manifest.mtp.clone().ok_or("Missing MTP adapter")?;
        let capacity = *spec
            .batch_draft_profiles
            .last()
            .ok_or("Missing warm batch capacity")?;
        let at = Instant::now();
        let mut cursor = 0;
        while cursor < committed.len() {
            let mut rows = 0;
            let mut segments = vec![];
            let mut endpoints = vec![];
            while cursor < committed.len() {
                let (i, round) = &committed[cursor];
                let tokens = round.tokens.len();
                if rows + tokens > capacity {
                    break;
                }
                let (warm, live) = spec.warm_plan(tokens)?;
                if live != tokens {
                    return Err("Accepted warm chain exceeds native capacity".into());
                }
                self.execution.activate_sequence(requests[*i].slot)?;
                self.upload_ids(&spec.input, &round.tokens)?;
                self.prepare_program_inputs(&warm.program, &round.tokens, &[])?;
                self.execution
                    .ensure_sequence_program(requests[*i].slot, &warm.program)?;
                segments.push(BatchSegment {
                    slot: requests[*i].slot,
                    tokens,
                });
                endpoints.push(BatchSegment {
                    slot: requests[*i].slot,
                    tokens: 1,
                });
                rows += tokens;
                cursor += 1;
            }
            if segments.is_empty() {
                return Err("Accepted warm chain exceeds native packet capacity".into());
            }
            // Each owner processes its complete causal chain once. Only dense
            // projection/MoE rows are joined; heads consume request endpoints.
            self.execute_spec_packet(&segments, BatchOperation::Warm)?;
            self.execute_spec_packet(&endpoints, BatchOperation::DraftHead)?;
        }
        let total: usize = committed.iter().map(|(_, c)| c.tokens.len()).sum();
        let seconds = at.elapsed().as_secs_f64();
        for (i, round) in committed {
            requests[*i].mtp.refresh_s += seconds * round.tokens.len() as f64 / total as f64;
        }
        Ok(())
    }
    fn generate_batch_drafts(
        &mut self,
        requests: &mut [&mut RequestState],
        drafts: &mut [(usize, DraftRound)],
    ) -> Result<()> {
        let spec = self.manifest.mtp.clone().ok_or("Missing MTP adapter")?;
        while drafts.iter().any(|(_, d)| !d.done) {
            let active: Vec<_> = drafts
                .iter()
                .enumerate()
                .filter(|(_, (_, d))| !d.done)
                .map(|(i, _)| i)
                .collect();
            let advancing: Vec<_> = active
                .iter()
                .copied()
                .filter(|&i| !drafts[i].1.round.drafts.is_empty())
                .collect();
            let mut segments = vec![];
            for &j in &advancing {
                let (i, draft) = &drafts[j];
                self.execution
                    .ensure_sequence_program(requests[*i].slot, &spec.draft_program)?;
                self.prepare_program_inputs(
                    &spec.draft_program,
                    &[*draft.round.drafts.last().unwrap()],
                    &[],
                )?;
                segments.push(BatchSegment {
                    slot: requests[*i].slot,
                    tokens: 1,
                });
            }
            if segments.len() >= 2 {
                let (plan, bindings) = self
                    .model_package
                    .cached_batch(&segments, orinfer_model_sdk::abi::BatchOperation::Draft)?;
                if !bindings.is_empty() {
                    return Err("Unexpected draft state table".into());
                }
                self.execution.execute_batch_operation(
                    orinfer_model_sdk::abi::BatchOperation::Draft,
                    segments.iter().map(|s| (s.slot, s.tokens)).collect(),
                    &plan,
                    true,
                    true,
                )?;
            } else if let Some(segment) = segments.first() {
                self.execution.activate_sequence(segment.slot)?;
                self.launch_rows(&spec.draft_program, 1, ExecutionPhase::Decode)?;
            }
            for j in active {
                let (i, draft) = &mut drafts[j];
                match self.with_request(requests[*i], |model, req| model.consume_draft(req, draft))
                {
                    Ok(()) => {}
                    Err(_) if requests[*i].failure().is_some() => {
                        draft.done = true;
                        draft.failed = true;
                    }
                    Err(error) => return Err(error),
                }
            }
        }
        Ok(())
    }
    pub(super) fn refresh_request_mtp(&self, req: &mut RequestState, head: bool) -> Result<bool> {
        let Some(spec) = &self.manifest.mtp else {
            return Ok(true);
        };
        let position = self.read_control(&self.manifest.position)? as usize;
        let draft_position = self.read_control(&spec.position)? as usize;
        if req.warm.tokens > position || draft_position != req.warm.tokens {
            return Err(format!(
                "Private MTP cursor mismatch: slot {}, target {position}, draft {draft_position}, warm {}",
                req.slot, req.warm.tokens
            ));
        }
        let lag = position - req.warm.tokens;
        let ring = spec
            .hidden_ring
            .as_ref()
            .and_then(|n| self.manifest.buffers.iter().find(|b| b.name == *n))
            .map(|b| b.shape[0])
            .unwrap_or(1);
        if req.mtp_blocked || lag > ring {
            req.mtp_blocked = true;
            return Ok(false);
        }
        if lag != 0 {
            let at = Instant::now();
            self.mtp_warm_state(
                spec,
                &req.history[req.warm.tokens + 1..position + 1],
                &|| false,
                ExecutionPhase::Decode,
                head,
            )?;
            req.warm.tokens = position;
            req.mtp.refresh_s += at.elapsed().as_secs_f64();
        }
        Ok(true)
    }
    pub(super) fn speculative_request_step(
        &mut self,
        req: &mut RequestState,
        max_rows: usize,
    ) -> Result<Vec<u32>> {
        let round = match self.prepare_speculative_round(req, max_rows)? {
            Preparation::Finished(tokens) => return Ok(tokens),
            Preparation::Round(round) => round,
        };
        self.verify_single_round(req, &round)?;
        self.commit_verified_round(req, round)
    }
    fn start_speculative_round(
        &mut self,
        req: &mut RequestState,
        max_rows: usize,
    ) -> Result<DraftPreparation> {
        if req.failure().is_some() {
            return Ok(DraftPreparation::Finished(vec![]));
        }
        let spec = self.manifest.mtp.clone().ok_or("Missing MTP adapter")?;
        let valid = self.refresh_request_mtp(req, true)?;
        let position = self.read_control(&self.manifest.position)? as usize;
        let remaining = req.limit - req.generated;
        let plan = valid
            .then(|| {
                spec.best_verification_plan(
                    remaining
                        .min(max_rows)
                        .min(spec.default_verification_tokens)
                        .min(self.manifest.max_context - position),
                )
            })
            .flatten();
        let pending = *req.history.last().ok_or("Missing pending token")?;
        let Some(plan) = plan else {
            self.execute_target_tail(req)?;
            let out = self.commit_ordinary_token(req)?;
            if valid {
                self.refresh_request_mtp(req, true)?;
            }
            return Ok(DraftPreparation::Finished(out));
        };
        let host_sampling = !req.sampling.is_greedy() || req.sampling.top_logprobs.is_some();
        let lookup = (self.prompt_lookup && !host_sampling)
            .then(|| req.prompt_lookup.propose(&req.history, plan.tokens - 1))
            .flatten();
        let is_lookup = lookup.is_some();
        if !is_lookup && let Some(save) = &spec.draft_snapshot_program {
            self.launch_program(save, ExecutionPhase::Decode)?;
        }
        let drafts = lookup.unwrap_or_default();
        if is_lookup {
            req.mtp.lookup_rounds += 1;
            req.mtp.lookup_proposed_tokens += drafts.len();
        }
        Ok(DraftPreparation::Draft(Box::new(DraftRound {
            round: PreparedRound {
                plan: plan.clone(),
                position,
                remaining,
                vocab: self.manifest.vocab,
                host_sampling,
                is_lookup,
                drafts,
                proposals: vec![],
                verification: vec![pending],
            },
            decoder: req.decoder.fork(),
            history: req.history.clone(),
            done: is_lookup,
            failed: false,
            started: Instant::now(),
        })))
    }

    /// Use the model's ordinary one-row packet for an MTP tail whenever that
    /// capability is declared. The packet owns target capture as well as state.
    fn execute_target_tail(&mut self, req: &mut RequestState) -> Result<()> {
        if self.manifest.batch_profiles.contains(&1) {
            let allow_replay = req.limit - req.generated >= 2;
            let segments = self.prepare_batch_inputs(&mut [req], &[(0, 1)])?;
            let (plan, _) = self
                .model_package
                .cached_batch(&segments, orinfer_model_sdk::abi::BatchOperation::Ordinary)?;
            self.execution.execute_batch(
                segments.iter().map(|s| (s.slot, s.tokens)).collect(),
                &plan,
                true,
                allow_replay,
            )
        } else {
            let pending = *req.history.last().ok_or("Missing target tail input")?;
            self.upload_ids(&self.manifest.token, &[pending])?;
            self.prepare_inputs(&[pending], &req.history[..req.history.len() - 1])?;
            self.upload_segment_controls(1)?;
            self.launch_program("decode", ExecutionPhase::Decode)?;
            if let Some(spec) = &self.manifest.mtp {
                self.mtp_capture(spec, 1, ExecutionPhase::Decode)?;
            }
            Ok(())
        }
    }
    fn consume_draft(&self, req: &mut RequestState, build: &mut DraftRound) -> Result<()> {
        let spec = self.manifest.mtp.as_ref().ok_or("Missing MTP adapter")?;
        let round = &mut build.round;
        let vocab = round.vocab;
        let host_sampling = round.host_sampling;
        let drafts = &mut round.drafts;
        let proposals = &mut round.proposals;
        let draft_decoder = &mut build.decoder;
        let draft_history = &mut build.history;
        let draft_options = crate::sampling::Options {
            top_logprobs: None,
            ..req.sampling.clone()
        };
        let i = drafts.len();
        let [selected, status] = self
            .execution
            .read_control_group([&spec.token, &spec.status])?;
        if status != 0 || selected < 0 || selected as usize >= vocab {
            return Err("Invalid request draft token".into());
        }
        let token = if !host_sampling {
            let token = match draft_decoder
                .greedy_candidate(selected as u32, vocab)
                .inspect_err(|error| {
                    if draft_decoder.failure().is_some() {
                        req.decoder.fail(error.clone());
                    }
                })? {
                Some(token) => token,
                None => {
                    let raw = self
                        .execution
                        .download_bytes(&spec.draft_logits, vocab * 4)?;
                    draft_decoder
                        .law(
                            &floats(&raw, crate::artifact::Dtype::F32),
                            draft_history,
                            &draft_options,
                        )
                        .inspect_err(|error| {
                            if draft_decoder.failure().is_some() {
                                req.decoder.fail(error.clone());
                            }
                        })?
                        .distribution
                        .draw(0.0)?
                }
            };
            if token != selected as u32 {
                self.upload_ids(&spec.token, &[token])?;
            }
            draft_decoder.consume(token).inspect_err(|error| {
                req.decoder.fail(error.clone());
            })?;
            token
        } else {
            let raw = self
                .execution
                .download_bytes(&spec.draft_logits, vocab * 4)?;
            let law = draft_decoder
                .law(
                    &floats(&raw, crate::artifact::Dtype::F32),
                    draft_history,
                    &draft_options,
                )
                .inspect_err(|error| {
                    if draft_decoder.failure().is_some() {
                        req.decoder.fail(error.clone());
                    }
                })?;
            let distribution = law.distribution;
            let token = distribution.draw(crate::sampling::counter_uniform(
                req.sampling.seed,
                crate::mtp::DRAFT_STREAM,
                (req.generated + i) as u64,
            ))?;
            proposals.push(crate::mtp::Proposal {
                token,
                distribution,
            });
            self.upload_ids(&spec.token, &[token])?;
            draft_decoder.consume(token).inspect_err(|error| {
                req.decoder.fail(error.clone());
            })?;
            token
        };
        drafts.push(token);
        draft_history.push(token);
        build.done = draft_decoder.finished() || drafts.len() == round.plan.tokens - 1;
        Ok(())
    }
    fn finish_drafts(req: &mut RequestState, mut build: DraftRound) -> Result<PreparedRound> {
        let round = &mut build.round;
        req.mtp.proposed_tokens += round.drafts.len();
        // Padding after EOS has no authority over the parser or committed tokens.
        round.drafts.resize(
            round.plan.tokens - 1,
            *round.drafts.last().ok_or("Empty draft profile")?,
        );
        round.verification.extend_from_slice(&round.drafts);
        req.mtp.draft_s += build.started.elapsed().as_secs_f64();
        Ok(build.round)
    }
    pub(super) fn prepare_speculative_round(
        &mut self,
        req: &mut RequestState,
        max_rows: usize,
    ) -> Result<Preparation> {
        let mut build = match self.start_speculative_round(req, max_rows)? {
            DraftPreparation::Finished(tokens) => return Ok(Preparation::Finished(tokens)),
            DraftPreparation::Draft(build) => *build,
        };
        while !build.done {
            if !build.round.drafts.is_empty() {
                let spec = self.manifest.mtp.as_ref().ok_or("Missing MTP adapter")?;
                self.prepare_program_inputs(
                    &spec.draft_program,
                    &[*build.round.drafts.last().unwrap()],
                    &[],
                )?;
                self.launch_rows(&spec.draft_program, 1, ExecutionPhase::Decode)?;
            }
            self.consume_draft(req, &mut build)?;
        }
        Ok(Preparation::Round(Self::finish_drafts(req, build)?))
    }
    pub(super) fn verify_single_round(
        &mut self,
        req: &mut RequestState,
        round: &PreparedRound,
    ) -> Result<()> {
        let plan = &round.plan;
        let verification = &round.verification;
        let at = Instant::now();
        let packet = self
            .manifest
            .mtp
            .as_ref()
            .is_some_and(|spec| spec.batch_verification_tokens.contains(&plan.tokens));
        self.upload_segment_controls(plan.tokens)?;
        if packet {
            self.execution
                .ensure_sequence_program(req.slot, &plan.program)?;
        }
        self.upload_ids(&self.manifest.input, verification)?;
        self.prepare_verification_inputs(
            &plan.program,
            verification,
            &req.history[..req.history.len() - 1],
        )?;
        if packet {
            self.execute_spec_packet(
                &[BatchSegment {
                    slot: req.slot,
                    tokens: plan.tokens,
                }],
                orinfer_model_sdk::abi::BatchOperation::Verification,
            )?;
        } else {
            self.launch_rows(&plan.program, plan.tokens, ExecutionPhase::Decode)?;
            self.launch_rows(&plan.capture_program, plan.tokens, ExecutionPhase::Decode)?;
        }
        req.mtp.verification_s += at.elapsed().as_secs_f64();
        Ok(())
    }
    fn sample_verified_round(
        &mut self,
        req: &mut RequestState,
        round: PreparedRound,
    ) -> Result<CommittedRound> {
        let spec = self.manifest.mtp.clone().ok_or("Missing MTP adapter")?;
        let PreparedRound {
            plan,
            position,
            remaining,
            vocab,
            host_sampling,
            is_lookup,
            drafts,
            proposals,
            verification: _,
        } = round;
        let target = self.read_controls(&spec.verification_tokens, plan.tokens)?;
        let status = self.read_controls(&spec.verification_status, plan.tokens)?;
        if target.iter().any(|&id| id as usize >= vocab) || status.iter().any(|&s| s != 0) {
            return Err("Invalid request verification output".into());
        }
        req.mtp.rounds += 1;
        let at = Instant::now();
        let committed = if !host_sampling {
            let mut committed = Vec::with_capacity(plan.tokens);
            let mut logits = None;
            let mut history = req.history.clone();
            for (row, &selected) in target.iter().enumerate() {
                let token = match req.decoder.greedy_candidate(selected, vocab)? {
                    Some(token) => token,
                    None => {
                        if logits.is_none() {
                            let raw = self.execution.download_bytes(
                                &spec.verification_logits,
                                plan.tokens * vocab * 4,
                            )?;
                            logits = Some(floats(&raw, crate::artifact::Dtype::F32));
                        }
                        req.decoder
                            .law(
                                &logits.as_ref().unwrap()[row * vocab..(row + 1) * vocab],
                                &history,
                                &req.sampling,
                            )?
                            .distribution
                            .draw(0.0)?
                    }
                };
                req.decoder.consume(token)?;
                committed.push(token);
                history.push(token);
                if drafts.get(row) != Some(&token) || req.decoder.finished() {
                    break;
                }
            }
            committed
        } else {
            let raw = self
                .execution
                .download_bytes(&spec.verification_logits, plan.tokens * vocab * 4)?;
            crate::sampling::verify(
                &proposals,
                &floats(&raw, crate::artifact::Dtype::F32),
                &req.history,
                &req.sampling,
                &mut req.decoder,
                req.generated,
                vocab,
            )?
        };
        if committed.is_empty() || committed.len() > remaining {
            return Err("Invalid speculative commit length".into());
        }
        req.mtp.sampling_s += at.elapsed().as_secs_f64();
        let accepted = if host_sampling {
            committed
                .iter()
                .zip(&proposals)
                .take_while(|(token, proposal)| **token == proposal.token)
                .count()
        } else {
            committed
                .iter()
                .zip(&drafts)
                .take_while(|(token, draft)| token == draft)
                .count()
        };
        req.mtp.accepted_draft_tokens += accepted;
        if is_lookup {
            req.mtp.lookup_accepted_draft_tokens += accepted;
            req.prompt_lookup
                .observe(accepted, drafts.len(), req.history.len() + committed.len());
        }
        Ok(CommittedRound {
            tokens: committed,
            plan,
            position,
            is_lookup,
        })
    }
    fn restore_verified_cursor(
        &mut self,
        req: &mut RequestState,
        round: &CommittedRound,
        restored: bool,
    ) -> Result<()> {
        let spec = self.manifest.mtp.clone().ok_or("Missing MTP adapter")?;
        let CommittedRound {
            tokens: committed,
            plan,
            position,
            is_lookup,
        } = round;
        let at = Instant::now();
        if !restored && (spec.commit_always || committed.len() < plan.tokens) {
            self.upload_ids(&spec.accepted_inputs, &[committed.len() as u32])?;
            self.launch_rows(&plan.restore_program, plan.tokens, ExecutionPhase::Decode)?;
        }
        if restored || spec.commit_always || committed.len() < plan.tokens {
            self.upload_ids(
                &self.manifest.position,
                &[(*position + committed.len()) as u32],
            )?;
            self.upload_ids(&spec.target_length, &[(*position + committed.len()) as u32])?;
        }
        req.mtp.restore_s += at.elapsed().as_secs_f64();
        req.history.extend_from_slice(committed);
        req.generated += committed.len();
        let at = Instant::now();
        if !*is_lookup && let Some(restore) = &spec.draft_restore_program {
            self.launch_program(restore, ExecutionPhase::Decode)?;
        }
        self.upload_ids(&spec.position, &[*position as u32])?;
        req.mtp.refresh_s += at.elapsed().as_secs_f64();
        Ok(())
    }
    fn finish_warmed_commit(
        &self,
        req: &mut RequestState,
        round: CommittedRound,
    ) -> Result<Vec<u32>> {
        let committed = round.tokens;
        req.warm.tokens = round.position + committed.len();
        self.upload_ids(
            &self.manifest.token,
            &[*committed.last().expect("nonempty commit")],
        )?;
        if self.read_control(&self.manifest.position)? as usize
            != req.input.len() + req.generated - 1
        {
            return Err("Private speculative position mismatch".into());
        }
        req.mtp.committed_tokens = req.generated;
        Ok(committed)
    }
    fn finish_verified_round(
        &mut self,
        req: &mut RequestState,
        round: CommittedRound,
        restored: bool,
    ) -> Result<Vec<u32>> {
        self.restore_verified_cursor(req, &round, restored)?;
        let spec = self.manifest.mtp.clone().ok_or("Missing MTP adapter")?;
        let at = Instant::now();
        self.mtp_warm(&spec, &round.tokens, &|| false, ExecutionPhase::Decode)?;
        req.mtp.refresh_s += at.elapsed().as_secs_f64();
        self.finish_warmed_commit(req, round)
    }
    pub(super) fn commit_verified_round(
        &mut self,
        req: &mut RequestState,
        round: PreparedRound,
    ) -> Result<Vec<u32>> {
        let committed = self.sample_verified_round(req, round)?;
        self.finish_verified_round(req, committed, false)
    }
}

#[cfg(test)]
#[path = "tests/verification.rs"]
mod verification_tests;

#[cfg(test)]
mod policy_tests {
    use super::*;

    #[test]
    #[ignore = "Requires real MTP model and exclusive GPU experiment lock"]
    fn validate_flash_warm_input_ownership() {
        #[derive(serde::Deserialize)]
        struct Fixture {
            model: std::path::PathBuf,
            cases: Vec<GenerationInput>,
            cuda_graph: String,
        }
        let fixture: Fixture = crate::model::read(&std::path::PathBuf::from(
            std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
        ))
        .unwrap();
        let mut model = ModelRuntime::load_with_options(
            &fixture.model,
            LoadOptions {
                cuda_graph: fixture.cuda_graph.parse().unwrap(),
                prefix_cache_bytes: 0,
                mtp_drafts: Some(3),
                ..Default::default()
            },
        )
        .unwrap();
        let spec = model.manifest.mtp.clone().unwrap();
        let mut requests = Vec::new();
        let mut committed = Vec::new();
        for (i, input) in fixture.cases.into_iter().take(2).enumerate() {
            let mut req = model.start_request(input, &|| false).unwrap();
            while req.prefilling {
                model
                    .with_request(&mut req, |m, r| {
                        m.advance_prefill(r, r.input.len().min(4096), false)
                    })
                    .unwrap();
            }
            let round = model
                .with_request(&mut req, |m, r| {
                    let Preparation::Round(round) = m.prepare_speculative_round(r, 4)? else {
                        return Err("Expected a live speculative round".into());
                    };
                    m.verify_single_round(r, &round)?;
                    let committed = m.sample_verified_round(r, round)?;
                    m.restore_verified_cursor(r, &committed, false)?;
                    Ok(committed)
                })
                .unwrap();
            committed.push((i, round));
            requests.push(req);
        }
        assert_eq!(requests.len(), 2);
        for req in &mut requests {
            model
                .with_request(req, |m, _| {
                    // A previous request/round may leave any control value.
                    // Mapping must use this owner's newly uploaded warm length.
                    m.upload_ids("MtpLength", &[0])?;
                    m.upload_ids(&spec.input, &[0; 8])
                })
                .unwrap();
        }
        model
            .warm_committed_batch(&mut requests.iter_mut().collect::<Vec<_>>(), &committed)
            .unwrap();
        for (i, round) in &committed {
            model
                .with_request(&mut requests[*i], |m, _| {
                    assert_eq!(m.read_control("MtpLength")? as usize, round.tokens.len());
                    assert_eq!(
                        m.read_controls(&spec.input, round.tokens.len())?,
                        round.tokens
                    );
                    assert_eq!(
                        m.read_control(&spec.position)? as usize,
                        round.position + round.tokens.len()
                    );
                    Ok(())
                })
                .unwrap();
        }
        for req in &mut requests {
            model.finish_request(req, false).unwrap();
        }
    }
    #[test]
    fn batch_cost_uses_measured_upper_margin_and_rejects_invalid_samples() {
        let mut p = Profile::default();
        p.observe_steady(0.2, 8, 8);
        assert!((p.batch_score().unwrap() - 0.025).abs() < 1e-12);
        p.observe_steady(0.4, 4, 16);
        assert!(p.batch_score().unwrap() > p.time.mean / p.tokens);
        // Ordinary 8-row decode taking 60 ms is faster than this MTP round.
        assert!(p.batch_score().unwrap() > 0.06 / 8.);
        p.observe_steady(f64::NAN, 8, 24);
        p.observe_steady(0.01, 0, 24);
        assert_eq!(p.trials(), 2);
        assert_eq!(p.last_progress, 16);
    }
    #[test]
    #[ignore = "Requires real MTP model and exclusive GPU experiment lock"]
    fn validate_batched_drafts_and_commits() {
        #[derive(serde::Deserialize)]
        struct Fixture {
            model: std::path::PathBuf,
            output: std::path::PathBuf,
            cases: Vec<GenerationInput>,
            cuda_graph: String,
        }
        let fixture: Fixture = crate::model::read(&std::path::PathBuf::from(
            std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
        ))
        .unwrap();
        assert!(!fixture.output.exists());
        let mut model = ModelRuntime::load_with_options(
            &fixture.model,
            LoadOptions {
                cuda_graph: fixture.cuda_graph.parse().unwrap(),
                prefix_cache_bytes: 0,
                ..Default::default()
            },
        )
        .unwrap();
        let spec = model.manifest.mtp.clone().unwrap();
        assert!(spec.batch_draft_profiles.last().is_some_and(|&n| n >= 8));
        assert!(spec.batch_commit);
        assert!(spec.batch_warm);
        let options = scheduler::Options {
            target_tpot_ms: 1000.,
            memory_reserve_bytes: 0,
            ..Default::default()
        };
        let source = &fixture.cases[0];
        let input = |lane: usize| {
            let mut tokens = source.input_tokens.clone();
            tokens.extend([32, 32 + lane as u32]);
            GenerationInput {
                input_tokens: tokens,
                images: source.images.clone(),
                max_new_tokens: 16,
                sampling: source.sampling.clone(),
                prefix_hints: vec![],
            }
        };
        let log_probs = |values: &[f32]| {
            let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let norm = values
                .iter()
                .map(|&x| (x as f64 - max).exp())
                .sum::<f64>()
                .ln()
                + max;
            values.iter().map(|&x| x as f64 - norm).collect::<Vec<_>>()
        };
        let mut rows = vec![];
        for count in [2, 3, 4, 8] {
            let mut baselines = vec![];
            let mut candidates = vec![];
            for lane in 0..count {
                let mut req = model.start_request(input(lane), &|| false).unwrap();
                while req.prefilling {
                    model.advance_requests(&mut [&mut req], &options).unwrap();
                }
                // Fix the shifted token history independently of a rounded
                // draft selection; compare the whole distribution at each step.
                let mut scores = vec![];
                let mut tokens = vec![];
                for _ in 0..3 {
                    let selected = model
                        .with_request(&mut req, |m, _| {
                            let raw = m
                                .execution
                                .download_bytes(&spec.draft_logits, m.manifest.vocab * 4)?;
                            scores.push(log_probs(&floats(&raw, crate::artifact::Dtype::F32)));
                            Ok(m.read_control(&spec.token)? as u32)
                        })
                        .unwrap();
                    tokens.push(selected);
                    model
                        .with_request(&mut req, |m, _| {
                            m.prepare_program_inputs(&spec.draft_program, &[selected], &[])?;
                            m.launch_rows(&spec.draft_program, 1, ExecutionPhase::Decode)
                        })
                        .unwrap();
                }
                baselines.push(scores);
                candidates.push(tokens);
                model.finish_request(&mut req, false).unwrap();
            }
            let mut requests = vec![];
            for lane in 0..count {
                let mut req = model.start_request(input(lane), &|| false).unwrap();
                while req.prefilling {
                    model.advance_requests(&mut [&mut req], &options).unwrap();
                }
                requests.push(req);
            }
            let mut max_kl = 0f64;
            for depth in 0..3 {
                for (lane, req) in requests.iter_mut().enumerate() {
                    let score = model
                        .with_request(req, |m, _| {
                            let raw = m
                                .execution
                                .download_bytes(&spec.draft_logits, m.manifest.vocab * 4)?;
                            Ok(log_probs(&floats(&raw, crate::artifact::Dtype::F32)))
                        })
                        .unwrap();
                    let baseline = &baselines[lane][depth];
                    let kl: f64 = baseline
                        .iter()
                        .zip(&score)
                        .map(|(b, a)| b.exp() * (b - a))
                        .sum();
                    assert!(kl < 0.05, "draft KL {count}/{lane}/{depth}: {kl}");
                    let token = candidates[lane][depth] as usize;
                    assert!(
                        (baseline[token] - score[token]).abs() < 0.1,
                        "draft selected NLL delta"
                    );
                    max_kl = max_kl.max(kl);
                    model
                        .execution
                        .ensure_sequence_program(req.slot, &spec.draft_program)
                        .unwrap();
                    model
                        .prepare_program_inputs(
                            &spec.draft_program,
                            &[candidates[lane][depth]],
                            &[],
                        )
                        .unwrap();
                }
                let segments: Vec<_> = requests
                    .iter()
                    .map(|r| BatchSegment {
                        slot: r.slot,
                        tokens: 1,
                    })
                    .collect();
                let (plan, bindings) = model
                    .model_package
                    .batch_operation(
                        &segments,
                        orinfer_model_sdk::abi::BatchOperation::Draft,
                        true,
                    )
                    .unwrap();
                assert!(bindings.is_empty());
                model
                    .execution
                    .execute_batch_operation(
                        orinfer_model_sdk::abi::BatchOperation::Draft,
                        segments.iter().map(|s| (s.slot, s.tokens)).collect(),
                        &plan,
                        true,
                        true,
                    )
                    .unwrap();
            }
            for req in &mut requests {
                model.finish_request(req, false).unwrap();
            }
            // Exercise complete native rounds with unequal histories, changing
            // membership and tail fallback, then cancellation/arena reuse.
            let mut outputs = vec![vec![]; count];
            requests.clear();
            for (lane, output) in outputs.iter_mut().enumerate() {
                let mut req = model.start_request(input(lane), &|| false).unwrap();
                while req.prefilling {
                    for step in model.advance_requests(&mut [&mut req], &options).unwrap() {
                        output.extend(step.tokens);
                    }
                }
                requests.push(req);
            }
            let at = Instant::now();
            for _ in 0..3 {
                let selected: Vec<_> = requests
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| r.limit - r.generated >= 4)
                    .map(|(i, _)| (i, 4))
                    .collect();
                assert_eq!(selected.len(), count);
                let mut active: Vec<_> = requests.iter_mut().collect();
                let mut steps = vec![];
                model
                    .speculative_batch_step(&mut active, &selected, &mut steps)
                    .unwrap();
                for step in steps {
                    outputs[step.request].extend(step.tokens);
                }
            }
            let round_s = at.elapsed().as_secs_f64();
            while requests.iter().any(|r| !r.is_finished()) {
                let mut active: Vec<_> = requests.iter_mut().collect();
                for step in model.advance_requests(&mut active, &options).unwrap() {
                    outputs[step.request].extend(step.tokens);
                }
            }
            for (lane, req) in requests.iter_mut().enumerate() {
                assert_eq!(req.generated, outputs[lane].len());
                model
                    .with_request(req, |m, r| {
                        assert_eq!(
                            m.read_control(&m.manifest.position)? as usize,
                            r.input.len() + r.generated - 1
                        );
                        Ok(())
                    })
                    .unwrap();
                model.finish_request(req, false).unwrap();
            }
            model.execution.reclaim_idle_state().unwrap();
            assert_eq!(model.execution.resident_kv_bytes(), 0);
            rows.push(serde_json::json!({"batch":count,"draft_max_kl":max_kl,"three_rounds_s":round_s,"outputs":outputs,"position_and_cleanup_passed":true}));
            std::fs::write(&fixture.output, serde_json::to_vec_pretty(&rows).unwrap()).unwrap();
        }
        let mut peers = vec![];
        for lane in 0..2 {
            let mut req = model.start_request(input(lane), &|| false).unwrap();
            while req.prefilling {
                model.advance_requests(&mut [&mut req], &options).unwrap();
            }
            peers.push(req);
        }
        peers[0]
            .decoder
            .fail("Injected request constraint failure".into());
        let before = peers[1].generated;
        let mut output = vec![];
        model
            .speculative_batch_step(
                &mut peers.iter_mut().collect::<Vec<_>>(),
                &[(0, 4), (1, 4)],
                &mut output,
            )
            .unwrap();
        assert!(output.iter().any(|o| o.request == 0 && o.tokens.is_empty()));
        assert!(peers[1].failure().is_none() && peers[1].generated > before);
        for req in &mut peers {
            model.finish_request(req, false).unwrap();
        }
    }
    #[test]
    fn profiles_warm_before_comparison_and_revisit_old_lengths() {
        let mut costs = BTreeMap::<usize, Profile>::new();
        let plans = [2, 4];
        for round in 0..3 {
            assert_eq!(
                select_profile(&plans, &costs, round, 0, f64::INFINITY, 0.1),
                Some((4, true))
            );
            costs
                .entry(4)
                .or_default()
                .observe(if round == 0 { 1.0 } else { 0.08 }, 3, round + 1);
        }
        assert!((costs[&4].score().unwrap() - 0.08 / 3.0).abs() < 1e-12);
        for round in 3..6 {
            assert_eq!(
                select_profile(&plans, &costs, round, 0, f64::INFINITY, 0.1),
                Some((2, true))
            );
            costs.entry(2).or_default().observe(0.06, 1, round + 1);
        }
        assert_eq!(
            select_profile(&plans, &costs, 7, 0, f64::INFINITY, 0.1),
            Some((4, false))
        );
        assert_eq!(
            select_profile(&plans, &costs, 64, 0, f64::INFINITY, 0.1),
            Some((4, true))
        );
        costs.get_mut(&4).unwrap().observe(0.08, 3, 64);
        // Ordinary decode advances progress without observing any profile.
        // Crossing the probe interval must recover exploration anyway.
        assert_eq!(
            select_profile(&plans, &costs, 128, 64, f64::INFINITY, 0.1),
            Some((2, true))
        );
        assert_eq!(
            select_profile(&plans, &costs, 127, 64, f64::INFINITY, 0.1),
            Some((4, false))
        );
        assert_eq!(
            select_profile(&plans, &costs, 7, 0, 0.07, 0.1),
            Some((2, false))
        );
        assert_eq!(select_profile(&plans, &costs, 128, 64, 0.01, 0.1), None);
        assert_eq!(
            select_profile(&[4], &costs, 7, 0, f64::INFINITY, 0.1),
            Some((4, false))
        );
    }
}
