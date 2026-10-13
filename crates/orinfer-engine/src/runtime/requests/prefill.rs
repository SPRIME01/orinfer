//! Resume a large forward pass at model-registered layer boundaries. Tokens
//! become committed only after its target body and shifted draft warm complete.
use super::*;
use crate::cuda::snapshot::{Range, Snapshot};

pub(super) struct Frame {
    pub plan: crate::model::PrefillPlan,
    next: usize,
    snapshot: Option<Snapshot>,
    regular_s: f64,
}

impl ModelRuntime {
    pub(super) fn release_prefill_frame(&mut self, req: &mut RequestState) -> Result<()> {
        if let Some(mut frame) = req.prefill_frame.take()
            && let Some(snapshot) = frame.snapshot.take()
        {
            self.execution.release_snapshot(snapshot)?;
        }
        Ok(())
    }
    pub(super) fn prefill_stage_cost(&self, req: &RequestState, chunk: usize) -> f64 {
        let plan = req.prefill_frame.as_ref().map(|f| &f.plan).or_else(|| {
            self.manifest
                .prefill_plans
                .iter()
                .find(|p| p.chunk_tokens == chunk)
        });
        if let Some(plan) = plan.filter(|p| !p.stages.is_empty()) {
            if let Some(frame) = &req.prefill_frame {
                return self.prefill_costs.stage_upper(
                    chunk,
                    frame.next,
                    req.offset + chunk,
                    plan.stages.len(),
                );
            }
            // A new tile must fit its slowest measured layer, not just its
            // first layer. An in-flight tile keeps its original extent.
            (0..plan.stages.len())
                .map(|stage| {
                    self.prefill_costs.stage_upper(
                        chunk,
                        stage,
                        req.offset + chunk,
                        plan.stages.len(),
                    )
                })
                .fold(0., f64::max)
        } else {
            self.prefill_costs.chunk_upper(chunk, req.offset + chunk)
        }
    }

    pub(super) fn advance_prefill_step(
        &mut self,
        req: &mut RequestState,
        chunk: usize,
        bounded: bool,
    ) -> Result<(usize, Vec<u32>)> {
        let plan = req
            .prefill_frame
            .as_ref()
            .map(|f| f.plan.clone())
            .or_else(|| {
                self.manifest
                    .prefill_plans
                    .iter()
                    .find(|p| p.chunk_tokens == chunk)
                    .cloned()
            });
        let Some(plan) = plan.filter(|p| !p.stages.is_empty()) else {
            return self.advance_prefill(req, chunk, bounded);
        };
        if chunk != plan.chunk_tokens || chunk > Self::prefill_boundary(req) - req.offset {
            return Err("Invalid resumed prefill extent".into());
        }
        let at = Instant::now();
        let cold_before = self.execution.batch_statistics.get();
        let mut frame = if let Some(frame) = req.prefill_frame.take() {
            frame
        } else {
            self.upload_ids(
                &self.manifest.input,
                &req.input[req.offset..req.offset + chunk],
            )?;
            self.prepare_inputs(
                &req.input[req.offset..req.offset + chunk],
                &req.input[..req.offset],
            )?;
            Frame {
                plan,
                next: 0,
                snapshot: None,
                regular_s: at.elapsed().as_secs_f64(),
            }
        };
        if let Some(snapshot) = frame.snapshot.take() {
            let resume = Instant::now();
            self.execution.restore_snapshot(&snapshot)?;
            self.execution.release_snapshot(snapshot)?;
            self.scheduler_statistics.prefill_resume_s += resume.elapsed().as_secs_f64();
            frame.regular_s += resume.elapsed().as_secs_f64();
        }
        // OOM while saving a continuation falls back to completing the same
        // forward pass, never to recomputing already-mutated recurrent layers.
        loop {
            let began = Instant::now();
            let load_before = self.execution.batch_statistics.get().kernel_load_s;
            let stage = frame.next;
            self.scheduler_statistics.layered_prefill_iterations += 1;
            if stage < frame.plan.stages.len() {
                self.upload_segment_controls(chunk)?;
                self.launch_program(&frame.plan.stages[stage], ExecutionPhase::Prefill)?;
                frame.next += 1;
                frame.regular_s += began.elapsed().as_secs_f64()
                    - (self.execution.batch_statistics.get().kernel_load_s - load_before);
                if frame.next == frame.plan.stages.len() {
                    if let Some(spec) = &self.manifest.mtp {
                        self.mtp_capture(spec, chunk, ExecutionPhase::Prefill)?;
                    }
                    if req.offset + chunk == req.input.len()
                        || req.checkpoints.contains(&(req.offset + chunk))
                    {
                        self.launch_program(&frame.plan.head_program, ExecutionPhase::Prefill)?;
                    }
                }
            } else if let Some(spec) = self.manifest.mtp.clone() {
                let end = req.offset + chunk;
                let warm_end =
                    end + usize::from(end < req.input.len() && !req.checkpoints.contains(&end));
                let remaining = warm_end.saturating_sub(req.warm.tokens + 1);
                if remaining > 0 {
                    let tokens = spec
                        .warm_plans
                        .iter()
                        .filter(|p| p.tokens <= remaining)
                        .map(|p| p.tokens)
                        .max()
                        .ok_or("Missing resumed MTP warm shape")?;
                    let offset = req.warm.tokens + 1;
                    self.mtp_warm_state(
                        &spec,
                        &req.input[offset..offset + tokens],
                        &|| false,
                        ExecutionPhase::Prefill,
                        false,
                    )?;
                    req.warm.tokens += tokens;
                    req.warm.seconds += began.elapsed().as_secs_f64();
                }
            }
            if bounded {
                self.scheduler_statistics.bounded_prefill_iterations += 1;
            }
            if frame.next < frame.plan.stages.len() {
                let suspend = Instant::now();
                let ranges = frame
                    .plan
                    .live_buffers
                    .iter()
                    .map(|name| {
                        (
                            name.clone(),
                            Range {
                                offset: 0,
                                bytes: self.execution.sizes[name],
                            },
                        )
                    })
                    .collect();
                let snapshot_plan = self
                    .execution
                    .plan_snapshot(ranges, &BTreeSet::new(), &[])?;
                if let Some(snapshot) = self.execution.snapshot(&snapshot_plan)? {
                    self.scheduler_statistics.prefill_suspend_s += suspend.elapsed().as_secs_f64();
                    self.scheduler_statistics.peak_prefill_frame_bytes = self
                        .scheduler_statistics
                        .peak_prefill_frame_bytes
                        .max(snapshot.bytes);
                    frame.regular_s += suspend.elapsed().as_secs_f64();
                    frame.snapshot = Some(snapshot);
                    req.prefill_frame = Some(frame);
                    self.observe_prefill_step(
                        chunk,
                        stage,
                        req.offset + chunk,
                        at,
                        cold_before,
                        bounded,
                    );
                    return Ok((0, vec![]));
                }
                continue;
            }
            let end = req.offset + chunk;
            let warm_end =
                end + usize::from(end < req.input.len() && !req.checkpoints.contains(&end));
            if self.manifest.mtp.is_some() && req.warm.tokens + 1 < warm_end {
                req.prefill_frame = Some(frame);
                self.observe_prefill_step(
                    chunk,
                    stage,
                    req.offset + chunk,
                    at,
                    cold_before,
                    bounded,
                );
                return Ok((0, vec![]));
            }
            self.prefill_costs
                .observe_at(chunk, end, frame.regular_s.max(0.000001));
            let tokens = self.after_prefill(req, chunk)?;
            self.observe_prefill_step(chunk, stage, end, at, cold_before, bounded);
            self.scheduler_statistics.prefill_tokens += chunk;
            *self
                .scheduler_statistics
                .prefill_chunk_histogram
                .entry(chunk)
                .or_default() += 1;
            return Ok((chunk, tokens));
        }
    }
    fn observe_prefill_step(
        &mut self,
        chunk: usize,
        stage: usize,
        context: usize,
        began: Instant,
        before: crate::scheduler::BatchExecutionStatistics,
        bounded: bool,
    ) {
        let seconds = began.elapsed().as_secs_f64();
        let after = self.execution.batch_statistics.get();
        let cold = after.kernel_load_s - before.kernel_load_s + after.sequence_capture_s
            - before.sequence_capture_s;
        self.prefill_costs
            .observe_stage(chunk, stage, context, (seconds - cold).max(0.000001));
        if bounded {
            self.scheduler_statistics.max_bounded_prefill_s =
                self.scheduler_statistics.max_bounded_prefill_s.max(seconds);
        }
    }
}
