//! Resume a large forward pass at model-registered layer boundaries. Tokens
//! become committed only after its target body and shifted draft warm complete.
use super::*;
use crate::cuda::snapshot::{Range, Workspace};
use std::{cell::RefCell, rc::Rc};

pub(super) struct Frame {
    pub plan: crate::model::PrefillPlan,
    pub tokens: usize,
    next: usize,
    workspace: Option<Rc<RefCell<Workspace>>>,
    regular_s: f64,
}

impl ModelRuntime {
    pub(super) fn release_prefill_frame(&mut self, req: &mut RequestState) -> Result<()> {
        drop(req.prefill_frame.take());
        self.execution.collect_snapshots()
    }
    pub(super) fn prefill_stage_cost(&self, req: &RequestState, chunk: usize) -> f64 {
        let plan = req
            .prefill_frame
            .as_ref()
            .map(|f| &f.plan)
            .or_else(|| self.manifest.prefill_plan(chunk));
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
            .or_else(|| self.manifest.prefill_plan(chunk).cloned());
        let Some(plan) = plan.filter(|p| !p.stages.is_empty()) else {
            return self.advance_prefill(req, chunk, bounded);
        };
        if !plan.accepts(chunk)
            || req
                .prefill_frame
                .as_ref()
                .is_some_and(|f| f.tokens != chunk)
            || chunk > Self::prefill_boundary(req) - req.offset
        {
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
            self.prepare_prefill_inputs(
                &plan.prefill_program,
                &req.input[req.offset..req.offset + chunk],
                &req.input[..req.offset],
            )?;
            Frame {
                plan,
                tokens: chunk,
                next: 0,
                workspace: None,
                regular_s: at.elapsed().as_secs_f64(),
            }
        };
        if let Some(workspace) = &frame.workspace {
            frame.regular_s += self.execution.resume_workspace(workspace)?;
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
                self.launch_rows(&frame.plan.stages[stage], chunk, ExecutionPhase::Prefill)?;
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
                        self.launch_rows(&frame.plan.head_program, chunk, ExecutionPhase::Prefill)?;
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
                if frame.workspace.is_none() {
                    let reserve = Instant::now();
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
                    frame.workspace = self.execution.reserve_workspace(ranges)?;
                    frame.regular_s += reserve.elapsed().as_secs_f64();
                }
                if let Some(workspace) = &frame.workspace {
                    self.execution.retain_workspace(workspace)?;
                    self.scheduler_statistics.peak_prefill_frame_bytes = self
                        .scheduler_statistics
                        .peak_prefill_frame_bytes
                        .max(workspace.borrow().bytes());
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
            // The target body no longer needs the shared continuation buffers.
            // MTP warm and endpoint commit operate on request-private state.
            if frame.workspace.take().is_some() {
                self.execution.collect_snapshots()?;
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
