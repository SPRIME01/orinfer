//! Compare the unified B=1 template against the dedicated single-request path.
use super::*;

pub(super) fn start_decoder(model: &mut ModelRuntime, source: &GenerationInput) -> RequestState {
    let mut source = input(source, 128);
    source
        .sampling
        .logit_bias
        .extend([(248044, -100.), (248046, -100.)]);
    let mut req = model.start_request(source, &|| false).unwrap();
    while req.prefilling {
        model
            .with_request(&mut req, |m, r| {
                m.advance_prefill(r, r.input.len().min(4096), false)
            })
            .unwrap();
    }
    req
}

fn step(model: &mut ModelRuntime, req: &mut RequestState, unified: bool) {
    let before = req.generated;
    let mut output = Vec::new();
    if unified {
        model
            .execute_batch_selected(&mut [req], &[(0, 1)], &mut output)
            .unwrap();
    } else {
        let tokens = model
            .with_request(req, |m, r| {
                let last = r.history.len() - 1;
                m.prepare_inputs(&r.history[last..], &r.history[..last])?;
                m.upload_segment_controls(1)?;
                m.launch_program("decode", ExecutionPhase::Decode)?;
                m.commit_ordinary_token(r)
            })
            .unwrap();
        output.push(StepOutput { request: 0, tokens });
    }
    assert_eq!(req.generated, before + 1);
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].tokens.len(), 1);
    assert!(!req.is_finished() && req.failure().is_none());
}

pub(super) fn state(model: &mut ModelRuntime, req: &mut RequestState) -> BTreeMap<String, String> {
    model
        .with_request(req, |m, r| {
            m.prefix_ranges(r.input.len() + r.generated - 1)?
                .into_iter()
                .map(|(name, range)| {
                    let raw = m
                        .execution
                        .download_bytes(&name, range.offset + range.bytes)?;
                    Ok((name, crate::artifact::sha256(&raw[range.offset..])))
                })
                .collect()
        })
        .unwrap()
}

pub(super) fn load(fixture: &Fixture) -> ModelRuntime {
    let model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            cuda_graph: fixture.cuda_graph.parse().unwrap(),
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        model
            .manifest
            .programs
            .contains_key("flash_decode/packet/inputs")
    );
    model
}

#[test]
#[ignore = "Requires unified Flash package and exclusive GPU experiment lock"]
fn validate_flash_single_decode_template() {
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!fixture.output.exists());
    let mut model = load(&fixture);
    let mut evidence = Vec::new();
    for (case_index, source) in fixture.cases.iter().enumerate() {
        let mut reference = start_decoder(&mut model, source);
        let mut candidate = start_decoder(&mut model, source);
        let mut max_kl: f64 = 0.;
        let mut max_nll_delta: f64 = 0.;
        let mut top3 = 0;
        for _ in 0..8 {
            // Freeze the target history even if rounding changes token choice.
            let token = *reference.history.last().unwrap();
            *candidate.history.last_mut().unwrap() = token;
            model
                .with_request(&mut candidate, |m, _| {
                    m.upload_ids(&m.manifest.token, &[token])
                })
                .unwrap();
            step(&mut model, &mut reference, false);
            step(&mut model, &mut candidate, true);
            let expected = flash_batch::distribution(&mut model, &mut reference);
            let actual = flash_batch::distribution(&mut model, &mut candidate);
            let choice = *candidate.history.last().unwrap() as usize;
            let kl = expected
                .iter()
                .zip(&actual)
                .map(|(a, b)| a.exp() * (a - b))
                .sum::<f64>();
            let delta = expected[choice] - actual[choice];
            let mut ranked: Vec<_> = expected.iter().enumerate().collect();
            ranked.select_nth_unstable_by(3, |a, b| b.1.total_cmp(a.1));
            let within_top3 = ranked[..3].iter().any(|(id, _)| *id == choice);
            assert!(kl.is_finite() && kl.abs() < 0.05 && delta.abs() < 0.1 && within_top3);
            max_kl = max_kl.max(kl.abs());
            max_nll_delta = max_nll_delta.max(delta.abs());
            top3 += usize::from(within_top3);
            assert_eq!(
                state(&mut model, &mut candidate),
                state(&mut model, &mut reference),
                "B=1 changed committed request state"
            );
        }
        let snapshot = state(&mut model, &mut candidate);
        model.finish_request(&mut reference, false).unwrap();
        let mut replacement = start_decoder(&mut model, source);
        step(&mut model, &mut replacement, true);
        model.finish_request(&mut replacement, false).unwrap();
        assert_eq!(state(&mut model, &mut candidate), snapshot);
        // Exercise both directions across the same request's graph/state arenas.
        step(&mut model, &mut candidate, false);
        step(&mut model, &mut candidate, true);
        model.finish_request(&mut candidate, false).unwrap();
        evidence.push(json!({"case_index":case_index,"teacher_forced_steps":8,
            "max_kl":max_kl,"max_abs_nll_delta":max_nll_delta,"top3":top3,
            "committed_state_equal":true,"cancel_reuse_isolated":true}));
    }
    let stats = model.scheduling_statistics();
    if fixture.cuda_graph != "off" {
        assert!(stats.batch_execution.dynamic_graph_captures > 0);
        assert!(stats.batch_execution.graph_hits > 0);
    }
    std::fs::write(
        fixture.output,
        serde_json::to_vec_pretty(&json!({"cases":evidence,"scheduler":stats,
            "seed":20261002,"scope":"Same quantized model, dedicated versus unified B=1"}))
        .unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "Requires unified Flash package and exclusive GPU experiment lock"]
fn benchmark_flash_single_decode_paths() {
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!fixture.output.exists());
    let mut model = load(&fixture);
    let mut results = Vec::new();
    for (case_index, source) in fixture.cases.iter().enumerate() {
        for unified in [false, true, true, false, false, true] {
            let mut req = start_decoder(&mut model, source);
            let mut samples = Vec::new();
            for iteration in 0..64 {
                let at = Instant::now();
                step(&mut model, &mut req, unified);
                if iteration >= 4 {
                    samples.push(at.elapsed().as_secs_f64());
                }
            }
            let seconds: f64 = samples.iter().sum();
            let tps = samples.len() as f64 / seconds;
            let route = if unified { "unified_b1" } else { "dedicated" };
            eprintln!("Single decode {} {route}: {tps:.4} TPS", case_index);
            results.push(
                json!({"case_index":case_index,"route":route,"committed_tokens":samples.len(),
                "decode_tps":tps,"samples_s":samples}),
            );
            model.finish_request(&mut req, false).unwrap();
        }
    }
    std::fs::write(
        fixture.output,
        serde_json::to_vec_pretty(&json!({"cases":results,
            "scheduler":model.scheduling_statistics(),"seed":20261002,
            "scope":"Rust real-model decode including inputs, sampling, commit and synchronization; no MTP"}))
        .unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "Diagnostic graph replay; requires unified Flash package and exclusive GPU lock"]
fn profile_flash_single_decode_paths() {
    use crate::execution::Invocation;
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!fixture.output.exists());
    let mut model = load(&fixture);
    let mut rows = Vec::new();
    for unified in [false, true] {
        let mut req = start_decoder(&mut model, &fixture.cases[0]);
        for _ in 0..8 {
            step(&mut model, &mut req, unified);
        }
        let plan = if unified {
            model
                .prepare_batch_inputs(&mut [&mut req], &[(0, 1)])
                .unwrap();
            model
                .model_package
                .batch_plan(&[BatchSegment {
                    slot: req.slot,
                    tokens: 1,
                }])
                .unwrap()
        } else {
            model.execution.activate_sequence(req.slot).unwrap();
            model.manifest.programs["decode"]
                .iter()
                .cloned()
                .map(|operation| Invocation {
                    operation,
                    sequence: Some(req.slot),
                    launch: None,
                    views: BTreeMap::new(),
                })
                .collect()
        };
        // These bounded replays intentionally stop updating CPU histories.
        // The request reserved 128 output tokens; 8 + 8 + 4 remains within it.
        let hot = model.execution.profile_invocations_hot(&plan, 8).unwrap();
        let trials = model.execution.profile_graph_operations(&plan).unwrap();
        let kernels: BTreeMap<_, _> = model
            .manifest
            .kernels
            .iter()
            .map(|k| (k.name.as_str(), k))
            .collect();
        let mut operations = Vec::new();
        for (index, op) in plan.iter().enumerate() {
            let mut times: Vec<_> = trials.iter().map(|t| t[index]).collect();
            times.sort_by(f32::total_cmp);
            let kernel = if let crate::model::Operation::Kernel { name } = &op.operation {
                kernels.get(name.as_str()).copied()
            } else {
                None
            };
            operations.push(
                json!({"operation":op.operation,"median_ms":times[times.len()/2],
                "symbol":kernel.map(|k|&k.symbol),"args":kernel.map(|k|&k.args),
                "module_sha256":kernel.map(|k|&k.module.sha256)}),
            );
        }
        let route = if unified { "unified_b1" } else { "dedicated" };
        let row = json!({"route":route,"operation_count":plan.len(),
            "hot_graph_ms":hot,"instrumented_graph_ms":trials.iter()
                .map(|t|t.iter().sum::<f32>()).collect::<Vec<_>>(),"operations":operations});
        eprintln!(
            "PROFILE {route}: {} operations, GPU hot {hot:?}",
            plan.len()
        );
        rows.push(row);
        model.finish_request(&mut req, false).unwrap();
    }
    std::fs::write(
        fixture.output,
        serde_json::to_vec_pretty(&json!({"rows":rows,"seed":20261002,
            "scope":"Diagnostic stale-input graph replay, not model throughput; external events per operation perturb timing, compare with two-event hot graph"}))
        .unwrap(),
    )
    .unwrap();
}
