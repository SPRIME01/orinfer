use super::*;

#[test]
#[ignore = "Requires native Flash batch package and exclusive GPU experiment lock"]
fn validate_flash_decode_batches() {
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!fixture.output.exists());
    assert!((4..=64).contains(&fixture.numerical_tokens));
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            cuda_graph: fixture.cuda_graph.parse().unwrap(),
            // Compare identical cold prompt execution. A partial prefix hit
            // changes prompt chunk boundaries and tests a different path.
            prefix_cache_bytes: 0,
            mtp_drafts: fixture.mtp_drafts,

            ..Default::default()
        },
    )
    .unwrap();
    assert!(!model.manifest.batch_profiles.is_empty());
    let spec = model.manifest.mtp.take();
    let options = scheduler::Options {
        memory_reserve_bytes: 256 << 20,
        ..Default::default()
    };
    // After real batched target steps, speculative execution must be able to
    // rebuild the draft from each request's captured hidden/state history.
    model.manifest.mtp = spec;
    let mtp_tested = model.manifest.mtp.is_some();
    if model.manifest.mtp.is_some() {
        let mut requests = vec![];
        for i in 0..8 {
            let item = input(&fixture.cases[i % fixture.cases.len()], 12);
            // Released GPU/host arenas can take a moment to appear in the
            // admission counters. Match the serving worker's retry behavior.
            let began = Instant::now();
            let admitted = loop {
                if model.can_admit_request(&item, &options).unwrap() {
                    break true;
                }
                if began.elapsed() >= std::time::Duration::from_secs(2) {
                    break false;
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            };
            if !admitted {
                eprintln!(
                    "MTP admission deferred after {} requests: free {}, fixed {}, KV {}, workspace {} bytes",
                    requests.len(),
                    model.admission_free_bytes().unwrap(),
                    model
                        .execution
                        .additional_state_bytes(&model.request_limits(&item).unwrap())
                        .unwrap(),
                    model
                        .execution
                        .additional_kv_bytes(
                            model.validate_generation(&item).unwrap(),
                            &model.request_limits(&item).unwrap()
                        )
                        .unwrap(),
                    model
                        .pending_workspace(model.validate_generation(&item).unwrap())
                        .unwrap()
                );
                break;
            }
            requests.push(model.start_request(item, &|| false).unwrap());
        }
        assert!(
            requests.len() >= 2,
            "Need two admitted requests for batch-to-MTP regression"
        );
        for request in &mut requests {
            eprintln!(
                "MTP prefill: slot {}, {} tokens",
                request.slot,
                request.input.len()
            );
            while request.prefilling {
                model.advance_requests(&mut [request], &options).unwrap();
            }
        }
        while requests.iter().any(|r| !r.is_finished() && r.generated < 3) {
            model
                .advance_requests(&mut requests.iter_mut().collect::<Vec<_>>(), &options)
                .unwrap();
        }
        assert!(
            requests
                .iter()
                .all(|r| r.generated >= 3 && r.failure().is_none())
        );
        for r in &mut requests[1..] {
            model.finish_request(r, false).unwrap();
        }
        let speculative_before = model.scheduler_statistics.speculative_iterations;
        while !requests[0].is_finished() {
            model
                .advance_requests(&mut [&mut requests[0]], &options)
                .unwrap();
        }
        assert!(requests[0].failure().is_none());
        assert!(model.scheduler_statistics.speculative_iterations > speculative_before);
        model.finish_request(&mut requests[0], true).unwrap();
    }
    let mtp_statistics = serde_json::to_value(model.scheduling_statistics()).unwrap();
    // Draft/verification working memory can reduce admission after its first
    // use. Validate ordinary batches with the real --mtp-drafts 0 deployment,
    // rather than silently skipping their numerical coverage under pressure.
    let mut model = if mtp_tested {
        eprintln!("Batch-to-MTP phase complete; reloading target-only model");
        drop(model);
        ModelRuntime::load_with_options(
            &fixture.model,
            LoadOptions {
                cuda_graph: fixture.cuda_graph.parse().unwrap(),
                prefix_cache_bytes: 0,
                mtp_drafts: Some(0),

                ..Default::default()
            },
        )
        .unwrap()
    } else {
        model
    };
    let mut rows = vec![];
    let batches = if fixture.batches.is_empty() {
        vec![2, 3, 4, 5, 7, 8, 9, 11, 16, 32]
    } else {
        fixture.batches.clone()
    };
    assert!(batches.contains(&8) && batches.iter().all(|n| (2..=128).contains(n)));
    for &count in &batches {
        eprintln!("Numerical batch {count}: admitting and prefilling");
        let mut requests = vec![];
        for i in 0..count {
            let item = input(
                &fixture.cases[i % fixture.cases.len()],
                fixture.numerical_tokens,
            );
            if !model.can_admit_request(&item, &options).unwrap() {
                break;
            }
            requests.push(model.start_request(item, &|| false).unwrap());
        }
        if requests.len() != count {
            let row =
                json!({"requested_batch":count,"admitted":requests.len(),"memory_deferred":true});
            eprintln!("{row}");
            rows.push(row);
            for request in &mut requests {
                model.finish_request(request, false).unwrap();
            }
            continue;
        }
        let mut outputs = vec![vec![]; count];
        let mut scores = vec![vec![]; count];
        // Complete every prompt before measuring decode. Otherwise a short
        // request can finish while its peers are still prefilling, and an
        // admitted cohort does not prove that the batch kernel actually ran.
        for i in 0..count {
            while requests[i].prefilling {
                for step in model
                    .advance_requests(&mut [&mut requests[i]], &options)
                    .unwrap()
                {
                    assert_eq!(step.request, 0);
                    assert_eq!(step.tokens.len(), 1);
                    outputs[i].extend(step.tokens);
                    scores[i].push(distribution(&mut model, &mut requests[i]));
                }
            }
            assert!(!requests[i].is_finished());
        }
        let batches_before = model
            .scheduler_statistics
            .batch_histogram
            .get(&count)
            .copied()
            .unwrap_or(0);
        while requests.iter().any(|r| !r.is_finished()) {
            let mut active: Vec<_> = requests.iter_mut().collect();
            for step in model.advance_requests(&mut active, &options).unwrap() {
                assert_eq!(step.tokens.len(), 1);
                outputs[step.request].extend(step.tokens);
                scores[step.request].push(distribution(&mut model, &mut requests[step.request]));
            }
        }
        assert!(
            model.scheduler_statistics.batch_histogram[&count] > batches_before,
            "The admitted cohort must execute a real decode batch of {count}"
        );
        for request in &mut requests {
            model.finish_request(request, true).unwrap();
        }
        for i in 0..count {
            let mut baseline = model
                .start_request(
                    input(
                        &fixture.cases[i % fixture.cases.len()],
                        fixture.numerical_tokens,
                    ),
                    &|| false,
                )
                .unwrap();
            while baseline.prefilling {
                model
                    .advance_requests(&mut [&mut baseline], &options)
                    .unwrap();
            }
            let mut kl = 0.;
            let mut nll_delta = 0.;
            let mut top3 = 0;
            let mut max_kl: f64 = 0.;
            let mut max_nll_delta: f64 = 0.;
            for (step, &token) in outputs[i].iter().enumerate() {
                let b = distribution(&mut model, &mut baseline);
                let a = &scores[i][step];
                let step_kl = b.iter().zip(a).map(|(b, a)| b.exp() * (b - a)).sum::<f64>();
                let step_delta = b[token as usize] - a[token as usize];
                kl += step_kl;
                nll_delta += step_delta;
                max_kl = max_kl.max(step_kl);
                max_nll_delta = max_nll_delta.max(step_delta.abs());
                let mut ranked: Vec<_> = b.iter().enumerate().collect();
                ranked.select_nth_unstable_by(3, |a, b| b.1.total_cmp(a.1));
                top3 += usize::from(ranked[..3].iter().any(|(id, _)| *id == token as usize));
                if step + 1 < outputs[i].len() {
                    *baseline.history.last_mut().unwrap() = token;
                    model
                        .with_request(&mut baseline, |m, _| {
                            m.upload_ids(&m.manifest.token, &[token])
                        })
                        .unwrap();
                    model
                        .advance_requests(&mut [&mut baseline], &options)
                        .unwrap();
                }
            }
            model.finish_request(&mut baseline, false).unwrap();
            let n = outputs[i].len() as f64;
            let row = json!({"batch":count,"lane":i,"case":i % fixture.cases.len(),"same_history_mean_kl":kl/n,"mean_nll_delta":nll_delta/n,"max_kl":max_kl,"max_abs_nll_delta":max_nll_delta,"selected_in_reference_top3":top3 as f64/n,"tokens":outputs[i]});
            eprintln!("{row}");
            assert!(
                max_kl.is_finite() && max_kl < 0.05 && max_nll_delta < 0.1,
                "Batch changed target distribution: {row}"
            );
            rows.push(row);
        }
    }
    assert!(
        rows.iter().any(|r| r["batch"] == 8),
        "A real batch 8 numerical comparison is required"
    );
    if !model.manifest.dynamic_batch_kernels.is_empty() {
        for count in [3, 5, 7, 9] {
            if !batches.contains(&count) {
                continue;
            }
            assert!(
                rows.iter().any(|r| r["batch"] == count),
                "A real dynamic batch {count} numerical comparison is required"
            );
        }
        if rows
            .iter()
            .any(|r| r["batch"].as_u64().is_some_and(|n| !n.is_power_of_two()))
        {
            let statistics = model.scheduling_statistics().batch_execution;
            if fixture.cuda_graph == "off" {
                assert!(statistics.dynamic_direct_iterations > 0);
            } else {
                assert!(statistics.dynamic_graph_captures > 0);
                assert!(statistics.graph_hits > 0);
            }
        }
    }
    let mut survivor = model
        .start_request(input(&fixture.cases[0], 8), &|| false)
        .unwrap();
    while survivor.generated < 2 {
        model
            .advance_requests(&mut [&mut survivor], &options)
            .unwrap();
    }
    let before = state(&mut model, &mut survivor);
    let mut other = model
        .start_request(input(&fixture.cases[1], 8), &|| false)
        .unwrap();
    assert_eq!(before, state(&mut model, &mut survivor));
    model.finish_request(&mut other, false).unwrap();
    model.execution.reclaim_idle_state().unwrap();
    assert_eq!(before, state(&mut model, &mut survivor));
    let limits = model.request_limits(&input(&fixture.cases[1], 8)).unwrap();
    assert_eq!(
        model.execution.additional_state_bytes(&limits).unwrap(),
        0,
        "Choose the retained warm arena, not an earlier reclaimed arena"
    );
    let mut replacement = model
        .start_request(input(&fixture.cases[1], 8), &|| false)
        .unwrap();
    assert_ne!(replacement.slot, survivor.slot);
    assert_eq!(before, state(&mut model, &mut survivor));
    while !survivor.is_finished() || !replacement.is_finished() {
        model
            .advance_requests(&mut [&mut replacement, &mut survivor], &options)
            .unwrap();
    }
    model.finish_request(&mut survivor, true).unwrap();
    model.finish_request(&mut replacement, true).unwrap();
    drop(before);
    model.execution.reclaim_idle_graphs().unwrap();
    model.execution.reclaim_idle_state().unwrap();
    assert!(model.reserved_requests.is_empty());
    assert_eq!(model.execution.active_kv_bytes(), 0);
    model.execution.reclaim_idle_state().unwrap();
    assert_eq!(model.execution.resident_kv_bytes(), 0);
    std::fs::write(fixture.output,serde_json::to_vec_pretty(&json!({"cases":rows,"cancel_reuse_state_equal":true,"batch_to_mtp":mtp_tested,"mtp_scheduler":mtp_statistics,"scheduler":model.scheduling_statistics()})).unwrap()).unwrap();
}

pub(super) fn distribution(model: &mut ModelRuntime, request: &mut RequestState) -> Vec<f64> {
    let raw = model
        .with_request(request, |m, _| {
            m.execution
                .download_bytes(&m.manifest.logits, m.manifest.vocab * 4)
        })
        .unwrap();
    log_probabilities(&floats(&raw, crate::artifact::Dtype::F32))
}

#[test]
#[ignore = "Requires native batched verification package and exclusive GPU lock"]
fn validate_flash_batched_verification() {
    use super::super::speculation::Preparation;
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!fixture.output.exists());
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
    assert!(
        !model
            .manifest
            .mtp
            .as_ref()
            .unwrap()
            .batch_verification_tokens
            .is_empty()
    );
    let spec = model.manifest.mtp.as_ref().unwrap().clone();
    let options = scheduler::Options::default();
    let mut results = vec![];
    for count in [2, 4, 8, 2, 4, 8] {
        let inputs = |lane| {
            let mut request = input(&fixture.cases[lane % fixture.cases.len()], 32);
            request.input_tokens.push((32 + lane) as u32);
            request
        };
        let mut references = vec![];
        let mut candidates = vec![];
        for lane in 0..count {
            let mut request = model.start_request(inputs(lane), &|| false).unwrap();
            while request.prefilling {
                model
                    .advance_requests(&mut [&mut request], &options)
                    .unwrap();
            }
            let Preparation::Round(round) = model
                .with_request(&mut request, |m, r| m.prepare_speculative_round(r, 4))
                .unwrap()
            else {
                panic!("Expected initialized MTP round")
            };
            candidates.push(round.verification.clone());
            model
                .with_request(&mut request, |m, r| m.verify_single_round(r, &round))
                .unwrap();
            let raw = model
                .with_request(&mut request, |m, _| {
                    m.execution
                        .download_bytes(&spec.verification_logits, 4 * m.manifest.vocab * 4)
                })
                .unwrap();
            let scores: Vec<_> = floats(&raw, crate::artifact::Dtype::F32)
                .chunks_exact(model.manifest.vocab)
                .map(log_probabilities)
                .collect();
            references.push(scores);
            model
                .with_request(&mut request, |m, r| m.commit_verified_round(r, round))
                .unwrap();
            model.finish_request(&mut request, false).unwrap();
        }
        let mut requests = vec![];
        let mut rounds = vec![];
        for (lane, candidate) in candidates.iter().enumerate() {
            let mut request = model.start_request(inputs(lane), &|| false).unwrap();
            while request.prefilling {
                model
                    .advance_requests(&mut [&mut request], &options)
                    .unwrap();
            }
            let Preparation::Round(round) = model
                .with_request(&mut request, |m, r| m.prepare_speculative_round(r, 4))
                .unwrap()
            else {
                panic!("Expected MTP round")
            };
            assert_eq!(&round.verification, candidate);
            model
                .execution
                .ensure_sequence_program(request.slot, &round.plan.program)
                .unwrap();
            model
                .upload_ids(&model.manifest.input, &round.verification)
                .unwrap();
            model
                .prepare_inputs(
                    &round.verification,
                    &request.history[..request.history.len() - 1],
                )
                .unwrap();
            model.upload_segment_controls(4).unwrap();
            rounds.push(round);
            requests.push(request);
        }
        let segments: Vec<_> = requests
            .iter()
            .map(|r| BatchSegment {
                slot: r.slot,
                tokens: 4,
            })
            .collect();
        let (plan, bindings) = model
            .model_package
            .batch_operation(
                &segments,
                orinfer_model_sdk::abi::BatchOperation::Verification,
                true,
            )
            .unwrap();
        assert!(bindings.is_empty());
        let at = Instant::now();
        model
            .execution
            .execute_batch_operation(
                orinfer_model_sdk::abi::BatchOperation::Verification,
                segments.iter().map(|s| (s.slot, s.tokens)).collect(),
                &plan,
                true,
                true,
            )
            .unwrap();
        let seconds = at.elapsed().as_secs_f64();
        let mut metrics = vec![];
        for (lane, round) in rounds.into_iter().enumerate() {
            let raw = model
                .with_request(&mut requests[lane], |m, _| {
                    m.execution
                        .download_bytes(&spec.verification_logits, 4 * m.manifest.vocab * 4)
                })
                .unwrap();
            for (row, actual) in floats(&raw, crate::artifact::Dtype::F32)
                .chunks_exact(model.manifest.vocab)
                .map(log_probabilities)
                .enumerate()
            {
                let reference = &references[lane][row];
                let kl: f64 = reference
                    .iter()
                    .zip(&actual)
                    .map(|(p, q)| p.exp() * (p - q))
                    .sum();
                let token = actual
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0;
                let higher = reference.iter().filter(|&&p| p > reference[token]).count();
                let delta = reference[token] - actual[token];
                assert!(
                    kl.is_finite() && kl < 0.05 && higher < 3,
                    "Batch {count}, lane {lane}, row {row}: KL {kl}, selected rank {higher}"
                );
                metrics.push(json!({"lane":lane,"row":row,"kl":kl,"nll_delta":delta,"top3":true}));
            }
            let output = model
                .with_request(&mut requests[lane], |m, r| {
                    m.commit_verified_round(r, round)
                })
                .unwrap();
            assert!(!output.is_empty());
            assert_eq!(
                model
                    .with_request(&mut requests[lane], |m, _| Ok(m
                        .read_control(&m.manifest.position)?
                        as usize))
                    .unwrap(),
                requests[lane].input.len() + requests[lane].generated - 1
            );
            while !requests[lane].is_finished() {
                model
                    .advance_requests(&mut [&mut requests[lane]], &options)
                    .unwrap();
            }
            assert!(requests[lane].failure().is_none());
            model.finish_request(&mut requests[lane], false).unwrap();
        }
        eprintln!("batched verification {count}: {seconds:.6}s");
        results.push(json!({"requests":count,"verification_s":seconds,"metrics":metrics}));
    }
    model.execution.reclaim_idle_graphs().unwrap();
    model.execution.reclaim_idle_state().unwrap();
    assert!(model.reserved_requests.is_empty());
    std::fs::write(
        fixture.output,
        serde_json::to_vec_pretty(&json!({"seed":20261002,"cases":results})).unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "Requires native Flash mixed package and exclusive GPU experiment lock"]
fn validate_flash_mixed_decoder_groups() {
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!fixture.output.exists());
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            cuda_graph: fixture.cuda_graph.parse().unwrap(),
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let options = scheduler::Options::default();
    let mut rows = vec![];
    let collect = std::env::var_os("ORINFER_COLLECT_MIXED_REFERENCE").is_some();
    let paired: Option<Vec<Vec<Vec<f64>>>> = std::env::var_os("ORINFER_MIXED_REFERENCE")
        .map(|path| crate::model::read(&PathBuf::from(path)).unwrap());
    let mut distributions = vec![];
    let mut cases = vec![
        (2, 128),
        (3, 128),
        (4, 128),
        (8, 128),
        (2, 512),
        (3, 512),
        (4, 512),
        (8, 512),
        (2, 128),
        (3, 128),
        (4, 128),
        (8, 128),
    ];
    let shared_graphs = fixture.cuda_graph == "full";
    if shared_graphs {
        cases.push((2, 128));
    }
    for (count, chunk) in cases {
        // Retain spare leases on repeated geometries so the stateless pieces
        // must survive a real membership change instead of reusing a slot graph.
        let mut spare = vec![];
        if shared_graphs && rows.len() >= 8 {
            for _ in 0..(rows.len() - 7) {
                spare.push(
                    model
                        .start_request(input(&fixture.cases[0], 1), &|| false)
                        .unwrap(),
                );
            }
        }
        let request_input = |lane| {
            let mut request = input(&fixture.cases[0], 4);
            // Distinct histories expose state-table aliasing across lanes.
            request.input_tokens.push(32 + lane as u32);
            request
        };
        let mut reference = vec![];
        for lane in 0..count {
            let mut request = model.start_request(request_input(lane), &|| false).unwrap();
            while request.prefilling {
                model
                    .advance_requests(&mut [&mut request], &options)
                    .unwrap();
            }
            model
                .advance_requests(&mut [&mut request], &options)
                .unwrap();
            reference.push(distribution(&mut model, &mut request));
            model.finish_request(&mut request, false).unwrap();
        }
        let mut requests = vec![];
        for lane in 0..count {
            let mut request = model.start_request(request_input(lane), &|| false).unwrap();
            while request.prefilling {
                model
                    .advance_requests(&mut [&mut request], &options)
                    .unwrap();
            }
            requests.push(request);
        }
        requests.push(
            model
                .start_request(input(&fixture.cases[1], 4), &|| false)
                .unwrap(),
        );
        let selected: Vec<_> = (0..count)
            .map(|i| (i, 1))
            .chain(std::iter::once((count, chunk)))
            .collect();
        let mut output = vec![];
        let at = Instant::now();
        model
            .execute_selected(
                &mut requests.iter_mut().collect::<Vec<_>>(),
                &selected,
                &mut output,
            )
            .unwrap();
        let elapsed = at.elapsed().as_secs_f64();
        assert_eq!(requests[count].offset, chunk);
        let mut scores = vec![];
        let mut group_distributions = vec![];
        for (lane, baseline) in reference.iter().enumerate() {
            assert!(output.iter().any(|s| s.request == lane));
            let actual = distribution(&mut model, &mut requests[lane]);
            let baseline = paired
                .as_ref()
                .map_or(baseline, |r| &r[rows.len() % r.len()][lane]);
            let kl: f64 = baseline
                .iter()
                .zip(&actual)
                .map(|(b, a)| b.exp() * (b - a))
                .sum();
            let token = *requests[lane].history.last().unwrap() as usize;
            let delta = baseline[token] - actual[token];
            assert!(
                collect || (kl.is_finite() && kl < 0.05 && delta.abs() < 0.1),
                "Group {count}, chunk {chunk}, lane {lane}: KL {kl}, NLL delta {delta}"
            );
            scores.push(json!({"lane":lane,"same_history_kl":kl,"nll_delta":delta}));
            group_distributions.push(actual);
        }
        let row = json!({"decoders":count,"chunk":chunk,"mixed_s":elapsed,"scores":scores});
        eprintln!("mixed decoder group {row}");
        rows.push(row);
        distributions.push(group_distributions);
        model.finish_request(&mut requests[count], false).unwrap();
        for request in &mut requests[..count] {
            while !request.is_finished() {
                model.advance_requests(&mut [request], &options).unwrap();
            }
            assert!(request.failure().is_none());
            model.finish_request(request, false).unwrap();
        }
        for request in &mut spare {
            model.finish_request(request, false).unwrap();
        }
    }
    if shared_graphs {
        let stats = model.execution.batch_statistics.get();
        assert!(
            stats.shared_graph_captures > 0 && stats.shared_graph_hits > 0,
            "Shared pieces must replay across membership changes: {stats:?}"
        );
    }
    model.execution.reclaim_idle_graphs().unwrap();
    model.execution.reclaim_idle_state().unwrap();
    assert!(model.reserved_requests.is_empty());
    if collect {
        std::fs::write(
            fixture.output.with_extension("distributions.json"),
            serde_json::to_vec(&distributions).unwrap(),
        )
        .unwrap();
    }
    std::fs::write(
        fixture.output,
        serde_json::to_vec_pretty(
            &json!({"seed":20261002,"cases":rows,"cancellation_passed":true}),
        )
        .unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "Requires native Flash mixed package and exclusive GPU experiment lock"]
fn validate_flash_mixed_segments() {
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!fixture.output.exists());
    assert!(fixture.cases.len() >= 2 && fixture.cases[1].input_tokens.len() >= 512);
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            cuda_graph: fixture.cuda_graph.parse().unwrap(),
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let options = scheduler::Options::default();
    let mut rows = vec![];
    for (round, chunk) in [16, 128, 512, 16, 128, 512].into_iter().enumerate() {
        let mut baseline = model
            .start_request(input(&fixture.cases[0], 4), &|| false)
            .unwrap();
        while baseline.prefilling {
            model
                .advance_requests(&mut [&mut baseline], &options)
                .unwrap();
        }
        let first = *baseline.history.last().unwrap();
        model
            .advance_requests(&mut [&mut baseline], &options)
            .unwrap();
        let reference = distribution(&mut model, &mut baseline);
        model.finish_request(&mut baseline, false).unwrap();

        let mut anchor = model
            .start_request(input(&fixture.cases[0], 4), &|| false)
            .unwrap();
        while anchor.prefilling {
            model
                .advance_requests(&mut [&mut anchor], &options)
                .unwrap();
        }
        assert_eq!(*anchor.history.last().unwrap(), first);
        let mut cold = model
            .start_request(input(&fixture.cases[1], 4), &|| false)
            .unwrap();
        let mut output = vec![];
        let before = model.execution.batch_statistics.get();
        let inputs_before = model.scheduler_statistics.batch_inputs_s;
        let plan_before = model.scheduler_statistics.batch_plan_s;
        let commit_before = model.scheduler_statistics.batch_commit_s;
        let at = Instant::now();
        model
            .execute_selected(
                &mut [&mut anchor, &mut cold],
                &[(0, 1), (1, chunk)],
                &mut output,
            )
            .unwrap();
        let elapsed = at.elapsed().as_secs_f64();
        assert_eq!(cold.offset, chunk);
        assert_eq!(anchor.generated, 2);
        assert!(output.iter().any(|s| s.request == 0));
        let actual = distribution(&mut model, &mut anchor);
        let kl: f64 = reference
            .iter()
            .zip(&actual)
            .map(|(b, a)| b.exp() * (b - a))
            .sum();
        let token = *anchor.history.last().unwrap();
        let delta = reference[token as usize] - actual[token as usize];
        let after = model.execution.batch_statistics.get();
        let row = json!({"round":round,"chunk":chunk,"decode_tokens":1,"mixed_s":elapsed,
            "direct_s":after.direct_s-before.direct_s,"kernel_load_s":after.kernel_load_s-before.kernel_load_s,
            "inputs_s":model.scheduler_statistics.batch_inputs_s-inputs_before,
            "plan_s":model.scheduler_statistics.batch_plan_s-plan_before,
            "commit_s":model.scheduler_statistics.batch_commit_s-commit_before,
            "same_history_kl":kl,"nll_delta":delta});
        eprintln!("forced mixed {row}");
        assert!(
            kl.is_finite() && kl < 0.05 && delta.abs() < 0.1,
            "Mixed decoder distribution changed: {row}"
        );
        rows.push(row);
        model.finish_request(&mut cold, false).unwrap();
        // Cancelling a partial prompt cannot release or mutate the decoder's state.
        while !anchor.is_finished() {
            model
                .advance_requests(&mut [&mut anchor], &options)
                .unwrap();
        }
        assert!(anchor.failure().is_none());
        model.finish_request(&mut anchor, false).unwrap();
    }
    model.execution.reclaim_idle_graphs().unwrap();
    model.execution.reclaim_idle_state().unwrap();
    assert!(model.reserved_requests.is_empty());
    std::fs::write(
        fixture.output,
        serde_json::to_vec_pretty(
            &json!({"seed":20261002,"cases":rows,"cancellation_passed":true}),
        )
        .unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "Requires native Flash package and exclusive GPU experiment lock"]
fn validate_flash_bounded_prefill() {
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!fixture.output.exists());
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            cuda_graph: fixture.cuda_graph.parse().unwrap(),
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),

            ..Default::default()
        },
    )
    .unwrap();
    let options = scheduler::Options {
        prefill_budget_ms: 200.,
        ..Default::default()
    };
    let mut rows = vec![];
    for source in &fixture.cases {
        let mut anchor = model
            .start_request(input(&fixture.cases[0], 32), &|| false)
            .unwrap();
        while anchor.prefilling {
            model
                .advance_requests(&mut [&mut anchor], &options)
                .unwrap();
        }
        let mut cold = model.start_request(input(source, 4), &|| false).unwrap();
        let began = Instant::now();
        let mut chunks = vec![];
        let mut scores = vec![];
        let mut output = vec![];
        let bounded_before = model.scheduler_statistics.bounded_prefill_iterations;
        let mixed_before = model.scheduler_statistics.mixed_iterations;
        let mut decode_during_prefill = 0;
        while !cold.is_finished() || !anchor.is_finished() {
            let (offset, generated) = (cold.offset, anchor.generated);
            let waiting = cold.prefilling && !anchor.is_finished();
            for step in model
                .advance_requests(&mut [&mut anchor, &mut cold], &options)
                .unwrap()
            {
                if step.request == 1 {
                    output.extend(step.tokens);
                    scores.push(distribution(&mut model, &mut cold));
                }
            }
            if cold.offset > offset {
                chunks.push((cold.offset - offset, waiting));
            }
            if waiting {
                decode_during_prefill += anchor.generated - generated;
            }
        }
        let elapsed = began.elapsed().as_secs_f64();
        assert!(!output.is_empty() && output.len() <= 4);
        assert!(decode_during_prefill > 0);
        if model
            .manifest
            .batch_layout
            .as_ref()
            .is_some_and(|l| !l.profiles.is_empty())
        {
            assert!(model.scheduler_statistics.mixed_iterations > mixed_before);
        } else {
            assert!(model.scheduler_statistics.bounded_prefill_iterations > bounded_before);
        }
        // A cold-only 2048/4096 tile must not block an already-live decoder.
        assert!(
            chunks
                .iter()
                .filter(|(_, mixed)| *mixed)
                .all(|(n, _)| *n <= 512)
        );
        for request in [&mut cold, &mut anchor] {
            model.finish_request(request, false).unwrap();
        }
        let mut baseline = model.start_request(input(source, 4), &|| false).unwrap();
        while baseline.prefilling {
            model
                .advance_requests(&mut [&mut baseline], &options)
                .unwrap();
        }
        let (mut kl, mut nll, mut top3) = (0., 0., 0);
        for (step, &token) in output.iter().enumerate() {
            let b = distribution(&mut model, &mut baseline);
            let a = &scores[step];
            kl += b.iter().zip(a).map(|(b, a)| b.exp() * (b - a)).sum::<f64>();
            nll += b[token as usize] - a[token as usize];
            let mut ranked: Vec<_> = b.iter().enumerate().collect();
            ranked.select_nth_unstable_by(3, |a, b| b.1.total_cmp(a.1));
            top3 += usize::from(ranked[..3].iter().any(|(id, _)| *id == token as usize));
            if step + 1 < output.len() {
                *baseline.history.last_mut().unwrap() = token;
                model
                    .with_request(&mut baseline, |m, _| {
                        m.upload_ids(&m.manifest.token, &[token])
                    })
                    .unwrap();
                model
                    .advance_requests(&mut [&mut baseline], &options)
                    .unwrap();
            }
        }
        model.finish_request(&mut baseline, false).unwrap();
        let n = output.len() as f64;
        let row = json!({"prompt_tokens":source.input_tokens.len(),"images":source.images.len(),
            "chunks":chunks,"decoder_tokens_during_prefill":decode_during_prefill,
            "wall_s":elapsed,"same_history_mean_kl":kl/n,"mean_nll_delta":nll/n,
            "selected_in_reference_top3":top3 as f64/n,"tokens":output});
        eprintln!("bounded prefill {row}");
        assert!(
            kl.is_finite() && nll.is_finite() && kl / n < 0.05 && (nll / n).abs() < 0.1,
            "Bounded chunks changed token quality: {row}"
        );
        rows.push(row);
    }
    model.execution.reclaim_idle_graphs().unwrap();
    model.execution.reclaim_idle_state().unwrap();
    assert!(model.reserved_requests.is_empty());
    assert_eq!(model.execution.resident_kv_bytes(), 0);
    std::fs::write(fixture.output,serde_json::to_vec_pretty(&json!({
        "seed":20261002,"scope":"Same quantized weights, cold large-chunk reference; not BF16 quantization acceptance",
        "cases":rows,"scheduler":model.scheduling_statistics(),"cleanup_passed":true
    })).unwrap()).unwrap();
}

#[test]
#[ignore = "Requires native Flash package and exclusive GPU experiment lock"]
fn validate_flash_prefix_cache() {
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!fixture.output.exists());
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            cuda_graph: fixture.cuda_graph.parse().unwrap(),
            prefix_cache_bytes: 2 << 30,
            mtp_drafts: Some(0),

            ..Default::default()
        },
    )
    .unwrap();
    let options = scheduler::Options::default();
    let mut rows = vec![];
    for source in &fixture.cases {
        for snapshot in model.prefix_cache.clear() {
            model.execution.release_snapshot(snapshot).unwrap();
        }
        let mut cold = model.start_request(input(source, 4), &|| false).unwrap();
        let mut expected = vec![];
        while !cold.is_finished() {
            for step in model.advance_requests(&mut [&mut cold], &options).unwrap() {
                expected.extend(step.tokens);
            }
        }
        let position = source.input_tokens.len() + cold.generated - 1;
        let ranges = model.prefix_ranges(position).unwrap();
        let committed = |model: &mut ModelRuntime, request: &mut RequestState| {
            model
                .with_request(request, |m, _| {
                    ranges
                        .iter()
                        .map(|(name, range)| {
                            let bytes = m
                                .execution
                                .download_bytes(name, range.offset + range.bytes)?;
                            Ok((name.clone(), bytes[range.offset..].to_vec()))
                        })
                        .collect::<Result<BTreeMap<String, Vec<u8>>>>()
                })
                .unwrap()
        };
        let expected_state = committed(&mut model, &mut cold);
        model.finish_request(&mut cold, true).unwrap();
        let mut warm = model.start_request(input(source, 4), &|| false).unwrap();
        assert_eq!(
            warm.prefix.statistics.cached_tokens,
            source.input_tokens.len()
        );
        let mut actual = vec![];
        while !warm.is_finished() {
            for step in model.advance_requests(&mut [&mut warm], &options).unwrap() {
                actual.extend(step.tokens);
            }
        }
        assert_eq!(
            actual, expected,
            "Full prefix restore changed deterministic continuation"
        );
        assert_eq!(
            committed(&mut model, &mut warm),
            expected_state,
            "Full prefix restore changed committed KV/GDN/conv/PLE/index/pending state"
        );
        rows.push(json!({"prompt_tokens":source.input_tokens.len(),"images":source.images.len(),
            "cached_tokens":warm.prefix.statistics.cached_tokens,"restore_s":warm.prefix.statistics.restore_s,
            "tokens_equal":true,"committed_state_equal":true}));
        model.finish_request(&mut warm, true).unwrap();
    }
    for snapshot in model.prefix_cache.clear() {
        model.execution.release_snapshot(snapshot).unwrap();
    }
    model.execution.reclaim_idle_graphs().unwrap();
    model.execution.reclaim_idle_state().unwrap();
    assert!(model.reserved_requests.is_empty());
    assert_eq!(model.execution.resident_kv_bytes(), 0);
    std::fs::write(
        fixture.output,
        serde_json::to_vec_pretty(&json!({
            "seed":20261002,"cases":rows,"cleanup_passed":true
        }))
        .unwrap(),
    )
    .unwrap();
}
