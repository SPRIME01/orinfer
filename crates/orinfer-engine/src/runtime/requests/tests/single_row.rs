use super::*;

fn committed_state(model: &mut ModelRuntime, req: &mut RequestState) -> BTreeMap<String, String> {
    model
        .with_request(req, |m, r| {
            m.prefix_ranges(r.input.len())?
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

#[test]
#[ignore = "Requires real dynamic prefill package and exclusive GPU experiment lock"]
fn validate_single_row_prefill() {
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
            mtp_drafts: fixture.mtp_drafts,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        model.manifest.select_prefill_plan(1).unwrap(),
        (1, "prefill", "head")
    );
    let mut rows = vec![];
    for source in &fixture.cases {
        let mut source = input(source, 4);
        source.input_tokens.truncate(1);
        assert!(source.images.is_empty());

        // Independently compare the same compressed weights using the original
        // single-row decode arithmetic, without draft warm-up or prefix reuse.
        let mtp = model.manifest.mtp.take();
        let plans = std::mem::take(&mut model.manifest.prefill_plans);
        let mut reference = model.start_request(input(&source, 4), &|| false).unwrap();
        model
            .with_request(&mut reference, |m, r| m.advance_prefill(r, 1, false))
            .unwrap();
        let expected = flash_batch::distribution(&mut model, &mut reference);
        model.finish_request(&mut reference, false).unwrap();
        model.manifest.prefill_plans = plans;

        let mut candidate = model.start_request(input(&source, 4), &|| false).unwrap();
        model
            .with_request(&mut candidate, |m, r| m.advance_prefill(r, 1, false))
            .unwrap();
        let actual = flash_batch::distribution(&mut model, &mut candidate);
        let kl: f64 = expected
            .iter()
            .zip(&actual)
            .map(|(a, b)| a.exp() * (a - b))
            .sum();
        let choice = actual
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        let mut ranked: Vec<_> = expected.iter().enumerate().collect();
        ranked.sort_unstable_by(|a, b| b.1.total_cmp(a.1));
        let top3 = ranked[..3].iter().any(|(token, _)| *token == choice);
        let nll_delta = expected[choice] - actual[choice];
        eprintln!(
            "Single-row token {}: KL={kl}, NLL delta={nll_delta}, top3={top3}",
            source.input_tokens[0]
        );
        // Use the same distribution/likelihood limits as the existing exact
        // tail checks: single-row GEMV and prefill MMA have different rounding.
        assert!(
            kl.is_finite() && kl.abs() < 0.05 && nll_delta.abs() < 0.1 && top3,
            "Single-row arithmetic: KL={kl}, NLL delta={nll_delta}, top3={top3}"
        );
        model.finish_request(&mut candidate, false).unwrap();
        model.manifest.mtp = mtp;

        let mut atomic = model.start_request(input(&source, 4), &|| false).unwrap();
        let expected_tokens = model
            .with_request(&mut atomic, |m, r| m.advance_prefill(r, 1, false))
            .unwrap()
            .1;
        let expected_state = committed_state(&mut model, &mut atomic);
        model.finish_request(&mut atomic, false).unwrap();

        // The scheduler's single selected row must execute prefill and capture
        // its hidden output, rather than the generation decode arena.
        let mut scheduled = model.start_request(input(&source, 4), &|| false).unwrap();
        let mut tokens = Vec::new();
        let mut iterations = 0;
        while scheduled.prefilling {
            let steps = model
                .advance_requests(
                    &mut [&mut scheduled],
                    &scheduler::Options {
                        memory_reserve_bytes: 0,
                        ..Default::default()
                    },
                )
                .unwrap();
            tokens.extend(steps.into_iter().flat_map(|s| s.tokens));
            iterations += 1;
            assert!(iterations <= 64, "Single-row prefill did not complete");
        }
        assert_eq!(tokens, expected_tokens);
        assert_eq!(committed_state(&mut model, &mut scheduled), expected_state);
        assert!(!scheduled.prefilling);
        let mut cancelled = model.start_request(input(&source, 4), &|| false).unwrap();
        model.finish_request(&mut cancelled, false).unwrap();
        assert_eq!(committed_state(&mut model, &mut scheduled), expected_state);
        model.finish_request(&mut scheduled, false).unwrap();
        rows.push(
            json!({"token":source.input_tokens[0], "decode_reference_kl":kl,
            "decode_reference_nll_delta":nll_delta,
            "decode_reference_top3":top3,"scheduled_tokens_equal":true,
            "complete_committed_state_equal":true,"cancellation_isolated":true,
            "scheduler_iterations":iterations}),
        );
    }
    std::fs::write(
        &fixture.output,
        serde_json::to_vec_pretty(&json!({"rows":rows,"seed":20261002})).unwrap(),
    )
    .unwrap();
}
