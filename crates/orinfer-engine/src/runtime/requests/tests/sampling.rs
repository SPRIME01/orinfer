//! Compare complete sampled requests and committed state using the same weights.
use super::*;

#[test]
#[ignore = "Requires GPU sampler package and the exclusive GPU experiment lock"]
fn validate_gpu_sampling_requests() {
    let f: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!f.output.exists());
    let mut model = ModelRuntime::load_with_options(
        &f.model,
        LoadOptions {
            cuda_graph: f.cuda_graph.parse().unwrap(),
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let sampler = model
        .manifest
        .sampling
        .clone()
        .expect("GPU sampling contract");
    let options = scheduler::Options::default();
    let mut results = vec![];
    for (k, p) in [(0, 1.), (0, 0.9), (40, 0.8)] {
        let mut expected = None;
        let mut timings = vec![];
        for gpu in [false, true] {
            model.manifest.sampling = gpu.then_some(sampler.clone());
            let mut source = input(&f.cases[0], 64);
            source.sampling = crate::sampling::Options {
                temperature: 0.8,
                top_k: k,
                top_p: p,
                presence_penalty: 0.2,
                frequency_penalty: 0.1,
                repetition_penalty: 1.1,
                seed: crate::sampling::EVALUATION_SEED,
                ..Default::default()
            };
            let mut req = model.start_request(source, &|| false).unwrap();
            let start = Instant::now();
            let mut output = vec![];
            let mut maintenance_calls = 0;
            while !req.is_finished() {
                for step in model
                    .advance_requests_with(&mut [&mut req], &options, || {
                        maintenance_calls += 1;
                    })
                    .unwrap()
                {
                    output.extend(step.tokens);
                }
            }
            let elapsed = start.elapsed().as_secs_f64();
            assert_eq!(output.len(), 64);
            assert!(maintenance_calls > 0);
            let final_state = single_decode::state(&mut model, &mut req);
            if let Some((tokens, state)) = &expected {
                assert_eq!(
                    &output, tokens,
                    "GPU sampling changed seeded request output"
                );
                assert_eq!(
                    &final_state, state,
                    "GPU sampling changed committed causal state"
                );
            } else {
                expected = Some((output.clone(), final_state));
            }
            model.finish_request(&mut req, false).unwrap();
            timings.push(json!({"gpu":gpu,"seconds":elapsed,"tokens":output.len(),"request_tps":output.len() as f64 / elapsed,"maintenance_calls":maintenance_calls}));
        }
        results.push(json!({"top_k":k,"top_p":p,"seeded_tokens_equal":true,"committed_state_equal":true,"timings":timings}));
    }
    std::fs::write(f.output, serde_json::to_vec_pretty(&results).unwrap()).unwrap();
}
