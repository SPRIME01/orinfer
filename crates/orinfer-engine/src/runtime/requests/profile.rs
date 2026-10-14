//! Ignored diagnostics: real-model kernel captures, partition state and GPU graphs.
use super::*;
use crate::execution::Invocation;
use serde_json::json;
use std::cell::RefCell;

thread_local! {
    static MARKS: RefCell<Option<Vec<(&'static str, Instant)>>> = const { RefCell::new(None) };
}
pub(super) fn mark(name: &'static str) {
    MARKS.with(|m| {
        if let Some(m) = m.borrow_mut().as_mut() {
            m.push((name, Instant::now()));
        }
    });
}
fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn prefill_invocations(
    model: &ModelRuntime,
    program: &str,
    rows: usize,
) -> Result<Vec<Invocation>> {
    let operations = &model.manifest.programs[program];
    if let Some(template) = model
        .manifest
        .row_execution
        .as_ref()
        .filter(|s| s.programs.contains(program))
    {
        template.specialize(
            program,
            operations,
            &model.manifest.dynamic_batch_kernels,
            rows,
        )
    } else {
        Ok(operations
            .iter()
            .cloned()
            .map(|operation| Invocation {
                operation,
                sequence: None,
                launch: None,
                views: BTreeMap::new(),
            })
            .collect())
    }
}

#[test]
#[ignore = "Captures real kernel arguments; requires model and exclusive GPU experiment lock"]
fn capture_kernel_arguments() {
    let fixture: serde_json::Value = crate::model::read(&std::path::PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    let output = std::path::PathBuf::from(fixture["output"].as_str().unwrap());
    std::fs::create_dir(&output).unwrap();
    let targets: BTreeSet<String> = serde_json::from_value(fixture["kernels"].clone()).unwrap();
    assert!(!targets.is_empty());
    let mut model = ModelRuntime::load_with_options(
        std::path::Path::new(fixture["model"].as_str().unwrap()),
        LoadOptions {
            cuda_graph: crate::execution::CudaGraphMode::Off,
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),
            ..LoadOptions::default()
        },
    )
    .unwrap();
    let mut captures = vec![];
    for phase in ["prefill", "decode"] {
        let source: GenerationInput = serde_json::from_value(fixture["cases"][0].clone()).unwrap();
        assert!(source.images.is_empty());
        let mut request = model.start_request(source, &|| false).unwrap();
        if phase == "decode" {
            while request.prefilling {
                model
                    .advance_requests(&mut [&mut request], &scheduler::Options::default())
                    .unwrap();
            }
        }
        model.with_request(&mut request, |model, req| {
            let (tokens, history, program) = if phase == "prefill" {
                let (chunk, program, _) = model.manifest.select_prefill_plan(req.input.len())?;
                if chunk != req.input.len() {
                    return Err("Kernel capture requires a single full prefill chunk".into());
                }
                (&req.input[..], &req.input[..0], program.to_owned())
            } else {
                let last = req.history.len() - 1;
                (&req.history[last..], &req.history[..last], "decode".to_owned())
            };
            model.execution.ensure_sequence_program(req.slot, &program)?;
            model.upload_ids(if phase == "decode" { &model.manifest.token } else { &model.manifest.input }, tokens)?;
            model.upload_segment_controls(tokens.len())?;
            model.prepare_inputs(tokens, history)?;
            let mut remaining = targets.iter().filter(|name| name.starts_with(&format!("{program}/"))).cloned().collect::<BTreeSet<_>>();
            for operation in model.manifest.programs[&program].clone() {
                let selected = match &operation {
                    crate::model::Operation::Kernel { name } if remaining.contains(name) => Some(name.clone()),
                    _ => None,
                };
                let mut records = BTreeMap::new();
                if let Some(name) = &selected {
                    let spec = model.manifest.kernels.iter().find(|k| k.name == *name).unwrap();
                    let directory = output.join(name.replace('/', "-"));
                    std::fs::create_dir(&directory).map_err(|e| e.to_string())?;
                    for argument in &spec.args {
                        let buffer_name = match argument {
                            crate::artifact::Argument::Buffer { name } | crate::artifact::Argument::BufferSlice { name, .. } => name,
                            _ => return Err("Kernel capture requires pointer-only arguments".into()),
                        };
                        let b = model.manifest.buffers.iter().find(|b| b.name == *buffer_name).unwrap();
                        if b.data.is_some() || records.contains_key(buffer_name) { continue; }
                        let bytes = model.execution.session.virtual_buffers.borrow().get(buffer_name)
                            .map_or(model.execution.sizes[buffer_name], |v| v.mapped.min(model.execution.sizes[buffer_name]));
                        let raw = model.execution.download_bytes(buffer_name, bytes)?;
                        let file = format!("{}.before.bin", records.len());
                        std::fs::write(directory.join(&file), &raw).map_err(|e| e.to_string())?;
                        records.insert(buffer_name.clone(), json!({"file":file,"bytes":bytes,"sha256":crate::artifact::sha256(&raw)}));
                    }
                }
                model.execution.execute_batch(vec![], &[Invocation { operation, sequence: None, launch: None, views: BTreeMap::new() }], false, false)?;
                if let Some(name) = selected {
                    let directory = output.join(name.replace('/', "-"));
                    for (index, (buffer, record)) in records.iter_mut().enumerate() {
                        let raw = model.execution.download_bytes(buffer, record["bytes"].as_u64().unwrap() as usize)?;
                        let file = format!("{index}.after.bin");
                        std::fs::write(directory.join(&file), &raw).map_err(|e| e.to_string())?;
                        record["after"] = json!({"file":file,"sha256":crate::artifact::sha256(&raw)});
                    }
                    captures.push(json!({"kernel":name,"directory":directory.file_name().unwrap().to_str().unwrap(),"buffers":records}));
                    remaining.remove(&name);
                    if remaining.is_empty() { break; }
                }
            }
            if !remaining.is_empty() { return Err("Selected capture kernel is absent from its program".into()); }
            Ok(())
        }).unwrap();
        model.finish_request(&mut request, false).unwrap();
    }
    assert_eq!(captures.len(), targets.len());
    std::fs::write(output.join("capture.json"), serde_json::to_vec_pretty(&json!({
        "model":fixture["model"],"fingerprint":model.stats.manifest_sha256,"seed":20261002,"captures":captures,
        "scope":"Real first prefill and decode kernel inputs and outputs, MTP off. State is request-private; individual launches for offline profiling, not performance acceptance. Weight payloads remain in the source model."
    })).unwrap()).unwrap();
}

#[test]
#[ignore = "Requires real model, tokenized requests and exclusive GPU experiment lock"]
fn profile_cached_generation() {
    let fixture: serde_json::Value = crate::model::read(&std::path::PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    let output = std::path::PathBuf::from(fixture["output"].as_str().unwrap());
    assert!(!output.exists());
    let sources: Vec<GenerationInput> = serde_json::from_value(fixture["cases"].clone()).unwrap();
    let repetitions = fixture["repetitions"].as_u64().unwrap_or(2) as usize;
    let eos: Vec<u32> = serde_json::from_value(fixture["eos"].clone()).unwrap();
    let prefix_cache_bytes = fixture["prefix_cache_bytes"]
        .as_u64()
        .map(|bytes| bytes as usize)
        .unwrap_or(12usize << 30);
    let mut model = ModelRuntime::load_with_options(
        std::path::Path::new(fixture["model"].as_str().unwrap()),
        LoadOptions {
            cuda_graph: crate::execution::CudaGraphMode::DecodeOnly,
            prefix_cache_bytes,
            mtp_drafts: Some(fixture["mtp_drafts"].as_u64().unwrap_or(3) as usize),
            ..LoadOptions::default()
        },
    )
    .unwrap();
    if let Some(regions) = fixture.get("parallel_regions") {
        *model.execution.parallel_regions.borrow_mut() =
            serde_json::from_value(regions.clone()).unwrap();
    }
    model
        .execution
        .parallel_gate_first
        .set(fixture["parallel_gate_first"].as_bool().unwrap_or(false));
    if fixture.get("parallel_priority").is_some() {
        *model.execution.parallel_priority.borrow_mut() = fixture["parallel_priority"]
            .as_str()
            .unwrap_or("off")
            .to_owned();
    }
    let mut rows = vec![];
    let mut deterministic = BTreeMap::new();
    for (case, source) in sources.iter().enumerate() {
        for repetition in 0..=repetitions {
            let tracing = fixture["nsys"].as_bool() == Some(true) && case == 0 && repetition == 1;
            if tracing {
                // SAFETY: Current CUDA context; Nsight owns collection for this diagnostic.
                unsafe {
                    crate::cuda::check(
                        cudarc::driver::sys::cuProfilerStart(),
                        "start diagnostic trace",
                    )
                    .unwrap();
                }
            }
            let started = Instant::now();
            let mut request = model
                .start_request(
                    GenerationInput {
                        input_tokens: source.input_tokens.clone(),
                        images: source.images.clone(),
                        max_new_tokens: source.max_new_tokens,
                        sampling: source.sampling.clone(),
                        prefix_hints: source.prefix_hints.clone(),
                    },
                    &|| false,
                )
                .unwrap();
            let mut tokens = vec![];
            let mut arrivals = vec![];
            let mut first = None;
            let mut first_count = 0;
            let mut ended = false;
            while !request.is_finished() && !ended {
                // Equal proposal widths isolate kernel arithmetic from the
                // normal latency-adaptive profile selection during validation.
                if fixture["fixed_speculation"].as_bool() == Some(true) {
                    request.speculation_costs.clear();
                }
                let steps = model
                    .advance_requests(&mut [&mut request], &scheduler::Options::default())
                    .unwrap();
                for step in steps {
                    if step.tokens.is_empty() {
                        continue;
                    }
                    let elapsed = started.elapsed().as_secs_f64();
                    if first.is_none() {
                        first = Some(elapsed);
                        first_count = step.tokens.len();
                    }
                    ended |= step.tokens.iter().any(|token| eos.contains(token));
                    arrivals.push(json!({"elapsed_s":elapsed,"tokens":step.tokens.len()}));
                    tokens.extend(step.tokens);
                }
            }
            assert!(!tokens.is_empty());
            if repetition > 0 && prefix_cache_bytes > 0 {
                assert!(request.prefix.statistics.cached_tokens > 0);
            }
            if prefix_cache_bytes == 0 {
                assert_eq!(request.prefix.statistics.cached_tokens, 0);
            }
            if fixture["mtp_drafts"].as_u64() == Some(0) {
                assert_eq!(request.mtp.rounds, 0);
            }
            if let Some(previous) = deterministic.get(&case) {
                assert_eq!(&tokens, previous, "Cached generation changed seeded output");
            } else {
                deterministic.insert(case, tokens.clone());
            }
            let wall_s = started.elapsed().as_secs_f64();
            if tracing {
                // SAFETY: The timed request completed; stop diagnostic collection.
                unsafe {
                    crate::cuda::check(
                        cudarc::driver::sys::cuProfilerStop(),
                        "stop diagnostic trace",
                    )
                    .unwrap();
                }
            }
            let last = arrivals.last().unwrap()["elapsed_s"].as_f64().unwrap();
            let decode_s = last - first.unwrap();
            let decode_tokens = tokens.len() - first_count;
            // Download after the timed generation. Compare every live cached
            // KV/GDN/conv/PLE/MTP range across arithmetic-preserving packages.
            let state_hashes = (fixture["state_hashes"].as_bool() == Some(true)).then(|| {
                model.execution.sync().unwrap();
                let position = model.read_control(&model.manifest.position).unwrap() as usize;
                crate::runtime::mtp_gpu_tests::complete_prefix_state(&model, position)
            });
            let row = json!({"case":case,"repetition":repetition,"warmup":repetition==0,
                "prompt_tokens":source.input_tokens.len(),"output_tokens":tokens,
                "first_step_tokens":first_count,"decode_tokens":decode_tokens,
                "decode_s":decode_s,"decode_tps":(decode_s>0.).then(||decode_tokens as f64/decode_s),
                "wall_s":wall_s,"ttft_s":first,"arrivals":arrivals,
                "prompt_through_first_token_tps":source.input_tokens.len() as f64/first.unwrap(),
                "prefix":request.prefix.statistics,"mtp":request.mtp,
                "sampling":source.sampling,"state_hashes":state_hashes,
                "finish_reason":if ended {"stop"}else{"length"}});
            eprintln!(
                "cached generation case={case} repetition={repetition}: {} TPS",
                row["decode_tps"]
            );
            rows.push(row);
            model.finish_request(&mut request, true).unwrap();
            std::fs::write(&output,serde_json::to_vec_pretty(&json!({"rows":rows,
                "scope":"Real full autoregressive requests with fixture-selected prefix cache and MTP, normal scheduler. Committed tokens only; decode excludes the complete first emitted step. Prompt throughput includes request setup and first token selection. Warmup excluded; seeded output equality checked."})).unwrap()).unwrap();
        }
    }
}

#[test]
#[ignore = "Requires real model and exclusive GPU experiment lock"]
fn profile_speculative_verification() {
    let fixture: serde_json::Value = crate::model::read(&std::path::PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    let output = std::path::PathBuf::from(fixture["output"].as_str().unwrap());
    assert!(!output.exists());
    let sources: Vec<GenerationInput> = serde_json::from_value(fixture["cases"].clone()).unwrap();
    let widths: Vec<usize> = serde_json::from_value(fixture["widths"].clone()).unwrap();
    let mut model = ModelRuntime::load_with_options(
        std::path::Path::new(fixture["model"].as_str().unwrap()),
        LoadOptions {
            cuda_graph: crate::execution::CudaGraphMode::DecodeOnly,
            prefix_cache_bytes: 0,
            ..LoadOptions::default()
        },
    )
    .unwrap();
    let mut rows = vec![];
    for source in sources {
        for &width in &widths {
            let mut tokens = vec![];
            model
                .generate(
                    &source.input_tokens,
                    None,
                    width + 1,
                    &source.sampling,
                    || false,
                    |t| {
                        tokens.push(t);
                        true
                    },
                )
                .unwrap();
            assert!(tokens.len() >= width);
            let mut request = model
                .start_request(
                    GenerationInput {
                        input_tokens: source.input_tokens.clone(),
                        images: vec![],
                        max_new_tokens: width + 16,
                        sampling: source.sampling.clone(),
                        prefix_hints: vec![],
                    },
                    &|| false,
                )
                .unwrap();
            while request.prefilling {
                model
                    .advance_requests(&mut [&mut request], &scheduler::Options::default())
                    .unwrap();
            }
            let spec = model.manifest.mtp.clone().unwrap();
            let verify = spec
                .verification_plans
                .iter()
                .find(|p| p.tokens == width)
                .unwrap();
            let position = source.input_tokens.len() as u32;
            model
                .upload_ids(&model.manifest.input, &tokens[..width])
                .unwrap();
            model
                .prepare_verification_inputs(
                    &verify.program,
                    &tokens[..width],
                    &source.input_tokens,
                )
                .unwrap();
            model.upload_segment_controls(width).unwrap();
            // Materialize demand-mapped KV/workspace before capturing the
            // instrumented graph. Timing replays advance recurrent state and
            // are diagnostic only; they do not establish output quality.
            model
                .launch_program(&verify.program, ExecutionPhase::Decode)
                .unwrap();
            let plan: Vec<_> = model.manifest.programs[&verify.program]
                .iter()
                .cloned()
                .map(|operation| Invocation {
                    operation,
                    sequence: None,
                    launch: None,
                    views: BTreeMap::new(),
                })
                .collect();
            let trials = model
                .execution
                .profile_graph_operations_at_position(
                    &plan,
                    Some((&model.manifest.position, position)),
                )
                .unwrap();
            let operations: Vec<_> = plan
                .iter()
                .enumerate()
                .map(|(index, op)| {
                    let name = match &op.operation {
                        crate::model::Operation::Kernel { name } => name.clone(),
                        _ => category(&model, op),
                    };
                    let ms = median(
                        &trials
                            .iter()
                            .map(|t| f64::from(t[index]))
                            .collect::<Vec<_>>(),
                    );
                    json!({"name":name,"median_ms":ms})
                })
                .collect();
            rows.push(json!({"prompt_tokens":source.input_tokens.len(),"verification_tokens":width,
                "program":verify.program,"operations":operations,
                "instrumented_gpu_median_ms":median(&trials.iter().map(|t|t.iter().map(|x|f64::from(*x)).sum()).collect::<Vec<_>>())}));
            std::fs::write(&output,serde_json::to_vec_pretty(&json!({"rows":rows,
                "scope":"Real model, teacher-forced target tokens; instrumented verification graph at fixed position. Recurrent state advances between replays. Diagnostic only, excludes draft/commit/refresh and is not end-to-end TPS or quality acceptance."})).unwrap()).unwrap();
            model.finish_request(&mut request, false).unwrap();
        }
    }
}
fn category(model: &ModelRuntime, op: &Invocation) -> String {
    match &op.operation {
        crate::model::Operation::Kernel { name } => {
            let parts: Vec<_> = name.split('/').collect();
            if let Some(layer) = parts
                .get(1)
                .and_then(|s| s.strip_prefix("layer"))
                .and_then(|s| s.parse::<usize>().ok())
            {
                format!(
                    "{}/{}/{}",
                    model.manifest.batch_layout.as_ref().unwrap().layers[layer],
                    if parts[0].starts_with("batch_gdn_") {
                        "batched_mixer"
                    } else if op.sequence.is_some() {
                        "private"
                    } else {
                        "shared"
                    },
                    parts[2]
                )
            } else {
                name.clone()
            }
        }
        crate::model::Operation::Copy {
            source,
            destination,
            ..
        } => {
            if destination.ends_with("_History") {
                "copy/conv_history".into()
            } else {
                format!("copy/{source}/{destination}")
            }
        }
        crate::model::Operation::Zero { destination, .. } => format!("zero/{destination}"),
    }
}

#[test]
#[ignore = "Requires real prefill model and exclusive GPU experiment lock"]
fn capture_prefill_ffn_inputs() {
    let fixture: serde_json::Value = crate::model::read(&std::path::PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    let output = std::path::PathBuf::from(fixture["output"].as_str().unwrap());
    std::fs::create_dir(&output).unwrap();
    let sources: Vec<GenerationInput> = serde_json::from_value(fixture["cases"].clone()).unwrap();
    let mut model = ModelRuntime::load_with_options(
        std::path::Path::new(fixture["model"].as_str().unwrap()),
        LoadOptions {
            cuda_graph: crate::execution::CudaGraphMode::Off,
            prefix_cache_bytes: 0,
            ..LoadOptions::default()
        },
    )
    .unwrap();
    model.manifest.mtp = None;
    let mut inputs = BTreeMap::new();
    for source in sources {
        let (chunk, program, _) = model
            .manifest
            .select_prefill_plan(source.input_tokens.len())
            .unwrap();
        let program = program.to_owned();
        model
            .generate(
                &source.input_tokens,
                None,
                1,
                &source.sampling,
                || false,
                |_| false,
            )
            .unwrap();
        model
            .execution
            .reset_sequence(&model.manifest.reset_buffers)
            .unwrap();
        model
            .prepare_visual(&source.input_tokens, &[], &|| false)
            .unwrap();
        // reset_sequence releases demand-mapped KV. Manual kernel stepping
        // needs the same mapping/workspace preparation as submit_program.
        model
            .execution
            .ensure_sequence_program(0, &program)
            .unwrap();
        model
            .upload_ids(&model.manifest.input, &source.input_tokens[..chunk])
            .unwrap();
        for operation in model.manifest.programs[&program].clone() {
            if let crate::model::Operation::Kernel { name } = &operation {
                let spec = model
                    .manifest
                    .kernels
                    .iter()
                    .find(|k| k.name == *name)
                    .unwrap();
                for family in ["GateUp", "Down"] {
                    let prefix = format!("L0_{family}");
                    let expanded = spec.args.iter().any(|a| matches!(a,crate::artifact::Argument::Buffer { name } if name == "TemporaryW8"));
                    if name.starts_with(&format!("{program}/layer0/"))
                        && spec.args.iter().any(|a| matches!(a,crate::artifact::Argument::Buffer { name } if name == &format!("{prefix}_WS")))
                        && spec.args.iter().any(|a| matches!(a,crate::artifact::Argument::Buffer { name } if name == "TemporaryA8"))
                    {
                        let scales = model.manifest.buffers.iter().find(|b| b.name == format!("{prefix}_S")).unwrap();
                        let width = scales.shape[1] * 128;
                        let activation = model.execution.download_bytes("TemporaryA8", chunk * width).unwrap();
                        let scale = model.execution.download_bytes("TemporaryAS", chunk * 2).unwrap();
                        let key = format!("{prefix}_m{chunk}");
                        std::fs::write(output.join(format!("{key}.a8")), &activation).unwrap();
                        std::fs::write(output.join(format!("{key}.scale.f16")), &scale).unwrap();
                        let mut entry = json!({"rows":chunk,"width":width,
                            "activation_sha256":crate::artifact::sha256(&activation),
                            "scale_sha256":crate::artifact::sha256(&scale)});
                        if expanded {
                            let rows = scales.shape[0];
                            let weight = model.execution.download_bytes("TemporaryW8", rows * width).unwrap();
                            std::fs::write(output.join(format!("{key}.w8")), &weight).unwrap();
                            entry["weight_shape"] = json!([rows, width]);
                            entry["weight_sha256"] = json!(crate::artifact::sha256(&weight));
                        }
                        inputs.insert(key, entry);
                    }
                }
            }
            model
                .execution
                .execute_batch(
                    vec![],
                    &[Invocation {
                        operation,
                        sequence: None,
                        launch: None,
                        views: BTreeMap::new(),
                    }],
                    false,
                    false,
                )
                .unwrap();
        }
    }
    std::fs::write(output.join("capture.json"),serde_json::to_vec_pretty(&json!({
        "fingerprint":model.stats.manifest_sha256,"seed":20261002,"inputs":inputs,
        "scope":"Actual A8 inputs and row scales before first-layer prefill FFN projections; no MTP."
    })).unwrap()).unwrap();
}

#[test]
#[ignore = "Requires real model and exclusive GPU experiment lock"]
fn capture_prefill_attention_inputs() {
    let fixture: serde_json::Value = crate::model::read(&std::path::PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    let output = std::path::PathBuf::from(fixture["output"].as_str().unwrap());
    std::fs::create_dir(&output).unwrap();
    let source: GenerationInput = serde_json::from_value(fixture["cases"][0].clone()).unwrap();
    assert!(source.images.is_empty());
    let mut model = ModelRuntime::load_with_options(
        std::path::Path::new(fixture["model"].as_str().unwrap()),
        LoadOptions {
            cuda_graph: crate::execution::CudaGraphMode::Off,
            prefix_cache_bytes: 0,
            ..LoadOptions::default()
        },
    )
    .unwrap();
    model.manifest.mtp = None;
    let length = source.input_tokens.len();
    let (chunk, program, _) = model.manifest.select_prefill_plan(length).unwrap();
    let program = program.to_owned();
    let base = length - chunk;
    assert!(base > 0 && base.is_multiple_of(chunk));
    model
        .generate(
            &source.input_tokens[..base],
            None,
            1,
            &source.sampling,
            || false,
            |_| false,
        )
        .unwrap();
    model
        .execution
        .ensure_sequence_program(0, &program)
        .unwrap();
    model
        .upload_ids(&model.manifest.input, &source.input_tokens[base..])
        .unwrap();
    for operation in model.manifest.programs[&program].clone() {
        if let crate::model::Operation::Kernel { name } = &operation {
            let spec = model
                .manifest
                .kernels
                .iter()
                .find(|k| k.name == *name)
                .unwrap();
            let reads = |buffer: &str| {
                spec.args.iter().any(
                    |a| matches!(a,crate::artifact::Argument::Buffer { name } if name == buffer),
                )
            };
            if reads("FullQ") && reads("PrefillK") && reads("PrefillV") {
                let mut records = BTreeMap::new();
                for (name, bytes) in [
                    ("FullQ", chunk * 6144 * 2),
                    ("FullGate", chunk * 6144 * 2),
                    ("PrefillK", length * 1024 * 2),
                    ("PrefillV", length * 1024 * 2),
                    ("Positions", chunk * 4),
                    ("SeqLength", 4),
                ] {
                    let raw = model.execution.download_bytes(name, bytes).unwrap();
                    let file = format!("{name}.bin");
                    std::fs::write(output.join(&file), &raw).unwrap();
                    records.insert(
                        name,
                        json!({"file":file,"bytes":bytes,"sha256":crate::artifact::sha256(&raw)}),
                    );
                }
                std::fs::write(output.join("capture.json"),serde_json::to_vec_pretty(&json!({
                    "fingerprint":model.stats.manifest_sha256,"seed":20261002,
                    "rows":chunk,"context":length,"base":base,"kernel":name,"inputs":records,
                    "scope":"Actual first full-attention layer inputs before the final real-model prefill chunk; no MTP."
                })).unwrap()).unwrap();
                return;
            }
        }
        model
            .execution
            .execute_batch(
                vec![],
                &[Invocation {
                    operation,
                    sequence: None,
                    launch: None,
                    views: BTreeMap::new(),
                }],
                false,
                false,
            )
            .unwrap();
    }
    panic!("Prepared model has no staged full-attention prefill reader");
}

#[test]
#[ignore = "Requires real model and exclusive GPU experiment lock"]
fn profile_prefill_stages() {
    let fixture: serde_json::Value = crate::model::read(&std::path::PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    let output = std::path::PathBuf::from(fixture["output"].as_str().unwrap());
    assert!(!output.exists());
    let sources: Vec<GenerationInput> = serde_json::from_value(fixture["cases"].clone()).unwrap();
    let mut model = ModelRuntime::load_with_options(
        std::path::Path::new(fixture["model"].as_str().unwrap()),
        LoadOptions {
            cuda_graph: crate::execution::CudaGraphMode::DecodeOnly,
            prefix_cache_bytes: 0,
            ..LoadOptions::default()
        },
    )
    .unwrap();
    model.manifest.mtp = None;
    let mut rows = vec![];
    for source in sources {
        let length = source.input_tokens.len();
        let (chunk, program, _) = model.manifest.select_prefill_plan(length).unwrap();
        let program = program.to_owned();
        let base = length - chunk;
        // Materialize the exact context and its KV/workspace before profiling.
        // Replays reset only the target position; they are timing diagnostics,
        // not a comparison of the recurrent state or generated text.
        model
            .generate(
                &source.input_tokens,
                None,
                1,
                &source.sampling,
                || false,
                |_| false,
            )
            .unwrap();
        model
            .upload_ids(&model.manifest.input, &source.input_tokens[base..])
            .unwrap();
        model
            .prepare_inputs(&source.input_tokens[base..], &source.input_tokens[..base])
            .unwrap();
        model
            .upload_ids(&model.manifest.position, &[base as u32])
            .unwrap();
        model
            .execution
            .ensure_sequence_program(0, &program)
            .unwrap();
        let plan = prefill_invocations(&model, &program, chunk).unwrap();
        let trials = model
            .execution
            .profile_graph_operations_at_position(
                &plan,
                Some((&model.manifest.position, base as u32)),
            )
            .unwrap();
        let mut groups: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for (index, op) in plan.iter().enumerate() {
            let values = groups
                .entry(category(&model, op))
                .or_insert(vec![0.; trials.len()]);
            for (trial, times) in trials.iter().enumerate() {
                values[trial] += f64::from(times[index]);
            }
        }
        rows.push(json!({"input_tokens":length,"profiled_chunk_tokens":chunk,"context_before_chunk":base,
            "program":program,"operation_count":plan.len(),
            "instrumented_graph_gpu_median_ms":median(&trials.iter().map(|t|t.iter().map(|x|f64::from(*x)).sum()).collect::<Vec<f64>>()),
            "gpu_group_median_ms":groups.iter().map(|(k,v)|(k.clone(),median(v))).collect::<BTreeMap<_,_>>() }));
        std::fs::write(&output,serde_json::to_vec_pretty(&json!({"rows":rows,
            "scope":"Last real-model prefill chunk at fixed target position; three instrumented graph trials after warm-up; head and host submission excluded. Recurrent state advances between replays. Diagnostic only, not end-to-end TPS or quality acceptance."})).unwrap()).unwrap();
    }
}

#[test]
#[ignore = "Captures real FFN inputs; requires model and exclusive GPU experiment lock"]
fn capture_decode_projections() {
    use std::io::Write;
    let fixture: serde_json::Value = crate::model::read(&std::path::PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    let output = std::path::PathBuf::from(fixture["output"].as_str().unwrap());
    std::fs::create_dir(&output).unwrap();
    let sources: Vec<GenerationInput> = serde_json::from_value(fixture["cases"].clone()).unwrap();
    let mut model = ModelRuntime::load_with_options(
        std::path::Path::new(fixture["model"].as_str().unwrap()),
        LoadOptions {
            cuda_graph: crate::execution::CudaGraphMode::Off,
            prefix_cache_bytes: 0,
            ..LoadOptions::default()
        },
    )
    .unwrap();
    model.manifest.mtp = None;
    let options = scheduler::Options::default();
    let mut samples = 0;
    for source in sources {
        let mut request = model.start_request(source, &|| false).unwrap();
        while request.prefilling {
            model
                .advance_requests(&mut [&mut request], &options)
                .unwrap();
        }
        model.with_request(&mut request, |model, req| {
            model.execution.ensure_sequence_program(req.slot, "decode")?;
            let operations = model.manifest.programs["decode"].clone();
            for _ in 0..8 {
                for operation in &operations {
                    if let crate::model::Operation::Kernel { name } = operation {
                        let pieces: Vec<_> = name.split('/').collect();
                        if let Some(layer) = pieces.get(1).and_then(|s| s.strip_prefix("layer")) {
                            let spec = model.manifest.kernels.iter().find(|k| k.name == *name).unwrap();
                            for (family, input) in [("GateUp", "Norm"), ("Down", "Activated")] {
                                if spec.args.iter().any(|a| matches!(a,
                                    crate::artifact::Argument::Buffer { name } if name == &format!("L{layer}_{family}_P"))) {
                                    if !spec.args.iter().any(|a| matches!(a,
                                        crate::artifact::Argument::Buffer { name } if name == input)) {
                                        return Err("Capture requires a source package with FP16 Norm/Activated inputs".into());
                                    }
                                    let stride = model.manifest.batch_layout.as_ref().unwrap().row_strides[input];
                                    let bytes = model.execution.download_bytes(input, stride)?;
                                    let mut file = std::fs::OpenOptions::new().create(true).append(true)
                                        .open(output.join(format!("L{layer}_{family}.f16"))).map_err(|e| e.to_string())?;
                                    file.write_all(&bytes).map_err(|e| e.to_string())?;
                                }
                            }
                        }
                    }
                    model.execution.execute_batch(vec![], &[Invocation {
                        operation: operation.clone(), sequence: None, launch: None, views: BTreeMap::new(),
                    }], false, false)?;
                }
                model.execution.sync()?;
                samples += 1;
            }
            Ok(())
        }).unwrap();
        model.finish_request(&mut request, false).unwrap();
    }
    std::fs::write(
        output.join("capture.json"),
        serde_json::to_vec_pretty(&json!({
            "model":fixture["model"],"fingerprint":model.stats.manifest_sha256,
            "samples_per_projection":samples,"seed":20261002,
            "scope":"Real sequential target-model decode inputs; no MTP."
        }))
        .unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "Requires real model and exclusive GPU experiment lock"]
fn profile_continuous_decode() {
    let fixture: serde_json::Value = crate::model::read(&std::path::PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    let output = std::path::PathBuf::from(fixture["output"].as_str().unwrap());
    assert!(!output.exists());
    let sources: Vec<GenerationInput> = serde_json::from_value(fixture["cases"].clone()).unwrap();
    let mut model = ModelRuntime::load_with_options(
        std::path::Path::new(fixture["model"].as_str().unwrap()),
        LoadOptions {
            cuda_graph: fixture["cuda_graph"].as_str().unwrap().parse().unwrap(),
            prefix_cache_bytes: 0,
            ..LoadOptions::default()
        },
    )
    .unwrap();
    // Isolate target decode. This is not a quality assessment or API benchmark.
    model.manifest.mtp = None;
    let options = scheduler::Options::default();
    let mut rows = vec![];
    let batches: Vec<usize> = fixture
        .get("batches")
        .map(|v| serde_json::from_value(v.clone()).unwrap())
        .unwrap_or_else(|| vec![1, 2, 4, 8, 16, 32]);
    assert!(batches.iter().all(|&b| (1..=128).contains(&b)));
    for count in batches {
        let mut requests: Vec<_> = (0..count)
            .map(|i| {
                let source = &sources[i % sources.len()];
                model
                    .start_request(
                        GenerationInput {
                            input_tokens: source.input_tokens.clone(),
                            images: vec![],
                            max_new_tokens: 160,
                            sampling: source.sampling.clone(),
                            prefix_hints: vec![],
                        },
                        &|| false,
                    )
                    .unwrap()
            })
            .collect();
        while requests.iter().any(|r| r.prefilling) {
            model
                .advance_requests(&mut requests.iter_mut().collect::<Vec<_>>(), &options)
                .unwrap();
        }
        for _ in 0..2 {
            model
                .advance_requests(&mut requests.iter_mut().collect::<Vec<_>>(), &options)
                .unwrap();
        }
        let mut walls = vec![];
        let mut phases: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let live_events = model.execution.profile_events(2).unwrap();
        model.execution.live_graph_ms.borrow_mut().clear();
        if count == 1 {
            model
                .execution
                .live_graph_events
                .set(Some([live_events[0], live_events[1]]));
        }
        for _ in 0..64 {
            MARKS.with(|m| *m.borrow_mut() = Some(vec![]));
            let started = Instant::now();
            let steps = model
                .advance_requests(&mut requests.iter_mut().collect::<Vec<_>>(), &options)
                .unwrap();
            mark("iteration_end");
            walls.push(started.elapsed().as_secs_f64() * 1000.);
            assert_eq!(steps.iter().map(|s| s.tokens.len()).sum::<usize>(), count);
            let marks = MARKS.with(|m| m.borrow_mut().take().unwrap());
            for pair in marks.windows(2) {
                phases
                    .entry(pair[1].0.into())
                    .or_default()
                    .push(pair[1].1.duration_since(pair[0].1).as_secs_f64() * 1000.);
            }
        }
        model.execution.live_graph_events.set(None);
        if fixture["live_only"].as_bool() == Some(true) {
            let gpu: Vec<_> = model
                .execution
                .live_graph_ms
                .borrow()
                .iter()
                .copied()
                .map(f64::from)
                .collect();
            assert_eq!(gpu.len(), walls.len());
            rows.push(json!({"batch":count,"iterations":walls.len(),
                "wall_mean_ms":walls.iter().sum::<f64>()/walls.len() as f64,
                "live_graph_mean_ms":gpu.iter().sum::<f64>()/gpu.len() as f64,
                "phases_mean_ms":phases.iter().map(|(k,v)|(k.clone(),v.iter().sum::<f64>()/v.len() as f64)).collect::<BTreeMap<_,_>>() }));
            model.finish_request(&mut requests[0], false).unwrap();
            continue;
        }
        let segments: Vec<_> = requests
            .iter()
            .map(|r| BatchSegment {
                slot: r.slot,
                tokens: 1,
            })
            .collect();
        let key: Vec<_> = segments.iter().map(|s| (s.slot, s.tokens)).collect();
        // Profiling replays advance state beyond the CPU histories. Reserve the
        // bounded extra steps before replay, including a possible slab boundary.
        for request in &requests {
            model
                .execution
                .ensure_sequence_program(request.slot, "prefill_m32")
                .unwrap();
        }
        let gpu: Vec<f64> = model
            .execution
            .profile_hot_graph(&key, 16)
            .unwrap()
            .into_iter()
            .map(f64::from)
            .collect();
        let plan = if count == 1 {
            model.manifest.programs["decode"]
                .iter()
                .cloned()
                .map(|operation| Invocation {
                    operation,
                    sequence: None,
                    launch: None,
                    views: BTreeMap::new(),
                })
                .collect()
        } else {
            model.model_package.batch_plan(&segments).unwrap()
        };
        let trials = model.execution.profile_graph_operations(&plan).unwrap();
        let mut groups: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let mut operations = Vec::with_capacity(plan.len());
        for (index, op) in plan.iter().enumerate() {
            let name = match &op.operation {
                crate::model::Operation::Kernel { name } => name.clone(),
                _ => category(&model, op),
            };
            operations.push(json!({"name":name,"median_ms":median(
                &trials.iter().map(|t|f64::from(t[index])).collect::<Vec<_>>()
            )}));
            let values = groups
                .entry(category(&model, op))
                .or_insert(vec![0.; trials.len()]);
            for (trial, times) in trials.iter().enumerate() {
                values[trial] += f64::from(times[index]);
            }
        }
        let mut subsets = BTreeMap::new();
        if let Some(filters) = fixture["kernel_subsets"].as_object() {
            // Isolated diagnostic graphs have stale inputs. Their times are
            // neither additive nor valid model executions; do this last.
            for (label, names) in filters {
                let names: BTreeSet<String> = serde_json::from_value(names.clone()).unwrap();
                let selected: Vec<_> = plan.iter().filter(|op| matches!(
                    &op.operation, crate::model::Operation::Kernel { name } if names.contains(name)
                )).cloned().collect();
                assert_eq!(selected.len(), names.len(), "Unknown subset kernel");
                assert!(!selected.is_empty());
                let times: Vec<_> = model
                    .execution
                    .profile_invocations_hot(&selected, 8)
                    .unwrap()
                    .into_iter()
                    .skip(1)
                    .map(f64::from)
                    .collect();
                subsets.insert(label.clone(), json!({"operations":selected.len(),
                    "median_ms":median(&times), "scope":"Isolated real kernels with stale inputs and two external timing events; diagnostic only, not additive and not model inference."}));
            }
        }
        let row = json!({"batch":count,"iterations":64,"wall_median_ms":median(&walls),
            "target_tokens_per_second":count as f64*1000./(walls.iter().sum::<f64>()/walls.len() as f64),
            "phases_median_ms":phases.iter().map(|(k,v)|(k.clone(),median(v))).collect::<BTreeMap<_,_>>(),
            "hot_graph_gpu_median_ms":median(&gpu),"operation_count":plan.len(),
            "instrumented_graph_gpu_median_ms":median(&trials.iter().map(|t| t.iter().map(|x|f64::from(*x)).sum()).collect::<Vec<f64>>()),
            "gpu_operations":operations,
            "isolated_subsets":subsets,
            "gpu_group_median_ms":groups.iter().map(|(k,v)|(k.clone(),median(v))).collect::<BTreeMap<_,_>>()});
        eprintln!("PROFILE {row}");
        rows.push(row);
        std::fs::write(&output,serde_json::to_vec_pretty(&json!({"scope":"64 real target decode iterations per fixed batch; no MTP, admission/prefill/API excluded. GPU operation timing uses external-event nodes in an instrumented replay, compare its overhead with the uninstrumented hot graph.","rows":rows})).unwrap()).unwrap();
        for request in &mut requests {
            model.finish_request(request, false).unwrap();
        }
    }
    if fixture["live_only"].as_bool() == Some(true) {
        std::fs::write(&output,serde_json::to_vec_pretty(&json!({"scope":"Live autoregressive iterations; two graph timing events, CPU stage timestamps; diagnostic only.","rows":rows})).unwrap()).unwrap();
    }
}

#[test]
#[ignore = "Captures real prefill partition intermediates; requires exclusive GPU lock"]
fn capture_prefill_partitions() {
    let spec: serde_json::Value = crate::model::read(&std::path::PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    let output = std::path::PathBuf::from(spec["output"].as_str().unwrap());
    std::fs::create_dir(&output).unwrap();
    let source: GenerationInput = serde_json::from_value(spec["case"].clone()).unwrap();
    let mut model = ModelRuntime::load_with_options(
        std::path::Path::new(spec["model"].as_str().unwrap()),
        LoadOptions {
            cuda_graph: crate::execution::CudaGraphMode::Off,
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let mut results = vec![];
    for partition in spec["partitions"].as_array().unwrap() {
        let label = partition["name"].as_str().unwrap();
        let chunks: Vec<usize> = serde_json::from_value(partition["chunks"].clone()).unwrap();
        assert_eq!(chunks.iter().sum::<usize>(), source.input_tokens.len());
        let mut req = model
            .start_request(
                GenerationInput {
                    input_tokens: source.input_tokens.clone(),
                    images: source.images.clone(),
                    max_new_tokens: 1,
                    sampling: source.sampling.clone(),
                    prefix_hints: vec![],
                },
                &|| false,
            )
            .unwrap();
        let directory = output.join(label);
        std::fs::create_dir(&directory).unwrap();
        let mut captures = vec![];
        for (index, &chunk) in chunks.iter().enumerate() {
            if index != 0 && index + 1 != chunks.len() {
                model
                    .with_request(&mut req, |m, r| m.advance_prefill(r, chunk, false))
                    .unwrap();
                continue;
            }
            let phase = if index == 0 { "first" } else { "last" };
            model
                .with_request(&mut req, |m, r| {
                    let (program, head) = if chunk == 1 {
                        ("decode".to_owned(), None)
                    } else {
                        let (_, program, head) = m.manifest.select_prefill_plan(chunk)?;
                        (program.to_owned(), Some(head.to_owned()))
                    };
                    m.upload_ids(
                        if chunk == 1 {
                            &m.manifest.token
                        } else {
                            &m.manifest.input
                        },
                        &r.input[r.offset..r.offset + chunk],
                    )?;
                    m.prepare_inputs(&r.input[r.offset..r.offset + chunk], &r.input[..r.offset])?;
                    m.upload_segment_controls(chunk)?;
                    m.execution.ensure_sequence_program(r.slot, &program)?;
                    for invocation in prefill_invocations(m, &program, chunk)? {
                        let name = match &invocation.operation {
                            crate::model::Operation::Kernel { name } => Some(name.clone()),
                            _ => None,
                        };
                        m.execution
                            .execute_batch(vec![], &[invocation], false, false)?;
                        if let Some(watches) = name
                            .as_ref()
                            .and_then(|n| spec["watches"].get(n))
                            .and_then(|v| v.as_array())
                        {
                            for watch in watches {
                                let buffer = watch["buffer"].as_str().unwrap();
                                let offset = watch["offset"].as_u64().unwrap_or(0) as usize;
                                let bytes = watch["bytes"].as_u64().unwrap() as usize;
                                let raw = m.execution.download_bytes(buffer, offset + bytes)?;
                                let file =
                                    format!("{phase}-{}.bin", watch["name"].as_str().unwrap());
                                std::fs::write(directory.join(&file), &raw[offset..])
                                    .map_err(|e| e.to_string())?;
                                captures.push(
                                    json!({"phase":phase,"kernel":name,"watch":watch,"file":file}),
                                );
                            }
                        }
                    }
                    if r.offset + chunk == r.input.len()
                        && let Some(head) = head
                    {
                        m.launch_rows(&head, chunk, ExecutionPhase::Prefill)?;
                    }
                    m.after_prefill(r, chunk)?;
                    Ok(())
                })
                .unwrap();
        }
        let raw = model
            .with_request(&mut req, |m, _| {
                m.execution
                    .download_bytes(&m.manifest.logits, m.manifest.vocab * 4)
            })
            .unwrap();
        std::fs::write(directory.join("logits.f32"), raw).unwrap();
        let state = if spec["capture_state"].as_bool() == Some(true) {
            model
                .with_request(&mut req, |m, _| {
                    m.prefix_ranges(source.input_tokens.len())?
                        .into_iter()
                        .map(|(name, range)| {
                            let raw = m
                                .execution
                                .download_bytes(&name, range.offset + range.bytes)?;
                            Ok((name, crate::artifact::sha256(&raw[range.offset..])))
                        })
                        .collect::<Result<BTreeMap<_, _>>>()
                })
                .unwrap()
        } else {
            BTreeMap::new()
        };
        model.finish_request(&mut req, false).unwrap();
        results.push(json!({"name":label,"chunks":chunks,"captures":captures,"state":state}));
        std::fs::write(
            output.join("capture.json"),
            serde_json::to_vec_pretty(&json!({"cases":results,"seed":20261002})).unwrap(),
        )
        .unwrap();
        eprintln!("prefill capture complete: {label}");
    }
}
