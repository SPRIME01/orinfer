//! The target's row arithmetic is shared; speculation changes causal commit only.
use super::*;

pub(super) fn register(m: &mut Manifest, moe: bool) -> Result<()> {
    let Some(spec) = m.mtp.as_ref() else {
        return Ok(());
    };
    if spec.draft_program != "mtp_draft"
        || !spec.commit_always
        || !spec.shared_verification_state
        || spec.position != "MtpStep"
        || spec.target_length != "SeqLength"
        || spec.accepted_inputs != "AcceptedInputs"
        || spec.hidden_ring.as_deref() != Some("MtpTargetHidden")
        || spec.warm_plans.len() != 1
        || spec.warm_plans[0].tokens != 512
        || spec.warm_plans[0].min_tokens != 1
        || spec.verification_plans.len() != 1
        || spec.verification_plans[0].tokens != 8
        || spec.verification_plans[0].min_tokens != 2
    {
        return Err("Unsupported Qwen compact MTP contract".into());
    }
    let hidden = m.batch_layout.as_ref().ok_or("Missing Qwen layout")?.hidden;
    let mut warm = kernels(
        "qwen_rows/mtp",
        &[
            "gather",
            "prepare",
            "embedding",
            "overlay",
            "concat",
            "fc_quant",
            "fc",
            "pre_norm",
            "pre_quant",
            "q",
            "k",
            "v",
            "kv_prepare",
            "attention",
            "attention_merge",
            "out_quant",
            "out",
            "post_norm",
            "post_quant",
        ],
    );
    let ffn = ffn_names(moe, true, false, false);
    warm.extend(kernels("qwen_rows/mtp", &ffn));
    warm.extend(kernels("qwen_rows/mtp", &["advance"]));
    let mut head = kernels(
        "qwen_rows/mtp",
        &[
            "final_norm",
            "head_rotate",
            "head_project",
            "partials",
            "merge",
            "selection",
        ],
    );
    head.push(Operation::Copy {
        source: "MtpLastHidden".into(),
        destination: "MtpCondition".into(),
        bytes: hidden * 2,
    });
    let mut draft = vec![Operation::Copy {
        source: "MtpToken".into(),
        destination: "MtpInput".into(),
        bytes: 4,
    }];
    draft.extend(warm.iter().skip(1).cloned());
    draft.extend(head.clone());
    m.programs.insert("mtp_warm".into(), warm);
    m.programs.insert("mtp_head".into(), head);
    m.programs.insert("mtp_draft".into(), draft);
    m.programs
        .insert("capture".into(), kernels("qwen_rows/capture", &["hidden"]));
    let mut verify = m.programs["qwen_rows/begin"].clone();
    let mut restore = vec![];
    let layers = &m
        .batch_layout
        .as_ref()
        .ok_or("Missing Qwen row layout")?
        .layers;
    for (layer, kind) in layers.iter().enumerate() {
        verify.extend(m.programs[&format!("qwen_rows/layer{layer}/pre")].clone());
        let mut private = if kind == "linear_attention" {
            let mut private = kernels(&format!("qwen_rows/layer{layer}/private"), &["gates"]);
            private.extend(kernels(
                &format!("qwen_rows/verify/layer{layer}"),
                &["history"],
            ));
            private.extend(kernels(
                &format!("qwen_rows/layer{layer}/private"),
                &["conv"],
            ));
            if super::deferred_gdn(m) {
                private.extend(kernels(
                    &format!("qwen_rows/layer{layer}/private"),
                    &["recurrent_materialize"],
                ));
            }
            private.extend(kernels(
                &format!("qwen_rows/verify/layer{layer}"),
                &["recurrent", "factors"],
            ));
            private.extend(kernels(
                &format!("qwen_rows/layer{layer}/private"),
                &["gated_norm"],
            ));
            restore.extend(kernels(
                &format!("qwen_rows/restore/layer{layer}"),
                &["state", "history"],
            ));
            private
        } else {
            m.programs[&format!("qwen_rows/layer{layer}/private")].clone()
        };
        m.programs.insert(
            format!("qwen_rows/verify/layer{layer}/private"),
            private.clone(),
        );
        verify.append(&mut private);
        verify.extend(m.programs[&format!("qwen_rows/layer{layer}/post")].clone());
    }
    verify.extend(m.programs["qwen_rows/end"].clone());
    verify.extend(kernels("qwen_rows/verify", &["norm"]));
    verify.extend(m.programs["qwen_rows/head"].clone());
    verify.extend(kernels("qwen_rows/verify", &["logits", "selections"]));
    m.programs.insert("verify".into(), verify);
    m.programs.insert("restore".into(), restore);
    m.programs.insert(
        "qwen_rows/verify/gather".into(),
        kernels("qwen_rows/verify", &["norm"]),
    );
    m.programs.insert(
        "qwen_rows/verify/select".into(),
        kernels("qwen_rows/verify", &["selections"]),
    );
    for (name, names) in [
        (
            "mtp_begin",
            &["gather", "prepare", "embedding", "overlay", "concat"][..],
        ),
        (
            "mtp_pre",
            &["fc_quant", "fc", "pre_norm", "pre_quant", "q", "k", "v"][..],
        ),
        (
            "mtp_private",
            &["kv_prepare", "attention", "attention_merge"][..],
        ),
        ("mtp_advance", &["advance"][..]),
        ("mtp_gather", &["final_norm"][..]),
        ("mtp_select", &["selection"][..]),
    ] {
        m.programs
            .insert(name.into(), kernels("qwen_rows/mtp", names));
    }
    let mut post = kernels(
        "qwen_rows/mtp",
        &["out_quant", "out", "post_norm", "post_quant"],
    );
    post.extend(kernels("qwen_rows/mtp", &ffn));
    m.programs.insert("mtp_post".into(), post);
    let batch_programs = m
        .programs
        .keys()
        .filter(|name| name.starts_with("mtp_") || name.starts_with("qwen_rows/verify/"))
        .cloned()
        .collect();
    if let Some(table) = m.buffers.iter().find(|b| b.name == "QwenCommitPointers") {
        if table.dtype != Dtype::U64
            || table.shape != [LANES * m.batch_layout.as_ref().unwrap().layers.len() * 8]
        {
            return Err("Invalid Qwen commit address table".into());
        }
        let mut operations = vec![];
        for (layer, kind) in m.batch_layout.as_ref().unwrap().layers.iter().enumerate() {
            if kind == "linear_attention" {
                operations.extend(kernels(
                    &format!("qwen_rows/commit/layer{layer}"),
                    &["state", "history"],
                ));
            }
        }
        m.programs.insert("qwen_rows/commit".into(), operations);
    }
    let spec = m.mtp.as_mut().unwrap();
    spec.batch_verification_tokens = (2..=8).collect();
    spec.mixed_verification_tokens = spec.batch_verification_tokens.clone();
    spec.batch_draft_profiles = vec![LANES];
    spec.batch_commit = true;
    spec.commit_state_table = m
        .buffers
        .iter()
        .find(|b| b.name == "QwenCommitPointers")
        .map(|b| b.name.clone());
    spec.batch_warm = true;
    spec.batch_programs = batch_programs;
    if spec.commit_state_table.is_some() {
        spec.batch_programs.push("qwen_rows/commit".into());
    }
    let kv = m.kv_cache.as_mut().ok_or("Missing MTP KV storage")?;
    for (name, position, tokens, length) in [
        ("mtp_warm", "MtpStep", 512, Some("MtpSegmentLength")),
        ("mtp_draft", "MtpStep", 1, None),
        ("verify", "Step", 8, Some("SegmentLength")),
    ] {
        kv.growth.insert(
            name.into(),
            KvGrowth {
                position: position.into(),
                tokens,
                length: length.map(str::to_owned),
                buffers: kv
                    .buffers
                    .keys()
                    .filter(|name| name.starts_with("Mtp") == position.starts_with("Mtp"))
                    .cloned()
                    .collect(),
            },
        );
    }
    Ok(())
}

pub(super) fn commit(
    m: &Manifest,
    segments: &[BatchSegment],
    include: bool,
) -> Result<abi::BatchPlan> {
    let spec = m.mtp.as_ref().ok_or("Missing Qwen MTP")?;
    let mut owners = BTreeSet::new();
    if !(2..=LANES).contains(&segments.len())
        || segments
            .iter()
            .any(|s| !owners.insert(s.slot) || !spec.batch_verification_tokens.contains(&s.tokens))
    {
        return Err("Invalid Qwen MTP commit packet".into());
    }
    let mut out = vec![];
    if spec.commit_state_table.is_some() {
        const COLUMNS: &[&str] = &[
            "VerifyK",
            "VerifyG",
            "VerifyUpdate",
            "AcceptedInputs",
            "SegmentLength",
            "State",
            "HistoryPrefixes",
            "History",
        ];
        let buffers: BTreeSet<_> = m.buffers.iter().map(|b| b.name.as_str()).collect();
        let roles: Vec<_> = (0..m.batch_layout.as_ref().unwrap().layers.len())
            .flat_map(|layer| {
                let buffers = &buffers;
                COLUMNS.iter().map(move |&role| {
                    let name = format!("L{layer}_{role}");
                    if buffers.contains(name.as_str()) {
                        Some(name)
                    } else {
                        buffers.contains(role).then(|| role.into())
                    }
                })
            })
            .collect();
        let bindings = crate::packet::private_bindings(segments, segments.len(), &roles);
        if include {
            append(
                m,
                &mut out,
                "qwen_rows/commit",
                segments.len(),
                None,
                &BTreeMap::new(),
            )?;
        }
        return Ok((out, bindings));
    }
    if include {
        for s in segments {
            append(
                m,
                &mut out,
                "restore",
                s.tokens,
                Some(s.slot),
                &BTreeMap::new(),
            )?;
        }
    }
    Ok((out, vec![]))
}

pub(super) fn draft(
    m: &Manifest,
    segments: &[BatchSegment],
    operation: abi::BatchOperation,
    include: bool,
) -> Result<abi::BatchPlan> {
    let body = operation != abi::BatchOperation::DraftHead;
    let head = operation != abi::BatchOperation::Warm;
    let mut owners = BTreeSet::new();
    let total = segments
        .iter()
        .try_fold(0usize, |n, s| n.checked_add(s.tokens))
        .ok_or("Draft row overflow")?;
    if segments.is_empty()
        || total > LANES
        || segments.iter().any(|s| {
            s.tokens == 0
                || !owners.insert(s.slot)
                || (operation != abi::BatchOperation::Warm && s.tokens != 1)
        })
    {
        return Err("Invalid Qwen draft packet".into());
    }
    let mut out = vec![];
    if !include {
        return Ok((out, vec![]));
    }
    let empty = BTreeMap::new();
    let hidden = m.batch_layout.as_ref().ok_or("Missing Qwen layout")?.hidden;
    let strides: Vec<(&str, usize)> = [
        "MtpEmbedding",
        "MtpConcat",
        "MtpHidden",
        "MtpNorm",
        "MtpR0",
        "MtpR1",
        "MtpFullX",
        "MtpFullQ",
        "MtpFullGate",
        "MtpMixer",
        "MtpMix",
        "MtpGateUp",
    ]
    .iter()
    .map(|&name| {
        let b = m
            .buffers
            .iter()
            .find(|b| b.name == name)
            .ok_or("Missing draft row buffer")?;
        Ok((
            name,
            b.shape[1..].iter().product::<usize>() * b.dtype.bytes(),
        ))
    })
    .collect::<Result<_>>()?;
    let mut offset = 0;
    let views: Vec<BTreeMap<String, BufferView>> = segments
        .iter()
        .map(|s| {
            let v = strides
                .iter()
                .map(|(name, stride)| {
                    (
                        (*name).into(),
                        BufferView {
                            buffer: (*name).into(),
                            offset: offset * stride,
                        },
                    )
                })
                .collect();
            offset += s.tokens;
            v
        })
        .collect();
    let copy = |out: &mut Vec<Invocation>,
                source: &str,
                destination: &str,
                bytes: usize,
                owner: usize,
                views: &BTreeMap<String, BufferView>| {
        out.push(Invocation {
            operation: Operation::Copy {
                source: source.into(),
                destination: destination.into(),
                bytes,
            },
            sequence: Some(owner),
            views: views.clone(),
            launch: None,
        });
    };
    if body {
        for (s, v) in segments.iter().zip(&views) {
            if operation == abi::BatchOperation::Draft {
                copy(&mut out, "MtpToken", "MtpInput", 4, s.slot, &empty);
            }
            // Warm gathers the complete target-hidden chain. A draft reuses
            // its previous normalized condition, exactly like the single path.
            let begin = &m.programs["mtp_begin"];
            let begin = if operation == abi::BatchOperation::Draft {
                &begin[1..]
            } else {
                &begin[..]
            };
            let rows = m.row_execution.as_ref().ok_or("Missing draft rows")?;
            let mut calls =
                rows.specialize("mtp_begin", begin, &m.dynamic_batch_kernels, s.tokens)?;
            for call in &mut calls {
                call.sequence = Some(s.slot);
                call.views = v.clone();
            }
            out.extend(calls);
        }
        append(m, &mut out, "mtp_pre", total, None, &empty)?;
        for (s, v) in segments.iter().zip(&views) {
            append(m, &mut out, "mtp_private", s.tokens, Some(s.slot), v)?;
        }
        append(m, &mut out, "mtp_post", total, None, &empty)?;
    }
    for (lane, (s, v)) in segments.iter().zip(&views).enumerate() {
        let hv = BTreeMap::from([(
            "HeadHidden".into(),
            BufferView {
                buffer: "HeadHidden".into(),
                offset: lane * hidden * 2,
            },
        )]);
        if body {
            append(m, &mut out, "mtp_advance", s.tokens, Some(s.slot), v)?;
            append(m, &mut out, "mtp_gather", 1, Some(s.slot), v)?;
            copy(
                &mut out,
                "MtpLastHidden",
                "MtpCondition",
                hidden * 2,
                s.slot,
                &empty,
            );
        }
        if head {
            copy(
                &mut out,
                "MtpCondition",
                "HeadHidden",
                hidden * 2,
                s.slot,
                &hv,
            );
        }
    }
    if head {
        append(m, &mut out, "qwen_rows/head", segments.len(), None, &empty)?;
        for (lane, s) in segments.iter().enumerate() {
            let v = BTreeMap::from([
                (
                    "HeadLogits".into(),
                    BufferView {
                        buffer: "HeadLogits".into(),
                        offset: lane * m.vocab * 4,
                    },
                ),
                (
                    "Pairs".into(),
                    BufferView {
                        buffer: "Pairs".into(),
                        offset: lane * 8,
                    },
                ),
            ]);
            copy(&mut out, "HeadLogits", "MtpLogits", m.vocab * 4, s.slot, &v);
            append(m, &mut out, "mtp_select", 1, Some(s.slot), &v)?;
        }
    }
    Ok((out, vec![]))
}
