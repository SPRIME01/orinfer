//! The target's row arithmetic is shared; speculation changes causal commit only.
use super::*;

pub(super) fn register(m: &mut Manifest) -> Result<()> {
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
    let warm = kernels(
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
            "gate",
            "up",
            "down_quant",
            "down",
            "advance",
        ],
    );
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
        bytes: 5120 * 2,
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
        if kind == "linear_attention" {
            verify.extend(kernels(
                &format!("qwen_rows/layer{layer}/private"),
                &["gates"],
            ));
            verify.extend(kernels(
                &format!("qwen_rows/verify/layer{layer}"),
                &["history"],
            ));
            verify.extend(kernels(
                &format!("qwen_rows/layer{layer}/private"),
                &["conv"],
            ));
            verify.extend(kernels(
                &format!("qwen_rows/verify/layer{layer}"),
                &["recurrent", "factors"],
            ));
            verify.extend(kernels(
                &format!("qwen_rows/layer{layer}/private"),
                &["gated_norm"],
            ));
            restore.extend(kernels(
                &format!("qwen_rows/restore/layer{layer}"),
                &["state", "history"],
            ));
        } else {
            verify.extend(m.programs[&format!("qwen_rows/layer{layer}/private")].clone());
        }
        verify.extend(m.programs[&format!("qwen_rows/layer{layer}/post")].clone());
    }
    verify.extend(m.programs["qwen_rows/end"].clone());
    verify.extend(kernels("qwen_rows/verify", &["norm"]));
    verify.extend(m.programs["qwen_rows/head"].clone());
    verify.extend(kernels("qwen_rows/verify", &["logits", "selections"]));
    m.programs.insert("verify".into(), verify);
    m.programs.insert("restore".into(), restore);
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
