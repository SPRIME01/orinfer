//! Concatenated dense/MoE rows with independent causal segments and endpoint heads.
use crate::{
    artifact::{Argument, Dtype, Kernel, Result},
    execution::{BufferView, Invocation},
    model::{Manifest, Operation},
};
use orinfer_model_sdk::architecture::BatchSegment;
use std::collections::{BTreeMap, BTreeSet};

const CAPACITIES: [usize; 5] = [32, 64, 128, 256, 1024];
const SEGMENTS: [usize; 4] = [1, 16, 128, 512];
const RESIDUAL: &str = "FlashMixedResidual";
const HEAD_RESIDUAL: &str = "FlashMixedHeadResidual";

pub(super) fn operands(m: &mut Manifest, kernels: &[Kernel]) -> Result<()> {
    let contracts: BTreeMap<_, _> = kernels.iter().map(|k| (k.name.as_str(), k)).collect();
    let buffers: BTreeMap<_, _> = m.buffers.iter().map(|b| (b.name.as_str(), b)).collect();
    let mut casts = BTreeMap::new();
    for kernel in kernels {
        let Some(parent) = kernel.name.strip_suffix("/bf16_input") else {
            continue;
        };
        let projection = contracts.get(parent).ok_or("Missing HC operand consumer")?;
        let names: Vec<_> = kernel
            .args
            .iter()
            .filter_map(|a| {
                if let Argument::Buffer { name } = a {
                    Some(name.as_str())
                } else {
                    None
                }
            })
            .collect();
        if !parent.starts_with("flash_mixed_m")
            || kernel.symbol != "hc_input_cast_kernel"
            || names.len() != 2
        {
            return Err("Invalid joined HC operand cast".into());
        }
        let input = names
            .iter()
            .find_map(|n| buffers.get(n).filter(|b| b.dtype == Dtype::F16))
            .ok_or("Missing FP16 HC operand")?;
        let output = names
            .iter()
            .find_map(|n| buffers.get(n).filter(|b| b.dtype == Dtype::Bf16))
            .ok_or("Missing BF16 HC operand")?;
        if input.data.is_some()
            || output.data.is_some()
            || input.bytes()? < output.bytes()?
            || !projection
                .args
                .iter()
                .any(|a| matches!(a,Argument::Buffer {name} if name==&output.name))
        {
            return Err("HC operand cast/consumer storage mismatch".into());
        }
        casts.insert(parent.to_owned(), kernel.name.clone());
    }
    for operations in m.programs.values_mut() {
        let mut result = Vec::new();
        for op in std::mem::take(operations) {
            if let Operation::Kernel { name } = &op
                && let Some(cast) = casts.remove(name)
            {
                result.push(Operation::Kernel { name: cast });
            }
            result.push(op);
        }
        *operations = result;
    }
    if !casts.is_empty() {
        return Err("Unregistered joined HC operand conversion".into());
    }
    Ok(())
}

pub(super) fn register(m: &mut Manifest) -> Result<()> {
    let layout = m.batch_layout.as_ref().ok_or("Missing Flash layout")?;
    if layout.profiles.is_empty() {
        if !m.prefill_batch_profiles.is_empty() || !layout.small_mixed_shapes.is_empty() {
            return Err("Incomplete Flash mixed batch layout".into());
        }
        return Ok(());
    }
    if layout.profiles.keys().copied().collect::<Vec<_>>() != [16, 128, 512]
        || layout.profiles.values().any(|v| v != "flash_recurrent")
        || layout.small_mixed_shapes != [16, 128, 512]
        || m.prefill_batch_profiles
            .iter()
            .map(|p| p.tokens)
            .collect::<Vec<_>>()
            != [256, 1024]
        || m.prefill_batch_profiles
            .iter()
            .any(|p| !matches!(p.kind.as_str(), "flash_mixed" | "flash_mixed_grouped"))
        || m.prefill_batch_profiles
            .windows(2)
            .any(|p| p[0].kind != p[1].kind)
        || m.buffers
            .iter()
            .find(|b| b.name == RESIDUAL)
            .is_none_or(|b| b.data.is_some() || b.bytes().ok() != Some(1024 * 20480))
    {
        return Err("Invalid Flash mixed causal/row layout".into());
    }
    for rows in CAPACITIES {
        let prefix = format!("flash_mixed_m{rows}");
        m.programs.insert(
            format!("{prefix}/begin"),
            super::plan::section(&prefix, "begin", 1, None),
        );
        for layer in 0..48 {
            let attention = usize::from(layer % 4 == 3);
            let pre = if rows > 128 { 11 } else { 12 };
            let post = if rows > 128 { 30 } else { 29 };
            for (stage, count) in [
                ("pre", pre - attention + usize::from(layer == 1)),
                ("post", post - attention),
            ] {
                let section = format!("layer{layer}/{stage}");
                m.programs.insert(
                    format!("{prefix}/{section}"),
                    super::plan::section(&prefix, &section, count, None),
                );
            }
        }
        m.programs.insert(
            format!("{prefix}/layer1/ple"),
            super::plan::section(&prefix, "layer1/ple", 7, None),
        );
    }
    for &rows in &m.batch_profiles {
        let prefix = format!("flash_mixed_head_m{rows}");
        m.programs.insert(
            format!("{prefix}/head"),
            super::plan::section(&prefix, "head", if rows <= 8 { 8 } else { 9 }, None),
        );
    }
    for tokens in SEGMENTS {
        let prefix = format!("flash_segment_m{tokens}");
        for layer in 0..48 {
            let section = format!("layer{layer}/private");
            let count = if layer % 4 == 3 { 19 } else { 2 };
            m.programs.insert(
                format!("{prefix}/{section}"),
                super::plan::section(&prefix, &section, count, None),
            );
        }
        m.programs.insert(
            format!("{prefix}/layer1/ple"),
            super::plan::section(&prefix, "layer1/ple", 2, None),
        );
        if m.mtp.is_some() {
            let name = format!("flash_mixed_capture_m{tokens}");
            m.programs
                .insert(name.clone(), super::plan::section(&name, "body", 1, None));
        }
    }
    if let Some(spec) = &m.mtp {
        for &tokens in &spec.batch_verification_tokens {
            if !(2..=8).contains(&tokens) {
                return Err("Unsupported Flash verification segment".into());
            }
            let prefix = format!("flash_verify_segment_m{tokens}");
            for layer in 0..48 {
                let section = format!("layer{layer}/private");
                m.programs.insert(
                    format!("{prefix}/{section}"),
                    super::plan::section(
                        &prefix,
                        &section,
                        if layer % 4 == 3 { 20 } else { 4 },
                        None,
                    ),
                );
            }
            m.programs.insert(
                format!("{prefix}/layer1/ple"),
                super::plan::section(&prefix, "layer1/ple", 3, None),
            );
            let capture = format!("flash_verify_capture_m{tokens}");
            m.programs.insert(
                capture.clone(),
                super::plan::section(&capture, "body", 1, None),
            );
        }
    }
    Ok(())
}

pub(super) fn validate_segments(m: &Manifest, segments: &[BatchSegment]) -> Result<usize> {
    if m.batch_layout
        .as_ref()
        .is_none_or(|l| l.profiles.is_empty())
        || segments.is_empty()
        || segments.len() > 128
    {
        return Err("Flash package lacks mixed causal segments".into());
    }
    let mut slots = BTreeSet::new();
    let total = segments.iter().try_fold(0usize, |n, s| -> Result<usize> {
        if !SEGMENTS.contains(&s.tokens) || !slots.insert(s.slot) {
            return Err("Invalid or repeated Flash causal segment".into());
        }
        n.checked_add(s.tokens).ok_or("Mixed row overflow".into())
    })?;
    CAPACITIES
        .into_iter()
        .find(|&n| n >= total)
        .ok_or("Flash mixed row capacity exceeded".into())
}

fn invoke(
    operation: Operation,
    sequence: Option<usize>,
    views: &BTreeMap<String, BufferView>,
) -> Invocation {
    Invocation {
        operation,
        sequence,
        views: views.clone(),
        launch: None,
    }
}
fn append(
    out: &mut Vec<Invocation>,
    m: &Manifest,
    program: &str,
    slot: Option<usize>,
    views: &BTreeMap<String, BufferView>,
) -> Result<()> {
    out.extend(
        m.programs
            .get(program)
            .ok_or_else(|| format!("Missing mixed stage {program}"))?
            .iter()
            .cloned()
            .map(|op| invoke(op, slot, views)),
    );
    Ok(())
}

pub(super) fn plan(m: &Manifest, segments: &[BatchSegment]) -> Result<Vec<Invocation>> {
    plan_kind(m, segments, Kind::Ordinary)
}

pub(super) fn verification_plan(
    m: &Manifest,
    segments: &[BatchSegment],
) -> Result<Vec<Invocation>> {
    plan_kind(m, segments, Kind::Verification)
}

pub(super) fn mixed_verification_plan(
    m: &Manifest,
    segments: &[BatchSegment],
) -> Result<Vec<Invocation>> {
    if !m
        .mtp
        .as_ref()
        .is_some_and(|s| !s.mixed_verification_tokens.is_empty())
    {
        return Err("Package lacks mixed prefill/verification".into());
    }
    plan_kind(m, segments, Kind::MixedVerification)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Ordinary,
    Verification,
    MixedVerification,
}

fn plan_kind(m: &Manifest, segments: &[BatchSegment], kind: Kind) -> Result<Vec<Invocation>> {
    let verified = |s: &BatchSegment| {
        kind != Kind::Ordinary
            && m.mtp.as_ref().is_some_and(|spec| {
                if kind == Kind::MixedVerification {
                    spec.mixed_verification_tokens.contains(&s.tokens)
                } else {
                    spec.batch_verification_tokens.contains(&s.tokens)
                }
            })
    };
    let head_used: usize = segments
        .iter()
        .map(|s| if verified(s) { s.tokens } else { 1 })
        .sum();
    let rows = if kind == Kind::Ordinary {
        validate_segments(m, segments)?
    } else {
        let mut slots = BTreeSet::new();
        let mut has_prefill = false;
        let mut has_verify = false;
        let total = segments.iter().try_fold(0usize, |total, s| {
            if !slots.insert(s.slot) {
                return Err("Repeated mixed sequence".to_string());
            }
            if verified(s) {
                has_verify = true;
            } else if kind == Kind::MixedVerification
                && SEGMENTS.contains(&s.tokens)
                && s.tokens > 1
            {
                has_prefill = true;
            } else {
                return Err("Invalid mixed verification segment".into());
            }
            total
                .checked_add(s.tokens)
                .ok_or("Verification row overflow".to_string())
        })?;
        if !(2..=128).contains(&segments.len())
            || head_used > 128
            || !has_verify
            || (kind == Kind::MixedVerification && !has_prefill)
        {
            return Err("Invalid mixed verification endpoints".into());
        }
        CAPACITIES
            .into_iter()
            .find(|&n| n >= total)
            .ok_or("Missing verification row body")?
    };
    let layout = m
        .batch_layout
        .as_ref()
        .filter(|l| !l.profiles.is_empty())
        .ok_or("Missing mixed row layout")?;
    // Put single-token rows first so the existing request-table mixers can
    // consume a contiguous subgroup without gathering projections or state.
    let ordered: Vec<_> = segments
        .iter()
        .filter(|s| s.tokens == 1)
        .chain(segments.iter().filter(|s| s.tokens != 1))
        .cloned()
        .collect();
    let segments = ordered.as_slice();
    let decoders = segments.iter().take_while(|s| s.tokens == 1).count();
    let parallel = kind == Kind::Ordinary && decoders >= 2 && !layout.state_columns.is_empty();
    let unified_decode = super::unified_decode::present(m);
    let private_capacity = if unified_decode {
        128
    } else {
        decoders.next_power_of_two().max(2)
    };
    let private_views: BTreeMap<_, _> = layout
        .row_strides
        .keys()
        .filter_map(|role| {
            let buffer = format!("FlashMixed_{role}");
            m.buffers.iter().any(|b| b.name == buffer).then(|| {
                (
                    if unified_decode {
                        format!("FlashDecode_{role}")
                    } else {
                        format!("BatchM{private_capacity}_{role}")
                    },
                    BufferView { buffer, offset: 0 },
                )
            })
        })
        .collect();
    let prefix = format!("flash_mixed_m{rows}");
    let dynamic = m
        .dynamic_batch_kernels
        .contains_key(&format!("{prefix}/begin/k0"));
    let empty = BTreeMap::new();
    let mut out = vec![];
    let mut starts = vec![];
    let mut start = 0;
    for s in segments {
        let width = super::plan::arena_width(s.tokens);
        let program = if verified(s) {
            format!("verify_m{}", s.tokens)
        } else if s.tokens == 1 {
            "decode".into()
        } else if m.row_execution.is_some() {
            format!(
                "prefill_m{}",
                m.prefill_plan(s.tokens)
                    .ok_or("Missing mixed prefill template")?
                    .chunk_tokens
            )
        } else {
            format!("prefill_m{}", s.tokens)
        };
        if m.vision.is_some() {
            let mut input_views = empty.clone();
            if !verified(s) && s.tokens > 1 && m.row_execution.is_some() {
                input_views.insert(
                    "M4096_Embedding".into(),
                    BufferView {
                        buffer: format!("M{width}_Embedding"),
                        offset: 0,
                    },
                );
            }
            let name = format!("{program}/begin/k0");
            let mut invocation = invoke(
                Operation::Kernel { name: name.clone() },
                Some(s.slot),
                &input_views,
            );
            if m.row_execution.is_some() && !verified(s) && s.tokens > 1 {
                invocation.launch = Some(
                    m.dynamic_batch_kernels
                        .get(&name)
                        .ok_or("Missing mixed overlay template contract")?
                        .launch(s.tokens)?,
                );
            }
            out.push(invocation);
        }
        let mut views: BTreeMap<_, _> = layout
            .row_strides
            .iter()
            .filter(|(n, _)| !matches!(n.as_str(), "Logits" | "Selected"))
            .map(|(name, stride)| {
                let name = format!("FlashMixed_{name}");
                (
                    name.clone(),
                    BufferView {
                        buffer: name,
                        offset: start * stride,
                    },
                )
            })
            .collect();
        views.insert(
            RESIDUAL.into(),
            BufferView {
                buffer: RESIDUAL.into(),
                offset: start * 20480,
            },
        );
        for (source, destination) in [
            (format!("M{width}_Embedding"), "FlashMixed_M1_Embedding"),
            (format!("M{width}_Ple"), "FlashMixed_M1_Ple"),
        ] {
            out.push(invoke(
                Operation::Copy {
                    source,
                    destination: destination.into(),
                    bytes: s.tokens * 5120,
                },
                Some(s.slot),
                &views,
            ));
        }
        starts.push((start, program, views));
        start += s.tokens;
    }
    if rows > start {
        for name in ["FlashMixed_M1_Embedding", "FlashMixed_M1_Ple"] {
            let pad = BTreeMap::from([(
                name.into(),
                BufferView {
                    buffer: name.into(),
                    offset: start * 5120,
                },
            )]);
            out.push(invoke(
                Operation::Zero {
                    destination: name.into(),
                    bytes: (rows - start) * 5120,
                },
                None,
                &pad,
            ));
        }
    }
    append(&mut out, m, &format!("{prefix}/begin"), None, &empty)?;
    for layer in 0..48 {
        if layer == 1 {
            append(&mut out, m, &format!("{prefix}/layer1/ple"), None, &empty)?;
            for (s, (_, _, views)) in segments.iter().zip(&starts) {
                append(
                    &mut out,
                    m,
                    &format!(
                        "{}_m{}/layer1/ple",
                        if verified(s) {
                            "flash_verify_segment"
                        } else {
                            "flash_segment"
                        },
                        s.tokens
                    ),
                    Some(s.slot),
                    views,
                )?;
            }
        }
        append(
            &mut out,
            m,
            &format!("{prefix}/layer{layer}/pre"),
            None,
            &empty,
        )?;
        // Padded private rows must never retain a previous batch's mixer output.
        if rows > start && !dynamic {
            for (name, stride) in &layout.row_strides {
                if matches!(
                    name.as_str(),
                    "Logits" | "Selected" | "M1_Embedding" | "M1_Ple"
                ) {
                    continue;
                }
                let name = format!("FlashMixed_{name}");
                if !m.buffers.iter().any(|b| b.name == name) {
                    continue;
                }
                let pad = BTreeMap::from([(
                    name.clone(),
                    BufferView {
                        buffer: name.clone(),
                        offset: start * stride,
                    },
                )]);
                out.push(invoke(
                    Operation::Zero {
                        destination: name,
                        bytes: (rows - start) * stride,
                    },
                    None,
                    &pad,
                ));
            }
        }
        if parallel {
            let program = if unified_decode {
                format!("flash_decode_state/layer{layer}")
            } else {
                format!("flash_private_m{private_capacity}/layer{layer}")
            };
            append(&mut out, m, &program, None, &private_views)?;
            for invocation in out.iter_mut().rev().take(m.programs[&program].len()) {
                if (unified_decode || private_capacity != decoders)
                    && let Operation::Kernel { name } = &invocation.operation
                {
                    invocation.launch = Some(
                        m.dynamic_batch_kernels
                            .get(name)
                            .ok_or("Missing mixed decoder subgroup launch")?
                            .launch(decoders)?,
                    );
                }
            }
        }
        for (s, (_, _, views)) in segments
            .iter()
            .zip(&starts)
            .filter(|(s, _)| !parallel || s.tokens != 1)
        {
            let private = &m.programs[&format!(
                "{}_m{}/layer{layer}/private",
                if verified(s) {
                    "flash_verify_segment"
                } else {
                    "flash_segment"
                },
                s.tokens
            )];
            for (i, op) in private.iter().enumerate() {
                out.push(invoke(op.clone(), Some(s.slot), views));
                if layer % 4 != 3 && i == usize::from(verified(s)) {
                    out.push(invoke(
                        Operation::Copy {
                            source: format!("M{}_HistoryOut", super::plan::arena_width(s.tokens)),
                            destination: format!("State_{layer}_conv"),
                            bytes: 61440,
                        },
                        Some(s.slot),
                        &empty,
                    ));
                }
            }
        }
        append(
            &mut out,
            m,
            &format!("{prefix}/layer{layer}/post"),
            None,
            &empty,
        )?;
    }
    let head_rows = *m
        .batch_profiles
        .iter()
        .find(|&&n| {
            n >= head_used
                && m.programs
                    .contains_key(&format!("flash_mixed_head_m{n}/head"))
        })
        .ok_or("Missing Flash endpoint head")?;
    if head_rows > head_used {
        let pad = BTreeMap::from([(
            HEAD_RESIDUAL.into(),
            BufferView {
                buffer: HEAD_RESIDUAL.into(),
                offset: head_used * 20480,
            },
        )]);
        out.push(invoke(
            Operation::Zero {
                destination: HEAD_RESIDUAL.into(),
                bytes: (head_rows - head_used) * 20480,
            },
            None,
            &pad,
        ));
    }
    let mut head_lane = 0;
    for (s, (first, program, views)) in segments.iter().zip(&starts) {
        let name = format!("{program}/end/k0");
        let mut invocation = invoke(
            Operation::Kernel { name: name.clone() },
            Some(s.slot),
            &empty,
        );
        if m.row_execution.is_some() && !verified(s) && s.tokens > 1 {
            invocation.launch = Some(
                m.dynamic_batch_kernels
                    .get(&name)
                    .ok_or("Missing mixed position template contract")?
                    .launch(s.tokens)?,
            );
        }
        out.push(invocation);
        if m.mtp.is_some() {
            append(
                &mut out,
                m,
                &format!(
                    "flash_{}_capture_m{}",
                    if verified(s) { "verify" } else { "mixed" },
                    s.tokens
                ),
                Some(s.slot),
                views,
            )?;
        }
        let copy = BTreeMap::from([
            (
                RESIDUAL.into(),
                BufferView {
                    buffer: RESIDUAL.into(),
                    offset: (first + if verified(s) { 0 } else { s.tokens - 1 }) * 20480,
                },
            ),
            (
                HEAD_RESIDUAL.into(),
                BufferView {
                    buffer: HEAD_RESIDUAL.into(),
                    offset: head_lane * 20480,
                },
            ),
        ]);
        out.push(invoke(
            Operation::Copy {
                source: RESIDUAL.into(),
                destination: HEAD_RESIDUAL.into(),
                bytes: if verified(s) { s.tokens * 20480 } else { 20480 },
            },
            Some(s.slot),
            &copy,
        ));
        head_lane += if verified(s) { s.tokens } else { 1 };
    }
    append(
        &mut out,
        m,
        &format!("flash_mixed_head_m{head_rows}/head"),
        None,
        &empty,
    )?;
    let mut head_lane = 0;
    for s in segments {
        if verified(s) {
            let spec = m.mtp.as_ref().unwrap();
            let views = BTreeMap::from([
                (
                    format!("BatchM{head_rows}_Logits"),
                    BufferView {
                        buffer: format!("BatchM{head_rows}_Logits"),
                        offset: head_lane * m.vocab * 4,
                    },
                ),
                (
                    format!("BatchM{head_rows}_Selected"),
                    BufferView {
                        buffer: format!("BatchM{head_rows}_Selected"),
                        offset: head_lane * 8,
                    },
                ),
            ]);
            for (source, destination, bytes) in [
                (
                    format!("BatchM{head_rows}_Logits"),
                    spec.verification_logits.clone(),
                    s.tokens * m.vocab * 4,
                ),
                (
                    format!("BatchM{head_rows}_Selected"),
                    "VerificationPairs".into(),
                    s.tokens * 8,
                ),
            ] {
                out.push(invoke(
                    Operation::Copy {
                        source,
                        destination,
                        bytes,
                    },
                    Some(s.slot),
                    &views,
                ));
            }
            out.push(invoke(
                Operation::Kernel {
                    name: format!("verify_m{}/head/k8", s.tokens),
                },
                Some(s.slot),
                &empty,
            ));
            head_lane += s.tokens;
            continue;
        }
        let mut views = BTreeMap::from([(
            format!("BatchM{head_rows}_Logits"),
            BufferView {
                buffer: format!("BatchM{head_rows}_Logits"),
                offset: head_lane * m.vocab * 4,
            },
        )]);
        out.push(invoke(
            Operation::Copy {
                source: format!("BatchM{head_rows}_Logits"),
                destination: m.logits.clone(),
                bytes: m.vocab * 4,
            },
            Some(s.slot),
            &views,
        ));
        for (offset, destination) in [(0, &m.token), (4, &m.status)] {
            let name = format!("BatchM{head_rows}_Selected");
            views.insert(
                name.clone(),
                BufferView {
                    buffer: name.clone(),
                    offset: head_lane * 8 + offset,
                },
            );
            out.push(invoke(
                Operation::Copy {
                    source: name,
                    destination: destination.clone(),
                    bytes: 4,
                },
                Some(s.slot),
                &views,
            ));
        }
        head_lane += 1;
    }
    if dynamic {
        for invocation in &mut out {
            if let Operation::Kernel { name } = &invocation.operation
                && name.starts_with(&format!("{prefix}/"))
            {
                invocation.launch = Some(
                    m.dynamic_batch_kernels
                        .get(name)
                        .ok_or("Missing joined row launch contract")?
                        .launch(start)?,
                );
            }
        }
    }
    Ok(out)
}
