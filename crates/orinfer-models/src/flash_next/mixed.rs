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
            .map(|p| (p.tokens, p.kind.as_str()))
            .collect::<Vec<_>>()
            != [(256, "flash_mixed"), (1024, "flash_mixed")]
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
    let rows = validate_segments(m, segments)?;
    let layout = m.batch_layout.as_ref().unwrap();
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
        let program = if s.tokens == 1 {
            "decode".into()
        } else {
            format!("prefill_m{}", s.tokens)
        };
        if m.vision.is_some() {
            out.push(invoke(
                Operation::Kernel {
                    name: format!("{program}/begin/k0"),
                },
                Some(s.slot),
                &empty,
            ));
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
                    &format!("flash_segment_m{}/layer1/ple", s.tokens),
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
        for (s, (_, _, views)) in segments.iter().zip(&starts) {
            let private = &m.programs[&format!("flash_segment_m{}/layer{layer}/private", s.tokens)];
            for (i, op) in private.iter().enumerate() {
                out.push(invoke(op.clone(), Some(s.slot), views));
                if layer % 4 != 3 && i == 0 {
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
        .find(|&&n| n >= segments.len())
        .ok_or("Missing Flash endpoint head")?;
    if head_rows > segments.len() {
        let pad = BTreeMap::from([(
            HEAD_RESIDUAL.into(),
            BufferView {
                buffer: HEAD_RESIDUAL.into(),
                offset: segments.len() * 20480,
            },
        )]);
        out.push(invoke(
            Operation::Zero {
                destination: HEAD_RESIDUAL.into(),
                bytes: (head_rows - segments.len()) * 20480,
            },
            None,
            &pad,
        ));
    }
    for (lane, (s, (first, program, views))) in segments.iter().zip(&starts).enumerate() {
        out.push(invoke(
            Operation::Kernel {
                name: format!("{program}/end/k0"),
            },
            Some(s.slot),
            &empty,
        ));
        if m.mtp.is_some() {
            append(
                &mut out,
                m,
                &format!("flash_mixed_capture_m{}", s.tokens),
                Some(s.slot),
                views,
            )?;
        }
        let copy = BTreeMap::from([
            (
                RESIDUAL.into(),
                BufferView {
                    buffer: RESIDUAL.into(),
                    offset: (first + s.tokens - 1) * 20480,
                },
            ),
            (
                HEAD_RESIDUAL.into(),
                BufferView {
                    buffer: HEAD_RESIDUAL.into(),
                    offset: lane * 20480,
                },
            ),
        ]);
        out.push(invoke(
            Operation::Copy {
                source: RESIDUAL.into(),
                destination: HEAD_RESIDUAL.into(),
                bytes: 20480,
            },
            Some(s.slot),
            &copy,
        ));
    }
    append(
        &mut out,
        m,
        &format!("flash_mixed_head_m{head_rows}/head"),
        None,
        &empty,
    )?;
    for (lane, s) in segments.iter().enumerate() {
        let mut views = BTreeMap::from([(
            format!("BatchM{head_rows}_Logits"),
            BufferView {
                buffer: format!("BatchM{head_rows}_Logits"),
                offset: lane * m.vocab * 4,
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
                    offset: lane * 8 + offset,
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
