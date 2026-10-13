//! Independent draft rows share projections; every QSA update owns a request.
use crate::{
    artifact::Result,
    execution::{BufferView, Invocation},
    model::{Manifest, Operation},
};
use orinfer_model_sdk::{abi::BatchOperation, architecture::BatchSegment};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn register(m: &mut Manifest) -> Result<()> {
    let Some(spec) = &m.mtp else { return Ok(()) };
    for &rows in &spec.batch_draft_profiles {
        if ![2, 4, 8].contains(&rows) {
            return Err("Unsupported Flash draft capacity".into());
        }
        for (stage, count) in [
            ("begin", 5),
            ("pre", 10),
            ("private", 19),
            ("post", 25),
            ("head", 8),
        ] {
            let prefix = format!("flash_draft_m{rows}");
            m.programs.insert(
                format!("{prefix}/{stage}"),
                super::plan::section(&prefix, stage, count, None),
            );
        }
        for tokens in 1..=rows {
            let prefix = format!("flash_draft_m{rows}");
            let section = format!("warm_m{tokens}/private");
            m.programs.insert(
                format!("{prefix}/{section}"),
                super::plan::section(&prefix, &section, 19, None),
            );
        }
    }
    Ok(())
}

pub(super) fn plan(
    m: &Manifest,
    segments: &[BatchSegment],
    operation: BatchOperation,
) -> Result<Vec<Invocation>> {
    let body = operation != BatchOperation::DraftHead;
    let head = operation != BatchOperation::Warm;
    let spec = m.mtp.as_ref().ok_or("Missing draft adapter")?;
    let used = segments.iter().try_fold(0usize, |sum, s| {
        sum.checked_add(s.tokens).ok_or("Draft row overflow")
    })?;
    let capacity = spec
        .batch_draft_profiles
        .iter()
        .copied()
        .find(|&n| n >= used)
        .ok_or("Unsupported draft batch capacity")?;
    let mut owners = BTreeSet::new();
    if segments.is_empty()
        || (operation == BatchOperation::Draft && segments.len() < 2)
        || segments.iter().any(|s| {
            s.tokens == 0
                || (operation != BatchOperation::Warm && s.tokens != 1)
                || !owners.insert(s.slot)
        })
    {
        return Err("Draft packet requires unique independent one-token rows".into());
    }
    let mut offset = 0;
    let offsets: Vec<_> = segments
        .iter()
        .map(|s| {
            let start = offset;
            offset += s.tokens;
            start
        })
        .collect();
    let prefix = format!("flash_draft_m{capacity}");
    let arena = format!("DraftBatchM{capacity}_");
    let mut out = vec![];
    let mut push = |operation, sequence, views: BTreeMap<String, BufferView>| {
        out.push(Invocation {
            operation,
            sequence,
            views,
            launch: None,
        });
    };
    let copy = |source: &str, destination: &str, bytes: usize| Operation::Copy {
        source: source.into(),
        destination: destination.into(),
        bytes,
    };
    let row = |name: &str, index: usize| -> Result<BufferView> {
        let buffer = m
            .buffers
            .iter()
            .find(|b| b.name == name)
            .ok_or("Missing draft row arena")?;
        let bytes = buffer.bytes()?;
        if !bytes.is_multiple_of(capacity) {
            return Err("Unaligned draft row arena".into());
        }
        Ok(BufferView {
            buffer: name.into(),
            offset: index * (bytes / capacity),
        })
    };
    let embedding = format!("{arena}DraftM{capacity}_Embedding");
    let condition = format!("{arena}DraftM{capacity}_Condition");
    let hidden = format!("{arena}DraftM{capacity}_HC");
    // Inactive lanes are initialized without touching any request state.
    for (name, stride) in if body {
        vec![(&embedding, 5120), (&condition, 20480)]
    } else {
        vec![(&hidden, 20480)]
    } {
        push(
            Operation::Zero {
                destination: name.clone(),
                bytes: capacity * stride,
            },
            None,
            BTreeMap::new(),
        );
    }
    for (lane, segment) in segments.iter().enumerate() {
        if body {
            if operation == BatchOperation::Warm {
                push(
                    Operation::Kernel {
                        name: format!("mtp_warm_m{}/begin/k0", segment.tokens),
                    },
                    Some(segment.slot),
                    BTreeMap::new(),
                );
            }
            for i in 1..=usize::from(m.vision.is_some()) + 1 {
                push(
                    Operation::Kernel {
                        name: format!("mtp_warm_m{}/begin/k{i}", segment.tokens),
                    },
                    Some(segment.slot),
                    BTreeMap::new(),
                );
            }
            for (source, destination, bytes) in [
                (
                    format!("DraftM{}_Embedding", segment.tokens),
                    &embedding,
                    segment.tokens * 5120,
                ),
                (
                    format!("DraftM{}_Condition", segment.tokens),
                    &condition,
                    segment.tokens * 20480,
                ),
            ] {
                push(
                    copy(&source, destination, bytes),
                    Some(segment.slot),
                    BTreeMap::from([(destination.clone(), row(destination, offsets[lane])?)]),
                );
            }
        } else {
            push(
                copy("DraftM1_Condition", &hidden, 20480),
                Some(segment.slot),
                BTreeMap::from([(hidden.clone(), row(&hidden, lane)?)]),
            );
        }
    }
    if body {
        for stage in ["begin", "pre"] {
            for op in &m.programs[&format!("{prefix}/{stage}")] {
                push(op.clone(), None, BTreeMap::new());
            }
        }
        // Dedicated draft arenas are row-major. Only interface buffers get a lane
        // view; causal scratch/control bindings still resolve within the owner slot.
        for (lane, segment) in segments.iter().enumerate() {
            let views: BTreeMap<String, BufferView> = m
                .buffers
                .iter()
                .filter(|b| {
                    b.name.starts_with(&arena)
                        && b.bytes().is_ok_and(|n| n.is_multiple_of(capacity))
                })
                .map(|b| Ok((b.name.clone(), row(&b.name, offsets[lane])?)))
                .collect::<Result<_>>()?;
            let section = if operation == BatchOperation::Warm {
                format!("warm_m{}/private", segment.tokens)
            } else {
                "private".into()
            };
            for op in &m.programs[&format!("{prefix}/{section}")] {
                push(op.clone(), Some(segment.slot), views.clone());
            }
            push(
                Operation::Kernel {
                    name: format!("mtp_warm_m{}/end/k0", segment.tokens),
                },
                Some(segment.slot),
                BTreeMap::new(),
            );
        }
        for op in &m.programs[&format!("{prefix}/post")] {
            push(op.clone(), None, BTreeMap::new());
        }
    }
    if head {
        for stage in ["head"] {
            for op in &m.programs[&format!("{prefix}/{stage}")] {
                push(op.clone(), None, BTreeMap::new());
            }
        }
    }
    let logits = format!("{arena}logits");
    let pairs = format!("{arena}pairs");
    for (lane, segment) in segments.iter().enumerate() {
        for (source, destination, bytes, extra) in [
            (&logits, spec.draft_logits.as_str(), m.vocab * 4, 0),
            (&pairs, spec.token.as_str(), 4, 0),
            (&pairs, spec.status.as_str(), 4, 4),
            (&hidden, "DraftM1_Condition", 20480, 0),
        ] {
            if !head && source != &hidden {
                continue;
            }
            if !body && source == &hidden {
                continue;
            }
            let endpoint = if source == &hidden {
                offsets[lane] + segment.tokens - 1
            } else {
                lane
            };
            let mut view = row(source, endpoint)?;
            view.offset += extra;
            push(
                copy(source, destination, bytes),
                Some(segment.slot),
                BTreeMap::from([(source.clone(), view)]),
            );
        }
    }
    Ok(out)
}
