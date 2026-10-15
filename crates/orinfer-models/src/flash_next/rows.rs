//! Shared token-row computation; owners and causal segments remain caller-owned.
use crate::{
    artifact::Result,
    execution::{BufferView, Invocation},
    model::{Manifest, Operation},
};
use std::collections::BTreeMap;

pub(super) fn verification(m: &mut Manifest) -> Result<()> {
    let Some(spec) = m.mtp.clone() else {
        return Ok(());
    };
    for plan in &spec.verification_plans {
        let n = plan.tokens;
        let width = super::plan::arena_width(n);
        let mut ops = Vec::new();
        if m.vision.is_some() {
            ops.push(Operation::Kernel {
                name: "mtp_verify/begin/k0".into(),
            });
        }
        for (source, destination, bytes) in [
            (
                format!("M{width}_Embedding"),
                "FlashDecode_M1_Embedding".into(),
                n * 5120,
            ),
            (
                format!("M{width}_Ple"),
                "FlashDecode_M1_Ple".into(),
                n * 5120,
            ),
        ] {
            ops.push(Operation::Copy {
                source,
                destination,
                bytes,
            });
        }
        ops.extend(m.programs["flash_decode/begin"].clone());
        for layer in 0..48 {
            if layer == 1 {
                ops.extend(m.programs["flash_decode/layer1/ple"].clone());
                ops.extend((0..3).map(|i| Operation::Kernel {
                    name: { format!("causal_rows/verify/layer1/ple/k{i}") },
                }));
            }
            ops.extend(m.programs[&format!("flash_decode/layer{layer}/pre")].clone());
            for i in 0..if layer % 4 == 3 { 20 } else { 4 } {
                ops.push(Operation::Kernel {
                    name: { format!("causal_rows/verify/layer{layer}/private/k{i}") },
                });
                if layer % 4 != 3 && i == 1 {
                    ops.push(Operation::Copy {
                        source: "M8_HistoryOut".into(),
                        destination: format!("State_{layer}_conv"),
                        bytes: 61440,
                    });
                }
            }
            ops.extend(m.programs[&format!("flash_decode/layer{layer}/post")].clone());
        }
        ops.push(Operation::Kernel {
            name: "causal_rows/verify/advance".into(),
        });
        ops.extend(m.programs["flash_decode/head"].clone());
        ops.extend([
            Operation::Copy {
                source: "FlashDecode_Logits".into(),
                destination: spec.verification_logits.clone(),
                bytes: n * m.vocab * 4,
            },
            Operation::Copy {
                source: "FlashDecode_Selected".into(),
                destination: "VerificationPairs".into(),
                bytes: n * 8,
            },
            Operation::Kernel {
                name: "causal_rows/verify/selections".into(),
            },
        ]);
        m.programs.insert(plan.program.clone(), ops);
        m.programs.insert(
            plan.capture_program.clone(),
            vec![Operation::Kernel {
                name: "causal_rows/verify/capture".into(),
            }],
        );
        let rows = m
            .row_execution
            .as_mut()
            .ok_or("Missing common verification rows")?;
        rows.programs.insert(plan.capture_program.clone());
        rows.program_capacities
            .insert(plan.capture_program.clone(), n);
        rows.programs.insert(plan.program.clone());
        rows.program_capacities.insert(plan.program.clone(), n);
    }
    Ok(())
}

pub(super) fn append(
    m: &Manifest,
    out: &mut Vec<Invocation>,
    program: &str,
    count: usize,
    sequence: Option<usize>,
    views: &BTreeMap<String, BufferView>,
) -> Result<()> {
    let rows = m
        .row_execution
        .as_ref()
        .ok_or("Missing shared row execution")?;
    let operations = m
        .programs
        .get(program)
        .ok_or("Missing shared row program")?;
    let mut calls = rows.specialize(program, operations, &m.dynamic_batch_kernels, count)?;
    for call in &mut calls {
        call.sequence = sequence;
        call.views = views.clone();
    }
    out.extend(calls);
    Ok(())
}
