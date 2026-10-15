//! Draft computation shares row templates; only QSA and capture own chronology.
use crate::{
    artifact::Result,
    model::{Manifest, Operation},
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn present(m: &Manifest) -> bool {
    m.dynamic_batch_kernels.contains_key("draft_rows/begin/k0")
}

pub(super) fn register(m: &mut Manifest) -> Result<()> {
    if !present(m) {
        return Ok(());
    }
    for stage in ["begin", "pre", "post"] {
        let prefix = format!("draft_rows/{stage}/k");
        let mut names: Vec<_> = m
            .dynamic_batch_kernels
            .keys()
            .filter_map(|name| {
                name.strip_prefix(&prefix)
                    .and_then(|n| n.parse::<usize>().ok())
                    .map(|i| (i, name.clone()))
            })
            .collect();
        names.sort_by_key(|(i, _)| *i);
        if names.is_empty() || names.iter().enumerate().any(|(at, (i, _))| at != *i) {
            return Err("Incomplete draft row stage".into());
        }
        m.programs.insert(
            format!("draft_rows/{stage}"),
            names
                .into_iter()
                .map(|(_, name)| Operation::Kernel { name })
                .collect(),
        );
    }
    m.programs.insert(
        "draft_rows/head".into(),
        super::plan::section("draft_rows", "head", 9, None),
    );
    Ok(())
}

pub(super) fn causal_begin(m: &Manifest, tokens: usize, gather: bool) -> Vec<Operation> {
    let mut ops = Vec::new();
    if tokens != 512 {
        ops.push(Operation::Copy {
            source: format!("DraftM{tokens}_Embedding"),
            destination: "DraftM512_Embedding".into(),
            bytes: tokens * 5120,
        });
    }
    if gather {
        ops.push(Operation::Kernel {
            name: "causal_rows/warm/gather".into(),
        });
    } else {
        ops.push(Operation::Copy {
            source: "DraftM1_Condition".into(),
            destination: "DraftM512_Condition".into(),
            bytes: 20480,
        });
    }
    ops.push(Operation::Kernel {
        name: "causal_rows/warm/length".into(),
    });
    if m.vision.is_some() {
        ops.push(Operation::Kernel {
            name: "causal_rows/warm/overlay".into(),
        });
    }
    ops
}

pub(super) fn configure(m: &mut Manifest) -> Result<()> {
    if !present(m) {
        return Ok(());
    }
    let head: Vec<_> = m.programs["flash_decode/head"]
        .iter()
        .map(|op| {
            let Operation::Kernel { name } = op else {
                return Err("Invalid common output head operation".to_owned());
            };
            Ok(Operation::Kernel {
                name: name.replace("flash_decode/head/", "draft_rows/head/"),
            })
        })
        .collect::<Result<_>>()?;
    m.programs.insert("draft_rows/head".into(), head.clone());
    let spec = m.mtp.clone().ok_or("Missing draft state contract")?;
    let mut head_program = vec![Operation::Copy {
        source: "DraftM1_Condition".into(),
        destination: "DraftRows_Hidden".into(),
        bytes: 20480,
    }];
    head_program.extend(head);
    head_program.extend([
        Operation::Copy {
            source: "FlashDecode_Logits".into(),
            destination: spec.draft_logits.clone(),
            bytes: m.vocab * 4,
        },
        Operation::Kernel {
            name: "draft_rows/selections".into(),
        },
    ]);
    for warm in &spec.warm_plans {
        let n = warm.tokens;
        let causal = super::causal_rows::present(m);
        let mut body = (0..2 + usize::from(m.vision.is_some()))
            .map(|i| Operation::Kernel {
                name: format!("mtp_warm_m{n}/begin/k{i}"),
            })
            .collect::<Vec<_>>();
        if causal {
            body = causal_begin(m, n, true);
        }
        body.extend([
            Operation::Copy {
                source: format!("DraftM{}_Embedding", if causal { 512 } else { n }),
                destination: "DraftRows_Embedding".into(),
                bytes: n * 5120,
            },
            Operation::Copy {
                source: format!("DraftM{}_Condition", if causal { 512 } else { n }),
                destination: "DraftRows_Condition".into(),
                bytes: n * 20480,
            },
        ]);
        for stage in ["begin", "pre"] {
            body.extend(m.programs[&format!("draft_rows/{stage}")].clone());
        }
        body.extend((0..19).map(|i| Operation::Kernel {
            name: if causal {
                format!("causal_rows/draft/k{i}")
            } else {
                format!("draft_rows/causal_m{n}/k{i}")
            },
        }));
        body.extend(m.programs["draft_rows/post"].clone());
        body.extend([
            Operation::Kernel {
                name: if causal {
                    "causal_rows/warm/advance".into()
                } else {
                    format!("mtp_warm_m{n}/end/k0")
                },
            },
            Operation::Kernel {
                name: if causal {
                    "causal_rows/warm/capture".into()
                } else {
                    format!("draft_rows/capture_m{n}")
                },
            },
        ]);
        if n == 1 {
            let mut draft = body[1..].to_vec();
            if causal {
                // Replace gather with the carried endpoint condition.
                draft = causal_begin(m, 1, false);
                draft.extend(body[causal_begin(m, 1, true).len()..].to_vec());
            }
            draft.extend(head_program.clone());
            m.programs.insert(spec.draft_program.clone(), draft);
        }
        m.programs.insert(warm.program.clone(), body);
        m.programs
            .insert(warm.head_program.clone(), head_program.clone());
    }
    let mut programs = BTreeSet::new();
    let mut capacities = BTreeMap::new();
    for stage in ["begin", "pre", "post", "head"] {
        let name = format!("draft_rows/{stage}");
        capacities.insert(name.clone(), if stage == "head" { 128 } else { 512 });
        programs.insert(name);
    }
    for warm in &spec.warm_plans {
        capacities.insert(warm.program.clone(), warm.tokens);
        programs.insert(warm.program.clone());
        capacities.insert(warm.head_program.clone(), 128);
        programs.insert(warm.head_program.clone());
    }
    capacities.insert(spec.draft_program.clone(), 1);
    programs.insert(spec.draft_program);
    let rows = m
        .row_execution
        .as_mut()
        .ok_or("Missing shared row execution")?;
    rows.programs.extend(programs);
    rows.program_capacities.extend(capacities);
    Ok(())
}
