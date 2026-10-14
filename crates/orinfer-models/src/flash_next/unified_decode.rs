//! One bounded decode template; shape choices substitute local implementations.
use crate::{
    artifact::Result,
    model::{Manifest, Operation},
};
use orinfer_model_sdk::operators::rows::{KernelVariant, RowExecution};
use std::collections::{BTreeMap, BTreeSet};

pub(super) const CAPACITY: usize = 128;
pub(super) const PREFIX: &str = "flash_decode";
pub(super) const ARENA: &str = "FlashDecode_";

pub(super) fn present(m: &Manifest) -> bool {
    m.dynamic_batch_kernels
        .contains_key("flash_decode/begin/k0")
}

pub(super) fn register(m: &mut Manifest) -> Result<()> {
    let layout = m.batch_layout.as_ref().ok_or("Missing decode row layout")?;
    if layout.state_columns.is_empty() || m.state_pointer_table.is_none() {
        return Err("Unified decode needs request-private address bindings".into());
    }
    for (role, stride) in &layout.row_strides {
        // Only the roles consumed by the decode template need a row arena.
        let name = format!("{ARENA}{role}");
        if let Some(b) = m.buffers.iter().find(|b| b.name == name)
            && (b.data.is_some() || b.bytes()? != stride * CAPACITY)
        {
            return Err("Invalid canonical decode row arena".into());
        }
    }
    for (section, count) in super::batching::stages(CAPACITY) {
        if section.ends_with("/private") {
            continue;
        }
        m.programs.insert(
            format!("{PREFIX}/{section}"),
            super::plan::section(PREFIX, &section, count, None),
        );
    }
    for name in ["inputs", "end", "outputs"] {
        let kernel = format!("flash_decode/packet/{name}");
        if !m.dynamic_batch_kernels.contains_key(&kernel) {
            return Err("Missing address-table decode packet I/O".into());
        }
        m.programs
            .insert(kernel.clone(), vec![Operation::Kernel { name: kernel }]);
    }
    m.programs.insert(
        "flash_decode_ple".into(),
        (0..2)
            .map(|i| Operation::Kernel {
                name: format!("flash_decode_ple/k{i}"),
            })
            .collect(),
    );
    for layer in 0..48 {
        m.programs.insert(
            format!("flash_decode_state/layer{layer}"),
            super::plan::section(
                "flash_decode_state",
                &format!("layer{layer}"),
                if layer % 4 == 3 { 19 } else { 3 },
                None,
            ),
        );
    }
    // Model validation/lowering sees local implementation dependencies once.
    // This is removed after the variants have been installed, never executed.
    let local = m
        .dynamic_batch_kernels
        .keys()
        .filter(|name| {
            name.starts_with("flash_decode/")
                && ["small", "tiny", "quad", "pair", "four", "oct"]
                    .iter()
                    .any(|tag| name.ends_with(&format!("/{tag}")))
        })
        .map(|name| Operation::Kernel { name: name.clone() })
        .collect::<Vec<_>>();
    if local.is_empty() {
        return Err("Missing decode local implementations".into());
    }
    m.programs.insert("flash_decode/local".into(), local);
    super::mixed::register(m)
}

pub(super) fn configure(m: &mut Manifest, kernels: &[crate::artifact::Kernel]) -> Result<()> {
    if !present(m) {
        return Ok(());
    }
    m.programs.remove("flash_decode/local");
    let programs = m
        .programs
        .keys()
        .filter(|name| name.starts_with("flash_decode/") && !name.ends_with("/private"))
        .cloned()
        .collect::<BTreeSet<_>>();
    let contracts: BTreeMap<_, _> = kernels.iter().map(|k| (k.name.as_str(), k)).collect();
    let mut variants: BTreeMap<String, Vec<KernelVariant>> = BTreeMap::new();
    for program in &programs {
        for (index, op) in m.programs[program].iter().enumerate() {
            let Operation::Kernel { name } = op else {
                continue;
            };
            let small = format!("{name}/small");
            let replacement = |kernel: String| {
                let split = format!("{kernel}/a8_split");
                if m.dynamic_batch_kernels.contains_key(&split) {
                    vec![split, kernel]
                } else {
                    vec![kernel]
                }
            };
            let skip = format!("{name}/small_skip");
            let weight_role = |kernel: &crate::artifact::Kernel, suffix: &str| {
                kernel.args.iter().any(|a|matches!(a,crate::artifact::Argument::Buffer {name} if name.contains("hc_") && name.contains(suffix)))
            };
            let hc_up = contracts
                .get(small.as_str())
                .is_some_and(|k| weight_role(k, "_up.weight"));
            let hc_injection = contracts
                .get(small.as_str())
                .is_some_and(|k| weight_role(k, "_inject.weight"));
            let hc_mix = index > 0
                && if let Operation::Kernel { name: previous } = &m.programs[program][index - 1] {
                    contracts
                        .get(format!("{previous}/small").as_str())
                        .is_some_and(|k| weight_role(k, "_up.weight"))
                } else {
                    false
                };
            if m.dynamic_batch_kernels.contains_key(&small) {
                let local = variants.entry(name.clone()).or_default();
                let bands: &[(usize, usize, &[&str])] = &[
                    (1, 1, &["tiny", "small"]),
                    (2, 2, &["pair", "tiny", "small"]),
                    (3, 3, &["quad", "small"]),
                    (4, 4, &["four", "quad", "small"]),
                    (5, 7, &["small"]),
                    (8, 8, &["oct", "small"]),
                ];
                for &(min_rows, max_rows, tags) in bands {
                    let kernel = tags
                        .iter()
                        .map(|tag| format!("{name}/{tag}"))
                        .find(|key| m.dynamic_batch_kernels.contains_key(key))
                        .ok_or("Missing local decode fallback")?;
                    let kernels = replacement(kernel);
                    if let Some(last) = local.last_mut()
                        && last.max_rows + 1 == min_rows
                        && last.kernels == kernels
                    {
                        last.max_rows = max_rows;
                    } else {
                        local.push(KernelVariant {
                            min_rows,
                            max_rows,
                            multiple_of: 1,
                            kernels,
                        });
                    }
                }
                if hc_up || hc_injection {
                    local.last_mut().unwrap().max_rows = CAPACITY;
                }
            } else if m.dynamic_batch_kernels.contains_key(&skip) {
                variants
                    .entry(name.clone())
                    .or_default()
                    .push(KernelVariant {
                        min_rows: 1,
                        max_rows: if hc_mix { CAPACITY } else { 8 },
                        multiple_of: 1,
                        kernels: vec![],
                    });
            }
            if let Some(parent) = name.strip_suffix("/a8_split")
                && m.dynamic_batch_kernels
                    .contains_key(&format!("{parent}/small"))
            {
                variants
                    .entry(name.clone())
                    .or_default()
                    .push(KernelVariant {
                        min_rows: 1,
                        max_rows: 8,
                        multiple_of: 1,
                        kernels: vec![],
                    });
            }
            let medium = format!("{name}/medium");
            if !hc_up && !hc_injection && !hc_mix && m.dynamic_batch_kernels.contains_key(&medium) {
                variants
                    .entry(name.clone())
                    .or_default()
                    .push(KernelVariant {
                        min_rows: 9,
                        max_rows: 16,
                        multiple_of: 1,
                        kernels: vec![medium],
                    });
            }
        }
    }
    let rows = m.row_execution.get_or_insert_with(|| RowExecution {
        capacity: CAPACITY,
        programs: BTreeSet::new(),
        program_capacities: BTreeMap::new(),
        variants: BTreeMap::new(),
    });
    for program in programs {
        rows.program_capacities.insert(program.clone(), CAPACITY);
        rows.programs.insert(program);
    }
    rows.variants.extend(variants);
    Ok(())
}
