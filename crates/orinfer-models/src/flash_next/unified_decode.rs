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

pub(super) const COMPUTATION_CAPACITY: usize = 1024;

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
            && (b.data.is_some()
                || b.bytes()?
                    != stride
                        * if matches!(role.as_str(), "Logits" | "Selected") {
                            CAPACITY
                        } else {
                            COMPUTATION_CAPACITY
                        })
        {
            return Err("Invalid canonical decode row arena".into());
        }
    }
    for (section, count) in super::batching::stages() {
        if section.ends_with("/private") {
            continue;
        }
        m.programs.insert(
            format!("{PREFIX}/{section}"),
            super::plan::section(PREFIX, &section, count, None),
        );
    }
    super::draft_rows::register(m)?;
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
                && (name.contains("/rows/") || name.contains("/skip/"))
        })
        .map(|name| Operation::Kernel { name: name.clone() })
        .collect::<Vec<_>>();
    if local.is_empty() {
        return Err("Missing compiled decode row selections".into());
    }
    m.programs.insert("flash_decode/local".into(), local);
    super::mixed::register(m)
}

pub(super) fn configure(m: &mut Manifest, kernels: &[crate::artifact::Kernel]) -> Result<()> {
    m.programs.remove("flash_decode/local");
    super::draft_rows::configure(m)?;
    for kernel in kernels
        .iter()
        .filter(|k| k.symbol == "rows_gdn_conv_kernel")
    {
        let section = kernel
            .name
            .strip_suffix("/k0")
            .ok_or("Invalid compact convolution slot")?;
        let program = m
            .programs
            .get_mut(section)
            .ok_or("Missing private convolution program")?;
        let copy = format!("{section}/k1");
        program.retain(|op| !matches!(op, Operation::Kernel { name } if name == &copy));
    }
    let programs = m
        .programs
        .keys()
        .filter(|name| {
            (name.starts_with("flash_decode/") && !name.ends_with("/private"))
                || [
                    "draft_rows/begin",
                    "draft_rows/pre",
                    "draft_rows/post",
                    "draft_rows/head",
                ]
                .contains(&name.as_str())
        })
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut variants: BTreeMap<String, Vec<KernelVariant>> = BTreeMap::new();
    let slots = programs
        .iter()
        .flat_map(|program| m.programs[program].iter())
        .filter_map(|op| {
            if let Operation::Kernel { name } = op {
                Some(name.as_str())
            } else {
                None
            }
        })
        .collect::<BTreeSet<_>>();
    for kernel in kernels {
        let Some((slot, variant)) = selection(&kernel.name)? else {
            continue;
        };
        if !slots.contains(slot) {
            return Err("Decode row selection has no canonical operation".into());
        }
        variants.entry(slot.into()).or_default().push(variant);
    }
    for choices in variants.values_mut() {
        choices.sort_by_key(|v| v.min_rows);
        if choices.windows(2).any(|v| v[0].max_rows >= v[1].min_rows) {
            return Err("Overlapping compiled decode row selections".into());
        }
    }
    let computation_capacity = COMPUTATION_CAPACITY;
    let rows = m.row_execution.get_or_insert_with(|| RowExecution {
        capacity: CAPACITY,
        programs: BTreeSet::new(),
        program_capacities: BTreeMap::new(),
        variants: BTreeMap::new(),
    });
    for program in programs {
        let capacity = if program.contains("/packet/") || program.ends_with("/head") {
            CAPACITY
        } else if program.starts_with("draft_rows/") {
            512
        } else {
            computation_capacity
        };
        rows.program_capacities.insert(program.clone(), capacity);
        rows.programs.insert(program);
    }
    rows.variants.extend(variants);
    // A single decoder is a supported packet; advertise it after mixed plans
    // are registered, since their independent head profiles start at two.
    m.batch_profiles.insert(0, 1);
    super::rows::verification(m)?;
    Ok(())
}

/// Build tools encode local choices in schema-1 binding identities. There is
/// no batch-specific algorithm, fusion or weight-name dispatch in serving.
fn selection(name: &str) -> Result<Option<(&str, KernelVariant)>> {
    if !name.starts_with("flash_decode/") && !name.starts_with("draft_rows/") {
        return Ok(None);
    }
    let (slot, range, skip) = if let Some((slot, range)) = name.split_once("/rows/") {
        (slot, range, false)
    } else if let Some((slot, range)) = name.split_once("/skip/") {
        (slot, range, true)
    } else {
        return Ok(None);
    };
    let (first, last) = range
        .split_once('-')
        .ok_or("Invalid decode row selection")?;
    let min_rows: usize = first.parse().map_err(|_| "Invalid decode row bound")?;
    let max_rows: usize = last.parse().map_err(|_| "Invalid decode row bound")?;
    if min_rows == 0 || min_rows > max_rows || max_rows > 1024 {
        return Err("Decode row selection outside its allocation".into());
    }
    Ok(Some((
        slot,
        KernelVariant {
            min_rows,
            max_rows,
            multiple_of: 1,
            kernels: if skip { vec![] } else { vec![name.into()] },
        },
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compiled_ranges_select_exact_rows_or_dynamic_tail() {
        for (range, first, last) in [("3-3", 3, 3), ("5-7", 5, 7), ("33-128", 33, 128)] {
            let name = format!("flash_decode/layer0/pre/k1/rows/{range}");
            let (slot, v) = selection(&name).unwrap().unwrap();
            assert_eq!(slot, "flash_decode/layer0/pre/k1");
            assert_eq!((v.min_rows, v.max_rows), (first, last));
            assert_eq!(v.kernels, [name]);
        }
        assert!(
            selection("flash_decode/layer0/post/k2/skip/1-128")
                .unwrap()
                .unwrap()
                .1
                .kernels
                .is_empty()
        );
        assert!(selection("flash_decode/layer0/post/k2").unwrap().is_none());
        assert!(selection("prefill/layer0/rows/invalid").unwrap().is_none());
        for range in ["0-3", "3-2", "1-1025", "x-8", "8", "8-8/a8_split"] {
            assert!(selection(&format!("flash_decode/head/k1/rows/{range}")).is_err());
        }
    }
}
