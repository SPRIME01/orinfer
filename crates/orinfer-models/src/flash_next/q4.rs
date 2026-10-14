//! Row-Q4 input transforms are part of the registered Flash execution recipe.
use crate::{
    artifact::{Access, Argument, Dtype, Kernel, Result},
    model::{Manifest, Operation},
};
use std::collections::{BTreeMap, BTreeSet};

/// Creation receives storage contracts before kernels are attached. Derive
/// precision variants from roles, never from package kernel enumeration.
pub(super) fn build(manifest: &mut Manifest) -> Result<()> {
    let shared_target = super::unified_decode::shared(manifest);
    let shared_draft = super::draft_rows::present(manifest);
    let buffers: BTreeMap<_, _> = manifest
        .buffers
        .iter()
        .map(|b| (b.name.as_str(), b))
        .collect();
    let mut rotations = BTreeMap::new();
    let mut workspaces = BTreeSet::new();
    for layer in 0..48 {
        let (role, n) = if layer % 4 == 3 {
            ("attn_q", 12288)
        } else {
            ("attn_gate", 6144)
        };
        let prefix = format!("Q4_blk.{layer}.{role}");
        let qkv_prefix = format!("Q4_blk.{layer}.attn_qkv");
        let qkv = buffers.contains_key(format!("{qkv_prefix}_packed").as_str());
        if qkv && (layer % 4 == 3 || !buffers.contains_key(format!("{prefix}_packed").as_str())) {
            return Err(format!(
                "Row-Q4 QKV requires a paired GDN Z projection: layer{layer}"
            ));
        }
        if !buffers.contains_key(format!("{prefix}_packed").as_str()) {
            continue;
        }
        workspaces.insert(("", 2560));
        for (suffix, dtype, shape) in [
            ("packed", Dtype::U16, vec![n, 640]),
            ("scale", Dtype::F16, vec![n]),
            ("signs", Dtype::I8, vec![2560]),
        ] {
            let name = format!("{prefix}_{suffix}");
            if buffers.get(name.as_str()).is_none_or(|b| {
                b.dtype != dtype || b.shape != shape || b.access != Access::Read || b.data.is_none()
            }) {
                return Err(format!("Invalid row-Q4 weight role: {name}"));
            }
        }
        if qkv {
            // QKV and Z share the same HC input and the stored Z sign vector.
            for (suffix, dtype, shape) in [
                ("packed", Dtype::U16, vec![10240, 640]),
                ("scale", Dtype::F16, vec![10240]),
            ] {
                let name = format!("{qkv_prefix}_{suffix}");
                if buffers.get(name.as_str()).is_none_or(|b| {
                    b.dtype != dtype
                        || b.shape != shape
                        || b.access != Access::Read
                        || b.data.is_none()
                }) {
                    return Err(format!("Invalid row-Q4 weight role: {name}"));
                }
            }
        }
        for rows in (1..=8).filter(|&n| n == 1 || !shared_target) {
            let program = if rows == 1 {
                "decode".into()
            } else {
                format!("verify_m{rows}")
            };
            // PLE adds nine decode slots and ten verification slots in layer1.
            let slot = if layer == 1 {
                if rows == 1 { 16 } else { 17 }
            } else if layer % 4 == 3 {
                6
            } else {
                7
            };
            let projection = format!("{program}/layer{layer}/k{}", slot - usize::from(qkv));
            if manifest.programs.get(&program).is_none_or(|ops| {
                !ops.iter()
                    .any(|op| matches!(op, Operation::Kernel {name} if name==&projection))
            }) {
                return Err(format!(
                    "Missing registered row-Q4 projection: {projection}"
                ));
            }
            rotations.insert(
                projection,
                format!("{program}/layer{layer}/k{slot}/q4_rotation"),
            );
        }
    }
    for layer in 0..48 {
        let role = if layer % 4 == 3 {
            "attn_output"
        } else {
            "ssm_out"
        };
        let prefix = format!("Q4_blk.{layer}.{role}");
        if !buffers.contains_key(format!("{prefix}_packed").as_str()) {
            continue;
        }
        for (suffix, dtype, shape) in [
            ("packed", Dtype::U16, vec![2560, 1536]),
            ("scale", Dtype::F16, vec![2560]),
            ("signs", Dtype::I8, vec![6144]),
        ] {
            let name = format!("{prefix}_{suffix}");
            if buffers.get(name.as_str()).is_none_or(|b| {
                b.dtype != dtype || b.shape != shape || b.access != Access::Read || b.data.is_none()
            }) {
                return Err(format!("Invalid row-Q4 attention output role: {name}"));
            }
        }
        workspaces.insert(("Out_", 6144));
        for rows in (1..=8).filter(|&n| n == 1 || !shared_target) {
            let program = if rows == 1 {
                "decode".into()
            } else {
                format!("verify_m{rows}")
            };
            let slot = if layer == 1 {
                if rows == 1 { 24 } else { 26 }
            } else if layer % 4 == 3 {
                if rows == 1 { 30 } else { 31 }
            } else if rows == 1 {
                15
            } else {
                16
            };
            let projection = format!("{program}/layer{layer}/k{slot}");
            if manifest.programs.get(&program).is_none_or(|ops| {
                !ops.iter()
                    .any(|op| matches!(op,Operation::Kernel {name} if name==&projection))
            }) {
                return Err(format!(
                    "Missing registered row-Q4 attention output: {projection}"
                ));
            }
            rotations.insert(projection.clone(), format!("{projection}/q4_rotation"));
        }
    }
    if buffers.contains_key("Q4_output_packed") {
        workspaces.insert(("", 2560));
        for (suffix, dtype, shape) in [
            ("packed", Dtype::U16, vec![248320, 640]),
            ("scale", Dtype::F16, vec![248320]),
            ("signs", Dtype::I8, vec![2560]),
        ] {
            let name = format!("Q4_output_{suffix}");
            if buffers.get(name.as_str()).is_none_or(|b| {
                b.dtype != dtype || b.shape != shape || b.access != Access::Read || b.data.is_none()
            }) {
                return Err(format!("Invalid row-Q4 output role: {name}"));
            }
        }
        let mut heads = vec!["head_m1/body/k5".to_owned()];
        if manifest.mtp.is_some() {
            if !shared_draft {
                heads.extend((1..=8).map(|rows| format!("mtp_head_m{rows}/body/k5")));
            }
            if !shared_target {
                heads.extend((2..=8).map(|rows| format!("verify_m{rows}/head/k5")));
            }
        }
        for projection in heads {
            if !manifest
                .programs
                .values()
                .flatten()
                .any(|op| matches!(op,Operation::Kernel {name} if name==&projection))
            {
                return Err(format!("Missing registered row-Q4 head: {projection}"));
            }
            rotations.insert(projection.clone(), format!("{projection}/q4_rotation"));
        }
    }
    if rotations.is_empty() {
        return Ok(());
    }
    for (prefix, k) in workspaces {
        for rows in 1..=8 {
            for (suffix, dtype, shape) in [
                ("Activation", Dtype::I8, vec![rows, k]),
                ("Scale", Dtype::F16, vec![rows]),
            ] {
                let name = format!("Q4_{prefix}M{rows}_{suffix}");
                if buffers.get(name.as_str()).is_none_or(|b| {
                    b.dtype != dtype
                        || b.shape != shape
                        || b.access != Access::ReadWrite
                        || b.data.is_some()
                }) {
                    return Err(format!("Invalid row-Q4 rotation workspace: {name}"));
                }
            }
        }
    }
    for operations in manifest.programs.values_mut() {
        let mut expanded = Vec::with_capacity(operations.len());
        for operation in std::mem::take(operations) {
            if let Operation::Kernel { name } = &operation
                && let Some(rotation) = rotations.get(name)
            {
                expanded.push(Operation::Kernel {
                    name: rotation.clone(),
                });
            }
            expanded.push(operation);
        }
        *operations = expanded;
    }
    Ok(())
}

/// Attach transforms published by the shared compute lowering. Recipe slot
/// counts describe logical operators; implementations can add prerequisites
/// without duplicating the model's single/batch/mixed execution recipes.
pub(super) fn prerequisites(manifest: &mut Manifest, kernels: &[Kernel]) -> Result<()> {
    let contracts: BTreeMap<_, _> = kernels.iter().map(|k| (k.name.as_str(), k)).collect();
    let executed: BTreeSet<_> = manifest
        .programs
        .values()
        .flatten()
        .filter_map(|op| {
            if let Operation::Kernel { name } = op {
                Some(name.clone())
            } else {
                None
            }
        })
        .collect();
    let mut pending = BTreeMap::new();
    for rotation in kernels {
        let Some(projection) = rotation.name.strip_suffix("/q4_rotation") else {
            continue;
        };
        let target = contracts
            .get(projection)
            .ok_or("Missing Q4 projection contract")?;
        if !(4..=5).contains(&rotation.args.len()) || !(5..=6).contains(&target.args.len()) {
            return Err("Unexpected Q4 prerequisite ABI".into());
        }
        for index in 0..2 {
            if !matches!((&rotation.args[index], &target.args[index]),
                (Argument::Buffer {name: a}, Argument::Buffer {name: b}) if a == b)
            {
                return Err("Q4 transform/projection workspace mismatch".into());
            }
        }
        let Argument::Buffer { name: signs } = &rotation.args[2] else {
            return Err("Missing Q4 rotation signs".into());
        };
        let Argument::Buffer { name: packed } = &target.args[4] else {
            return Err("Missing Q4 projection weights".into());
        };
        if !signs.starts_with("Q4_")
            || !signs.ends_with("_signs")
            || !packed.starts_with("Q4_")
            || !packed.ends_with("_packed")
        {
            return Err("Invalid Q4 prerequisite operands".into());
        }
        pending.insert(projection.to_owned(), rotation.name.clone());
    }
    for operations in manifest.programs.values_mut() {
        // Aliases are independent programs. A prerequisite executed in one
        // program must never suppress the transform in another alias.
        let existing: BTreeSet<_> = operations
            .iter()
            .filter_map(|op| match op {
                Operation::Kernel { name } => Some(name.clone()),
                _ => None,
            })
            .collect();
        let mut expanded = Vec::with_capacity(operations.len());
        for operation in std::mem::take(operations) {
            if let Operation::Kernel { name } = &operation
                && let Some(rotation) = pending.get(name)
                && !existing.contains(rotation)
            {
                expanded.push(Operation::Kernel {
                    name: rotation.clone(),
                });
            }
            expanded.push(operation);
        }
        *operations = expanded;
    }
    if pending.iter().any(|(projection, rotation)| {
        !executed.contains(projection.as_str()) && !executed.contains(rotation.as_str())
    }) {
        return Err("Unregistered Q4 prerequisite".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn contract(name: &str, args: &[&str]) -> Kernel {
        let asset = serde_json::json!({"file":"test.cubin","sha256":"0".repeat(64)});
        serde_json::from_value(serde_json::json!({
            "name":name,"symbol":"main_kernel","module":asset,"source":asset,"host_abi":asset,
            "grid":[1,1,1],"block":[128,1,1],"shared_memory_bytes":0,"cooperative":false,
            "args":args.iter().map(|name| serde_json::json!({"kind":"buffer","name":name})).collect::<Vec<_>>()
        })).unwrap()
    }

    #[test]
    fn prerequisites_cover_every_program_alias_and_are_idempotent() {
        let mut m = manifest(0);
        m.programs.clear();
        let projection = "prefill_m4096/layer0/k6";
        let rotation = format!("{projection}/q4_rotation");
        let kernels = vec![
            contract(
                projection,
                &["A", "AS", "Out", "Q4_test_scale", "Q4_test_packed"],
            ),
            contract(&rotation, &["A", "AS", "Q4_test_signs", "Input"]),
        ];
        for preinstalled in [false, true] {
            m.programs.clear();
            // Cover both destructive lookup across aliases and the case where
            // just one alias already contains its prerequisite.
            for program in ["prefill", "prefill_m2048", "prefill_m4096"] {
                let mut ops = vec![];
                if preinstalled && program == "prefill" {
                    ops.push(Operation::Kernel {
                        name: rotation.clone(),
                    });
                }
                ops.push(Operation::Kernel {
                    name: projection.into(),
                });
                m.programs.insert(program.into(), ops);
            }
            for _ in 0..2 {
                prerequisites(&mut m, &kernels).unwrap();
                for ops in m.programs.values() {
                    assert_eq!(ops.len(), 2);
                    assert!(matches!(&ops[0], Operation::Kernel { name } if name == &rotation));
                    assert!(matches!(&ops[1], Operation::Kernel { name } if name == projection));
                }
            }
        }
    }

    #[test]
    fn shared_lowering_expands_fixed_dynamic_and_mixed_recipes() {
        for section in [
            "flash_batch_m2/layer0/pre",
            "flash_dynamic_m4/layer0/pre",
            "flash_mixed_m128/layer0/pre",
        ] {
            let mut m = manifest(0);
            m.programs.clear();
            let projection = format!("{section}/k6");
            let rotation = format!("{projection}/q4_rotation");
            m.programs.insert(
                section.into(),
                vec![Operation::Kernel {
                    name: projection.clone(),
                }],
            );
            let mut kernels = vec![
                contract(
                    &projection,
                    &[
                        "A",
                        "AS",
                        "Out",
                        "Q4_blk.0.attn_qkv_scale",
                        "Q4_blk.0.attn_qkv_packed",
                    ],
                ),
                contract(&rotation, &["A", "AS", "Q4_blk.0.attn_gate_signs", "Input"]),
            ];
            prerequisites(&mut m, &kernels).unwrap();
            assert!(
                matches!(&m.programs[section][0], Operation::Kernel {name} if name == &rotation)
            );
            assert!(
                matches!(&m.programs[section][1], Operation::Kernel {name} if name == &projection)
            );
            prerequisites(&mut m, &kernels).unwrap();
            assert_eq!(m.programs[section].len(), 2);
            m.programs.get_mut(section).unwrap().remove(0);
            kernels[1].args[0] = Argument::Buffer {
                name: "wrong".into(),
            };
            assert!(
                prerequisites(&mut m, &kernels)
                    .unwrap_err()
                    .contains("workspace mismatch")
            );
            kernels[1].args[0] = Argument::Buffer { name: "A".into() };
            m.programs.clear();
            assert!(
                prerequisites(&mut m, &kernels)
                    .unwrap_err()
                    .contains("Unregistered")
            );
        }
    }

    fn manifest(layer: usize) -> Manifest {
        let mut m: Manifest = serde_json::from_value(serde_json::json!({
            "schema_version":2,"target":"sm_87","model":"test","chunk_tokens":4096,
            "max_context":262144,"vocab":16,"toolchain":{},"buffers":[],
            "reset_buffers":[],"input":"Input","token":"Token","status":"Status",
            "logits":"Logits","position":"Position","weight_bytes":0,
            "weight_parameters":0,"weight_scope":"test"
        }))
        .unwrap();
        let (role, n) = if layer % 4 == 3 {
            ("attn_q", 12288)
        } else {
            ("attn_gate", 6144)
        };
        for (suffix, dtype, shape) in [
            ("packed", "u16", vec![n, 640]),
            ("scale", "f16", vec![n]),
            ("signs", "i8", vec![2560]),
        ] {
            m.buffers.push(
                serde_json::from_value(serde_json::json!({
                    "name":format!("Q4_blk.{layer}.{role}_{suffix}"),"dtype":dtype,"shape":shape,
                    "layout":"native_contiguous","alignment":256,"access":"read",
                    "data":{"tensor":"weight","sha256":"0".repeat(64)}
                }))
                .unwrap(),
            );
        }
        for rows in 1..=8 {
            for (suffix, dtype, shape) in [
                ("Activation", "i8", vec![rows, 2560]),
                ("Scale", "f16", vec![rows]),
            ] {
                m.buffers.push(serde_json::from_value(serde_json::json!({
                    "name":format!("Q4_M{rows}_{suffix}"),"dtype":dtype,"shape":shape,
                    "layout":"native_contiguous","alignment":256,"access":"read_write","data":null
                })).unwrap());
            }
            let program = if rows == 1 {
                "decode".into()
            } else {
                format!("verify_m{rows}")
            };
            let slot = if layer == 1 {
                if rows == 1 { 16 } else { 17 }
            } else if layer % 4 == 3 {
                6
            } else {
                7
            };
            m.programs.insert(
                program.clone(),
                vec![
                    Operation::Kernel {
                        name: format!("{program}/layer{layer}/k{slot}"),
                    },
                    Operation::Copy {
                        source: "HistoryOut".into(),
                        destination: format!("State_{layer}_conv"),
                        bytes: 61440,
                    },
                ],
            );
        }
        m.programs.insert(
            "mtp_draft".into(),
            vec![Operation::Kernel {
                name: "mtp_draft/layer48/k6".into(),
            }],
        );
        m
    }
    #[test]
    fn plan_before_kernel_attachment_covers_ple_and_all_verification_widths() {
        for layer in [0, 1, 3] {
            let mut m = manifest(layer);
            assert!(m.kernels.is_empty());
            build(&mut m).unwrap();
            for (program, ops) in &m.programs {
                if program == "mtp_draft" {
                    assert_eq!(ops.len(), 1);
                } else {
                    assert_eq!(ops.len(), 3);
                    assert!(
                        matches!(&ops[0],Operation::Kernel {name} if name.ends_with("/q4_rotation"))
                    );
                    assert!(matches!(&ops[2], Operation::Copy { bytes: 61440, .. }));
                }
            }
        }
    }
    #[test]
    fn invalid_rotation_storage_is_rejected_and_w8_is_unchanged() {
        let mut m = manifest(0);
        m.buffers[2].shape = vec![128];
        assert!(
            build(&mut m)
                .unwrap_err()
                .contains("Invalid row-Q4 weight role")
        );
        m.buffers.clear();
        build(&mut m).unwrap();
        assert_eq!(m.programs["decode"].len(), 2);
    }

    #[test]
    fn qkv_and_z_reuse_one_rotation_before_both_projections() {
        let mut m = manifest(0);
        for (suffix, dtype, shape) in [
            ("packed", "u16", vec![10240, 640]),
            ("scale", "f16", vec![10240]),
        ] {
            m.buffers.push(
                serde_json::from_value(serde_json::json!({
                    "name":format!("Q4_blk.0.attn_qkv_{suffix}"),"dtype":dtype,"shape":shape,
                    "layout":"native_contiguous","alignment":256,"access":"read",
                    "data":{"tensor":"weight","sha256":"0".repeat(64)}
                }))
                .unwrap(),
            );
        }
        for (program, ops) in &mut m.programs {
            if program != "mtp_draft" {
                ops.insert(
                    0,
                    Operation::Kernel {
                        name: format!("{program}/layer0/k6"),
                    },
                );
            }
        }
        build(&mut m).unwrap();
        for (program, ops) in &m.programs {
            if program != "mtp_draft" {
                assert_eq!(ops.len(), 4);
                assert!(
                    matches!(&ops[0],Operation::Kernel {name} if name.ends_with("/k7/q4_rotation"))
                );
                assert!(matches!(&ops[1],Operation::Kernel {name} if name.ends_with("/k6")));
                assert!(matches!(&ops[2],Operation::Kernel {name} if name.ends_with("/k7")));
            }
        }
        m.buffers.retain(|b| b.name != "Q4_blk.0.attn_gate_packed");
        assert!(
            build(&mut m)
                .unwrap_err()
                .contains("requires a paired GDN Z")
        );
    }

    #[test]
    fn head_rotation_covers_aliases_without_changing_prefill_head() {
        let mut m = manifest(0);
        for (suffix, dtype, shape) in [
            ("packed", "u16", vec![248320, 640]),
            ("scale", "f16", vec![248320]),
            ("signs", "i8", vec![2560]),
        ] {
            m.buffers.push(
                serde_json::from_value(serde_json::json!({
                    "name":format!("Q4_output_{suffix}"),"dtype":dtype,"shape":shape,
                    "layout":"native_contiguous","alignment":256,"access":"read",
                    "data":{"tensor":"weight","sha256":"0".repeat(64)}
                }))
                .unwrap(),
            );
        }
        let head = Operation::Kernel {
            name: "head_m1/body/k5".into(),
        };
        m.programs.get_mut("decode").unwrap().push(head.clone());
        m.programs.insert("head_m1".into(), vec![head]);
        m.programs.insert(
            "head_m4096".into(),
            vec![Operation::Kernel {
                name: "head_m4096/body/k5".into(),
            }],
        );
        build(&mut m).unwrap();
        assert!(
            matches!(&m.programs["head_m1"][0],Operation::Kernel {name} if name=="head_m1/body/k5/q4_rotation")
        );
        assert_eq!(m.programs["head_m4096"].len(), 1);
        assert_eq!(
            m.programs["decode"]
                .iter()
                .filter(
                    |op| matches!(op,Operation::Kernel {name} if name.ends_with("/q4_rotation"))
                )
                .count(),
            2
        );
        // Head-only variants need the same activation workspace validation.
        m.buffers.retain(|b| !b.name.starts_with("Q4_blk."));
        m.buffers
            .iter_mut()
            .find(|b| b.name == "Q4_M1_Scale")
            .unwrap()
            .shape = vec![2];
        assert!(
            build(&mut m)
                .unwrap_err()
                .contains("Invalid row-Q4 rotation workspace")
        );
    }

    #[test]
    fn attention_output_rotations_cover_ple_without_an_input_q4_variant() {
        for layer in [0, 1, 3, 47] {
            let mut m = manifest(layer);
            m.buffers.clear();
            let role = if layer % 4 == 3 {
                "attn_output"
            } else {
                "ssm_out"
            };
            for (suffix, dtype, shape) in [
                ("packed", "u16", vec![2560, 1536]),
                ("scale", "f16", vec![2560]),
                ("signs", "i8", vec![6144]),
            ] {
                m.buffers.push(serde_json::from_value(serde_json::json!({
                    "name":format!("Q4_blk.{layer}.{role}_{suffix}"),"dtype":dtype,"shape":shape,
                    "layout":"native_contiguous","alignment":256,"access":"read",
                    "data":{"tensor":"weight","sha256":"0".repeat(64)}
                })).unwrap());
            }
            for rows in 1..=8 {
                for (suffix, dtype, shape) in [
                    ("Activation", "i8", vec![rows, 6144]),
                    ("Scale", "f16", vec![rows]),
                ] {
                    m.buffers.push(serde_json::from_value(serde_json::json!({
                        "name":format!("Q4_Out_M{rows}_{suffix}"),"dtype":dtype,"shape":shape,
                        "layout":"native_contiguous","alignment":256,"access":"read_write","data":null
                    })).unwrap());
                }
                let program = if rows == 1 {
                    "decode".to_owned()
                } else {
                    format!("verify_m{rows}")
                };
                let slot = if layer == 1 {
                    if rows == 1 { 24 } else { 26 }
                } else if layer % 4 == 3 {
                    if rows == 1 { 30 } else { 31 }
                } else if rows == 1 {
                    15
                } else {
                    16
                };
                m.programs
                    .get_mut(&program)
                    .unwrap()
                    .push(Operation::Kernel {
                        name: format!("{program}/layer{layer}/k{slot}"),
                    });
            }
            build(&mut m).unwrap();
            for (program, ops) in &m.programs {
                if program != "mtp_draft" {
                    assert!(
                        matches!(&ops[ops.len()-2],Operation::Kernel {name} if name.ends_with("/q4_rotation"))
                    );
                    assert!(matches!(&ops[1], Operation::Copy { bytes: 61440, .. }));
                }
            }
            m.buffers
                .iter_mut()
                .find(|b| b.name == "Q4_Out_M4_Activation")
                .unwrap()
                .shape = vec![4, 2560];
            assert!(
                build(&mut m)
                    .unwrap_err()
                    .contains("Invalid row-Q4 rotation workspace")
            );
        }
    }
}
