//! Compressed expert banks and exact A8 split steps owned by the Flash recipe.
use crate::{
    artifact::{Argument, Dtype, Kernel, Result},
    model::{Manifest, Operation},
};
use orinfer_model_sdk::abi::ExpertBank;
use std::collections::BTreeMap;

pub(super) fn build(manifest: &mut Manifest, kernels: &[Kernel]) -> Result<()> {
    if kernels.is_empty() && manifest.buffers.iter().any(|b| b.name.starts_with("E8P_")) {
        return Err("Missing exact expert execution contracts".into());
    }
    let contracts: BTreeMap<_, _> = kernels.iter().map(|k| (k.name.as_str(), k)).collect();
    let mut splits = BTreeMap::new();
    for kernel in kernels {
        if let Some(projection) = kernel.name.strip_suffix("/a8_split") {
            let target = contracts
                .get(projection)
                .ok_or("Missing exact expert projection")?;
            if !matches!(
                target.symbol.as_str(),
                "exact_e8p_int4_kernel" | "exact_e8p_shared_input_kernel"
            ) || target.args.len() != 13
                || kernel.symbol != "pack_exact_a8_kernel"
                || kernel.args.len() != 4
            {
                return Err("Unexpected exact expert ABI".into());
            }
            for (split_index, project_index) in [(1, 3), (2, 4), (3, 8)] {
                let (Argument::Buffer { name: a }, Argument::Buffer { name: b }) =
                    (&kernel.args[split_index], &target.args[project_index])
                else {
                    return Err("Exact expert workspace must be a whole buffer".into());
                };
                if a != b {
                    return Err("Exact expert split/projection workspace mismatch".into());
                }
            }
            splits.insert(projection.to_owned(), kernel.name.clone());
        }
    }
    if kernels.iter().any(|k| {
        matches!(
            k.symbol.as_str(),
            "exact_e8p_int4_kernel" | "exact_e8p_shared_input_kernel"
        ) && !splits.contains_key(&k.name)
    }) {
        return Err("Missing exact expert A8 split".into());
    }
    for operations in manifest.programs.values_mut() {
        let mut expanded = Vec::with_capacity(operations.len());
        for operation in std::mem::take(operations) {
            if let Operation::Kernel { name } = &operation
                && let Some(split) = splits.remove(name)
            {
                expanded.push(Operation::Kernel { name: split });
            }
            expanded.push(operation);
        }
        *operations = expanded;
    }
    if !splits.is_empty() {
        return Err("Unregistered exact expert execution step".into());
    }
    Ok(())
}

pub(super) fn banks(manifest: &Manifest, kernels: &[Kernel]) -> Result<Vec<ExpertBank>> {
    let buffers: BTreeMap<_, _> = manifest
        .buffers
        .iter()
        .map(|b| (b.name.as_str(), b))
        .collect();
    (0..48)
        .map(|layer| {
            let prefix = format!("W_blk.{layer}.");
            let names: Vec<_> = ["ffn_gate_up_exps", "ffn_down_exps"]
                .into_iter()
                .flat_map(|role| {
                    ["packed", "scale"].into_iter().map({
                        let prefix = prefix.clone();
                        move |part| format!("{prefix}{role}.weight_{part}")
                    })
                })
                .collect();
            let experts = buffers
                .get(names[0].as_str())
                .and_then(|b| b.shape.first())
                .copied()
                .ok_or("Missing expert bank")?;
            let mut consumers = BTreeMap::new();
            for k in kernels {
                if !k
                    .args
                    .iter()
                    .any(|a| matches!(a, Argument::Buffer { name } if name == &names[0]))
                {
                    continue;
                }
                // Exported host argument order, not the TileLang function's
                // declaration order. The exact path adds two nibble planes and sum.
                let histogram = if matches!(
                    k.symbol.as_str(),
                    "exact_e8p_int4_kernel" | "exact_e8p_shared_input_kernel"
                ) && k.args.len() == 13
                {
                    2
                } else if k.args.len() == 11 {
                    4
                } else {
                    return Err(format!("{}: unsupported expert consumer ABI", k.name));
                };
                let Some(Argument::Buffer { name }) = k.args.get(histogram) else {
                    return Err("Unsupported grouped expert histogram ABI".into());
                };
                let b = buffers
                    .get(name.as_str())
                    .ok_or("Missing expert histogram")?;
                if b.dtype != Dtype::I32 || b.shape != [experts] {
                    return Err(format!("{}: unsupported expert consumer ABI", k.name));
                }
                consumers.insert(k.name.clone(), histogram);
            }
            if consumers.is_empty() && !kernels.is_empty() {
                return Err("Missing expert gate/up consumers".into());
            }
            Ok(ExpertBank {
                experts,
                buffers: names,
                consumers,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kernel(name: &str, symbol: &str, args: &[&str]) -> Kernel {
        let identity = serde_json::json!({"file":"kernel.cubin","sha256":"0".repeat(64)});
        serde_json::from_value(serde_json::json!({
            "name":name,"symbol":symbol,"module":identity,"source":identity,"host_abi":identity,
            "grid":[1,1,1],"block":[128,1,1],"shared_memory_bytes":0,"cooperative":false,
            "args":args.iter().map(|name| serde_json::json!({"kind":"buffer","name":name})).collect::<Vec<_>>()
        })).unwrap()
    }

    fn fixture() -> (Manifest, Vec<Kernel>) {
        let mut m: Manifest = serde_json::from_value(serde_json::json!({
            "schema_version":1,"target":"sm_87","model":"test","chunk_tokens":4096,
            "max_context":262144,"vocab":16,"toolchain":{},"buffers":[],
            "reset_buffers":[],"input":"Input","token":"Token","status":"Status",
            "logits":"Logits","position":"Position","weight_bytes":0,
            "weight_parameters":0,"weight_scope":"test"
        }))
        .unwrap();
        m.programs.insert(
            "decode".into(),
            vec![Operation::Kernel {
                name: "decode/layer0/k29".into(),
            }],
        );
        let projection = kernel(
            "decode/layer0/k29",
            "exact_e8p_int4_kernel",
            &[
                "AS", "Book", "Counts", "High", "Low", "Offsets", "Out", "P", "Sum", "TC", "TE",
                "TR", "WS",
            ],
        );
        let split = kernel(
            "decode/layer0/k29/a8_split",
            "pack_exact_a8_kernel",
            &["A", "High", "Low", "Sum"],
        );
        (m, vec![projection, split])
    }

    #[test]
    fn exact_split_precedes_projection_and_validates_actual_abi() {
        let (mut m, kernels) = fixture();
        build(&mut m, &kernels).unwrap();
        assert!(
            matches!(&m.programs["decode"][0], Operation::Kernel {name} if name.ends_with("/a8_split"))
        );
        assert!(
            matches!(&m.programs["decode"][1], Operation::Kernel {name} if name.ends_with("/k29"))
        );
        let (mut m, mut kernels) = fixture();
        kernels[1].args[1] = Argument::Buffer {
            name: "wrong".into(),
        };
        assert!(
            build(&mut m, &kernels)
                .unwrap_err()
                .contains("workspace mismatch")
        );
        let (mut m, kernels) = fixture();
        assert!(
            build(&mut m, &kernels[..1])
                .unwrap_err()
                .contains("Missing")
        );
        let (mut m, kernels) = fixture();
        m.programs.clear();
        assert!(
            build(&mut m, &kernels)
                .unwrap_err()
                .contains("Unregistered")
        );
    }
}
