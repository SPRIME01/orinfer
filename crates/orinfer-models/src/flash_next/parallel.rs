//! Fork only branches certified independent by their generated argument access.
use crate::{
    artifact::{Argument, Kernel, Result},
    model::{Manifest, Operation},
};
use std::collections::BTreeMap;

pub(super) fn build(manifest: &mut Manifest, kernels: &[Kernel]) -> Result<()> {
    manifest.graph_branches.clear();
    if !kernels
        .iter()
        .any(|k| k.name.starts_with("decode/") && !k.readonly_arguments.is_empty())
    {
        return Ok(());
    }
    let ops = manifest
        .programs
        .get("decode")
        .ok_or("Missing decode program")?;
    let positions: BTreeMap<_, _> = ops
        .iter()
        .enumerate()
        .filter_map(|(i, op)| {
            if let Operation::Kernel { name } = op {
                Some((name.as_str(), i))
            } else {
                None
            }
        })
        .collect();
    let select = |layer: usize, weight: &str| -> Result<&str> {
        let prefix = format!("decode/layer{layer}/");
        let mut found = kernels.iter().filter(|k| {
            k.name.starts_with(&prefix)
                && k.args
                    .iter()
                    .any(|a| matches!(a, Argument::Buffer { name } if name == weight))
        });
        let k = found
            .next()
            .ok_or_else(|| format!("Missing branch projection {weight}"))?;
        if found.next().is_some() {
            return Err("Ambiguous branch projection".into());
        }
        Ok(k.name.as_str())
    };
    let adjacent = |name: &str, offset: isize| -> Result<&str> {
        let i = positions
            .get(name)
            .ok_or("Branch endpoint is not executed")?;
        let Some(Operation::Kernel { name }) =
            i.checked_add_signed(offset).and_then(|i| ops.get(i))
        else {
            return Err("Missing adjacent branch operation".into());
        };
        Ok(name.as_str())
    };
    let mut branches = Vec::new();
    for layer in 0..48 {
        let weight = |suffix: &str| format!("W_blk.{layer}.{suffix}");
        for part in ["hc_attn", "hc_ffn"] {
            let root = select(layer, &weight(&format!("{part}_down.weight_value")))?;
            let inject = select(layer, &weight(&format!("{part}_inject.weight_value")))?;
            branches.push([root, inject, inject, adjacent(inject, 1)?].map(str::to_owned));
        }
        if layer % 4 != 3 {
            let qkv = select(layer, &format!("Q4_blk.{layer}.attn_qkv_packed"))?;
            let gate = select(layer, &format!("Q4_blk.{layer}.attn_gate_packed"))?;
            let beta = select(layer, &weight("ssm_beta.weight_weight"))?;
            let ab = adjacent(beta, 1)?;
            branches.push([qkv, gate, ab, adjacent(ab, 1)?].map(str::to_owned));
        }
        let expert = select(layer, &weight("ffn_gate_up_exps.weight_packed"))?;
        let shared = select(layer, &weight("ffn_gate_shexp.weight_weight"))?;
        let gate = select(layer, &weight("ffn_gate_inp_shexp.weight_value"))?;
        branches.push([expert, adjacent(shared, -1)?, gate, adjacent(gate, 1)?].map(str::to_owned));
    }
    branches.sort_by_key(|r| positions[r[0].as_str()]);
    manifest.graph_branches.insert("decode".into(), branches);
    orinfer_model_sdk::model::validate_graph_branches(manifest, kernels)
}
