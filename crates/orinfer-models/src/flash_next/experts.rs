//! The grouped E8P ABI puts the per-expert histogram in host argument 4.
use crate::{
    artifact::{Argument, Dtype, Kernel, Result},
    model::Manifest,
};
use orinfer_model_sdk::abi::ExpertBank;
use std::collections::BTreeMap;

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
                let Some(Argument::Buffer { name }) = k.args.get(4) else {
                    return Err("Unsupported grouped expert histogram ABI".into());
                };
                let b = buffers
                    .get(name.as_str())
                    .ok_or("Missing expert histogram")?;
                if k.args.len() != 11 || b.dtype != Dtype::I32 || b.shape != [experts] {
                    return Err(format!("{}: unsupported expert consumer ABI", k.name));
                }
                consumers.insert(k.name.clone(), 4);
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
