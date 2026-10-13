//! Generated, typed tensor-role bindings live in the native implementation.
use crate::artifact::Result;
use orinfer_model_sdk::{abi::plans::KernelBinding, catalog::Catalog};
mod generated {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
pub fn kernels(catalog: &Catalog, all: bool) -> Result<Vec<crate::artifact::Kernel>> {
    check(catalog)?;
    generated::BINDINGS
        .iter()
        .filter(|b| all || b.plan_contract)
        .map(|b| catalog.bind(b.definition, b.name.into(), b.grid, arguments(b.arguments)?))
        .collect()
}
fn arguments(index: usize) -> Result<Vec<crate::artifact::Argument>> {
    generated::ARGUMENTS
        .get(index)
        .ok_or("Unknown native argument group")?
        .iter()
        .map(|a| a.bind(generated::ROLES))
        .collect()
}
pub(crate) fn buffer_argument(name: &str, position: usize) -> Result<String> {
    match kernel_arguments(name)?.get(position) {
        Some(crate::artifact::Argument::Buffer { name }) => Ok(name.clone()),
        _ => Err("Continuation must bind a whole mutable buffer".into()),
    }
}
pub(crate) fn kernel_arguments(name: &str) -> Result<Vec<crate::artifact::Argument>> {
    let binding = generated::BINDINGS
        .iter()
        .find(|b| b.name == name)
        .ok_or_else(|| format!("Missing native continuation binding {name}"))?;
    arguments(binding.arguments)
}
fn check(catalog: &Catalog) -> Result<()> {
    use sha2::{Digest, Sha256};
    catalog.validate()?;
    let mut value = serde_json::to_value(catalog).map_err(|e| e.to_string())?;
    value
        .as_object_mut()
        .ok_or("Invalid catalog object")?
        .remove("binding_signature");
    let raw = serde_json::to_vec(&value).map_err(|e| e.to_string())?;
    if hex::encode(Sha256::digest(raw)) != generated::CATALOG_ID {
        return Err("Compiled catalog identity differs from native role registry".into());
    }
    if catalog.binding_signature != generated::SIGNATURE || generated::BINDINGS.is_empty() {
        return Err("Kernel catalog does not match this native binding registry; rebuild the execution package".into());
    }
    Ok(())
}
pub fn export() -> Result<Vec<KernelBinding>> {
    generated::BINDINGS
        .iter()
        .map(|b| {
            Ok(KernelBinding {
                name: b.name.into(),
                definition: b.definition,
                grid: b.grid,
                args: arguments(b.arguments)?,
                dynamic: b.dynamic,
                plan_contract: b.plan_contract,
            })
        })
        .collect()
}
pub fn dynamic(
    catalog: &Catalog,
) -> Result<
    std::collections::BTreeMap<String, orinfer_model_sdk::operators::dynamic::DynamicBatchKernel>,
> {
    check(catalog)?;
    generated::BINDINGS
        .iter()
        .filter_map(|b| b.dynamic.map(|i| (b.name, i)))
        .map(|(name, i)| {
            Ok((
                name.into(),
                catalog
                    .dynamic
                    .get(i)
                    .ok_or("Unknown dynamic launch definition")?
                    .bind(name),
            ))
        })
        .collect()
}
