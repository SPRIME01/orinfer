//! Length is a launch scalar; semantic state slots have one compiled binding.
use crate::{
    artifact::Result,
    model::{Manifest, Operation},
};

pub(super) fn present(m: &Manifest) -> bool {
    m.dynamic_batch_kernels
        .contains_key("causal_rows/verify/layer0/private/k0")
}

pub(super) fn register(m: &mut Manifest) {
    for family in ["target", "verify", "restore"] {
        for layer in 0..48 {
            let section = if family == "restore" {
                format!("layer{layer}")
            } else {
                format!("layer{layer}/private")
            };
            let count = if family == "restore" {
                if layer == 1 {
                    3
                } else if layer % 4 == 3 {
                    1
                } else {
                    2
                }
            } else if family == "verify" {
                if layer % 4 == 3 { 20 } else { 4 }
            } else if layer % 4 == 3 {
                19
            } else {
                2
            };
            let name = format!("causal_rows/{family}/{section}");
            m.programs.insert(
                name.clone(),
                super::plan::section(&format!("causal_rows/{family}"), &section, count, None),
            );
        }
        if family != "restore" {
            let prefix = format!("causal_rows/{family}");
            m.programs.insert(
                format!("{prefix}/layer1/ple"),
                super::plan::section(
                    &prefix,
                    "layer1/ple",
                    if family == "verify" { 3 } else { 2 },
                    None,
                ),
            );
        }
    }
    m.programs.insert(
        "causal_rows/draft".into(),
        super::plan::section("causal_rows", "draft", 19, None),
    );
    for (name, kernel) in [
        ("causal_rows/verify/capture", "causal_rows/verify/capture"),
        ("causal_rows/target/capture", "causal_rows/target/capture"),
    ] {
        m.programs.insert(
            name.into(),
            vec![Operation::Kernel {
                name: kernel.into(),
            }],
        );
    }
}

pub(super) fn configure(m: &mut Manifest) -> Result<()> {
    let mut entries = Vec::new();
    for name in m.programs.keys() {
        if name.starts_with("causal_rows/") || name == "mtp_restore" {
            entries.push((
                name.clone(),
                if name.contains("/verify/") || name.contains("/restore/") || name == "mtp_restore"
                {
                    8
                } else {
                    512
                },
            ));
        }
    }
    let rows = m
        .row_execution
        .as_mut()
        .ok_or("Missing causal row execution")?;
    for (name, cap) in entries {
        rows.programs.insert(name.clone());
        rows.program_capacities.insert(name, cap);
    }
    Ok(())
}
