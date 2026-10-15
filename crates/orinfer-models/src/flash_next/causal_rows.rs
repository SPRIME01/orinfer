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
    if !present(m) {
        return;
    }
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
    for (old, family, widths) in [
        ("flash_segment", "target", vec![1, 16, 128, 512]),
        ("flash_verify_segment", "verify", (2..=8).collect()),
    ] {
        for n in widths {
            for layer in 0..48 {
                let section = format!("layer{layer}/private");
                m.programs.insert(
                    format!("{old}_m{n}/{section}"),
                    m.programs[&format!("causal_rows/{family}/{section}")].clone(),
                );
            }
            m.programs.insert(
                format!("{old}_m{n}/layer1/ple"),
                m.programs[&format!("causal_rows/{family}/layer1/ple")].clone(),
            );
        }
    }
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
    for n in [1, 16, 128, 512] {
        m.programs.insert(
            format!("flash_mixed_capture_m{n}"),
            m.programs["causal_rows/target/capture"].clone(),
        );
    }
    for n in 2..=8 {
        m.programs.insert(
            format!("flash_verify_capture_m{n}"),
            m.programs["causal_rows/verify/capture"].clone(),
        );
    }
}

pub(super) fn configure(m: &mut Manifest) -> Result<()> {
    if !present(m) {
        return Ok(());
    }
    let mut entries = Vec::new();
    for name in m.programs.keys() {
        if name.starts_with("causal_rows/")
            || name.starts_with("flash_segment_m")
            || name.starts_with("flash_verify_segment_m")
            || name.starts_with("flash_mixed_capture_m")
            || name.starts_with("flash_verify_capture_m")
            || name.starts_with("restore_m")
        {
            entries.push((
                name.clone(),
                if name.contains("/verify/")
                    || name.contains("/restore/")
                    || name.starts_with("restore_m")
                    || name.starts_with("flash_verify")
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
