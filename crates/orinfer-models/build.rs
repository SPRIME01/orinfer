use std::{env, fs, path::PathBuf};
fn main() {
    println!("cargo:rerun-if-env-changed=ORINFER_BINDINGS_SOURCE");
    let destination = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR")).join("bindings.rs");
    if let Some(source) = env::var_os("ORINFER_BINDINGS_SOURCE") {
        println!(
            "cargo:rerun-if-changed={}",
            PathBuf::from(&source).display()
        );
        fs::copy(source, destination).expect("copy generated native bindings");
    } else {
        fs::write(destination, "pub const CATALOG_ID: &str = \"\";\npub const SIGNATURE: &str = \"\";\npub const ROLES: &[&str] = &[];\npub const ARGUMENTS: &[&[orinfer_model_sdk::catalog::RoleArgument]] = &[];\npub const BINDINGS: &[orinfer_model_sdk::catalog::RoleBinding] = &[];\n").expect("write empty binding registry");
    }
}
