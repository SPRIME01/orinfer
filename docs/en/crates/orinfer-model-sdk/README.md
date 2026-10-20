# orinfer-model-sdk

[简体中文](../../../../crates/orinfer-model-sdk/README.md) | **English**

Shared data contracts and version 1 of the C ABI for standalone model execution packages. A package depends only on this SDK, not the main engine. Rust implementations can export a `cdylib`; implementations in other languages can use the [C header](../../../../crates/orinfer-model-sdk/include/orinfer_model.h).

Configuration and plan descriptors are serialized at construction time; batches are transferred as C arrays. The allocator that owns memory must free it, and the native library must remain loaded until the associated objects are destroyed. SDK Rust traits are for implementation within a package only; they are not passed across the dynamic-library boundary.

See [Model execution packages](../../model-packages.md) for protocol and resource ownership. The ABI version and crate version are managed separately: changes to cross-library structs or function tables require an ABI version change, not merely additional fields.
