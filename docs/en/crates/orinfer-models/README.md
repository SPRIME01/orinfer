# orinfer-models

[简体中文](../../../../crates/orinfer-models/README.md) | **English**

Native model implementations live in the unified workspace. Model adapters and precision policies register independently and share execution components, TileLang sources, build tools, and tests. Deployment produces a `.so + cubin` execution package.

The current `int8_quality` paths cover Qwen3.8-27B (`Qwen3_5`) and Flash Next, including mixed precision, fusion, batching, MTP, and visual position semantics. The model library links only to `orinfer-model-sdk`, not `orinfer-engine`; cross-library communication uses C ABI v1 exclusively.

Flash Next accepts the current dynamic execution contract only: 1–4096 prefill rows, up to 1024 shared target rows, up to 512 draft/warm rows, and 2–8 MTP verification rows. Ordinary decode, mixed requests, and MTP reuse compute templates, with state ownership, capture, and commit handled separately. Legacy fixed-batch and per-length execution plans are no longer generated. Q4 uses a fixed mixed-precision policy. Input rotation is bound to consumer workspaces and symbol tables, and runs once before the first consumer of shared QKV/Z.

```bash
cargo build --release --locked --offline -p orinfer-models
cargo test --locked --offline -p orinfer-models
```

See [Model execution packages](../../model-packages.md) for build steps, package contracts, and validation.
