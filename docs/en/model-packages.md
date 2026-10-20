# Model execution packages

[简体中文](../model-packages.md) | **English**

The online engine loads model-specific execution logic through a **versioned C ABI**. Model adapters, compute policies, shared runtime components, kernels, and build tools remain in one source workspace. Deployment packages contain a native `.so`, matching cubins, and configuration/layout contracts; the model's weights stay in its model directory. An execution package does not require a separate source repository.

Labels such as A8/A4 describe compute policies, **not uniform precision across every operator**. Packages must declare their actual mixed-precision behavior.

The engine core owns the API, scheduler, CUDA context and streams, memory, CUDA Graphs, and prefix indexing. Packages own configuration checks, model execution order, fusion, dynamic batches, state bindings, and vision positions. Adding a model family does not require core architecture enums or dispatch branches. The current native library is `orinfer-models`, registering `int8_quality` for Qwen3.5 and Flash Next.

```text
MODEL_DIR/
  config.json, tokenizer.json, chat_template.jinja, ...
  cache/model.json
  cache/weights/*.safetensors
  cache/cpu/                          # Optional CPU lookup tables
  cache/packages/<sha256>/
    package.json
    lib/model.so
    lib/bindings.rs, lib/BUILD.txt    # Generated binding sources and rebuild instructions
    assets.safetensors               # Or individual cubin/source/host ABI files
    LICENSE
```

Schema-1 `cache/model.json` pins the entire `execution_package` by digest. The shared cache defaults to `~/.cache/orinfer/packages` and can be overridden with `ORINFER_EXECUTION_CACHE`. Package schema 1 / runtime ABI 1 describe SM87, architecture and compute policy, supported configuration, buffers, kernel ABI, and `.so` identity. The package's `describe` endpoint reports identity and capabilities, which the runtime cross-checks against the manifest. Content digests govern package identity; source-workspace organization and ABI compatibility are separate concerns. Incompatible protocol changes require a new ABI version.

## Native ABI and execution bindings

The `.so` depends on `orinfer-model-sdk`, not `orinfer-engine`. Its `orinfer_model_v1` export returns a C function table for describe/create/batch/visual operations and corresponding resource release functions. See [orinfer_model.h](../../crates/orinfer-model-sdk/include/orinfer_model.h). Rust structs and traits never cross the dynamic-library boundary.

The package `catalog` stores shared assets, compiled implementation definitions, and dynamic-row launch templates. Definitions reference cubin/source/host ABI indices, symbols, blocks, shared memory, cooperative flags, ordered argument types, and read-only contracts. They do **not** store layer numbers, weight names, per-layer arguments, grids, or full execution programs. One cubin may expose distinct ABI/launch contracts; do not deduplicate definitions by binary hash alone.

Offline-generated mappings are compiled into Rust tensor-role tables, shared call templates, per-layer argument tables, and deduplicated specialization selectors inside the `.so`. These describe implementations and storage roles; execution order and fusion are produced by the registered Rust model adapter. Catalog and compiled bindings have separate identities. Changes to implementations, argument types, offsets, or dynamic launches require rebuilding a matching native library. Runtime validation checks catalog identity, buffer dtype/shape/layout, and bounds, without Python or JIT compilation.

Layers with equivalent behavior share call templates and specialization tables. Each layer supplies only its layer index, weights, state, workspace roles, and required scalars. Instantiation preserves exact argument order, slice offsets, scalar bit patterns, grid, and dynamic-launch contracts; computation and precision are never inferred from weight names. Continuation-argument lookup queries templates directly instead of expanding an entire registry. Publication checks exact round-trips for each instance, for both model adapters.

Flash Next's HC lossless-cache exponent and QSA radix shift use integer scalar arguments under the existing ABI. Request-private QSA reuses its state wrapper. QSA stripe counts are selected offline. Adjacent row-count ranges merge only when every argument, asset, and launch contract matches. Static and dynamic integer projection sizes use one template and candidate space; dynamic fallbacks are measured at several real row counts, invalidated when source/compiler/device identity changes. Unselected candidates are excluded from releases; there is no online tuning.

Normal online creation exports only final programs and bindings referenced by local specializations and the expert-residency contract. Prefusion build inputs are not exposed as executable launches. `plan-model` and `validate-model` use an internal `build_bindings` option to retain the complete build contract for offline rebuilding, asset checks, and diagnostics. This changes neither schema 1 nor C ABI 1. Runtime per-layer nodes still need instantiation; CUDA Graph replay does not interpret templates.

Creation passes configuration, data contracts, and a compact compilation catalog as JSON. A separate `orinfer_model_plans_v1` export uses C arrays for generated kernel-argument bindings and kernel/copy/zero programs, leaving the primary `orinfer_model_v1` table unchanged. Table and record sizes are explicit; arrays, strings, and nested arguments share an owner and are released together. Per-layer bindings and complete programs do not make a JSON serialization round-trip.

The dynamic batch hot path also uses C arrays, returning instructions, buffer views, dynamic-launch arguments, and request-state bindings together. Allocators release their own native memory; the library remains loaded until model and CUDA resources have been destroyed. Model instances are bound to the GPU execution thread.

### Expert residency and CPU input assets

Limited expert residency is selected through creation-time `expert_contracts`, yielding more complete native bindings. The model library returns `expert_banks` describing counts, expert-contiguous tensors, and I32 routing-count argument positions for consuming kernels. The core reads these counts through actual bindings and manages global SLRU and CUDA VMM pages without knowing architecture-specific names or dimensions. Full residency builds only the verification contracts needed by the model plan; the runtime does not read per-layer `kernel_contracts` from package JSON.

Models may declare hashed `input_assets`. The optional `orinfer_model_inputs_v1` entry point accepts a program name, current tokens, and request history, then returns upload data for named buffers while retaining ABI v1. The core validates access rights and upload bounds, and finishes uploads before the library frees data.

Flash Next performs CPU W8 embedding and E8P PLE lookup inside the package. A bounded row cache is shared; n-gram histories are supplied by requests. CPU assets survive model cloning and execution-package updates. The package distinguishes base-model, verification, and shifted-draft inputs; the core does not register model-specific input dispatch.

MTP verification plans may declare compact recurrent updates that are always committed, and optional save/restore programs for draft branches. The scheduler saves before drafting and restores before refreshing the accepted prefix. Flash Next uses these general interfaces to preserve GDN, convolution, PLE, HC, and QSA pending/index state.

Packages do not create hidden CUDA contexts or streams, or manage engine GPU memory directly. The core executes explicit programs and does not rebuild plans across the package boundary on a Graph hit. The state-binding cache holds at most 16 slot/segment-length combinations and only caches logical names; GPU addresses are resolved from the current request arena. Prefix identities include the package digest so old state snapshots are never reused across different compute policies or layouts.

## Building and updating

The unified source tree contains:

- `crates/orinfer-engine`: generic runtime, with no model registration.
- `crates/orinfer-model-sdk`: plan, state, and C ABI contracts.
- `crates/orinfer-models`: model adapters, independently registered precision policies, and registry.
- `kernels/`: reusable TileLang kernels and fusions.
- `tools/`: offline compilation, weight preparation, assembly, and validation.

Adapters express architecture semantics and computation composition; precision policies select available precision paths. A policy may serve multiple adapters and fusion may cross component boundaries. Only verified architecture/policy combinations are registered; unimplemented modes do not receive placeholder fallbacks. Packages bind only assets actually required by the model configuration.

```bash
make build
# Or build the native model library alone:
cargo build --release --locked --offline -p orinfer-models
```

That library supplies registered model logic. A deployable library also needs generated binding modules for the current kernel ABI and physical tensor roles. The publisher uses `ORINFER_BINDINGS_SOURCE` to compile the module. The native library depends on the SDK, not the engine. The package retains `lib/model.so` plus generated `lib/bindings.rs` and `lib/BUILD.txt` so the source workspace and `Cargo.lock` can reproduce it. The latter files do not execute online.

To rebuild the native library and catalog while preserving weights and cubins:

```bash
PYTHONPATH=. .venv/bin/python -m tools.model.catalog \
  --model /path/to/prepared-model --output /path/to/rebuilt-model
```

This requires the project's pinned Rust toolchain and cached Cargo dependencies; it builds with `--locked --offline`. Immutable weights are hardlinked when on the same filesystem; model descriptors and `.so` are separate files. The output is loader-verified and published atomically without modifying the source. Normal deployment needs no compiler.

`tools.model.attach_execution` can replace the native library **only when the binding registry is identical**. Ordinary workspace libraries without generated bindings, or libraries with different catalog identities, are rejected. Rebuild via `tools.model.catalog` when registered model logic changes.

Both model data and packages remain schema 1. The loader accepts current catalogs with native binding exports, not older per-layer `kernels` packages or a schema-2 compatibility path. Offline build inputs may still hold expanded bindings for exact comparisons and generated-Rust compilation.

## Installing execution packages

After downloading the engine, matching execution archive, and `SHA256SUMS`, verify the downloads. Installation requires no Python, CUDA initialization, or Rust toolchain:

```bash
sha256sum --ignore-missing -c SHA256SUMS
orinfer package install /path/to/package.tar.xz
orinfer plan-model /path/to/new-model
orinfer validate-model /path/to/new-model
```

The installer detects `.tar.xz` and legacy `.tar.gz` by file content. It validates the digest, schema/ABI, SM87 target, aarch64 ELF library, and SHA256 for all loose or safetensors-embedded kernel assets **without running native package code**. Path traversal, symlinks, and duplicate archive entries are rejected. Successful verification is followed by atomic cache installation; failures clean temporary directories. An installed digest is never overwritten. Extraction and full package checks happen only at installation, not every startup.

Cache selection follows `ORINFER_EXECUTION_CACHE`, then `$XDG_CACHE_HOME/orinfer/packages`, then `~/.cache/orinfer/packages`. `--cache /path/to/packages` changes the install destination for this invocation only. Set `ORINFER_EXECUTION_CACHE` for subsequent runtime use of a custom cache.

`plan-model` loads the native library on CPU and generates its plan; `validate-model` also verifies model weights, CPU tables, and every kernel file. Neither initializes the GPU. Normal online startup checks large safetensors weights and CPU tables structurally; `--verify-weights` enables full content verification. Frontend files, CPU asset metadata, native library, kernel hashes, and configuration signatures remain checked. The optional JSON creation field `verify_weights` controls model-library checks of CPU tables without changing schema 1 or ABI v1; updated native libraries can enable the new behavior.

### Installing released v0.1.3 packages

The v0.1.3 binary predates `package install`. After downloading and verifying its assets with `SHA256SUMS`, extract the execution package's digest directory into the shared cache using system tar:

```bash
export ORINFER_EXECUTION_CACHE="$HOME/.cache/orinfer/packages"
mkdir -p "$ORINFER_EXECUTION_CACHE"
tar -xf /path/to/package.tar.gz -C "$ORINFER_EXECUTION_CACHE"
orinfer validate-model /path/to/prepared-model
```

### Creating an offline archive

Model builders, not ordinary deployment users, create `.tar.xz` execution packages with Python:

```bash
PYTHONPATH=. .venv/bin/python tools/model/package.py archive /path/to/package /path/to/package.tar.xz
```

## Compact release layouts

After offline tuning, GPU weights are packed into standard safetensors shards of approximately 2 GiB. CPU lookup data is packed into contiguous encoded segments. Tensor names, physical layouts, quantization bits, and CPU row ordering are unchanged, while indices, CPU offsets, and hashes are refreshed. The model directory holds only model data; its matching execution package is written to a separate cache:

```bash
PYTHONPATH=. .venv/bin/python tools/model/compact.py \
  --model /path/to/prepared-model --output /path/to/compact-model \
  --package-cache ~/.cache/orinfer/packages
orinfer validate-model /path/to/compact-model
```

Flash Next publication uses stable component and layer-group names: main-model experts in three-layer shards, other backbone weights in eight-layer shards, and Q4 projections, output head, losslessly compressed HC, MTP, and vision in separate components. CPU tables reuse merged contiguous-row shards. No shard exceeds 2 GiB by default; oversized components fail publication. Repacking retains each tensor's bytes, layout, and identity, without requantization or kernel recompilation:

```bash
PYTHONPATH=. .venv/bin/python -m tools.model.flash_next.repack \
  --model /path/to/optimized-model --output /path/to/published-model
orinfer validate-model /path/to/published-model
```

Published model outputs exclude `cache/packages`. Hugging Face and ModelScope repositories distribute configurations, tokenizers, model descriptors, GPU weights, and CPU tables. Matching execution packages are released independently on GitHub, installed into the shared cache first, and located by `execution_package` digest. Repacking does not change that digest.

A package's `assets.safetensors` stores cubins, generated source, and host ABI as named U8 tensors, deduplicating byte-identical assets. A schema-1 file identity may include a `tensor` field; SHA256 checks that tensor's raw bytes. The native `.so` stays a separate file. The engine reuses bundle mmap and the header index to read named assets without unpacking thousands of tiny files; CUDA modules load individually. This layout requires an engine version supporting named assets.

Offline tuners obtain the current bindings through `plan-model`, expand and modify them for compilation, and regenerate the native module and catalog afterward. Online format remains unchanged.

## Validation

`make check` runs formatting, Clippy, and unit tests for the engine, SDK, and model crates, including cross-language C ABI loading, invalid-input rejection, and atomic-update behavior. Package updates should also compare complete programs, launch arguments, buffers, and state contracts. GPU checks require matching model weights and cover request state, changed-input Graph replay, prefix restoration, and MTP branches.

The generic loading-path GPU fixture can run independently:

```bash
bash tools/operators/run.sh tools/model/validate_model_package_gpu.py artifacts/model-package/check01
```

It covers all three Graph modes, changed inputs, and complete logits, but does not establish production-model quality or throughput.

Current native Flash Next packages can be rebuilt with `tools/model/flash_next/build.py`, retuning integer projections against real matrices with content checks and resume support. The rebuild preserves verified state, floating-point reductions, and joint expert combinations from the input package. It requires an existing compatible execution package; intermediate quantized weights are not deployable models. See [Flash Next builds](tools/model/flash_next/README.md).
