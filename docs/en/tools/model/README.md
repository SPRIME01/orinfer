# Offline model builds

Python is used for weight preparation, TileLang compilation, tuning, and validation. Rust handles online model loading, execution, and serving.

## Supported build entry points

| Model | Input | Build entry point |
| --- | --- | --- |
| Qwen3.8-27B | Fully quantized, custom-codebook Q4A8 checkpoint | [qwen35/build.py](qwen35/README.md) |
| Qwen3.8-Flash-Next | Current schema-1 model and matching execution package | [flash_next/build.py](flash_next/README.md) |

The current 27B deployment uses two-stage integer E8P Q4A8. Flash Next uses E8P Q2A8 experts with quality-oriented mixed precision. See [Quantization](../quantization/README.md) for the storage and encoding contracts.

```bash
make python-env
make build
bash tools/operators/run.sh tools/model/qwen35/build.py artifacts/qwen35-build \
  --source /path/to/converted --model-output artifacts/models/qwen35-ready \
  --gpu-lock-held
bash tools/operators/run.sh tools/model/flash_next/build.py artifacts/flash-build \
  --model /path/to/native-flash-model --model-output artifacts/models/flash-ready
```

The compiler environment is pinned to `orinfer-compiler:0.1.2`. See [Build and release](../build/README.md) for the image recipe and mount rules. External inputs are mounted read-only through `ORINFER_CHECKPOINT_DIR`. All GPU work shares `artifacts/gpu-experiment.lock`. Use new output directories; published artifacts must remain immutable.

## Artifacts and shared components

Prepared model directories retain the HF configuration, tokenizer, chat template, standard safetensors shards, and `cache/model.json`. A digest binds the model descriptor to a standalone execution package containing the Rust model library, deduplicated cubins, sources, and host ABI. Both the data format and C ABI are version 1. The online engine does not accept a bare AOT manifest.

- `compile.py`, `row_contract.py`, `row_selection.py`: dynamic-row factories, host ABI contracts, and offline selection.
- `catalog.py`, `binding_templates.py`: shared compile definitions and native parameter templates.
- `publication.py`, `compact.py`: immutable asset reuse, safetensors merging, and atomic publication.
- `package.py`: offline execution-package verification, XZ packaging, internal installer, and frontend asset binding. For deployment, use Rust's `orinfer package install`.
- `attach_execution.py`: attach a matching execution package to a model data directory.

Rebuilt execution packages must be checked for program coverage, parameters, dynamic launches, state, and buffer contracts. Changed weight layouts require preparing weights again; recompiling the same computation may reuse compatible shards.

## Validation and measurement

```bash
./target/release/orinfer validate-model /path/to/prepared-model
./target/release/orinfer plan-model /path/to/prepared-model
./target/release/orinfer run-model /path/to/prepared-model examples/requests.json
make check
make check-offline
make check-gpu MODEL=/path/to/prepared-model OUTPUT=artifacts/check-gpu/new-run
```

`validate-model` fully verifies files and payloads; `plan-model` produces a CPU-only execution plan. GPU acceptance covers dynamic batching, request isolation, prefix restoration, MTP, and image input. Quantization quality is checked separately against same-model BF16/FP8 fixed-history references. See [Quality evaluation](../eval/README.md).

`run.sh` records complete Rust requests and machine state; `profile.sh` and `profile_summary.py` support Nsight profiling. Operator microbenchmarks, short-output profiles, and end-to-end request throughput should be reported separately; see [Performance](../../docs/performance.md).
