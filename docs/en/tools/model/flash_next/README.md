# Flash Next offline build

[简体中文](../../../../../tools/model/flash_next/README.md) | **English**

Rust executes online inference. TileLang is used only for offline compilation. Deployment directories contain Hugging Face configuration/tokenizer assets, `cache/model.json`, standard safetensors weights, and a digest-pinned execution package. Data format and C ABI are both version 1.

## Rebuilding a current native model

To retune integer projections and publish a matching execution package for a model already in the current native format:

```bash
make build
bash tools/operators/run.sh tools/model/flash_next/build.py artifacts/flash-build \
  --model /path/to/native-model --model-output artifacts/models/flash-ready
```

Input **must** include a matching execution package defining architecture, request state, programs, and binding contracts. This is not a general safetensors importer and cannot deploy intermediate quantized weights. Existing vision, MTP, context-capacity, and request-state semantics are preserved from the input; features are not inferred from directory names.

The builder derives Q4, W8, and fused SwiGLU arguments directly from bound weight shapes/dtypes and host ABI—not by parsing emitted CUDA. Identical matrix geometries share compiled assets. A deterministic sample of real weights and scales evaluates candidates. Exact row-count specializations cover 1/2/3/4/8/16/32; other sizes use dynamic row arguments. Dynamic decode candidates are measured over several real row counts for each geometry. Declared prefill programs are tuned at 512/2048/4096 rows by default (`--prefill-tuning-rows` overrides this). Implementations must handle small row counts, unaligned tails, full capacity, exact integer dot products, scaling/rounding boundaries, and CUDA Graph replay with changed inputs.

Experts' routing, up projection, and down projection retain the **jointly validated** selector table from the input; individual tiles must not be replaced independently. Real expert banks are checked against full E/N/K geometry, both legal row tiles, short tails, and maximum capacity. HC, QSA, vision, and causal-state implementations retain their fixed identities when rebuilding integer projections.

Build cache defaults to `artifacts/build-cache/flash-next` (`--cache` overrides it). Keys include construction arguments, actual weight identity, source, compiler, CUDA, and device. Invalid content is rebuilt. Each cache item publishes atomically, and the model publishes only after Rust plan verification. Resume an interrupted rebuild:

```bash
bash tools/operators/run.sh --resume tools/model/flash_next/build.py artifacts/flash-build \
  --model /path/to/native-model --model-output artifacts/models/flash-ready
```

For an already accepted native package, `--prefill-only` restricts rebuilding to declared prefill projections while retaining other verified operators and expert combinations. This affects only offline build scope; runtime continues using dynamic row templates.

Inspect construction contracts without a GPU:

```bash
PYTHONPATH=. .venv/bin/python -m tools.model.flash_next.build \
  --model /path/to/native-model --model-output artifacts/models/flash-ready \
  --output artifacts/flash-plan --plan-only
```

Build records live in the selected output directory and are not part of the online model protocol. Final packages merge/deduplicate assets and contain only needed cubins. After rebuilding, rerun real-request accuracy, complete-state, and performance acceptance; local operator checks are not sufficient.

## Computation and state contracts

Experts use custom integer E8P Q2A8. GDN QKV/Z, QSA Q/gate, and the shared target/MTP output head use signed-block128 rotation, per-output-row scales, and codebook Q4A8. GDN/QSA output projections, shared FFN, and draft Q/O use W8; other sensitive paths retain mixed precision. Full A8 bit-plane decomposition uses **two INT4 Tensor Core dot products** to reconstruct the INT32 result; it is not A4 quantization. Weights are kept in one compressed bank.

Prefill templates accept 1–4096 rows and chunk longer prompts. Ordinary decode, mixed batches, and MTP verification share target compute with up to 1024 rows; draft/warm share up to 512. Row count selects local launches, not duplicated model workflows or padded fake requests. Local choices are measured offline, not hard-coded by scenario. Auxiliary kernels use dynamic templates; HC lossless exponents and QSA radix shifts use scalar ABI arguments where geometry is unchanged.

QSA KV is group-64 INT8 and recurrent GDN state is FP32. KV, pending buffers, compressed indices, convolution/PLE histories, positions, and MTP hidden rings are request-private. Compare all state across chunking, restore, capture, and commit. Mixed execution packets explicitly carry prefill/verification roles; role boundaries contribute to Graph identity.

## Deployment

```bash
./target/release/orinfer serve artifacts/models/flash-ready \
  --model qwen-flash-next --listen 0.0.0.0:8088 \
  --cuda-graph decode_only --mtp-drafts 3 --prefix-cache-mib 512
```

MTP supports 1–7 drafts; `0` disables it and `auto` uses the package default. Stochastic sampling uses p/q acceptance and residual correction. Short tails and scheduler time budgets may narrow verification width. Prefix restoration preserves complete target/draft state. History-based continuation drafts require explicit `--prompt-lookup`.

See [Serving](../../../serving.md) for requests, deployment defaults, and images. Vision capacity is declared by the input model; text-only rebuilds do not expand vision or context capacity.

## Original weights and reference tools

`make python-env` creates the pinned CPU `uv` environment. GPU tools share the compiler image and GPU lock. Reference BF16 weights are pinned to a revision and read matrix-by-matrix or expert-by-expert without downloading a full model:

```bash
bash tools/quantization/run_flash_next.sh \
  --index /path/to/original/model.safetensors.index.json \
  --revision SOURCE_COMMIT_SHA --stage all \
  --output artifacts/quantization/flash-next/converted

PYTHONPATH=. .venv/bin/python -m tools.model.flash_next.weights \
  --converted artifacts/quantization/flash-next/converted \
  --index /path/to/original/model.safetensors.index.json \
  --config /path/to/original/config.json --frontend /path/to/original \
  --output artifacts/models/flash-next-quantized
```

These produce **intermediate quantized weights**, not a deployable native package. `native.py`, `mtp.py`, and `reference/` contain offline mathematical references. Production package rebuilds use `build.py` with host ABI binding and export supplied by `compiler.py`. Legacy fixed-batch/per-length plan migration has been removed; historical intermediate packages cannot be converted directly into deployment packages.

Numerical acceptance uses seed 20261002, compares top-three probabilities, KL/NLL, and task results along the same teacher-forced histories, and does not automatically fail on different token choices. A partial-layer reference is not a whole-model BF16 baseline. `checks/` covers kernels, tails, and Graph replay; Rust GPU tests cover state, prefix, MTP, and mixed requests. Measure performance with full Rust CLI requests; do not extrapolate TPS from microbenchmarks.

Source responsibilities: `construction.py`/`build.py` own production rebuilding; `decode_tuning.py` selects integer candidates; `weights.py`/`checkpoint.py` read offline weights; `native.py`/`mtp.py` retain mathematical references. Shared operators live under `kernels/model/`, and the online adapter is under `crates/orinfer-models/src/flash_next/`.
