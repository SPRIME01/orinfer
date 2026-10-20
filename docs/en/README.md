# Orinfer

[简体中文](../../README.md) | **English**

LLM and vision-language inference on NVIDIA Jetson, powered by Rust and TileLang.

Orinfer is an inference engine targeting **Jetson AGX Orin 64GB / CUDA SM87**. Rust handles online execution and serving; TileLang operators are compiled offline into standalone GPU execution packages. It supports text, single-image, and multi-image inference for Qwen3.8-27B and Qwen3.5-4B (`Qwen3_5`), and Qwen3.8-Flash-Next (`qwen4_exp`). Development builds also include Index-Translate-2B/9B adapters. Image input is wired up for translation, although translation quality on complex images still needs improvement.

## Features

- OpenAI-compatible Chat Completions and native Ollama chat/generate APIs with streaming, thinking, sampling, strict function calling, JSON Schema constraints, and token log probabilities.
- [Offline JSONL inference](batch.md) with continuous batching, incremental result persistence, and resumable jobs—no HTTP server required.
- Two-stage E8P Q4/W8 mixed weights for 27B, 4B, and Index-Translate-2B/9B models; custom E8P Q2 experts and mixed Q4/W8 projections for Flash Next. Computation uses INT8 mixed precision, retaining BF16/FP16/FP32 on sensitive paths.
- Chunked prefill, isolated per-request state, prefix caching, native MTP, and continuous batching, with admission queuing when memory is insufficient. For Flash Next, `--mtp-drafts 1..7` selects the maximum draft count; `0` disables MTP, while `auto` dynamically selects draft depth using concurrency, context, measured execution cost, and per-depth acceptance, subject to package limits.
- Prefix caching for mixed architectures, INT8 KV, and demand-mapped memory. Model packages can be configured for contexts up to 256K tokens.
- CUDA Graph modes: `decode_only` (default), `full`, and `off`.

## Quick start

Deployment requires **aarch64 Linux and an NVIDIA CUDA driver**. It does **not** require Python, PyTorch, TileLang, or the Rust toolchain.

Download an engine release archive, the matching model execution package, and `SHA256SUMS` from the [upstream releases](https://github.com/iMMIQ/orinfer/releases). Verify the downloads, unpack the engine archive, and add `bin/orinfer` to your `PATH`. Download the prepared model directory separately; it contains the model configuration, tokenizer, chat template, and native quantized safetensors. The `execution_package` digest in the model descriptor must match the installed execution package.

**Release archives do not include model weights. Orinfer cannot directly load arbitrary Hugging Face, AWQ, or GGUF directories.**

Starting with v0.1.4, release archives use `.tar.xz`, and execution packages are installed with `package install` (which also accepts legacy `.tar.gz` packages). See [legacy v0.1.3 installation instructions](model-packages.md#installing-released-v013-packages) if using an older release.

```bash
sha256sum --ignore-missing -c SHA256SUMS
orinfer package install /path/to/matching-execution.tar.xz
orinfer validate-model /path/to/prepared-model
orinfer serve /path/to/prepared-model --model qwen3.8-27b
```

The server listens on `0.0.0.0:8088` by default. Set `ORINFER_API_KEY` to require Bearer authentication for `/v1` routes.

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Introduce yourself in one sentence."}],"max_tokens":512,"stream":true}'
```

Endpoints include `/health`, `/v1/models`, and `/v1/chat/completions`, plus native Ollama endpoints such as `/api/chat` and `/api/generate`. Chat supports streamed text, tools, thinking, constrained JSON, and single/multiple images. The [serving guide](serving.md) explains request parameters and provides configurations for [OpenCode](../../examples/opencode.json) and [Pi](../../examples/pi-models.json).

## Common configuration

| Option | Default | Purpose |
| --- | --- | --- |
| `--listen` | `0.0.0.0:8088` | Listen address |
| `--default-request-params` | `{}` | Default JSON request parameters; explicit request fields take precedence |
| `--cuda-graph` | `decode_only` | CUDA Graph mode |
| `--resident-experts` | `full` | Keep all experts resident; for Flash Next, specify an equivalent per-layer capacity and stream other experts from SSD |
| `--prefix-cache-mib` | `12288` | GPU prefix-cache budget; `0` disables it |
| `--max-active-requests` | `32` | Active request limit (1–128) |
| `--max-batch-tokens` | `128` | Decode/verification row budget; mixed prefill can use package capacity |
| `--max-speculative-tokens` | `0` | Total MTP verification-row budget; `0` uses the execution package limit |
| `--prefill-budget-ms` | `200` | Predicted-time target for mixed prefill |

Actual concurrency and context capacity depend on the execution package and available memory. Environment variables use the `ORINFER_*` prefix. Shared execution packages are installed under `~/.cache/orinfer/packages` by default; override this with `ORINFER_EXECUTION_CACHE`. See the [serving guide](serving.md) and `orinfer serve --help` for the complete configuration.

## Performance and scope

On a **Jetson AGX Orin 64GB**, using the current development build and a matching optimized Flash Next execution package, execution-layer prefill on a 512-token input measured **1,008.9 tokens/s**. In the streaming Rust CLI Chat API, generating Rust code from scratch with `medium` thinking and a 16K total output budget yielded 6.2K–7.6K tokens. Single-stream decode measured **29.49–29.72 tokens/s** with MTP disabled and **44.99–45.82 tokens/s** with up to three drafts: a **1.51–1.55×** speedup.

The [performance reference](performance.md) includes detailed prefill, API time-to-first-token, code-generation throughput, prefix-cache, and 27B results. These measurements apply to rebuilt execution packages; the test configurations, warmup counts, and timing definitions are documented there.

**Index-Translate-2B:** quantized weights occupy approximately **1.23 GB / 4.32 bits per weight**. In development-build Rust CLI fixed-length measurements, prefill at 512/2K/8K tokens was approximately **7.6K/7.9K/6.5K tokens/s**; decode with MTP off was approximately **108/105/94 tokens/s**. The ordinary request path measured approximately **113/109/98 tokens/s**. The goal of 150 decode tokens/s without MTP has **not** been reached. See the [2B measurement details](performance.md#index-translate-2b-2026-10-18) for methodology, text quality spot checks, and image-quality limitations.

**Index-Translate-9B:** quantized weights occupy approximately **5.44 GB / 4.51 bits per weight**. This adapter reuses the execution components but retains separate embedding and output heads. Rust CLI fixed-length prefill at 512/2K/8K tokens measured approximately **2,153/2,220/1,956 tokens/s**; decode with MTP off measured approximately **33.4/32.6/29.8 tokens/s**. See the [9B measurements](performance.md#index-translate-9b-2026-10-18) and BF16 spot comparisons.

**Qwen3.5-4B:** quantized weights are approximately **2.58 GB / 4.42 bits per weight**. Fixed-length Rust CLI prefill at 512/2K/8K tokens measured **3,279/3,401/2,798 tokens/s**, with MTP-off decode at **54.45/52.34/45.68 tokens/s**. See the [4B measurements](performance.md#qwen35-4b-2026-10-19).

The engine currently targets only the hardware and architectures described above. Video, audio, `n > 1`, and cloud-storage APIs are not supported. The [serving guide](serving.md) describes the supported JSON Schema subset. Quantization quality is checked against fixed scenarios and matched-history probabilities from same-source BF16/FP8 references; comprehensive LLM benchmarks and an independent BF16 256K-context quality evaluation remain outstanding. See [quality evaluation](tools/eval/README.md).

## Development and documentation

```bash
git clone https://github.com/SPRIME01/orinfer.git
cd orinfer
cargo fetch --locked
make build       # Build Rust only; toolchain pinned in rust-toolchain.toml
make python-env  # Create/sync the CPU .venv with uv
make check       # Formatting, compilation, Clippy, Rust and CPU tests
```

The offline Python environment is necessary only to quantize weights, compile TileLang operators, or rebuild models. Deploying released, prepared models and matching execution packages does not require it. The `.so + cubin` execution-package format is described in [execution-package contracts and builds](model-packages.md).

- [Build and release](tools/build/README.md): compiler image, CPU/GPU checks, and release artifacts.
- [Models and execution packages](tools/model/README.md): offline preparation, layout, MTP, and context extension.
- [Serving](serving.md): API, images, caching, concurrency, and resource controls.
- [Vision builds](tools/vision/README.md), [API validation](tools/api/README.md), and [TileLang kernels](kernels/README.md).

Online Rust code lives in `crates/orinfer-{engine,api,cli}`; model adapters and compute policies are in `crates/orinfer-models`, and the shared ABI is in `crates/orinfer-model-sdk`. GPU operators and offline tools live in `kernels/` and `tools/`, respectively.

## License

**LGPL-3.0-or-later**; see [LICENSE](../../LICENSE) and [COPYING](../../COPYING). Third-party dependencies and model weights retain their respective licenses.
