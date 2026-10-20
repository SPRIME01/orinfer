# Images and multi-image inference

[简体中文](../../../../tools/vision/README.md) | **English**

The 27B and Flash Next models share the ViT implementation in `kernels/vision/encoder.py`. Each text architecture declares its own merger output dimension, feature injection, and MRoPE state. The [27B native builder](../model/qwen35/README.md) creates a vision-enabled model, while the [Flash Next native rebuild](../model/flash_next/README.md) preserves the input model's vision contract.

Rust handles online image preprocessing, position generation, request-private feature tables, and execution. Vision uses one dynamic patch template for single images and ordered multi-image inputs. **Video is not supported.** The default ceiling is 32,768 patches (8,192 visual tokens), subject to context and per-request memory budgets. See [Serving](../../serving.md) for request examples.

## Kernel and reference checks

Use the pinned compiler image and shared GPU lock. Output paths must not exist beforehand:

```bash
bash tools/operators/run.sh tools/vision/validate_kernels.py artifacts/vision/kernels-check
bash tools/operators/run.sh tools/vision/validate_mrope.py artifacts/vision/mrope-check
```

`validate_mrope.py` checks query/KV/indexer rotation, unaligned compressed blocks, and CUDA Graph replay with changed feature inputs. `official_reference.py` uses the checkpoint's native Transformers vision model to export independent BF16 or FP16 features; always record the reference dtype.

`preprocess_reference.py` exports preprocessing fixtures for the current model, which are then checked by an ignored Rust test:

```bash
bash tools/operators/run.sh tools/vision/preprocess_reference.py artifacts/vision/preprocessor-check \
  --model /path/to/prepared-model
ORINFER_IMAGE_REFERENCE="$PWD/artifacts/vision/preprocessor-check" \
  cargo test --offline checkpoint_image_processor_reference -- --ignored --nocapture
```

The Rust `make check-gpu` acceptance suite checks full state, chunking, prefix restore, and isolation of text following images.

## API acceptance

Against a prepared, running server:

```bash
python3 tools/vision/smoke.py --output artifacts/vision/smoke.json
python3 tools/vision/scenarios.py --output artifacts/vision/scenarios.json
python3 tools/vision/limits.py --manifest /path/to/prepared-model \
  --output artifacts/vision/limits.json
```

All tools accept `--base-url` and `--model`. Smoke tests cover one image, ordered multi-image input, SSE, and state isolation. Scenarios cover counting, multimodal history, function calling, and queuing. Capacity tests verify limit rejection, recovery after errors, and concurrent image queuing against the default 32,768-patch limit. Scenarios use seed 20261002 and record answers, usage, finish reasons, and latency. Task quality is assessed separately against the original BF16/FP8 model.
