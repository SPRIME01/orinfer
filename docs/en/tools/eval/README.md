# Quality evaluation tools

[简体中文](../../../../tools/eval/README.md) | **English**

`quick_quality.py` compares baseline and candidate token probabilities over the **same teacher-forced history**. Probabilities are normalized over the full vocabulary, and the baseline's top-three token IDs are explicitly queried. Different selected token IDs are diagnostic evidence, not an automatic quality failure.

`scoring_common.py` validates fixed seeds, history identity, probability fields, and task outcomes. `run_quality.py` connects to a configured scoring endpoint and builds evaluation requests from `fixtures/quick-quality-scenarios.json`. You must supply the reference server.

```bash
python3 -m unittest discover -s tools/eval -p 'test_*.py' -v
python3 tools/eval/run_quality.py --help
```

Use official BF16 or same-model FP8 as the quantization quality reference. Comparisons against the same AWQ weights are for implementation diagnostics only. Scenario success, target NLL, and long-context tasks require separate evaluation.

For translation models, `offline_quality prepare` accepts `--fixtures fixtures/translation-quality-scenarios.json`. The fixtures cover Chinese, English, Japanese, and German translation, terminology, JSON placeholders, negation, and business tone. Reference translations define fixed probability-scoring histories; free generation may produce other correct translations. These spot checks do not constitute a comprehensive multilingual benchmark. Evaluate vision probabilities and API-level image/multi-image tasks separately.

## Offline BF16 comparison

`reference_bf16.py` uses a Transformers version supporting Qwen3_5 to load raw safetensors one layer at a time, avoiding the need to keep the full BF16 model resident on the GPU. The reference executes full-sequence prefill; `score-model` performs normal prefill and token-by-token teacher-forced decode. Reports label the two paths separately and preserve full-vocabulary-normalized top-three probabilities, requested-token probabilities, and the actual history SHA256.

```bash
python3 -m tools.eval.offline_quality prepare --checkpoint MODEL_DIR --output artifacts/quality/requests.json
python3 -m tools.eval.reference_bf16 --checkpoint BF16_DIR --revision SOURCE_COMMIT \
  --requests artifacts/quality/requests.json --output artifacts/quality/bf16.json
python3 -m tools.eval.offline_quality queries --requests artifacts/quality/requests.json \
  --baseline artifacts/quality/bf16.json --output artifacts/quality/queries.json
target/release/orinfer score-model MODEL_DIR artifacts/quality/queries.json > artifacts/quality/candidate.json
python3 -m tools.eval.offline_quality compare --requests artifacts/quality/queries.json \
  --baseline artifacts/quality/bf16.json --candidate artifacts/quality/candidate.json \
  --output artifacts/quality/comparison.json
```

Serialize GPU jobs using `flock artifacts/gpu-experiment.lock`. Reference and candidate use seed 20261002 and identical original token IDs. Scenario answers freeze the scoring history; task correctness under independent free generation still requires validation. Probability comparisons are not a complete LLM benchmark and do not independently establish equivalence of BF16 autoregressive generation.

If the raw checkpoint will not fit on disk, use `--remote-repo REPO --revision COMMIT` with a directory retaining the original configuration, tokenizer, and weight index. `COMMIT` must be a pinned 40-character commit. The reader checks HTTP Range boundaries, fetches original tensors by component in parallel, and records the SHA256 of bytes actually read; it never reconstructs original tensors from quantized weights. `--read-workers` controls network concurrency (1–16). Optional `--tensor-cache DIR` consumes prefetched standard safetensors, validates source revision and payload hash, and deletes files once consumed. This streaming path is for quality comparison. Its network-inclusive runtime must not be reported as model startup performance.

`prepare --long-context-tokens 512 2048 8192` adds deterministic middle-document retrieval cases. These retain the full Chat template and produce actual token lengths close to, but no greater than, the requested budget. The scoring history fixes the correct retrieved answer. The reference `--batch-size` defaults to 1; larger batches reduce weight reads, but BF16 rounding can change with GEMM shape. Use a single-stream reference when accepting single-stream quality.

Image-scoring cases add an `images` array. Each image contains `grid_height`, `grid_width`, and normalized FP32 patches in `pixels`, matching the online `ImageInput` contract. `prompt_ids` must include the native template and expanded image-placeholder IDs. Reference and candidate use identical patches. The reference runs the official BF16 vision encoder, feature injection, and MRoPE, not features exported by the candidate. Input fingerprints include image dimensions, SHA256 of little-endian FP32 pixels, and image order to prevent mismatched comparisons. This measures from vision encoding through token probabilities; compare raw-image decode/resize independently, and validate task results with free generation.

The Transformers image reference requires explicit `mm_token_type_ids`: expanded image-pad tokens are `1`; text and delimiters are `0`. Preserve this field for free generation as well so MRoPE is used. The deployed tokenizer must incorporate added-token settings from `tokenizer_config.json` and match actual Transformers token IDs; identical rendered strings do not necessarily tokenize identically.
