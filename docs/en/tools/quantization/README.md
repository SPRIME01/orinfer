# Offline quantization

[简体中文](../../../../tools/quantization/README.md) | **English**

CPU utilities use the project's `uv` environment; GPU encoders run in the pinned compiler image. Encoding, storage repacking, and execution-package building are separate steps. Quantized outputs require independent quality acceptance before deployment.

## Current representations

| Application | Weight representation | Scaling and computation |
| --- | --- | --- |
| Flash Next experts and PLE | Integer E8P Q2, with one U16 encoding eight weights | Per-output-row FP16 scale; decoded to INT8 |
| 27B backbone and output head | Two-stage integer E8P Q4, one U32 encoding eight weights | Fixed coefficients `[4,1]`, per-output-row FP16 scale, A8 computation |
| Index-Translate-2B/9B backbone | Two-stage integer E8P Q4 | Fixed coefficients `[4,1]`, per-output-row FP16 scale, A8 computation |
| Selected Flash Next non-expert projections | Rotated scalar-codebook Q4 | Signed-block128 rotation, per-output-row FP16 scale, A8 computation |

Sensitive paths retain W8, FP16, BF16, or FP32. Actual storage accounting includes scales, bases, rotation signs, padding, and precision exceptions. No permanently resident expanded W8 copy is stored.

## E8P encoding and layout

`e8p.py` derives an integer basis from lattice geometry; `e8p_gpu.py` fits original floating-point matrices with a TileLang nearest-neighbor encoder. Basis and ordering are part of the format identity. `vq.py` provides E8P codeword decoding, selector tables, lossless K64 repacking, and safetensors numerical fixtures.

`e8p_q4.py` specifies two-stage Q4 storage and its CPU reference. `e8p_q4_gpu.Encoder.fit()` fits two codewords in blocks; `Encoder.refine(raw, original)` alternates updates against the *original* floating-point weights and performs a joint beam-8 search. Current 27B encoding uses coefficients `[4,1]` and beam-8. Both stages share one basis; there is only one FP16 scale per output row, not per-group floating-point scales. Saved codewords use U32 `[N,K/8]` and are losslessly rearranged into K128 compute layout by the builder.

Callers retain original weights in output-row chunks (for example, 128 rows at a time). `raw` must come from original BF16/FP8, not dequantized community weights. Storage layout, FP16 scales, rotation, and coefficients must match the execution package.

```bash
.venv/bin/python -m unittest tools.quantization.test_e8p tools.quantization.test_e8p_q4
bash tools/operators/run.sh tools/quantization/check_e8p_q4.py artifacts/check-e8p-q4
bash tools/operators/run.sh tools/quantization/check_e8p_search.py artifacts/check-e8p-search
```

`check_e8p_direct.py` verifies full A8 bit-plane decomposition and reconstruction of original INT32 results from two INT4 Tensor Core products. This is **not A4 quantization**. `evaluate_e8p_q4.py` compares component fixtures with original-weight sources and reports reconstruction error and projection latency. Measure model NLL and full-request performance separately.

## Preparing weights and reference models

See [Flash Next builds](../model/flash_next/README.md) for pinned-source conversion, PLE, non-expert preparation, and original BF16 teacher references.

`row_q4.py` and `row_q4_gpu.py` implement scalar Q4 with codebook `±(3,10,19,31,46,67,93,127)`. `qwen35.py` prepares the 27B model in chunks, including scalar Q4, W8 exceptions, and original tensors. **That path does not perform two-stage E8P encoding or beam-8 optimization.** The [27B native builder](../model/qwen35/README.md) needs a checkpoint with two-stage encoding already completed.

Source metadata must pin the repository/revision, configuration, and safetensors index. Remote tensors may be read by byte range; `--source-dir` reuses locally downloaded shards. Preparation utilities verify source identity, encoder implementation, seed, and completed chunks. They do not mix sources or accept incomplete outputs.

`qwen_e8p.py` supports Index-Translate-2B/9B and 27B, performing two-stage fitting and beam-8 directly from official BF16 shards. The source directory must contain original Hugging Face files plus `source.json` with a pinned `repo` and 40-character `revision`. `--shard FILE` processes one shard at a time. Completed blocks retain checksums; temporary blocks are removed only after a complete shard is written, allowing the caller to release encoded source shards.

Embeddings, shared output heads, and large vision matrices also use E8P Q4. Sensitive output projections and MTP matrices remain W8; small sensitive patch, position, and norm parameters remain at original precision. For the 2B model, the tied output head uses the same compressed embedding bank, without an expanded copy. The 9B model separately encodes embedding and output head according to its official configuration; weight-sharing semantics are preserved.

```bash
bash tools/quantization/run.sh qwen_e8p --source-dir /path/to/original \
  --output artifacts/quantization/index-translate-2b-e8p-q4a8
```

Tiny E8P queries return the top eight results directly in the original FP64 norm/code scoring order to avoid degeneracy in equidistant candidate search. Thresholds preserve the original scores; weights are not zero-clipped, nor are beam width or iterations reduced. `check_e8p_search.py` contains exhaustive, tiny-value, and changed-input Graph comparisons.

`rotation.py` supplies chunked and full Hadamard transforms. `block_ldlq.py` retains CPU math references for covariance factors and error feedback; it does not imply full-model calibration or fine-tuning. Rotation does not automatically preserve equivalence across activation quantization, nonlinearities, and rounding boundaries.

Quantization quality is compared to the **same model's original BF16**, or explicitly labeled FP8 when resources are limited. With fixed seed 20261002 and teacher-forced histories, compare task results, target NLL, probabilities, and long-context behavior. Differences in selected tokens do not automatically imply failure; local reconstruction error alone cannot establish end-to-end model quality.
