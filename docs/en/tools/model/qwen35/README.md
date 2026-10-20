# Qwen3.5 Dense native Q4A8 build

[简体中文](../../../../../tools/model/qwen35/README.md) | **English**

The current 27B baseline uses the official model's beam-8-reencoded, two-stage E8P Q4A8 quantized weights. It uses signed-block128 rotation, two fixed integer codebook stages, and per-output-row scales. Embeddings, mixer output projections, and MTP draft matrices retain W8; small sensitive weights and vision paths remain floating point. Local artifacts: `artifacts/quantization/qwen35-e8p-beam8-20261016/converted` and `artifacts/models/qwen3.8-27b-e8p-q4a8-current`.

The same builder supports Index-Translate-2B (24 layers) and Index-Translate-9B (32 layers), encoded from pinned BF16 source by `tools/quantization/qwen_e8p.py`. GDN/GQA, FFN, vision, MTP, dynamic rows, and state control are reused according to configuration, without duplicating execution paths. The 9B model has separate E8P Q4 embedding and output-head banks. The 2B model ties them to one bank: selected embedding rows are decoded and inverse-rotated on demand; output projection uses the same compressed weights with rotated A8 computation. Large vision matrices use Q4A8 with original FP16 bias/GELU boundaries; small patch, position-table, and norm parameters remain floating point, and sensitive output/MTP matrices remain W8. Average bits per weight account for those exceptions.

Qwen3.5-4B uses this same build entry point with its official 2,560-hidden, 9,216-FFN, 32-layer configuration and tied embedding/output head. When `generation_config.json` is missing, the builder pins EOS and padding using the model configuration and tokenizer, including model-end and chat-end markers; explicitly supplied generation configuration retains its original bytes.

## Quantized-weight contract

Two-stage E8P Q4 uses:

- `quant_method=orinfer_e8p_q4_int8`
- `coefficients=[4,1]`
- `basis_tensor=orinfer.e8p_q4.basis`
- `layout=u32-low-coarse-high-residual`
- Compressed matrix U32 `[N,K/8]`, losslessly repacked to `[K/128,N,16]` at build time
- One F16 scale per output row

The model reuses native execution plans, shared integer compute, and offline tile tuning. Once the nonzero basis is verified, direct INT4 MMA splits full A8 into two bit planes and reconstructs the original INT32 dot product. The INT8 decoding path reads the same compressed bank; autotuning chooses which implementation to use.

```bash
make build
bash tools/operators/run.sh tools/model/qwen35/build.py artifacts/qwen35-build \
  --source /path/to/converted --model-output artifacts/models/qwen35-q4a8 \
  --gpu-lock-held
```

The builder accepts **only complete quantized outputs** and reads previously validated safetensors shards by identity. Two-stage E8P codewords are only rearranged; other tensors retain original bytes. Expanded W8 copies are never persisted and weights are not requantized. Necessary floating layouts/constants occupy one extra shard. Default context capacity is 262,144 tokens and can be lowered with `--context`.

Published directories contain HF configuration/tokenizer assets, `cache/model.json`, standard shards, and a digest-pinned schema-1 execution package, like Flash Next. The package has a Rust model library and merged/deduplicated cubin/host ABI. Layer order, state commit, and batching are registered in Rust, not serialized into data files.

At build time, HF `tokenizer_config.json` added-token entries are merged into the deployment `tokenizer.json` while checking that token IDs stay unchanged. The chat template comes from a standalone file if present, otherwise `tokenizer_config.json`; its bytes are preserved, checked, and pinned before compilation. This aligns the Rust tokenizer with Transformers thinking, tool, and vision markers without running Transformers online.

## Integer compute and dynamic rows

Direct INT4 represents each E8P integer as `E=2n+1` and the complete A8 activation as `A=lo+16hi`. Two bit-plane products are accumulated, `(a+b)·sum(A)` is restored, then the original weight and token scales are applied in order. **No A8 information is discarded.** Compressed weights are pipelined in K128 slabs through CTA shared memory. One, two, or four warp groups may divide the K dimension; INT32 partial sums are merged inside shared memory before the existing single rescale, without another global reduction. One offline tuner selects legal split-K/N-tile combinations based on measurements, not model names or request types.

Prefill, ordinary decode, and MTP target verification share row compute templates. Common 1/2/3/4/8/16/32 row counts are auto-tuned offline; other lengths use dynamic-argument cubins. Dynamic prefill selects one setting using total GPU time across actual row sizes, then full-request acceptance. Decode fallback balances relative cost at sampled sizes. Both check tails and numeric contracts. Projections write directly into QKV and gate/up output ranges instead of concatenating giant weight matrices.

Gate/up occupy separate planes of one temporary buffer, each with a fixed capacity of 4096 rows. SwiGLU and rotation use explicit plane offsets regardless of actual row count, preventing layout drift for short tails or single-row decode. Requests own FP32 GDN, convolution history, INT8 KV, and MTP state; accepted MTP prefixes use compact update factors.

Prefill can pause at declared layer boundaries while preserving the true row count of the same large token block. The shared runtime retains hidden and FP32 residual values across layer boundaries without mixing private request state into shared workspaces. Adjacent layers run while time budget remains, based on measured costs; there is no fixed number of layers per group. Flash Next and 27B share layer registration, dynamic-row invocation, and address-table binding.

Ordinary decode handles 1–128 independent requests' private GDN/attention with `independent_rows` around the existing one-row compute body. This does not introduce batch-size-specific implementations.

## MTP, batching, and state

Multi-request MTP reuses ABI roles Verification, Draft, Commit, Warm, and DraftHead. Mixed prefill/verification uses `MixedRows` with explicit role boundaries. Shared target pre/post compute and draft projections are batched, while each request's causal state advances independently. Ordinary requests collect final positions; verification requests collect every verification position. After acceptance, GDN factors, convolution history, and positions are restored/committed, then each request's draft conditioning advances in a batch. Single-request verification uses the same packet; CLI draft ceiling and runtime cost model control width.

For each matrix geometry/output dtype/stride/compression layout, the package publishes one full-capacity dynamic projection fallback. Offline tuning covers geometries, tile boundaries, and neighboring row counts, selecting the best representative total GPU latency. Rejected candidate cubins are not published. AOT variants for 1/2/3/4/8/16/32 rows share the same template. For other sizes up to 128, a bounded dynamic specialization selected from offline measurement accepts the **actual** row count, not padded empty requests.

The E8P INT8 Tensor Core path asynchronously stages compressed K128 slabs and activations into CTA shared memory, decodes original codewords, and accumulates. Candidate tiles include larger M to amortize decode, measuring short tails, 512, and 4096 rows. Only block-local compressed staging is added, with no resident expanded weights. GDN sequential recurrence unroll counts are chosen by row-count tuning and checked for elementwise FP32-state/FP16-output equivalence; unselected options are not released.

Residual addition, zero-centered RMSNorm, block128 rotation, and row A8 quantization are fused into a dynamic-row kernel while retaining FP32 residuals and both original FP16 rounding boundaries. GDN A/B FP16 Tensor Core projections select tiles by actual row counts without altering MMA K accumulation order. Shared quantization helpers use TileLang macros to preserve bindings and output types.

Wide FFN SwiGLU and block128 rotation run across column tiles and derive one output-row A8 scale from the local maxima without new per-group scales. Causal attention shares row templates and offline-selects query-head groupings and split limits. Runtime supplies actual rows and KV length. INT8 KV, FP32 softmax, short-vector FP32 tree merge, FP16 output boundaries, and gating are preserved. Compact striped KV reads avoid excessive unrolled instruction streams, and tuning includes partial KV tiles. Prefill, decode, and MTP reuse the same compute body, with Rust binding each request's KV and positions.

`tools/model/row_tuning.py` selects and merges adjacent selection regions using the same rules as Flash Next. Only selected cubins enter execution packages. Workspace contains temporary rotation and bounded partial-attention results, never expanded weight caches.

## Vision, convolution, and acceptance

Single-/multi-image input uses shared ViT kernels: one dynamic patch template handles 4–32768 patches, and the merger groups patches in fours. Quantization, execution, and vision behavior are checked against real checkpoint semantics.

Run two-stage E8P numeric, tail, and Graph tests with `tools/quantization/check_e8p_q4.py` and `tools/quantization/check_e8p_direct.py` using the GPU wrapper. Convolution time tiles 1/2/4/8/16 are measured offline and reuse eight K partial sums, a 16-lane reduction, and original FP16 boundaries. MTP, short prefill tails, and full blocks use the same compute. Private ordinary-decode convolution snapshots the old tail and commits history in place. Batch address tables isolate histories without a separate history-copy kernel. E8P sign/parity correction merges into the final XOR without changing codewords or INT32 dots.

Operator checks:

```bash
bash tools/operators/run.sh tools/model/qwen35/check_kernels.py artifacts/qwen35-kernels \
  --gpu-lock-held
bash tools/operators/run.sh tools/model/qwen35/check_rows.py artifacts/qwen35-rows
bash tools/operators/run.sh tools/model/qwen35/check_decode.py artifacts/qwen35-decode
```

Build cache defaults to `artifacts/build-cache/qwen35`. Resume interrupted builds with wrapper `--resume`; validated compiler assets are reused. **New execution packages still require real-model loading, generation, prefix restore, concurrency, MTP, probability, and whole-state acceptance.** Passing operator checks is not proof of end-to-end quality.
