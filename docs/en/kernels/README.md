# TileLang kernels

Target: **CUDA SM87**. `operators/` contains primitive operators; `model/` contains model projections, fusions, GDN, and attention.

Kernels compile offline into cubins with a real host ABI; Rust executes them online through the CUDA Driver API. See [Model tools](../tools/model/README.md) for export and assembly. Every implementation must validate numerical results, unaligned tails, state semantics, and CUDA Graph replay **after changing the inputs**.

## Shared compute components

- `model/integer_core.py` provides shared DP4A/Tensor Core compute loops, tail handling, and output scaling for W8, scalar-codebook Q4, and dual-W8 fused SwiGLU. Weight loaders vary by encoding. Gate and up are each rounded to FP16 before SwiGLU. Outer factories provide geometry/ABI validation and launch mapping; two schedules are selected offline rather than cloning compute by model or request type.
- `model/floating_core.py` shares MMA loops across general floating projections, grouped BF16, HC, and vision Linear. Layout, operand orientation, and sequential group reductions are fixed at compile time. K tails are padded within tiles. HC's K16 accumulation and intermediate rounding may not be replaced with parallel reductions. Conv3D bias rounding and Linear/GELU outputs preserve their own epilogues.
- `model/e8p_core.py` shares full A8 bit-plane decomposition, dual INT4 MMA, and output reconstruction/scaling. Grouped, K-major pipelined, and unique-route cached-input paths share arithmetic but preserve separate data movement. The unique-route path requires at most one row per expert, guaranteed by the caller's routing contract; it is not hard-coded for model E/N/K.

TileLang macros inline these components. Runtime does not interpret Python or choose codecs; compiled host ABIs expose weight/state buffers and necessary dynamic row parameters. The 27B two-stage E8P Q4A8 path shares E8P arithmetic, rotation/A8, and dynamic-row scheduling. The builder validates encoding and K128 layout.

BF16 vision epilogues use explicit PTX round-to-nearest-even writes, preserving Conv3D's separate dot rounding and Linear/GELU precision boundaries without unnecessary NVRTC BF16 vector round-trips. FP16 remains automatically vectorized. `tools/vision/validate_kernels.py` checks both precisions, K/N tails, and changed-input replay.

`tools/model/flash_next/checks/shared_compute.py` compares W8/Q4 integer dots, FP16/FP32 outputs, and fused SwiGLU to an independent CPU reference, including unaligned N, dynamic tails, and changed-input replay. Floating/expert paths are separately checked against reduction/routing contracts; comparing two implementations that share the same component is not independent validation.

`model/moe.py` implements FP32 top-k routing, deterministic expert dispatch, and output combination. A8 quantization and fused SwiGLU reuse `operators/op30_activation_quantization.py`. Integer E8P experts decode tiles inside `model/e8p_exact.py`, `model/e8p_prefill.py`, and related schedules, without keeping expanded weights resident.

`model/hyperconnection.py` implements Qwen4 gated-residual branch normalization, low-rank projections, mixing, and residual updates, retaining BF16 weights and FP32 accumulation. Small-row paths fuse down/SiLU and up/branch mixing separately while preserving original FP16 intermediate rounding. `model/ple.py` implements PLE gating, dilated convolution, and history updates. `tools/model/flash_next/ple.py` provides CPU n-gram hashing and bounded E8P row caching. Validate via `tools/operators/hyperconnection.py` and `ple.py`.

`model/integer_vq.py` retains the E8P INT8 reference used for offline build/validation and input-side block128 rotation. `model/rotation_a8.py` fuses rotation, optional SwiGLU, and per-row A8 quantization without changing FP16 rounding boundaries. Rotation must precede the corresponding projection; down-projection rotation occurs after SwiGLU. Validate via `tools/operators/e8p_selector.py` and [Flash Next model checks](../tools/model/flash_next/README.md).

`model/int8_projection.py` and `model/q4_projection.py` supply per-row-scaled W8 and codebook-Q4 projections. `tools/model/flash_next/native.py` builds offline reference models; `tools/model/flash_next/build.py` produces deployment execution packages.

QSA uses group-four compressed indexing, exact radix top-k, and group-64 INT8 KV in `model/qsa.py`. `model/qsa_attention.py` computes attention with shared-KV-tile FP16 Tensor Cores while softmax and accumulation remain FP32. Maximum context is 262,144 tokens. Compressed index, pending ring, and KV scales are request state. See `tools/operators/qsa*.py` for kernel checks and `tools/model/flash_next/long_context.py` for actual long-context runs. [Quantization](../tools/quantization/README.md) documents conversion and independent BF16 references.

`model/flash_mtp.py` provides complete HC input normalization, embedding/branch fusion, and per-position MTP history. `model/greedy.py` handles multirow verification output. Target verification saves convolution, PLE, and QSA pending histories. GDN stores keys, decay, and FP32 updates, and `model/gdn_sequence.py` replays accepted prefixes into FP32 state rather than cloning full matrices at each position. Commits follow accepted length; live cursors isolate rejected KV and compressed-index suffixes. `tools/model/flash_next/mtp.py` supplies offline greedy acceptance, draft refresh, and full-session restoration. MTP shares base-model embeddings/output head while using separate INT8 KV. Verify operators with `tools/operators/flash_mtp.py`, actual requests with `tools/model/flash_next/checks/mtp.py`, and fused/GDN/direct-expert variants with `tools/operators/flash_mtp_optimization.py`.
