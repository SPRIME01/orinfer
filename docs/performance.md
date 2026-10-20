# Performance reference

Measured results on **Jetson AGX Orin 64GB / CUDA SM87**, with dates given below. On October 12, 2026, the device ran in MAXN at a GPU frequency of 1300.5 MHz; tests did not alter global power or frequency settings. Measurements use the Rust release CLI and matching optimized execution packages. Changes to native libraries, kernels, or ABI require rebuilding the execution package; CPU-only scheduler changes may reuse it.

**Read the measurement boundary before comparing numbers.** Execution-layer prefill, fixed-length decode, HTTP end-to-end TPS, TTFT, total JSONL job TPS, and GPU-only kernel times are different metrics. Token rates use actual token counts. Full-prefix-hit results are warm-cache upper bounds, not cold-request performance.

## Index-Translate-9B (2026-10-18)

Source: official [IndexTeam/Index-Translate-9B](https://huggingface.co/IndexTeam/Index-Translate-9B), revision `4e06279e62352dab96bbeb6fa0579bca5e822d75`. Reuses Qwen3_5 dynamic-row, GDN/GQA, FFN, vision, MTP, and state components with 32 layers, hidden size 4096, and separate embedding/output heads.

Backbone, embeddings, output head, and large vision matrices use signed-block128 rotation and two-stage E8P Q4A8 (`[4,1]`, beam-8). Sensitive outputs/MTP use W8, with floating-point small matrices/norms. There is no activation calibration. Quantized payload including scales/codebooks and precision exceptions: **5,440,207,936 bytes / 4.509 bits per weight**. Deployed read-only data including derived RoPE constants: 5,473,766,976 bytes. KV is INT8, recurrent GDN FP32. Context is configured to 256K but measured here only to 8K.

Rust release `run-model`, MTP disabled, `decode_only`; two consecutive runs per input, taking the second, with 65 generated tokens. Prefill excludes the output head; decode times the subsequent 64 tokens. This is *not* naturally ending HTTP throughput.

| Input tokens | Prefill tokens/s | Decode tokens/s | First-token time (s) |
| ---: | ---: | ---: | ---: |
| 512 | 2,152.6 | 33.42 | 0.241 |
| 2,048 | 2,219.6 | 32.62 | 0.926 |
| 8,192 | 1,956.4 | 29.82 | 4.190 |

Ready-after-load time was **4.65 s** with warm filesystem cache; a full `validate-model` weight check took **6.93 s** separately. Neither establishes disk-cold startup performance.

Against same-source BF16 at seed 20261002 and identical teacher-forced histories: 8 text translation cases / 130 positions had target NLL **+0.02370 nats/token**, **96.15%** matching top-1, all choices within BF16 top-3, and no changes at 110 high-confidence positions. At 2K/8K (26 positions), NLL difference was **-0.07549**, 96.15% top-1 agreement, all in top-3. One 8K difference was equivalent phrasing ("load compressed model" versus "load a compressed model"). Across notification-image and ordered red/blue two-image cases (17 positions), NLL was **+0.08327**, 94.12% top-1 agreement, all within BF16 top-3.

Actual Chat API checks covered MTP off/up to three drafts, SSE, full-prefix hits, eight-way concurrency, terminology, preserved JSON, one image, and two images. Requests used greedy decoding, thinking off, 256-token output budgets, and natural stopping. MTP produced equivalent but not always identical Chinese renderings of "the model only needs loading once"; cold and three subsequent full-cache-hit single-stream responses used the same wording. Simple image translation and red/blue ordering worked; double-image responses could produce redundant labels, and complex images were not accepted for quality.

Ordinary 2/3/4/5/8/32-way batches plus 24 matched-history verification/mixed cases (width 2/4/8 × requests 1/2/4/8) had zero KL and selected-token NLL differences. Cancellation/slot reuse and partial 32/512-token prefix restoration preserved target/draft KV, GDN, convolution, and positions. A 16-record translation JSONL job completed naturally with resume avoiding duplicate writes. Full validation, `make check`, and `make check-offline` passed. These are limited state and quality checks, **not a comprehensive benchmark or 256K validation**.

## Index-Translate-2B (2026-10-18)

Source: official [IndexTeam/Index-Translate-2B](https://huggingface.co/IndexTeam/Index-Translate-2B), revision `9bc08445315723eb9b77d1b93505915eb03c3f01`. Two-stage E8P Q4A8 covers backbone, tied embedding/output head, and large vision matrices without activation calibration; sensitive output/MTP use W8. Quantized payload: **1,228,931,712 bytes / 4.323 bits per weight**; deployed read-only data with RoPE etc.: 1,262,497,024 bytes. INT8 KV, FP32 GDN, configured 256K context, tested here to 8K.

The adapter reuses 27B dynamic-row, private state, vision, and MTP components. Offline attention tuning includes 8/16-way splits (16 selected for short rows), keeping existing large-prefill pipelines. Private single-row convolution commits history in place without a separate copy; parity correction is folded into the final E8P XOR. Conversion acceptance measured reference warm Graphs at 512/8K in **8.72/10.03 ms**, excluding host-request overhead.

### Fixed-length CLI execution

Rust release `run-model` with MTP and prefix caching disabled, `decode_only`, 65 forced output tokens, two runs per input, taking the second. Measurements alternated baseline/optimized/optimized packages. Prefill excludes the head; decode measures the subsequent 64 tokens.

| Input tokens | Baseline prefill TPS | Optimized prefill TPS | Baseline decode TPS | Optimized decode TPS |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 7,680 | 7,630 | 106.15 | 108.41 |
| 2,048 | 7,988 | 7,908 | 103.03 | 105.09 |
| 8,192 | 6,564 | 6,537 | 92.77 | 94.37 |

Relative to a baseline already incorporating attention tuning, decode improved ~1.7–2.1% while prefill moved by less than 1%. Warm-filesystem load-to-ready changed 2.72 → 2.77 s; it is not a cold-start comparison.

### Ordinary greedy decode concurrency

Rust request-path cohort benchmark, MTP/prefix disabled. All requests complete independent prefill first, followed by four warmup decode steps and 16 measured steps; 512 inputs were repeated twice and averaged, 2K/8K once per concurrency setting. Timings include request state, input preparation, and result commit, but exclude prefill, HTTP, and queues. No EOS logit bias is added. This differs from the `run-model` conversion-reference path above.

| Active requests | Aggregate TPS, 512 input | Per-request TPS | Gain over earlier implementation |
| ---: | ---: | ---: | ---: |
| 1 | 113.2 | 113.2 | +0.7% |
| 2 | 215.1 | 107.5 | +4.0% |
| 4 | 390.1 | 97.5 | +7.4% |
| 8 | 636.4 | 79.5 | +12.1% |
| 16 | 835.7 | 52.2 | +16.4% |
| 32 | 1,125.1 | 35.2 | +24.1% |
| 64 | 1,358.5 | 21.2 | +42.1% |
| 128 | 1,469.2 | 11.5 | +39.9% |

| Active requests | Aggregate TPS, 2K input | Aggregate TPS, 8K input |
| ---: | ---: | ---: |
| 1 | 109.5 | 98.7 |
| 8 | 541.7 | 347.8 |
| 32 | 855.7 | 450.5 |
| 128 | 1,036.8 | 501.1 |

Actual 128-way decode executed at all three lengths without reaching GPU memory limits in this test. The server/execution active-request ceiling is 128; default server cap remains 32. With 512 inputs, per-request speed stays within 10%/20% of single-stream speed at roughly 2/5 active requests.

Optimizations batch ownership-based token/position/state readback; upload stable private input asynchronously; avoid arena switches for already mapped writable KV; and cache immutable role bindings in the native model library. Dynamic projection fallback handles actual row counts rather than padding or enumerating variants; exact AOT sizes remain 1/2/3/4/8/16/32.

For irregular 512-input batch sizes 5/12/24/31/33/65, aggregate TPS was **463/709/1015/1127/1100/1285**, versus earlier **264/502/657** at 5/12/24. The 65-row case still loses about 5% against 64 due to a tile boundary; tuning is measured, not manually switched.

At 128 active 512-input requests, step time fell **121.9 → 87.1 ms**, with Graph execution accounting **85.6 → 79.0 ms** and overhead outside that interval **36.3 → 8.1 ms**. That interval also contains input prep, control transfers, and commit—not solely CPU arithmetic. Updating only the shared executor yielded about 30% more throughput at 128; updating the execution package raised it to ~40%. At 2K/8K, 128-way gains were ~28%/14%; private long-context attention remained limiting.

**The target of 150 TPS single-stream decode without MTP has not been reached.** Current ordinary request-path 512-token-input single-stream decode is ~113 TPS; 150 requires no more than 6.67 ms per step.

### HTTP Chat concurrency

A fixed translation request used 221 input tokens and naturally produced 183 tokens, with thinking/MTP disabled, seed 20261002, and `--max-active-requests 128 --max-batch-tokens 128 --cuda-graph decode_only`. Each setting was repeated twice. Throughput is real usage output over complete HTTP time, including prefill, queues, Graph capture, and SSE. Identical prompts hitting the full prefix are an *upper bound on caching*, not representative of varied translation requests.

| Arriving requests | No prefix cache TPS (before → after) | Full-hit TPS (before → after) | Current no-cache TTFT P95 (s) | Current full-hit TTFT P95 (s) |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 628.0 → 750.2 | 894.0 → 1,089.6 | 2.49 | 0.15 |
| 64 | 639.8 → 813.0 | 894.3 → 1,234.2 | 5.50 | 0.94 |
| 128 | 625.6 → 791.2 | 934.7 → 1,268.5 | 14.30 | 2.98 |

Peak active counts and real batches matched configured levels. For full-cache-hit 128-way HTTP requests, completion P50 dropped **24.32 → 17.86 s**. Across 1,064 actual translation calls, all naturally ended at 183 output tokens with content matching the earlier single-stream baseline. API overhead, admission, capture, and small tail batches constrain gains relative to fixed cohorts.

Matched-history owner-by-owner checks across 5/8/12/16/24/32/33/64/128 requests with 512/2K inputs yielded 1,288 positions with zero maximum KL and target NLL difference; all selections were within reference top-3. Cancellation, reuse, and reclamation passed. These compare the **same quantized model**, not original BF16.

Earlier single-stream SSE acceptance used [fixed translation prompts](../fixtures/translation-benchmark.json), thinking off, greedy, seed 20261002, 512 output budget, and 512 MiB prefix cache. With one warmup and three identical full-prefix-hit runs per setting, complete-HTTP median output rates were **112.97 TPS without MTP (183 tokens)** and **152.46/153.33/148.77 TPS** with up to 1/2/3 drafts. Time, identifiers, and three paragraphs were preserved despite wording changes. MTP figures do **not** meet the separate 150-TPS-without-MTP goal.

Against original same-source BF16: eight text cases / 130 teacher-forced positions had **+0.03385 nats/token** target NLL, **96.92%** top-1 agreement, all choices within BF16 top-3, and 96 unchanged high-confidence positions. A 2K/8K terminology subset / 30 positions had +0.13515 NLL and correctly completed the target translations. Convolution/E8P decode optimization preserved probabilities/top choices at 177 text, context, and vision positions; 24 MTP verification/mixed scenarios had zero KL/NLL differences. Partial 32/512-token prefix state and 78 attention cases, all 65,536 E8P codewords, convolution tails, and changed-input Graph replay passed.

Vision remains limited. Simple notification and red/blue image tests passed, while extra-newline two-image, menu time/price, and complex multi-image cases produced duplication/recognition errors. Original BF16 also failed some menu cases. Returning the vision merger alone to FP16 did not reliably fix them; the selected Q4 policy remains. Across 17 image-history positions, NLL was **+0.24353**, with 94.12% of top choices inside BF16 top-3. This is **not a passed comprehensive vision-quality evaluation**. BF16 image references supply modality IDs for MRoPE; tokenization was checked against original HF added tokens.

### Whole-file JSONL jobs

Comparison of `batch --scheduler-policy latency` versus `throughput` on identical 2B weights/package, 64 active requests, 128 decode rows, `decode_only`, MTP/prefix/thinking disabled, greedy seed 20261002, and 256 output budget per request. Three separate-process runs were alternated; warm filesystem caches were not cleared. Values include scan/load/preprocess/queue/generation/write time. TPS is actual total output over *complete CLI duration*.

| File | Input tokens per record | Latency-policy duration | Throughput-policy duration | Speedup | Aggregate TPS (latency → throughput) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 200 distinct short translations | 88–90 | 24.02 s | 18.89 s | 1.27× | 564.6 → 717.0 |
| 96 mixed-length translations | 86–2,327 | 23.14 s | 16.97 s | 1.36× | 163.2 → 222.0 |

The mixed file had 48 short requests and 16 each around 227/647/2,327 input tokens. All runs ended without errors, truncation, or duplicate IDs; short jobs produced ~13.5K output tokens and mixed jobs ~3.8K. Short-file ranges were latency 23.89–24.03 s and throughput 18.82–18.91 s; mixed-throughput 16.95–16.99 s. One mixed-latency run had a model-load storage I/O timeout and took 52.93 s; it remained in the median, but its wait is not a scheduling improvement.

Throughput mode relaxes latency admission quotas and mandatory decode time, probes unknown joint shapes, and selects by full measured cost. Explicit 64-way concurrency exceeds the CLI default of 32; higher concurrency is not always faster and per-request TTFT/ITL may worsen. These comparisons predate global prefix planning and output-length estimation.

Ten joint-prefill matched-history cases at 512/2K and 2/8 requests differed from serial quantized references by at most **0.000154 mean KL** and **+0.001832 mean target NLL**; all choices were within top-3. Full-hit dual requests had zero differences. Cold prefill, deterministic-seed reorder, cancellation, slot reuse, and private state passed. JSONL failure/truncation handling, interrupted 200-record resume, HTTP/SSE comparison, and two-draft MTP translation/sampling/JSON-schema/stop passed. These checks are not original BF16 quantization acceptance.

### Ten thousand real translation requests

Used official [OPUS-100 English–Chinese corpus](https://opus.nlpl.eu/opus-100.php), deduplicated with fixed seed 20261002 into disjoint sets of 1,024 tuning and 10,000 acceptance records. Each sentence was translated independently with greedy decoding, thinking off, up to 2,048 output tokens, and natural stopping. Input length P50/P90/P99/max was **41/77/135/731 tokens**; output **15/45/95/765**. This is a short-sentence-dominated workload, not long-document translation.

Same 2B weights/package, 64 active, throughput policy, `decode_only`, MTP/prefix off. Prior baseline commit `1999fc3` already included throughput scheduling. New changes added budget-bounded whole-file token-prefix planning, long-request priority and observed completion lengths, branch snapshot selection, asynchronous bounded result persistence, and 4 MiB pinned-pool chunked uploads for large inputs. HTTP default behavior and the package were not changed.

| 10,000-record metric | Previous offline mode | Revised offline mode |
| --- | ---: | ---: |
| Total CLI invocation | 358.81 s | **314.78 s** |
| Ready-to-last-completion | 324.30 s | **309.67 s** |
| Actual generated tokens | 217,193 | 217,196 |
| Full-invocation output TPS | 605.3 | **690.0** |
| Post-ready output TPS | 669.7 | **701.4** |

The previous startup had about 30 seconds of storage I/O wait, absent in the new run. Thus the full-call 1.14× speedup is **not entirely an optimization gain**; ready-to-completion speedup is **1.047×**. One complete 10K run per version was measured. Separate 1,024-record tuning favored 64 over 128 active requests and no MTP over two drafts. For similar short-sentence workloads, try `--max-active-requests 64 --mtp-drafts 0 --prefix-cache-mib 0`; the CLI default remains 32.

Whole-job iterations fell **4,108 → 3,402**, including one-active-request tail iterations **577 → 1**. Execution-plan preparation fell **37.11 → 31.51 s**. Full dynamic packet compute and host preparation still dominate independent short sentences.

With the same current binary, same 10K inputs, and unchanged other settings, switching to `latency` produced **535.89 s / 405.3 TPS**, 217,206 generated tokens, all naturally finished. Compared against that *current* latency-policy baseline, current throughput mode completes the job **1.70× faster**. This must not be confused with the 1.047× post-ready improvement over the previous throughput implementation. Both use local CLI; HTTP overhead is absent.

Of the 10K translations, **9,999 were exactly identical**; chrF++ changed **23.2760 → 23.2758**, Chinese-tokenized BLEU **33.3474 → 33.3474**. This is a same-weight scheduler regression check, not new BF16 quantization acceptance. Noisy multi-genre OPUS absolute scores do not substitute for domain evaluation.

A separate 1,024-sentence test built 32 document groups, each with the same artificial 301-word translation guide, then shuffled them. Sentences were not repeated. With 64 active, 128 MiB prefix cache, MTP off, `decode_only`, global prefix planning reduced total invocation **100.63 → 63.85 s (1.58×)** and ready-to-completion **96.15 → 58.81 s (1.63×)**. Cache-hit requests increased **19 → 707/1,024**; actual prefill fell **436,414 → 192,593 tokens**, median restore ~**0.70 ms**. Both runs completed naturally; output totals were 21,016/20,965.

With shared-prefix planning, 790/1,024 translations were identical, chrF++ **22.9454 → 22.8491**, BLEU **33.1588 → 33.0351**. A 1,000-sample paired bootstrap at seed 20261002 found BLEU/chrF++ p-values **0.129/0.139**, not a statistically detected metric change in this sample—not proof of exact equivalence. Changed checkpoints/chunks can alter rounding and subsequent wording. Partial checkpoint restoration at 32/384/394/400/401 tokens matched same-chunk cold execution for subsequent tokens and all FP32 GDN, convolution, target/draft KV, and positions.

Tests also covered uploads over 256 KiB with caller buffer release/overwrite and staging reuse; cancellation/isolation, fixed-history probabilities, asynchronous writer errors, resume without loading GPU when complete, interrupted/corrupt-tail resume, HTTP/SSE, and MTP/schema/logprobs/stop. A separate shared-workspace KV-growth fix for Flash Next passed dual-request 512/2K matched-history KL/NLL 0, with 27B and Flash variants naturally completing. A final 1,024-record release-binary run generated 21,144 tokens in **35.40 s**, with ready-state processing ~31.0 s versus ~31.3 s previously.

## CPU sampling preprocessing (2026-10-17)

Without penalties or logit bias, CPU preprocessing now converts FP32→FP64 with addition of zero to preserve signed-zero semantics; full-input finiteness and grammar masking remain. The full sampling CDF, bitwise scores, and seeded selections matched the generic path. GPU kernels/weights/native package were unchanged. On synthetic 248,320-logit top-20 sampling with 8K history, CPU median fell **1.585 → 1.276 ms (-19.5%)**.

Real Rust SSE requests: 617 input tokens, 64 output budget, max three MTP drafts, medium thinking with 32-token cap, temperature 0.7/top-k 20/top-p 0.8/seed 20261002, `decode_only`, full expert residency; one warmup, three full-prefix-hit runs per case. These are whole-HTTP TPS medians.

| Case | Before TPS | After TPS | Change |
| --- | ---: | ---: | ---: |
| Flash Next, no logprobs | 37.14 | 37.63 | +1.32% |
| Flash Next, top-three logprobs | 33.83 | 33.68 | -0.46% |
| 27B, no logprobs | 23.73 | 23.96 | +1.00% |
| 27B, top-three logprobs | 22.11 | 22.31 | +0.90% |

End-to-end effects were ~1%; CPU microbench speedup is **not** decode speedup. All 16 paired API responses matched content, reasoning, usage, and finish reason; penalty/JSON schema checks and `make check` passed.

## Rust Graph hits and probability reporting (2026-10-17)

Batch Graph hits now reuse bound kernel modules without redundant address-plan expansion/registration checks. COW, live-workspace saves, arena invalidation, and indirect Graph address validation still occur. Ordinary, verify, draft, warm, and mixed use the shared executor. Both sides compared identical weights/native libraries/cubins.

Logprob reporting shares history/penalty/bias/grammar preprocessing with sampling; a linear top-N selection sorts only N values and normalizes over the full vocabulary. Sampling CDF, randomness, and MTP acceptance remain unchanged. Tiny but nonzero tail probabilities no longer disappear merely through cumulative rounding; genuinely underflowed or masked probabilities keep documented floors.

Rust SSE request setup: 617 input tokens, max three drafts, medium/32-token reasoning cap, seed 20261002, `decode_only`, fully resident experts, one warmup and three full-prefix-hit runs. Greedy output budget 128, non-greedy temperature 0.7/top-k 20/top-p 0.8 and budget 64. The figures are HTTP TPS medians.

| Case | Before TPS | After TPS | Change |
| --- | ---: | ---: | ---: |
| Flash Next greedy, 1 request | 46.92 | 47.73 | +1.72% |
| Flash Next greedy, 4 requests | 75.25 | 74.20 | -1.40% |
| Flash Next greedy, 8 requests | 112.68 | 114.76 | +1.85% |
| Flash Next stochastic, no logprobs | 36.24 | 37.63 | +3.84% |
| Flash Next stochastic, top-three logprobs | 22.84 | 33.69 | +47.50% |
| 27B greedy, 1 request | 26.03 | 26.15 | +0.48% |
| 27B greedy, 4 requests | 58.60 | 58.99 | +0.67% |
| 27B greedy, 8 requests | 80.41 | 80.19 | -0.27% |
| 27B stochastic, no logprobs | 23.49 | 23.67 | +0.78% |
| 27B stochastic, top-three logprobs | 17.81 | 22.07 | +23.89% |

Ordinary greedy MTP varied by about ±2%, not evidence of a general decode gain. Gains concentrated on probability reporting. In a separate 248,320-logit/8K-history CPU test, top-20 sample plus top-three logprobs improved **18.75 → 4.80 ms (-74%)**; not model TPS.

Four-way MTP Nsight Systems trace (last 100 synchronized packet Graphs, excluding shared-fragment/direct-kernel intervals) found host gaps from previous stream synchronization to Graph submission at median **1.051 → 0.856 ms**, P95 **1.160 → 1.189 ms**. For packets with at least 10 ms GPU time, host gap improved only **1.083 → 1.022 ms**. These host gaps include prep and control transfers; they are not GPU acceleration measurements.

All 90 warm paired API responses matched content/reasoning/usage/finish reason out of 120 pairs; five cold pairs differed in content under nondeterministic scheduling while other fields matched. Fixed-history 27B (24 cases) and Flash Next (16 cases) target/mixed comparisons had zero KL and selected-token NLL changes. Additional complete-state recovery, direct/capture/warm-replay, draft/commit, penalty/mask/bias/tie/top-k tests passed. Graph off/full prefix, sampling, MTP, logprobs, JSON, image, and multi-image checks passed; Rust format, Clippy, and tests passed.

## Rust input transfers and control readback (2026-10-17)

Small inputs are uploaded asynchronously from session-owned bounded pinned slabs, consumed in stream order. Slots cannot be reused until DMA completes; KV growth, copy-on-write, and request switching retain synchronization. Control values are read back in batches; Flash Next CPU embedding/PLE returns immutable shared rows. Pinned-pool capacity is about 4 MiB on demand; this test version retained bounded synchronous transfers for inputs above 256 KiB (a later JSONL improvement chunks larger transfers as described above). Rebuilt Flash Next native library used unchanged weights, kernels, and exported plan.

Real SSE: 617 input tokens, 96 output budget, medium thinking/32-token cap, greedy, seed 20261002, `decode_only`, fully resident experts. Each concurrency setting had one warmup and two full-prefix-hit runs, taking the median of complete HTTP output TPS.

| Model | Maximum MTP drafts | 1 request (before → after) | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash Next | 0 | 27.12 → 26.90 | 46.32 → 46.93 | 77.64 → 79.04 | 120.67 → 123.32 |
| Flash Next | 3 | 46.92 → 47.56 | 43.54 → 44.40 | 74.49 → 75.98 | 110.73 → 112.95 |
| 27B | 0 | 10.89 → 10.93 | 20.40 → 20.57 | 37.43 → 37.66 | 60.86 → 63.68 |
| 27B | 3 | 24.64 → 24.78 | 44.88 → 46.40 | 55.00 → 52.92 | 80.62 → 83.00 |

Longer 256-token reverse-order comparisons found Flash Next 1-request no-MTP **27.15 → 27.11 TPS (-0.14%)**, and 27B four-way MTP **60.17 → 61.96 TPS (+2.98%)**, showing that short-run dips could be measurement noise. Single-stream no-MTP Nsight warm Graph host gaps fell **0.977 → 0.751 ms**; 128 CLI output tokens matched, but the trace is not an end-to-end TPS result.

Of 180 paired 96-token API responses, all 120 warm pairs matched; two cold 27B eight-way responses differed. Cold/warm 256-token 27B concurrency also differed on the old baseline itself, so whole-output reproduction under changing schedules is not guaranteed. Fixed-history target/verify/mixed comparisons (27B 24, Flash Next 16) had zero maximum KL and selected-token NLL difference. Graph off/full prefix, penalties, non-greedy MTP, logprobs, JSON, and image cases passed.

## 27B layer-wise prefill and batched MTP (2026-10-17)

27B and Flash Next share layer registration, dynamic rows, private address bindings, and runtime scheduling. Prefill pauses at model-declared layer boundaries without changing real block size; adjacent layers use measured costs and remaining budget rather than fixed group counts. Independent 1–128-request GDN/attention wraps one-row compute in address tables. Ordinary/verification/mixed packets reuse shared target compute and explicit role/state ownership. Batch draft, commit, and warm use existing components.

Compared same official beam-8 E8P Q4A8 weights on Rust release `serve`/real SSE Chat, max four MTP drafts, `decode_only`, thinking off, greedy seed 20261002. Fixed code tasks were padded to 512/2K/8K inputs with 128-token output budget. Two full-HTTP rounds per case include queues/prefill/generation. Cold = prefix cache disabled; hit = full input prefix restored, **not cold/warm service startup**. Baseline had the `ff8ca0e` KV-prefetch package and prerefactor scheduler.

| Scenario | Baseline aggregate TPS | Current aggregate TPS |
| --- | ---: | ---: |
| Cold 2 requests, 512 + 8K | 9.07–9.21 | **9.81–9.97** |
| Cold 4, 2×512 + 2K + 8K | 11.37–11.76 | **14.63–15.03** |
| Cold 8, 2×512 + 3×2K + 3×8K | 9.78–10.07 | **11.51–11.79** |
| Full hit 4, same 4-request pattern | 32.77–33.87 | **58.91–64.67** |
| Full hit 8, same 8-request pattern | 49.21–50.09 | **69.31–90.81** |

Worst cold-request TTFT dropped **38.17–39.87 → 28.82–29.27 s** at four-way and **91.84–95.38 → 77.59–80.51 s** at eight-way. With an 8K cold prompt arriving 200 ms before a 512-token prompt, short-request queue/init wait fell **6.176 → 0.022 s** and TTFT **6.947 → 1.616 s**. If the 8K prompt was fully cached, short cold TTFT was **2.293 s** versus 2.374 s. SSE chunks may combine MTP tokens; chunk gaps are not per-token ITL.

A separate 16-way cold test (8×512 + 8×2K) **regressed** from **21.35 → 20.67 TPS**, and worst TTFT worsened **64.86 → 78.75 s**. Existing cold-admission restrictions were unchanged; the refactor does not universally improve concurrency.

The package grew from 173 to **181 distinct cubins** (eight added private-state dynamic implementations), not per-batch enumerations. 409 logical programs and 5,326 local bindings include shared segments, layer args, and 64 prefill stages; they are not distinct cubins. Each request adds 2,208,512 bytes of private scratch; the global address table adds 917,504 bytes, without resident expanded weights.

Adjacent-process `run-model` comparisons with MTP/prefix off, warmup then two single-token runs, found prefill **678.00/689.51/618.11 → 678.94/689.63/617.53 TPS** at 512/2K/8K (within 0.15%). Normal 64-token decode averaged **10.74 TPS**; four-draft MTP fixed-prefix fixtures ~**24.8 TPS**. These forced-length checks are not natural coding throughput.

Ordinary B2/B3/B4/B8 and 24 matched-history verification/mixed cases (width 2/4/8 × requests 1/2/4/8) had zero KL and selected-token NLL differences. Layer pause at 4K, time-budget continuation, workspace alternation, cancellation/reuse, and partial prefix restoration preserved all recurrent state. Rebuilt Flash Next contracts matched the original package; `make check` and `make check-offline` passed. **Whole-model Compute Sanitizer was unavailable because GPU debugging was disabled; no memcheck pass is claimed.** Final packaged-model validation/load and 166 teacher-forced probability points matched the earlier quantized path; BF16 top-1 agreement stayed **98.19%**, all choices in BF16 top-3, all **150/150** high-confidence points matching.

## Flash Next short-request regression (2026-10-15)

Under full expert residency, group-64 INT8 KV, and `decode_only`, the shared dynamic-row ordinary-decode template delivered aggregate TPS at 1/2/4/8/33 concurrent requests of **26.08/43.33/73.82/112.51/167.14**, each within 1% of the preceding package. Two MTP code-generation cases measured **39.09/51.76 TPS** before and **39.04/51.83** after, with identical generated token counts; the first three MTP iterations were excluded.

Three warm 512-token CLI prefills averaged **987.6 TPS** before and **986.3 TPS** after. Consecutive-process load-to-ready was **18.29 versus 18.16 s** without clearing filesystem caches. Template interpretation occurs during loading/planning, not during Graph replay; no measurable inference regression was observed.

## Flash Next full code generation

Text-only model built from an official checkpoint with integer E8P Q2A8 experts, Q4A8 input/output-head projections, W8 output projections/shared FFN, and BF16/FP16/FP32 where precision sensitive. Fully resident experts, group-64 INT8 KV, FP32 recurrent GDN. No images, other concurrent requests, or history-based prompt-lookup drafts.

Full code generation used the Rust `orinfer serve` streaming Chat API, including actual chat template, scheduling, state restoration, MTP, CUDA work, and SSE transfer. Python was a client/timer only.

```bash
./target/release/orinfer serve /path/to/optimized-flash-next \
  --model qwen-flash-next --cuda-graph decode_only \
  --resident-experts full --mtp-drafts 3 \
  --prefix-cache-mib 1024 --memory-reserve-mib 512
```

MTP-off used the same package/settings with `--mtp-drafts 0`. Draft widths were selected adaptively per request, not forced. Requests: `enable_thinking=true`, `reasoning_effort=medium`, `temperature=0`, `seed=20261002`, `thinking_token_budget=16384`, `max_completion_tokens=16384`. Reasoning and code share the 16K budget, with space reserved for reasoning termination and at least one response token.

The task was to implement a safe Rust streaming frame decoder from scratch with a big-endian u32 length prefix, arbitrary input fragments, maximum-frame check, reset, and stdlib unit tests. Unrelated repository helper text varied prompts to 512/2048/8191 input tokens.

For each length/MTP configuration, one initial and one identical full-prefix-hit request were run. Table values are the **single warm full-request run** per cell. Outputs were 6,176–7,598 tokens including thinking and code, all naturally ended with `finish_reason=stop`.

| Input tokens | Generated tokens | MTP-off decode TPS | MTP-on decode TPS | Speedup | MTP reasoning-stage TPS | MTP code-stage TPS | Draft acceptance |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 6,276 | 29.72 | 44.99 | 1.51× | 41.46 | 49.37 | 74.8% |
| 2,048 | 6,176 | 29.64 | 45.21 | 1.53× | 44.02 | 46.62 | 76.4% |
| 8,191 | 7,598 | 29.49 | 45.82 | 1.55× | 43.14 | 49.30 | 77.4% |

Decode TPS is `(completion_tokens - 1) / (total HTTP time - TTFT)`, including Rust scheduling, token transfers, synchronization, and SSE tail handling; only **committed** output tokens count, including thinking—not rejected drafts. TTFT starts with request submission and ends at the first valid reasoning/content fragment. SSE fragments may combine tokens. Stage TPS is approximate using API usage and reasoning/body timing boundaries; it does not treat chunk count as token count.

Full-request decode measured approximately **45 TPS**; code-only stage measured **46.62–49.37 TPS**. This **does not reach 50 TPS for complete generation**, and the code-only rate must not replace the reasoning-inclusive decode measure.

MTP-on and MTP-off matched reasoning, code, token counts, and finish reasons for every corresponding request. Generated code compiled in all three cases. Generated unit tests passed for 512 and 8,191-token prompts. The 2,048-token case had one flawed generated test repeating the first frame-header byte (10 passing, 1 failing); correcting only that test input while leaving the decoder unchanged yielded 11 passes. These are MTP-toggle consistency checks, not independent BF16/FP8 quality acceptance.
