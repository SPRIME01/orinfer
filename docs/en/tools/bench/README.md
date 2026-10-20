# Benchmark tools

[简体中文](../../../../tools/bench/README.md) | **English**

See [Performance reference](../../performance.md) for measured Flash Next Rust CLI/Chat API results and historical 27B results. That reference distinguishes execution-layer prefill, input tokens versus API TTFT, decode excluding TTFT, and full-prefix-hit latency. **Whole-request throughput measured here must not be substituted for those metrics.**

`reference_bench.py` measures prefill/TTFT and decode against an OpenAI-compatible SSE endpoint; `test_reference_bench.py` tests token chunks and timing intervals. Record reference server configuration and identity in `reference-lock.json` in the output directory. Supply explicit input tokens via `--seed-tokens /path/to/input-tokens.json`. This workload requires token-ID input/output, fixed-length generation, and optionally separate timing diagnostics; ordinary Chat requests are not interchangeable.

`run_aot_artifact.sh MANIFEST NEW_OUTPUT` builds the Rust CLI and runs an AOT fixture while recording process and machine identity. Use `tools/model/run.sh` for full-model inference. `sample_machine.py` and `sample_continuous.py` read temperatures, clock speeds, memory, and power without changing global settings. GPU jobs share `artifacts/gpu-experiment.lock`. `validate_plan.py` validates benchmark/quick-quality configurations on CPU; it does not measure performance.

## Code generation, MTP, and translation

`code_generation.py` measures combinations of thinking, MTP, and prefix caching through the same Chat requests. Input uses the `concurrency.py` case format described below:

```bash
python3 -m tools.bench.code_generation --requests code-requests.json \
  --repetitions 3 --output artifacts/code-generation.json
```

Each case gets one warmup followed by repeated requests. Reports retain full answer and reasoning text, usage, first-token/first-content timing, actual cached-token hit rate, and scheduler counter deltas. TPS is **committed completion tokens divided by complete HTTP time**, including reasoning, prefill, and transport. MTP drafts and SSE chunks are not token counts. Empty answers and budget truncation are reported separately; an early-ended response is not valid evidence of complete coding performance or quality. For cold measurements, restart your own server or change the prefix. Keep service and inputs fixed, with exclusive API access during measurement.

Translation workloads can use `--requests fixtures/translation-benchmark.json`; check all three paragraphs, timing information, and identifiers. The fixed-batch `benchmark_flash_decode_template` diagnostic controls length through engine output budget without artificial EOS logit bias, so ordinary greedy decode is not accidentally converted into CPU full-vocabulary sampling. It excludes HTTP and queuing time.

## Concurrency and health statistics

`concurrency.py` measures 1–128 concurrent real SSE requests. Input has the shape `{"cases":[{"id":"case","request":{... Chat API body ...}}]}`:

```bash
python3 tools/bench/concurrency.py --requests requests.json \
  --concurrency 1 2 4 8 16 32 128 --repetitions 3 --output artifacts/concurrency.json
```

Reports include true usage-based output-token counts, whole-round throughput, TTFT percentiles, cached tokens, and per-request results. Throughput includes queuing, prefill, HTTP, and output; it is **not GPU-only decode TPS**. SSE chunk gaps are not necessarily token ITL. Repeated prompts may hit prefix cache, so vary input or disable caching for cold comparisons. `/health.scheduler_statistics` exposes actual prefill compute tokens, engine-committed decode tokens, batch histograms, mixed iterations, and peak active requests.

Health data also covers admission, request startup, prefix restore, prefill completion, batch input preparation/planning/commit, and CUDA Graphs. `batch_execution` reports batch Graph hits/misses, captured kernel/copy/memset node counts, eviction, address-change invalidation, and component timing. Single-request/MTP captures are separately counted under `sequence_captures/sequence_capture_s`.

These counters accumulate **host time**; replay can include synchronization waits, so they are not CUDA-event GPU durations. `request_start_s` includes `prefix_restore_s`, and batch Graph components are already part of `compute_s`. Single-program capture may occur during startup or compute; do not sum overlapping counters as independent time.

The tools take health snapshots before sending requests and after completion, then report `scheduler_delta` and `admission_delta`. Cumulative peaks are not subtracted. Use exclusive server access; counter resets after restart are treated as errors. `admission_statistics` records counts per round, full-cache-hit text requests, admission histograms, and wall time, including policy, memory checks, and startup.

## Joint versus independent prefill

`benchmark_prefill_scheduling` is an ignored real-GPU test comparing joint prefill against large, per-request prefill for the *same model*. Its fixture provides `model`, a new `output` path, `cuda_graph`, `batches` (2–8), `repetitions`, `output_tokens`, and text-only `cases` with `GenerationInput` fields `input_tokens`, `max_new_tokens`, and `sampling`. Use `temperature: 0` and `seed: 20261002`. Stop your own GPU service and hold the experiment lock:

```bash
ORINFER_BATCH_FIXTURE=artifacts/prefill-scheduling-fixture.json \
  flock artifacts/gpu-experiment.lock \
  cargo test --release --offline -p orinfer-engine benchmark_prefill_scheduling -- --ignored --nocapture
```

The test disables MTP and prefix reuse, alternates path order, and separately records admission, aggregate time to prefill completion, per-request TTFT, and subsequent batch decode. Prefill timing includes head and first-token selection; TTFT also includes admission. Decode begins after the cohort has finished prefill, so this does not measure online mixed-scheduling ITL.

Joint paths use package-declared chunks; independent paths use their largest valid per-request chunk. Different blocks may choose different quantization codebooks. Token differences are diagnostic, not an independent BF16/FP8 quality result. Each request also checks final positions and KV reclamation.

## Mixed warm/cold requests

`mixed.py` warms one text `anchor`, launches a specified number of prefix-hit anchors, and injects a cold request after each stream has delivered a set number of valid output chunks.

Input shape:

```json
{
  "anchor": {"model":"MODEL_ID","messages":[{"role":"user","content":"..."}]},
  "cold_cases": [{"id":"cold","anchor_count":2,"request":{},
    "expected_prompt_tokens":8192,"target_text":"CODE-641"}]
}
```

Set `temperature: 0` and `seed: 20261002` on all requests. Anchors must generate long enough to remain active. Use a distinct identifier at the beginning of each cold prompt to prevent cross-round cache hits. `anchor_count` restricts which concurrency counts a case runs under (omitted means all). Expected prompt tokens and target text are optional.

```bash
python3 -m tools.bench.mixed --requests mixed-requests.json \
  --anchor-counts 1 2 4 8 --trigger-chunks 4 --output artifacts/mixed.json
```

Reports include no-injection controls, cold-request TTFT, anchor output throughput, overlapping SSE chunk intervals during long prefill, and engine-counter deltas. The boundary-crossing stall is included; chunks are not token ITL. Tests require full prefix hits for anchors and no more than 5% cold-prompt cache hits, verified through admission counters, with no other traffic. The server's actual adaptive MTP settings apply; run separate pure-decode/disabled-MTP baselines.

## Heterogeneous delayed-arrival acceptance

`acceptance.py` validates delayed arrival of short/long text, images, multiple images, and multiround branching. JSON supports `cases: [{id, request, expected_words}]`, optional `warmup: [id]`, and `rounds: [{id, jobs: [{case, delay_s}]}]` with at most 128 requests submitted per round.

Answer checks are limited to unambiguous labels/short answers and ignore case/separators. Concurrent throughput comes from actual usage. Results retain scheduler/admission deltas, active/queued peaks, and per-request outcomes. Incomplete SSE and incorrect expected answers fail the run.

```bash
python3 -m tools.bench.acceptance --requests acceptance-requests.json \
  --output artifacts/acceptance.json
```

## Flash Next GPU packet tests

`validate_flash_decode_batches` accepts the same `ORINFER_BATCH_FIXTURE` shape and checks 2/3/4/8/16/32 requests against matched-history logits/top-3/NLL, images, ordered multi-image inputs, cancellation/slot reuse, and continued MTP after target batching. It requires a native Flash batch package and exclusive GPU lock:

```bash
ORINFER_BATCH_FIXTURE=/absolute/path/to/flash-fixture.json \
  flock artifacts/gpu-experiment.lock \
  cargo test --release --offline -p orinfer-engine validate_flash_decode_batches -- --ignored --nocapture
```

This disables prefix cache and compares identical weights, cold prefill, and teacher-forced histories against single-sequence execution. It establishes arithmetic/state isolation in the batch path, not original BF16/FP8 quantization quality. Use real Chat/SSE `concurrency.py` and `acceptance.py` for functional throughput/cache acceptance.

`validate_flash_heterogeneous_scheduling` uses the same fixture. The first case must contain at least 4,096 input tokens; the second must be short. It tests two cold prefills and mixing an already decoding short request with a long prefill. It verifies large native chunks for standalone long requests, bounded chunk sizes with active decoders, continued decode progress, and KL/NLL/top-3 under identical chunking and teacher-forced history. Independent large-chunk versus interrupted-chunk first-token differences are reported separately and are not state-isolation or BF16-quality verdicts. `validate_flash_bounded_prefill` accepts either cost-selected joint or independent execution instead of requiring every iteration to be mixed. Run under the GPU lock.

`validate_flash_long_generation_reuse` uses the same fixture with a stress prompt and a follow-up prompt; the latter must include a native 4,096-token prefill. The test first captures cold follow-up logits and full committed state, then forces 16,384 decode tokens (including MTP), caches generated prefixes, reuses slots, and compares cold/fully cached follow-ups against reference chunking, logits, and KV/GDN/convolution/PLE/MTP state. Engine tests intentionally continue beyond EOS to reach stress length; separately check task correctness via real Chat. Requires the GPU lock.

## Scheduled mixed-arrival trace

Use `python -m tools.bench.scheduling --requests TRACE.json --output RESULT.json` to replay a fixed arrival schedule (up to 128 requests). Input example:

```json
{"requests":[{"id":"short-1","at_s":0.1,"request":{"model":"MODEL_ID","messages":[{"role":"user","content":"Hello"}],"max_tokens":16}}]}
```

Arrivals are scheduled independently of prior request completion; multiple jobs may arrive together. Reports include arrival-time error, TTFT, whole-request time, and largest SSE chunk gap at P50/P95/P99, failure count, valid output TPS, successful-request prefix hit ratios (requests and tokens), and scheduler statistics. Chunk gaps must not be treated as token ITL. Compare scheduler versions using the same trace, package, cache/MTP/Graph settings, and both cold and warmed caches.

## Vision-stage profiling

`profile_vision_stages` is an ignored real-model diagnostic using a fixture with `model`, a new `output` path, and image-bearing `GenerationInput` cases. It binds actual image grids, measures end-to-end warm vision Graphs and per-operator GPU latency, and optionally adds NVTX ranges to launches with `ncu: true`. Image decoding, resizing, feature copying, and text generation are **excluded**. Measure complete image translation separately via Chat API.

```bash
ORINFER_BATCH_FIXTURE=/absolute/path/to/vision-fixture.json \
  flock artifacts/gpu-experiment.lock \
  cargo test --release -p orinfer-engine profile_vision_stages -- --ignored --nocapture --test-threads=1
```
