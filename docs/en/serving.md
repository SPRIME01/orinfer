# Serving guide

[简体中文](../serving.md) | **English**

## Startup and model loading

```bash
./target/release/orinfer validate-model /path/to/prepared-model
./target/release/orinfer serve /path/to/prepared-model \
  --model qwen3.8-27b --listen 0.0.0.0:8088 --cuda-graph decode_only
```

The service loads the model once and processes requests through one GPU worker. The CPU uses `tokenizer.json`, `chat_template.jinja`, and `generation_config.json` from the model directory to render and tokenize requests. There is no online Python, quantization, or kernel JIT.

Model weights are read in parallel into bounded pinned buffers while uploads run asynchronously. `--load-workers N` allows 1–32 readers; the default uses available CPU cores, capped at four. Staging memory is limited to 256 MiB and released after loading. `--verify-weights` performs more expensive full content checks at startup.

All execution-package kernel assets are still hash-verified. CUDA modules for ordinary text, configured MTP, vision, and small-batch plans are preloaded. Others are loaded on first use and bound before Graph capture, so uncommon shapes may have first-use overhead.

During model preparation, frontend file SHA256 digests are pinned. If modifying them intentionally, confirm their token mapping and semantics match the checkpoint, then rerun `python3 tools/model/package.py pin-assets MODEL_DIR` (which does not change weights). Model files must remain unchanged while loading.

Execution packages are resolved first from the shared cache by digest, then from `cache/packages/<digest>` inside the model. Set `ORINFER_EXECUTION_CACHE` to override the default `$XDG_CACHE_HOME/orinfer/packages` (or `~/.cache/orinfer/packages` without XDG). A present but corrupt package is an error. Custom safetensors layouts retain `orin.layout.<tensor>` markers; the execution-package digest pins native logic and kernels.

### Expert residency

`serve`, `run-model`, and `score-model` accept `--resident-experts full|N`, defaulting to `full`. In Flash Next, limited residency requires an execution package declaring expert paging. Older libraries must be reattached using matching native model tools; weights and cubins do not change.

`N` is equivalent per-layer capacity. The 48 backbone layers share a global SLRU cache of up to `48 × N` experts without fixed per-layer quotas. Valid values are 1–512; 512 is effectively full residency. MTP experts remain fully resident. Missing experts are loaded on demand from compressed model-file weights/scales and mapped into stable CUDA virtual addresses; expanded INT8 weights are not permanently cached.

```bash
./target/release/orinfer serve /path/to/prepared-flash-model \
  --model qwen-flash-next --resident-experts 256
```

Limited mode performs CPU scheduling and SSD reads after routing and automatically selects `cuda_graph=off`. Full residency retains the chosen Graph mode. Capacity is counted in experts, not strict bytes; CUDA allocation granularity and shared pages add overhead. If one prefill or batch routes to more experts than the limit, extra experts are mapped temporarily and later released. Smaller caches increase I/O and may reduce throughput. Keep model files readable and unchanged throughout serving.

### Default request parameters

`--default-request-params` accepts a JSON object supplying Chat fields omitted by clients:

```bash
./target/release/orinfer serve /path/to/prepared-model \
  --default-request-params '{"enable_thinking":true,"temperature":0.7}'
```

Without this option, ordinary API defaults apply. Explicit request values—including `false`, empty arrays, and `null`—override deployment defaults. An explicit `null` clears the deployment default and then follows that field's normal null handling. Objects replace objects; there is no recursive merge. Clients must still supply `model` and `messages`, which cannot be deployment defaults. Default fields and combinations are validated before the GPU model loads.

The pairs `enable_thinking`/`reasoning_effort` and `max_tokens`/`max_completion_tokens` override as groups: supplying either request field suppresses the group's deployment default. Therefore `enable_thinking=false` or `reasoning_effort=none` disables an enabled deployment default. Conflicts *within* the request return 400. Explicit `stream=false/null` suppresses default `stream_options`; explicit `logprobs=false/null` suppresses default `top_logprobs`. Inspect `/health.default_request_params`.

## Ollama API

Running `orinfer serve` without a single model path also exposes native Ollama APIs at `http://127.0.0.1:8088`:

- `GET /api/version`, `GET /api/tags`, `POST /api/show`, `GET /api/ps`: version, registered models, model details, and loaded models; discovery is CPU-only.
- `POST /api/chat` and `POST /api/generate`: Chat and generation using native chat templates, streaming NDJSON (`application/x-ndjson`) by default; `stream=false` returns one JSON object.
- `POST /api/copy` and `DELETE /api/delete`: alias copying and deregistration after active requests finish. Imported weight files remain at their user-owned paths.

```bash
curl http://127.0.0.1:8088/api/chat \
  -d '{"model":"index-translate-2b","messages":[{"role":"user","content":"Translate to English: 你好"}],"think":false,"options":{"num_predict":256},"keep_alive":"60s"}'
```

Supports Boolean or named-level `think`, base64 PNG/JPEG/WebP image messages, tools and tool-result histories, `format="json"` or JSON Schema, and `logprobs/top_logprobs`. `options` supports temperature, top-p, top-k, seed, stop, presence/frequency/repetition penalties and positive `num_predict`. Omitted fields inherit deployment defaults; explicit `think=false` overrides default reasoning. Tool arguments are complete JSON objects, emitted only after full parse when streaming; incomplete invalid arguments return errors.

`keep_alive` accepts numeric seconds, durations such as `"1h30m"`, negative indefinite retention, or `0` to unload after completion. Empty messages/prompts preload; an empty request with `keep_alive=0` unloads. Names without tags accept `:latest`. Both APIs share admission, continuous scheduling, prefix caching and cancellation.

Different models may run simultaneously when memory permits, including simultaneous cold starts; each model has only one worker. Under memory pressure, the manager evicts least-recently-used idle models, including models requested with `keep_alive=-1`. It waits only if active requests/in-flight loads still hold necessary memory, and does not interrupt active generation. Each in-flight load reserves memory until ready or failed.

Responses report actual tokens, cache counts and total duration in nanoseconds, without manufacturing unsupported stage timings or physical VRAM figures. `/api/ps.estimated_load_bytes` is an admission estimate. `ORINFER_API_KEY` protects both `/api` and `/v1`.

Import models through the CLI as native Orinfer packages. Ollama registry pull/push/create, embedding, raw prompt, suffix, Go-template overrides, legacy token context, and automatic truncation/sliding windows are unsupported and return explicit errors. Context is fixed by the execution package; `options.num_ctx` may be absent or equal that size.

## Chat API

Available routes: `GET /health`, `GET /v1/models`, and `POST /v1/chat/completions`. With `ORINFER_API_KEY` set, `/v1` requires `Authorization: Bearer <key>`.

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"What is 37×24 minus 19×13?"}],"temperature":0,"max_tokens":512,"enable_thinking":true,"stream":true,"stream_options":{"include_usage":true}}'
```

Supported sampling fields include `temperature`, `top_p`, `top_k`, presence/frequency/repetition penalties, signed 64-bit `seed`, `logit_bias`, EOS, and `stop`. Temperature, top-p, top-k, and repetition penalty default to the model's generation configuration; presence/frequency penalties default to zero. The default seed is 20261002. `logit_bias` keys are tokenizer token IDs and values range from -100 to 100.

Thinking is off by default. `enable_thinking=true` or non-`none` `reasoning_effort` enables it; reasoning is returned in the extension field `reasoning_content`. `none` disables it, `minimal/low` map to the checkpoint's `low`, `medium` remains `medium`, and `high/xhigh/max` map to `xhigh`. Explicit `enable_thinking=false` disables thinking. Native model templates require a user message; place system/developer instructions first.

`thinking_token_budget` imposes a hard limit on the *initial* reasoning segment, including 0. At the limit, decoding forces `</think>` before continuing the answer or tool calls. That terminator counts as a completion token, not a reasoning token. The cap shrinks automatically to total output budget minus two, reserving the terminator and at least one response token. When thinking is enabled with this parameter, total output budget must be at least two. The limit does not apply when thinking is disabled. It is per-request state shared by MTP target and drafts. Unpenalized greedy requests without probability reporting may reuse GPU argmax under constraints; forced single-token termination needs no CPU logits download. Other constrained paths have CPU fallbacks, and sampling preserves the target distribution. Effort expresses model preference and is not a substitute for the hard cap.

The default output budget is 8,192 tokens, reduced to fit remaining context. Specify **either** `max_tokens` or `max_completion_tokens`, never both. Prompt, history, images, reasoning, and output all count toward package context capacity. Explicit budgets beyond capacity return HTTP 400 before admission. Request bodies are limited to 32 MiB. Context overflow uses `error.code=context_length_exceeded` so clients can compact history and retry; other invalid parameters use `invalid_request`.

`stream` defaults to false. True streams SSE deltas followed by `[DONE]`; `stream=null` uses default behavior, and `tools=null` means no tools. With `stream_options.include_usage=true`, regular chunks have `usage:null` and a final `choices:[]` chunk carries complete usage, including reasoning and cached prompt tokens. Disconnects cancel generation. Error streams report errors; partial output is not presented as completed success.

`logprobs=true` reports content-token `token/logprob/bytes/top_logprobs` (0–20 alternatives). SSE attaches probabilities to content deltas. Probabilities come from the target model after penalties/bias/constraints but **before** top-k/top-p truncation. Greedy `temperature=0` probabilities use a T=1 softmax. `bytes` reconstruct UTF-8 characters spanning token boundaries. Reasoning, tool protocol markers, and tool arguments are excluded from content probabilities. Logprob reporting downloads logits to CPU and adds overhead.

Non-streaming logprobs reserve bounded CPU memory based on output budget and count against `--preprocess-memory-mib` until the HTTP body is complete. Excessively large single requests are rejected with 400 before preparation. Streaming sends probabilities incrementally; slow-client buffering remains bounded.

Invalid requests return HTTP 400 with `error.message/type/param/code` and `x-request-id`. `/health.chat_capabilities` reports current support. `store=false`, `metadata`, `safety_identifier`, and `service_tier=auto/default` are accepted without implementing cloud completion storage, moderation, or service-tier scheduling. `store=true`, `n>1`, audio, video, and unknown fields are rejected.

### JSON output

`response_format.type` accepts `text`, `json_object`, or `json_schema`. JSON modes use Rust llguidance for token-level constraints. Supported schema features include nested objects/arrays, types, required properties, enum/const, and compilable local `$ref`. External references, excessive depth/size, and unguaranteeable constraints fail during request preparation with 400.

```json
{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Return the result of 2+3 as JSON"}],
  "response_format": {"type": "json_schema", "json_schema": {
    "name": "answer", "strict": true,
    "schema": {"type": "object", "properties": {"answer": {"type": "integer"}},
      "required": ["answer"], "additionalProperties": false}
  }},
  "max_completion_tokens": 128
}
```

Thinking can be combined with JSON output; the answer remains constrained. MTP target and drafts share the parser constraints, and rejected drafts do not advance parser state. If the budget ends first, `finish_reason=length` may accompany incomplete JSON. Always inspect the finish reason. Constrained output does not allow `stop` because it could truncate JSON or tool arguments.

### Images and multi-image input

The 27B model and Flash Next packages with vision encoders support images. User messages may interleave text with ordered images. Sources: PNG, JPEG, WebP, HTTP(S), or base64 data URIs. Images in history are re-encoded.

```json
{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Compare these two images"},
    {"type": "image_url", "image_url": {"url": "https://example.com/first.png"}},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  ]}],
  "max_tokens": 512
}
```

`detail=low` uses roughly a 256×256 pixel budget; `auto/high` use the smaller of the checkpoint budget and package vision capacity. Preprocessing preserves aspect ratio and aligns dimensions to 32 pixels; each 32×32 region occupies one prompt token. Compressed images are capped at 24 MiB apiece; URL reads time out after 20 seconds. Patch, feature, or total-context overflow returns 400. Sending images to a text-only model also returns 400.

Each image runs separate bidirectional vision attention; merger output is injected at its image tokens. Text uses interleaved MRoPE. Vision encoding executes each time, even when the corresponding processed text prefix is cached. Video and audio are unsupported. See [Vision tools](tools/vision/README.md).

### Tool calling

Supports `tools`, `tool_choice`, and `parallel_tool_calls`. Native XML tool calls are converted to standard `tool_calls` with JSON-string arguments. Clients execute tools and send `role=tool` messages containing matching `tool_call_id` values and the conversation history.

Streaming first emits index/id/type/name, then incremental `function.arguments`; clients concatenate arguments by index. `strict=true`, `tool_choice=required`, a named tool choice, or `parallel_tool_calls=false` enables constrained decoding for names, schemas, and call counts. Required forces at least one call, named choice allows only that tool, and disabled parallel calls permit at most one. Internally, constrained calls use a JSON envelope; public API and history remain standard `tool_calls`.

Unconstrained `auto` calls retain the native XML protocol. Unknown functions or missing/wrongly typed arguments are returned to the client as calls; the client validates them and can send an error via `role=tool` so the model may retry. The server never runs tools. XML argument values are converted using declared types; unconvertible values remain strings. Output too malformed to identify a function is an error. On output-budget exhaustion, `finish_reason=length` preserves the partial call prefix in streaming and non-streaming modes: **do not execute truncated calls**. When historical arguments are not valid JSON objects, the native template preserves the original string as `__orinfer_raw_arguments` instead of guessing corrections.

### OpenCode and Pi

The [OpenCode example](../../examples/opencode.json) uses `@ai-sdk/openai-compatible`, a default coding agent, automatic compaction, image/tool/thinking support, and `reasoning_content` history passthrough without an agent-step limit. Copy to project `opencode.json` and use the model ID configured with `--model`:

```bash
opencode run --model orinfer/qwen-flash-next 'Inspect and fix failing tests in this project'
opencode run --model orinfer/qwen-flash-next --variant low 'Inspect and fix failing tests in this project'
```

Thinking defaults off; low/medium/high impose 512/1024/2048 reasoning-token caps within an 8,192-token total output cap. Tool permissions retain OpenCode defaults.

The [Pi example](../../examples/pi-models.json) targets `earendil-works/pi`. Merge the provider into Pi's `models.json` (default `~/.pi/agent`). It uses `openai-completions` and allows model/thinking selection with `/model` and `/thinking`:

```bash
pi --provider orinfer --model qwen-flash-next --thinking low
```

`orinfer-local` in the example is a placeholder API key for an unauthenticated local server; replace it with the real secret or environment reference if authentication is enabled. Pi sends its thinking budget via `thinking_token_budget`; adjust `thinkingBudgets` in Pi settings. Examples do not claim a fixed cache lifetime or enable long retention. User attachments and client read tools supply images. Model IDs, context windows, and output limits must match the actual served package.

## Execution and caching

### CUDA Graph

`serve`, `run-model`, and `score-model` support `--cuda-graph decode_only|full|off`:

- `decode_only` (default): Graphs for ordinary decode, MTP drafting/verification/restoration, and short refreshes; text prefill, vision encoding, and initial MTP warmup launch directly.
- `full`: capture and use Graphs for all execution plans.
- `off`: submit equivalent kernels, copies, and memset operations directly.

Prefill tails do not use Graphs in `decode_only`, even if they reuse a decode plan. Address/workspace changes invalidate or recapture affected Graphs.

Per-request Graphs are captured on first program use, not eagerly at model load. Batch Graphs are captured after member configurations recur; common power-of-two batch sizes are cached with count/node limits. Transient batches and plans exceeding node budgets run directly. Under admission memory pressure, idle Graphs are evicted before idle request state. First capture time belongs to request warmup.

After a short request, each KV buffer may retain up to one CUDA allocation-granularity page, cleared for reuse in that slot. Larger mappings are released; idle pages are also freed under pressure. Admission charges only newly needed KV pages; an idle slot's pages do not offset future memory needs of other active requests. New requests prefer idle slots requiring the least new allocation, and admission is recalculated after reclamation. With no active requests, transient memory shortages retry in the queue for up to two seconds; continued shortages return 503 `service_unavailable` without blocking other admissible work.

Flash Next supports 2,048/4,096-token base-model prefill blocks and 512-token MTP warmup blocks; `full` can Graph both. Indexing and temporary scoring advance with active context; maximum cache context remains 256K and Graph virtual addresses remain stable.

### Multi-token prediction (MTP)

Packages with native `mtp` plans enable MTP automatically for supported text, image, thinking, tools, and sampling requests. Unpenalized greedy decoding uses GPU top-1. Random sampling uses the same target/draft sampling transformations, accepts with `min(1,p/q)`, and samples from the normalized positive residual `(p-q)+` on rejection to preserve the target-model distribution.

Optional `--prompt-lookup` (off by default) drafts continuations from the request's own token history. It may help code editing with large preserved inputs or repeated templates but can slow ordinary generation. It is considered only for unpenalized greedy output without requested logprobs; every candidate is verified by the target. Bounded lookup windows, draft limits, and other requests' time budgets control verification width. Prolonged low acceptance temporarily returns to model drafts. Random sampling uses native MTP.

`--mtp-drafts 0` excludes draft/verification plans and their exclusive weights at load time while retaining base-model, vision, and batch state machinery. Re-enabling MTP loads the unchanged model's draft weights normally.

On rejection, per-request GDN, convolution, positions, and live KV state are restored. A fixed seed controls the sampling random stream, but adaptive verification length follows measured costs, so output is **not guaranteed to match token-by-token** across MTP versus ordinary decode. Both preserve the target distribution. When fewer output tokens remain than a verification block, ordinary decode handles the tail.

Batch-capable Flash packages let 2–8 decoders share draft projection, target verification, accepted-state commit, and grouped warmup of shifted tokens over real causal segments. Each request drafts again only at its own endpoint; convolution, GDN, QSA, positions, sampling, and constraint parsers remain private. Other concurrency, mixed prefill, and output tails use ordinary target batching. Packages without batch MTP continue per-request selection.

Draft width is selected within the CLI limit using measured whole-iteration cost and committed tokens, including draft, verify, commit, and warmup. Multi-request scheduling first measures ordinary target-batch cost. Low-benefit conclusions carry across context ranges and are periodically re-probed; warmup/capture/module-load costs are excluded from stable estimates. Exploration is bounded by other requests' decode budgets. Benefits depend on acceptance and concurrency; short requests may finish before tuning stabilizes. See [Model tools](tools/model/README.md).

### Prefix cache

`--prefix-cache-mib` defaults to 12,288 MiB; zero disables reuse. The cache grows on demand rather than allocating its budget at startup. Checkpoints preserve complete KV, FP32 GDN, convolution, positions, and MTP state. A Rust radix tree matches exact token/image identity and selects reusable endpoints by measured prefill and restore costs.

KV intervals are immutable and shared by reference; recurrent state is stored separately. Fully aligned KV intervals share physical memory via CUDA VMM while preserving each request's virtual Graph addresses. Boundary fragments and GDN/convolution/PLE/incomplete compression state are copied per request. Shared intervals are read-only and copy-on-write; shared mappings are removed before arena reuse. Creating a checkpoint still saves prefix data. Short prefixes can be skipped if they interfere with large-block execution.

Final prompts, periodic and cost-qualified branch checkpoints, and completed-generation prefixes may be saved. A generated-output prefix retains its longest reusable prompt; if budget is inadequate, its snapshot is skipped. Normal eviction may still remove prompts. Reloading a model clears the cache.

Hits skip corresponding text-backbone compute; vision encoding, uncached prompt tails, and generation still run. `usage.prompt_tokens_details.cached_tokens` reports actual restored tokens (request SSE usage to see it). VMM avoids copying aligned portions but still requires boundary/recurrent restoration. Full-prompt hits near maximum context are not guaranteed; memory pressure can evict or skip snapshots.

`prompt_cache_key` (1–64 characters) and `prompt_cache_retention=in-memory|24h` are accepted as routing/retention preferences. With one worker, the key neither partitions the exact-identity cache nor changes output. This is an evictable **memory cache**, not guaranteed 24-hour retention or SSD persistence; inspect `/health.chat_capabilities.prompt_cache`. Automatic reuse also works without these request fields.

## Concurrency, context, and resources

Continuous batching is active when the package supplies the required operators; otherwise requests execute serially. Check `/health.continuous_batching`. Requests own isolated KV, GDN, convolution, position, vision, and sampling state, while weights and execution workspaces are shared.

Cold single-request prefill uses large blocks. Compatible short requests may combine prefill. With active decoders, smaller chunks are selected by predicted cost. Packages may specialize common batch sizes; dynamic AOT rows support other real sizes from 2–128 without online compilation or padding fake requests. Requests enter/leave at iteration boundaries, without waiting solely to fill a batch.

Flash Next batching shares projections, MoE routing/experts, and output head; GDN, convolution, PLE, QSA/KV/index/pending, and MTP remain request-private. Common capacities are 2/4/8/16/32/64/128, with dynamic shapes for other sizes. Mixed-capable packages combine real decode rows with prefill fragments for shared HC/projection/expert computation, using batched GDN/QSA for one-token decoder subgroups and private causal advancement for longer segments. Independent long prefills retain the large-block path.

With decoders present, the scheduler chooses the largest compiled prefill block fitting the predicted *whole mixed iteration*, including input preparation, MTP warmup, cache snapshots, and first-token finishing. Bounds incorporate context-dependent monotonic estimates, rolling P95 positive errors, and EWMA margins. Cold start, long context, and checkpoints can still exceed the soft budget. A minimum effective chunk guarantees progress, using only real tail tokens. Vision encoding runs independently. Check `prefill_chunk_histogram`, `max_bounded_prefill_s`, and `decode_budget_overruns`.

Flash packages with private address tables run one set of GDN/QSA kernels per layer for all requests, isolating temporary buffers and committing convolution history in batches. Graph replay refreshes request addresses; padded rows have null bindings. A lone request uses M1 programs.

KV and MTP hidden rings allocate according to total token budgets. Oversized private allocations are shrunk on slot reuse; under memory pressure one idle slot may be retained while other idle buffers are reclaimed. Actual active request counts depend on memory and per-request budgets; requests exceeding admission capacity queue.

Defaults: **32 active requests**, **128 queued requests**; full queues return 429. Admission checks CUDA free memory and Linux `MemAvailable` with system reserve, shrinks prefix cache first, then queues. Queue order considers initialization, image encoding, reusable cache endpoints, and remaining prefill cost, with waiting-time priority smoothing. After 30 seconds, oldest jobs receive preference. Image identity is computed once and cache matching tracks checkpoint updates. Large cold prefills ordinarily admit at most two early, while short requests can admit together. Shared-prefix requests may wait for the first to compute the common prefix. A request blocked by insufficient memory after 30 seconds temporarily prevents new admissions until space frees or it is canceled/times out.

Measured elapsed prefill cost is charged against scheduling quotas. With active decoders, time since last committed token influences the next work choice; at least one prefill token advances periodically, incrementing `prefill_progress_overrides` if the soft target cannot be met. `--target-tpot-ms` is a soft engine-token-commit target, **not** a hard HTTP streaming latency guarantee. `max_decode_gap_s` and `decode_budget_overruns` measure engine gaps; HTTP chunk gaps must be measured separately.

Active batch size also adapts to measured neighboring batch costs and throughput, probing unknown sizes. Graphs for new member combinations execute directly first, capture on recurrence, and may be replayed for final steps. Shared segments can reuse Graphs by operator, rows, views, and addresses; private state remains per request, with address tables updated between member changes. Whole-batch and shared Graphs have a joint budget of 12,288 operation nodes. A separate bounded CPU plan cache retains only actual operator buffer views. `batch_execution.shared_graph_hits` and `shared_graph_captures` report reuse. `decode_only` still applies only to decode/MTP; Graphs for mixed prefill require `full`.

Context capacity is set by the package, up to 262,144 tokens. Long prompts prefill in chunks; dense first-pass attention complexity is unchanged. Group-64 INT8 KV retains FP16 scales; CUDA VMM maps physical pages to executed positions while keeping Graph addresses stable. Single-layer FP16 temporary prefill KV workspace is shared and grows on demand. Recurrent GDN state stays FP32.

### Server options

| Option | Default | Behavior |
| --- | --- | --- |
| `--listen` | `0.0.0.0:8088` | HTTP listen address |
| `--model` | `qwen3.8-27b` | API model ID |
| `--default-request-params` | `{}` | JSON defaults; explicit request wins |
| `--scheduler-policy` | `latency` | Online latency priority; `throughput` optimizes whole-job completion |
| `--cuda-graph` | `decode_only` | Graph policy |
| `--resident-experts` | `full` | Per-layer-equivalent backbone expert residency; global limited cache; MTP fully resident |
| `--verify-weights` | off | Full loaded-weight/CPU-table content checks at startup |
| `--prefix-cache-mib` | `12288` | GPU prefix-cache budget in MiB; 0 disables |
| `--max-active-requests` | `32` | 1–128 active requests |
| `--max-speculative-tokens` | `0` | Total MTP verification-row budget; 0 follows package capacity |
| `--max-batch-tokens` | `128` | 1–128 decode/verification rows; mixed prefill may use package capacity |
| `--prefill-budget-ms` | `200` | Soft predicted-cost bound on latency-policy mixed chunks |
| `--target-tpot-ms` | `400` | Soft time-between-engine-token-commits target in latency mode |
| `--memory-reserve-mib` | `1024` | Admission reserve beyond system headroom |
| `--preprocess-workers` | `2` | Parallel image/template/tokenizer CPU tasks |
| `--preprocess-memory-mib` | `2048` | Preprocessing and queued-image memory budget |
| `--queue-timeout-ms` | `0` | Queue wait deadline; 0 means unlimited, not a compute deadline |
| `--output-timeout-ms` | `60000` | Terminate when output channel cannot advance (excludes prefill) |
| `--drain-timeout-ms` | `30000` | HTTP drain period after SIGINT/SIGTERM |
| `--gpu-lock` | `artifacts/gpu-experiment.lock` | Shared GPU worker lock |

The default `latency` policy chooses combined or independent prefill by measured whole-iteration cost. Multiple cold requests do not automatically force long inputs into small blocks; unknown combined shapes do not supersede valid native large blocks. Mixed batches and independent prefill share a decode time budget; prompt work is followed by bounded decode bursts (including MTP) while long-waiting prefills still progress. `throughput` removes time quotas and cold-prefill count restrictions and can probe unknown combined shapes, retaining memory admission and isolation; see [Offline batch jobs](batch.md). Compare `/health` counters `independent_prefill_choices`, `joint_prefill_choices`, and `decode_burst_iterations`. Logs report queue time, engine-side TTFT since model-queue admission, generation time, and reasoning tokens. Engine token commit can precede complete client-visible text.

`/health` reports starting/ready/draining/failed plus scheduling, admission, cache restore, and Graph counters. GPU worker failures yield 503; CUDA or internal state faults require restart. See [Benchmark tools](tools/bench/README.md) for measurements.

Startup always checks configuration, execution-library and kernel identity, and the structure/dtype/shape/layout/offsets of weights and CPU-table safetensors, but normally skips SHA256 scans of large weights. Add `--verify-weights` to `serve`, `run-model`, or `score-model` for full content verification. `validate-model` always checks all weight contents. Default structural verification cannot detect corruption that changes only tensor values.

## Diagnostic CLI

`plan-model` loads the package's native library on CPU and prints generated plans. `validate-model` verifies model, native library, kernels, and tensor hashes; neither initializes the GPU. See [Execution packages](model-packages.md) for ABI and migration guidance.

`run-model MODEL_DIR REQUESTS.json` takes token-ID requests for fixed-block, ordinary-decode diagnostic benchmarking and resets per-request state. Input length must be divisible by a declared prefill plan; this is **not** a benchmark for arbitrary-length Chat, MTP, or caching. For those, use the Chat API or Rust `Model::generate`.

`score-model` scores frozen token histories. See [API validation](tools/api/README.md) for fixed scenario tests and [Quality evaluation](tools/eval/README.md) for BF16/FP8 references.
