# Offline JSONL inference

`orinfer batch` loads a model and processes local files directly, without an HTTP server or Python. It shares the Chat template, tokenizer, sampling, stop handling, continuous batching, prefix cache, and model execution packages with `serve`.

```bash
orinfer batch /path/to/model \
  --model index-translate-2b \
  --input requests.jsonl --output results.jsonl \
  --max-active-requests 64 --mtp-drafts 0 --prefix-cache-mib 512 \
  --default-request-params '{"temperature":0,"seed":20261002,"enable_thinking":false,"max_completion_tokens":512}'
```

Concurrency, CUDA Graph, MTP, cache, and preprocessing-budget options are shared with `serve`. The default is at most 32 active requests. A completed request frees a slot for another, subject to memory admission. Whether concurrency 64 or 128 is faster depends on input and output lengths; maximum concurrency does not imply proportional throughput.

## Scheduling and input ordering

`batch` defaults to `--scheduler-policy throughput`, which optimizes total file completion time. When capacity is available, it continues admitting work without the online inter-token targets or the admission restriction on two long prefills. It reuses dynamic execution templates, selecting combined or separate prefill based on measured iteration costs. Unknown combined shapes may be probed; a prefill does not require an enforced decode interval afterward. Under this policy, `--prefill-budget-ms` and `--target-tpot-ms` do not constrain iteration duration, so individual TTFT or token gaps may increase. Memory limits, active counts, package capacity, output budgets, and stop rules remain enforced; MTP remains cost-adaptive.

Use `--scheduler-policy latency` to restore the online strategy (also the default for `serve`). Resume runs may switch policies.

In throughput mode, `--batch-order auto` (default) scans text requests and extracts real token prefixes through the same Chat frontend. With prefix caching off, longer requests are scheduled first. With caching on, it uses a compressed prefix tree, schedules heavier shared subtrees preferentially, and supplies checkpoint hints for shared branches. The prefix index and planning workspace together consume no more than one-quarter of the preprocessing budget, storing at most 4,096 prefix tokens per request. If the budget is insufficient, scheduling falls back to input order. Images and non-string content bypass text planning but still use bounded preprocessing.

Set `--batch-order input` to disable whole-file planning; prepared requests may still be reordered by the continuous GPU scheduler.

Throughput scheduling learns expected input/output length ratios from naturally completed requests, admits work by estimated remaining output alongside existing cohorts, and treats `max_completion_tokens` as a ceiling rather than an expected length. New work fills slots as requests finish; long-waiting requests still receive priority.

Global text planning also reduces disposable prompt and generation-continuation cache snapshots. Known shared full prompts, partial checkpoints, and request state continue to follow the common caching rules. For short, unrelated translation jobs, `--prefix-cache-mib 0` can help; measure the cache benefit separately for long shared prefixes.

## Input format

Input is **UTF-8 JSONL**, with one OpenAI Batch-style Chat request per line:

```json
{"custom_id":"paragraph-1","method":"POST","url":"/v1/chat/completions","body":{"messages":[{"role":"user","content":"Translate into Chinese: The service is ready."}]}}
```

A direct Chat request is also accepted, optionally with `custom_id`:

```json
{"custom_id":"paragraph-2","messages":[{"role":"user","content":"Translate into English: 服务已就绪。"}],"max_tokens":128}
```

If `model` is omitted, it defaults to the CLI's `--model`; an explicit model must match. Other Chat fields follow the [serving guide](serving.md), including supported images, ordered multi-image input, and tool calls. Local images may use data URLs; remote image downloads follow server-side restrictions. Each input record gets its own model template; separate paragraphs are not concatenated into one prompt.

- `custom_id` must be unique and between 1 and 256 bytes. If omitted, it defaults to the physical line number (`line-N`). Blank lines are skipped; duplicate IDs fail before the GPU model is loaded.
- A single line can be at most 32 MiB. Syntax and request-validation errors are recorded as per-line errors while other lines proceed. Job-wide errors (file read failures or oversized lines) stop the job.
- Offline requests return complete responses. Explicit `stream=true` or `stream_options` is invalid. Streaming settings supplied by deployment defaults are disabled for batch; other defaults merge according to the usual explicit-request precedence.
- Keep input files, referenced images, and model artifacts unchanged throughout a job and its resumption. The program checks for input file changes but does not persist immutable snapshots of remote images.

The [translation batch fixture](../fixtures/translation-batch.jsonl) can be used as input. Set sufficient output budgets so truncated translations are not mistaken for complete results.

## Output and resuming

Results are appended **in completion order**. Match them back to input records with `custom_id`. Successful records follow OpenAI Batch response structure:

```json
{"id":"batch-paragraph-1","custom_id":"paragraph-1","response":{"status_code":200,"request_id":"chatcmpl-...","body":{"choices":[{"index":0,"message":{"role":"assistant","content":"The service is ready."},"finish_reason":"stop"}],"usage":{"prompt_tokens":24,"completion_tokens":8,"total_tokens":32}}},"error":null}
```

An error record includes `custom_id`, `response:null`, and `error.code/message`. Complete success responses also include normal model, usage, thinking, and tool fields where applicable.

New jobs refuse to overwrite an existing output file. The companion `results.jsonl.meta.json` stores the input SHA256, model descriptor identity, and default request parameters; preserve it with the results.

To resume, add `--resume` to the original command:

```bash
orinfer batch /path/to/model \
  --model index-translate-2b \
  --input requests.jsonl --output results.jsonl \
  --max-active-requests 64 --mtp-drafts 0 --prefix-cache-mib 512 \
  --default-request-params '{"temperature":0,"seed":20261002,"enable_thinking":false,"max_completion_tokens":512}' \
  --resume
```

Committed successful and truncated responses are skipped; failed or unfinished requests are retried. Retry records with the same ID are appended, with the last entry authoritative. Truncated responses are counted separately: resuming does not repeatedly regenerate them with the same budget. Use a new output file after increasing the budget.

Changes to input, model descriptor, model ID, request defaults, or MTP configuration invalidate the resume metadata. Scheduling policy, read order, concurrency, and memory budgets may change.

Completed results enter a bounded writer queue and are written by a separate thread. Data is synchronized approximately once per second and at shutdown. Queued results count against the preprocessing memory budget. SIGINT/SIGTERM stop new work, drain already queued results, then exit; write or sync failures fail the job. On resume, only an incomplete trailing line without a newline is removed; corruption in committed lines is an error. A process lock prevents concurrent appenders. This mechanism does **not** guarantee exactly-once execution of input side effects or durability of writes not yet synchronized at power loss.

Standard output is a JSON summary including scheduling policy (`scheduler_policy`), input order (`batch_order`), planned request count/prefix tokens/planning time (`planned_requests`, `planning_prefix_tokens`, `planning_s`), totals, successes, truncations, failures, unfinished and skipped records, requests processed in this invocation, generated tokens, peak active requests, and elapsed time.

`output_tps` is *tokens generated in this invocation divided by total invocation time*. It includes scanning, model loading, planning, preprocessing, queuing, and persistence; it is **not pure decode TPS**. Exit status is 0 only when all requests complete normally; errors, truncation, or interruption result in a nonzero status.
