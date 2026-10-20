# API validation

`smoke.py` acts only as a client of a running real-model server. It covers arbitrary input lengths, EOS, SSE and usage accounting, fixed seeds, stop sequences, queuing, isolation after disconnects, error formats, integer-argument tool calls, and tool-result follow-ups. The output path must not exist. If the server requires authentication, the client reads `ORINFER_API_KEY` from the environment.

```bash
python3 tools/api/smoke.py --base-url http://127.0.0.1:8088/v1 \
  --output artifacts/api-smoke-results.json
```

`compatibility.py` additionally checks nullable parameters, signed seeds, field-specific errors, JSON/schema constraints with thinking, strict and forced tool choice with incremental arguments, target log probabilities, Unicode bytes, and request isolation. With server-side MTP enabled, the same suite covers its constraints and probability paths; providing a tokenizer directory also validates token-ID bias.

```bash
python3 tools/api/compatibility.py --model qwen-flash-next \
  --tokenizer-dir /path/to/prepared-model \
  --output artifacts/api-compatibility.json
```

The optional `--long-logprobs` case requests up to 512 output tokens with 20 candidate log probabilities per token. It validates non-streaming responses over 512 KiB and takes longer to run.

`template_reference.py` is an offline Transformers-based reference tool. It renders the original checkpoint template to token IDs and compares against the Rust template/tokenizer for plain text, tools, tool history, and thinking. JSON key ordering is first normalized to match Rust.

```bash
python3 tools/api/template_reference.py --tokenizer-dir /path/to/tokenizer-dir \
  --output artifacts/chat-reference.json
ORINFER_TOKENIZER_DIR=/path/to/tokenizer-dir \
ORINFER_CHAT_REFERENCE="$PWD/artifacts/chat-reference.json" \
  cargo test --offline checkpoint_matches_reference_tokens -- --ignored
```

The [OpenCode configuration](../../examples/opencode.json) and [Pi configuration](../../examples/pi-models.json) can be used for coding workflows; see [OpenCode and Pi](../../docs/serving.md#opencode-and-pi). Clients execute tools and return their results to the Chat API. Test in an isolated directory: verify file reads/edits, running tests, tool-error responses, thinking, compaction, and session continuation. Do not commit the test directory, generated files, session databases, or logs.

To validate single-image, multi-image, image-history, and thinking templates, use a built vision manifest and a local PNG:

```bash
python3 tools/api/template_reference.py --tokenizer-dir /path/to/tokenizer-dir \
  --vision-model /path/to/prepared-model-dir --image /path/to/image.png \
  --output artifacts/chat-image-reference.json
ORINFER_TOKENIZER_DIR=/path/to/tokenizer-dir \
ORINFER_CHAT_REFERENCE="$PWD/artifacts/chat-image-reference.json" \
  cargo test --offline checkpoint_matches_reference_tokens -- --ignored
```

The reference side uses the original Transformers template and `Qwen3VLProcessor` to produce complete input IDs, including native image-token expansion. The Rust side compares token by token along the actual Chat-request processing path.
