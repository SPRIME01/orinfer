# API 验证

`smoke.py` 只作为客户端连接运行中的真实模型服务，覆盖任意输入长度、EOS、SSE/usage、固定 seed、stop、排队和断开后的状态隔离、错误格式，以及带整数参数的工具调用和结果回传。输出使用新路径；启用服务鉴权时从环境变量 `ORINFER_API_KEY` 读取凭据。

```bash
python3 tools/api/smoke.py --base-url http://127.0.0.1:8088/v1 \
  --output artifacts/api-smoke-results.json
```

`compatibility.py` 进一步验证 nullable 参数、有符号 seed、字段定位错误、JSON/schema 与 thinking、严格/指定工具及参数增量、target logprobs、Unicode bytes 和请求隔离。服务开启 MTP 时，同一套测试覆盖其约束及概率路径；传入模型目录还会验证 token-ID bias。

```bash
python3 tools/api/compatibility.py --model qwen-flash-next \
  --tokenizer-dir /path/to/prepared-model \
  --output artifacts/api-compatibility.json
```

增加 `--long-logprobs` 会另外生成最多 512 tokens、每 token 20 个候选概率的完整响应，检查超过 512 KiB 的非流式输出；此项耗时更长。

`template_reference.py` 是离线参考工具，需要 Transformers。它使用原始 checkpoint 模板生成 token IDs，对照 Rust 模板和分词器；包含文本、tools、工具历史和 thinking 四种场景。JSON 对象的 key 顺序先与 Rust 一致化。

```bash
python3 tools/api/template_reference.py --tokenizer-dir /path/to/tokenizer-dir \
  --output artifacts/chat-reference.json
ORINFER_TOKENIZER_DIR=/path/to/tokenizer-dir \
ORINFER_CHAT_REFERENCE="$PWD/artifacts/chat-reference.json" \
  cargo test --offline checkpoint_matches_reference_tokens -- --ignored
```

[OpenCode 配置](../../examples/opencode.json)和[Pi 配置](../../examples/pi-models.json)提供日常编码接入；配置与命令见[服务指南](../../docs/serving.md#opencode-与-pi)。客户端执行工具并把结果发送回 Chat API。使用独立测试目录，验证读取、修改文件、运行测试、工具错误回传、thinking、压缩与继续会话；测试目录、生成文件、会话数据库和日志保存在忽略的本地工作区。

增加图片、多图、图片历史和 thinking 模板对照时，使用已构建的视觉 manifest 和本机 PNG 图片：

```bash
python3 tools/api/template_reference.py --tokenizer-dir /path/to/tokenizer-dir \
  --vision-model /path/to/prepared-model-dir --image /path/to/image.png \
  --output artifacts/chat-image-reference.json
ORINFER_TOKENIZER_DIR=/path/to/tokenizer-dir \
ORINFER_CHAT_REFERENCE="$PWD/artifacts/chat-image-reference.json" \
  cargo test --offline checkpoint_matches_reference_tokens -- --ignored
```

参考端由 Transformers 原始模板和 Qwen3VLProcessor 生成完整 input IDs，包括其原生图片 token 展开；Rust 端通过实际 Chat 请求处理路径做逐 token 比较。
