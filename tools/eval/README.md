# 质量评测工具

`quick_quality.py`分析同一teacher-forced历史下的baseline/candidate token概率；概率按完整词表归一化，并显式查询baseline top-3 IDs。token身份不同用于诊断，不单独作为质量失败条件。

`scoring_common.py`校验固定seed、历史身份、概率字段及任务结果。`run_quality.py`连接配置的评分接口，使用`fixtures/quick-quality-scenarios.json`生成评测请求。参考服务需自行提供。

```bash
python3 -m unittest discover -s tools/eval -p 'test_*.py' -v
python3 tools/eval/run_quality.py --help
```

官方BF16或同模型FP8用作量化质量参考；同AWQ权重对照只用于实现诊断。场景检查、target NLL与长上下文任务需要分别评估。

翻译模型可在 `offline_quality prepare` 时指定 `--fixtures fixtures/translation-quality-scenarios.json`，覆盖中英日德翻译、术语、JSON 占位符、否定和商务语气。参考译文用于固定概率评分历史；自由生成允许其他准确译法。这组快速检查不代表完整多语言 benchmark，图片和多图另做视觉概率及 API 任务验收。

## 离线 BF16 对照

`reference_bf16.py`使用支持Qwen3_5的Transformers，从原始safetensors逐层加载权重，避免整份BF16权重常驻GPU。参考执行完整序列prefill；`score-model`执行正常prefill及逐token teacher-forced decode。报告明确标记两条路径，保留完整词表归一化的top-3、指定token概率与实际历史SHA256。

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

GPU任务通过`flock artifacts/gpu-experiment.lock`串行执行。参考与candidate使用固定seed20261002和相同原始token IDs。场景答案用于冻结评分历史，独立自由生成仍需验证任务结果；该概率比较不代表完整LLM benchmark，也不单独证明BF16 autoregressive路径。

磁盘放不下原始checkpoint时，可在保留原始配置、tokenizer和权重index的目录上使用`--remote-repo REPO --revision COMMIT`。COMMIT必须是固定的40位提交，读取器校验HTTP Range边界，按文本组件并行读取原始tensor，记录实际读取字节的SHA256；不经过量化权重重建。`--read-workers`控制网络并发，范围1–16。可选的`--tensor-cache DIR`消费预取的标准safetensors，校验来源提交和payload hash，加载后删除已消费文件。该流式读取用于质量对照，耗时包含网络，不能当作模型启动性能。

`prepare --long-context-tokens 512 2048 8192`追加确定性的中部资料检索场景。保留完整Chat模板，实际token长度接近且不超过给定预算；评分历史固定为正确资料答案。参考的`--batch-size`默认1，可合批以减少读取权重的次数；BF16舍入也会随GEMM形状变化，单流验收应使用单流参考。

图片评分在每个case增加`images`数组，每项为`grid_height`、`grid_width`及归一化FP32 patch数组`pixels`，与在线`ImageInput`契约一致。`prompt_ids`须包含原生模板和展开后的图片占位token；参考和candidate共享相同patch输入。参考使用官方BF16视觉编码器、特征注入和MRoPE，不读取candidate视觉特征。输入指纹包含图片尺寸、FP32 little-endian像素SHA256和图片顺序，防止错误配对。该对照覆盖编码器至token概率；原始图片解码/resize应另用预处理参考验证，任务结果仍需独立自由生成。

Transformers 的图像参考需要显式传入 `mm_token_type_ids`：展开的 image-pad 为 1，文本及界定符为 0；自由生成也必须保留该字段，确保使用 MRoPE。部署 tokenizer 应合并 `tokenizer_config.json` 的 added-token 配置，并与 Transformers 对照实际 token IDs；相同渲染字符串仍可能被切成不同 tokens。
