# 性能工具

当前 Flash Next 的 Rust CLI / Chat API 实测与 27B 历史参考见[性能参考](../../docs/performance.md)。该文档分别列出执行层 prefill、输入 tokens / API TTFT、排除 TTFT 的 decode，以及完整 prefix 命中时延；本目录各工具的整请求吞吐不能直接替代这些指标。

`reference_bench.py`测量兼容OpenAI SSE接口的prefill/TTFT与decode时间；`test_reference_bench.py`检查token chunk和计时间隔。

参考服务的配置及身份须记录在输出目录的`reference-lock.json`中，输入token由`--seed-tokens /path/to/input-tokens.json`显式提供。请求要求服务支持token-ID输入/输出、固定长度生成和可选的独立计时诊断；不能直接用普通Chat请求代替。

`run_aot_artifact.sh MANIFEST NEW_OUTPUT`构建Rust CLI并执行AOT fixture，保留实际运行身份和机器状态。模型完整推理使用`tools/model/run.sh`。

`sample_machine.py`和`sample_continuous.py`只读记录机器温度、频率、内存和功耗；不会改动全局设置。GPU运行入口共享`artifacts/gpu-experiment.lock`。

`validate_plan.py`校验benchmark与快速评测配置，属于CPU检查，不产生性能测量。

`code_generation.py` 用相同 Chat 请求测量 thinking、MTP 和 prefix cache 的组合，输入采用下方 `concurrency.py` 的 cases 格式：

```bash
python3 -m tools.bench.code_generation --requests code-requests.json \
  --repetitions 3 --output artifacts/code-generation.json
```

每个 case 先执行一次预热，再重复请求；输出保留全部正文、思考文本、usage、首 token、首正文时间、实际缓存 token 命中率和 scheduler 差值。吞吐按已提交的 completion tokens 除以完整 HTTP 耗时计算，包含 thinking、prefill 和传输，不将 MTP 草稿或 SSE chunk 当作 token。报告单独列出空正文及预算截断，不能用提前结束的响应替代完整代码性能或质量验收。测冷请求需先重启自己的服务或使用新前缀；服务配置和输入保持一致，测量期间应独占接口。

该工具也适用于翻译任务：以 `--requests fixtures/translation-benchmark.json` 测量完整译文，核对三个段落、时间和标识符。固定 batch 的 `benchmark_flash_decode_template` 诊断由引擎输出预算控制长度，不额外添加 EOS logit bias，避免将普通 greedy 测成需要整词表回读的 CPU 采样路径；它不包含 HTTP 或排队时间。

`concurrency.py`以真实SSE请求测量1..128并发，输入为`{ "cases": [{ "id": "case", "request": { ...Chat API body... } }] }`：

```bash
python3 tools/bench/concurrency.py --requests requests.json \
  --concurrency 1 2 4 8 16 32 128 --repetitions 3 --output artifacts/concurrency.json
```

报告包括真实usage输出token数、整轮吞吐、TTFT分位数、缓存tokens及每请求结果。吞吐包含排队、prefill、HTTP和输出，不能直接当作GPU纯decode TPS；流式chunk间隔也不能当作token ITL。重复请求会命中prefix cache；测冷请求时需构造不同输入或关闭cache。`/health.scheduler_statistics`提供prefill实际计算tokens、引擎提交的decode tokens、batch直方图、混合迭代及峰值活跃数。

健康接口还提供准入、request startup、prefix恢复、prefill完成、batch输入准备/计划/提交及Graph统计。`batch_execution`记录batch Graph命中、未命中、捕获的kernel/copy/memset操作数、淘汰、地址变化失效和分项耗时；单请求/MTP程序运行期间的新捕获另记`sequence_captures/sequence_capture_s`。这些是累计主机计时，replay包含同步等待，不能视为CUDA event的GPU时间。`request_start_s`包含`prefix_restore_s`，batch Graph分项包含在`compute_s`中；单程序捕获可能发生在startup或计算阶段，不能把这些重叠计时重复相加。

工具在发送请求前、整轮完成后取得健康快照，并输出`scheduler_delta`和`admission_delta`；累计峰值不作差值。测量期间应独占服务，重启导致计数器回退时会报错。`admission_statistics`记录各轮处理的请求数、完整缓存命中的文本请求数、准入直方图及墙钟时间；它包含策略、显存准入检查和request startup。

`benchmark_prefill_scheduling`是默认跳过的真实GPU基准，比较同一模型的联合prefill与逐请求大块prefill。夹具包含`model`、新的`output`路径、`cuda_graph`、`batches`（2..8）、`repetitions`、`output_tokens`及`cases`；case使用`GenerationInput`字段`input_tokens`、`max_new_tokens`、`sampling`，只接受文本输入。采样应设置`temperature: 0`、`seed: 20261002`。关闭本项目GPU服务后，在独占实验锁下运行：

```bash
ORINFER_BATCH_FIXTURE=artifacts/prefill-scheduling-fixture.json \
  flock artifacts/gpu-experiment.lock \
  cargo test --release --offline -p orinfer-engine benchmark_prefill_scheduling -- --ignored --nocapture
```

基准关闭MTP和prefix复用，交换两条路径的测试顺序，分别记录准入、所有请求完成prefill的吞吐、每请求TTFT和后续batch decode。Prefill计时包含head和首token选择，TTFT另含准入；整个cohort完成prefill后才开始decode，未测量在线混合调度的ITL。联合路径使用算子包声明的分块，逐请求路径使用其最大可用单请求块；不同块可能采用不同量化码本。输出token差异用于诊断，不能替代独立BF16/FP8质量验收。每个请求另校验最终位置及KV回收。

`mixed.py`先预热一个文本`anchor`，启动指定数量的缓存命中请求，在每条流收到若干有效输出片段后注入一个冷请求。输入为`{ "anchor": { ...Chat API body... }, "cold_cases": [{ "id": "cold", "anchor_count": 2, "request": { ... }, "expected_prompt_tokens": 8192, "target_text": "CODE-641" }] }`。所有请求使用`temperature: 0`、`seed: 20261002`；anchor须足够长以保持生成。每个并发档位的冷请求应在输入开头使用不同标识，避免跨轮缓存；`anchor_count`限定case适用的档位，省略时适用于所有档位。提示token数和任务答案字段可省略。

```bash
python3 -m tools.bench.mixed --requests mixed-requests.json \
  --anchor-counts 1 2 4 8 --trigger-chunks 4 --output artifacts/mixed.json
```

报告包含无注入对照、长请求TTFT、anchor输出吞吐、长请求prefill期间重叠的SSE片段间隔及引擎计数器差值。跨越注入边界的停顿也计入；片段间隔不是token ITL。要求anchor完整命中prefix，冷请求命中不超过5%，通过准入计数检查无其他流量。测试采用服务当前配置，包括自适应MTP；纯decode吞吐与关闭MTP的基准应另测。

`acceptance.py`执行带延迟到达的异构任务验收，覆盖长短文本、图片、多图和多轮分支。JSON含`cases: [{id, request, expected_words}]`、可选`warmup: [id]`与`rounds: [{id, jobs: [{case, delay_s}]}]`；每轮最多128个提交请求。答案检查仅用于明确的标签/简短答案任务，忽略大小写和分隔符；并发吞吐从真实usage计算。输出保存每轮scheduler/admission差值、活跃/排队峰值及每请求结果，拒绝不完整SSE或任务答案错误。

```bash
python3 -m tools.bench.acceptance --requests acceptance-requests.json \
  --output artifacts/acceptance.json
```

`validate_flash_decode_batches` 使用同一 `ORINFER_BATCH_FIXTURE` 格式，检查 2/3/4/8/16/32 请求的同历史 logits/top3/NLL、图片与多图、取消/槽位复用，以及 target batch 后继续 MTP。需 native Flash batch 包和独占 GPU 锁：

```bash
ORINFER_BATCH_FIXTURE=/absolute/path/to/flash-fixture.json \
  flock artifacts/gpu-experiment.lock \
  cargo test --release --offline -p orinfer-engine validate_flash_decode_batches -- --ignored --nocapture
```

该检查关闭 prefix cache，对照同权重、相同冷 prefill 和 teacher-forced 历史的单序列执行，验证 batch 迁移的算术与隔离；不替代原始 BF16/FP8 量化质量验收。性能与缓存命中的功能验收另用真实 Chat/SSE 的 `concurrency.py`、`acceptance.py` 测量。

`validate_flash_heterogeneous_scheduling` 使用相同夹具格式，第一条 case 的输入须至少 4096 tokens，第二条为短请求。它分别检查两条 cold prefill 和已开始 decode 的短请求与长 prefill 混跑，确认长请求保留原生大块、活跃 decoder 期间块长受限、连续 decode 有推进，以及相同分块和 teacher-forced 历史下的 KL/NLL/top3；同样需要独占 GPU 锁。独立最大块与中断分块的首 token 分布差异另行记录，不当作状态隔离或 BF16 量化质量验收。`validate_flash_bounded_prefill` 接受按成本选择的联合或独立路径，不要求每轮都混合。

`validate_flash_long_generation_reuse` 使用相同夹具格式，前两条 case 为压力提示和后续验证提示；后者须包含原生 4096-token prefill。它先记录后续提示的冷请求 logits 与完整已提交状态，再强制执行 16384 个 decode tokens（含 MTP）、保存生成缓存并复用槽位，检查冷请求和完整 prefix 命中与参考的分块、logits、KV/GDN/卷积/PLE/MTP 状态一致。引擎测试刻意继续处理 EOS 来保证压力长度，任务输出正确性另用真实 Chat API 验证；同样需要独占 GPU 锁。

固定到达时间的混合请求验收使用 `python -m tools.bench.scheduling --requests TRACE.json --output RESULT.json`。输入为 `{"requests":[{"id":"short-1","at_s":0.1,"request":{"model":"MODEL_ID","messages":[{"role":"user","content":"Hello"}],"max_tokens":16}}]}`，最多 128 条；同一时刻可提交多条请求。到达时间不依赖前一请求是否完成，报告客户端到达误差、TTFT、整请求耗时和最大流式 chunk 间隔的 P50/P95/P99，以及失败数、有效输出吞吐、成功请求的 prefix 请求/token 命中率和调度统计。chunk 间隔不能当成 token ITL。比较调度版本时使用相同 trace、模型包、缓存/MTP/Graph 配置，分别测试冷缓存与预热后的缓存。

`profile_vision_stages` 是默认跳过的真实模型诊断，使用 `ORINFER_BATCH_FIXTURE` 中的 `model`、新 `output` 路径和带 `images` 的 `GenerationInput` cases。它按实际图片网格绑定视觉程序，记录整段热 Graph 与逐算子 GPU 时延；`ncu: true` 对实际 launch 添加 NVTX 区间。图片解码、resize、特征复制和文本生成不在该计时内，完整图片翻译性能须另测 Chat API。通过同一个 GPU 锁运行 `cargo test --release -p orinfer-engine profile_vision_stages -- --ignored --nocapture --test-threads=1`。

冷启动端到端测量使用 `python -m tools.bench.startup --model MODEL_DIR --request REQUEST.json --output artifacts/startup --server-args --mtp-drafts 0 --cuda-graph decode_only`。每轮启动独立 Rust CLI，通过 HTTP 等待就绪并提交首条流式 Chat 请求；记录进程启动到 ready、首个非空文本／thinking／工具片段及请求完成的耗时、usage 和原始输出，默认三轮并报告中位数。GPU 锁等待和退出清理不计入，文件缓存保持原状；这衡量进程冷启动，不等于重启机器后的冷盘读取。工具在子进程退出前持有共享 GPU 锁；`--gpu-lock-timeout 60` 可等待当前 GPU 工作结束。
