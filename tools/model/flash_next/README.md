# Flash Next 模型准备

提供自有 Q2A8 权重、Rust 在线适配器，以及离线完整推理与原始 BF16 参考。权重转换产物先通过下述部署构建生成 `cache/model.json` 和匹配执行包，再交给 CLI。

执行器固定使用已采用的 SM87 路径：专家 E8P＋block128 旋转、普通投影 W8、关键系数 BF16/FP32、group-64 INT8 KV、FP32 GDN 状态。prefill 按长度选择专家 tile；MTP 支持 1..7 个草稿和完整接受前缀提交。架构与编码约束位于 [`architecture-contract.json`](../../../configs/architecture-contract.json)，加载前拒绝不匹配的配置。

小批次验证使用精确的 E8P 短码表和字节置换；router 按小批次选择输出 tile。已有原生模型可用 `tools/model/flash_next/optimize_decode.py` 重建这些算子，权重逐字节保留，新执行包仍合并为 safetensors 资产容器：

```bash
ORINFER_CHECKPOINT_DIR=/path/to/native-model \
  ORINFER_EXECUTION_CACHE=/path/to/orinfer/packages \
  bash tools/operators/run.sh tools/model/flash_next/optimize_decode.py artifacts/flash-decode-build \
  --model /path/to/native-model --model-output artifacts/models/flash-decode
```

共享计算策略位于 `kernel_policy.py`，`lowering.py` 按实际权重角色和导出 ABI 生成实现及前置步骤。Q4 权重存在时，保留的 W8 参考权重不会覆盖它；行数只决定 GEMV/Tensor Core 和 tile。经过认证的 K64 E8P bank 在小 batch 上使用完整 A8 位平面分解，其他形状读取同一压缩 bank 的 INT8 路径。固定 batch、动态 batch 和混合 prefill/decode 构建都执行此 lowering，Rust 模型适配器把旋转与位平面拆分插入原有算子位置，私有因果状态不进入共享策略。

并发数值验收固定 seed，并在相同 teacher-forced 历史下逐步比较概率分布和 NLL；greedy token 一致率单独记录。近似浮点路径不以逐字相同为通过条件，取消、恢复、状态隔离和提交仍分别验证。`validate_flash_decode_batches` 的夹具可用 `numerical_tokens` 指定 4–64 个输出位置，包括提前 EOS 后的强制历史。

已有执行包可在不修改权重的情况下重新生成全部形状：

```bash
bash tools/operators/run.sh tools/model/flash_next/lowering.py artifacts/flash-policy-build \
  --model /path/to/native-model --model-output artifacts/models/flash-policy
```

## 权重准备

CPU 工具使用 `make python-env` 创建的 uv 环境，GPU 转换与验证使用固定编译镜像及 GPU 锁。固定原始 checkpoint revision，流式读取 BF16，不需要保存完整原始模型：

```bash
bash tools/quantization/run_flash_next.sh \
  --index /path/to/original/model.safetensors.index.json \
  --revision SOURCE_COMMIT_SHA --stage all \
  --output artifacts/quantization/flash-next/converted

PYTHONPATH=. .venv/bin/python -m tools.model.flash_next.weights \
  --converted artifacts/quantization/flash-next/converted \
  --index /path/to/original/model.safetensors.index.json \
  --config /path/to/original/config.json --frontend /path/to/original \
  --output artifacts/models/flash-next-e8p-a8
```

MTP 另用 `--component mtp --stage all` 转换，再发布到独立目录。target/draft 必须来自同一源、revision、编码与文本配置；共享 embedding 和输出头，分别保有 KV/index 状态。

发布目录保留 HF config、tokenizer、chat template 与标准 safetensors index。物理 tensor layout 存在 shard metadata 中；发布逐字节保留已转换权重，默认在验证后消费临时 shard，`--keep-temporary-shards` 可保留。`checkpoint.py` 读取发布目录，不依赖转换进度文件。模型格式和权重编码未因目录整理而改变。

## Rust 部署

在线包含 W8 embedding、CPU E8P PLE 查表、48 层 HC/GDN/QSA/MoE、256K INT8 KV，以及完整请求状态和 prefix cache。prefill 提供 4096／2048／512／128／16 token 档位，尾部用真实 M=1 执行；不填充虚假 token。基础包支持单个活跃请求、其余排队；下述 batch 构建可启用多请求 decode。可选原生 MTP 支持贪心和带惩罚项的随机采样。

```bash
make build
bash tools/operators/run.sh tools/model/flash_next/prepare.py artifacts/flash-build \
  --checkpoint /path/to/flash-target --model-output artifacts/models/flash-serving
PYTHONPATH=. .venv/bin/python -m tools.model.flash_next.package artifacts/models/flash-serving
./target/release/orinfer serve artifacts/models/flash-serving \
  --model qwen-flash-next --max-active-requests 1 \
  --cuda-graph decode_only --prefix-cache-mib 512
```

可选 MTP 执行包共享主模型 embedding 和完整输出头，增加单层 draft、512 行 HC 环和验证/提交程序。在线支持每轮 1–7 个草稿，默认最多 3 个；用 `serve --mtp-drafts 7` 指定上限，`auto` 沿用包默认值，`0` 关闭 MTP 执行并跳过草稿专用权重的上传。短尾、上下文边界和调度 token 预算不足时自动缩小验证批次或使用普通 decode；prefix checkpoint 保留 P−1 的 draft 状态，恢复时用真实 continuation 连接。模型文件与可共享的状态、workspace 保留；所指定的验证档位必须存在于执行包中。

```bash
bash tools/operators/run.sh tools/model/flash_next/prepare_mtp.py artifacts/flash-mtp-build \
  --checkpoint /path/to/flash-target --mtp-checkpoint /path/to/flash-draft \
  --base-model artifacts/models/flash-serving \
  --model-output artifacts/models/flash-serving-mtp
PYTHONPATH=. .venv/bin/python -m tools.model.flash_next.package artifacts/models/flash-serving-mtp
# serve 使用上面的相同参数，改为 flash-serving-mtp 目录；包含 MTP 的包自动启用。
```

代码修改或重复模板可额外指定 `--prompt-lookup`，从本请求历史中提取待验证续写；默认关闭，普通生成可能因较差候选而变慢。它仅适用于无惩罚 greedy 且未请求 logprobs 的请求，上限仍由 `--mtp-drafts` 控制。

验证阶段保存 GDN 的 FP32 紧凑更新、卷积/PLE 历史和每个 QSA pending 前缀；全部接受和部分接受均执行提交。Draft 分支会恢复其原始 pending 块，再以 target HC 重算已提交位置。随机采样使用 p/q 接受和残差分布修正。

构建会按实际 host ABI 导出并绑定 cubin，保存同权重的固定历史 logits；打包时逐项比较 Rust 注册计划与离线计划。CPU 表按行读取，embedding／PLE 解码缓存分别限制为 8／32 MiB，部署不用 Python。中断构建可用 `--reuse-data` 校验并复用已写入的 GPU 权重；CPU 表通过 hardlink 复用。

可用 `score-model MODEL_DIR artifacts/flash-build/score-requests.json` 核对部署输出。这个对照验证执行迁移，独立 BF16 质量评估仍按下文进行。服务兼容 OpenCode 的 `@ai-sdk/openai-compatible` provider，模型 ID 为 `qwen-flash-next`；建议输出预算至少 4096 tokens。

## 图片与多图

在已准备的文本目录上附加视觉编码器；如需 MTP，先准备上述 MTP 文本目录；沿用同源 checkpoint revision，仅下载 `model.visual.*` 原始 BF16 张量，不下载整份模型。文本权重和已有 cubin 通过 hardlink 复用，视觉权重以 FP16 保存，计算使用 FP16/FP32。原始权重转换仅允许 FP16 subnormal 范围内的舍入，拒绝溢出或更大的改动。

```bash
PYTHONPATH=. .venv/bin/python -m tools.vision.source \
  --model artifacts/models/flash-serving-mtp --output artifacts/checkpoints/flash-vision
bash tools/operators/run.sh tools/model/flash_next/prepare_vision.py artifacts/flash-vision-build \
  --base-model artifacts/models/flash-serving-mtp \
  --checkpoint artifacts/checkpoints/flash-vision \
  --model-output artifacts/models/flash-serving-mm
PYTHONPATH=. .venv/bin/python -m tools.model.flash_next.package artifacts/models/flash-serving-mm
./target/release/orinfer serve artifacts/models/flash-serving-mm \
  --model qwen-flash-next --max-active-requests 1 \
  --cuda-graph decode_only --prefix-cache-mib 512 --mtp-drafts 7
```

视觉编码器与 27B 共享 TileLang 算子和执行组件，merger 输出为 2560 维；QSA 查询、KV 和压缩索引使用交错三轴 MRoPE，MTP 读取移位后的图像特征。PLE 保留原始 image token ID。多图按消息次序独立编码，特征和位置索引属于请求私有状态；图片身份参与 prefix 匹配。

默认单图最多 8192 patches（2048 image tokens），多图合计最多 16384 image tokens；可用 `--max-patches`、`--max-features` 调整构建容量。`--compile-cache` 可复用已有 TileLang 编译缓存。上下文仍为 256K、KV 仍为 INT8；视觉编码不被文本 prefix cache 省略。请求格式见[服务使用](../../../docs/serving.md)，独立编码器、chat template 和 MRoPE 的验证入口见[图文工具](../../vision/README.md)。

## 多请求 decode

在已发布的文本或多模态包上离线增加 batch 算子。先执行 `make build`，使用新的目标目录；权重逐字节复用，shape-only 编译不需要加载第二份模型：

```bash
bash tools/operators/run.sh tools/model/flash_next/batching.py artifacts/flash-batch-build \
  --model artifacts/models/flash-serving-mm \
  --model-output artifacts/models/flash-serving-batch
./target/release/orinfer serve artifacts/models/flash-serving-batch \
  --model qwen-flash-next --max-active-requests 32 \
  --cuda-graph decode_only --prefix-cache-mib 512 --mtp-drafts 7
```

不需要再次执行 `package.py`。可用 `--compile-cache` 复用完成的 TileLang 缓存。2/4/8/16/32/64/128 档共享 HC、普通投影、路由、专家、共享专家和输出头；其余请求数补零到下一档。GDN、PLE 卷积和 QSA 沿用已验证的单序列状态 kernel，只推进真实 lane。MTP 捕获每个请求的 target HC，调度器根据收益在 target batch 和逐请求 MTP 之间选择。CPU PLE 的行缓存共享，历史独立。

冷 prefill 使用原有大块；已有 decoder 时，与 decode 交替执行按 `--prefill-budget-ms` 和在线耗时估计选择的真实小块。预算包含输入准备、target、MTP 预热及 checkpoint 工作的实测耗时，是软目标；至少推进一个 token。每请求 KV/context 预留参与内存准入，超出活跃或内存预算的请求排队。128 个提交请求不要求同时在显存中驻留 128 份满上下文状态。Graph 按槽位和形状缓存；取消、槽位复用和 prefix 恢复保持请求隔离。

进一步把 GDN/QSA 私有阶段改成跨请求并行 kernel：

```bash
bash tools/operators/run.sh tools/model/flash_next/private_mixers.py artifacts/flash-private-build \
  --model artifacts/models/flash-serving-batch \
  --model-output artifacts/models/flash-serving-parallel
```

该构建复用已有权重和批量投影，保持 schema 1。地址表按请求、层、缓冲区列存储，列顺序属于 `batch_layout.state_columns` 的存储布局；只上传当前批次及其空行。TileLang M1 算子在降低到 CUDA 前增加请求维度，保留原有 CTA 几何、计算顺序和精度；GDN/卷积、QSA KV/scale/index/pending、多图坐标和临时索引/softmax 缓冲均通过私有地址访问。20 个临时缓冲区按 256 字节对齐合并为每请求一块连续内存，避免大量小块分配；各层顺序复用此块，持续状态保持独立。空行不读写状态，卷积历史用批量 kernel 提交。单请求沿用 M1 程序，prefill 与 MTP 的执行语义不变。

为非常见请求数量附加动态行数回退：

```bash
bash tools/operators/run.sh tools/model/flash_next/dynamic_batch.py artifacts/flash-dynamic-build \
  --model artifacts/models/flash-serving-parallel \
  --model-output artifacts/models/flash-serving-dynamic
```

2/4/8/16/32/64/128 档仍使用原有专用计划；其他数量复用邻近容量的 AOT 算子，以真实行数执行 HC、投影、路由、dispatch、输出头和私有 GDN/QSA。专家 workspace 保留容量，实际 Counts/TileCount 和 launch 上界只覆盖有效分派。无需清零或计算虚构请求，线上不编译。动态 launch 合约记录容量及受校验的整数表达式，仍使用 schema 1；权重和精度不变。发布前检查 CUDA 签名与 host ABI 的参数数量，行数显式绑定为标量，避免依赖 affine shape 的隐式推导。

## 离线执行与验证

所有输出放在新的 artifact 目录。场景包含实际 chat template、thinking、固定历史 top3 概率和 NLL；seed 为 20261002：

```bash
bash tools/operators/run.sh tools/model/flash_next/native.py artifacts/flash-check \
  --checkpoint /path/to/flash-target --context 262144 --chunk 4096 --graph on \
  --max-new-tokens 64

# 可选 MTP：在上述入口加 --mtp-checkpoint /path/to/flash-draft --mtp-drafts 3
# --mtp-vocab-size 0 使用完整草稿头；默认 65536，只限制草稿，不限制 target。
# --mtp-adaptive 可选启用 1/3/7 动态深度。

bash tools/operators/run.sh tools/model/flash_next/checks/mtp.py artifacts/flash-mtp-check \
  --checkpoint /path/to/flash-target --mtp-checkpoint /path/to/flash-draft \
  --lengths 512 2048 8192 --drafts 3 7

bash tools/operators/run.sh tools/model/flash_next/long_context.py artifacts/flash-long-check \
  --checkpoint /path/to/flash-target --context 262144

bash tools/operators/run.sh tools/model/flash_next/checks/w8_kernels.py artifacts/flash-w8-kernels
bash tools/operators/run.sh tools/model/flash_next/checks/w8.py artifacts/flash-w8-requests \
  --checkpoint /path/to/flash-target --draft /path/to/flash-draft
```

W8 对照保留朴素路径用于数值回归，逐位置比较 logits 和 target/draft 状态；它不作为部署策略。可设 `ORINFER_OPERATOR_NSYS=1` 并给 `checks/w8.py` 加 `--trace-only`，单独采集诊断 trace。生成/接受率测试与独立量化质量评估分别进行。

独立 BF16 对照使用 `reference/`：先在编译镜像中用 `scenes.py` 固定请求，再在 CPU 环境读取被选中的原始 embedding/PLE，最后执行分层 teacher：

```bash
PYTHONPATH=. .venv/bin/python -m tools.model.flash_next.reference.inputs \
  --index /path/to/original/model.safetensors.index.json --config /path/to/original/config.json \
  --scenes /path/to/scenes.json --revision SOURCE_COMMIT_SHA --output artifacts/flash-inputs

bash tools/model/run_flash_teacher.sh \
  --index /path/to/original/model.safetensors.index.json --config /path/to/original/config.json \
  --scenes /path/to/scenes.json --revision SOURCE_COMMIT_SHA \
  --inputs artifacts/flash-inputs --output artifacts/flash-bf16
```

把完整 teacher 的 `results.json` 传给 `native.py --baseline`，比较相同历史的 token 质量和概率。仅运行部分层不能作为整模型质量基线。

## 源码边界

`weights.py`/`checkpoint.py` 管理权重，`native.py`/`mtp.py` 是离线执行参考，`reference/` 保留独立数学与原始权重读取，`checks/` 与 `tests/` 管理回归验证。算子留在公共 `kernels/model/`，不按模型复制。

`orinfer-models/src/flash_next` 提供在线适配器和 CPU 查表，复用 SDK 和 CUDA 执行器；QSA KV/scale/index/pending、GDN 与卷积均为请求私有状态。Python 执行器用于离线计划、数值参考和 MTP 研究。

## 混合 prefill/decode

已带动态 decode 和 K64 精确 E8P 专家的包可增加真正的混合执行。共享 HC、投影、路由和专家计算按拼接后的实际 token 行数运行；GDN、QSA、卷积、PLE 和 MTP 隐藏历史仍按请求分别推进。只有各片段的最后一行进入输出头。现有权重逐字节复用，线上不需要编译。

```bash
bash tools/operators/run.sh tools/model/flash_next/mixed.py artifacts/flash-mixed-build \
  --model /path/to/optimized-flash-model --model-output artifacts/models/flash-mixed
bash tools/operators/run.sh tools/model/flash_next/mixed_hc.py artifacts/flash-mixed-hc-build \
  --model artifacts/models/flash-mixed --model-output artifacts/models/flash-mixed-hc
```

混合片段支持 16、128、512 token，拼接容量最高 1024 行；独立长 prefill 继续使用原有 2048/4096 档。BF16 HC 操作数在矩阵乘法前准备，保留原来的 FP16 舍入边界。kernel 首次装载、graph capture 和 prefill 完成时的 head/cache/MTP 预热分别计时，避免影响后续普通批次的耗时预测；请求总时延仍包含这些开销。

Orin 上可用 `--prefill-budget-ms 350 --target-tpot-ms 400` 起步调优。350 ms 是混合批次的预测预算，400 ms 是 decode 间隔的软目标；首次加载算子仍可能超过目标。MTP 验证继续使用现有独立执行计划，普通 decode 可与 prefill 合并。


### 批量 MTP

在已经具备混合 HC 和私有地址表的包上增加共享目标验证，再增加独立草稿与接受后的批量预热：

```bash
bash tools/operators/run.sh tools/model/flash_next/mixed_verify.py artifacts/flash-verify-build \
  --model /path/to/flash-mixed-hc --model-output artifacts/models/flash-verify
bash tools/operators/run.sh tools/model/flash_next/mixed_draft.py artifacts/flash-draft-build \
  --model artifacts/models/flash-verify --model-output artifacts/models/flash-draft
```

共享验证支持每请求 2–8 行，保留每个候选位置的 logits 和精确提交因子。草稿共享 2/4/8 行的投影、HC、专家与 Q4 输出头；QSA、KV、索引和 pending 保持请求私有。接受后按真实的整段移位 token 拼接预热，单个包最多 8 行，只为每个请求的端点计算草稿头。CLI 对 2–8 个 decoder 按完整轮次成本自适应选择，其他配置仍可使用普通 target batch。

新增的 CUDA 编译只包含因子保存和完整行 Q4 草稿头等必要 kernel，其余复用现有 cubin。权重不重新量化；新增执行逻辑需要重建原生库、校验完整计划并重新进行 GPU/CLI 验收。程序声明属于模型包，关闭 MTP 时一并移除其专用 batch 程序。在线格式和 C ABI 表保持版本 1。

各验证宽度的临时 prefix/提交因子在同一请求内复用最大容量，不共享请求之间的状态；验证和接受提交完成后才复用。当前布局由每请求 115.07 MiB 降为 26.30 MiB，不改变原 kernel 的形状、索引或复制长度。

### 分块数值与真实尾块

生产构建在混合执行包之后追加以下步骤，复用现有权重并重建原生模型库：

```bash
bash tools/operators/run.sh tools/model/flash_next/canonical_reductions.py artifacts/flash-canonical-build \
  --model /path/to/flash-mixed-draft --model-output artifacts/models/flash-canonical
bash tools/operators/run.sh tools/model/flash_next/prefill_tails.py artifacts/flash-tails-build \
  --model artifacts/models/flash-canonical --model-output artifacts/models/flash-exact
```

QSA 的选中 KV 固定按 256-token 区间归约，再按固定顺序合并八份 FP32 部分结果；最后一份同时覆盖不足四 token 的 pending 尾部。GDN 固定 FP32 归约布局，HC down 按四路 stream 分别累加后依次求和。这些数值边界不随 batch 或 prefill 大小改变；权重、INT8 KV 和 Tensor Core 精度保持原配置。

2–15-token prefill 尾块复用 16 行的投影与专家 cubin，只为真实长度编译因果算子。填充行不推进 KV、GDN、卷积、PLE、MTP 历史或位置，输出头选最后一条真实残差。尾块捕获和 MTP 验证使用独立的程序名。此改动需要新的执行包；更新 CLI 本身不会改变已发布 cubin 的行为。

### 可暂停的大块前向

当前模型库为 512/2048/4096-token 前向注册按层切分的阶段，边界在全部融合和 Q4 前置转换完成后生成。阶段引用现有操作和 cubin；运行时在切换序列或执行 batch 前保存仍存活的 HC/PLE 工作区，插入 decode/MTP 后恢复。同一请求连续执行时不复制工作区，备份分配在块内复用。不发布尚未完成所有层的 prefix 状态。MTP warm 也按已有 warm profile 分步执行。

对已完成上述数值与尾块构建的模型，重建原生执行包即可启用：

```bash
make build
PYTHONPATH=. .venv/bin/python -m tools.model.catalog \
  --model /path/to/flash-exact --output /path/to/flash-layered
```

该命令重新生成匹配 catalog 的模型库，复用权重和 cubin。只替换 CLI 不会为旧执行包增加阶段。`/health` 的 `layered_prefill_iterations`、`prefill_suspend_s`、`prefill_resume_s` 和 `peak_prefill_frame_bytes` 可用于确认实际执行及开销。
