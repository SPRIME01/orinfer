# Flash Next 离线构建

在线推理由 Rust 执行，TileLang 只参与离线编译。部署目录使用 HF 配置与 tokenizer、`cache/model.json`、标准 safetensors 权重，以及由 digest 固定的执行包。数据格式与 C ABI 均为版本 1。

## 重建当前原生模型

已有当前格式的模型，用一个入口重新调优整数投影并发布匹配的执行包：

```bash
make build
bash tools/operators/run.sh tools/model/flash_next/build.py artifacts/flash-build \
  --model /path/to/native-model --model-output artifacts/models/flash-ready
```

输入必须包含匹配的执行包：它提供架构、请求状态、程序与绑定契约。这个入口不是任意 safetensors 导入器，也不能用来直接部署量化中间产物。现有图片、MTP、上下文容量和请求状态语义随输入保留；不通过猜测目录名启用功能。

构建直接从绑定的权重形状、dtype 和 host ABI 得到 Q4、W8、融合 SwiGLU 的构造参数，不从生成的 CUDA 文本恢复计算配置。相同矩阵几何共享编译结果，使用一组确定的真实权重与 scale 测量候选。固定行数保留 1/2/3/4/8/16/32 特化，其余使用动态行参数。decode 动态候选按几何分布测量多个实际行数；prefill 根据模型声明的程序识别，默认针对 512/2048/4096 行测量，可用 `--prefill-tuning-rows` 调整。所有动态实现仍检查短行、非对齐尾部与容量上限；输出必须精确保持整数点积、scale 与舍入边界，并通过修改输入后的 CUDA Graph replay。

专家的路由、上投影和下投影保留输入中已验证的联合选择表，不能分别换 tile。构建另外读取真实专家 bank，在完整 E/N/K 几何上验证两种合法行 tile、短尾与最大容量。HC、QSA、视觉和因果状态实现保留固定身份，不因整数投影重建而改写。

缓存默认在 `artifacts/build-cache/flash-next`，可用 `--cache` 指定。键包含构造参数、真实权重身份、源码、编译器、CUDA 和设备；内容校验失败自动重建该项。单项完成后原子发布，整个模型在 Rust 计划检查通过后发布。失败可重跑同一条命令，只需在包装器上增加 `--resume`：

```bash
bash tools/operators/run.sh --resume tools/model/flash_next/build.py artifacts/flash-build \
  --model /path/to/native-model --model-output artifacts/models/flash-ready
```

对已经验收的原生包，可加 `--prefill-only` 只重建其声明的 prefill 投影，沿用其他固定身份的算子与专家组合。它只限制离线构建范围，运行时仍使用同一动态行模板。

只清点构造契约不需要 GPU：

```bash
PYTHONPATH=. .venv/bin/python -m tools.model.flash_next.build \
  --model /path/to/native-model --model-output artifacts/models/flash-ready \
  --output artifacts/flash-plan --plan-only
```

构建记录保存在指定输出目录，不属于在线模型协议。最终包合并资产、去重绑定，只包含所需 cubin。重建完成仍须进行真实请求的精度、完整状态及性能验收；局部算子检查不能代替端到端验收。

## 计算与状态契约

专家使用自有整数 E8P Q2A8；GDN QKV/Z、QSA Q/gate 和 target/MTP 共享输出头采用 signed-block128 旋转、输出行 scale 的码表 Q4A8。GDN/QSA 输出投影、共享 FFN 与草稿 Q/O 保留 W8，其余敏感部分维持混合精度。E8P 的完整 A8 位平面分解用两个 INT4 Tensor Core 点积恢复 INT32 结果，不等于 A4 量化。权重只保留一份压缩 bank。

prefill 模板接受 1–4096 行，更长输入分块。普通 decode、混合 batch 与 MTP 验证共享最大 1024 行的 target 计算；draft/warm 共享最大 512 行。行数只决定局部 launch，不复制完整模型执行流程，也不补出虚假请求。局部实现选择来自离线测量，不手写性能分界。辅助算子统一动态模板，HC 无损指数、QSA radix shift 等不改计算几何的值通过标量传入。

QSA KV 为 group-64 INT8，GDN 持续状态为 FP32。KV、pending、压缩索引、卷积/PLE 历史、位置与 MTP 隐藏环均为请求私有状态。改变分块、恢复、捕获或提交时必须比较全部状态。混合 packet 显式记录 prefill/验证角色，角色边界也参与 Graph 身份。

## 部署

```bash
./target/release/orinfer serve artifacts/models/flash-ready \
  --model qwen-flash-next --listen 0.0.0.0:8088 \
  --cuda-graph decode_only --mtp-drafts 3 --prefix-cache-mib 512
```

MTP 包支持 1–7 个草稿；`0` 关闭，`auto` 沿用包默认。随机采样使用 p/q 接受与残差分布修正。短尾与调度预算可缩小验证宽度；prefix 恢复保留完整 target/draft 状态。历史续写候选仅通过显式 `--prompt-lookup` 启用。

服务、请求默认参数、图片和多图输入见[服务说明](../../../docs/serving.md)。视觉容量由输入模型声明，文本重建不扩大视觉或上下文容量。

## 原始权重与参考工具

CPU 依赖由 `make python-env` 创建的 uv 环境管理。GPU 工具统一使用固定编译镜像和共享 GPU 锁；原始 BF16 固定 revision，按矩阵或专家读取，不要求保存完整模型：

```bash
bash tools/quantization/run_flash_next.sh \
  --index /path/to/original/model.safetensors.index.json \
  --revision SOURCE_COMMIT_SHA --stage all \
  --output artifacts/quantization/flash-next/converted

PYTHONPATH=. .venv/bin/python -m tools.model.flash_next.weights \
  --converted artifacts/quantization/flash-next/converted \
  --index /path/to/original/model.safetensors.index.json \
  --config /path/to/original/config.json --frontend /path/to/original \
  --output artifacts/models/flash-next-quantized
```

这些是量化中间权重，不能直接作为当前原生部署包。`native.py`、`mtp.py` 与 `reference/` 提供离线数学对照；正式执行包重建统一使用上方 `build.py`，共用 `compiler.py` 的 host ABI 绑定与编译产物导出。旧的固定 batch／逐长度计划升级入口已移除，不支持把历史中间包转换为部署包。

数值验证固定 seed 20261002，对同一 teacher-forced 历史比较 top3 概率、KL/NLL 和任务结果；token 不完全一致不自动判失败。完整 BF16 参考位于 `reference/`，部分层结果不能视为整模型基线。`checks/` 提供 kernel、尾部和 replay 验证，Rust GPU 测试覆盖状态、prefix、MTP 和混合请求。性能以 Rust CLI 完整请求为准，不能用微测外推 TPS。

源码职责：`construction.py`/`build.py` 管理正式重建，`decode_tuning.py` 管理整数候选；`weights.py`/`checkpoint.py` 读取离线权重，`native.py`/`mtp.py` 保留数学参考。公共算子在 `kernels/model/`，在线架构适配器在 `crates/orinfer-models/src/flash_next/`。
