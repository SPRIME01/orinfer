# 性能参考

本页记录 Jetson AGX Orin 64GB / CUDA SM87 上的实测参考，各节注明测量日期。2026-10-12 的机器为 MAXN 模式、GPU 频率 1300.5 MHz；测试未修改全局功耗或频率设置。结果对应 Rust release CLI 和匹配的优化执行包；执行库、kernel 或 ABI 变化时需重建执行包，单独更新 CPU 调度可复用原执行包。表中的 tokens/s 均按实际 token 数计算。

## 27B 分层 prefill 与批量 MTP（2026-10-17）

27B 与 Flash Next 共用分层注册、动态行调用、私有地址绑定及运行时调度。Prefill 保持原 token 块，在模型声明的层边界暂停；相邻层按剩余时间预算及实测层成本连续执行，不固定每组层数。普通 decode 的私有 GDN／attention 从原单行计算体包装成动态 1–128 请求 kernel。批量 target 验证、draft、commit、warm 和混合 prefill／verification 共用原有计算体及显式请求角色，分别绑定请求状态。

同一份官方 beam8 两级 E8P Q4A8 权重，Rust release CLI `serve` 和真实 SSE Chat API；MTP 草稿上限 4、`decode_only`、thinking 关闭、greedy、seed 20261002。固定代码任务填充至 512／2K／8K 输入，输出预算 128 tokens。下表每组重复两轮，统计完整一轮的总输出 TPS，包含排队、prefill 和生成，不等同于纯 decode TPS。“冷”表示 prefix cache 关闭；“命中”表示输入前缀完整命中，并非服务启动冷热。对照为 `ff8ca0e` 的 KV 预取执行包与本轮重构前的运行时。

| 场景 | 对照总 TPS | 当前总 TPS |
| --- | ---: | ---: |
| 冷 2 并发，512 + 8K | 9.07–9.21 | **9.81–9.97** |
| 冷 4 并发，2×512 + 2K + 8K | 11.37–11.76 | **14.63–15.03** |
| 冷 8 并发，2×512 + 3×2K + 3×8K | 9.78–10.07 | **11.51–11.79** |
| 全命中 4 并发，同上 4 路输入 | 32.77–33.87 | **58.91–64.67** |
| 全命中 8 并发，同上 8 路输入 | 49.21–50.09 | **69.31–90.81** |

冷 4 路最长首 token 等待由 38.17–39.87 s 降到 28.82–29.27 s，冷 8 路由 91.84–95.38 s 降到 77.59–80.51 s。8K 冷请求先到、512-token 请求晚到 200 ms 时，短请求的排队／初始化时间从 6.176 s 降到 0.022 s，首 token 等待从 6.947 s 降到 **1.616 s**。缓存命中的 8K 请求与晚到的 512-token 冷请求交错时，短请求首 token 等待为 **2.293 s**，对照 2.374 s。SSE chunk 可能包含多个 MTP token，这些 chunk 间隔不能当作单 token ITL。

16 路冷请求另测一轮 8×512 + 8×2K：总 TPS 从 21.35 变为 **20.67**，最长首 token 等待从 64.86 s 增至 **78.75 s**。该场景仍有回退和明显调度长尾；本轮没有修改既有冷请求准入限制，不能据此宣称任意并发都改善。

执行包从 173 增至 **181 个不同 cubin**，增加的是 8 个动态私有状态实现，没有枚举 batch 大小。409 个逻辑程序、5326 个局部绑定包含共享片段、逐层参数及 64 个 prefill 阶段，均不等于独立 cubin。每请求增加 2,208,512 bytes 私有临时区，全局地址表增加 917,504 bytes；没有展开权重常驻副本。

另用 `run-model` 相邻进程对照，关闭 MTP/prefix、每档预热一次后测两次单 token 输出，512／2K／8K 主干 prefill 从 678.00／689.51／618.11 变为 **678.94／689.63／617.53 TPS**，差异均在 0.15% 内。固定 64 输出的普通 decode 两次均值为 **10.74 TPS**。4 草稿 MTP 固定前缀夹具约 **24.8 TPS**，此项为强制长度的状态与性能对照，不代表自然代码生成吞吐。

普通 B2/B3/B4/B8，以及验证宽度 2/4/8 × 请求数 1/2/4/8 × 纯验证/混合 prefill 的 24 组同历史对照，最大 KL 与所选 token NLL 差异均为 0。4K 分层暂停、按预算连续执行多层、交替覆盖 workspace、取消复用及部分前缀恢复后的持久状态一致。Flash Next 重建后的执行契约与原包完全相同，真实 GPU 分层恢复检查通过。`make check` 和 `make check-offline` 通过。本轮整模型 Compute Sanitizer 因设备禁用 GPU debugging 未能执行，不能视为 memcheck 通过。

本机当前模型目录已更新到重建后的执行包，并通过完整文件校验和实际加载。从最终目录重跑的 166 个 teacher-forced 概率记录与原量化执行路径逐项相同；官方 BF16 top1 一致率保持 **98.19%**，选择全部位于 BF16 top3，高置信位置一致 **150/150**。


## Flash Next 短请求回归（2026-10-15）

普通 decode 使用同一动态行模板，专家全量常驻、INT8 KV、`decode_only`；数据为固定长度夹具的完整 Rust 请求回归。

相同 Rust release 请求链连续复测，普通 decode 并发 1／2／4／8／33 为 26.08／43.33／73.82／112.51／167.14 TPS，与上一包的差异均不到 1%。两组代码生成 MTP，原包为 39.09／51.76 TPS，新包为 39.04／51.83 TPS，输出 token 相同。MTP 统计排除最初三个迭代。

Rust release CLI 三个热态 512-token 请求合计 prefill 原包 987.6 TPS、新包 986.3 TPS；加载到就绪分别 18.29 s、18.16 s。加载为连续进程复测，未清理系统缓存，不作为冷启动加速结论。模板仅参与加载和计划生成，Graph replay 不解释模板，未观察到推理性能回退。

## Flash Next 完整生成

使用官方 checkpoint 构建的文本模型：自有整数 E8P Q2A8 专家、Q4A8 输入投影与输出头、W8 输出投影与共享 FFN，精度敏感路径保留 BF16/FP16/FP32。专家全量常驻，KV 为 group-64 INT8，GDN 持续状态为 FP32。测试不含图片、其他并发请求或历史续写草稿。

完整代码生成通过 `orinfer serve` 的流式 Chat Completions API 执行，覆盖实际 chat template、请求调度、状态恢复、MTP 和 CUDA 计算。Python 客户端只发请求、接收 SSE 和计时。服务配置：

```bash
./target/release/orinfer serve /path/to/optimized-flash-next \
  --model qwen-flash-next --cuda-graph decode_only \
  --resident-experts full --mtp-drafts 3 \
  --prefix-cache-mib 1024 --memory-reserve-mib 512
```

关闭 MTP 的对照使用相同执行包和配置，只将 `--mtp-drafts` 改为 `0`。草稿长度使用正常的请求级自适应策略，没有固定验证宽度。

请求参数为 `enable_thinking=true`、`reasoning_effort=medium`、`temperature=0`、`seed=20261002`、`thinking_token_budget=16384`、`max_completion_tokens=16384`。思考和代码共享 16K 总输出上限；API 为思考结束标记和至少一个答案 token 预留空间。

场景要求从零实现安全 Rust 流式帧解码器：大端 u32 长度头、任意碎片输入、最大帧限制、复位和标准库单元测试。提示包含与实现无关的仓库参考 helper，以构造不同输入长度。完整请求测试的输入分别为 512、2048、8191 tokens。

### 完整代码生成

每种输入、每个 MTP 配置各运行一次首请求和一次相同提示的完整 prefix 命中请求。下表采用后者的暖态数据，每格为一次完整请求的测量值。生成 6176–7598 tokens，包含思考和代码；所有请求均自然结束，`finish_reason=stop`。

| 输入 tokens | 实际输出 tokens | MTP 关闭 decode TPS | MTP 开启 decode TPS | 加速比 | MTP 思考阶段 TPS | MTP 代码阶段 TPS | 草稿接受率 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 6276 | 29.72 | 44.99 | 1.51× | 41.46 | 49.37 | 74.8% |
| 2048 | 6176 | 29.64 | 45.21 | 1.53× | 44.02 | 46.62 | 76.4% |
| 8191 | 7598 | 29.49 | 45.82 | 1.55× | 43.14 | 49.30 | 77.4% |

Decode TPS 为 `(completion_tokens - 1) / (完整 HTTP 请求耗时 - TTFT)`，包含 Rust 调度、token 传输、同步与 SSE 尾部处理；只统计提交的 tokens，包含思考，不统计被拒绝的草稿。TTFT 从发请求计时，到收到首个有效思考或正文片段。SSE 片段可能合并多个 tokens；阶段 TPS 按 API usage 和思考/正文片段时间边界近似计算，不将片段数当作 token 数。

完整请求的 decode 为约 45 TPS，代码阶段为 46.62–49.37 TPS；这些结果没有达到完整请求 50 TPS，不能用代码阶段吞吐替代包含思考的完整 decode。

两种 MTP 配置下，全部请求的思考文本、代码、token 数和停止原因一致。三个输入长度生成的代码均可编译；512 与 8191-token 场景的生成单元测试全部通过。2048-token 场景有一个生成测试重复输入首个帧头字节，原始结果为 10 通过、1 失败；仅修正该测试输入、保持生成的解码器实现不变后，11 项通过。这验证了本次 MTP 开关的一致性，不替代独立 BF16/FP8 量化质量评估。
