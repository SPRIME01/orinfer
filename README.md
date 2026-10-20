# Orinfer

[简体中文](README.md) | [English](docs/en/README.md)

LLM and vision-language inference on NVIDIA Jetson, powered by Rust and TileLang.

面向 **Jetson AGX Orin 64GB / CUDA SM87** 的推理引擎：Rust 负责在线运行和服务，TileLang 算子离线编译为独立 GPU 包。支持 Qwen3.8-27B（`Qwen3_5`）和 Qwen3.8-Flash-Next（`qwen4_exp`）的文本、图片和多图推理。开发版新增 Qwen3.5-4B 和 Index-Translate-2B/9B 翻译适配；图片输入已接通，复杂图片的翻译质量仍需改进。

## 功能

- OpenAI Chat Completions 与 Ollama 原生聊天/生成 API，支持流式输出、thinking、采样、严格函数调用、JSON Schema 约束和 token 概率。
- [JSONL 离线推理](docs/batch.md)：连续批处理、逐条落盘和断点续跑，无需 HTTP 服务。
- 27B、4B 与 Index-Translate-2B/9B 使用两级 E8P Q4/W8 混合权重；Flash Next 使用自有 E8P Q2 专家和 Q4/W8 混合投影。以 INT8 混合精度计算，关键路径保留 BF16/FP16/FP32。
- 分块 prefill、独立请求状态、prefix cache、原生 MTP 和连续批处理，内存不足时排队。MTP 根据并发、上下文、执行耗时和逐深度接受情况自动选择投机深度，也可回退普通 decode。`--mtp-drafts 1..7` 设置草稿上限，`0` 关闭，`auto` 使用执行包的上限。
- 混合架构 prefix cache、INT8 KV 和按需映射内存；模型包可配置最长 256K 上下文。
- CUDA Graph 提供 `decode_only`、`full`、`off` 三种模式，默认 `decode_only`。

## 快速开始

部署需要 aarch64 Linux 和 NVIDIA CUDA Driver，**不需要 Python、PyTorch、TileLang 或 Rust 编译器**。

从 [Releases](https://github.com/iMMIQ/orinfer/releases) 下载引擎发行包、模型匹配的执行包和 `SHA256SUMS`，校验下载文件后解压引擎，将 `bin/orinfer` 加入 `PATH`。模型目录单独下载，包含配置、tokenizer、chat template 和原生量化 safetensors；模型描述中的 `execution_package` 必须与执行包 digest 一致。发行包不附带权重，**当前不能直接加载任意 HF、AWQ 或 GGUF 目录**。

从 v0.1.4 起，发行包统一使用 `.tar.xz`，执行包通过 `package install` 安装；该命令也接受旧 `.tar.gz`。v0.1.3 的安装方式见[旧版安装说明](docs/model-packages.md#安装已发布的-v013)。

```bash
sha256sum --ignore-missing -c SHA256SUMS
orinfer package install /path/to/matching-execution.tar.xz
orinfer validate-model /path/to/prepared-model
orinfer serve /path/to/prepared-model --model qwen3.8-27b
```

默认监听 `0.0.0.0:8088`。设置 `ORINFER_API_KEY` 可启用 `/v1` 的 Bearer 鉴权。

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"用一句话介绍你自己。"}],"max_tokens":512,"stream":true}'
```

提供 `/health`、`/v1/models` 和 `/v1/chat/completions`，支持流式文本、工具、thinking、JSON 约束与图片/多图。[服务使用指南](docs/serving.md)包含参数说明，以及 [OpenCode](examples/opencode.json) 和 [Pi](examples/pi-models.json) 接入配置。

## 常用配置

| 配置 | 默认值 | 用途 |
| --- | --- | --- |
| `--listen` | `0.0.0.0:8088` | 监听地址 |
| `--default-request-params` | `{}` | JSON 请求默认值，显式请求优先 |
| `--cuda-graph` | `decode_only` | Graph 模式 |
| `--resident-experts` | `full` | 专家全量常驻；Flash Next 可指定每层等效容量，按需从 SSD 加载 |
| `--prefix-cache-mib` | `12288` | GPU 前缀缓存预算；0 关闭 |
| `--max-active-requests` | `32` | 活跃请求上限，支持 1–128 |
| `--max-batch-tokens` | `128` | 普通 decode 行预算；混合 prefill 可使用执行包容量 |
| `--max-speculative-tokens` | `0` | MTP 总验证行预算；`0` 使用执行包允许的容量 |
| `--prefill-budget-ms` | `200` | 混合 prefill 的预测耗时目标 |

实际并发和上下文容量受执行包及可用内存限制。环境变量统一使用 `ORINFER_*`；共享模型包默认安装到 `~/.cache/orinfer/packages`，可用 `ORINFER_EXECUTION_CACHE` 指定位置。完整配置见[服务使用指南](docs/serving.md)和 `orinfer serve --help`。

## 性能与范围

Jetson AGX Orin 64GB、当前开发版本及匹配的 Flash Next 优化执行包：512-token 输入的执行层 prefill 为 **1008.9 tokens/s**。通过 Rust CLI 的流式 Chat API，从零生成 Rust 代码，开启 `medium` thinking、总输出预算 16K，实际生成 6.2K–7.6K tokens 时，单流 decode 关闭 MTP 为 **29.49–29.72 tokens/s**，开启最多 3 个草稿后为 **44.99–45.82 tokens/s**，提升 **1.51–1.55 倍**。

详细 prefill、API 首 token 时延、代码阶段吞吐、prefix cache 命中及 27B 测量结果见[性能参考](docs/performance.md)。数据对应重建后的执行包，测试配置、预热次数和计时口径均在该文档中说明。

Index-Translate-2B 的量化权重约 **1.23 GB / 4.32 bit**。开发版 Rust CLI 固定长度基准中，512／2K／8K prefill 约 **7.6K／7.9K／6.5K tokens/s**，关闭 MTP 的 decode 约 **108／105／94 tokens/s**；普通请求路径约 **113／109／98 tokens/s**。关闭 MTP 的 150 TPS 目标尚未达到。测量口径、文本快速评测和图片质量限制见[同页记录](docs/performance.md#index-translate-2b2026-10-18)。

Index-Translate-9B 的量化权重约 **5.44 GB / 4.51 bit**，复用相同执行组件并保留独立 embedding/输出头。Rust CLI 固定长度基准中，512／2K／8K prefill 约 **2153／2220／1956 tokens/s**，关闭 MTP 的 decode 约 **33.4／32.6／29.8 tokens/s**。测试范围与 BF16 快速质量对照见[9B 测量](docs/performance.md#index-translate-9b2026-10-18)。

Qwen3.5-4B 的量化权重约 **2.58 GB / 4.42 bit**，复用相同动态执行组件和共享 embedding/输出头。Rust CLI 固定长度基准中，512／2K／8K prefill 约 **3279／3401／2798 tokens/s**，关闭 MTP 的 decode 约 **54.45／52.34／45.68 tokens/s**。测量口径与小样本 BF16 对照见[4B 测量](docs/performance.md#qwen35-4b2026-10-19)。

目前只支持上述硬件和架构，不支持视频、音频、`n>1` 或云端存储接口。JSON Schema 支持范围见[服务使用指南](docs/serving.md)。量化质量使用同源 BF16/FP8 的固定场景和同历史概率对照；完整 LLM benchmark 和独立 BF16 256K 质量评测尚未覆盖，见[质量评测](tools/eval/README.md)。

## 开发与文档

```bash
git clone https://github.com/iMMIQ/orinfer.git
cd orinfer
cargo fetch --locked
make build       # 仅构建 Rust；工具链固定在 rust-toolchain.toml
make python-env  # uv 创建并同步 CPU .venv
make check       # 格式、编译、clippy、Rust/CPU 测试
```

自行量化、编译 TileLang 算子或重新构建模型时才需要离线 Python 环境；下载已发布模型及匹配执行包后可直接部署。执行包的 `.so + cubin` 格式见[执行包接口与构建](docs/model-packages.md)。

- [构建与发布](tools/build/README.md)：编译镜像、CPU/GPU 检查与发行包。
- [模型与执行包](tools/model/README.md)：离线准备、布局、MTP 和上下文扩展。
- [服务使用](docs/serving.md)：API、图片、缓存、并发和资源配置。
- [图文构建](tools/vision/README.md) · [API 验证](tools/api/README.md) · [TileLang kernels](kernels/README.md)。

在线代码位于 `crates/orinfer-{engine,api,cli}`，模型适配器及计算策略位于 `crates/orinfer-models`，公共 ABI 位于 `crates/orinfer-model-sdk`；GPU 算子与离线工具分别位于 `kernels/`、`tools/`。

## 许可证

**LGPL-3.0-or-later**，见 [LICENSE](LICENSE) 和 [COPYING](COPYING)。第三方依赖及模型权重遵循各自许可证。
