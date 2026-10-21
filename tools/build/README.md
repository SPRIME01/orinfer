# 构建、检查与发布

在线服务只依赖 Rust 二进制、CUDA 驱动、准备好的模型目录和算子包。Python、PyTorch、TileLang 均属于离线环境。

## 编译环境

```bash
make compiler-image
make check-offline
```

`compiler.Dockerfile` 使用公开 NVIDIA Jetson CUDA 12.6/PyTorch 2.9.1 镜像的固定 digest，使用固定版本 uv 管理镜像内的 `/opt/venv`，安装 TileLang 0.1.15、Transformers 5.20.1、Tokenizers 0.23.3 和 Hub 2.2.0，模型和编译缓存通过运行时挂载提供。`compiler-requirements.txt` 固定完整 Python 依赖和构建工具，由 uv 同步并移除基底遗留的无关服务依赖。Jetson 原生 Torch 2.9.1/torchvision 0.24.1 保持不变，构建后检查 Torch 仍链接 CUDA12.6；系统依赖由基底镜像 digest 固定。GPU 编译环境保留 Python3.10 和 NumPy2.2.6；TVM FFI、Z3 遵守 TileLang 约束。默认镜像名为 `orinfer-compiler:0.1.2`，可用 `ORINFER_OPERATOR_IMAGE` 覆盖。

GPU 编译入口：

NVRTC 后端使用 CUDA Python 12.6.2.post1；部分离线原生算子参考保留 Triton 3.5.1。两者与 TileLang 一同固定在编译镜像中，在线部署只加载已编译的原生执行包。

```bash
ORINFER_CHECKPOINT_DIR=/path/to/converted \
  bash tools/operators/run.sh tools/model/qwen35/build.py artifacts/build/qwen35 \
  --source /path/to/converted --model-output artifacts/models/qwen35-ready \
  --gpu-lock-held
```

GPU 入口使用调用者 UID/GID 及补充设备组，生成的模型/cache 归调用者所有。仓库与输出目录按当前绝对路径挂载；外部 checkpoint 通过 `ORINFER_CHECKPOINT_DIR` 只读挂载。模型输入约束与构建参数见[模型构建说明](../model/README.md)。27B 构建器读取完成量化的自有码表权重；Flash Next 重建器读取当前原生模型及其固定执行契约。两者均按已声明的原生量化与执行契约构建。

新环境可使用合成输入完成最小 GPU 编译验收：

```bash
bash tools/operators/run.sh tools/build/smoke.py artifacts/build/compiler-smoke
```

它从空缓存编译实际生产 norm 与动态行数 greedy 归约，检查非对齐尾部、3/7 行共用 cubin、仿射 shape 的标量 ABI、输出 guard 和改变输入的 graph replay，并导出 AOT。

## 自动检查

`make check` 包含 Rust/Ruff 格式、Python 静态检查、编译、clippy、Rust/CPU Python 测试和验收配置校验；GitHub Actions 在 x86_64 和 aarch64 runner 执行同一入口；原生 C ABI 加载测试在 aarch64 执行，读取 `rust-toolchain.toml`。`make fmt` 统一格式，`make lint` 执行格式和静态检查；Rust 编译将死代码视为错误，clippy 以零警告为通过条件。使用 `uv` 隔离 CPU 依赖：`make python-env` 创建 Python3.12 的 `.venv` 并同步固定版本；检查默认使用 `.venv/bin/python`，CPU 依赖隔离在虚拟环境中，可通过 `PYTHON` 覆盖解释器；CI 和编译镜像固定 uv0.13.1，Rust 版本由 `rust-toolchain.toml` 固定。Python 源码仍兼容 GPU 编译环境的 3.10 语法。

`make check-offline` 在编译镜像中检查全部离线模块引用，并在 CPU 上执行需要 Torch 的参考数学及动态行数 ABI 测试。

```bash
make check-gpu MODEL=/path/to/prepared-model OUTPUT=artifacts/check-gpu/new-run
```

该入口生成原生 Chat 模板的文本、图片、多图夹具，固定 seed20261002，依次验证 off/decode_only/full 模式的动态 batch、取消/重排/复用隔离、请求丢弃清理和同历史概率。原串行计划只在测试构建中作为独立参考存在，生产整请求生成和服务使用同一状态机。独立 BF16 质量验收另行执行。

GPU 入口获取 `artifacts/gpu-experiment.lock`；锁被占用时退出。运行时保持功耗、频率及全局缓存设置；先释放自己部署的空闲服务，保留其他服务。

## 发布

提交源码、更新 workspace 版本并建立指向该提交的 annotated tag 后，运行：

```bash
python3 tools/release/package.py --version v0.2.0 \
  --operators /path/to/qwen3_5-execution.tar.xz \
  --operators /path/to/flash-next-execution.tar.xz \
  --output artifacts/releases/v0.2.0
```

默认仅生成和校验产物。工具要求工作区干净，使用 `Cargo.lock` 构建，验证 aarch64/版本，打包完整源码、许可证、依赖声明、模型打包工具和架构契约，并逐个验证原生执行包及实际库身份。引擎、执行包和随附的源码归档统一使用 XZ 压缩（`.tar.xz`，preset 6）；输入旧 `.tar.gz` 执行包会经过验证并重新执行 XZ 压缩。资产名称取自架构规格 ID、计算策略、GPU 目标和版本，例如 `orinfer-exec-qwen35-2b-a8-sm87-v0.2.0.tar.xz`。规格 ID 由 `configs/architecture-contract.json` 中的配置签名选择器识别；`a8` 对应 `int8_quality` 混合精度策略。每种规格分别发布，重复规格与策略会阻止发布；`RELEASE.json` 保留完整包 digest 和架构规格 ID；完整离线工具及 Flash Next 文档随包提供。权重通过模型仓库单独发布。部署使用 Rust CLI 的 `orinfer package install` 安装执行包；详见[安装说明](../../docs/model-packages.md#安装执行包)。

从 0.1.2 起，引擎使用包含 `.so` 与 cubin 的 schema 1 原生执行包。0.1.1 及更早格式的模型需通过当前构建工具重新发布。当前原生格式的模型可以用[执行库更新工具](../../docs/model-packages.md)生成新目录，保留权重和 kernel 资产。

实际发布时添加 `--publish --notes /path/to/release-notes.md`。工具核对 tag 与源码 revision，推送 tag，创建 GitHub Release，再重新下载并核对所有上传资产的 SHA256。每个 release 以独立版本发布，已发布资产保持原样。
