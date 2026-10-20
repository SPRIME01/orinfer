# 模型执行包

**简体中文** | [English](en/model-packages.md)

在线引擎通过版本化 C ABI 加载模型执行逻辑。模型适配器、计算策略、共享执行组件、kernel和构建工具在同一源码workspace维护。部署包包含原生 `.so`、匹配的 cubin 和配置/布局契约；权重仍在模型目录中。部署包不要求对应独立源码仓库。A8/A4 是计算策略，包应声明实际混合精度，不代表所有算子精度相同。

核心负责 API、调度、CUDA context/stream、内存、graph 和 prefix 索引；包负责配置校验、模型执行顺序、融合、动态 batch、状态绑定和视觉位置。增加模型族不需要在核心添加枚举或分派分支。当前原生库为 `orinfer-models`，注册 Qwen3.5 和 Flash Next 的 `int8_quality` 路径。

```text
MODEL_DIR/
  config.json, tokenizer.json, chat_template.jinja, ...
  cache/model.json
  cache/weights/*.safetensors
  cache/cpu/                         # 可选 CPU 查表数据
  cache/packages/<sha256>/
    package.json
    lib/model.so
    lib/bindings.rs, lib/BUILD.txt    # 生成模块的对应源码与重建方法
    assets.safetensors               # 或独立 cubin/source/host ABI 文件
    LICENSE
```

`model.json` schema 1 的 `execution_package` 固定整个包。共享缓存位于 `~/.cache/orinfer/packages`，可通过 `ORINFER_EXECUTION_CACHE` 覆盖。包 schema 1 / runtime ABI 1 描述目标 SM87、架构与计算策略、支持的配置、buffer 契约、kernel ABI 和 `.so` 身份。包自身通过 `describe` 返回身份及能力，运行时与 manifest 交叉检查。部署包以内容摘要固定；公共 ABI 的兼容性与源码组织分别管理。破坏协议兼容性的变化必须更新 ABI。

`.so` 使用 `orinfer-model-sdk`，不依赖 `orinfer-engine`。入口 `orinfer_model_v1` 返回 C 函数表，包含 describe/create/batch/visual 和对应资源释放函数。完整 C 声明位于 [`orinfer_model.h`](../crates/orinfer-model-sdk/include/orinfer_model.h)。Rust 数据结构及 trait 不跨动态库边界。

包内 `catalog` 只保存共享资产、编译实现定义和动态行数 launch 模板。定义包含 cubin/source/host ABI 索引、symbol、block、shared memory、cooperative、实际参数类型顺序及只读契约；不保存层号、权重名、逐层参数、grid 或完整执行程序。同一 cubin 可有多个不同 ABI/launch 契约，不能仅按二进制 hash 合并定义。

离线编译得到的参数映射编译为 `.so` 内的 Rust 张量角色表、共享调用模板、逐层参数表和去重特化选择表；该映射只记录实现与存储角色，执行顺序和融合仍由注册的 Rust 模型适配器生成。目录与映射分别固定身份，交换实现、参数类型、偏移或动态 launch 后必须重新构建匹配的库。运行时校验目录身份、buffer dtype/shape/layout 和边界，不依赖 Python 或在线编译。

同类层共享调用模板和特化表，每层仅提供层号、权重、状态、workspace 角色及必要标量。模板实例化严格保留实参顺序、slice 偏移、标量位模式、grid 和动态 launch 契约；不会从权重名称推测计算或精度。独立 continuation 参数查找直接查询模板，不为查一个参数展开整个注册表。此因子分解在发布时逐条做精确往返校验，适用于两种模型适配器。

Flash Next 的 HC 无损缓存指数和 QSA radix shift 通过已有 ABI 的整数标量绑定，共享相同计算实现；独立请求的 QSA 复用同一私有状态包装器。QSA 条带数离线测量选择。相邻行数区间仅在全部实参、资产及动态 launch 契约相同时合并。整数投影的固定行数和动态行数使用相同模板与候选空间，动态回退在多个实际行数上测量，选择结果随编译器、设备和源码身份失效；发布包不包含未选中的候选，不在运行时调优。

在线创建只导出最终程序、全部局部特化候选和专家常驻契约引用的绑定；融合前的构建输入不注册为可执行 launch。`plan-model` 和 `validate-model` 通过内部创建参数 `build_bindings` 保留完整构建契约，供离线重建、完整资产校验和诊断。它不改变 schema 1 或 C ABI 1；运行时的实际逐层执行节点仍需实例化，CUDA Graph replay 不查询或解释模板。

动态 launch 的全行数验证只在离线构建使用的 `plan-model` 和 `validate-model` 执行，相同容量和表达式只验证一次。正常加载检查每个绑定的参数类型和容量 ABI；实际 launch 继续检查真实行数、算术溢出、除零和非零 grid，不在启动时穷举所有行数。此行为由引擎实现，无需重建执行包。

创建时用 JSON 传递配置、数据契约与小型编译目录。独立入口 `orinfer_model_plans_v1` 用 C 数组导出生成的 kernel 参数绑定和 kernel/copy/zero 程序；主 `orinfer_model_v1` 表保持 ABI 1。导出接口显式声明表及记录大小，数组、字符串和嵌套参数由同一个 owner 管理并释放。逐层绑定和完整程序不经过 JSON 序列化往返。动态 batch 热路径继续使用 C 数组，一次返回指令、buffer views、动态 launch 参数和请求状态绑定。原生内存由分配方释放，库保持加载直到模型及 CUDA 资源释放完成。模型实例与 GPU 执行线程绑定。

有限专家常驻模式通过创建请求的 `expert_contracts` 选择更完整的原生绑定契约。模型库返回 `expert_banks`：专家数、按专家连续存储的张量，以及消费 kernel 的 I32 路由计数参数位置。核心按实际绑定读取计数，管理全局 SLRU 和 CUDA VMM 页，不识别模型专属名称或维度。默认全量常驻只构建模型计划所需的验证契约；不从执行包 JSON 读取逐层 `kernel_contracts`。

模型可声明 hash 固定的 `input_assets`，通过独立可选入口 `orinfer_model_inputs_v1` 接收执行程序名、当前 token 和请求历史，返回命名 buffer 的上传数据；原有 ABI v1 函数表不变。核心校验访问权限与上传范围，并在库释放数据前完成上传。Flash Next 在包内完成 CPU W8 embedding／E8P PLE 查表，有预算的行缓存共享，n-gram 历史从请求提供。CPU 资产随模型克隆和执行包更新一起保留。主模型、验证和 shifted draft 输入由模型包区分，核心不注册架构专属输入分派。

MTP 的验证计划可声明始终提交紧凑 recurrent 更新，也可提供 draft 分支的保存/恢复程序；调度器在提出草稿前保存、刷新已提交前缀前恢复。Flash Next 使用这些通用接口保留 GDN、卷积、PLE、HC 和 QSA pending/index 的完整状态。

包不创建独立 CUDA context 或隐藏 stream，不直接管理引擎的显存。核心执行明确的程序；CUDA graph 命中时不再跨包重建程序。状态绑定缓存最多 16 个槽位/段长组合，仅缓存逻辑名称，GPU 地址仍从当前请求 arena 解析。prefix 身份包含包 digest；不同执行策略和状态布局不会复用旧快照。

## 构建与更新

统一源码入口：

- `crates/orinfer-engine`：通用运行设施，不注册具体模型。
- `crates/orinfer-model-sdk`：计划、状态和原生 C ABI 契约。
- `crates/orinfer-models`：模型适配器、独立的精度策略选择和注册表。
- `kernels/`：可复用的 TileLang 算子与融合实现。
- `tools/`：离线编译、权重准备、组装与验证。

适配器描述架构语义与执行组合，策略决定可用精度路径；同一策略可被多个适配器支持，融合可以跨执行组件边界。只注册已经验证的架构/策略组合，不为尚未实现的模式提供占位回退。部署时只绑定模型配置实际需要的 kernel 资产。

```bash
make build
# 或仅构建原生模型库：
cargo build --release --locked --offline -p orinfer-models
```

该产物提供注册的模型逻辑；可部署库还需要离线构建工具生成匹配当前 kernel ABI 和物理 tensor 角色的绑定模块。发布器通过 `ORINFER_BINDINGS_SOURCE` 编译该模块，原生库只依赖 SDK，不链接引擎；结果位于包的 `lib/model.so`。包同时保存生成的 `lib/bindings.rs` 和 `lib/BUILD.txt`，与发行版内的 workspace 源码及 Cargo.lock 一起用于重建该库；这些文件不参与在线执行。

在保留权重和 cubin 的情况下，重建原生库和目录：

```bash
PYTHONPATH=. .venv/bin/python -m tools.model.catalog \
  --model /path/to/prepared-model --output /path/to/rebuilt-model
```

工具需要项目固定 Rust 工具链及已缓存的 Cargo 依赖，使用 `--locked --offline` 构建；同文件系统的不可变权重通过 hardlink 复用，模型描述和 `.so` 使用独立文件。构建输出经加载器验证后原子发布，不修改源目录。普通在线启动仅需要已发布的包，不需要 Rust 编译器。

`tools.model.attach_execution` 可替换**已具有相同绑定注册表**的原生库；未带绑定表的普通 workspace 库或目录身份不同的库会被拒绝。更新注册模型逻辑后优先使用上述重建工具。

在线数据与包均保留 schema 1；当前加载器只接受 `catalog` 和原生绑定导出接口，不保留逐层 `kernels` 包或 schema 2 的兼容加载分支。离线构建输入仍可包含展开的 bindings，用于精确对照与生成 Rust 模块。

## 安装执行包

下载引擎、匹配的执行包和 `SHA256SUMS` 后，先检查下载内容。安装命令不依赖 Python、CUDA 初始化或 Rust 编译器：

```bash
sha256sum --ignore-missing -c SHA256SUMS
orinfer package install /path/to/package.tar.xz
orinfer plan-model /path/to/new-model
orinfer validate-model /path/to/new-model
```

安装器按文件内容识别 `.tar.xz` 和旧 `.tar.gz`，校验 package digest、schema/ABI、SM87 目标、aarch64 ELF 库和全部独立／safetensors 内嵌 kernel 资产的 SHA256；安装期间不执行库代码。路径越界、链接和重复归档成员被拒绝，全部校验成功才原子写入缓存，失败清理临时目录，已安装的同 digest 包不会被覆盖。解压与校验只发生在安装时，不增加后续启动工作。

默认安装位置与运行时一致：优先 `ORINFER_EXECUTION_CACHE`，其次 `$XDG_CACHE_HOME/orinfer/packages`，最后 `~/.cache/orinfer/packages`。`--cache /path/to/packages` 只改变本次安装位置；自定义位置需要同时为后续运行设置 `ORINFER_EXECUTION_CACHE`。

`plan-model` 会在 CPU 加载原生执行库并生成计划；`validate-model` 还完整校验权重、CPU 表和全部 kernel 文件，均不初始化 GPU。在线启动默认只检查大体积权重和 CPU 表的结构，`--verify-weights` 开启完整内容校验；配置签名仍校验，前端资源、CPU 资产元数据、执行库和 kernel 身份仍校验 hash。JSON 创建请求的可选 `verify_weights` 字段由模型库处理 CPU 表，C ABI v1 和 schema 1 不变；更新执行库后即可使用新的 CPU 表加载行为。

### 安装已发布的 v0.1.3

`package install` 尚未包含在 v0.1.3 二进制中。该版本下载并通过 `SHA256SUMS` 校验后，可用系统 tar 将执行包的 digest 目录解压到共享缓存，再运行 `validate-model`：

```bash
export ORINFER_EXECUTION_CACHE="$HOME/.cache/orinfer/packages"
mkdir -p "$ORINFER_EXECUTION_CACHE"
tar -xf /path/to/package.tar.gz -C "$ORINFER_EXECUTION_CACHE"
orinfer validate-model /path/to/prepared-model
```

### 离线归档

模型构建者使用 Python 工具生成新的 `.tar.xz` 执行包；普通部署无需此步骤：

```bash
PYTHONPATH=. .venv/bin/python tools/model/package.py archive /path/to/package /path/to/package.tar.xz
```

## 合并发布布局

离线调优结束后，将 GPU 权重合并成约 2 GiB 的标准 safetensors 分片，CPU 查表按连续编码段合并。张量名称、物理布局、量化位模式及 CPU 行序保持不变；更新索引、CPU 文件偏移与身份 hash。模型目录只包含模型数据，匹配的执行包独立写入指定缓存：

```bash
PYTHONPATH=. .venv/bin/python tools/model/compact.py \
  --model /path/to/prepared-model --output /path/to/compact-model \
  --package-cache ~/.cache/orinfer/packages
orinfer validate-model /path/to/compact-model
```

Flash Next 的正式权重发布采用稳定的组件和层组名称：主模型专家每三层一片，其余原权重每八层一片，Q4 投影、输出头、HC 无损压缩表示、MTP 和视觉各自独立；CPU 表复用已合并的连续行分片。单片默认不超过 2 GiB，组件超限时拒绝发布。重打包保留每个张量的字节、布局和身份，不重新量化或编译 kernel：

```bash
PYTHONPATH=. .venv/bin/python -m tools.model.flash_next.repack \
  --model /path/to/optimized-model --output /path/to/published-model
orinfer validate-model /path/to/published-model
```

输出不包含 `cache/packages`。Hugging Face 和 ModelScope 模型仓库只发布配置、tokenizer、模型描述、GPU 权重与 CPU 表；匹配执行包通过 GitHub Release 独立发行，先安装到公共执行缓存再加载模型。模型描述中的 `execution_package` 固定对应包 digest，重打包不改变它。

包内 `assets.safetensors` 用具名 U8 tensor 收纳 cubin、生成源码与 host ABI，去重字节相同的资产。schema 1 的文件身份可带 `tensor` 字段，SHA256 校验对应 tensor 的原始字节；原生 `.so` 保持独立文件。引擎复用 bundle 的 mmap 和头部索引，直接读取具名资产，不解包成小文件；CUDA 模块仍分别加载。此布局需要支持具名资产的引擎版本。

模型托管仓库不包含执行包。包可单独归档、安装，模型通过 `execution_package` digest 在共享缓存中找到它。合并用于最终发布。离线调优工具通过 `plan-model` 从 Rust 库读取当前绑定，在编译阶段展开和修改，再重新生成原生模块及目录；在线格式保持一致。

## 验证

`make check` 统一执行引擎、SDK及模型 crate 的格式、clippy、单元测试，同时覆盖 C ABI 跨语言加载、错误拒绝和更新原子性。更新还应比较完整程序、launch 参数、buffer 与状态契约；GPU 验证需要匹配模型权重，覆盖请求状态、graph 输入变化、prefix 恢复及 MTP 分支。

通用加载链的 GPU 夹具可独立运行：

```bash
bash tools/operators/run.sh tools/model/validate_model_package_gpu.py artifacts/model-package/check01
```

夹具覆盖三种 graph 模式、输入变化和完整 logits，不代替真实模型质量或性能验收。

Flash Next 的当前原生包可通过 `tools/model/flash_next/build.py` 重新生成整数投影与执行包。该入口读取模型固定的架构绑定契约，按真实矩阵调优，支持内容校验与断点恢复；状态、浮点归约和联合专家组合保持输入中的已验证实现。输入须包含匹配的执行包，量化中间权重不能直接作为部署模型。用法见 [Flash Next 构建](../tools/model/flash_next/README.md)。
