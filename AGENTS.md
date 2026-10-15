# 工程约定

本文件适用于整个仓库；用户明确指令优先。

## 范围

- 硬件固定本机Jetson AGX Orin 64GB，aarch64/CUDA SM87。
- 支持Qwen3.8-27B（Qwen3_5）和Qwen3.8-Flash-Next（qwen4_exp）的文本、图片和多图输入；不包含视频。
- 在线运行时用Rust，生产GPU kernel以TileLang为源；Python只用于离线转换、编译、参考和验证。
- 模型实现、公共执行组件、TileLang kernel和离线构建工具在统一workspace维护；不按模型×精度强制拆仓。orinfer-models分别注册模型适配器与计算策略，部署包内Rust cdylib通过版本化C ABI生成执行程序、动态batch和状态绑定，匹配的TileLang cubin与库同包发布。公共CUDA执行器不硬编码模型族、层数、维度或层序；新增架构不增加核心分派枚举。
- 第一阶段包含prefix cache与多请求batching；缓存恢复和batch调度必须保持完整KV/GDN/卷积/MTP状态及请求隔离。
- 项目许可证为LGPL-3.0-or-later，保留第三方版权和许可证声明。

## 实现

- 按真实checkpoint配置核对GDN、GQA、RoPE、zero-centered RMSNorm、embedding和head语义。
- GDN持续状态为FP32；KV、卷积、GDN和位置状态均属于请求私有状态。
- 改变prefill/decode、恢复或分支行为时校验全部状态，不只校验KV。
- 单份常驻W4；scale、zero、LUT、padding和持久重复表示计入权重平均bits。临时W8不作永久权重缓存。
- Flash Next扩展采用Q2A8和质量优先的混合精度；其QSA长上下文KV使用group-64对称INT8及FP16 scale，压缩索引与未完成块也纳入请求私有状态；Q2专家直接索引固定地址的压缩权重库，默认全量常驻；`--resident-experts N` 使用模型包声明的专家bank与路由计数，按需从SSD加载到CUDA VMM页，并共享全局SLRU。有限模式暂关闭CUDA Graph，MTP专家全量常驻；临时路由集合和页粒度不计作严格字节上限，不缓存整份展开W8。HC保留BF16权重；PLE表使用受预算限制的CPU缓存，其n-gram历史与卷积历史必须纳入请求状态及prefix恢复。
- Flash Next采用已确定的混合量化方案：专家保留整数E8P Q2A8；主干GDN QKV/Z、QSA Q/gate和target/MTP共享输出头采用signed-block128旋转、输出行scale的Q4A8。GDN/QSA输出投影、共享FFN和MTP草稿Q/O保留W8；其余精度敏感部分维持原混合精度。原生适配器校验该策略，不保留逐矩阵试验精度分支。
- Flash Next使用我们自己的整数E8P＋旋转量化表示，运行时反量化到INT8计算，不沿用社区Q2量化编码。当前优先完成算子和真实端到端推理，允许不校准或小样本校准；原始BF16按矩阵或专家分块读取，不需要保存完整BF16模型。完整校准需单独验证专家覆盖和层间误差，不能把反量化后的社区Q2称为原始BF16基线。
- E8P可使用无损K64压缩权重重排及完整A8位平面分解，以两个INT4 Tensor Core乘积恢复原INT32点积；这不是A4量化，必须保留原scale与FP16边界，并验证非零码本前提。未验证的MTP、并发及大prefill配置保留INT8路径，读取同一份压缩bank。
- 在线模型采用HF配置/tokenizer目录和`cache/model.json`数据描述；标准分片safetensors保留当前W4表示，默认在线校验dtype、shape、物理layout和容器边界；权重payload hash由`--verify-weights`显式开启，`validate-model`始终完整校验。配置签名仍校验，前端资源、CPU资产元数据、原生执行库和kernel身份仍固定并校验hash。模型数据通过execution_package digest固定独立模型执行包，第一阶段采用INT8为主、质量优先的混合精度，关键部分可FP16/FP32；不实现force_int8或其他格式导入。离线AOT manifest只作构建中间产物，在线模型数据描述与执行包均保留schema 1，只支持当前原生模型包结构；不提供旧结构或schema 2兼容分支。算子夹具格式独立。
- 服务协议位于orinfer-api；CLI只处理命令。CUDA执行器、加载器、模型适配器、计算策略与生成/视觉/MTP控制分离；权重、序列状态和workspace分作用域管理。
- 执行包catalog仅保存共享编译定义/资产及动态launch模板；逐层tensor角色与参数映射以共享调用模板、逐层参数表和去重特化表离线编译进Rust模型库；禁止在生成模块逐层复制整份特化绑定，经orinfer_model_plans_v1的有owner C数组导出，在线不解析展开的kernels列表或Python计划。目录身份与原生绑定注册表必须匹配，重建后校验完整程序/参数及全部请求状态。plan-model保留构建契约以支持离线重建；在线只导出最终程序、局部特化和专家常驻契约引用的绑定，不将融合前的构建输入注册为可执行launch。
- Rust按实际导出host ABI绑定参数、grid/block/shared；检查CUDA返回值、unsafe边界、资源生存期和stream顺序。
- Flash Next普通decode（1–128请求）、混合batch和MTP验证共享1–1024 token行的target执行模板；单／多流草稿和warm共享1–512行draft模板，输出头复用相同实现。因果段、接受前缀因子、捕获、提交和请求隔离独立处理，不复制投影／HC／FFN／专家计算流程。整数投影和专家按1/2/3/4/8/16/32离线特化，其余行数使用动态参数cubin；TileLang自动调优只选择不改变scale、dtype与归约边界的合法tile，路由与专家tile必须一致。运行时消费编译生成的行数选择表，单流也使用相同模板；辅助算子不保留8/16分界。不得恢复旧的逐容量在线batch planner、空行补齐或混合decode子组回退；离线中间绑定只用于构建最终模板，MTP短尾也走一行packet，单请求target程序仅保留状态准备／参考用途。Graph分支按模板槽位注册，全部特化验证读写依赖后复用；不手写batch档位，不在线调优；缓存按源码、编译器和设备身份失效。
- CUDA graph需要稳定地址和预分配workspace；改输入后replay验证实际执行。
- 共享片段与完整packet复用同一套Graph分支捕获；片段只启用完整且无歧义的fork/join区间。动态槽位展开的连续辅助kernel属于同一边界，重复的私有状态槽位不得关闭无关共享分支；优先级按实际launch grid判断。模型声明支持batch验证时，单请求MTP也复用Verification packet，不另走逐算子验证链。
- Flash Next因果段与MTP状态控制使用动态长度kernel：target段及draft最大512行，验证与接受前缀恢复最大8行；运行时显式传递真实段长，不从程序名推断。GDN head-major因子按真实行数定stride，卷积/PLE历史与隐藏环使用发布buffer的物理几何。协议允许长度、调度分块和轻量程序别名不代表独立cubin；不得恢复逐长度状态kernel枚举。验证/恢复/捕获必须沿动态行launch传递段长，并保持每个请求的owner及状态视图。
- Flash Next最终包通过prefill_tuning.py和converge_prefill.py离线调优并发布局部选择表，不手写性能分界；只发布被选择的实现。混合因果段及MTP warm支持连续1–512行，不按16/128/512拆段；剩余长度与延迟预算决定实际片段。MixedRows在现有ABI v1 flags中显式携带前置prefill片段数，禁止从行数猜测prefill/验证角色；缓存身份包含该角色边界。warm用一个min_tokens=1的容量模板，MtpLength必须在映射动态KV前更新。最终执行包重打包资产，移除未引用的候选及旧cubin。共享大行数计算通过row_tuning.py与converge_kernels.py测量发布，不保留32/128人工性能分界；专家路由与上下投影共同选择。HC injection 的直接与矩阵 tile 统一BF16 MMA K16累加顺序，单行参考也保持相同契约；不得把并行FP32求和与Tensor Core当成数值相同的候选。动态绑定合并只改身份，保留原始grid、实参表达式和token到专家行倍率。动态MTP在映射KV之前写入真实验证长度。视觉使用一个动态patch模板，merger launch按实际patch数除以4；提高视觉容量不得扩大现有文本程序的行数契约。MTP验证、恢复、捕获各用一个容量8、min_tokens=2的入口，运行时传递真实长度，固定形状参考包仅作离线构建输入。
- Cooperative launch需显式声明并校验occupancy，不假定普通launch可跨block同步。
- 性能以完整链及真实模型为准。合成探针、微测外推和四输出profile不能替代完整请求TPS。
- 质量对照官方BF16或同模型FP8，固定seed20261002及teacher-forced历史；token不完全一致不是自动失败。任务结果、target NLL和长上下文另行判断。

## 开发

- 先用rg检查调用者和依赖；保持修改小而完整。不要默认启动subagent。
- CPU Python依赖用uv管理项目`.venv`，通过`make python-env`同步固定版本，不直接更新主机Python环境。`make check`执行Rust格式/编译/clippy/测试及CPU评测检查；GPU验证用`bash tools/operators/run.sh`的固定编译镜像。
- GPU工作通过`artifacts/gpu-experiment.lock`串行，保留其他服务；不自行改全局频率、功耗或清缓存。
- 当前原生Flash Next的整数算子重建使用`tools/model/flash_next/build.py`：从权重几何和host ABI读取构造契约，真实权重调优，编译对象按内容校验并原子缓存，最终模型经Rust计划检查后原子发布。它需要模型固定的执行契约，不是任意裸权重导入；HC/QSA/状态及专家联合选择表保留输入已验证实现，专家真实几何另行验证。不得将历史升级脚本串联目录作为正式重建接口。
- 可变的HC无损指数、QSA radix shift使用ABI标量复用cubin；私有状态包装器必须保留标量、列stride及字节偏移。QSA条带数离线测量；动态整数投影调优覆盖多个实际行数及容量上限，不只测一个短长度。相邻选择区间仅在完整参数/launch契约一致时合并，不跨越空洞或隐式改变专家路由tile。
- 新kernel验证数值、非对齐尾部和实际graph replay；状态改动另测分块、恢复和请求隔离。
- 投影组件按权重读取、计算调度、输出处理组合：W8/Q4及融合SwiGLU复用integer_core，BF16矩阵复用floating_core，E8P的位平面、双INT4乘积和输出复用e8p_core。这些是编译期组件，不因模型、prefill/decode或MTP复制计算循环；访存调度、浮点归约和舍入契约显式保留。27B的码表Q4A8接入复用同一Q4解码和整数计算组件。调优缓存必须包含内联组件源码身份。
- Flash Next适配器仅接受当前动态prefill、target/draft行模板和因果状态契约；不恢复旧固定容量计划的注册、回退或逐长度改写。Q4前置变换按section、激活/scale工作区及符号表关联消费者，QKV/Z共享变换必须先于首个消费者，并对每个程序别名单独生成；禁止以算子序号补丁代替依赖关系。
- 源码库保留实现、必要构建工具、测试、示例与使用文档。不提交研究过程、agent交付记录、历史实验报告或本机环境证据。
- 权重、二进制、cubin、cache、日志及trace保留在忽略的artifacts/target目录；不提交密钥或token。
