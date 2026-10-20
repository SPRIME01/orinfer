# TileLang kernels

目标为CUDA SM87。`operators/`包含基础算子，`model/`包含模型使用的投影、融合、GDN和attention实现。

kernel经离线编译生成cubin和真实host ABI，在线由Rust CUDA Driver执行。导出格式与模型组装入口见[模型工具](../tools/model/README.md)。新增实现应检查完整数值、非对齐尾部、状态语义和改变输入后的graph replay。

## 共享计算组件

- `model/integer_core.py`提供W8、标量码表Q4及双W8融合SwiGLU共用的DP4A/Tensor Core循环、尾部处理和输出缩放。权重loader按编码选择；gate/up各自先舍入FP16再执行SwiGLU。外层工厂只保留几何校验、ABI和launch映射。两个调度由离线调优选择，不按模型或请求场景复制。
- `model/floating_core.py`提供通用浮点投影、分组BF16、HC投影与视觉Linear共用的MMA循环。输入布局、操作数方向和顺序group归约在编译时固定；K尾部在tile内补零，HC的K16累加与中间舍入不能被平行归约替换。视觉的Conv3D bias舍入与Linear/GELU输出仍由各自epilogue保持。
- `model/e8p_core.py`提供完整A8位平面拆分、双INT4 MMA和输出恢复/缩放。普通分组、K-major流水和唯一路由缓存输入共享算术组件，保留各自的数据移动调度。唯一路由路径要求每专家至多一行，由调用方的路由契约保证，不绑定某个模型的E/N/K。

组件通过TileLang宏内联，在线不解释Python或选择codec。编译后的host ABI仍是权重/状态buffer与必要的动态行参数。Qwen 27B 的两级 E8P Q4A8 复用相同 E8P 算术、旋转/A8 和动态行调度；其编码与 K128 布局由构建器校验。

BF16视觉epilogue使用显式PTX RNE舍入/写入，保留Conv3D的独立dot舍入及Linear/GELU的精度边界，避免NVRTC生成的BF16向量类型往返转换。FP16保留自动向量化。`tools/vision/validate_kernels.py`验证两种精度、K/N尾部和修改输入后的Graph replay。

`tools/model/flash_next/checks/shared_compute.py`用独立CPU整数点积验证W8/Q4、FP16/FP32输出与融合SwiGLU，包括非对齐N、动态尾部及改输入后的Graph replay。浮点和专家另外验证既有归约/路由契约；不能只比较共用同一组件的两个实现。

`model/moe.py`提供FP32 top-k路由、确定性激活分发和输出合并；A8量化及融合SwiGLU复用`operators/op30_activation_quantization.py`。整数E8P专家通过`model/e8p_exact.py`、`model/e8p_prefill.py`等调度在Tile内解码，不缓存整份展开权重。

`model/hyperconnection.py`提供Qwen4 gated residual的分支归一化、低秩投影、混合及残差更新，保留BF16权重和FP32累加。小行数路径将down/SiLU和up/分支混合分别融合，保持原始FP16中间舍入。`model/ple.py`提供PLE门控、膨胀卷积及历史更新；CPU的n-gram哈希与有容量限制的E8P行缓存入口为`tools/model/flash_next/ple.py`。验证入口为`tools/operators/hyperconnection.py`和`ple.py`。

`model/integer_vq.py`保留离线构建和验证所需的E8P INT8参考实现，以及输入侧block128旋转。`model/rotation_a8.py`将旋转、可选SwiGLU和行级A8量化融合，保留原始FP16舍入边界。旋转必须在对应投影前执行，down的旋转位于SwiGLU之后。数值验证入口为`tools/operators/e8p_selector.py`，整模型状态和请求验证入口见[Flash Next](../tools/model/flash_next/README.md)。

`model/int8_projection.py`与`model/q4_projection.py`提供行scale的W8和码表Q4投影。`tools/model/flash_next/native.py`组装离线参考模型；正式整数算子重建入口为`tools/model/flash_next/build.py`，部署包由Rust在线执行。QSA使用`model/qsa.py`的group-4压缩索引、精确radix top-k及group-64 INT8 KV，`model/qsa_attention.py`用共享KV tile的FP16 Tensor Core计算attention，softmax和累加保持FP32；上下文上限为262144 token。压缩索引、pending ring和KV scale均纳入请求状态。算子验证入口为`tools/operators/qsa*.py`，真实长上下文执行入口为`tools/model/flash_next/long_context.py`。转换和独立BF16对照用法见[量化工具](../tools/quantization/README.md)。

`model/flash_mtp.py`提供MTP的完整HC输入归一化、embedding/分支融合和逐位置历史保存；`model/greedy.py`支持多行验证输出。目标模型验证时保存卷积、PLE及QSA pending历史；GDN保存键、衰减和FP32更新量，以`model/gdn_sequence.py`重放接受前缀到FP32状态，避免逐位置复制完整矩阵。按接受长度提交，KV和压缩索引的拒绝后缀通过live cursor隔离。`tools/model/flash_next/mtp.py`负责离线greedy接受、草稿刷新及完整session恢复；MTP共用主模型的embedding和输出头，另有独立INT8 KV。算子验证入口为`tools/operators/flash_mtp.py`，真实请求对照入口为`tools/model/flash_next/checks/mtp.py`，融合、GDN提交和专家直接索引验证入口为`tools/operators/flash_mtp_optimization.py`。

`model/sampling.py`提供共享GPU随机采样：FP64惩罚与bias处理，先top-k再nucleus截断，按分数及token ID稳定排序。概率质量使用有界U64定点数，最大未归一化截断误差为`vocab / scale`，scale由词表大小确定并保证总和不溢出。整数radix累计不依赖batch组成；随机数仍由Rust按seed和已提交位置产生。计数、mask及直方图均为可重建workspace，不进入prefix快照。普通采样和无需概率报告的grammar采样走GPU；top_logprobs与MTP的p/q概率及拒绝校正保留CPU参考路径。算子数值、尾部、异常输入及改变输入后的Graph检查使用`tools/operators/sampling.py`。
