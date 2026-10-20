# 离线量化

**简体中文** | [English](../../docs/en/tools/quantization/README.md)

CPU 工具使用项目 uv 环境，GPU 编码使用固定编译镜像。编码、存储重排和执行包构建分别完成；量化产物通过独立质量验收后才能部署。

## 当前表示

| 用途 | 权重表示 | 缩放与计算 |
| --- | --- | --- |
| Flash Next 专家及 PLE | 整数 E8P Q2，U16 描述 8 个权重 | 输出行 FP16 scale，解码到 INT8 |
| 27B 主干与输出头 | 两级整数 E8P Q4，U32 描述 8 个权重 | 固定系数 `[4,1]`，输出行 FP16 scale，A8 计算 |
| Index-Translate-2B/9B 主干 | 两级整数 E8P Q4 | 固定系数 `[4,1]`，输出行 FP16 scale，A8 计算 |
| Flash Next 部分非专家投影 | 旋转标量码表 Q4 | signed-block128，输出行 FP16 scale，A8 计算 |

敏感部分保留 W8、FP16、BF16 或 FP32。scale、basis、旋转符号、padding 和例外权重均计入实际空间，不保存常驻展开 W8 副本。

## E8P 编码与布局

`e8p.py` 从格点几何生成整数 basis，`e8p_gpu.py` 使用 TileLang 最近邻编码器拟合原始浮点矩阵。basis 及其顺序属于格式身份；`vq.py` 提供 E8P 码字解码、selector 表、无损 K64 重排与 safetensors 数值夹具。

`e8p_q4.py` 是两级 Q4 的 CPU 参考与存储定义。`e8p_q4_gpu.Encoder.fit()` 分块拟合两个码字，`Encoder.refine(raw, original)` 以原始浮点权重执行交替更新及 beam8 联合搜索。当前 27B 使用 `[4,1]` 和 beam8；两个阶段共享 basis，每个输出行只有一个 FP16 scale，没有逐 group 浮点缩放。保存的码字为 U32 `[N,K/8]`，构建器无损重排为计算用的 K128 布局。

调用者按输出行分块持有原始权重，例如每次 128 行；`raw` 必须来自原始 BF16/FP8，不能用社区量化权重的反量化值替代。存储、FP16 scale、旋转和系数均须与执行包匹配。

```bash
.venv/bin/python -m unittest tools.quantization.test_e8p tools.quantization.test_e8p_q4
bash tools/operators/run.sh tools/quantization/check_e8p_q4.py artifacts/check-e8p-q4
bash tools/operators/run.sh tools/quantization/check_e8p_search.py artifacts/check-e8p-search
```

`check_e8p_direct.py` 验证完整 A8 位平面和两个 INT4 Tensor Core 点积恢复原 INT32 结果；这不是 A4 量化。`evaluate_e8p_q4.py` 比较有原始权重来源的组件夹具，输出重构误差与投影时延；模型 NLL 和完整请求性能另外测量。

## 权重准备与参考

Flash Next 的固定来源转换、PLE、非专家权重准备和原始 BF16 teacher 见[模型构建](../model/flash_next/README.md)。

`row_q4.py` 与 `row_q4_gpu.py` 实现当前标量 Q4 组件，码表为 `±(3,10,19,31,46,67,93,127)`。`qwen35.py` 为 27B 分块准备标量 Q4、W8 例外及原始张量；该入口本身不执行两级 E8P 编码或 beam8 优化。27B E8P 部署应向[原生构建器](../model/qwen35/README.md)提供已完成两级编码的 checkpoint，而不能把标量 Q4 准备完成视为 E8P 完成。

源信息需包含固定的 repository/revision、配置和 safetensors 索引。远端权重可按字节范围读取；已下载分片通过 `--source-dir` 复用。准备工具核对来源、编码实现、seed 和已完成分块，不混用不同来源或不完整产物。

`qwen_e8p.py` 支持 Index-Translate-2B/9B 和 27B，从官方 BF16 分片直接执行两级拟合和 beam8 搜索。输入目录包含原始 HF 文件以及记录固定 `repo`、40 位 `revision` 的 `source.json`；`--shard FILE` 可逐片编码。已完成块保存校验和，完整分片写入后才清理其临时块；调用方可释放已编码源分片。embedding、共享输出头和视觉大矩阵也量化为 E8P Q4；敏感输出投影和 MTP 矩阵保留 W8，patch、位置表、norm 等较小敏感参数保留原始数据。2B 的 tied 输出头读取同一份压缩 embedding，不保存展开副本；9B 按官方配置分别编码 embedding 和输出头，不改变权重共享语义。

```bash
bash tools/quantization/run.sh qwen_e8p --source-dir /path/to/original \
  --output artifacts/quantization/index-translate-2b-e8p-q4a8
```

极小 E8P 查询按原 FP64 评分的 norm/code 顺序直接返回 top-8，避免等距候选搜索退化；阈值保持原评分结果，不将模型权重截成零，也不减少 beam 宽度或迭代次数。`check_e8p_search.py` 包含穷举、极小值和换输入 Graph 对照。

`rotation.py` 提供分块与完整 Hadamard 变换；`block_ldlq.py` 保留协方差因子及误差反馈的 CPU 数学参考，不表示已完成整模型校准或微调。激活量化、非线性和舍入边界不会因旋转而自动等价。

量化质量使用同模型原始 BF16；资源不足时可用明确标注的 FP8。固定 seed 20261002 与 teacher-forced 历史，比较任务结果、目标 NLL、概率和长上下文行为。token 选择不完全相同不自动判失败，局部重构误差也不能单独证明整模型质量。
