# Offline quantization

Use the project uv environment for CPU tools and the fixed compiler image for GPU fitting. These codecs are experimental. Validate each checkpoint against an original BF16/FP8 reference; execution completion and quality acceptance are separate. Short scene tests establish initial quality evidence, not general benchmark performance.

`e8p.py` generates our integer basis from lattice geometry without importing an external quantizer. `e8p_gpu.py` fits original floating matrices using a fused TileLang nearest-neighbor encoder, input rotation, and two weight-only scale updates. The stored table is part of the format: indices must never be interpreted with a different basis or basis ordering.

Flash Next 的固定源转换、safetensors 发布、完整请求和 MTP 验证、原始 BF16 teacher 用法见[模型准备](../model/flash_next/README.md)。执行路径使用自有 E8P 编码；下面的其他 codec 保留为独立量化研究工具，不属于 Flash Next 部署路径。

`vq.py` defines independent experimental integer-vector fixtures in standard safetensors. VQ4 has U8 `[N,K/4]` indices and I8 `[256,4]` tables. E8P has U16 `[N,K/8]` indices and an even-integer I8 `[256,8]` basis; sign/parity and the unit shift reconstruct the final integer coordinates exactly. Both use F16 `[N]` scales and either empty or I8 `[K]` rotation signs. `gpu_layout()` uses group-major indices and little-endian packed U32 tables. Optional U16 `[N,K/128]` patches contain one valid bit, eight signed-value bits and seven position bits, adding exactly 0.125 bit/weight; replacement values use the same row scale.

Block128 rotations are normalized input-side Hadamard transforms. The corresponding floating weight transform is mathematically equivalent before quantization; A8/FP16 boundaries add numerical error. Store/charge signs, tables, patches, scales and padding in the footprint, and include activation transformation in complete-FFN performance. These codecs do not establish full-model quality, and the E8P decoder is not a reproduction of the full QuIP# quantization/training pipeline.

## 两级整数 E8P Q4A8（实验）

`e8p_q4.py` 的每个 U32 保存两个 E8P U16 码字，描述 8 个权重，载荷恰为 4bit/weight。默认还原为 `q = 4 * E8P(coarse) + E8P(residual)`，整数范围为 `[-55,55]`；两个阶段共享原有 basis。系数是整个 codec 的常数，每个输出行只有一个 FP16 scale，没有逐 group 浮点缩放。保存格式为标准 safetensors，包含 `packed/scales/signs/basis`，metadata 固定 codec、码字顺序、旋转及系数。

CPU `fit()` 是数值参考；`e8p_q4_gpu.Encoder.fit()` 复用 TileLang 最近邻编码器，按真实存储的 FP16 scale 交替拟合两个码字。转换时由调用方分块，例如每次 128 个输出行，不需要完整 BF16 checkpoint 或校准数据。`Weights.dequantize()` 返回旋转坐标中的浮点权重；与原矩阵比较时须使用同一 `vq.rotate()`。

`Encoder.refine(raw, original)` 对已有 `[4,1]` 码字执行 8 次交替更新和两轮 beam8 联合搜索；`raw` 必须是对应的原始浮点权重，不能用已量化权重反量化后替代。每次保留误差更低的行码字及其真实 FP16 scale，不改变存储格式、旋转或在线解码器。调用方同样按输出行分块。

粗码字 top-8 由 `kernels/model/e8p_encode.py` 精确搜索：先求 512 个 basis/parity 组的最优距离，以第八小的组距离界限筛选，再展开可能命中的符号组合。保留边界并列组；距离比较带 FP32 舍入余量，接近的候选用 FP64 复核，以码字顺序打破并列。无需生成 `[vectors,65536]` 距离矩阵；八个候选的残差最近邻批量执行。该 top-8 是精确搜索，但整个两级交替拟合仍不是全局最优加性编码的保证。

```python
from tools.quantization.e8p_q4_gpu import Encoder

encoder = Encoder()
initial = encoder.fit(raw, signs)
improved = encoder.refine(raw, initial)
```

`kernels/model/e8p_q4.py` 在投影内部解码到 INT8，复用 `integer_core` 的 Tensor Core/DP4A 调度与完整 K 的 INT32 累加。支持动态 token 行及输出 stride，不保留常驻 W8 副本；激活使用同一 signed-block128 旋转和 A8 行 scale。27B 构建器和 Rust 行执行器接受隔离候选的 `orinfer_e8p_q4_int8` 策略，要求固定 `[4,1]` 系数、U32 `[N,K/8]` 码字和匹配的 basis；不改变现有标量 Q4 部署权重。模型质量需另做 BF16 对照。

```bash
.venv/bin/python -m unittest tools.quantization.test_e8p_q4
bash tools/operators/run.sh tools/quantization/check_e8p_q4.py artifacts/check-e8p-q4
bash tools/operators/run.sh tools/quantization/check_e8p_search.py artifacts/check-e8p-search
```

真实权重组件比较可运行 `evaluate_e8p_q4.py`，使用同一 wrapper，追加 `--weights /path/to/fixtures [--full-matrix /path/to/matrix.safetensors]`。输入为含 F32 `weight` 和 I8 `signs` 的 safetensors，编号样本另附同名 `.json` 源信息；应从固定版本的原始 BF16 转换，不能使用已量化权重作为质量参考。输出分别记录权重重构误差和投影 kernel 时延；这两项不能替代模型 NLL、任务质量或完整请求 TPS 验收。

`block_ldlq.py` provides regularized covariance factors and a CPU oracle for reverse block error feedback following the BlockLDLQ principle. `rotation.py` provides offline forward/inverse transforms for the Flash Next dimensions using an order20 Paley Hadamard and a power-of-two factor. A two-sided transform must restore the output coordinates before nonlinearities; its reference representation is not an online operator package. These helpers do not include QuIP# fine-tuning or full-model calibration.

## Qwen 27B 码表 Q4A8

`qwen35.py` 从固定 revision 的原始 BF16 分块转换，复用 Flash Next 的 `row_q4.py` 编码；`row_q4_gpu.py` 是其有界 GPU 拟合实现。128 维带符号 Hadamard 旋转后，每个权重保存 4bit 索引，码表为 `±(3,10,19,31,46,67,93,127)`，每个输出行保存一个 FP16 scale。运行时的目标是查表还原 INT8 并累加完整 K，不在内循环逐 group 缩放。它是标量非线性码表，不是两级残差 E8P。

embedding、GDN/QSA 输出投影与 MTP 大矩阵保留 W8；视觉、norm 等保留原精度。报告将 scale、共享旋转符号、码表及全部例外权重计入实际 bits/weight。此权重策略未经过完整模型质量验收，不提供校准或 MTP 接受率保证。

源目录需要 `config.json`、原始 `model.safetensors.index.json`，以及记录 `repo` 和固定 `revision` 的 `source.json`；tokenizer 等前端资源也放在该目录。远端权重按字节范围读取；已有原始分片时可加 `--source-dir`。无需完整保存远端 BF16：

```bash
bash tools/quantization/run.sh qwen35 \
  --source /path/to/pinned-source-metadata --output artifacts/quantization/qwen27b
```

`--inventory-only` 仅清点；`--chunk-mib` 默认 32。GPU 编码按块持有共享锁，块完成后同步并释放临时显存。重复相同命令会检查已完成分块并继续；源、编码实现、seed 或分块配置变化会拒绝混用。最终按组件/层合并标准 safetensors，验证后删除临时分块副本。

输出仅为量化权重。继续使用[27B 原生构建器](../model/qwen35/README.md)生成 Rust/TileLang 执行包和在线模型目录，再执行 `validate-model` 与实际生成检查；量化中间目录不能直接部署。
