# Offline quantization

Use the project uv environment for CPU tools and the fixed compiler image for GPU fitting. These codecs are experimental. Validate each checkpoint against an original BF16/FP8 reference; execution completion and quality acceptance are separate. Short scene tests establish initial quality evidence, not general benchmark performance.

`e8p.py` generates our integer basis from lattice geometry without importing an external quantizer. `e8p_gpu.py` fits original floating matrices using a fused TileLang nearest-neighbor encoder, input rotation, and two weight-only scale updates. The stored table is part of the format: indices must never be interpreted with a different basis or basis ordering.

Flash Next 的固定源转换、safetensors 发布、完整请求和 MTP 验证、原始 BF16 teacher 用法见[模型准备](../model/flash_next/README.md)。执行路径使用自有 E8P 编码；下面的其他 codec 保留为独立量化研究工具，不属于 Flash Next 部署路径。

`vq.py` defines independent experimental integer-vector fixtures in standard safetensors. VQ4 has U8 `[N,K/4]` indices and I8 `[256,4]` tables. E8P has U16 `[N,K/8]` indices and an even-integer I8 `[256,8]` basis; sign/parity and the unit shift reconstruct the final integer coordinates exactly. Both use F16 `[N]` scales and either empty or I8 `[K]` rotation signs. `gpu_layout()` uses group-major indices and little-endian packed U32 tables. Optional U16 `[N,K/128]` patches contain one valid bit, eight signed-value bits and seven position bits, adding exactly 0.125 bit/weight; replacement values use the same row scale.

Block128 rotations are normalized input-side Hadamard transforms. The corresponding floating weight transform is mathematically equivalent before quantization; A8/FP16 boundaries add numerical error. Store/charge signs, tables, patches, scales and padding in the footprint, and include activation transformation in complete-FFN performance. These codecs do not establish full-model quality, and the E8P decoder is not a reproduction of the full QuIP# quantization/training pipeline.

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

输出仅为量化权重，仍需支持该编码的 27B 加载与算子构建，不能直接交给现有 27B 部署加载器。
