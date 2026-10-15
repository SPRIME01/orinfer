# orinfer-models

统一 workspace 内的原生模型实现。模型适配器与精度策略分别注册，共享执行组件、TileLang源码、构建工具和测试；部署时生成 `.so + cubin` 包。

当前支持 Qwen3.8-27B（Qwen3_5）及 Flash Next 的 `int8_quality`，包含混合精度、融合、batch、MTP和视觉位置语义。模型库只链接 `orinfer-model-sdk`，不依赖 `orinfer-engine`；跨库仅使用 C ABI v1。

Flash Next 只接受当前动态执行契约：prefill 1–4096行、共享target最多1024行、draft/warm最多512行、MTP验证2–8行。普通decode、混合请求和MTP复用计算模板，状态所有权、捕获与提交分别处理；不再构造旧的固定batch或逐长度执行计划。Q4使用固定混合精度策略，输入旋转按消费者的工作区与符号表绑定，在共享QKV/Z的首个消费者之前执行一次。

```bash
cargo build --release --locked --offline -p orinfer-models
cargo test --locked --offline -p orinfer-models
```

构建、包契约和验证见[模型执行包](../../docs/model-packages.md)。
