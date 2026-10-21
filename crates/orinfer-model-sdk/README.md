# orinfer-model-sdk

独立模型执行包的共享数据契约和 C ABI v1。模型包的依赖限定为此 SDK；Rust 实现可以导出 `cdylib`，其他语言实现可使用 [C 头文件](include/orinfer_model.h)。

配置与计划描述在创建阶段序列化；batch 使用 C 数组传递。内存由分配方释放，原生库保持加载到对象销毁。SDK 中的 Rust trait 用于包内实现，跨库调用统一使用 C ABI。

协议和资源所有权见[模型执行包](../../docs/model-packages.md)。ABI 版本与 crate 版本分别管理；修改跨库结构或函数表时，必须同步升级 ABI 版本。
