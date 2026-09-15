# 国产平台适配记录

## 当前状态

当前正式实现和性能结论基于 NVIDIA RTX 3060 Laptop、CUDA/WSL。华为昇腾、摩尔线程和沐曦平台尚未完成编译、运行和性能验证，因此不能声称已经完成国产平台适配或认证。

## 适配目标

计划依次验证：

1. 摩尔线程 MUSA
2. 沐曦 MXMACA
3. 华为昇腾 CANN / Ascend C

## 保持不变的公共部分

- 输入文件和 v2 packed 文件格式；
- MXFP8/NVFP4 的 scale、编码、舍入公式；
- CPU reference/golden；
- FP32、FP16、BF16 输出定义；
- 误差、压缩率和日志字段。

## 每个平台需要适配的部分

- 设备内存申请、拷贝和释放；
- kernel 启动和 event 计时接口；
- `maximum`/`finalize_max` 归约；
- MXFP8/NVFP4 量化与 packed 写出；
- FP32/FP16/BF16 反量化输出；
- CMake/toolchain 和设备架构参数。

MUSA 和 MXMACA 可先评估 CUDA API 兼容层；昇腾需要使用 Ascend C 重写线程、片上内存和归约实现，不能只替换编译器名称。

## 验收标准

每个平台必须使用同一组 CPU golden 完成：

- packed data 逐字节一致；
- scale 逐字节一致；
- NVFP4 每字节两个 4-bit 元素，尾部高 nibble 为零；
- FP32/FP16/BF16 输出一致；
- 1M、4M、16M 性能数据；
- 最大绝对误差、MAE、MSE、压缩率和逻辑带宽；
- 平台、驱动、编译器和运行时版本记录。

## 性能优化口径

CUDA 版本的量化 kernel 已完成多轮优化。迁移到国产平台后不直接复用 CUDA 的性能结论：不同平台的带宽、向量宽度、线程模型和特殊函数吞吐不同，必须先完成正确性适配，再针对该平台重新 profile 和调优。workspace 复用、减少拷贝等端到端优化也需要在每个平台重新测量。

## 远程操作流程

```text
本地公共源码/CPU golden
    -> git push
远程平台 clone/pull
    -> 使用平台 SDK 编译
    -> 运行同一组 golden 和边界测试
    -> 保存性能/环境日志
    -> 回传结果并更新本文件
```

连接信息不写入仓库；使用 SSH key，不在脚本或文档中保存密码。
