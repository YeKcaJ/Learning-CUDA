# 选题 2 总结报告

本报告保留 2026-09-09 阶段的实现和测量口径；后续两种融合量化优化见
[优化日志](../OPTIMIZATION_LOG.md)，当前目录与命令以 [核心文档](../Core/README.md) 为准。
2026-09-11 的结构整理将旧工程移到 legacy、CPU reference 移到 Core/reference，以下旧路径属于历史记录。

日期：2026-09-09。项目：MXFP8 / NVFP4 低精度软件模拟、量化、打包和 CUDA 反量化。
作者目录：YeKcaJ。源码及配置的使用方式见 [README](../README.md)。

## 完成范围

| 要求 | 实现与验证 |
|---|---|
| FP32 / FP16 矩阵输入 | 带尺寸与 dtype 标识的小端二进制；FP16 输入与其 FP32 精确展开对照 |
| 配置文件 | Python 标准库 TOML，支持题目所有配置字段，非法值明确报错 |
| MXFP8 | E4M3FN 数据、E8M0 scale、默认 32 元素分组 |
| NVFP4 | E2M1 数据、E4M3 局部 scale、FP32 全局 scale，默认 16 元素分组 |
| per-tensor / block-wise | 两种模式均实现；tensor 为明确标注的软件对照模式 |
| nearest / stochastic | 冻结 nearest 规则；按 seed/index 重现的随机元素舍入 |
| 真实 4-bit 存储 | 两元素一字节，尾部高 nibble 补零，读回校验 padding |
| FP32 / FP16 / BF16 输出 | GPU 内转换后直接写目标类型数组，保存时区分 magic |
| 误差、压缩率、性能 | 144 组误差记录；kernel、主机调用、CPU 对照和三种输出性能 |
| 测试 | 每格式 29 个 CTest、固定 golden 哈希、GPU 三类 sanitizer |
| nsys 分析 | 两种格式均成功采集 CUDA kernel/API/传输数据，并指导 scale kernel 优化 |

国产平台适配（摩尔线程、沐曦、华为昇腾）尚未实施；实际 T4、原生 FP8/FP4 路径也未验证，不计入本次完成声明。适配计划见 [国产平台适配记录](PLATFORM_ADAPTATION.md)。

## 格式与数学过程

详细字段和舍入规则分别见 [冻结 v1 规范](REFERENCE_SPEC.md) 与 [扩展 v2 规范](EXTENDED_SPEC.md)。
默认块边界按矩阵行主序展平后划分，不按每行重新开始。

MXFP8 每组求最大绝对值 m，用 `e=clamp(ceil(log2(m/448))+127,0,254)` 保存 E8M0 指数，
scale 为 `2^(e-127)`。量化 `x/scale` 为 E4M3FN，反量化 `decode(code)*scale`。
全零组 e=127；极小值除法下溢时 e=0。E4M3FN 有 1 位符号、4 位指数、3 位尾数，最大有限幅值 448。

NVFP4 先求张量最大绝对值 M，计算 `g=M/(6*448)`；全零时 g=1，正数下溢时采用最小正 FP32。
组最大值 m 对应局部 scale `b=encode_e4m3(m/(6*g))`，有效 scale 为 `g*decode_e4m3(b)`。
元素编码为 E2M1，其幅值表是 `{0,0.5,1,1.5,2,3,4,6}`。局部 scale 舍入后再用于元素编码，
确保量化和反量化使用同一有效 scale。有效 scale 为零时输出正零，避免 0/0。
偶数元素位于字节低四位、奇数元素位于高四位，每次 packed store 由一个线程独占整个字节。

tensor 模式只保存一组局部 scale；NVFP4 仍保留全局乘局部公式。它是题目要求的策略对照，
不是把标准 MXFP8/NVFP4 的固定 32/16 元素布局改名。v2 文件记录 group、模式、舍入及 seed。

nearest 沿用 CPU 冻结枚举的等距规则，而非 IEEE nearest-even。stochastic 根据相邻幅值
间的相对距离决定上舍入概率，scale 本身仍采用 nearest。FP16/BF16 最终转换使用 nearest-even。
有限离群值属于误差测试；NaN/Inf 量化输入明确拒绝，不声称存在有意义的有限量化误差。

## 实现与优化历程

1. CPU reference 以枚举候选定义可复现答案，冻结输入、payload 和 FP32 反量化输出。
2. v1 CUDA 与 frozen golden 对照，修复 NVFP4 尾部 packed 写越界和零 scale 分支。
3. 公共输出模板实现三种 dtype，独立 16 位 oracle 检查转换边界，不以 FP32 容差判断 16 位结果。
4. v2 增加独立 `Workspace` 管理显存和 event，重复测试复用资源。旧 CLI 和旧 benchmark 不删除。
5. NVFP4 全局最大值改为多 CTA 局部归约和第二级归约，GPU 上直接计算 g，消除量化中间的主机回传。
6. E4M3 由枚举 256 编码改为正幅值二分；中点与相邻 FP32 值仍逐字节匹配原 CPU。
7. 首次将二分用于 E2M1 反而更慢，改为七个固定中点比较；保留此负结果，未把试验性方案当作成功优化。
8. nsys 发现逐小块 scale kernel 成为 NVFP4 瓶颈，改为每 256 线程处理多组 16/32 元素，
   使用 width=16/32 的 warp shuffle 归约，减少小 CTA 和 shared-memory 同步。
9. 默认块索引使用编译期常量除法；tensor 模式固定 scale 下标为零，避免逐元素运行时 64 位除法。

本次没有尝试融合到 GEMM 或推理引擎，因此只声明格式模拟与量化/反量化算子的性能，不声称模型端到端加速。

## 正确性与复现

| 检查 | MXFP8 | NVFP4 |
|---|---:|---:|
| CTest | 29/29 | 29/29 |
| v1 输入和 golden SHA256 | 15/15 | 15/15 |
| v2 生成/边界用例 | 299 | 299 |
| 新快速编码候选样本 | 10760 | 10046 |
| 旧/新路径 memcheck | 0 errors，0 bytes leaked | 0 errors，0 bytes leaked |
| 旧/新路径 racecheck | 0 errors，0 warnings | 0 errors，0 warnings |
| 旧/新路径 synccheck | 0 errors | 0 errors |

新用例覆盖 0~65、255/256/257/1023/1024/1025 长度，tensor/block、nearest/stochastic、
正负零、极小有限值、scale 下溢、输出溢出。FP16 输入使用 Python `struct` 的 binary16 解码
进行独立展开对照。文件测试检查截断、尾随字节、非法版本、尺寸溢出、配置错误、保存后独立读取。
随机中点上舍入频率约 0.49，固定 seed 可重现且改 seed 会改变编码；这不替代完整随机数质量评估。

输入矩阵实验共 `2 dtype * 3 distribution * 2 format * 2 mode * 2 rounding * 3 output = 144` 组，
全部 CPU 量化字节与 GPU 输出比较通过。日志归档及 SHA256 清单位于
[validation-20260909](records/_archive/2026-09-09-legacy-cli/validation-20260909/summary.json) 和 [SHA256.json](records/_archive/2026-09-09-legacy-cli/validation-20260909/SHA256.json)。

GPU 检查器使用 CUDA 12.9.79 官方独立包；压缩包 SHA256：
`e23aad21132ff58b92a22aad372a7048793400b79c625665d325d4ecec6979bf`。
本机临时路径 `/tmp/quant-tools.2mGHlO/cuda_sanitizer_api-linux-x86_64-12.9.79-archive/compute-sanitizer/compute-sanitizer`。
清理临时目录后需重新解压并通过 SANITIZER_BIN 指定，不依赖这个路径永久存在。

## 误差与压缩率

矩阵尺寸 128x129，seed=1234；uniform 为 [-3,3]，normal 为 N(0,1)，outlier 在正态输入两端加入 +/-1000。
下表为 FP32 输入、默认 block、nearest、FP32 输出，误差相对原始输入：

| 格式 | 分布 | 最大绝对误差 | MAE | MSE |
|---|---|---:|---:|---:|
| MXFP8 | uniform | 0.124997 | 0.034683 | 0.002206 |
| MXFP8 | normal | 0.201626 | 0.018006 | 0.000711 |
| MXFP8 | outlier | 24.000000 | 0.020909 | 0.070478 |
| NVFP4 | uniform | 0.499591 | 0.133239 | 0.030989 |
| NVFP4 | normal | 0.483356 | 0.071848 | 0.009300 |
| NVFP4 | outlier | 2.197041 | 0.072834 | 0.010478 |

此尺寸的 payload 压缩率：MXFP8 为 3.8788 倍，NVFP4 为 7.1080 倍；包含输入/输出文件头后
分别为 3.8641 和 7.0594 倍。FP16 输入的 payload 压缩率减半，不能照抄 FP32 的压缩率。
NVFP4 在本组离群值上的最大误差小于 MXFP8，不说明它普遍更准确；其 normal/uniform MAE 更高。
tensor 模式下，NVFP4 outlier MAE 从 block 的 0.07283 升到 0.80005，说明局部缩放保护普通值的意义。
所有模式及三种输出的完整原始结果见 [144 组汇总](records/_archive/evaluation/evaluation-warp-20260909/summary.json)。

## 性能结果

GPU：RTX 3060 Laptop，6 GiB，驱动 596.08；Ubuntu 24.04 WSL；CUDA 12.0.140，Release，sm_75。
设备代码未使用 Ampere 以上指令；在本机运行 sm_75 代码通过，但未在真实 T4 上实测。
硬件、编译器、CPU 信息和源码哈希见 [environment.json](records/_archive/2026-09-09-legacy-cli/performance-warp-20260909/environment.json)。
串行测量，预热 3 次、正式 20 次，基线/优化顺序交替。设备 event 计时不含 H2D/D2H 或分配。

| 格式 | 元素数 | 枚举+小 CTA 基线 ms | 优化 median ms | 优化 P95 ms | 加速比 |
|---|---:|---:|---:|---:|---:|
| MXFP8 | 1M | 0.566272 | 0.124880 | 0.152576 | 4.53x |
| MXFP8 | 4M | 2.237392 | 0.468352 | 0.847872 | 4.78x |
| MXFP8 | 16M | 8.769024 | 1.642496 | 2.674688 | 5.34x |
| NVFP4 | 1M | 0.322560 | 0.143360 | 0.156672 | 2.25x |
| NVFP4 | 4M | 1.153536 | 0.452608 | 0.872448 | 2.55x |
| NVFP4 | 16M | 4.548096 | 1.755648 | 2.736128 | 2.59x |

基线也采用新的两级全局归约及相同 Workspace；表中加速来自编码和组内 scale 归约优化，
不是把原始 CLI 的端到端时间除以新 kernel 时间。NVFP4 全局归约改造没有单独做控制变量加速比。

| 格式 | 16M FP32 反量化 ms | FP16 ms | BF16 ms | 16M 量化 host_api ms |
|---|---:|---:|---:|---:|
| MXFP8 | 0.406048 | 0.365056 | 0.381392 | 19.634121 |
| NVFP4 | 0.341856 | 0.304032 | 0.334848 | 20.015056 |

16M 优化量化逻辑吞吐为 MXFP8 51.392 GB/s、NVFP4 43.600 GB/s。
CPU reference 在 1M 量化上中位数分别为 1345.309 ms 和 93.764 ms；GPU 含分配/传输/释放的
量化调用分别为 2.3465 ms 和 1.8261 ms。CPU 是为清晰和确定性编写的串行枚举参考，
不能把此差异当作对优化 SIMD/多线程 CPU 库的加速承诺。CPU 预热一次、测三次。
所有规模、输出 dtype、P95 和逻辑 GB/s 见 [完整性能表](records/_archive/2026-09-09-legacy-cli/performance-warp-20260909/RESULTS.md)。

## nsys 分析与局限

nsys 2024.6 与当前驱动组合曾无法取得 kernel 数据，升级到 2026.1.3 后可采集。
采集单一 4M 规模，命令与原始结果由 `tools/profile.py` 保存。
最终 NVFP4 报告中旧 `build_scales` 中位数约 1.285 ms，多组 warp 版本约 0.393 ms，
验证小 CTA 调度与同步确为可改善部分。不同路径调用次数不相同，应比每次耗时，不直接比累计占比。
结果见 [NVFP4 nsys](records/_archive/2026-09-09-legacy-cli/profiling-warp-20260909/nvfp4_stats.txt) 和
[MXFP8 nsys](records/_archive/2026-09-09-legacy-cli/profiling-warp-20260909/mxfp8_stats.txt)。

nsys API 耗时含 CPU 等待，不能与 GPU 时间相加。报告也包含首次初始化、预热和 host_api 分支，
最终性能数字取无 profiler 的测试。笔记本 GPU 的电源、温度、WSL 调度影响 P95，未锁频。
数据驻留模式重复同一输入，缓存可能影响小规模结果；逻辑字节数不包含归约临时流量。
误差实验的单次 kernel 时间用于日志完整性，不替代上述正式基准。后续可做 pinned-memory 异步流水、
算子融合、真实 T4/国产 GPU 验证、ncu 分析和独立库封装。
