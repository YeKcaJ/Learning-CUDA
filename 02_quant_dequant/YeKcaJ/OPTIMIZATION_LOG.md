# 性能优化记录

> 2026-09-10 目录迁移：正式源码位于 `Core/`，旧 `CUDACommon/pipeline*` 是转发文件。
> 操作请以 [Core 程序文档](Core/README.md) 为准，性能脚本使用 `Core/build/pipeline_mxfp8` / `pipeline_nvfp4`。
> 本文下面的 B0 数值、源码哈希和旧路径保留为采样时的历史记录；不能用新转发文件的哈希替代旧实验标识。
> 编码辅助已抽出为 `Core/codec.cuh`，编译单元发生变化，后续优化前需重新测量 baseline；此次迁移不宣称加速。

本文是本轮性能实验的工作文档：先说明当前程序，再固定基线，最后逐项记录修改。
已有完整功能报告见 [FINAL_REPORT.md](FINAL_REPORT.md)，不要把历史性能数据与本轮结果混用。

## 1. 本轮 baseline 是什么

**B0 = 当前 pipeline 默认计算路径，作为这轮新优化的起点。**

当前实现已经做过 warp 分组归约、快速编码等优化，但对接下来的实验来说，它就是“优化前”。
本轮目标是：同一输入、格式、缩放和舍入规则下，保持量化字节与 scale 不变，降低耗时。

| 名称 | 实际含义 | 本轮用途 |
|---|---|---|
| B0 | 当前 `pipeline` 的默认路径 | 后续优化前后比较的起点 |
| `quant_optimized` | benchmark 中默认路径的输出名称 | B0 和改进版都看这一行，再按结果目录区分版本 |
| `quant_enumeration` | pipeline 内置的小块 shared 归约 + 枚举编码对照 | 辅助对照，不是本轮 B0 |
| 旧 `build/benchmark` | `CUDACommon/benchmark.cu` 生成的旧端到端测试 | 本轮暂不使用 |
| CPU reference | 生成正确答案的串行 CPU 算法 | 验证正确性，不是 GPU 优化前版本 |

例如：比较 `opt01-baseline` 的 `quant_optimized` 与 `opt01-after` 的 `quant_optimized`。
不要拿旧枚举路径耗时除以改进版耗时，作为本轮改动的加速比。

## 2. 当前程序结构

### 2.1 从哪里进入程序

```text
正常处理文件：
tools/quantize.py run --config ...
    → Python 解析 TOML、选择格式
    → CUDAMXFP8/build/pipeline 或 CUDANVFP4/build/pipeline
    → pipeline_main.cu 的 main()
    → 读取输入 → GPU 量化 → GPU 反量化 → CPU 对照 → 保存文件和 JSON

正式性能测试：
tools/benchmark.py
    → 两个格式的 build/pipeline --benchmark 元素数 重复次数
    → pipeline_main.cu 的 benchmark()
    → 生成固定输入 → 预热 → 重复计时 → 汇总

nsys 分析：
tools/profile.py
    → nsys profile 包围同一个 pipeline --benchmark
    → 生成 .nsys-rep、CUDA kernel/API/传输统计
```

这三个入口最终使用同一套 pipeline 算子，但输入方式和测量目的不同。

### 2.2 各文件负责什么

| 文件 | 职责 | 优化时是否重点看 |
|---|---|---|
| [CUDACommon/pipeline.cuh](CUDACommon/pipeline.cuh) | CUDA kernel、编码、Workspace 显存管理、CPU 扩展参考 | **主要修改位置**；注意不要误改同文件内的 CPU oracle |
| [CUDACommon/pipeline_main.cu](CUDACommon/pipeline_main.cu) | C++ 入口、基准循环、日志、算法自测 | 确认计时范围和测试配置 |
| [CUDACommon/pipeline_io.h](CUDACommon/pipeline_io.h) | FP32/FP16 输入、v2 量化文件和模式参数 | 纯 kernel 优化通常不改 |
| [CUDACommon/output.cuh](CUDACommon/output.cuh) | FP32/FP16/BF16 输出转换、比较、保存 | 改输出路径时检查 |
| `CUDAMXFP8/main.cu`、`CUDANVFP4/main.cu` | 旧 CLI/kernel，并提供复用的编解码函数 | 不是当前默认 pipeline kernel 的位置；仍有依赖，不能直接删除 |
| `CPUMXFP8`、`CPUNVFP4` | 冻结 CPU 实现、固定输入和 golden | 不为追求 GPU 提速而更改答案 |
| `CUDACommon/tests/reference.cpp/.h` | 将 CPU 实现接入对照测试 | 正确性桥接 |
| `tools/benchmark.py`、`tools/profile.py` | 自动执行、汇总、保存环境和报告 | 保持实验条件一致 |

两个格式的 `pipeline` 由同一份源码编译而来。NVFP4 的 CMake 定义 `TEST_NVFP4`，
MXFP8 不定义。`pipeline_main.cu` 临时改名并包含旧 `main.cu` 来复用函数，随后提供自己的 `main()`。
CPU reference 在独立编译单元链接进来。目前不是已经拆好的独立算子库。

## 3. GPU 实际执行什么

以下是本轮固定的 **block + nearest** 模式，描述 `Workspace::launch_quant(false)`。

### 3.1 MXFP8 量化

```text
FP32 input
    → build_block_scales：每 32 个元素归约最大绝对值，生成 E8M0 scale
    → quantize_kernel<false>：归一化后编码成 E4M3，每元素 1 字节
    → data + scales
```

scale kernel 每个 CUDA 线程块 256 线程，处理 8 个量化组，使用 warp shuffle。
元素编码每线程处理一个元素，以二分查找选最近 E4M3 候选。
默认 block 模式不需要全局最大值归约；tensor 模式才需要。

### 3.2 NVFP4 量化

```text
FP32 input
    → maximum：多个 CTA 分别扫描，写局部最大绝对值
    → finalize_max：归约局部最大值，在 GPU 上计算 global_scale
    → build_block_scales：每 16 个元素归约，并将局部 scale 编码为 E4M3
    → quantize_kernel<false>：编码为 E2M1，两元素打包成一个字节
    → packed data + block_scales + global_scale
```

scale kernel 每个 256 线程块处理 16 组；只有各组首线程写 scale。
元素编码每线程处理两个元素并独占一个 packed 字节，避免两个线程竞争写高低 nibble。
nearest E2M1 使用七个固定中点比较。全局 scale 在 GPU 上计算，量化序列中间不回传 CPU。

### 3.3 反量化与显存

两种格式都调用 `dequantize_kernel<T>`：每线程恢复一个元素，FP32 中间计算后转为 T。
T 为 `float`、`__half` 或 `__nv_bfloat16`，分别对应 FP32、FP16、BF16。
NVFP4 先拆出 nibble，再乘 `global_scale * decode(block_scale)`。

`Workspace` 管理 input、data、scales、归约临时数组、输出显存和 CUDA events。
resident 测试在重复之间复用显存，输入只上传一次；host_api 测试每次重新分配、上传、下载和释放。
改变线程分工时必须继续保证尾部、负零、舍入和 packed padding 的行为。

## 4. 本轮固定实验条件

| 项目 | B0 条件 |
|---|---|
| 主分析规模 | 4M = 4,194,304 个元素；1M/16M 用于检查其他规模是否退化 |
| 输入 | FP32，1×N，标准正态分布，C++ mt19937 seed=20260909 |
| 量化格式 | MXFP8 和 NVFP4 分别记录 |
| 缩放/舍入 | block / nearest，块大小分别 32 / 16 |
| 反量化主指标 | FP32；另保留 FP16/BF16 结果 |
| 正式计时 | 每项预热 3 次，正式重复 20 次 |
| GPU | RTX 3060 Laptop，6144 MiB，驱动 596.08 |
| 编译 | CUDA 12.0.140，Release，当前两工程缓存指定 sm_75 |
| nsys | 2026.1.3，profile 脚本固定 4M、正式重复 5 次 |

**当前 benchmark 不读取 TOML。修改 configs 下的文件不会改变上述性能实验。**
`pipeline_main.cu::benchmark()` 直接创建默认 `Options`；TOML 只作用于 `quantize.py run`。
性能输入和误差实验的输入也不是同一份，不应把不同数据的误差和时间拼成一个实验。

### 4.1 哪一行才是要记录的时间

| 输出项 | 计时范围 | 用法 |
|---|---|---|
| `quant_optimized / resident_gpu` | CUDA event 包围整段设备量化序列，含归约、scale、编码，不含传输/分配 | **量化主要指标** |
| `dequant_fp32 / resident_gpu` | FP32 反量化 kernel | **反量化主要指标** |
| `quant_optimized / host_api` | 每次主机量化调用，含分配、上传、计算、下载和释放 | 单独观察端到端收益 |
| nsys 的 `Med (ns)` | 单个 kernel 的中位数 | 找耗时阶段，不直接当正式 benchmark 成绩 |
| `quantize.py run` 的单次日志 | 无正式预热/重复，可能受初始化影响 | 验证功能，不评价优化收益 |

本轮先记录 median 和 P95，单位统一 ms；nsys 的 ns 除以 1,000,000 才是 ms。
event 的整段序列时间不要求等于各 kernel 中位数之和，序列内还可能包含启动间隔。
API 等待与 GPU 执行可能重叠，不能直接把 CPU API、kernel、Memcpy 三张表相加。

## 5. B0 已有记录

正式基线已运行，不需要为了建立文档再覆盖它：

- [性能表](records/opt01-baseline/RESULTS.md)、[原始汇总](records/opt01-baseline/summary.json)。
- [采集环境与源码哈希](records/opt01-baseline/environment.json)，记录时间 2026-09-09 17:06:49 +08:00。
- [MXFP8 nsys 统计](records/opt01-profile/mxfp8_stats.txt)、[NVFP4 nsys 统计](records/opt01-profile/nvfp4_stats.txt)。
- 同目录 `.nsys-rep` 用于查看时间线，`.sqlite` 是可重新导出的缓存。

### 5.1 B0 正式性能（不带 nsys）

| 格式 | 元素数 | 量化 median ms | 量化 P95 ms | FP32 反量化 median ms | FP32 反量化 P95 ms |
|---|---:|---:|---:|---:|---:|
| MXFP8 | 1M | 0.124928 | 0.129024 | 0.031744 | 0.031744 |
| MXFP8 | 4M | 0.468320 | 0.468992 | 0.111616 | 0.113472 |
| MXFP8 | 16M | 1.636352 | 1.650688 | 0.403456 | 0.406528 |
| NVFP4 | 1M | 0.142336 | 0.200704 | 0.025600 | 0.059552 |
| NVFP4 | 4M | 0.449520 | 0.456448 | 0.088064 | 0.090016 |
| NVFP4 | 16M | 1.737728 | 1.744896 | 0.339968 | 0.340992 |

### 5.2 nsys 的初步观察（不是原因定论）

根据已保存的 4M 统计，当前 MXFP8 的元素编码中位数约 0.411 ms、block scale 约 0.099 ms；
当前 NVFP4 的 block scale 约 0.394 ms、元素编码约 0.088 ms。
因此 MXFP8 先研究编码成本，NVFP4 先研究 scale 编码和该阶段的线程利用。
nsys 只能先定位阶段，具体指令、寄存器或带宽限制需用控制变量实验或 ncu 验证。

nsys 报告中 `quantize_kernel<(bool)0>` 是当前路径，`<(bool)1>` 是枚举路径。
本脚本还采集内置对照、预热和 host_api 分支，调用次数不相同，不能按累计占比直接排名优化对象。

### 5.3 源码身份

建档时三个 pipeline 文件与 `opt01-baseline/environment.json` 记录的 SHA256 一致。
当前分支 `mxfp8-YeKcaJ`，HEAD 为 `5b449ca25fc5d27e16671657a18822a75f7849b1`。
**工作区存在未提交修改，HEAD 不能代表全部 B0 实现。哈希用于核对，不是可恢复的代码备份。**
首次改算子前，应单独保存 B0 源码副本或建立经过审查的 baseline commit；本次只建文档，未代你提交。

```text
CUDACommon/pipeline.cuh
6b64d028e5156567c97821cd87db44f7c158ef814812ee3356e0aae527708200
CUDACommon/pipeline_main.cu
7121feedbf9ff8a3d2a6657aae938f0cc0aeef0eb00e41c3bf0f8f9e63d1b1b5
CUDACommon/pipeline_io.h
e1415eb3b7486d3d568267e9797d30b8f3089473430ae42ecbea183382fb438a
```

这三个哈希不覆盖全部编译依赖。保留 B0 时还需包括两种旧 CUDA 源码、CPU reference、
output.cuh、reference 桥接、CMake 与测试脚本，不要只备份一个 kernel 文件。

## 6. 一次优化按什么顺序做

1. 选择一个对象，例如 NVFP4 block scale；写清 nsys 观察和一个待验证假设。
2. 保存 B0 源码，只做一项改动，不同时改格式、输入或舍入规则。
3. 重新构建并验证正确性，再测不带 profiler 的性能。
4. 重跑 nsys，检查目标阶段是否减少耗时、是否把开销转移到其他阶段。
5. 填下面的记录，结论为保留、撤回或待复测。一次更快不能说明稳定提速。

从项目根目录执行以下命令，输出目录必须是新名称：

```bash
# 会重新构建两个工程，运行 CTest 并核对冻结输入/golden 哈希。
bash CUDACommon/run_tests.sh

# 涉及访存或同步改动时，再指定可用的检查器，运行三类 GPU 检查。
# SANITIZER_BIN=/absolute/path/to/compute-sanitizer bash CUDACommon/run_tests.sh

python3 tools/benchmark.py --directory records/opt01-after --repeats 20
python3 tools/profile.py --directory records/opt01-after-profile

# SQLite 缓存过旧时强制重新导出；不会修改 .nsys-rep。
nsys stats --force-export=true --report cuda_gpu_kern_sum \
  records/opt01-after-profile/nvfp4.nsys-rep
```

纯性能优化通过条件：量化字节、scale、目标 dtype 输出保持规定行为，测试通过，
主要规模更快且其他规模无不可接受退化。压缩率与同一输入上的量化误差应不变。
若量化结果发生变化，先作为正确性/算法变更排查，不能只记一个更高的速度。

## 7. OPT-01 实验记录（待填写）

当前状态：已保存 B0 性能和 nsys；尚未在本记录下确认新的算子改动。

| 项目 | 填写内容 |
|---|---|
| 日期、实验人 | 待填写 |
| 对象 | 待选择：MXFP8/NVFP4，量化/反量化 |
| 对照版本 | B0；后续实验需填写具体版本或源码归档路径 |
| 改进版本/源码归档 | 待填写 |
| nsys 观察 | 哪个 kernel、什么规模、单次耗时多少 |
| 优化假设 | 为什么这个改动可能减少耗时，尚未证实 |
| 唯一改动 | 函数名、改动内容，不只写“优化代码” |
| 正确性 | CTest、量化字节/scale 对照、输出验证结果及日志路径 |
| GPU 检查 | memcheck/racecheck/synccheck 的结果；未跑则明确写未跑 |
| 报告路径 | before/after benchmark 和 nsys 路径 |
| 结论 | 待填写：保留 / 撤回 / 待复测，以及原因 |

下面按对象填写，量化和反量化不要混成一个耗时：

| 对象 | N | B0 median ms | 修改后 median ms | B0 P95 ms | 修改后 P95 ms | 加速比 |
|---|---:|---:|---:|---:|---:|---:|
| 待填写 | 1M | 待填 | 待测 | 待填 | 待测 | 待算 |
| 待填写 | 4M | 待填 | 待测 | 待填 | 待测 | 待算 |
| 待填写 | 16M | 待填 | 待测 | 待填 | 待测 | 待算 |

加速比 = 同范围的 B0 median / 修改后 median。P95 也应独立比较。
下一项实验复制第 7 节为 OPT-02，保留旧记录，包括没有提速的实验。
