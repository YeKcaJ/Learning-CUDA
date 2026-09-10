# MXFP8 / NVFP4 软件量化与 CUDA 反量化

选题 2。当前正式核心集中在 [Core/](Core/README.md)，其中的程序文档逐个说明函数作用并给出完整操作步骤。
旧版 CPU reference 和 CUDA CLI 保留，旧路径仅作为兼容入口或历史实现。
支持 FP32/FP16 输入，MXFP8、NVFP4 packed 输出，block/tensor 缩放，nearest/stochastic
舍入，以及 FP32/FP16/BF16 反量化。所有低精度编码为软件模拟，不使用 FP8/FP4 Tensor Core。

## 构建与测试

环境：Linux/WSL、CMake >=3.18、C++17、CUDA Toolkit（本机 12.0）、Python >=3.11。
默认源码目标为 sm_75，可通过 CMake 参数指定架构；本机在 RTX 3060 Laptop 上验证 sm_75 产物。
这不等于已经在真实 T4 上跑过。无需更换系统驱动或 CUDA Toolkit。

```bash
cmake -S Core -B Core/build -DCMAKE_BUILD_TYPE=Release
cmake --build Core/build -j 4
ctest --test-dir Core/build --output-on-failure
```

生成 `Core/build/pipeline_mxfp8`、`Core/build/pipeline_nvfp4`，7 个核心 CTest 入口覆盖工具、文件、算法和冻结哈希。
旧两套各 29 项兼容测试仍可用 `bash CUDACommon/run_tests.sh` 执行；旧 benchmark 不用于新的性能实验。

## 配置化执行

项目根目录现在分为：`input/` 保存输入，`results/` 保存对应运行输出，`records/` 保存历史和批量实验记录；均与 Core 平级。

日常输入可自动归档：`python3 Core/tools/quantize.py generate --dtype fp16`。
脚本打印 `input/fp16/<月日>/<编号>.fp16`；随后用
`python3 Core/tools/quantize.py run --config Core/configs/nvfp4.toml --input <生成路径>`
即可自动输出到 `results/fp16/<月日>/<输入编号>/nvfp4/run-N/`。
重复运行不会覆盖旧结果，输入清单和结果 JSON 记录 SHA256。详细布局见 [Core 文档](Core/README.md)。

从项目根目录运行，输出前缀必须不存在，避免覆盖实验记录：

```bash
python3 Core/tools/quantize.py generate \
  --rows 128 --cols 129 --dtype fp16 --distribution normal
python3 Core/tools/quantize.py run --config Core/configs/nvfp4.toml \
  --input input/fp16/910/1.fp16
```

将示例输入路径替换为生成命令打印的路径。输出自动归档到 results 的对应输入/格式/run-N 目录，包含 `.lpq`（packed+scales）、`.bf16`（按配置选择）、`.json`（误差、压缩率、计时）。
终端默认显示分组对齐的中文摘要；`run` 命令追加 `--json` 可恢复单行 JSON。
落盘 JSON 的字段和精度不变，批量 `evaluate` 仍使用 JSON 行输出。
修改 TOML 的 scale_mode、rounding、seed、output_type 即可改变配置，使用新前缀保存结果。
`Core/configs/mxfp8.toml` 是另一种格式的示例。每次执行默认比较 CPU oracle，不覆盖冻结 golden。

量化文件可独立读取，无需原始输入或 golden：

```bash
./Core/build/pipeline_nvfp4 --dequant-file results/fp16/910/1/nvfp4/run-1/result.lpq \
  results/fp16/910/1/nvfp4/run-1/restored.fp32 fp32
```

FP16 输入使用 `FP16INP1` 文件头；详见 [扩展规范](EXTENDED_SPEC.md)。tensor 模式为缩放策略
对照，不声称符合标准固定 block 布局。默认 block 模式的 payload 已与冻结 CPU 对照验证。

## 完整误差与性能实验

```bash
python3 Core/tools/quantize.py evaluate --directory records/my-evaluation
python3 Core/tools/benchmark.py --directory records/my-performance --repeats 20
```

误差实验覆盖 144 组组合并保存 summary.json。性能实验为 1M/4M/16M、预热 3 次、
正式 20 次；同一驻留 GPU 流程内交替比较小块 shared 归约/枚举与 warp 分组/快速编码，另测三种反量化输出和量化
端到端调用。在 1M 上额外测 CPU 枚举 reference（不是优化过的 CPU 库）。
完整结果见 [最终报告](FINAL_REPORT.md)；保留的汇总、性能和环境记录已移入 records，部分逐样例二进制此前已清出项目并备份。

## GPU 检查与 nsys

```bash
SANITIZER_BIN=/absolute/path/to/compute-sanitizer bash CUDACommon/run_tests.sh
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  -o profiles/pipeline_mxfp8 ./Core/build/pipeline_mxfp8 --benchmark 4194304 5
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
  profiles/pipeline_mxfp8.nsys-rep
```

也可用 `python3 Core/tools/profile.py --directory records/my-profile` 串行采集两种格式。

NVFP4 更换可执行路径和报告名。`quantize_kernel<true>` 为内部枚举对照；MXFP8 block+nearest 默认看 `mxfp8_quantize_fused_kernel`，NVFP4 默认看 `quantize_kernel<false>` 及其 scale/归约阶段。
nsys 包含预热及端到端分支，不用于发布无 profiler 的最终性能数字。新版驱动配新版 nsys；
本机 2024.6 曾缺失 kernel 数据，2026.1.3 已成功采集。Windows 查看器应同版或更新。

## 代码位置

| 目录/文件 | 职责 |
|---|---|
| CPUMXFP8 / CPUNVFP4 | 冻结 CPU 算法、v1 固定输入和 golden |
| CUDAMXFP8 / CUDANVFP4 | 旧版 CLI/kernel 与各自 CMake 构建入口 |
| Core/pipeline.cuh | 可配置设备计算、显存复用、CPU 扩展参考 |
| Core/codec.cuh | 独立编解码辅助，不再包含旧 CUDA main.cu |
| Core/pipeline_io.h | 输入与 v2 文件协议、尺寸验证 |
| Core/pipeline_main.cu | 新 CLI、性能基准和算法回归入口 |
| Core/output.cuh | 三种输出类型转换、比较与写出 |
| Core/tests | 冻结 CPU 桥接、文件/配置与工具测试 |
| Core/tools / Core/configs | 正式脚本和示例配置 |
| CUDACommon / tools / configs | 旧测试、兼容转发和保留的旧配置 |

GPU 核心已与旧 CUDA 入口分离；CPU 桥接仍依赖目录外冻结源码，因此 Core 不是可脱离仓库安装的独立包。
详细函数索引、文件格式与逐步命令见 [核心程序文档](Core/README.md)。

## 提交检查

按训练营要求在 `2026-summer-project` 的对应路径提交 PR。当前工作分支是
`mxfp8-YeKcaJ`；本次没有自动切换分支、提交或推送。提交应包含源码、配置、测试、规范、
报告和必要日志，不应包含 build 二进制、临时 profiler 数据。历史仓库中已跟踪的 build
文件不会因为新增 .gitignore 自动取消跟踪，正式提交前需单独审查。
