# MXFP8 / NVFP4 量化程序

选题 2。正式程序只维护 **Core**，使用入口是 `Core/tools/quantize.py`。
支持 FP32/FP16 输入、MXFP8/NVFP4 packed 权重、FP32/FP16/BF16 反量化输出。
支持 block/tensor 缩放与 nearest/stochastic 舍入。FP16 输入先在 CPU 精确展开为 FP32，GPU 量化输入目前统一为 FP32；不支持 BF16 输入。

## 找代码

| 想看什么 | 直接打开 |
|---|---|
| MXFP8 默认量化 kernel | [Core/kernels/mxfp8_quantize.cuh](Core/kernels/mxfp8_quantize.cuh) |
| NVFP4 默认 packed 量化 kernel | [Core/kernels/nvfp4_quantize.cuh](Core/kernels/nvfp4_quantize.cuh) |
| 两种格式的反量化 kernel | [Core/kernels/dequantize.cuh](Core/kernels/dequantize.cuh) |
| E4M3/E2M1 编码、scale 公式 | [codec.cuh](Core/kernels/codec.cuh)、[scale.cuh](Core/kernels/scale.cuh) |
| 哪些选项启动哪些 kernel | [Core/runtime/workspace.cu](Core/runtime/workspace.cu) |
| 程序入口与文件任务 | [main.cpp](Core/app/main.cpp)、[run.cu](Core/app/run.cu) |
| 函数作用与完整操作步骤 | [Core/README.md](Core/README.md) |

## 根目录分工

```text
YeKcaJ/
├── Core/                  正式源码、配置、CPU reference 与测试
├── input/                 输入：月日 / 编号.fp16或.fp32
├── output/                分为 weights、tensors、logs 的运行输出
├── records/               性能实验、nsys 和验证证据
├── docs/                  题目 PDF、文件规范、阶段报告
├── legacy/                旧实现与旧入口存档，不参与正式构建
├── OPTIMIZATION_LOG.md     每次优化的性能记录
└── README.md              本页
```

## 构建与使用

环境：Linux/WSL、CMake ≥ 3.18、CUDA Toolkit、C++17、Python ≥ 3.11。
本机使用 CUDA 12.0、RTX 3060 Laptop；默认构建 sm_75，可通过 CMake 指定架构。
以下命令均在项目根目录执行。

```bash
cmake -S Core -B Core/build -DCMAKE_BUILD_TYPE=Release
cmake --build Core/build -j 4
ctest --test-dir Core/build --output-on-failure
```

生成两个计算后端 `pipeline_mxfp8`、`pipeline_nvfp4`，由同一个 Python 入口按配置选择。

```bash
python3 Core/tools/quantize.py generate --dtype fp16 --rows 128 --cols 129
python3 Core/tools/quantize.py run --config Core/configs/mxfp8.toml \
  --input input/914/1.fp16
```

输入路径请替换成 generate 实际打印的路径。修改配置中的 `output_type` 选择 fp32/fp16/bf16。
输出自动保存为 `output/<输入类型>/<月日>/<编号>/<量化格式>/run-N/{weights,tensors,logs}/`，分别存放 packed 权重、反量化张量和 JSON 日志。
输入文件带格式和尺寸头，不能直接把无头原始数组改扩展名使用。

## 性能优化

```bash
python3 Core/tools/benchmark.py --directory records/my-benchmark --repeats 20
python3 Core/tools/profile.py --directory records/my-profile
```

选择未使用过的记录目录。benchmark 固定 FP32、block+nearest，测试 1M/4M/16M；比较前后 `quant_optimized / resident_gpu` 的中位数与 P95。
nsys 默认量化看 `mxfp8_quantize_fused_kernel` 或 `nvfp4_quantize_fused_kernel`；NVFP4 还包含 `maximum`、`finalize_max`。
`quant_enumeration` 是内部对照，不能替代已保存的优化前版本。

规范：[冻结 v1](docs/REFERENCE_SPEC.md)、[正式 v2](docs/EXTENDED_SPEC.md)。
最新优化见 [优化日志](OPTIMIZATION_LOG.md)；[阶段报告](docs/FINAL_REPORT.md) 保留原测量日期。
旧路径迁移说明见 [legacy/README.md](legacy/README.md)。提交源码时排除 build 和可重建产物；构建 Core 不需要 legacy。
