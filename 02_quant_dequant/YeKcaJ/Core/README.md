# 核心程序说明

正式程序由一个 Python 配置入口和两个 CUDA 计算后端组成。Core 包含全部源码和冻结参考数据，可以单独复制、重新构建；不要复制旧 build 缓存。
这里的文件按职责划分，优化算子主要看 `kernels/`。

## 1. 目录与调用关系

```text
Core/
├── app/                    程序入口和文件任务
│   ├── main.cpp            分发命令、统一处理异常
│   ├── commands.h          文件任务、benchmark、自测的接口声明
│   └── run.cu              量化/反量化流程与误差日志
├── kernels/                GPU 算子和设备编码辅助
│   ├── mxfp8_quantize.cuh   MXFP8 block+nearest 融合量化
│   ├── nvfp4_quantize.cuh   NVFP4 block+nearest 融合量化/打包
│   ├── dequantize.cuh      两种格式的三种类型反量化
│   ├── reduce.cuh          全局最大值的两级归约
│   ├── fallback_quantize.cuh  tensor、随机舍入、内部对照
│   ├── codec.cuh           E4M3/E2M1 编码解码
│   └── scale.cuh           scale 字节与有效乘数
├── runtime/                GPU 执行管理
│   ├── workspace.cuh       显存、上传、下载的接口
│   ├── workspace.cu        唯一的正式 kernel 启动实现
│   └── cuda_utils.cuh      显存 RAII、event、错误检查
├── io/                     输入、packed、反量化文件读写
├── common/                 数据结构、随机数、输出转换、墙钟计时
├── reference/              CPU 校验，独立于 GPU 编码器
│   ├── oracle.cpp/.h       block/tensor、两种舍入的 CPU 对照
│   ├── frozen.cpp/.h       冻结 CPU 算法桥接
│   ├── mxfp8/              原 CPU 源码及 tests/data、tests/golden
│   └── nvfp4/              原 CPU 源码及 tests/data、tests/golden
├── benchmarks/benchmark.cu  正式计时、统计与 JSONL
├── tests/                  算法回归、文件与工具测试
├── tools/                  quantize.py、benchmark.py、profile.py
├── configs/                两种格式的 TOML 示例
└── CMakeLists.txt           构建定义
```

运行方向：`tools/quantize.py → app/main.cpp → app/run.cu → runtime/workspace.cu → kernels/`。
`app/run.cu` 另外调用 `io/` 读写文件、`reference/` 校验结果。
`runtime/` 和 `kernels/` 不包含 CPU oracle 或文件读写代码；benchmark、自测分别编译，不再混在 main 中。
`.cuh` 是 CUDA 头文件，里面可以定义 kernel；正式 kernel 定义只由 `runtime/workspace.cu` 包含并编译。

## 2. 找到要优化的 kernel

| 路径/配置 | GPU 执行顺序 |
|---|---|
| MXFP8 block+nearest | `mxfp8_quantize_fused_kernel` |
| NVFP4 block+nearest | `maximum → finalize_max → nvfp4_quantize_fused_kernel` |
| tensor 或 stochastic | 按需全局归约 → `build_scales` 或 `build_block_scales` → `quantize_kernel<false>` |
| 内部 baseline | 按需全局归约 → `build_scales → quantize_kernel<true>` |
| 任意格式反量化 | `dequantize_kernel<T>`，T 为 float、__half 或 __nv_bfloat16 |

默认融合条件是 `block + nearest`，与输入文件是 FP32 还是 FP16 无关。
FP16 文件先在 CPU 精确展开为 FP32，再上传；这不是直接读取 FP16 显存的 kernel。BF16 当前仅支持输出。

## 3. 函数索引

| 文件 | 函数/类型 | 具体作用 |
|---|---|---|
| [kernels/mxfp8_quantize.cuh](kernels/mxfp8_quantize.cuh) | `mxfp8_quantize_fused_kernel` | 每 warp 32 元素，归约最大值、计算/广播 E8M0 scale、编码 E4M3；复用寄存器输入 |
| [kernels/nvfp4_quantize.cuh](kernels/nvfp4_quantize.cuh) | `nvfp4_quantize_fused_kernel` | 每线程两个元素、每 8 lane 一个分组；算 E4M3 scale，再编码/打包两个 E2M1 |
| [kernels/dequantize.cuh](kernels/dequantize.cuh) | `dequantize_kernel<T>` | 解码数据、乘有效 scale，转换到 T；NVFP4 提取对应高/低 nibble |
| [kernels/reduce.cuh](kernels/reduce.cuh) | `maximum` / `finalize_max` | 生成局部最大值，再归约为张量最大值并计算 NVFP4 global_scale |
| 同上 | `block_maximum` | 固定 256 线程块，warp 内归约后合并 8 个最大值，仅一次整块同步 |
| [kernels/scale.cuh](kernels/scale.cuh) | `scale_code` / `effective` | 编码 scale 字节；恢复两种格式的实际乘数 |
| [kernels/codec.cuh](kernels/codec.cuh) | `encode_e4m3_nearest` | 用 FP32 指数/尾数直接生成 E4M3；中点向较小幅值，不是 nearest-even |
| 同上 | `encode` | 分发快速 nearest、随机舍入、内部枚举；E2M1 nearest 使用固定中点比较 |
| 同上 | `decode_e4m3` / `magnitude` | 恢复 E4M3 值或编码的非负幅值 |
| 同上 | `encode_e4m3` / `encode_e2m1` | 内部对照用的枚举编码器，不能代表当前默认性能 |
| [kernels/fallback_quantize.cuh](kernels/fallback_quantize.cuh) | `build_scales` / `build_block_scales` | 非融合路径的 shared 或 warp 分组 scale 归约 |
| 同上 | `quantize_kernel<Baseline>` | 用已有 scale 编码，支持随机舍入和枚举对照 |
| [runtime/workspace.cuh](runtime/workspace.cuh) | `Workspace` | 一次任务的显存、尺寸、配置和 CUDA events |
| 同上 | `upload(values)` / `upload(packed)` | 上传 FP32 输入，或上传已有 packed 权重 |
| 同上 | `download` / `download_output<T>` | 下载 packed 元数据/字节，或 T 类型张量 |
| [runtime/workspace.cu](runtime/workspace.cu) | `launch_quant` / `launch_dequant<T>` | 选择量化路径，或启动指定输出类型的反量化 |
| [runtime/cuda_utils.cuh](runtime/cuda_utils.cuh) | `DeviceBuffer` / `Events::measure` / `check_cuda` | 显存自动释放、设备序列计时、CUDA 错误转异常 |
| [app/main.cpp](app/main.cpp) | `main` | 分发文件任务、独立反量化、benchmark、自测 |
| [app/run.cu](app/run.cu) | `run_pipeline` | 参数检查、读取、上传、量化、CPU 比较及 packed 文件往返检查 |
| 同上 | `finish<T>` | 反量化、CPU 对比、保存张量，计算误差/压缩率并打印 JSON |
| 同上 | `restore_file` / `restore<T>` | 从已有 .lpq 独立反量化，不需要原始输入或 golden |
| [io/tensor_io.h](io/tensor_io.h) | `read_input` | 校验输入头、形状和长度，展开 FP16，拒绝 NaN/Inf |
| 同上 | `read_packed` / `write_packed` | v2 头、data、scale 的读写及格式检查 |
| 同上 | `read_scalar` / `write_scalar` / `output_file` | 逐字段二进制读写，创建输出父目录 |
| [io/output.cuh](io/output.cuh) | `write<T>` | 保存带形状与输出类型头的反量化张量 |
| [reference/output_compare.cuh](reference/output_compare.cuh) | `compare<T>` | 与 CPU 输出比较，16 位结果逐位比较 |
| [common/types.h](common/types.h) | `Options` / `Tensor` / `Packed` | 配置、统一 FP32 主机输入、低精度结果 |
| 同上 | `ceil_div` / `data_size` / `checked_count` / `validate` | 向上取整、packed 字节数、溢出及 block 大小检查 |
| [common/output_type.cuh](common/output_type.cuh) | `Format<T>::convert` | 输出类型转换和文件标识，16 位采用 nearest-even |
| [common/numeric.h](common/numeric.h) | `random_unit` / `magnitude2` | 按 seed/index 生成随机数；返回 E2M1 幅值 |
| [common/timing.h](common/timing.h) | `elapsed` | 主机墙钟毫秒，与设备 event 计时分开 |
| [reference/oracle.cpp](reference/oracle.cpp) | `reference_encode` / `reference_quantize` / `reference_dequantize` | 独立 CPU 扩展对照；nearest 保留枚举，stochastic 线性找邻值 |
| 同上 | `compare_packed` | 比较形状、分组、data/scales 及 global_scale 位模式 |
| [reference/frozen.cpp](reference/frozen.cpp) | `cpu_reference` / `cpu_encode*` / `cpu_decode4` | 调用未改变的原 CPU 实现，适配两种格式字段 |
| [benchmarks/benchmark.cu](benchmarks/benchmark.cu) | `benchmark` / `bench_dequant<T>` | 固定 FP32 数据，预热并测试量化、三种反量化、host_api 和 CPU |
| 同上 | `percentile` / `stats` | 计算 median/P95 并输出 JSONL，逻辑 GB/s 不等于实测 DRAM 带宽 |
| [tests/regression.cu](tests/regression.cu) | `self_test` / `check_encoder` / `encoder_probe` | 尾部、模式、类型、重复一致性及编码边界验证；probe 是测试专用 kernel |

Python：`quantize.py` 的 `generate`/`generate_automatic` 创建带头输入，`read_config` 检查 TOML，`run` 调用后端并保存日志，`format_summary` 打印中文摘要；`benchmark.py` 保存性能与源码哈希，`profile.py` 采集并检查两种融合 kernel 的 nsys 数据。

## 4. 操作步骤

以下从项目根目录执行。配置、输入输出布局、后端名称沿用原来的方式。

**① 构建和正确性检查**

```bash
cmake -S Core -B Core/build -DCMAKE_BUILD_TYPE=Release
cmake --build Core/build -j 4
ctest --test-dir Core/build --output-on-failure
```

8 个 CTest 入口覆盖工具、归约边界、两种格式的文件/类型测试、算法回归与冻结哈希。
`tests/reduction.cu` 独立检查局部/全局最大值及 global_scale，覆盖 warp 边界、跨步扫描尾部、零和极端有限值。
也可运行 `./Core/build/pipeline_mxfp8 --self-test` 或 `./Core/build/pipeline_nvfp4 --self-test`。
冻结文件只验证，不重新生成；关闭 BUILD_TESTING 会去掉算法自测代码，普通文件任务仍有 CPU 对照。

**② 生成输入**

```bash
python3 Core/tools/quantize.py generate --dtype fp32 --rows 1024 --cols 1024
```

支持 `--dtype fp16`。自动保存到 `input/<月日>/<编号>.<dtype>`，同一编号可同时存在 `.fp16` 和 `.fp32`；同时写清单，记下终端打印的实际路径。

**③ 配置并运行**

修改 `Core/configs/mxfp8.toml` 或 `nvfp4.toml`：output_type 选择 fp32/fp16/bf16；融合路径用 `scale_mode="block"`、`rounding="nearest"`。

```bash
python3 Core/tools/quantize.py run --config Core/configs/mxfp8.toml \
  --input input/914/1.fp32
python3 Core/tools/quantize.py run --config Core/configs/nvfp4.toml \
  --input input/914/1.fp32
```

替换示例日期和编号。输出在 `output/914/1/<格式>/`，首次运行生成 `fp32_fp16.lpq`、`fp32_fp16.fp16` 和 `fp32_fp16.json`，重复运行追加 `_2`。
重复执行增加 run 编号。外部输入使用 `--prefix` 指定新的输出前缀。

**④ 独立反量化已有权重**

```bash
./Core/build/pipeline_mxfp8 --dequant-file \
  results/fp32/911/1/mxfp8/run-1/result.lpq \
  results/fp32/911/1/mxfp8/run-1/restored.bf16 bf16
```

NVFP4 使用另一个后端。输出路径应另取未使用的名称；底层文件写出会覆盖同名文件。

**⑤ 测性能并记录**

```bash
python3 Core/tools/benchmark.py --directory records/my-before --repeats 20
```

固定 FP32 正态输入、block+nearest、预热 3 次，测 1M/4M/16M。查看 RESULTS.md。
算子优化比较同规模 `quant_optimized / resident_gpu` 的 median、P95、逻辑带宽；反量化按相同输出 dtype 比。
输入读取、FP16 展开、CPU 校验、文件落盘不在驻留 GPU 计时内。单次 run 用于正确性和误差，不用于正式性能结论。

**⑥ 用 nsys 定位阶段**

```bash
python3 Core/tools/profile.py --directory records/my-profile
nsys stats --force-export=true --report cuda_gpu_kern_sum \
  records/my-profile/mxfp8.nsys-rep
```

NVFP4 将报告名换为 nvfp4.nsys-rep。看第 2 节列出的默认 kernel；报告也含内部对照和预热，不能把累计 Time (%) 当作单次默认流程占比。
改完 kernel 后重新构建、跑测试，再将 benchmark 保存到新的目录，并更新根目录 OPTIMIZATION_LOG.md。

完整误差组合：`python3 Core/tools/quantize.py evaluate --directory records/my-evaluation`（144 组）。
GPU 检查：使用兼容驱动的 `compute-sanitizer --tool memcheck ./Core/build/pipeline_nvfp4 --self-test`；更换 tool 可检查 racecheck/synccheck，MXFP8 同理。
