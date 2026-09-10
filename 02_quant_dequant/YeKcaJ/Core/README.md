# 核心模块程序文档

本目录是当前正式实现的唯一维护位置。先读第 2 节了解调用关系，再按第 3～8 节查函数，最后按第 9 节操作。

## 输入输出自动归档（2026-09-10）

建议日常运行不再手写输出前缀。输入按类型、月日子目录和编号；结果按相同输入标识、量化格式和运行次数保存。使用已有 `results` 目录，不另建拼写为 reslut 的目录。

```text
input/fp16/910/1.fp16           9 月 10 日第 1 份 FP16 输入
input/fp16/910/1.json           生成参数、种子、形状、时间、SHA256
input/fp16/910/2.fp16           同日下一份 FP16 输入
input/fp32/910/1.fp32           FP32 单独编号
results/fp16/910/1/nvfp4/run-1/
  result.lpq                       packed 权重
  result.bf16                      反量化输出，类型由配置决定
  result.json                      指标、实际配置、输入绝对路径和 SHA256
results/fp16/910/1/nvfp4/run-2/  同输入再次运行，不覆盖 run-1
results/fp16/910/1/mxfp8/run-1/  同输入的另一种量化格式
```

日期目录按运行机器本地日期自动生成，不设年份层；同月日已有文件时继续递增编号，完整年份保留在 JSON 时间字段中。月日为“月份 + 两位日期”，例如 910、1001。FP16 与 FP32 的同名编号不意味着数值完全相同，具体以各自清单中的形状、分布和 seed 为准。运行失败可能留下部分文件和已占用编号，重试分配新 run，不覆盖失败记录。

```bash
# 从项目根目录执行，终端会打印实际生成路径。
python3 Core/tools/quantize.py generate --dtype fp16 --rows 128 --cols 129
# 使用上一步打印的路径；下方仅以 2026-09-10 首份输入为例。
python3 Core/tools/quantize.py run --config Core/configs/nvfp4.toml \
  --input input/fp16/910/1.fp16
```

显式 `--output` 和 `--prefix` 仍然有效。自动输出只接受项目下 `input/<dtype>/<月日>/<name>.<dtype>` 的文件，并核对输入 magic；外部文件需提供 `--prefix`。历史记录已迁入根目录 `records/`，批量 evaluate、正式 benchmark 和 nsys 的实验目录也请放入 records，避免与日常输出混用。

## 1. 目录与依赖

```text
Core/
  CMakeLists.txt          一次构建两种格式，并注册核心测试
  pipeline_main.cu        程序主入口、计时、误差日志、自测
  pipeline.cuh            正式 GPU kernel、显存管理、CPU 扩展对照
  codec.cuh               E4M3/E2M1 基础编解码、CUDA 错误检查
  pipeline_io.h           配置结构、输入和 .lpq 文件读写
  output.cuh              FP32/FP16/BF16 转换、输出和比较
  tools/
    quantize.py            配置化运行、输入生成、144 组误差评估
    benchmark.py           正式重复计时、环境与报告记录
    profile.py             nsys 时间线与统计采集
  configs/
    mxfp8.toml             MXFP8 示例配置，默认 FP16 输出
    nvfp4.toml             NVFP4 示例配置，默认 BF16 输出
  tests/
    reference.h/.cpp       冻结 CPU reference 的桥接接口
    test_pipeline.py       文件、配置与端到端测试
    test_tools.py          工具路径和报告生成测试
  build/                   本地生成，不提交
```

边界说明：

- 正式 GPU 程序不再 `#include` 旧的 `CUDAMXFP8/main.cu` 或 `CUDANVFP4/main.cu`。
- `tests/reference.cpp` 仍编译目录外的 `CPUMXFP8/main.cpp` / `CPUNVFP4/main.cpp`。冻结输入和 golden 保留原位置，因此不能只复制 Core 就独立构建。
- `CUDACommon/pipeline*`、`CUDACommon/output.cuh` 和 CPU 桥接旧路径是转发文件，不是另一套实现。旧 CUDA CLI、旧 benchmark 和旧回归测试继续保留。
- 根目录 `tools/quantize.py`、`benchmark.py`、`profile.py` 转发到本目录，使用本目录的构建产物。根目录旧 `configs/` 保留兼容，但后续修改配置请使用 `Core/configs/`。
- 历史实验记录已归档到根目录 `records/`，内部 JSON 的旧路径和哈希保持原样；`tools/record_validation.py` 仍服务于原来的完整历史验证流程，不是本目录 7 项测试的记录器。

## 2. 调用关系

```text
Core/tools/quantize.py run
  -> 读取 TOML，选择 Core/build/pipeline_mxfp8 或 pipeline_nvfp4
  -> pipeline_main.cu::main
     -> read_input：读入 FP32/FP16，内存中统一 FP32
     -> Workspace::upload：上传输入
     -> Workspace::launch_quant：计算 scale 并编码
     -> Workspace::download：下载 packed 权重
     -> reference_quantize + compare_packed：CPU 逐字节对照
     -> write_packed/read_packed：写文件并回读验证
     -> finish<T>：反量化、CPU 比较、写张量、统计误差
  -> Python 保存 JSON 日志并显示中文摘要
```

默认 block 模式的 GPU 路径：

```text
MXFP8：build_block_scales -> quantize_kernel<false>
NVFP4：maximum -> finalize_max -> build_block_scales -> quantize_kernel<false>
反量化：dequantize_kernel<T>
```

`tensor` 模式使用 `build_scales` 生成全张量共享的 scale，是软件比较扩展，不代表标准固定块布局。内部 `Baseline=true` 路径使用 `build_scales` 和枚举编码；它不是旧版 CLI，也不是后续优化自动选用的 B0。

## 3. pipeline_main.cu：主入口和测试流程

| 函数 | 具体作用 |
|---|---|
| `elapsed(t)` | 返回从主机时间点 t 到现在的毫秒数，用于主机流程/CPU 计时，不是 kernel 时间。 |
| `percentile(x,p)` | 排序非空样本；中位数在偶数样本时取中间两值均值，P95 用 nearest-rank。 |
| `stats(op,scope,n,samples,bytes)` | 输出一条 JSON 性能记录，包括中位数、P95 和逻辑带宽；bytes 是估算的必要数据量。 |
| `finish<T>(...)` | 启动目标类型反量化，下载结果，与 CPU 反量化后转换到 T 的结果比较；写输出张量，计算相对原始输入的最大误差、MAE、MSE、压缩率和单次计时。若输出非有限则误差字段记 null。 |
| `restore<T>(q,path)` | 仅反量化已有 .lpq：上传 q、执行 kernel、CPU 比较并写出，不需要原始 FP32 输入。 |
| `bench_dequant<T>(w,repeats)` | 同一显存上预热 3 次，再重复计时反量化；不把上传和下载计入 resident_gpu。 |
| `benchmark(n,repeats)` | 固定 seed=20260909 的 FP32 正态输入；交替测内部枚举路径和默认路径，校验一致后测三种反量化输出。另测含分配/传输/释放的 host_api；仅 1M 规模额外测 CPU 三次。 |
| `encoder_probe`（自测 kernel） | 每线程编码一个指定数值，输出未打包的编码字节；跳过 scale 计算，独立验证编码器。 |
| `check_encoder()` | 检查正负零、相邻编码中点及两侧、随机数；比较 CPU/GPU 编码，并检查随机舍入约 1/2 的中点分布及种子变化。 |
| `self_test(cpu_dir)` | 覆盖 0～65、255/256/257、1023/1024/1025 长度，两种 scale/舍入及三种输出；检查重复执行、冻结 CPU 输入和极端有限值。一次输出 299 个 pipeline case，另有编码器检查。 |
| `main(argc,argv)` | 分发 benchmark、自测、独立反量化或普通完整流程；校验参数和输入输出路径，捕获异常后输出 `pipeline=FAIL` 并返回非零。 |

## 4. pipeline.cuh：核心计算

### 4.1 编码和 scale 辅助函数

| 函数 | 具体作用 |
|---|---|
| `random_unit(seed,index)` | 对 seed 和元素下标做固定整数混合，生成 (0,1) 内随机数；不依赖线程调度，CPU/GPU 可重现。 |
| `magnitude2(code)` | 查 E2M1 的 8 个非负幅值：0、0.5、1、1.5、2、3、4、6。调用方负责符号。 |
| `magnitude(code,four)` | 根据 four 选择 E2M1 幅值或 E4M3 解码值，供邻值查找使用。 |
| `encode(value,four,stochastic,seed,index,baseline)` | 将已除以 scale 的数值编码；默认 E4M3 二分查邻值，E2M1 nearest 用固定中点比较；随机舍入按距离概率选择相邻值。baseline 的 nearest 使用枚举辅助函数。 |
| `scale_code(m,global)` | 根据组最大绝对值 m 编码 scale：MXFP8 为 E8M0，NVFP4 为 E4M3。处理全零和下溢。 |
| `effective(scales,state,group)` | 解码实际乘数：MXFP8 为 `2^(scale字节-127)`；NVFP4 为 `state[1] * decode_e4m3(scales[group])`。 |

MXFP8 默认每 32 元素一组，scale 指数字节为 `clamp(ceil(log2(m/448))+127,0,254)`；全零使用 127，正数比例下溢使用 0。NVFP4 每 16 元素一组，`global_scale=M/(6*448)`，再将 `m/(6*global_scale)` 编码成 E4M3 block scale；全零 global=1，正数下溢保留 FP32 最小正 subnormal。

### 4.2 六个正式 kernel

| Kernel | 输入和输出 | 线程分工与用途 |
|---|---|---|
| `maximum` | FP32 input -> partial | 每块 256 线程，跨步扫描并共享内存归约，每块输出一个局部最大绝对值。NVFP4 和 tensor 模式需要它。 |
| `finalize_max` | partial -> state[0/1] | 一个 256 线程块归约全局最大 M；state[0]=M，state[1]=global_scale。全局参数留在 GPU。 |
| `build_scales` | input/state -> scales | 每块 32 线程处理一个量化分组；tensor 模式直接使用全局 M。用于 tensor 和内部对照路径。 |
| `build_block_scales` | input/state -> scales | 默认 block 路径，每块 256 线程处理 8 个 MXFP8 组或 16 个 NVFP4 组；shuffle width=32/16 隔离分组，组首线程写 scale。尾部补零但仍参与 shuffle。 |
| `quantize_kernel<Baseline>` | input/scales/state -> data | 每块 256 线程。MXFP8 一线程写一个 E4M3 字节；NVFP4 一线程写一个 packed 字节，偶数元素在低 4 位、奇数在高 4 位。奇数尾部高 4 位清零；scale 非正时编码正零。 |
| `dequantize_kernel<T>` | data/scales/state -> output | 每块 256 线程，每线程恢复一个元素；NVFP4 先提取对应 nibble。FP32 中间值乘有效 scale，最后转换到 FP32/FP16/BF16。 |

### 4.3 资源管理和启动函数

| 类/函数 | 具体作用 |
|---|---|
| `DeviceBuffer<T>(n)` / 析构 | 申请 n 个 T 的显存，对象销毁时释放。复制构造/赋值禁用，避免重复释放。 |
| `Events()` / 析构 | 创建/销毁一对 CUDA event，允许重复使用。 |
| `Events::measure(f)` | 记录起始 event，调用 f 提交工作，检查启动错误，记录终止 event 并等待，返回设备序列毫秒数。 |
| `Workspace(count,options)` | 计算元素数/组数/归约块数，申请输入、临时最大值、state、data、scales 和输出显存。输出按 FP32 容量预留，兼容两种 16 位类型。 |
| `Workspace::upload(vector<float>)` | 检查长度后把原始 FP32 数组上传。 |
| `Workspace::launch_quant(baseline=false)` | 按格式和模式选择全局归约、scale kernel，再启动编码/打包；同一默认 stream 保证执行顺序。 |
| `Workspace::launch_dequant<T>()` | 启动目标输出类型的反量化 kernel；空输入不启动。 |
| `Workspace::download(t)` | 下载 data、scales 和 NVFP4 global，附上 t 的形状，返回 Packed；不写文件。 |
| `Workspace::upload(Packed)` | 上传已有 packed 权重和 global，用于从文件独立反量化。 |
| `Workspace::download_output<T>()` | 按 T 的字节宽度下载 n 个输出元素。 |
| `Workspace` 析构 | 编译器自动析构成员，释放各 DeviceBuffer 和 Events。 |

### 4.4 CPU 扩展对照

| 函数 | 具体作用 |
|---|---|
| `reference_encode` | nearest 调用冻结 CPU 编码函数；随机舍入用 CPU 线性搜索邻值，不复用 GPU 二分。 |
| `reference_quantize` | 用 CPU 计算相同的全局/分组 scale、编码和打包，支持 tensor/stochastic 扩展，返回 Packed。 |
| `reference_dequantize` | 在 CPU 解码 Packed 并返回 FP32 数组，供各目标 dtype 转换后比较。 |
| `compare_packed` | 比较形状、分组、data/scales 字节及 global 的 FP32 位模式；不一致即抛异常。它不负责比较所有 Options 元数据，文件测试另外检查头字段。 |

## 5. codec.cuh 与 output.cuh

| 函数 | 具体作用 |
|---|---|
| `pipeline::check_cuda` / `CUDA_CHECK` | 把 CUDA API 失败转换成包含表达式、源码行和 CUDA 描述的异常。 |
| `pipeline::decode_e4m3` | 解码 E4M3FN，处理符号、正常数、subnormal 和 NaN 编码。 |
| `pipeline::encode_e4m3` | 枚举 256 个编码，跳过非有限项，选择误差最小的有限编码；中点保留先出现的编码，零返回正零。 |
| `pipeline::encode_e2m1` | 枚举 8 个幅值，选择最近幅值并拼接符号位，包括负零。 |
| `cuda_output::Format<T>::convert` | float 原样返回，half 用 `__float2half_rn`，BF16 用 `__float2bfloat16_rn`；16 位输出采用 nearest-even。 |
| `cuda_output::parse_dtype` | 旧 CLI 的兼容辅助函数，读取末尾 `--output-dtype` 并缩短 argc；正式 main 不使用它。 |
| `cuda_output::write<T>` | 写 36 字节输出头和行主序 T 数组；每元素 FP32 为 4 字节，FP16/BF16 为 2 字节。 |
| `cuda_output::compare<T>` | 比较 GPU 输出与 CPU FP32 对照转换到 T 后的值。FP32 有限值容差 1e-6 且检查零符号；16 位逐位比较；NaN 比类别，Inf 要同号。打印 mismatch 数，失败抛异常。 |

注意：量化的 nearest 中点规则与 FP16/BF16 输出的 nearest-even 不同，不能混用。

## 6. pipeline_io.h：结构和文件函数

| 结构/函数 | 具体作用 |
|---|---|
| `Options` | 保存 tensor/stochastic 开关、标准 block 大小和 seed。 |
| `Tensor` | 保存 rows、cols、原输入元素字节数及统一 FP32 的 values。 |
| `Packed` | 保存形状、每组元素数、选项、global 和 data/scales 数组。 |
| `ceil_div(n,d)` | n/d 向上取整，避免使用 n+d-1 导致加法溢出；d 由调用方保证非零。 |
| `data_size(n)` | 返回纯 data 大小：MXFP8 n 字节，NVFP4 ceil(n/2) 字节。 |
| `checked_count(rows,cols)` | 验证尺寸和乘法/FP32 字节数溢出，再返回元素数。 |
| `validate(options)` | 要求 MXFP8 block=32、NVFP4 block=16；tensor 通过开关表示。 |
| `read_scalar<T>` / `write_scalar<T>` | 按字段宽度读取/写入标量，避免 C++ 结构体填充影响格式；当前运行平台为小端。 |
| `output_file(path)` | 创建父目录并打开二进制输出；C++ 层会截断已有文件，Python run 提前拒绝已有结果。 |
| `read_input(path)` | 校验输入 magic/version、尺寸和完整长度；读取 FP32 或将 FP16 无损扩展到 FP32；拒绝 NaN/Inf。 |
| `write_packed(path,q)` | 写 v2 头、data、scales；NVFP4 global 存在头中。 |
| `read_packed(path)` | 校验当前编译格式、版本、形状、分组、长度、global 和 scale 编码，检查奇数 NVFP4 高 nibble 填充为零。 |

文件布局如下，均保留行主序形状：

| 文件 | 头部 | 数据区 |
|---|---|---|
| `.fp32` / `.fp16` 输入 | 28 字节，FP32INP1 / FP16INP1、version、rows、cols | 原始 FP32 / FP16 元素 |
| `.lpq` 权重 | 72 字节，LPQUANT2、version=2、格式、形状、group、mode、rounding、seed、global、两段长度 | data 后接 scales；NVFP4 每字节两个元素 |
| `.fp32` / `.fp16` / `.bf16` 输出 | 36 字节，FP32DEQ1 / FP16DEQ1 / BF16DEQ1、version、rows、cols、count | 反量化目标类型元素 |
| `.json` | JSON 文本 | 原始输入到最终输出的误差、压缩率、单次性能、CPU 对照和路径 |

输入和输出即使扩展名相同也不是同一种文件头，不能将输出直接当作输入。详细字节布局见根目录 `EXTENDED_SPEC.md`，v1 冻结规则见 CPU 目录的规范。

## 7. Python 工具函数

| 文件/函数 | 具体作用 |
|---|---|
| `quantize.read_config(path)` | TOML 解析、默认值合并、键名/枚举/类型检查；target_gpu 只是记录标签，不选择设备。 |
| `quantize.executable(fmt)` | 定位 Core/build/pipeline_mxfp8 或 pipeline_nvfp4，不存在则提示先构建。 |
| `quantize.automatic_prefix(source,cfg)` | 校验项目 input 路径及 dtype 文件头，按输入标识创建独占 run-N 目录，返回 result 前缀。 |
| `quantize.input_sha256(source)` | 分块计算输入 SHA256，让结果日志能够追溯实际输入字节。 |
| `quantize.format_summary(record,log)` | 输出中文摘要；内部 number 处理显示精度和非有限值，status 区分通过/失败/未校验，section 按中英文显示宽度对齐。 |
| `quantize.run(cfg,source,prefix,console_json)` | 拒绝覆盖输入/已有产物，调用 CUDA 程序 verify 模式，解析最后一行 JSON，保存 prefix.json 并显示摘要或原始 JSON。 |
| `quantize.generate(path,rows,cols,dtype,distribution,seed)` | 固定种子生成 uniform/normal/outlier 输入，以 FP32/FP16 编码分批写入新文件。 |
| `quantize.generate_automatic(rows,cols,dtype,distribution,seed)` | 按本地月日子目录分配输入编号，生成文件及参数 JSON 清单，返回输入路径。 |
| `quantize.main()` | 分发 generate、run、evaluate；evaluate 生成 6 份输入并执行 144 个配置组合，写 summary.json。 |
| `benchmark.main()` | 顺序执行两种格式的 1M/4M/16M benchmark，保存每次 stdout JSONL、汇总 JSON、环境/源码哈希和 RESULTS.md。内部 command 查询硬件/编译器信息。 |
| `profile.main()` | 用 nsys 对两种格式的 4M、5 次重复 benchmark 采集，保存 rep/capture.log/stats.txt/version.txt；检查报告中确实存在 quantize_kernel。 |

## 8. 测试函数

`tests/reference.cpp` 中 `cpu_reference(values)` 调用原 CPU 量化和反量化并统一字段；`cpu_decode4`、`cpu_encode4` 直接调用冻结 E4M3 函数；`cpu_encode2` 仅 NVFP4 调用 E2M1，MXFP8 分支返回占位零且不应使用。

`tests/test_pipeline.py` 的 `PipelineTests`：

| 函数 | 验证内容 |
|---|---|
| `setUp` / `tearDown` | 创建/清理本次测试专用临时目录，不写冻结文件。 |
| `input` | 创建带合法头部的小张量测试输入。 |
| `run_pipeline` | 调用指定 binary，断言成功/失败；成功时返回权重、输出路径和 JSON。 |
| `test_modes_and_output_roundtrip` | 两种输入、scale、舍入、三种输出组合；检查头部并独立反量化，输出文件必须逐字节一致。 |
| `test_fp16_means_actual_half_values` | FP16 输入与其精确扩展的 FP32 输入产生相同 packed 权重。 |
| `test_empty` | 空张量仅有文件头，误差为零。 |
| `test_nonfinite_input_rejected` | NaN 和正负 Inf 输入必须失败。 |
| `test_overflow_output_logged` | FP16 溢出必须记录非有限输出且误差为 null。 |
| `test_bad_input_headers` / `test_bad_packed_headers` | 截断、附加数据、错误 magic/version、尺寸或 group 必须拒绝。 |
| `test_invalid_options` | 非法 scale/舍入/dtype 以及重复次数 0 必须失败。 |
| `test_config_parser` | TOML 默认值/合法选项以及非法键和值检查。 |
| `test_console_summary_and_json` | 中文摘要、JSON 输出和日志内容一致，正确显示失败/未校验/非有限状态。 |

`tests/test_tools.py`：`load_tool` 加载指定脚本；`test_benchmark_paths_and_manifest` 模拟子进程检查 6 次调用和源码哈希；`test_profile_paths_and_missing_data` 模拟 nsys 成功/缺失数据；`test_legacy_help_entrypoints` 实际调用旧 Python 入口的 --help。模拟测试不等于实际完成性能采集。

## 9. 最后操作步骤

以下命令均在 WSL 项目根目录执行，不在 Core 或旧 CUDA 子目录中执行。

### 步骤 1：编译当前核心

环境需要 CMake >=3.18、C++17、CUDA Toolkit、可用的 NVIDIA GPU、Python >=3.11；脚本只依赖 Python 标准库。默认目标 sm_75，已在 RTX 3060 Laptop 验证，不代表已在实体 T4 验证。

```bash
cd /home/jky/Learning-CUDA-project/02_quant_dequant/YeKcaJ
cmake -S Core -B Core/build -DCMAKE_BUILD_TYPE=Release
cmake --build Core/build -j 4
```

产物是 `Core/build/pipeline_mxfp8` 和 `Core/build/pipeline_nvfp4`。脚本固定使用此目录；改用其他 CMake 构建目录时请直接调用对应二进制。指定 GPU 架构可在配置时追加 `-DCMAKE_CUDA_ARCHITECTURES=86`，但性能对比前后必须保持一致。

### 步骤 2：先跑正确性测试

```bash
ctest --test-dir Core/build --output-on-failure
```

共 7 个 CTest 入口：工具测试 1 个，每种格式各有文件测试、算法自测、冻结哈希检查。一个入口包含很多内部样例，不是只有 7 个输入张量。旧两套 29 项测试没有删除，需要完整兼容回归时执行 `bash CUDACommon/run_tests.sh`。

### 步骤 3：生成一份输入

```bash
python3 Core/tools/quantize.py generate \
  --rows 128 --cols 129 --dtype fp16 --distribution normal --seed 1234
```

自动创建根目录 input 下的日期编号，终端会打印路径，不覆盖已有输入。以下以已迁入的 `input/fp16/910/1.fp16` 为例；新生成文件请使用实际打印的路径。

### 步骤 4：运行两种格式

```bash
python3 Core/tools/quantize.py run --config Core/configs/mxfp8.toml \
  --input input/fp16/910/1.fp16
python3 Core/tools/quantize.py run --config Core/configs/nvfp4.toml \
  --input input/fp16/910/1.fp16
```

结果分别位于 `results/fp16/910/1/mxfp8/run-N/` 和 `nvfp4/run-N/`，每次得到 result.lpq、result.fp16/bf16、result.json。改变 output_type 即改变输出类型；run 末尾加 `--json` 可显示原始 JSON。

### 步骤 5：验证权重可以独立反量化

```bash
./Core/build/pipeline_nvfp4 --dequant-file results/fp16/910/1/nvfp4/run-1/result.lpq \
  results/fp16/910/1/nvfp4/run-1/restored.fp32 fp32
```

使用与 .lpq 格式一致的程序；不需要原始输入。最后一个参数也可为 fp16 或 bf16。直接调用 C++ 会覆盖同名输出，使用新路径保存。

### 步骤 6：完整误差评估

```bash
python3 Core/tools/quantize.py evaluate --directory records/core-evaluation-02
```

保存 144 组结果和 summary.json。看 cpu_quant_match / cpu_dequant_match 是否通过，再看 max_abs_error、mae、mse 和压缩率；不要把 CPU 一致性等同于量化无损。

### 步骤 7：记录正式性能 baseline

```bash
python3 Core/tools/benchmark.py --directory records/core-baseline-01 --repeats 20
```

查看 RESULTS.md 和 summary.json；环境和源码哈希在 environment.json。量化看 `quant_optimized + resident_gpu`，反量化看 `dequant_fp32/fp16/bf16 + resident_gpu`。保存中位数、P95、逻辑 GB/s；修改后换目录运行，再按相同格式/规模/类型/口径比较。

量化 resident_gpu 包含 scale/global 归约和编码，不含显存分配与传输；host_api 包含分配、上传、计算、下载和释放。单次 run 的主机流程不含释放。逻辑 GB/s 不计中间 scratch 流量，也不是硬件 DRAM 带宽测量。迁移改变编译单元和哈希，应重新测 baseline，不能宣称迁移带来加速。

### 步骤 8：用 nsys 定位耗时

```bash
python3 Core/tools/profile.py --directory records/core-profile-01
nsys stats --force-export=true --report cuda_gpu_kern_sum \
  records/core-profile-01/mxfp8.nsys-rep
nsys stats --force-export=true --report cuda_gpu_kern_gb_sum \
  records/core-profile-01/nvfp4.nsys-rep
```

打开 .nsys-rep 查看时间线，或看 *_stats.txt。报告包含预热、内部枚举路径和 host_api 循环，不能把所有实例总耗时当成一次量化时间。最终性能数字用不带 profiler 的步骤 7；nsys 本身不能证明某条指令或占用率就是瓶颈。

### 步骤 9：修改 kernel 后进行 GPU 内存检查

```bash
compute-sanitizer --tool memcheck --error-exitcode 99 \
  ./Core/build/pipeline_mxfp8 --self-test ./CPUMXFP8
compute-sanitizer --tool memcheck --error-exitcode 99 \
  ./Core/build/pipeline_nvfp4 --self-test ./CPUNVFP4
```

按需将 memcheck 换为 racecheck 或 synccheck；工具不在 PATH 时使用其真实绝对路径。驱动较新时需匹配的 nsys/Compute Sanitizer。工具不可用不算测试通过。

## 10. 本次迁移验证

记录见同目录 `MIGRATION_TESTS.md`。性能历史保留在根目录 `OPTIMIZATION_LOG.md`；本次是结构整理，不作为性能优化结论。
