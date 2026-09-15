# 摩尔线程 MUSA 后端

已在 MTT S4000（48 GiB、驱动/SDK 5.1.0、mp_22）验证。构建后执行的是真实 MUSA GPU kernel，CPU 仅作参考校验。CUDA 使用原有后端。

## 文件与函数

| 文件/函数 | 作用 |
|---|---|
| `include/cuda_runtime.h` | 将项目使用的内存、拷贝、event 和错误接口名称映射到 MUSA，只有 MUSA 构建包含此目录 |
| `include/cuda_fp16.h`、`cuda_bf16.h` | 引入 MUSA 的 FP16/BF16 类型及转换接口 |
| `quantize.cuh::musa_quantize_vectorized_kernel` | float4 加载、子组 shuffle 求最大值、一次 shared scale 广播；MXFP8 合并写 4 字节，NVFP4 合并写 2 字节，尾部回退标量 |
| `../../runtime/workspace.cu::launch_quant` | 根据后端、格式和配置选择融合/通用路径；NVFP4 先计算全局 scale |
| `../../kernels/reduce.cuh::block_maximum` | 32-lane 逻辑子组 shuffle，再用 8 个 shared 槽合并，保留 grid-stride 扫描 |
| `../../kernels/fallback_quantize.cuh` | tensor、stochastic 及内部枚举对照路径，非融合 block scale 使用子组 shuffle |
| `../../common/device_math.cuh` | FP64 商/积一次 RN 转回 FP32，保留次正规数并与 CPU 逐位对照；除法替代原高开销 `__fdiv_rn` |
| `../../kernels/dequantize.cuh` | 共用反量化 kernel，恢复 scale 后输出 FP32/FP16/BF16 |

设备报告 warpSize=128，但 SDK 同步 shuffle 使用 32-lane 逻辑子组；已独立验证 width=4/8/16/32。融合 kernel 的纯 shuffle 广播曾在零 scale/极小值回归失败，因此正式实现保留一次 shared scale 广播。MUSA 当前 10 项 CTest（比 CUDA 多一项 shuffle 测试）。

已完成归约/向量化和严格除法两轮优化，逐步数据见 [摩尔优化记录](../../../docs/MUSA_OPTIMIZATION_LOG.md)。设备算术测试现覆盖百万对输入；前后双侧输出哨兵已加入流程测试。分阶段计时、MUPTI 与检查工具状态见 [诊断工具](tools/README.md)。

## 操作步骤

在项目根目录执行，要求 Python 3.11+、CMake 3.18+ 和 MUSA SDK：

```bash
mthreads-gmi
/usr/local/musa/bin/mcc --version
cmake -S Core -B Core/build-musa \
  -DLP_BACKEND=MUSA -DMUSA_ROOT=/usr/local/musa -DMUSA_ARCH=mp_22 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build Core/build-musa -j 4
ctest --test-dir Core/build-musa --output-on-failure
```

生成输入并运行（把示例路径换成 generate 打印的实际路径）：

```bash
python3 Core/tools/quantize.py generate --dtype fp32 --rows 128 --cols 129
python3 Core/tools/quantize.py run --backend musa \
  --config Core/configs/musa_mxfp8.toml --input input/915/1.fp32
python3 Core/tools/quantize.py run --backend musa \
  --config Core/configs/musa_nvfp4.toml --input input/915/1.fp32
```

配置 `output_type` 可选 `fp32`、`fp16`、`bf16`。输入支持 FP32 和 FP16，FP16 先在主机展开为 FP32；不支持 BF16 输入。结果仍写到 `output/<月日>/<输入编号>/<格式>/`，同一前缀对应 `.lpq` 权重、目标类型张量和 `.json` 日志。

从已有权重独立反量化：

```bash
Core/build-musa/pipeline_mxfp8 --dequant-file \
  output/915/1/mxfp8/fp32_fp16.lpq output/915/1/mxfp8/restored.bf16 bf16
```

完整评估和基准（每次使用新目录）：

```bash
python3 Core/tools/quantize.py evaluate --backend musa --directory records/musa-eval-02
python3 Core/tools/benchmark.py --backend musa --directory records/musa-bench-02 --repeats 20
```

benchmark 固定 FP32、block+nearest，测试 1M/4M/16M；`resident_gpu` 使用 MUSA event。`quant_optimized` 表示当前默认实现，不表示已达到平台极限。NVIDIA nsys/ncu 和 `tools/profile.py` 仍仅适用于 CUDA。

本次实测、限制及其他平台进度见 [适配记录](../../../docs/PLATFORM_ADAPTATION.md)。
