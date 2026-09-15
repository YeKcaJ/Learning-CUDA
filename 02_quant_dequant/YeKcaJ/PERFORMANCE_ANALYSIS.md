# 当前性能分析

## 结论

当前 kernel 已经不能仅凭 benchmark 认定达到硬件极限，因为 WSL 中 Nsight Compute 无法连接 CUDA driver，缺少 occupancy、寄存器、SM 利用率和实际 DRAM 吞吐数据。不过现有证据表明：MXFP8 量化 kernel 已经很快，继续改动线程内编码的收益可能有限；完整流程的主要瓶颈已经转移到显存分配、Host↔Device 拷贝和同步。

## 当前数据

RTX 3060 Laptop，FP32 输入，block/nearest，预热 3 次、计时 20 次：

| 格式/操作 | 1M median | 4M median | 16M median |
|---|---:|---:|---:|
| MXFP8 量化 | 0.025600 ms | 0.070656 ms | 0.269824 ms |
| MXFP8 FP32 反量化 | 0.034816 ms | 0.111616 ms | 0.408480 ms |
| NVFP4 量化 | 0.048928 ms | 0.143360 ms | 0.522144 ms |

Nsight Systems 对 4M MXFP8 的当前向量化 kernel 测得中位数约 0.069114 ms。相同采集中，Host API 时间主要来自 `cudaMalloc`、`cudaMemcpy` 和 `cudaEventSynchronize`，因此 host_api 的约 5--6 ms 不能代表 kernel 本身的速度。

## 如何判断是否接近极限

1. **先固定输入和环境**：同一 GPU、FP32、block/nearest、相同元素数，预热 3 次，重复至少 20 次，记录 median 和 P95。
2. **区分三种时间**：只比较 `resident_gpu` 判断 kernel；比较 `host_api` 判断完整流程；用 Nsight Systems 判断分配、拷贝、同步和 kernel 的时间线。
3. **检查规模扩展**：如果元素数扩大 4 倍，kernel 时间大致也扩大 4 倍，说明主要受数据处理量限制；如果小规模明显更慢，通常是启动和同步开销。
4. **确认正确性不回退**：packed data、scale 必须逐字节匹配 CPU reference。
5. **获得硬件计数器后再下最终结论**：Nsight Compute 需要成功采集 occupancy、寄存器、memory throughput 和 warp stall；当前 WSL 的 `ERR_NVGPUCTRPERM`/driver stub 问题尚未解决，因此目前只能做阶段性判断。

## 可复现命令

普通基准：

```bash
python3 Core/tools/benchmark.py --directory records/new-run --repeats 20
```

Nsight Systems：

```bash
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  -o records/nsys/mxfp8 Core/build/pipeline_mxfp8 \
  --benchmark 4194304 20
nsys stats --force-export=true --report \
  cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
  records/nsys/mxfp8.nsys-rep
```

## 下一步优先级

1. 优先优化完整流程：复用 workspace，减少 `cudaMalloc/cudaFree`，减少 Host↔Device 往返。
2. 对 NVFP4 继续观察全局归约和 kernel 启动开销；`float2` 实验收益不足，已不纳入优化。
3. 对 MXFP8 暂停盲目改变编码逻辑；先解决 Nsight Compute 权限，再根据寄存器、occupancy 和 stall 数据决定是否继续。

数据来源：[MXFP8 benchmark](records/07-mxfp8-vectorized/RESULTS.md)、[Nsight Systems 分析](records/09-nsys-mxfp8/SUMMARY.md)。

## Nsight Compute 结论（2026-09-14）

GPU 性能计数器已恢复，完成了 4M 输入的正式采集：

| kernel | DRAM throughput | SM throughput | achieved occupancy | duration |
|---|---:|---:|---:|---:|
| MXFP8 vectorized | 90.28% | 81.95% | 88.92% | 71.14 us |
| NVFP4 fused quantize | 56.08% | 84.46% | 88.33% | 106.59 us |
| NVFP4 `maximum` | 88.01% | 32.50% | 88.55% | 61.31 us |

MXFP8 已同时接近 DRAM 和 SM 吞吐上限，occupancy、寄存器和 spilling 都不是主要问题；Nsight Compute 对全局 store 布局估计仍有约 4.5% 的局部改进空间。因此可以认为 MXFP8 已接近当前实现条件下的实用极限，后续只适合做小幅 store/coalescing 调整，不适合继续大改线程分工。

NVFP4 的 fused quantize 是计算受限（SM 84.46%，DRAM 56.08%），而 `maximum` 是访存受限（DRAM 88.01%，SM 32.50%）。所以 NVFP4 仍有两个明确方向：

1. `nvfp4_quantize_fused_kernel`：减少编码、除法、分支和谓词开销；测试每线程处理更多 packed byte，但要观察寄存器和 active threads/warp。
2. `maximum`：测试每线程多加载几个 FP32、调整 partial block 数和线程块大小；目标是减少约 61 us 的全局最大值归约时间。

不建议继续投入 `float2` 输入加载：此前实测收益约为 ±1%，且不稳定。完整报告：[MXFP8 NCU](results/ncu/mxfp8-test-analysis.md)、[NVFP4 NCU](results/ncu/nvfp4-test.ncu-rep)、[NVFP4 maximum NCU](results/ncu/nvfp4-maximum.ncu-rep)。
