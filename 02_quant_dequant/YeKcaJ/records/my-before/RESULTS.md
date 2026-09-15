# 性能测试结果

GPU 驻留计时使用 CUDA event；host_api 包括量化的分配、传输和释放。
内部对照保留 shared scale 归约与枚举元素编码；两种格式的默认 block+nearest 均使用 scale/编码融合 kernel，NVFP4 另含两级全局归约。内部对照的 scale 也可能共享优化，优化前后请比较留存的 quant_optimized 记录。
中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。

| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |
|---|---|---|---:|---:|---:|---:|
| mxfp8 | quant_enumeration | resident_gpu | 1048576 | 0.566272 | 0.568320 | 9.316 |
| mxfp8 | quant_optimized | resident_gpu | 1048576 | 0.034832 | 0.035840 | 151.460 |
| mxfp8 | dequant_fp32 | resident_gpu | 1048576 | 0.031744 | 0.032768 | 166.194 |
| mxfp8 | dequant_fp16 | resident_gpu | 1048576 | 0.024464 | 0.024576 | 129.925 |
| mxfp8 | dequant_bf16 | resident_gpu | 1048576 | 0.026624 | 0.026624 | 119.385 |
| mxfp8 | quant_optimized | host_api | 1048576 | 2.077935 | 2.286439 | 2.539 |
| mxfp8 | quant_reference | cpu | 1048576 | 1250.040458 | 1251.694456 | 0.004 |
| mxfp8 | dequant_fp32 | cpu | 1048576 | 13.101327 | 14.292152 | 0.403 |
| mxfp8 | quant_enumeration | resident_gpu | 4194304 | 2.236416 | 2.239488 | 9.436 |
| mxfp8 | quant_optimized | resident_gpu | 4194304 | 0.122880 | 0.127840 | 171.733 |
| mxfp8 | dequant_fp32 | resident_gpu | 4194304 | 0.111616 | 0.112640 | 189.064 |
| mxfp8 | dequant_fp16 | resident_gpu | 4194304 | 0.103440 | 0.106496 | 122.912 |
| mxfp8 | dequant_bf16 | resident_gpu | 4194304 | 0.109568 | 0.124928 | 116.037 |
| mxfp8 | quant_optimized | host_api | 4194304 | 4.715789 | 6.858041 | 4.475 |
| mxfp8 | quant_enumeration | resident_gpu | 16777216 | 7.861104 | 8.050688 | 10.738 |
| mxfp8 | quant_optimized | resident_gpu | 16777216 | 0.436592 | 0.458432 | 193.339 |
| mxfp8 | dequant_fp32 | resident_gpu | 16777216 | 0.405488 | 0.408576 | 208.170 |
| mxfp8 | dequant_fp16 | resident_gpu | 16777216 | 0.364544 | 0.366592 | 139.506 |
| mxfp8 | dequant_bf16 | resident_gpu | 16777216 | 0.377856 | 0.403456 | 134.591 |
| mxfp8 | quant_optimized | host_api | 16777216 | 14.665010 | 16.937566 | 5.756 |
| nvfp4 | quant_enumeration | resident_gpu | 1048576 | 0.227296 | 0.246848 | 21.048 |
| nvfp4 | quant_optimized | resident_gpu | 1048576 | 0.110080 | 0.118784 | 43.461 |
| nvfp4 | dequant_fp32 | resident_gpu | 1048576 | 0.043520 | 0.054272 | 109.930 |
| nvfp4 | dequant_fp16 | resident_gpu | 1048576 | 0.037888 | 0.061440 | 70.919 |
| nvfp4 | dequant_bf16 | resident_gpu | 1048576 | 0.040960 | 0.048832 | 65.600 |
| nvfp4 | quant_optimized | host_api | 1048576 | 1.658086 | 1.906245 | 2.885 |
| nvfp4 | quant_reference | cpu | 1048576 | 87.518673 | 91.841793 | 0.055 |
| nvfp4 | dequant_fp32 | cpu | 1048576 | 11.278055 | 11.333151 | 0.424 |
| nvfp4 | quant_enumeration | resident_gpu | 4194304 | 0.719904 | 0.726016 | 26.582 |
| nvfp4 | quant_optimized | resident_gpu | 4194304 | 0.143616 | 0.149504 | 133.248 |
| nvfp4 | dequant_fp32 | resident_gpu | 4194304 | 0.091136 | 0.093184 | 209.978 |
| nvfp4 | dequant_fp16 | resident_gpu | 4194304 | 0.081920 | 0.086016 | 131.200 |
| nvfp4 | dequant_bf16 | resident_gpu | 4194304 | 0.090112 | 0.091008 | 119.273 |
| nvfp4 | quant_optimized | host_api | 4194304 | 4.075745 | 4.706533 | 4.695 |
| nvfp4 | quant_enumeration | resident_gpu | 16777216 | 2.815824 | 3.150848 | 27.184 |
| nvfp4 | quant_optimized | resident_gpu | 16777216 | 0.520192 | 0.529408 | 147.150 |
| nvfp4 | dequant_fp32 | resident_gpu | 16777216 | 0.339968 | 0.343072 | 225.157 |
| nvfp4 | dequant_fp16 | resident_gpu | 16777216 | 0.302080 | 0.305952 | 142.319 |
| nvfp4 | dequant_bf16 | resident_gpu | 16777216 | 0.331776 | 0.357344 | 129.580 |
| nvfp4 | quant_optimized | host_api | 16777216 | 13.855218 | 17.003300 | 5.525 |
