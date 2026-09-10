# 优化日志

## 第 0 次：优化前基准（2026-09-10）

条件：RTX 3060 Laptop，FP32 输入，block / nearest，预热 3 次、计时 20 次。
以下为设备驻留耗时，单位 ms；反量化列为中位数。

| 格式 | 元素数 | 量化中位数 | 量化 P95 | FP32反量化 | FP16反量化 | BF16反量化 |
|---|---:|---:|---:|---:|---:|---:|
| MXFP8 | 1M | 0.123904 | 0.130048 | 0.031744 | 0.024576 | 0.027648 |
| MXFP8 | 4M | 0.467968 | 0.474112 | 0.111616 | 0.103424 | 0.108944 |
| MXFP8 | 16M | 1.634304 | 1.679360 | 0.404928 | 0.364544 | 0.381952 |
| NVFP4 | 1M | 0.142336 | 0.178048 | 0.025600 | 0.020272 | 0.022528 |
| NVFP4 | 4M | 0.450944 | 0.470016 | 0.088064 | 0.078848 | 0.086032 |
| NVFP4 | 16M | 1.742800 | 2.141184 | 0.339824 | 0.302080 | 0.332800 |

nsys 定位（4M，本次采集的 kernel 中位数）：
- MXFP8：编码 0.425761 ms，scale 计算 0.102718 ms，优先优化编码。
- NVFP4：scale 计算 0.408688 ms，编码 0.091735 ms，全局归约两阶段合计约 0.072145 ms，优先关注 scale。

正确性：7/7 核心测试通过。尚未修改 kernel。
数据来源：[benchmark](records/01-before/benchmark/RESULTS.md)、[MXFP8 nsys](records/_archive/duplicate-runs/opt02-v1-profile/mxfp8_stats.txt)、[NVFP4 nsys](records/_archive/duplicate-runs/opt02-v1-profile/nvfp4_stats.txt)。

## 第 1 次：MXFP8 直接编码 + scale 融合（2026-09-10）

改动：nearest 用 FP32 指数/尾数直接生成 E4M3；block 模式归约、scale 广播、编码合为一个 kernel。随机舍入、NVFP4 和反量化算法不变。

量化性能（同第0次条件，单位 ms；加速比相对第0次中位数）：

| 元素数 | 优化前 | 仅直接编码 | 直接编码+融合 | 融合 P95 | 加速比 |
|---|---:|---:|---:|---:|---:|
| 1M | 0.123904 | 0.055728 | 0.034816 | 0.040960 | 3.56x |
| 4M | 0.467968 | 0.199680 | 0.123312 | 0.128896 | 3.79x |
| 16M | 1.634304 | 0.708608 | 0.435648 | 0.737280 | 3.75x |

nsys：4M 默认路径由两个 kernel 合为 `mxfp8_quantize_fused_kernel`，中位数 0.143548 ms（与 benchmark 分开计量）。
复测：1M/4M/16M 中位数为 0.034816/0.122880/0.432128 ms；16M P95仍有波动。NVFP4 16M同环境旧/新对照为1.746352/1.737728 ms，未见稳定回退。
正确性：7/7测试通过，覆盖中点两侧、随机FP32位模式和跨scale分组；Compute Sanitizer memcheck/racecheck/synccheck均0错误。
结论：保留；本轮仅提升量化，反量化未优化。
数据：[分步](records/02-mxfp8-direct-encode/benchmark/RESULTS.md)、[融合](records/03-mxfp8-fused-scale/benchmark/RESULTS.md)、[复测](records/03-mxfp8-fused-scale/repeat/RESULTS.md)、[nsys](records/03-mxfp8-fused-scale/profile/mxfp8_stats.txt)。
