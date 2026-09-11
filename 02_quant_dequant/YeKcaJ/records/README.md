# 实验记录

本目录按**优化轮次**编号组织；归档内容不再出现在主视图，但**没有任何文件被删除**。

## 目录结构

```
records/
├── 01-before/                    优化前基准
│   ├── benchmark/                benchmark.py 输出（MXFP8/NVFP4，1M/4M/16M）
│   └── source/                   优化前源码备份 + 编译选项
├── 02-mxfp8-direct-encode/       第 1 步：nearest 直接编码 E4M3
│   └── benchmark/
├── 03-mxfp8-fused-scale/         第 2 步：scale 归约+广播+编码融合
│   ├── benchmark/
│   ├── repeat/                   复测
│   ├── profile/                  nsys 采集
│   └── validation/               正确性测试 + sanitizer + 源码备份
├── 04-nvfp4-e4m3-scale-fused/    第 3 步：NVFP4 scale 直接编码 + 归约融合（最终态）
│   └── benchmark/
└── _archive/                     归档，保留但不参与主线
    ├── 2026-09-09-legacy-cli/    旧版 CLI 时代的性能与 profile 记录
    ├── evaluation/               误差评估汇总
    ├── demo/                     旧演示输入
    └── duplicate-runs/           与主线重复的测量（数字差异 < 1%）
```

## MXFP8 量化中位数（resident_gpu，预热 3 + 重复 20）

| 元素数 | 优化前 `01-before` | 直接编码 `02-mxfp8-direct-encode` | 编码+融合 `03-mxfp8-fused-scale` |
|---|---:|---:|---:|
| 1M | 0.123904 | 0.055728 | **0.034816** |
| 4M | 0.467968 | 0.199680 | **0.123312** |
| 16M | 1.634304 | 0.708608 | **0.435648** |

两项改动的贡献（以 4M 为例）：

```
直接编码    0.467968 -> 0.199680   2.34x
scale 融合  0.199680 -> 0.123312   1.62x
合计        0.467968 -> 0.123312   3.79x
```

## NVFP4 量化中位数（resident_gpu，预热 3 + 重复 20）

| 元素数 | 优化前 `01-before` | 直接编码+融合 `04-nvfp4-e4m3-scale-fused` |
|---|---:|---:|
| 1M | 0.142336 | **0.063488** |
| 4M | 0.450944 | **0.142336** |
| 16M | 1.742800 | **0.519136** |

两步改动的贡献（以 4M 为例）：

```
scale 直接编码  0.520704 -> 0.238592   2.18x
归约+编码融合   0.238592 -> 0.142336   1.68x
合计            0.450944 -> 0.142336   3.17x
```

注：`01-before` 的 NVFP4 数字（0.450944）与步骤 1 的起点（0.520704）来自不同
采集批次，GPU 状态有别；加速比统一以 `01-before` 为基准计算。

本轮**未能运行 Compute Sanitizer**：Deb 包自带 2022.4.1 版与驱动 596.08 不兼容
（用最小 CUDA 程序验证为环境问题）。替代证据为 CPU oracle 逐字节比对与分组边界
专项测试，详见 `../OPTIMIZATION_LOG.md` 第 2 次记录。

## 各目录对应关系（重组前 -> 重组后）

| 重组后路径 | 重组前名称 |
|---|---|
| `01-before/benchmark/` | `core-baseline-01/` |
| `01-before/source/` | `opt02-v1-source/` |
| `02-mxfp8-direct-encode/benchmark/` | `mxfp8-opt1-direct/` |
| `03-mxfp8-fused-scale/benchmark/` | `mxfp8-opt1-fused/` |
| `03-mxfp8-fused-scale/repeat/` | `mxfp8-opt1-repeat/` |
| `03-mxfp8-fused-scale/profile/` | `mxfp8-opt1-profile/` |
| `03-mxfp8-fused-scale/validation/` | `mxfp8-opt1-validation/` |
| `_archive/2026-09-09-legacy-cli/` | `performance-*`、`profiling-*`、`opt01-*`、`validation-20260909` |
| `_archive/evaluation/` | `core-evaluation-01`、`evaluation-warp-20260909` |
| `_archive/demo/` | `core-demo` |
| `_archive/duplicate-runs/` | `opt02-before`、`opt02-v1-check`、`opt02-v1-profile`、`core-profile-01`、`core-profile-02` |

### 关于 `_archive/duplicate-runs/`

这些目录是**同质重复测量**，保留是为了不消灭证据，但主线分析不需要它们：

- `opt02-before`、`opt02-v1-check`：与 `01-before/benchmark` 测同一份代码，1M 中位数完全相同（0.123904），4M/16M 差异 < 1%。
- `opt02-v1-profile`、`core-profile-01`：同为"优化前"的 nsys（均含 `build_block_scales`、无 `mxfp8_quantize_fused_kernel`）。
- `core-profile-02`：与 `03-mxfp8-fused-scale/profile` 同为"优化后"（均含 `mxfp8_quantize_fused_kernel`）。

判断依据是 nsys 报告中的 kernel 名，见各目录 `mxfp8_stats.txt`。

## 源码备份

| 代码版本 | 备份位置 |
|---|---|
| 优化前 | `01-before/source/source.tar.gz` |
| 优化后（最终态） | `03-mxfp8-fused-scale/validation/source-after.tar.gz` |

NVFP4 优化（第 2 次）的源码即当前 `Core/`，其 SHA256 记录在
`04-nvfp4-e4m3-scale-fused/benchmark/environment.json`；中间态"仅 scale 直接编码"
同样没有独立源码备份。

**中间态"仅直接编码"没有源码备份**，仅有 `02-mxfp8-direct-encode/benchmark/environment.json` 中的 SHA256。如需复现该中间态，需从优化后源码中移除 `mxfp8_quantize_fused_kernel` 并恢复 `launch_quant()` 的三段式路径。

历史 JSON、哈希清单和采集日志中的**旧绝对路径保留不变**，以反映当时的运行环境；它们不是当前路径索引。

## 新增实验

批量 evaluate、benchmark、nsys 的 `--directory` 请指定本目录下的新名字，命名沿用 `NN-<描述>/<类型>/`，不覆盖旧记录。