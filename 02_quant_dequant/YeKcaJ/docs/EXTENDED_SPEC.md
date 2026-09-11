# 可配置接口与扩展格式 v2

本文件补充 `REFERENCE_SPEC.md`，不替换或重新生成旧版 CPU golden。旧命令继续读写 v1。
新 `pipeline` 入口写 v2，自定义封装不是 NVIDIA 库的二进制 ABI。

## 输入与配置

输入使用小端、行主序二进制，头部 28 字节：`magic[8]`、`version:uint32=1`、
`rows:uint64`、`cols:uint64`，之后为元素数组。`FP32INP1` 对应 float32，
`FP16INP1` 对应 IEEE binary16。尺寸范围为非负 int64。FP16 在主机无损扩展为 FP32 后量化。
空张量允许；截断、尾随字节、尺寸溢出和 NaN/Inf 输入明确拒绝。异常值实验使用有限离群值。

`Core/tools/quantize.py` 使用 Python 3.11+ 标准库 `tomllib` 解析 TOML，不手写字符串解析器。
支持 format、block_size、scale_mode、output_type、rounding、target_gpu、seed。
未知键和非法值报错。block_size 固定为 MXFP8=32、NVFP4=16；不支持任意非标准块尺寸。
target_gpu 仅为用户的报告标签，真实性能环境由 `Core/tools/benchmark.py` 查询保存。

## 缩放

block 模式与 v1 一样按展平数组分组，允许跨行。
tensor 模式将整个张量视为一组，只存一个局部 scale（空张量存零个）。
这是用于比较缩放策略的软件扩展模式，不声称是标准 MXFP8/NVFP4 固定 block 格式。

MXFP8：组最大绝对值 m，`e=clamp(ceil(log2(m/448))+127,0,254)`，`s=2^(e-127)`。
m=0 时 e=127；正 m 的除法结果下溢至零时 e=0。数据为 E4M3FN，重建 `decode(data)*s`。

NVFP4：全局最大值 M，`g=M/(6*448)`，全零张量 g=1。
正 M 的 g 下溢时夹到最小正 FP32 subnormal，避免零除。
每组 `b=encode_e4m3(m/(6*g))`，m=0 时 b=0，`s=g*decode_e4m3(b)`。
数据由 `input/s` 编为 E2M1，s<=0 时编正零。重建值 `decode_e2m1(data)*s`。
tensor 模式仍保留 NVFP4 分层公式，但只有一个 b；两级 scale 不代表两种独立模式。

## 舍入

nearest 保留冻结规则：饱和至有限范围，等距时保留枚举顺序中先出现者。
E4M3FN 的 NaN 编码不用于量化，E4M3 的零结果使用正零；E2M1 符号独立，保留负零。
这不是 IEEE nearest-even；FP16/BF16 最终输出转换才使用 nearest-even。

stochastic 仅用于元素，不用于 scale：在幅值邻值 a<=x<=b 之间，以上舍入概率
`p=(x-a)/(b-a)` 选 b，否则选 a，随后恢复符号。范围之外先饱和；精确可表示值不变。
按 seed:uint32 与展平元素下标生成无状态均匀随机数：

```text
x = seed XOR uint32(index) XOR (uint32(index>>32) * 0x9e3779b9)
x += 0x9e3779b9
x ^= x>>16; x *= 0x85ebca6b
x ^= x>>13; x *= 0xc2b2ae35
x ^= x>>16
u = (double(x)+0.5)/2^32
```

所有整数运算按 uint32 模 2^32。u<p 时上舍入；固定 seed 可重现，与线程调度无关。
CPU 随机参考用线性邻值查找，GPU 用二分，二者均检查中点概率和改 seed 的效果。

## 量化文件

v2 头部固定 72 字节，全部小端，不直接写 C++ struct，避免 padding 差异：

| 偏移 | 字段 | 类型 |
|---:|---|---|
| 0 | magic = LPQUANT2 | 8 bytes |
| 8 | version = 2 | uint32 |
| 12 | format = 8(MXFP8) / 4(NVFP4) | uint32 |
| 16 | rows | uint64 |
| 24 | cols | uint64 |
| 32 | group（tensor 时为 max(1,count)） | uint64 |
| 40 | tensor = 0/1 | uint32 |
| 44 | stochastic = 0/1 | uint32 |
| 48 | seed | uint32 |
| 52 | global_scale（MXFP8 为 1） | float32 |
| 56 | data_bytes | uint64 |
| 64 | scale_bytes | uint64 |
| 72 | data，随后 scales | byte arrays |

NVFP4 每字节包含两个元素，偶数下标在低 nibble；奇数尾部的高 nibble 必须为零。
读回检查尺寸、长度、模式、scale 和 padding。反量化输出沿用 36 字节的
FP32DEQ1 / FP16DEQ1 / BF16DEQ1 格式，不需要另行生成 16 位冻结 golden。

## 指标口径

- max_abs_error/MAE/MSE：原始输入（FP16 时为读入的 half 值）与最终输出的差异。
- 若输出 dtype 溢出为 Inf，误差置为 JSON null 并记录 nonfinite_output，不能伪报零误差。
- compression_payload：输入元素字节数 / (data + scales + NVFP4 的 4 字节 global scale)。
- compression_file：(输入 28 字节头+payload)/(量化 72 字节头+data+scales)。
- quant_kernel_ms：event 包围整个设备量化序列，包括归约、scale 和编码，不含 H2D/D2H。
- quant_setup_upload_compute_download_ms：一次量化的主机分配、上传、设备计算和下载；不含后续释放。
- dequant_kernel_ms：event 包围反量化 kernel。
- logical_GBps：逻辑输入输出字节数/耗时，不是硬件实际访问总量或带宽利用率。
- resident_gpu 基准复用所有显存/event；host_api 基准每次重新分配、传输并释放。
- 新基准 median 为标准中位数；P95 为 nearest-rank；旧 benchmark 保留历史定义。
