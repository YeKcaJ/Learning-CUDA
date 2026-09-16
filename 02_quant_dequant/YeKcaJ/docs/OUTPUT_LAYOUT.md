# 输入、输出与记录目录

## `output/`：最终交付输出

题目要求的输出全部放在这里。每次运行按日期、输入编号、格式和平台隔离：

```text
output/<月日>/<输入编号>/<格式>/<backend>/
├── <输入dtype>_<输出dtype>.lpq         低精度权重：header + packed data + scale
├── <输入dtype>_<输出dtype>.<输出dtype>  反量化张量：按行主序二进制
└── <输入dtype>_<输出dtype>.json         误差、压缩率、kernel时间、带宽
```

CUDA 和 MUSA 的文件不能共用目录；同一路径不能覆盖已有结果。

## `input/`：输入文件

```text
input/<月日>/<编号>.fp32|.fp16       普通功能输入
input/benchmark-v1/                  固定性能输入，只放 normal_N.fp32 和 manifest.json
input/evaluation-v1/                 固定误差评估输入：uniform、normal、outlier
```

误差评估的最终结果放到 `output/evaluation-v1/<backend>/`，只保留 `uniform.json`、`normal.json`、`outlier.json` 和 `summary.json`；144 组组合的中间文件使用临时目录，不把最终输出混进 `records/`。

## `records/`：过程证据

只保存 benchmark 原始 JSONL、环境和哈希、nsys/NCU/检查工具日志、平台适配过程、历史版本和重复实验归档。它用于审计和复现，不是用户运行后寻找三类交付文件的目录。

可以清理 `output/` 旧结果和可重建的 `Core/build/`；不要删除 `input/benchmark-v1/` 或 `records/` 原始证据。
