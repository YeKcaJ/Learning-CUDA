# 实验记录

本目录存放历史和后续批量实验记录；根目录 `input/` 存日常输入，`results/` 只存对应的单次运行输出。

2026-09-10 从 results 原样迁入 11 个目录：

- `opt01-baseline`、`opt01-profile`：优化轮次的 baseline 和 nsys 报告。
- `performance-*`、`profiling-*`：历史各阶段性能与分析记录。
- `validation-20260909`：历史正确性日志和哈希清单。
- `evaluation-warp-20260909`、`core-evaluation-01`：两轮评估汇总，逐样例产物此前已清出项目并备份。
- `core-demo`：保留旧演示输入，内容与 `input/fp16/910/1.fp16` 相同。

历史 JSON、哈希清单和采集日志中的旧绝对路径保留，以反映当时运行环境；它们不是当前路径索引。迁移没有重写记录或生成新的性能结论。

后续批量 evaluate、benchmark、nsys 的 `--directory` 请指定本目录下的新名字，不覆盖旧记录。恢复原位置时可将对应子目录移回 results，本次没有永久删除文件。
