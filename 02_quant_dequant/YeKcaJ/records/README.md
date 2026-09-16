# records 目录说明

`records/` 只保存实验过程和原始证据，不保存用户运行后要提交的最终输出。

分类规则：

- `01-before`～`12-*`：CUDA 优化轮次；
- `musa-*`：摩尔线程适配与优化；
- `controlled-v1`：跨平台固定输入复测；
- `benchmark-audit`：条件、哈希和审计清单；
- `_archive`：旧版、重复实验和历史快照；
- 各轮目录中的 `benchmark/`、`profile/`、`validation/`、`repeat/` 分别表示性能、nsys、正确性和复测证据。

- `01-before`～`12-*`：CUDA 各优化轮次的 benchmark、验证和 profile；
- `musa-opt01`、`musa-opt02`、`musa-s4000-*`：摩尔线程适配与测试；
- `controlled-v1`：固定输入的 CUDA/MUSA 复测；
- `benchmark-audit`、`_archive`：条件审计和历史归档。

每个 benchmark 目录中的 `summary.json`、JSONL、`environment.json` 和日志是原始证据。新实验使用新的目录名，禁止覆盖旧记录。

题目要求的三类最终输出统一查看 `output/`，目录格式见 [`docs/OUTPUT_LAYOUT.md`](../docs/OUTPUT_LAYOUT.md)。
