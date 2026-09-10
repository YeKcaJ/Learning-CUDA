#!/usr/bin/env python3
"""串行运行性能基准，保存原始 JSONL、硬件信息和可读汇总；不覆盖旧结果。"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


# 调用两种格式的 pipeline --benchmark；固定 1M/4M/16M，不读取 TOML。
# 输入生成、预热和计时由 pipeline_main.cu 的 benchmark 完成。
def main():
    # ===== 1. 参数与独立结果目录 =====
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 1000:
        parser.error("repeats must be 1..1000")
    directory = Path(args.directory).resolve()
    # 每轮使用新目录，避免覆盖之前的 baseline 或优化记录。
    directory.mkdir(parents=True, exist_ok=False)

    # 执行只读环境查询；命令失败时抛出异常，避免生成不完整环境记录。
    def command(*cmd):
        return subprocess.check_output(cmd, text=True).strip()

    # ===== 2. 硬件、工具版本与源码标识 =====
    # 记录 Core 源码/配置/构建定义及外部 CPU 算法；哈希不能代替源码备份。
    metadata = dict(
        date=datetime.datetime.now().astimezone().isoformat(),
        repeats=args.repeats,
        warmups=3,
        gpu=command(
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"
        ),
        nvcc=command("nvcc", "--version"),
        cpu=command("lscpu"),
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ROOT.rglob("*"))
            if p.is_file()
            and "build" not in p.relative_to(ROOT).parts
            and "__pycache__" not in p.parts
            and (
                p.suffix in (".cu", ".cuh", ".cpp", ".h", ".py", ".toml")
                or p.name == "CMakeLists.txt"
            )
        },
    )
    for fmt in ("MXFP8", "NVFP4"):
        source = ROOT.parent / ("CPU" + fmt) / "main.cpp"
        metadata["source_sha256"]["../CPU" + fmt + "/main.cpp"] = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
    (directory / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # ===== 3. 串行执行，保存每种格式和规模的原始 JSONL =====
    # 不并行启动 GPU 任务，以免资源争用影响计时。
    records = []
    for fmt in ("mxfp8", "nvfp4"):
        binary = ROOT / "build" / ("pipeline_" + fmt)
        for n in (1 << 20, 4 << 20, 16 << 20):
            print(f"Running {fmt}: elements={n}, repeats={args.repeats}", flush=True)
            result = subprocess.run(
                [str(binary), "--benchmark", str(n), str(args.repeats)],
                capture_output=True,
                text=True,
                check=True,
            )
            (directory / f"{fmt}_{n}.jsonl").write_text(result.stdout)
            records.extend(
                json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
            )
    # ===== 4. 合并 JSON 与可读报告 =====
    # resident_gpu 与 host_api 计时范围不同；后续优化应对比同名、同范围指标。
    # quant_enumeration 是内部对照，当前默认实现对应 quant_optimized。
    (directory / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
    lines = [
        "# 性能测试结果",
        "",
        "GPU 驻留计时使用 CUDA event；host_api 包括量化的分配、传输和释放。",
        "内部对照使用 shared scale 归约与枚举编码；MXFP8 默认使用 scale/直接编码融合 kernel，NVFP4 默认保留分组归约和快速编码。不是原始 CLI 的完整实现对比。",
        "中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。",
        "",
        "| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['format']} | {row['op']} | {row['scope']} | {row['elements']} | {row['median_ms']:.6f} | {row['p95_ms']:.6f} | {row['logical_GBps']:.3f} |"
        )
    (directory / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
