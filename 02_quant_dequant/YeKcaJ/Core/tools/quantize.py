#!/usr/bin/env python3
"""配置与文件入口；数值量化和反量化在 C++/CUDA 中执行。需要 Python 3.11+。"""

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import tomllib
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent


# 自动结果只接受 input/<dtype>/<月日>/<name>，外部输入继续手动指定 --prefix。
def automatic_prefix(source, cfg):
    source = Path(source).resolve()
    try:
        relative = source.relative_to((PROJECT / "input").resolve())
    except ValueError:
        raise ValueError("external input requires --prefix; automatic paths use project input/")
    if len(relative.parts) != 3 or relative.parts[0] not in ("fp16", "fp32"):
        raise ValueError("automatic input layout must be input/fp16|fp32/monthday/name")
    dtype, day, filename = relative.parts
    if len(day) not in (3, 4) or not day.isascii() or not day.isdigit() or source.suffix != "." + dtype:
        raise ValueError("automatic input date or dtype suffix is invalid")
    datetime.datetime(2000, int(day[:-2]), int(day[-2:]))
    with source.open("rb") as stream:
        if stream.read(8) != (b"FP16INP1" if dtype == "fp16" else b"FP32INP1"):
            raise ValueError("input header does not match its dtype directory")
    directory = PROJECT / "results" / dtype / day / Path(filename).stem / cfg["format"]
    directory.mkdir(parents=True, exist_ok=True)
    # mkdir 独占编号；重复/并发运行不会复用同一个结果目录。
    index = 1
    while True:
        run_dir = directory / f"run-{index}"
        try:
            run_dir.mkdir()
            return run_dir / "result"
        except FileExistsError:
            index += 1


# 分块计算哈希，避免为了日志再次将大输入完整读入内存。
def input_sha256(source):
    with Path(source).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

# ===== 1. 配置与可执行文件定位 =====


# 解析 TOML、补默认值并检查类型/取值；target_gpu 仅用于记录，不选择设备。
def read_config(path):
    with Path(path).open("rb") as stream:
        cfg = tomllib.load(stream)
    allowed = {
        "format",
        "block_size",
        "scale_mode",
        "output_type",
        "rounding",
        "target_gpu",
        "seed",
    }
    if set(cfg) - allowed:
        raise ValueError(f"unknown configuration keys: {sorted(set(cfg) - allowed)}")
    if cfg.get("format") not in ("mxfp8", "nvfp4"):
        raise ValueError("format must be mxfp8 or nvfp4")
    defaults = dict(
        block_size=32 if cfg["format"] == "mxfp8" else 16,
        scale_mode="block",
        output_type="fp32",
        rounding="nearest",
        target_gpu="unspecified",
        seed=1234,
    )
    cfg = defaults | cfg
    for key, values in {
        "scale_mode": ("tensor", "block"),
        "output_type": ("fp32", "fp16", "bf16"),
        "rounding": ("nearest", "stochastic"),
    }.items():
        if cfg[key] not in values:
            raise ValueError(f"invalid {key}: {cfg[key]}")
    if type(cfg["block_size"]) is not int or cfg["block_size"] != (
        32 if cfg["format"] == "mxfp8" else 16
    ):
        raise ValueError("block_size must match the standard format (32/16)")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] <= 0xFFFFFFFF:
        raise ValueError("seed must be a uint32")
    if not isinstance(cfg["target_gpu"], str):
        raise ValueError("target_gpu must be a string")
    return cfg


# 两种格式分别构建一个 pipeline 程序，Python 不实现量化 kernel。
def executable(fmt):
    path = ROOT / "build" / ("pipeline_" + fmt)
    if not path.is_file():
        raise ValueError(f"build the CUDA project first: {path}")
    return path


# ===== 2. 终端摘要与单次完整运行 =====


# 把 CUDA 程序的 JSON 指标排成中文摘要；文件日志仍保留原始数值精度。
def format_summary(record, log):
    # 仅改变显示精度；非有限结果不显示成正常误差值。
    def number(value, unit=""):
        if value is None or not math.isfinite(value):
            return "不可用（存在非有限结果）"
        return f"{value:.6g}" + (f" {unit}" if unit else "")

    # 区分校验通过、失败和未执行，不能把未校验当作通过。
    def status(value):
        return "通过" if value is True else "未校验" if value is None else "失败"

    # 向摘要追加一个分区，统一标签对齐和分区间距。
    def section(title, rows):
        lines.append(f"[{title}]")
        for label, value in rows:
            # 中文通常占两个终端列，不能用字符串长度直接对齐。
            width = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in label)
            lines.append(f"  {label}{' ' * max(1, 24 - width)}: {value}")
        lines.append("")

    lines = ["量化与反量化结果", "=" * 56]
    section(
        "配置",
        [
            ("量化格式", record["format"].upper()),
            (
                "输入 -> 输出",
                f"{record['input_dtype'].upper()} -> {record['output_dtype'].upper()}",
            ),
            ("元素数量", f"{record['elements']:,}"),
            ("缩放 / 舍入", f"{record['scale_mode']} / {record['rounding']}"),
            ("随机种子", record["seed"]),
            ("目标 GPU（配置标签）", record["target_gpu"]),
        ],
    )
    section(
        "CPU 对照",
        [
            ("量化编码与 scale", status(record["cpu_quant_match"])),
            ("反量化结果", status(record["cpu_dequant_match"])),
            ("非有限输出数量", record["nonfinite_output"]),
        ],
    )
    section(
        "误差与压缩率",
        [
            ("最大绝对误差", number(record["max_abs_error"])),
            ("平均绝对误差 MAE", number(record["mae"])),
            ("均方误差 MSE", number(record["mse"])),
            ("有效数据压缩率", number(record["compression_payload"], "倍")),
            ("完整文件压缩率", number(record["compression_file"], "倍")),
        ],
    )
    section(
        "单次性能（非正式基准）",
        [
            ("量化设备序列", number(record["quant_kernel_ms"], "ms")),
            ("反量化 kernel", number(record["dequant_kernel_ms"], "ms")),
            ("量化主机流程", number(record["quant_setup_upload_compute_download_ms"], "ms")),
            ("量化逻辑带宽", number(record["quant_logical_GBps"], "GB/s")),
            ("反量化逻辑带宽", number(record["dequant_logical_GBps"], "GB/s")),
        ],
    )
    lines.extend(["  主机流程含分配、上传、计算及下载，不含释放。", ""])
    lines.append("[文件]")
    # 长路径单独一行，便于选择复制，不截断真实文件名。
    for label, path in [
        ("输入", record["input"]),
        ("量化权重", record["packed"]),
        ("反量化张量", record["output"]),
        ("JSON 日志", log),
    ]:
        lines.extend([f"  {label}:", f"    {path}"])
    return "\n".join(lines)


# 生成输出路径 -> 调用 pipeline 做 CPU/GPU 对照 -> 保存 JSON -> 打印摘要。
def run(cfg, source, prefix=None, console_json=False):
    source = Path(source).resolve()
    binary = executable(cfg["format"])
    source_hash = input_sha256(source)
    prefix = automatic_prefix(source, cfg) if prefix is None else Path(prefix).resolve()
    packed, output, log = (
        Path(str(prefix) + ".lpq"),
        Path(str(prefix) + "." + cfg["output_type"]),
        Path(str(prefix) + ".json"),
    )
    if source in (packed, output, log):
        raise ValueError("output paths must differ from input")
    # 默认不覆盖已有结果；换前缀即可保留不同配置的实验记录。
    for path in (packed, output, log):
        if path.exists():
            raise ValueError(f"output exists, use a new prefix: {path}")
    # 参数顺序对应 app/run.cu；verify 启用 CPU 量化逐字节校验。
    command = [
        str(binary),
        str(source),
        str(packed),
        str(output),
        cfg["scale_mode"],
        cfg["rounding"],
        cfg["output_type"],
        str(cfg["block_size"]),
        str(cfg["seed"]),
        "verify",
    ]
    # 子程序失败时立即抛出异常，不将失败运行写成正常日志。
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    record = json.loads(result.stdout.splitlines()[-1])
    record.update(
        target_gpu=cfg["target_gpu"], input=str(source), packed=str(packed), output=str(output),
        input_sha256=source_hash, config=dict(cfg),
        created_at=datetime.datetime.now().astimezone().isoformat(),
    )
    log.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False) if console_json else format_summary(record, log))
    return record


# ===== 3. 固定种子输入生成 =====


# 自动生成 input/<dtype>/<月日>/<编号>，并写旁边的生成参数清单。
def generate_automatic(rows, cols, dtype, distribution, seed):
    today = datetime.date.today()
    directory = PROJECT / "input" / dtype / f"{today.month}{today.day:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        path = directory / f"{index}.{dtype}"
        manifest = path.with_suffix(".json")
        if path.exists() or manifest.exists():
            index += 1
            continue
        try:
            generate(path, rows, cols, dtype, distribution, seed)
            break
        except FileExistsError:
            index += 1
    record = dict(input=str(path.resolve()), dtype=dtype, rows=rows, cols=cols,
                  distribution=distribution, seed=seed, input_sha256=input_sha256(path),
                  created_at=datetime.datetime.now().astimezone().isoformat())
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return path


# 生成小端 v1 输入文件；outlier 在正态数据两端加入 +/-1000 的离群值。
def generate(path, rows, cols, dtype, distribution, seed):
    if rows < 0 or cols < 0 or rows * cols > 1 << 26:
        raise ValueError("invalid dimensions or more than 2^26 elements")
    path = Path(path)
    rng = random.Random(seed)
    values = [
        rng.uniform(-3, 3) if distribution == "uniform" else rng.gauss(0, 1)
        for _ in range(rows * cols)
    ]
    if distribution == "outlier" and values:
        values[0] = 1000
        if len(values) > 1:
            values[-1] = -1000
    path.parent.mkdir(parents=True, exist_ok=True)
    # 文件布局：28 字节头（magic、version、rows、cols）后接行主序数据。
    with path.open("xb") as stream:
        stream.write(
            struct.pack("<8sIQQ", b"FP32INP1" if dtype == "fp32" else b"FP16INP1", 1, rows, cols)
        )
        code = "f" if dtype == "fp32" else "e"
        # 分批打包写入，避免一次创建整个张量的 struct 参数列表。
        for offset in range(0, len(values), 65536):
            chunk = values[offset : offset + 65536]
            stream.write(struct.pack("<" + code * len(chunk), *chunk))


# ===== 4. 命令入口与批量误差评估 =====


# run 执行一个配置；generate 生成输入；evaluate 穷举配置并保存评估结果。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("run")
    p.add_argument("--config", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--prefix", help="手动输出前缀；省略时按 input 目录结构自动分配结果")
    p.add_argument("--json", action="store_true", help="终端输出原始单行 JSON，便于脚本解析")
    p = commands.add_parser("generate")
    p.add_argument("--output", help="手动输入文件路径；省略时存入 input/类型/月日/编号")
    p.add_argument("--rows", type=int, default=128)
    p.add_argument("--cols", type=int, default=129)
    p.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--distribution", choices=["uniform", "normal", "outlier"], default="normal")
    p.add_argument("--seed", type=int, default=1234)
    p = commands.add_parser("evaluate")
    p.add_argument("--directory", required=True, help="new output directory")
    args = parser.parse_args()
    if args.command == "run":
        run(read_config(args.config), args.input, args.prefix, console_json=args.json)
    elif args.command == "generate":
        if args.output:
            generate(args.output, args.rows, args.cols, args.dtype, args.distribution, args.seed)
            path = Path(args.output).resolve()
        else:
            path = generate_automatic(args.rows, args.cols, args.dtype, args.distribution, args.seed)
        print(f"输入文件: {path}")
    else:
        directory = Path(args.directory).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        records = []
        # 共 2 输入类型 * 3 分布 * 2 格式 * 2 scale 模式 * 2 舍入 * 3 输出 = 144 组。
        # 同一输入用于不同配置；这是误差评估，不是重复计时的正式 benchmark。
        for dtype in ("fp32", "fp16"):
            for distribution in ("uniform", "normal", "outlier"):
                source = directory / f"{distribution}.{dtype}"
                generate(source, 128, 129, dtype, distribution, 1234)
                for fmt in ("mxfp8", "nvfp4"):
                    for mode in ("block", "tensor"):
                        for rounding in ("nearest", "stochastic"):
                            for output in ("fp32", "fp16", "bf16"):
                                cfg = dict(
                                    format=fmt,
                                    block_size=32 if fmt == "mxfp8" else 16,
                                    scale_mode=mode,
                                    rounding=rounding,
                                    output_type=output,
                                    seed=1234,
                                    target_gpu="RTX 3060 Laptop",
                                )
                                record = run(
                                    cfg,
                                    source,
                                    directory
                                    / f"{distribution}_{dtype}_{fmt}_{mode}_{rounding}_{output}",
                                    console_json=True,
                                )
                                records.append(dict(record, distribution=distribution))
        (directory / "summary.json").write_text(
            json.dumps(records, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            print(exc.stderr)
        raise SystemExit(str(exc))
