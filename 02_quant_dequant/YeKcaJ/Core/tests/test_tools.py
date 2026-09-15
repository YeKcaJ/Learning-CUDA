"""不启动 profiler 或正式性能测试，检查迁移后的工具路径和报告生成。"""
import importlib.util
import datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

CORE = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location("core_test_" + name, CORE / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ToolTests(unittest.TestCase):
    def test_comparison_rejects_changed_conditions(self):
        tool = load_tool("compare_benchmarks")
        env = dict(protocol="fixed-fp32-v1", formal=True,
                   input=dict(dtype="fp32", shape="1 x elements", elements=[1 << 20, 4 << 20, 16 << 20],
                              generator_seed=20260909, sha256=tool.INPUT_HASHES),
                   quantization=dict(scale_mode="block", rounding="nearest", seed=1234,
                                     block_size=dict(mxfp8=32, nvfp4=16)),
                   metric=dict(op="quant_optimized", scope="resident_gpu", statistic="median_ms",
                               p95="nearest-rank", output_types=["fp32", "fp16", "bf16"]),
                   warmups=3, repeats=20)
        tool.validate_pair(env, dict(env))
        for change in ({"input": {"sha256": {"1M": "different"}}}, {"repeats": 5},
                       {"formal": False}, {"metric": {"scope": "host_api"}}):
            with self.assertRaises(ValueError):
                tool.validate_pair(env, dict(env, **change))
            with self.assertRaises(ValueError):
                tool.validate_pair(dict(env, **change), dict(env, **change))

    def test_backend_selection(self):
        tool = load_tool("quantize")
        with patch.object(Path, "is_file", return_value=True):
            self.assertEqual(tool.executable("nvfp4", "musa"), CORE / "build-musa" / "pipeline_nvfp4")
            self.assertEqual(tool.executable("mxfp8"), CORE / "build" / "pipeline_mxfp8")
            with self.assertRaises(ValueError):
                tool.executable("mxfp8", "unknown")

    def test_automatic_inputs_and_result_numbering(self):
        tool = load_tool("quantize")
        with tempfile.TemporaryDirectory() as folder, patch.object(tool, "PROJECT", Path(folder)), \
             patch.object(tool.datetime, "date") as date:
            date.today.return_value = datetime.datetime(2026, 9, 10).date()
            first = tool.generate_automatic(1, 33, "fp16", "normal", 1234)
            original = first.read_bytes()
            second = tool.generate_automatic(1, 33, "fp16", "normal", 1234)
            fp32 = tool.generate_automatic(1, 33, "fp32", "normal", 1234)
            self.assertEqual(first.relative_to(folder).as_posix(), "input/910/1.fp16")
            self.assertEqual(second.name, "2.fp16")
            self.assertEqual(fp32.name, "1.fp32")
            self.assertEqual(original, second.read_bytes())
            metadata = json.loads(first.with_name(first.name + ".json").read_text())
            self.assertEqual(metadata["input_sha256"], tool.input_sha256(first))
            self.assertEqual(metadata["seed"], 1234)
            cfg = {"format": "nvfp4", "output_type": "fp32"}
            one = tool.automatic_prefix(first, cfg, "cuda")
            two = tool.automatic_prefix(first, cfg, "cuda")
            self.assertEqual(one.relative_to(folder).as_posix(),
                             "output/910/1/nvfp4/cuda/fp16_fp32")
            self.assertEqual(two.name, "fp16_fp32")
            self.assertEqual(tool.automatic_prefix(first, {"format": "mxfp8", "output_type": "fp32"}, "musa").name, "fp16_fp32")
            self.assertEqual(first.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                tool.generate(first, 1, 33, "fp16", "normal", 99)
            self.assertEqual(first.read_bytes(), original)
            with self.assertRaises(ValueError):
                tool.automatic_prefix(Path(folder) / "external.fp16", cfg)
            first.write_bytes(fp32.read_bytes())
            with self.assertRaises(ValueError):
                tool.automatic_prefix(first, cfg)

    def test_benchmark_paths_and_manifest(self):
        tool = load_tool("benchmark")
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            fmt = Path(command[0]).name.removeprefix("pipeline_")
            record = dict(format=fmt, op="quant_optimized", scope="resident_gpu",
                          elements=int(command[2]), median_ms=1, p95_ms=1, logical_GBps=1)
            return subprocess.CompletedProcess(command, 0, json.dumps(record) + "\n", "")

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "benchmark"
            with patch.object(sys, "argv", ["benchmark", "--directory", str(output)]), \
                 patch.object(tool, "fixed_inputs", return_value={str(n): "hash" for n in tool.SIZES}), \
                 patch.object(tool, "file_hash", return_value="binary-hash"), \
                 patch.object(tool.subprocess, "check_output", return_value="test environment"), \
                 patch.object(tool.subprocess, "run", side_effect=fake_run):
                tool.main()
            self.assertEqual(len(calls), 6)
            self.assertEqual({Path(c[0]).parent for c in calls}, {CORE / "build"})
            metadata = json.loads((output / "environment.json").read_text())
            for name in ("runtime/workspace.cu", "kernels/codec.cuh", "CMakeLists.txt",
                         "reference/nvfp4/main.cpp", "app/main.cpp"):
                self.assertIn(name, metadata["source_sha256"])
            self.assertEqual(len(json.loads((output / "summary.json").read_text())), 6)
            self.assertTrue((output / "RESULTS.md").is_file())

    def test_fixed_benchmark_rejects_modified_input(self):
        tool = load_tool("benchmark")
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for n in tool.SIZES:
                (directory / f"normal_{n}.fp32").write_bytes(b"fixed corpus")
            expected = {str(n): tool.file_hash(directory / f"normal_{n}.fp32") for n in tool.SIZES}
            with patch.object(tool, "EXPECTED_INPUT_HASHES", expected):
                first = tool.fixed_inputs(CORE / "build", directory)
                self.assertEqual(tool.fixed_inputs(CORE / "build", directory), first)
                (directory / f"normal_{tool.SIZES[0]}.fp32").write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    tool.fixed_inputs(CORE / "build", directory)
            # 删除本地清单也不能把新输入冒充冻结的 v1 数据。
            (directory / "manifest.json").unlink()
            with self.assertRaises(ValueError):
                tool.fixed_inputs(CORE / "build", directory)

    def test_profile_paths_and_missing_data(self):
        tool = load_tool("profile")
        for valid in (True, False):
            with self.subTest(valid=valid), tempfile.TemporaryDirectory() as folder:
                calls = []

                def fake_run(command, **kwargs):
                    calls.append(command)
                    text = "pipeline::nvfp4_quantize_fused_kernel pipeline::mxfp8_quantize_fused_kernel" if valid else "SKIPPED"
                    return subprocess.CompletedProcess(command, 0, text, "")

                output = Path(folder) / "profile"
                with patch.object(sys, "argv", ["profile", "--directory", str(output)]), \
                     patch.object(tool.subprocess, "check_output", return_value="test nsys"), \
                     patch.object(tool.subprocess, "run", side_effect=fake_run):
                    if valid:
                        tool.main()
                    else:
                        with self.assertRaises(RuntimeError):
                            tool.main()
                for call in calls:
                    if call[1] == "profile":
                        self.assertEqual(Path(call[-4]).parent, CORE / "build")
                self.assertTrue((output / "mxfp8_stats.txt").is_file())

    def test_formal_help_entrypoints(self):
        for name in ("quantize", "benchmark", "profile"):
            result = subprocess.run([sys.executable, str(CORE / "tools" / (name + ".py")),
                                     "--help"], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
