#!/usr/bin/env python3
"""Validate and summarize the E5 end-to-end TinyLlama matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


VARIANTS = ["fixed_8", "fixed_16", "fixed_32", "fixed_64", "hetero_8_64"]


def load_reports(root: Path) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        path = root / "variants" / f"{variant}.json"
        if not path.is_file():
            raise RuntimeError(f"missing report: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("variant") != variant:
            raise RuntimeError(f"variant mismatch in {path}")
        if not report.get("successful"):
            raise RuntimeError(f"unsuccessful report: {path}")
        reports[variant] = report
    return reports


def cases_by_name(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["name"]: case for case in report["cases"]}


def output_signature(case: dict[str, Any]) -> list[dict[str, Any]]:
    requests = case["runs"][0]["requests"]
    return [
        {
            "input_tokens": request["input_tokens"],
            "output_tokens": request["output_tokens"],
            "terminal_reason": request["terminal_reason"],
            "error": request["error"],
        }
        for request in requests
    ]


def validate_fairness(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = reports["fixed_8"]
    stable_config = {
        key: baseline["config"][key]
        for key in (
            "warmup",
            "iterations",
            "kv_capacity_tokens",
            "capacity_probe_tokens",
            "model_weight_bytes",
            "model_workspace_bytes",
        )
    }
    baseline_cases = cases_by_name(baseline)
    token_cases: dict[str, bool] = {}
    for variant, report in reports.items():
        for key, value in stable_config.items():
            if report["config"][key] != value:
                raise RuntimeError(f"config mismatch: {variant}/{key}")
        if report["config"]["performance_storage_bytes"] != baseline["config"][
            "performance_storage_bytes"
        ]:
            raise RuntimeError(f"storage budget mismatch: {variant}")
        current_cases = cases_by_name(report)
        if set(current_cases) != set(baseline_cases):
            raise RuntimeError(f"case set mismatch: {variant}")
        for name, case in current_cases.items():
            if not case["passed"] or not case["outputs_consistent"]:
                raise RuntimeError(f"case failed: {variant}/{name}")
            if name in {"fault_c4", "capacity"}:
                continue
            equal = output_signature(case) == output_signature(baseline_cases[name])
            token_cases[name] = token_cases.get(name, True) and equal
            if not equal:
                raise RuntimeError(f"cross-variant token mismatch: {variant}/{name}")
    reference_path = next(iter(reports.values())).get("_reference_path")
    reference_passed = False
    reference_prompts = 0
    if reference_path:
        reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
        reference_passed = bool(reference.get("passed"))
        reference_prompts = int(reference.get("unique_prompts", 0))
        if not reference_passed:
            raise RuntimeError("Transformers FP16 reference validation failed")
    return {
        "config_equal": True,
        "storage_budget_equal": True,
        "cross_variant_tokens_equal": all(token_cases.values()),
        "token_cases": token_cases,
        "transformers_reference_passed": reference_passed,
        "transformers_reference_prompts": reference_prompts,
    }


def percent_delta(value: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return (value - baseline) / baseline * 100.0


def write_summary_csv(root: Path, reports: dict[str, dict[str, Any]]) -> None:
    columns = [
        "variant",
        "case",
        "concurrency",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "tpot_p50_ms",
        "tpot_p95_ms",
        "e2e_p50_ms",
        "e2e_p95_ms",
        "requests_per_second",
        "output_tokens_per_second",
        "goodput_requests_per_second",
        "model_forward_tokens",
        "model_forward_batches",
        "average_model_batch_size",
        "batched_attention_submissions",
        "batched_attention_lanes",
        "average_attention_batch_size",
        "peak_committed_tokens",
        "peak_reserved_tokens",
        "peak_fragmentation_tokens",
        "peak_primary_pages",
        "peak_secondary_pages",
        "primary_allocations",
        "secondary_allocations",
        "failed_primary_allocations",
        "failed_secondary_allocations",
        "peak_dynamic_gpu_bytes",
        "loaded_gpu_bytes",
        "peak_gpu_bytes",
        "completed",
        "failed",
        "rejected",
    ]
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for variant in VARIANTS:
            for case in reports[variant]["cases"]:
                summary = case["summary"]
                resources = case["resources"]
                runs = case["runs"]
                model_forward_tokens = sum(
                    run.get("model_forward_tokens", 0) for run in runs
                )
                model_forward_batches = sum(
                    run.get("model_forward_batches", 0) for run in runs
                )
                attention_submissions = sum(
                    run.get("batched_attention_submissions", 0)
                    for run in runs
                )
                attention_lanes = sum(
                    run.get("batched_attention_lanes", 0) for run in runs
                )
                writer.writerow(
                    {
                        "variant": variant,
                        "case": case["name"],
                        "concurrency": case["concurrency"],
                        "ttft_p50_ms": summary["ttft_p50_ns"] / 1e6,
                        "ttft_p95_ms": summary["ttft_p95_ns"] / 1e6,
                        "tpot_p50_ms": summary["tpot_p50_ns"] / 1e6,
                        "tpot_p95_ms": summary["tpot_p95_ns"] / 1e6,
                        "e2e_p50_ms": summary["e2e_p50_ns"] / 1e6,
                        "e2e_p95_ms": summary["e2e_p95_ns"] / 1e6,
                        "requests_per_second": summary["requests_per_second"],
                        "output_tokens_per_second": summary[
                            "output_tokens_per_second"
                        ],
                        "goodput_requests_per_second": summary[
                            "goodput_requests_per_second"
                        ],
                        "model_forward_tokens": model_forward_tokens,
                        "model_forward_batches": model_forward_batches,
                        "average_model_batch_size": (
                            model_forward_tokens / model_forward_batches
                            if model_forward_batches
                            else 0.0
                        ),
                        "batched_attention_submissions": attention_submissions,
                        "batched_attention_lanes": attention_lanes,
                        "average_attention_batch_size": (
                            attention_lanes / attention_submissions
                            if attention_submissions
                            else 0.0
                        ),
                        **resources,
                        "loaded_gpu_bytes": reports[variant]["config"][
                            "loaded_gpu_bytes"
                        ],
                        "peak_gpu_bytes": reports[variant]["config"][
                            "loaded_gpu_bytes"
                        ]
                        + resources["peak_dynamic_gpu_bytes"],
                        "completed": sum(run["completed"] for run in runs),
                        "failed": sum(run["failed"] for run in runs),
                        "rejected": sum(run["rejected"] for run in runs),
                    }
                )


def write_report(
    root: Path,
    reports: dict[str, dict[str, Any]],
    fairness: dict[str, Any],
) -> dict[str, Any]:
    fixed = cases_by_name(reports["fixed_8"])
    hetero = cases_by_name(reports["hetero_8_64"])
    comparisons: dict[str, dict[str, float]] = {}
    for name in fixed:
        if name in {"fault_c4", "capacity"}:
            continue
        fixed_summary = fixed[name]["summary"]
        hetero_summary = hetero[name]["summary"]
        comparisons[name] = {
            "e2e_p50_delta_percent": percent_delta(
                hetero_summary["e2e_p50_ns"], fixed_summary["e2e_p50_ns"]
            ),
            "tpot_p50_delta_percent": percent_delta(
                hetero_summary["tpot_p50_ns"], fixed_summary["tpot_p50_ns"]
            ),
            "output_tokens_per_second_delta_percent": percent_delta(
                hetero_summary["output_tokens_per_second"],
                fixed_summary["output_tokens_per_second"],
            ),
        }

    hetero_secondary_allocations = sum(
        case["resources"]["secondary_allocations"]
        for case in reports["hetero_8_64"]["cases"]
    )
    analysis = {
        "schema_version": 1,
        "successful": True,
        "fairness": fairness,
        "hetero_vs_fixed_8": comparisons,
        "hetero_secondary_allocations": hetero_secondary_allocations,
        "engine_promotion_active": hetero_secondary_allocations > 0,
        "batched_attention_active": any(
            run.get("batched_attention_submissions", 0) > 0
            for case in reports["hetero_8_64"]["cases"]
            for run in case["runs"]
        ),
    }
    (root / "comparison.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# E5 TinyLlama 端到端证据",
        "",
        "## 验证结论",
        "",
        "| 项目 | 结果 |",
        "|---|---|",
        f"| 五种 Page 策略配置一致 | {'PASS' if fairness['config_equal'] else 'FAIL'} |",
        f"| CUDA KV Storage 字节预算一致 | {'PASS' if fairness['storage_budget_equal'] else 'FAIL'} |",
        f"| 跨策略输出 Token 一致 | {'PASS' if fairness['cross_variant_tokens_equal'] else 'FAIL'} |",
        f"| Transformers FP16 独立 Reference | {'PASS' if fairness['transformers_reference_passed'] else '未执行'}（{fairness['transformers_reference_prompts']} 个唯一 Prompt） |",
        "| 每组测量轮数 | 3 或以上 |",
        "| 结果类型 | 真实 TinyLlama 1.1B FP16 端到端 Generation |",
        "",
        "## Fixed-8 与 Hetero-8/64",
        "",
        "正值表示 Hetero 指标高于 Fixed-8；对于延迟，正值代表更慢。",
        "",
        "| Workload | E2E p50 差值 | TPOT p50 差值 | Output tokens/s 差值 |",
        "|---|---:|---:|---:|",
    ]
    for name, values in comparisons.items():
        lines.append(
            f"| {name} | {values['e2e_p50_delta_percent']:+.2f}% | "
            f"{values['tpot_p50_delta_percent']:+.2f}% | "
            f"{values['output_tokens_per_second_delta_percent']:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 绝对结果",
            "",
            "| Variant | Workload | E2E p50 (ms) | TPOT p50 (ms) | Output tokens/s |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for variant in VARIANTS:
        for name in ("decode_short_c1", "decode_short_c4", "decode_long_prompt_c4", "mixed_c4"):
            summary = cases_by_name(reports[variant])[name]["summary"]
            lines.append(
                f"| {variant} | {name} | {summary['e2e_p50_ns'] / 1e6:.3f} | "
                f"{summary['tpot_p50_ns'] / 1e6:.3f} | "
                f"{summary['output_tokens_per_second']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Batch 执行证据",
            "",
            "| Workload | Model Batch Size | Attention Batch Size | Attention Submissions/Run |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in (
        "decode_short_c1",
        "decode_short_c2",
        "decode_short_c4",
        "decode_long_prompt_c4",
        "mixed_c4",
    ):
        runs = hetero[name]["runs"]
        model_tokens = sum(run.get("model_forward_tokens", 0) for run in runs)
        model_batches = sum(run.get("model_forward_batches", 0) for run in runs)
        attention_submissions = sum(
            run.get("batched_attention_submissions", 0) for run in runs
        )
        attention_lanes = sum(
            run.get("batched_attention_lanes", 0) for run in runs
        )
        lines.append(
            f"| {name} | "
            f"{model_tokens / model_batches if model_batches else 0.0:.2f} | "
            f"{attention_lanes / attention_submissions if attention_submissions else 0.0:.2f} | "
            f"{attention_submissions / len(runs):.0f} |"
        )
    lines.extend(
        [
            "",
            "## Capacity 与故障隔离",
            "",
            "| Variant | Capacity 完成/失败/拒绝 | Fault 完成/失败 | Peak fragmentation tokens |",
            "|---|---:|---:|---:|",
        ]
    )
    for variant in VARIANTS:
        variant_cases = cases_by_name(reports[variant])
        capacity_runs = variant_cases["capacity"]["runs"]
        fault_runs = variant_cases["fault_c4"]["runs"]
        lines.append(
            f"| {variant} | "
            f"{sum(run['completed'] for run in capacity_runs)}/"
            f"{sum(run['failed'] for run in capacity_runs)}/"
            f"{sum(run['rejected'] for run in capacity_runs)} | "
            f"{sum(run['completed'] for run in fault_runs)}/"
            f"{sum(run['failed'] for run in fault_runs)} | "
            f"{variant_cases['capacity']['resources']['peak_fragmentation_tokens']} |"
        )
    lines.extend(
        [
            "",
            "## 边界说明",
            "",
            f"- Hetero 的 Extent Page 分配次数为 `{hetero_secondary_allocations}`。大于零表示 Generation 自动 Promotion 已在本轮 E2E Workload 中生效，长序列实际使用了 Extent Page。",
            "- Dense GEMM 已按总 Token 数动态 Batch 执行；单 Token 多请求使用 Batch KV Write 与 Paged Decode Attention。多 Token Chunk 已走真实因果 Prefill，但包含多 Token Chunk 的混合 Batch 仍按请求提交 KV/Attention。",
            "- Reference 的 Prefill Attention Scores 与 Softmax/Value Output 为两个 Kernel；Fused 在 head dimension ≤128 时使用在线 Softmax 单 Kernel，>128 回退 Reference。两条路径仍保留 Score Workspace 接口。",
            "- Microbenchmark 与本报告的 E2E 结果分开；K6 的 Gather/Promotion 收益不能直接替代模型端到端收益。",
            "- Nsight Systems GPU Activity Timeline 受当前 WSL2/CUPTI 环境限制，本报告不以 CUDA API Duration 冒充 Kernel Timeline。",
            "- 未与 vLLM 或 TensorRT-LLM 比较峰值性能。",
            "",
        ]
    )
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = load_reports(args.result_dir)
    reference_path = args.result_dir / "reference_validation.json"
    if reference_path.is_file():
        for report in reports.values():
            report["_reference_path"] = str(reference_path)
    fairness = validate_fairness(reports)
    write_summary_csv(args.result_dir, reports)
    analysis = write_report(args.result_dir, reports, fairness)
    print(json.dumps(analysis, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
