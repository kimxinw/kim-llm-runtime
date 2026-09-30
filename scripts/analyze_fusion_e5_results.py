#!/usr/bin/env python3
"""Validate and summarize a two-implementation E5 matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


IMPLEMENTATION_LABELS = {
    "reference": "Reference",
    "fused": "Fused",
    "fused-cursor": "Fused+Cursor",
}
IMPLEMENTATION_SWITCHES = {
    "reference": "Fusion OFF",
    "fused": "Fusion ON / Cursor OFF",
    "fused-cursor": "Fusion ON / Cursor ON",
}
VARIANTS = ("fixed_8", "fixed_16", "fixed_32", "fixed_64", "hetero_8_64")
NON_PERFORMANCE_CASES = {"fault_c4", "capacity"}
CONFIG_KEYS = (
    "warmup",
    "iterations",
    "kv_capacity_tokens",
    "capacity_probe_tokens",
    "performance_storage_bytes",
    "capacity_storage_bytes",
    "model_weight_bytes",
    "model_workspace_bytes",
    "logical_static_gpu_bytes",
    "loaded_gpu_bytes",
)


def column_prefix(implementation: str) -> str:
    return implementation.replace("-", "_")


def load_reports(
    root: Path, implementations: tuple[str, str]
) -> dict[str, dict[str, dict[str, Any]]]:
    reports: dict[str, dict[str, dict[str, Any]]] = {}
    for implementation in implementations:
        implementation_reports: dict[str, dict[str, Any]] = {}
        for variant in VARIANTS:
            path = root / implementation / "variants" / f"{variant}.json"
            if not path.is_file():
                raise RuntimeError(f"missing report: {path}")
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("variant") != variant:
                raise RuntimeError(f"variant mismatch in {path}")
            if not report.get("successful"):
                raise RuntimeError(f"unsuccessful report: {path}")
            implementation_reports[variant] = report
        reports[implementation] = implementation_reports
    return reports


def cases_by_name(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["name"]: case for case in report["cases"]}


def output_signature(case: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "input_tokens": request["input_tokens"],
            "output_tokens": request["output_tokens"],
            "terminal_reason": request["terminal_reason"],
            "error": request["error"],
        }
        for request in case["runs"][0]["requests"]
    ]


def outcome_signature(case: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "accepted": run["accepted"],
            "rejected": run["rejected"],
            "completed": run["completed"],
            "failed": run["failed"],
            "resources_reclaimed": run["resources_reclaimed"],
            "expected_outcome": run["expected_outcome"],
        }
        for run in case["runs"]
    ]


def percent_delta(value: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return (value - baseline) / baseline * 100.0


def validate(
    root: Path,
    reports: dict[str, dict[str, dict[str, Any]]],
    baseline: str,
    candidate: str,
) -> dict[str, Any]:
    reference_validation_path = root / "reference_validation.json"
    if not reference_validation_path.is_file():
        raise RuntimeError("missing Transformers reference validation")
    reference_validation = json.loads(
        reference_validation_path.read_text(encoding="utf-8")
    )
    if not reference_validation.get("passed"):
        raise RuntimeError("Transformers FP16 reference validation failed")

    commits = {
        report["git_commit"]
        for implementation_reports in reports.values()
        for report in implementation_reports.values()
    }
    if len(commits) != 1:
        raise RuntimeError(f"source commit mismatch: {sorted(commits)}")

    measurement_runs = 0
    token_cases = 0
    outcome_cases = 0
    for variant in VARIANTS:
        baseline_report = reports[baseline][variant]
        candidate_report = reports[candidate][variant]
        for key in CONFIG_KEYS:
            if baseline_report["config"][key] != candidate_report["config"][key]:
                raise RuntimeError(f"config mismatch: {variant}/{key}")

        baseline_cases = cases_by_name(baseline_report)
        candidate_cases = cases_by_name(candidate_report)
        if set(baseline_cases) != set(candidate_cases):
            raise RuntimeError(f"case set mismatch: {variant}")

        for name, baseline_case in baseline_cases.items():
            candidate_case = candidate_cases[name]
            measurement_runs += len(baseline_case["runs"])
            measurement_runs += len(candidate_case["runs"])
            if not baseline_case["passed"] or not candidate_case["passed"]:
                raise RuntimeError(f"case failed: {variant}/{name}")
            if not baseline_case["outputs_consistent"] \
                or not candidate_case["outputs_consistent"]:
                raise RuntimeError(f"unstable outputs: {variant}/{name}")
            if name in NON_PERFORMANCE_CASES:
                if outcome_signature(baseline_case) \
                    != outcome_signature(candidate_case):
                    raise RuntimeError(f"outcome mismatch: {variant}/{name}")
                outcome_cases += 1
            else:
                if output_signature(baseline_case) \
                    != output_signature(candidate_case):
                    raise RuntimeError(f"token mismatch: {variant}/{name}")
                token_cases += 1

    return {
        "source_commit": next(iter(commits)),
        "config_equal": True,
        "case_sets_equal": True,
        "cross_implementation_tokens_equal": True,
        "cross_implementation_outcomes_equal": True,
        "transformers_reference_passed": True,
        "transformers_reference_prompts": int(
            reference_validation.get("unique_prompts", 0)
        ),
        "token_cases": token_cases,
        "outcome_cases": outcome_cases,
        "measurement_runs": measurement_runs,
    }


def build_comparisons(
    reports: dict[str, dict[str, dict[str, Any]]],
    baseline: str,
    candidate: str,
) -> list[dict[str, Any]]:
    base = column_prefix(baseline)
    cand = column_prefix(candidate)
    comparisons: list[dict[str, Any]] = []
    for variant in VARIANTS:
        baseline_cases = cases_by_name(reports[baseline][variant])
        candidate_cases = cases_by_name(reports[candidate][variant])
        for name, baseline_case in baseline_cases.items():
            if name in NON_PERFORMANCE_CASES:
                continue
            baseline_summary = baseline_case["summary"]
            candidate_summary = candidate_cases[name]["summary"]
            comparisons.append(
                {
                    "variant": variant,
                    "case": name,
                    "concurrency": baseline_case["concurrency"],
                    f"{base}_e2e_p50_ms": baseline_summary["e2e_p50_ns"] / 1e6,
                    f"{cand}_e2e_p50_ms": candidate_summary["e2e_p50_ns"] / 1e6,
                    "e2e_p50_delta_percent": percent_delta(
                        candidate_summary["e2e_p50_ns"],
                        baseline_summary["e2e_p50_ns"],
                    ),
                    f"{base}_tpot_p50_ms": baseline_summary["tpot_p50_ns"] / 1e6,
                    f"{cand}_tpot_p50_ms": candidate_summary["tpot_p50_ns"] / 1e6,
                    "tpot_p50_delta_percent": percent_delta(
                        candidate_summary["tpot_p50_ns"],
                        baseline_summary["tpot_p50_ns"],
                    ),
                    f"{base}_output_tokens_per_second": baseline_summary[
                        "output_tokens_per_second"
                    ],
                    f"{cand}_output_tokens_per_second": candidate_summary[
                        "output_tokens_per_second"
                    ],
                    "output_tokens_per_second_delta_percent": percent_delta(
                        candidate_summary["output_tokens_per_second"],
                        baseline_summary["output_tokens_per_second"],
                    ),
                }
            )
    return comparisons


def write_csv(root: Path, comparisons: list[dict[str, Any]]) -> None:
    with (root / "fusion_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(
            output, fieldnames=list(comparisons[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(comparisons)


def write_report(
    root: Path,
    validation: dict[str, Any],
    comparisons: list[dict[str, Any]],
    baseline: str,
    candidate: str,
) -> None:
    base = column_prefix(baseline)
    cand = column_prefix(candidate)
    base_label = IMPLEMENTATION_LABELS[baseline]
    cand_label = IMPLEMENTATION_LABELS[candidate]
    lines = [
        f"# {base_label}/{cand_label} TinyLlama E5 正式矩阵",
        "",
        "## 验证结论",
        "",
        "| 项目 | 结果 |",
        "|---|---|",
        "| 两套实现绑定同一提交 | PASS |",
        "| 配置、模型与 Storage Budget 一致 | PASS |",
        "| Case 集合一致 | PASS |",
        f"| 跨实现 Token 一致 | PASS（{validation['token_cases']} 个策略/Case） |",
        f"| 故障与容量结果一致 | PASS（{validation['outcome_cases']} 个策略/Case） |",
        f"| Transformers FP16 Reference | PASS（{validation['transformers_reference_prompts']} 个唯一 Prompt） |",
        f"| 测量 Run 总数 | {validation['measurement_runs']} |",
        "",
        f"## {cand_label} 相对 {base_label}",
        "",
        f"延迟差值为负表示 {cand_label} 更快；吞吐差值为正表示 {cand_label} 更快。",
        "",
        f"| Page 策略 | Workload | E2E p50（{base_label} → {cand_label}） | E2E 差值 | TPOT 差值 | Output tokens/s 差值 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['variant']} | {row['case']} | "
            f"{row[f'{base}_e2e_p50_ms']:.3f} → {row[f'{cand}_e2e_p50_ms']:.3f} ms | "
            f"{row['e2e_p50_delta_percent']:+.2f}% | "
            f"{row['tpot_p50_delta_percent']:+.2f}% | "
            f"{row['output_tokens_per_second_delta_percent']:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            f"- {base_label}（{IMPLEMENTATION_SWITCHES[baseline]}）与 {cand_label}（{IMPLEMENTATION_SWITCHES[candidate]}）仅编译开关不同；模型、请求、分页策略、容量、Warmup、迭代数和计时边界保持一致。",
            "- 每个实现包含五种分页策略和九类 Case；性能比较排除故障注入与容量 Case。",
            "- 本结果仅适用于记录的 TinyLlama FP16、RTX 3060 和当前 CUDA/Driver 环境，不能外推到其他模型或硬件。",
            "- 实现间收益与 Heterogeneous/Fixed Page 策略收益分别统计，不能相互替代。",
            "",
        ]
    )
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline", choices=sorted(IMPLEMENTATION_LABELS), default="reference"
    )
    parser.add_argument(
        "--candidate", choices=sorted(IMPLEMENTATION_LABELS), default="fused"
    )
    args = parser.parse_args()
    if args.baseline == args.candidate:
        parser.error("--baseline and --candidate must differ")
    return args


def main() -> None:
    args = parse_args()
    implementations = (args.baseline, args.candidate)
    reports = load_reports(args.result_dir, implementations)
    validation = validate(
        args.result_dir, reports, args.baseline, args.candidate
    )
    comparisons = build_comparisons(reports, args.baseline, args.candidate)
    write_csv(args.result_dir, comparisons)
    result = {
        "schema_version": 1,
        "successful": True,
        "validation": validation,
        "comparisons": comparisons,
    }
    (args.result_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(
        args.result_dir, validation, comparisons, args.baseline, args.candidate
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
