#!/usr/bin/env python3
"""Validate E5 Fixed-8 outputs against independent Transformers FP16."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import transformers

from validate_tinyllama_generation import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--runtime-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = json.loads(args.runtime_json.read_text(encoding="utf-8"))
    unique: dict[tuple[int, ...], list[int]] = {}
    covered_cases: list[str] = []
    for case in runtime["cases"]:
        if case["fault_workload"] or case["capacity_workload"]:
            continue
        covered_cases.append(case["name"])
        for request in case["runs"][0]["requests"]:
            prompt = tuple(int(token) for token in request["input_tokens"])
            output = [int(token) for token in request["output_tokens"]]
            previous = unique.setdefault(prompt, output)
            if previous != output:
                raise RuntimeError("runtime outputs disagree for identical prompt")

    grouped: dict[int, list[tuple[tuple[int, ...], list[int]]]] = defaultdict(list)
    for prompt, output in unique.items():
        grouped[len(prompt)].append((prompt, output))

    model = load_model(args.manifest, args.weights)
    eos_token_id = int(model.config.eos_token_id)
    comparisons: list[dict[str, object]] = []
    with torch.inference_mode():
        for prompt_length, entries in sorted(grouped.items()):
            input_ids = torch.tensor(
                [list(prompt) for prompt, _ in entries],
                dtype=torch.long,
                device="cuda",
            )
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=32,
                do_sample=False,
                eos_token_id=eos_token_id,
                pad_token_id=eos_token_id,
                use_cache=True,
            )
            for index, (prompt, expected) in enumerate(entries):
                actual = generated[index, prompt_length:].cpu().tolist()
                comparisons.append(
                    {
                        "prompt_length": prompt_length,
                        "prompt_tokens": list(prompt),
                        "runtime_tokens": expected,
                        "reference_tokens": actual,
                        "tokens_equal": expected == actual,
                    }
                )

    report = {
        "schema_version": 1,
        "checkpoint": runtime["checkpoint"],
        "checkpoint_revision": runtime["checkpoint_revision"],
        "covered_cases": covered_cases,
        "unique_prompts": len(comparisons),
        "comparisons": comparisons,
        "all_tokens_equal": all(item["tokens_equal"] for item in comparisons),
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "reference_dtype": "float16",
            "attention_implementation": "eager",
            "reference_prefill": "batched_by_prompt_length",
            "runtime_prefill": "multi_token_causal_chunks",
        },
    }
    report["passed"] = report["all_tokens_equal"] and len(comparisons) > 0
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
