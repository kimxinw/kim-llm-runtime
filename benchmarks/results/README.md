# Benchmark result policy

This directory stores formal, commit-bound Release benchmark evidence. Debug
and smoke outputs belong in `/tmp` and must not be presented as performance
results.

`scripts/run_release_matrix.sh` creates one directory per run:

```text
benchmarks/results/<source-commit>_<UTC timestamp>/
├── MANIFEST.txt
├── SHA256SUMS
├── summary.csv
├── capacity_comparison.csv
├── cpu/<workload>/
│   ├── cpu_metadata.json.gz
│   ├── cpu_metadata.csv.gz
│   └── run.log
└── cuda/<workload>/
    ├── cuda_data_path.json.gz
    ├── cuda_data_path.csv.gz
    └── run.log
```

A formal run must satisfy all of the following:

1. The source worktree is clean before execution.
2. Every report records the exact full source commit.
3. Release contract tests pass before benchmarks start.
4. Every workload reports `successful: true`, zero unexpected failures,
   valid invariants, and complete resource release.
5. Raw JSON/CSV files are compressed without modification after generation.
6. `SHA256SUMS` matches every artifact in the run directory.

The consolidated `summary.csv` contains only capacity and latency-summary
records. `capacity_comparison.csv` contains the derived Fixed-8/Fixed-64 versus
Hetero-8/64 percentages used by the report. Per-operation raw samples remain
in the per-workload JSON/CSV files.

K6 uses `scripts/run_k6_release_matrix.sh` and stores an explicit runtime
variant level below both `cpu/` and `cuda/`:

```text
benchmarks/results/<source-commit>_<UTC timestamp>_k6/
├── MANIFEST.txt
├── SHA256SUMS
├── REPORT.md
├── summary.csv
├── runtime_comparison.csv
├── promotion_break_even.csv
├── promotion_break_even_curve.csv
├── cpu/<hetero|fixed_8|fixed_16|fixed_32|fixed_64>/<workload>/
└── cuda/<hetero|fixed_8|fixed_16|fixed_32|fixed_64>/<workload>/
```

The break-even curve is an analytical repeat model built from measured CUDA
Event primitives. A row with `no_promotion_samples` is intentionally not
assigned a break-even point. Failed profiler attempts may be retained with
their diagnostics, but they cannot be presented as a completed Nsight case
study.

K6 profiler runs use `scripts/run_k6_profile.sh`. Set
`KIM_KV_PROFILE_COMPONENT` to `all`, `ncu`, or `nsys`; the latter two allow a
supported profiler to finish without misrepresenting the other profiler's
environmental failure:

```text
benchmarks/results/<source-commit>_<UTC timestamp>_k6_profile/
├── MANIFEST.txt
├── SHA256SUMS
├── NCU_SUMMARY.md                 # added after reviewed NCU analysis
├── NCU_SUMMARY.csv                # added after reviewed NCU analysis
├── hetero/ncu/<kernel>/
│   ├── kernels.ncu-rep
│   ├── metrics.csv
│   └── run/
└── fixed_8/ncu/<kernel>/
```

Nsight Compute collection is filtered by kernel name so that the launch limit
cannot be consumed entirely by an earlier kernel. Nsight Systems collection is
valid only when both `cuda_gpu_kern_sum` and `cuda_api_sum` contain data. The
script treats a missing or header-only report as an error even when `nsys`
itself exits successfully.

E5 adds a separate real-model end-to-end matrix. It must not be mixed with K6
Data Path samples:

```text
benchmarks/results/<source>_<UTC timestamp>_e5/
├── MANIFEST.txt
├── SOURCE_SHA256SUMS             # present for pre-commit pinned runs
├── SHA256SUMS
├── REPORT.md
├── comparison.json
├── reference_validation.json
├── summary.csv
└── variants/
    ├── fixed_8.json
    ├── fixed_16.json
    ├── fixed_32.json
    ├── fixed_64.json
    └── hetero_8_64.json
```

`scripts/run_e5_end_to_end.sh` runs Release contracts, three or more measured
rounds, the five page strategies, a Transformers FP16 reference, capacity and
fault workloads, analysis, and checksums. A clean committed tree is required
for a formal run. `KIM_KV_ALLOW_DIRTY=1` is development-only; such evidence
must say `precommit_source_hash_pinned` and record exact changed-source hashes.

Two-implementation E5 A/B matrices use `scripts/run_fusion_e5_matrix.sh`.
`KIM_KV_FUSION_E5_IMPLEMENTATIONS` names the baseline and candidate
(`reference fused` by default, or `fused-no-cursor fused`). Since the
Descriptor Cursor became the Fused default, `fused` means Fusion ON / Cursor ON
and `fused-no-cursor` is the Cursor-off baseline; archives made before that
change used `fused` for Cursor OFF and `fused-cursor` for Cursor ON.
`KIM_KV_E5_SUITE` selects the `standard` nine-case suite (default) or the
`long` 512/1024-token suite, which has no fault or capacity cases and skips the
per-implementation page-strategy analysis. The directory suffix is `fusion_e5`
for the default pair and suite, and `<baseline>_<candidate>[_long]_e5`
otherwise:

```text
benchmarks/results/<source-commit>_<UTC timestamp>_<suffix>/
├── MANIFEST.txt
├── SHA256SUMS
├── REPORT.md
├── comparison.json
├── fusion_comparison.csv
├── analysis.log
├── reference_validation.{json,log}
└── <baseline|candidate>/
    ├── REPORT.md                  # standard suite only
    ├── analysis.log               # standard suite only
    ├── comparison.json            # standard suite only
    ├── summary.csv                # standard suite only
    ├── reference_validation.json
    └── variants/<fixed_8|fixed_16|fixed_32|fixed_64|hetero_8_64>.{json,log}
```

Both builds come from the same clean commit and differ only in their
compile-time switches. A single matrix run has about 5% run-to-run E2E p50
variation for identical code, so a smaller E2E delta is not a resolved effect.
`SHA256SUMS` uses relative paths; archives made before that change list
absolute paths from the worktree that produced them.

Compute Sanitizer evidence uses `scripts/run_compute_sanitizer_matrix.sh` and
`KIM_KV_SANITIZER_VARIANTS` (`reference`, `fused`, `fused-no-cursor`). The script
checks cleanliness after writing its output, so formal runs set
`KIM_KV_RESULTS_ROOT` outside the worktree and copy the directory here
afterwards; its `SHA256SUMS` uses relative paths:

```text
benchmarks/results/<source-commit>_<UTC timestamp>_compute_sanitizer/
├── MANIFEST.txt
├── SHA256SUMS
├── summary.tsv
└── logs/<variant>_<tool>_<test>.log
```
