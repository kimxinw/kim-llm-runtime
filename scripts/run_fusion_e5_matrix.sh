#!/usr/bin/env bash

set -euo pipefail

readonly script_directory="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
    pwd
)"
readonly project_root="$(cd -- "${script_directory}/.." && pwd)"
cd "${project_root}"

readonly source_commit="$(git rev-parse --verify HEAD)"
readonly short_commit="${source_commit:0:12}"
readonly timestamp_utc="$(date -u +%Y%m%dT%H%M%SZ)"
readonly results_root="${KIM_KV_RESULTS_ROOT:-${project_root}/benchmarks/results}"
readonly result_directory="${KIM_KV_FUSION_E5_RESULT_DIR:-${results_root}/${short_commit}_${timestamp_utc}_fusion_e5}"
readonly manifest="${KIM_KV_MODEL_MANIFEST:-/home/xinwang/workspaces/kim-kvcache-e2-model/tinyllama-1.1b-chat-fp16.manifest}"
readonly weights="${KIM_KV_MODEL_WEIGHTS:-/home/xinwang/workspaces/kim-kvcache-e2-model/tinyllama-1.1b-chat-fp16.weights}"
readonly reference_python="${KIM_KV_REFERENCE_PYTHON:-/home/xinwang/miniconda3/envs/vllm/bin/python}"
readonly warmup="${KIM_KV_E5_WARMUP:-1}"
readonly iterations="${KIM_KV_E5_ITERATIONS:-3}"
readonly kv_capacity_tokens="${KIM_KV_E5_KV_CAPACITY_TOKENS:-8192}"
readonly capacity_probe_tokens="${KIM_KV_E5_CAPACITY_PROBE_TOKENS:-512}"
readonly build_jobs="${KIM_KV_BUILD_JOBS:-4}"
readonly cuda_device="${KIM_KV_CUDA_DEVICE:-0}"
readonly -a implementations=(reference fused)
readonly -a variants=(fixed_8 fixed_16 fixed_32 fixed_64 hetero)

if [[ ! -f "${manifest}" ]] || [[ ! -f "${weights}" ]]; then
    echo "error: set KIM_KV_MODEL_MANIFEST and KIM_KV_MODEL_WEIGHTS" >&2
    exit 2
fi
if [[ ! -x "${reference_python}" ]]; then
    echo "error: set KIM_KV_REFERENCE_PYTHON to Python with torch/transformers" >&2
    exit 2
fi
if [[ -n "$(git status --porcelain=v1 --untracked-files=normal)" ]]; then
    echo "error: formal Fusion E5 evidence requires a clean committed worktree" >&2
    git status --short >&2
    exit 2
fi
if [[ -e "${result_directory}" ]]; then
    echo "error: result directory already exists: ${result_directory}" >&2
    exit 2
fi

cuda_compiler="${CUDACXX:-}"
if [[ -z "${cuda_compiler}" ]] && command -v nvcc >/dev/null 2>&1; then
    cuda_compiler="$(command -v nvcc)"
fi
if [[ -z "${cuda_compiler}" ]]; then
    for cache_path in \
        build-k5-cuda-reference/CMakeCache.txt \
        build-k5-cuda-fused/CMakeCache.txt \
        build-k5-cuda-release/CMakeCache.txt; do
        if [[ -f "${cache_path}" ]]; then
            cuda_compiler="$(
                sed -n 's/^CMAKE_CUDA_COMPILER:[^=]*=//p' \
                    "${cache_path}" | head -n 1
            )"
        fi
        if [[ -n "${cuda_compiler}" ]]; then
            break
        fi
    done
fi
if [[ -z "${cuda_compiler}" ]] || [[ ! -x "${cuda_compiler}" ]]; then
    echo "error: CUDA compiler not found; set CUDACXX" >&2
    exit 2
fi
readonly cuda_compiler
readonly cuda_root="$(cd -- "$(dirname -- "${cuda_compiler}")/.." && pwd)"

preset_for_implementation()
{
    echo "cuda-release-$1"
}

build_directory_for_implementation()
{
    echo "build-k5-cuda-$1"
}

expected_fusion_for_implementation()
{
    if [[ "$1" == fused ]]; then
        echo ON
    else
        echo OFF
    fi
}

report_variant_name()
{
    if [[ "$1" == hetero ]]; then
        echo hetero_8_64
    else
        echo "$1"
    fi
}

mkdir -p "${result_directory}"

echo "Configuring, building, and testing CPU Release"
cmake --preset cpu-release
cmake --build --preset cpu-release --parallel "${build_jobs}"
ctest --preset cpu-release

for implementation in "${implementations[@]}"; do
    preset="$(preset_for_implementation "${implementation}")"
    build_directory="$(build_directory_for_implementation "${implementation}")"
    expected_fusion="$(expected_fusion_for_implementation "${implementation}")"

    echo "Configuring, building, and testing ${preset}"
    cmake --preset "${preset}" \
        "-DCMAKE_CUDA_COMPILER=${cuda_compiler}" \
        "-DCUDAToolkit_ROOT=${cuda_root}"
    cmake --build --preset "${preset}" --parallel "${build_jobs}"
    CUDA_VISIBLE_DEVICES="${cuda_device}" ctest --preset "${preset}"

    if ! grep -Fq \
        "KIM_KV_ENABLE_FUSED_ATTENTION:BOOL=${expected_fusion}" \
        "${build_directory}/CMakeCache.txt"; then
        echo "error: ${implementation} does not have Fusion=${expected_fusion}" >&2
        exit 2
    fi

    implementation_directory="${result_directory}/${implementation}"
    mkdir -p "${implementation_directory}/variants"
    benchmark="${project_root}/${build_directory}/tools/kim_kv_tinyllama_e2e_benchmark"
    if [[ ! -x "${benchmark}" ]]; then
        echo "error: benchmark not found: ${benchmark}" >&2
        exit 2
    fi

    for variant in "${variants[@]}"; do
        report_variant="$(report_variant_name "${variant}")"
        echo "Running Fusion E5 ${implementation}/${report_variant}"
        CUDA_VISIBLE_DEVICES="${cuda_device}" \
            "${benchmark}" \
                --manifest "${manifest}" \
                --weights "${weights}" \
                --variant "${variant}" \
                --output "${implementation_directory}/variants/${report_variant}.json" \
                --warmup "${warmup}" \
                --iterations "${iterations}" \
                --kv-capacity-tokens "${kv_capacity_tokens}" \
                --capacity-probe-tokens "${capacity_probe_tokens}" \
                --git-commit "${source_commit}" \
                >"${implementation_directory}/variants/${report_variant}.log" 2>&1
    done
done

CUDA_VISIBLE_DEVICES="${cuda_device}" \
    "${reference_python}" scripts/validate_e5_reference.py \
        --manifest "${manifest}" \
        --weights "${weights}" \
        --runtime-json "${result_directory}/reference/variants/fixed_8.json" \
        --output "${result_directory}/reference_validation.json" \
        >"${result_directory}/reference_validation.log"

for implementation in "${implementations[@]}"; do
    cp "${result_directory}/reference_validation.json" \
        "${result_directory}/${implementation}/reference_validation.json"
    python3 scripts/analyze_e5_results.py \
        --result-dir "${result_directory}/${implementation}" \
        >"${result_directory}/${implementation}/analysis.log"
done

python3 scripts/analyze_fusion_e5_results.py \
    --result-dir "${result_directory}" \
    >"${result_directory}/analysis.log"

{
    echo "schema_version=1"
    echo "stage=reference_fused_e5"
    echo "source_commit=${source_commit}"
    echo "working_tree_clean=true"
    echo "timestamp_utc=${timestamp_utc}"
    echo "gpu_device=${cuda_device}"
    echo "warmup=${warmup}"
    echo "iterations=${iterations}"
    echo "kv_capacity_tokens=${kv_capacity_tokens}"
    echo "capacity_probe_tokens=${capacity_probe_tokens}"
    echo "implementations=${implementations[*]}"
    echo "variants=${variants[*]}"
    echo "model_manifest=${manifest}"
    echo "model_weights=${weights}"
    echo "cuda_root=${cuda_root}"
    echo "cuda_compiler=${cuda_compiler}"
    if [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
        echo "gpu=$(
            /usr/lib/wsl/lib/nvidia-smi \
                --query-gpu=name,driver_version,memory.total,compute_cap \
                --format=csv,noheader | sed -n "$((cuda_device + 1))p"
        )"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        echo "gpu=$(
            nvidia-smi \
                --query-gpu=name,driver_version,memory.total,compute_cap \
                --format=csv,noheader | sed -n "$((cuda_device + 1))p"
        )"
    fi
    sha256sum "${manifest}" "${weights}"
    for implementation in "${implementations[@]}"; do
        build_directory="$(build_directory_for_implementation "${implementation}")"
        sha256sum \
            "${build_directory}/tools/kim_kv_tinyllama_e2e_benchmark"
    done
} >"${result_directory}/MANIFEST.txt"

find "${result_directory}" -type f ! -name SHA256SUMS -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum >"${result_directory}/SHA256SUMS"

echo "Fusion E5 evidence: ${result_directory}"
