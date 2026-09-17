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
readonly result_directory="${KIM_KV_SANITIZER_RESULT_DIR:-${results_root}/${short_commit}_${timestamp_utc}_compute_sanitizer}"
readonly cuda_device="${KIM_KV_CUDA_DEVICE:-0}"
readonly build_jobs="${KIM_KV_BUILD_JOBS:-4}"

read -r -a variants <<< "${KIM_KV_SANITIZER_VARIANTS:-reference fused}"
read -r -a sanitizer_tools <<< "${KIM_KV_SANITIZER_TOOLS:-memcheck racecheck initcheck}"
read -r -a test_binaries <<< "${KIM_KV_SANITIZER_TESTS:-paged_attention_fusion_test cuda_kv_test cuda_engine_kv_backend_test cuda_model_runner_test}"

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

compute_sanitizer="${KIM_KV_COMPUTE_SANITIZER:-}"
if [[ -z "${compute_sanitizer}" ]] && command -v compute-sanitizer >/dev/null 2>&1; then
    compute_sanitizer="$(command -v compute-sanitizer)"
fi
if [[ -z "${compute_sanitizer}" ]] \
    && [[ -x "${cuda_root}/bin/compute-sanitizer" ]]; then
    compute_sanitizer="${cuda_root}/bin/compute-sanitizer"
fi
if [[ -z "${compute_sanitizer}" ]] || [[ ! -x "${compute_sanitizer}" ]]; then
    echo "error: compute-sanitizer not found; set KIM_KV_COMPUTE_SANITIZER" >&2
    exit 2
fi
readonly compute_sanitizer

build_directory_for_variant()
{
    local variant="$1"
    case "${variant}" in
        reference)
            echo "build-k5-cuda-reference"
            ;;
        fused)
            echo "build-k5-cuda-fused"
            ;;
        *)
            echo "error: unsupported sanitizer variant: ${variant}" >&2
            return 2
            ;;
    esac
}

preset_for_variant()
{
    local variant="$1"
    echo "cuda-release-${variant}"
}

tool_arguments()
{
    local tool="$1"
    case "${tool}" in
        memcheck)
            echo "--leak-check full --report-api-errors explicit"
            ;;
        racecheck)
            echo "--racecheck-detect-level warn --racecheck-report all"
            ;;
        initcheck)
            echo "--check-api-memory-access yes"
            ;;
        *)
            echo "error: unsupported sanitizer tool: ${tool}" >&2
            return 2
            ;;
    esac
}

validate_tool_summary()
{
    local tool="$1"
    local log_path="$2"
    case "${tool}" in
        racecheck)
            grep -Fq \
                "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)" \
                "${log_path}"
            ;;
        memcheck|initcheck)
            grep -Fq "ERROR SUMMARY: 0 errors" "${log_path}"
            ;;
    esac
}

mkdir -p "${result_directory}/logs"
readonly summary_path="${result_directory}/summary.tsv"
printf "variant\ttool\ttest\tstatus\texit_code\tduration_seconds\tlog\n" \
    > "${summary_path}"

if [[ "${KIM_KV_SANITIZER_SKIP_BUILD:-0}" != 1 ]]; then
    for variant in "${variants[@]}"; do
        preset="$(preset_for_variant "${variant}")"
        echo "Configuring and building ${preset}"
        cmake --preset "${preset}" \
            "-DCMAKE_CUDA_COMPILER=${cuda_compiler}" \
            "-DCUDAToolkit_ROOT=${cuda_root}"
        cmake --build --preset "${preset}" --parallel "${build_jobs}"
    done
fi

for variant in "${variants[@]}"; do
    build_directory="$(build_directory_for_variant "${variant}")"
    expected_fusion="OFF"
    if [[ "${variant}" == fused ]]; then
        expected_fusion="ON"
    fi
    if ! grep -Fq \
        "KIM_KV_ENABLE_FUSED_ATTENTION:BOOL=${expected_fusion}" \
        "${build_directory}/CMakeCache.txt"; then
        echo "error: ${variant} build does not have Fusion=${expected_fusion}" >&2
        exit 2
    fi

    for tool in "${sanitizer_tools[@]}"; do
        read -r -a specific_arguments <<< "$(tool_arguments "${tool}")"
        for test_binary in "${test_binaries[@]}"; do
            executable="${project_root}/${build_directory}/tests/${test_binary}"
            if [[ ! -x "${executable}" ]]; then
                echo "error: test binary not found: ${executable}" >&2
                exit 2
            fi

            relative_log="logs/${variant}_${tool}_${test_binary}.log"
            log_path="${result_directory}/${relative_log}"
            echo "Running ${variant}/${tool}/${test_binary}"
            started_at="$(date +%s)"
            set +e
            CUDA_VISIBLE_DEVICES="${cuda_device}" \
                "${compute_sanitizer}" \
                    --tool "${tool}" \
                    --error-exitcode 86 \
                    --check-exit-code yes \
                    --target-processes application-only \
                    --print-limit 0 \
                    "${specific_arguments[@]}" \
                    "${executable}" \
                    >"${log_path}" 2>&1
            exit_code=$?
            set -e
            finished_at="$(date +%s)"
            duration_seconds="$((finished_at - started_at))"

            status="PASS"
            if [[ "${exit_code}" -ne 0 ]] \
                || ! validate_tool_summary "${tool}" "${log_path}"; then
                status="FAIL"
            fi
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "${variant}" "${tool}" "${test_binary}" "${status}" \
                "${exit_code}" "${duration_seconds}" "${relative_log}" \
                >> "${summary_path}"
            echo "${status}: ${variant}/${tool}/${test_binary} (${duration_seconds}s)"
        done
    done
done

working_tree_clean=true
if [[ -n "$(git status --porcelain=v1 --untracked-files=normal)" ]]; then
    working_tree_clean=false
fi

{
    echo "schema_version=1"
    echo "stage=reference_fused_compute_sanitizer"
    echo "source_commit=${source_commit}"
    echo "timestamp_utc=${timestamp_utc}"
    echo "working_tree_clean=${working_tree_clean}"
    echo "variants=${variants[*]}"
    echo "tools=${sanitizer_tools[*]}"
    echo "tests=${test_binaries[*]}"
    echo "cuda_visible_device=${cuda_device}"
    echo "cuda_root=${cuda_root}"
    echo "cuda_compiler=${cuda_compiler}"
    echo "compute_sanitizer=${compute_sanitizer}"
    echo "compute_sanitizer_version=$(
        "${compute_sanitizer}" --version | tail -n 1
    )"
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
} > "${result_directory}/MANIFEST.txt"

(
    cd "${result_directory}"
    find . -type f ! -name SHA256SUMS -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 sha256sum > SHA256SUMS
)

failed_runs="$(awk -F '\t' 'NR > 1 && $4 != "PASS" { count++ } END { print count + 0 }' "${summary_path}")"
if [[ "${failed_runs}" -ne 0 ]]; then
    echo "Compute Sanitizer matrix failed: ${failed_runs} run(s) failed" >&2
    echo "Results: ${result_directory}" >&2
    exit 1
fi

echo "Compute Sanitizer matrix passed"
echo "Results: ${result_directory}"
