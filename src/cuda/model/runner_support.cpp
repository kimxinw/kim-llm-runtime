#include "runner_internal.cuh"

#include <limits>

namespace kimkvcache::cuda_model_runner_detail {
namespace {

constexpr std::size_t kAlignment = 256;

[[nodiscard]] std::size_t aligned(std::size_t value) noexcept
{
    return (value + kAlignment - 1) & ~(kAlignment - 1);
}

[[nodiscard]] bool addRegion(
    std::size_t bytes,
    std::size_t& cursor,
    std::size_t& offset) noexcept
{
    cursor = aligned(cursor);
    offset = cursor;
    if (bytes > std::numeric_limits<std::size_t>::max() - cursor) {
        return false;
    }
    cursor += bytes;
    return true;
}

[[nodiscard]] bool multiply(
    std::size_t value,
    std::uint32_t factor,
    std::size_t& result) noexcept
{
    if (factor == 0
        || value > std::numeric_limits<std::size_t>::max() / factor) {
        return false;
    }
    result = value * factor;
    return true;
}

} // namespace

CudaModelRunnerStatus failure(
    CudaModelRunnerError error,
    int native_code,
    std::string detail)
{
    return CudaModelRunnerStatus{error, native_code, std::move(detail)};
}

CudaModelRunnerStatus cudaFailure(cudaError_t error, std::string detail)
{
    CudaModelRunnerError const kind = error == cudaErrorMemoryAllocation
        ? CudaModelRunnerError::AllocationFailed
        : CudaModelRunnerError::SubmissionFailed;
    return failure(kind, static_cast<int>(error), std::move(detail));
}

CudaModelRunnerStatus cublasFailure(
    cublasStatus_t status,
    std::string detail)
{
    return failure(
        CudaModelRunnerError::CublasFailed,
        static_cast<int>(status),
        std::move(detail)
    );
}

CudaModelRunnerStatus kvFailure(
    EngineKvStatus status,
    std::string detail)
{
    CudaModelRunnerError kind = CudaModelRunnerError::KvBackendFailed;
    if (status.error == EngineKvError::SubmissionFailed) {
        kind = CudaModelRunnerError::SubmissionFailed;
    } else if (status.error == EngineKvError::ExecutionFailed) {
        kind = CudaModelRunnerError::ExecutionFailed;
    }
    detail += ": ";
    detail += toString(status.error);
    return failure(kind, static_cast<int>(status.error), std::move(detail));
}

bool makeWorkspaceLayout(
    TinyLlamaConfig const& config,
    std::uint32_t max_batch_size,
    WorkspaceLayout& layout) noexcept
{
    std::size_t cursor = 0;
    std::size_t hidden_bytes =
        static_cast<std::size_t>(config.hidden_size) * sizeof(KvScalar);
    std::size_t kv_bytes = static_cast<std::size_t>(
        config.kv_head_count * config.head_dimension
    ) * sizeof(KvScalar);
    std::size_t intermediate_bytes =
        static_cast<std::size_t>(config.intermediate_size)
        * sizeof(KvScalar);
    std::size_t logits_bytes =
        static_cast<std::size_t>(config.vocabulary_size) * sizeof(float);
    std::size_t score_bytes =
        static_cast<std::size_t>(config.attention_head_count)
        * config.max_position_embeddings * sizeof(float);
    std::size_t token_id_bytes = sizeof(std::uint32_t);
    std::size_t kv_write_batch_item_bytes =
        sizeof(DeviceLayerKvWriteBatchItem);
    std::size_t attention_batch_item_bytes =
        sizeof(DevicePagedDecodeBatchItem);
    bool const scaled = multiply(hidden_bytes, max_batch_size, hidden_bytes)
        && multiply(kv_bytes, max_batch_size, kv_bytes)
        && multiply(intermediate_bytes, max_batch_size, intermediate_bytes)
        && multiply(logits_bytes, max_batch_size, logits_bytes)
        && multiply(score_bytes, max_batch_size, score_bytes)
        && multiply(token_id_bytes, max_batch_size, token_id_bytes)
        && multiply(
            kv_write_batch_item_bytes,
            max_batch_size,
            kv_write_batch_item_bytes
        )
        && multiply(
            attention_batch_item_bytes,
            max_batch_size,
            attention_batch_item_bytes
        );
    bool const valid = scaled
        && addRegion(token_id_bytes, cursor, layout.token_ids)
        && addRegion(token_id_bytes, cursor, layout.positions)
        && addRegion(hidden_bytes, cursor, layout.hidden)
        && addRegion(hidden_bytes, cursor, layout.normalized)
        && addRegion(hidden_bytes, cursor, layout.query)
        && addRegion(kv_bytes, cursor, layout.key)
        && addRegion(kv_bytes, cursor, layout.value)
        && addRegion(hidden_bytes, cursor, layout.attention)
        && addRegion(hidden_bytes, cursor, layout.projection)
        && addRegion(intermediate_bytes, cursor, layout.gate)
        && addRegion(intermediate_bytes, cursor, layout.up)
        && addRegion(intermediate_bytes, cursor, layout.activated)
        && addRegion(logits_bytes, cursor, layout.logits)
        && addRegion(token_id_bytes, cursor, layout.greedy_token)
        && addRegion(score_bytes, cursor, layout.attention_scores)
        && addRegion(
            kv_write_batch_item_bytes,
            cursor,
            layout.kv_write_batch_items
        )
        && addRegion(
            attention_batch_item_bytes,
            cursor,
            layout.attention_batch_items
        );
    if (valid) {
        layout.bytes = aligned(cursor);
    }
    return valid && layout.bytes != 0;
}

} // namespace kimkvcache::cuda_model_runner_detail

namespace kimkvcache {

CudaTinyLlamaModelRunner::Impl::~Impl()
{
    if (stream != nullptr) {
        static_cast<void>(cudaStreamSynchronize(stream));
    }
    if (device_workspace != nullptr) {
        static_cast<void>(cudaFree(device_workspace));
    }
    if (device_weights != nullptr) {
        static_cast<void>(cudaFree(device_weights));
    }
    if (cublas != nullptr) {
        static_cast<void>(cublasDestroy(cublas));
    }
}

KvScalar const* CudaTinyLlamaModelRunner::Impl::weight(
    std::string const& name) const noexcept
{
    WeightTensorDescriptor const* tensor = manifest.find(name);
    return tensor == nullptr ? nullptr
        : reinterpret_cast<KvScalar const*>(device_weights + tensor->offset);
}

CudaModelRunnerStatus CudaTinyLlamaModelRunner::Impl::checkLaunch(
    std::string const& operation) const
{
    cudaError_t const error = cudaGetLastError();
    return error == cudaSuccess
        ? CudaModelRunnerStatus{}
        : cuda_model_runner_detail::cudaFailure(error, operation);
}

CudaModelRunnerStatus CudaTinyLlamaModelRunner::Impl::gemv(
    KvScalar const* matrix,
    KvScalar const* input,
    KvScalar* output,
    std::uint32_t rows,
    std::uint32_t columns,
    std::string const& operation) const
{
    return gemmBatch(
        matrix, input, output, rows, columns, 1, operation
    );
}

CudaModelRunnerStatus CudaTinyLlamaModelRunner::Impl::gemmBatch(
    KvScalar const* matrix,
    KvScalar const* input,
    KvScalar* output,
    std::uint32_t rows,
    std::uint32_t columns,
    std::uint32_t batch_size,
    std::string const& operation) const
{
    float const alpha = 1.0F;
    float const beta = 0.0F;
    cublasStatus_t const status = cublasGemmEx(
        cublas,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(rows),
        static_cast<int>(batch_size),
        static_cast<int>(columns),
        &alpha,
        matrix,
        CUDA_R_16F,
        static_cast<int>(columns),
        input,
        CUDA_R_16F,
        static_cast<int>(columns),
        &beta,
        output,
        CUDA_R_16F,
        static_cast<int>(rows),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    return status == CUBLAS_STATUS_SUCCESS
        ? CudaModelRunnerStatus{}
        : cuda_model_runner_detail::cublasFailure(status, operation);
}

CudaModelRunnerStatus CudaTinyLlamaModelRunner::Impl::logitsGemv(
    KvScalar const* matrix,
    KvScalar const* input,
    float* output) const
{
    return logitsGemmBatch(matrix, input, output, 1);
}

CudaModelRunnerStatus CudaTinyLlamaModelRunner::Impl::logitsGemmBatch(
    KvScalar const* matrix,
    KvScalar const* input,
    float* output,
    std::uint32_t batch_size) const
{
    float const alpha = 1.0F;
    float const beta = 0.0F;
    TinyLlamaConfig const& config = manifest.config;
    cublasStatus_t const status = cublasGemmEx(
        cublas,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(config.vocabulary_size),
        static_cast<int>(batch_size),
        static_cast<int>(config.hidden_size),
        &alpha,
        matrix,
        CUDA_R_16F,
        static_cast<int>(config.hidden_size),
        input,
        CUDA_R_16F,
        static_cast<int>(config.hidden_size),
        &beta,
        output,
        CUDA_R_32F,
        static_cast<int>(config.vocabulary_size),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    return status == CUBLAS_STATUS_SUCCESS
        ? CudaModelRunnerStatus{}
        : cuda_model_runner_detail::cublasFailure(status, "lm_head");
}

} // namespace kimkvcache
