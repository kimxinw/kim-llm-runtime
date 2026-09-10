#pragma once

#include "kim-kv/engine/generation.h"
#include "kim-kv/engine/engine_kv.h"
#include "kim-kv/model/tinyllama_config.h"
#include "kim-kv/core/kv_layout.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace kimkvcache {

enum class CudaModelRunnerError : std::uint8_t {
    None,
    InvalidArgument,
    InvalidState,
    ManifestError,
    DataFileError,
    AllocationFailed,
    CublasFailed,
    SubmissionFailed,
    ExecutionFailed,
    KvBackendFailed,
};

[[nodiscard]] constexpr std::string_view toString(
    CudaModelRunnerError error) noexcept
{
    switch (error) {
    case CudaModelRunnerError::None:
        return "none";
    case CudaModelRunnerError::InvalidArgument:
        return "invalid_argument";
    case CudaModelRunnerError::InvalidState:
        return "invalid_state";
    case CudaModelRunnerError::ManifestError:
        return "manifest_error";
    case CudaModelRunnerError::DataFileError:
        return "data_file_error";
    case CudaModelRunnerError::AllocationFailed:
        return "allocation_failed";
    case CudaModelRunnerError::CublasFailed:
        return "cublas_failed";
    case CudaModelRunnerError::SubmissionFailed:
        return "submission_failed";
    case CudaModelRunnerError::ExecutionFailed:
        return "execution_failed";
    case CudaModelRunnerError::KvBackendFailed:
        return "kv_backend_failed";
    }
    return "unknown";
}

struct CudaModelRunnerStatus final {
    CudaModelRunnerError error{CudaModelRunnerError::None};
    int native_code{0};
    std::string detail{};

    [[nodiscard]] bool ok() const noexcept
    {
        return error == CudaModelRunnerError::None;
    }

    [[nodiscard]] explicit operator bool() const noexcept
    {
        return ok();
    }
};

// Optional correctness-only capture. Stable inference does not perform these
// device-to-host copies. layer_indices uses zero-based decoder layer numbers.
struct ModelRunnerDebugCapture final {
    std::vector<std::uint32_t> layer_indices{};
    std::vector<KvScalar> embedding{};
    std::vector<std::vector<KvScalar>> layer_hidden_states{};
    std::vector<KvScalar> final_hidden_state{};
};

struct ModelTokenResult final {
    CudaModelRunnerStatus status{};
    std::uint32_t greedy_token_id{0};
    std::vector<float> logits{};

    [[nodiscard]] bool ok() const noexcept
    {
        return status.ok();
    }
};

struct ModelRunnerOutputOptions final {
    bool copy_logits{true};
};

struct CudaModelRunnerOptions final {
    // Workspace is allocated once at construction and never resized in the
    // token path. This is the total flattened token-lane limit across chunks.
    std::uint32_t max_batch_size{8};

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return max_batch_size != 0;
    }
};

struct CudaModelRunnerCreateResult;

class CudaTinyLlamaModelRunner final : public GenerationModelRunner {
public:
    struct Impl;

    ~CudaTinyLlamaModelRunner();
    CudaTinyLlamaModelRunner(CudaTinyLlamaModelRunner const&) = delete;
    CudaTinyLlamaModelRunner& operator=(
        CudaTinyLlamaModelRunner const&) = delete;
    CudaTinyLlamaModelRunner(CudaTinyLlamaModelRunner&&) noexcept;
    CudaTinyLlamaModelRunner& operator=(
        CudaTinyLlamaModelRunner&&) noexcept;

    [[nodiscard]] TinyLlamaConfig config() const noexcept;
    [[nodiscard]] std::uint64_t deviceWeightBytes() const noexcept;
    [[nodiscard]] std::uint64_t deviceWorkspaceBytes() const noexcept;
    [[nodiscard]] std::uint32_t maxBatchSize() const noexcept;

    // Executes exactly one pre-tokenized position. The request must already
    // exist in EngineKvBackend and expected_position must equal its committed
    // token count. Logits and greedy token are copied to host only after the KV
    // transaction commits successfully.
    [[nodiscard]] ModelTokenResult forwardToken(
        RequestId request_id,
        std::uint32_t token_id,
        std::uint32_t expected_position,
        ModelRunnerDebugCapture* debug_capture = nullptr,
        ModelRunnerOutputOptions output_options = {}
    );

    [[nodiscard]] TinyLlamaConfig generationConfig() const noexcept override;
    [[nodiscard]] GenerationStepResult generationForwardToken(
        RequestId request_id,
        std::uint32_t token_id,
        std::uint32_t expected_position
    ) override;

    [[nodiscard]] GenerationBatchResult generationForwardBatch(
        std::vector<GenerationBatchItem> const& batch
    ) override;

    [[nodiscard]] GenerationBatchResult generationForwardChunks(
        std::vector<GenerationChunkItem> const& chunks
    ) override;

    [[nodiscard]] std::uint32_t generationMaxBatchSize() const noexcept override;

private:
    explicit CudaTinyLlamaModelRunner(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_{};

    friend struct CudaModelRunnerCreateResult;
    friend CudaModelRunnerCreateResult createCudaTinyLlamaModelRunner(
        std::string const&,
        std::string const&,
        EngineKvBackend&,
        EngineStream,
        CudaModelRunnerOptions
    );
};

struct CudaModelRunnerCreateResult final {
    CudaModelRunnerStatus status{};
    std::unique_ptr<CudaTinyLlamaModelRunner> runner{};

    [[nodiscard]] bool ok() const noexcept
    {
        return status.ok() && runner != nullptr;
    }
};

// kv_backend and stream are borrowed and must outlive the returned runner.
// Model weights and all per-token workspace are owned by the runner itself.
[[nodiscard]] CudaModelRunnerCreateResult createCudaTinyLlamaModelRunner(
    std::string const& manifest_path,
    std::string const& data_path,
    EngineKvBackend& kv_backend,
    EngineStream stream,
    CudaModelRunnerOptions options = {}
);

} // namespace kimkvcache
