#pragma once

#include "kim-kv/engine/engine_kv.h"
#include "kim-kv/model/tinyllama_config.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace kimkvcache {

enum class GenerationError : std::uint8_t {
    None,
    InvalidArgument,
    RuntimeStopped,
    ModelFailed,
    KvBackendFailed,
    InternalError,
};

[[nodiscard]] constexpr std::string_view toString(
    GenerationError error) noexcept
{
    switch (error) {
    case GenerationError::None:
        return "none";
    case GenerationError::InvalidArgument:
        return "invalid_argument";
    case GenerationError::RuntimeStopped:
        return "runtime_stopped";
    case GenerationError::ModelFailed:
        return "model_failed";
    case GenerationError::KvBackendFailed:
        return "kv_backend_failed";
    case GenerationError::InternalError:
        return "internal_error";
    }
    return "unknown";
}

enum class GenerationTerminalReason : std::uint8_t {
    EosToken,
    MaxNewTokens,
    Cancelled,
    RuntimeStopped,
    Failed,
};

[[nodiscard]] constexpr std::string_view toString(
    GenerationTerminalReason reason) noexcept
{
    switch (reason) {
    case GenerationTerminalReason::EosToken:
        return "eos_token";
    case GenerationTerminalReason::MaxNewTokens:
        return "max_new_tokens";
    case GenerationTerminalReason::Cancelled:
        return "cancelled";
    case GenerationTerminalReason::RuntimeStopped:
        return "runtime_stopped";
    case GenerationTerminalReason::Failed:
        return "failed";
    }
    return "unknown";
}

struct SamplingConfig final {
    std::uint32_t max_new_tokens{0};
    // nullopt selects the model's configured EOS token. Greedy is the only
    // sampling policy in E3.
    std::optional<std::uint32_t> eos_token_id{};
};

class GenerationCancellationToken final {
public:
    void cancel() noexcept;
    [[nodiscard]] bool cancelled() const noexcept;

private:
    std::atomic<bool> cancelled_{false};
};

struct GenerationRequest final {
    RequestId request_id{kInvalidRequestId};
    std::vector<std::uint32_t> prompt_token_ids{};
    SamplingConfig sampling{};
    std::shared_ptr<GenerationCancellationToken> cancellation{};
};

struct GenerationUsage final {
    std::uint32_t prompt_tokens{0};
    std::uint32_t completion_tokens{0};
    std::uint32_t total_tokens{0};
};

struct GenerationMetrics final {
    std::uint64_t ttft_ns{0};
    std::uint64_t tpot_ns{0};
    std::uint64_t e2e_ns{0};
};

struct GenerationTerminal final {
    RequestId request_id{kInvalidRequestId};
    GenerationTerminalReason reason{GenerationTerminalReason::Failed};
    GenerationError error{GenerationError::InternalError};
    std::string detail{};
    std::vector<std::uint32_t> output_token_ids{};
    GenerationUsage usage{};
    GenerationMetrics metrics{};

    [[nodiscard]] bool ok() const noexcept
    {
        return error == GenerationError::None
            && (reason == GenerationTerminalReason::EosToken
                || reason == GenerationTerminalReason::MaxNewTokens);
    }
};

struct GenerationStepResult final {
    bool success{false};
    std::uint32_t greedy_token_id{0};
    std::string detail{};
};

struct GenerationBatchItem final {
    RequestId request_id{kInvalidRequestId};
    std::uint32_t token_id{0};
    std::uint32_t expected_position{0};
};

// One request-local contiguous model segment. token_ids are laid out at
// [expected_position, expected_position + token_ids.size()).
struct GenerationChunkItem final {
    RequestId request_id{kInvalidRequestId};
    std::vector<std::uint32_t> token_ids{};
    std::uint32_t expected_position{0};
};

struct GenerationBatchResult final {
    bool success{false};
    std::vector<GenerationStepResult> steps{};
    std::string detail{};
};

// Narrow model SPI used by the E3 single-request loop. It deliberately
// exposes only the greedy next token; logits/debug captures remain runner
// specific correctness facilities.
class GenerationModelRunner {
public:
    virtual ~GenerationModelRunner() = default;

    GenerationModelRunner(GenerationModelRunner const&) = delete;
    GenerationModelRunner& operator=(GenerationModelRunner const&) = delete;

    [[nodiscard]] virtual TinyLlamaConfig generationConfig() const noexcept = 0;
    [[nodiscard]] virtual GenerationStepResult generationForwardToken(
        RequestId request_id,
        std::uint32_t token_id,
        std::uint32_t expected_position
    ) = 0;

    // Executes independent sequence positions as one model batch. The default
    // implementation preserves compatibility by forwarding each item through
    // generationForwardToken(); CUDA runners override this with a real dense
    // batch. A successful result always contains exactly batch.size() steps,
    // with request-local failures represented by the corresponding step.
    [[nodiscard]] virtual GenerationBatchResult generationForwardBatch(
        std::vector<GenerationBatchItem> const& batch
    );

    // Executes one or more request-local chunks with a shared total-token
    // budget. The compatibility implementation advances each chunk through
    // the scalar SPI; accelerated runners override this with true prefill.
    [[nodiscard]] virtual GenerationBatchResult generationForwardChunks(
        std::vector<GenerationChunkItem> const& chunks
    );

    [[nodiscard]] virtual std::uint32_t generationMaxBatchSize() const noexcept;

protected:
    GenerationModelRunner() = default;
    GenerationModelRunner(GenerationModelRunner&&) noexcept = default;
    GenerationModelRunner& operator=(GenerationModelRunner&&) noexcept = default;
};

// E3 owns exactly one synchronous request at a time. stop() is thread-safe and
// is observed at token boundaries. The iteration scheduler builds admission
// and multi-request lifecycle semantics around this unchanged E3 contract.
class SingleRequestGenerationRuntime final {
public:
    SingleRequestGenerationRuntime(
        EngineKvBackend& kv_backend,
        GenerationModelRunner& model_runner
    ) noexcept;

    SingleRequestGenerationRuntime(
        SingleRequestGenerationRuntime const&) = delete;
    SingleRequestGenerationRuntime& operator=(
        SingleRequestGenerationRuntime const&) = delete;

    [[nodiscard]] GenerationTerminal generate(
        GenerationRequest const& request
    );

    void stop() noexcept;
    [[nodiscard]] bool stopped() const noexcept;

private:
    EngineKvBackend* kv_backend_{nullptr};
    GenerationModelRunner* model_runner_{nullptr};
    std::atomic<bool> stopped_{false};
    std::atomic<bool> generating_{false};
};

} // namespace kimkvcache
