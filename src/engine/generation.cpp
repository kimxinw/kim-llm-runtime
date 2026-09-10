#include "kim-kv/engine/generation.h"

#include <chrono>
#include <cstddef>
#include <limits>
#include <new>
#include <utility>

namespace kimkvcache {
namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] std::uint64_t elapsedNanoseconds(
    Clock::time_point begin,
    Clock::time_point end) noexcept
{
    auto const elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end - begin
    ).count();
    return elapsed > 0 ? static_cast<std::uint64_t>(elapsed) : 0;
}

} // namespace

GenerationBatchResult GenerationModelRunner::generationForwardBatch(
    std::vector<GenerationBatchItem> const& batch)
{
    GenerationBatchResult result;
    if (batch.empty()) {
        result.detail = "generation batch must not be empty";
        return result;
    }
    try {
        result.steps.reserve(batch.size());
        for (GenerationBatchItem const& item : batch) {
            result.steps.push_back(generationForwardToken(
                item.request_id, item.token_id, item.expected_position
            ));
        }
    } catch (std::bad_alloc const&) {
        result.steps.clear();
        result.detail = "generation batch result allocation failed";
        return result;
    }
    result.success = true;
    return result;
}

GenerationBatchResult GenerationModelRunner::generationForwardChunks(
    std::vector<GenerationChunkItem> const& chunks)
{
    GenerationBatchResult result;
    if (chunks.empty()) {
        result.detail = "generation chunk batch must not be empty";
        return result;
    }
    try {
        result.steps.resize(chunks.size());
        std::vector<std::size_t> offsets(chunks.size(), 0);
        std::size_t remaining = 0;
        for (std::size_t index = 0; index < chunks.size(); ++index) {
            if (chunks[index].token_ids.empty()) {
                result.steps[index] = {
                    false, 0, "generation chunk must not be empty",
                };
            } else {
                ++remaining;
            }
        }
        while (remaining != 0) {
            std::vector<GenerationBatchItem> wave;
            std::vector<std::size_t> wave_indices;
            wave.reserve(remaining);
            wave_indices.reserve(remaining);
            for (std::size_t index = 0; index < chunks.size(); ++index) {
                if (offsets[index] >= chunks[index].token_ids.size()) {
                    continue;
                }
                wave.push_back(GenerationBatchItem{
                    chunks[index].request_id,
                    chunks[index].token_ids[offsets[index]],
                    chunks[index].expected_position
                        + static_cast<std::uint32_t>(offsets[index]),
                });
                wave_indices.push_back(index);
            }
            GenerationBatchResult wave_result = generationForwardBatch(wave);
            if (!wave_result.success
                || wave_result.steps.size() != wave.size()) {
                result.steps.clear();
                result.detail = wave_result.detail.empty()
                    ? "generation chunk compatibility wave failed"
                    : std::move(wave_result.detail);
                return result;
            }
            for (std::size_t lane = 0; lane < wave_indices.size(); ++lane) {
                std::size_t const index = wave_indices[lane];
                result.steps[index] = std::move(wave_result.steps[lane]);
                if (!result.steps[index].success
                    || ++offsets[index] == chunks[index].token_ids.size()) {
                    offsets[index] = chunks[index].token_ids.size();
                    --remaining;
                }
            }
        }
    } catch (std::bad_alloc const&) {
        result.steps.clear();
        result.detail = "generation chunk result allocation failed";
        return result;
    }
    result.success = true;
    return result;
}

std::uint32_t GenerationModelRunner::generationMaxBatchSize() const noexcept
{
    return std::numeric_limits<std::uint32_t>::max();
}

void GenerationCancellationToken::cancel() noexcept
{
    cancelled_.store(true, std::memory_order_release);
}

bool GenerationCancellationToken::cancelled() const noexcept
{
    return cancelled_.load(std::memory_order_acquire);
}

SingleRequestGenerationRuntime::SingleRequestGenerationRuntime(
    EngineKvBackend& kv_backend,
    GenerationModelRunner& model_runner) noexcept
    : kv_backend_(&kv_backend)
    , model_runner_(&model_runner)
{
}

void SingleRequestGenerationRuntime::stop() noexcept
{
    stopped_.store(true, std::memory_order_release);
}

bool SingleRequestGenerationRuntime::stopped() const noexcept
{
    return stopped_.load(std::memory_order_acquire);
}

GenerationTerminal SingleRequestGenerationRuntime::generate(
    GenerationRequest const& request)
{
    Clock::time_point const started = Clock::now();
    GenerationTerminal terminal;
    terminal.request_id = request.request_id;

    bool expected_idle = false;
    if (!generating_.compare_exchange_strong(
            expected_idle, true, std::memory_order_acq_rel)) {
        terminal.error = GenerationError::InvalidArgument;
        terminal.detail = "single-request runtime is already generating";
        terminal.metrics.e2e_ns = elapsedNanoseconds(started, Clock::now());
        return terminal;
    }

    struct GeneratingGuard final {
        std::atomic<bool>& value;
        ~GeneratingGuard()
        {
            value.store(false, std::memory_order_release);
        }
    } generating_guard{generating_};

    TinyLlamaConfig const config = model_runner_->generationConfig();
    auto const cancelled = [&request]() noexcept {
        return request.cancellation != nullptr
            && request.cancellation->cancelled();
    };

    bool request_created = false;
    Clock::time_point first_token_time{};
    Clock::time_point last_token_time{};
    bool emitted_token = false;

    auto finish = [&](GenerationTerminalReason reason,
                      GenerationError error,
                      std::string detail) {
        terminal.reason = reason;
        terminal.error = error;
        terminal.detail = std::move(detail);
        terminal.usage.completion_tokens = static_cast<std::uint32_t>(
            terminal.output_token_ids.size()
        );
        terminal.usage.total_tokens = terminal.usage.prompt_tokens
            + terminal.usage.completion_tokens;

        if (request_created) {
            EngineKvStatus const released = kv_backend_->releaseRequest(
                request.request_id
            );
            request_created = false;
            if (!released.ok()) {
                terminal.reason = GenerationTerminalReason::Failed;
                terminal.error = GenerationError::KvBackendFailed;
                terminal.detail = "release request failed: ";
                terminal.detail += toString(released.error);
            }
        }

        Clock::time_point const finished = Clock::now();
        terminal.metrics.e2e_ns = elapsedNanoseconds(started, finished);
        if (emitted_token) {
            terminal.metrics.ttft_ns = elapsedNanoseconds(
                started, first_token_time
            );
            if (terminal.output_token_ids.size() > 1) {
                terminal.metrics.tpot_ns = elapsedNanoseconds(
                    first_token_time, last_token_time
                ) / (terminal.output_token_ids.size() - 1);
            }
        }
        return terminal;
    };

    if (stopped()) {
        return finish(
            GenerationTerminalReason::RuntimeStopped,
            GenerationError::RuntimeStopped,
            "runtime is stopped"
        );
    }
    if (!config.valid() || request.request_id == kInvalidRequestId
        || request.prompt_token_ids.empty()
        || request.sampling.max_new_tokens == 0) {
        return finish(
            GenerationTerminalReason::Failed,
            GenerationError::InvalidArgument,
            "request id, prompt, model config, and max_new_tokens are required"
        );
    }
    terminal.usage.prompt_tokens = static_cast<std::uint32_t>(
        request.prompt_token_ids.size()
    );
    if (request.prompt_token_ids.size() > config.max_position_embeddings
        || request.sampling.max_new_tokens
            > config.max_position_embeddings
                - request.prompt_token_ids.size() + 1) {
        return finish(
            GenerationTerminalReason::Failed,
            GenerationError::InvalidArgument,
            "prompt and completion exceed the model position limit"
        );
    }

    std::uint32_t const eos_token = request.sampling.eos_token_id.value_or(
        config.eos_token_id
    );
    if (eos_token >= config.vocabulary_size) {
        return finish(
            GenerationTerminalReason::Failed,
            GenerationError::InvalidArgument,
            "EOS token is outside the model vocabulary"
        );
    }
    for (std::uint32_t token : request.prompt_token_ids) {
        if (token >= config.vocabulary_size) {
            return finish(
                GenerationTerminalReason::Failed,
                GenerationError::InvalidArgument,
                "prompt token is outside the model vocabulary"
            );
        }
    }
    if (cancelled()) {
        return finish(
            GenerationTerminalReason::Cancelled,
            GenerationError::None,
            "request cancelled before prefill"
        );
    }

    EngineKvStatus const created = kv_backend_->createRequest(
        request.request_id
    );
    if (!created.ok()) {
        return finish(
            GenerationTerminalReason::Failed,
            GenerationError::KvBackendFailed,
            std::string("create request failed: ")
                + std::string(toString(created.error))
        );
    }
    request_created = true;

    std::uint32_t next_token = 0;
    for (std::size_t position = 0;
         position < request.prompt_token_ids.size();
         ++position) {
        if (stopped()) {
            return finish(
                GenerationTerminalReason::RuntimeStopped,
                GenerationError::RuntimeStopped,
                "runtime stopped during prefill"
            );
        }
        if (cancelled()) {
            return finish(
                GenerationTerminalReason::Cancelled,
                GenerationError::None,
                "request cancelled during prefill"
            );
        }
        GenerationStepResult step = model_runner_->generationForwardToken(
            request.request_id,
            request.prompt_token_ids[position],
            static_cast<std::uint32_t>(position)
        );
        if (!step.success) {
            return finish(
                GenerationTerminalReason::Failed,
                GenerationError::ModelFailed,
                std::move(step.detail)
            );
        }
        if (step.greedy_token_id >= config.vocabulary_size) {
            return finish(
                GenerationTerminalReason::Failed,
                GenerationError::ModelFailed,
                "model returned a token outside the vocabulary"
            );
        }
        next_token = step.greedy_token_id;
    }

    for (std::uint32_t generated = 0;
         generated < request.sampling.max_new_tokens;
         ++generated) {
        if (stopped()) {
            return finish(
                GenerationTerminalReason::RuntimeStopped,
                GenerationError::RuntimeStopped,
                "runtime stopped during decode"
            );
        }
        if (cancelled()) {
            return finish(
                GenerationTerminalReason::Cancelled,
                GenerationError::None,
                "request cancelled during decode"
            );
        }

        try {
            terminal.output_token_ids.push_back(next_token);
        } catch (std::bad_alloc const&) {
            return finish(
                GenerationTerminalReason::Failed,
                GenerationError::InternalError,
                "completion token allocation failed"
            );
        }
        last_token_time = Clock::now();
        if (!emitted_token) {
            first_token_time = last_token_time;
            emitted_token = true;
        }
        if (next_token == eos_token) {
            return finish(
                GenerationTerminalReason::EosToken,
                GenerationError::None,
                "EOS token generated"
            );
        }
        if (generated + 1 == request.sampling.max_new_tokens) {
            return finish(
                GenerationTerminalReason::MaxNewTokens,
                GenerationError::None,
                "max_new_tokens reached"
            );
        }

        std::uint32_t const position = terminal.usage.prompt_tokens + generated;
        GenerationStepResult step = model_runner_->generationForwardToken(
            request.request_id, next_token, position
        );
        if (!step.success) {
            return finish(
                GenerationTerminalReason::Failed,
                GenerationError::ModelFailed,
                std::move(step.detail)
            );
        }
        if (step.greedy_token_id >= config.vocabulary_size) {
            return finish(
                GenerationTerminalReason::Failed,
                GenerationError::ModelFailed,
                "model returned a token outside the vocabulary"
            );
        }
        next_token = step.greedy_token_id;
    }

    return finish(
        GenerationTerminalReason::Failed,
        GenerationError::InternalError,
        "generation loop exited without a terminal reason"
    );
}

} // namespace kimkvcache
