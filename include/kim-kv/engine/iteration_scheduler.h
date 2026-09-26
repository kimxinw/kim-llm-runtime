#pragma once

#include "kim-kv/engine/generation.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace kimkvcache {

enum class SchedulerRequestState : std::uint8_t {
    Created,
    Waiting,
    Running,
    Cancelling,
    Terminal,
    Rejected,
};

[[nodiscard]] constexpr std::string_view toString(
    SchedulerRequestState state) noexcept
{
    switch (state) {
    case SchedulerRequestState::Created:
        return "created";
    case SchedulerRequestState::Waiting:
        return "waiting";
    case SchedulerRequestState::Running:
        return "running";
    case SchedulerRequestState::Cancelling:
        return "cancelling";
    case SchedulerRequestState::Terminal:
        return "terminal";
    case SchedulerRequestState::Rejected:
        return "rejected";
    }
    return "unknown";
}

enum class SchedulerAdmissionError : std::uint8_t {
    None,
    InvalidArgument,
    RuntimeStopped,
    DuplicateRequest,
    MaxActiveRequests,
    KvTokenBudget,
    InternalError,
};

[[nodiscard]] constexpr std::string_view toString(
    SchedulerAdmissionError error) noexcept
{
    switch (error) {
    case SchedulerAdmissionError::None:
        return "none";
    case SchedulerAdmissionError::InvalidArgument:
        return "invalid_argument";
    case SchedulerAdmissionError::RuntimeStopped:
        return "runtime_stopped";
    case SchedulerAdmissionError::DuplicateRequest:
        return "duplicate_request";
    case SchedulerAdmissionError::MaxActiveRequests:
        return "max_active_requests";
    case SchedulerAdmissionError::KvTokenBudget:
        return "kv_token_budget";
    case SchedulerAdmissionError::InternalError:
        return "internal_error";
    }
    return "unknown";
}

struct IterationSchedulerConfig final {
    std::uint32_t max_active_requests{0};
    // Total sequence positions consumed by all model batches in one iteration.
    std::uint32_t max_batched_tokens{0};
    // Worst-case committed sequence tokens reserved at admission time.
    std::uint64_t max_kv_tokens{0};
    // A prefill request may advance by at most this many prompt positions in
    // one iteration. Decode requests always advance by at most one position.
    std::uint32_t prefill_chunk_size{16};

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return max_active_requests != 0
            && max_batched_tokens != 0
            && max_kv_tokens != 0
            && prefill_chunk_size != 0;
    }
};

struct SchedulerAdmissionResult final {
    bool accepted{false};
    SchedulerAdmissionError error{SchedulerAdmissionError::InternalError};
    std::string detail{};

    [[nodiscard]] bool ok() const noexcept
    {
        return accepted && error == SchedulerAdmissionError::None;
    }
};

struct SchedulerIterationResult final {
    std::uint64_t iteration{0};
    std::uint32_t model_forward_tokens{0};
    std::uint32_t model_forward_batches{0};
    std::uint32_t prefill_tokens{0};
    std::uint32_t decode_tokens{0};
    std::uint32_t terminals_produced{0};
};

struct IterationSchedulerSnapshot final {
    std::uint64_t iteration_count{0};
    std::uint64_t accepted_count{0};
    std::uint64_t rejected_count{0};
    std::uint64_t waiting_count{0};
    std::uint64_t running_count{0};
    std::uint64_t cancelling_count{0};
    std::uint64_t terminal_count{0};
    std::uint64_t reserved_kv_tokens{0};
    std::uint64_t model_forward_tokens{0};
    std::uint64_t model_forward_batches{0};
    std::uint64_t prefill_tokens{0};
    std::uint64_t decode_tokens{0};
    bool stopped{false};

    [[nodiscard]] constexpr std::uint64_t activeCount() const noexcept
    {
        return waiting_count + running_count + cancelling_count;
    }
};

// Synchronous iteration-level FIFO scheduler. submit/cancel/runIteration/drain
// are externally serialized. stop() is thread-safe and is observed at the
// current token boundary; drain() then produces all remaining terminals.
class IterationSchedulerRuntime final {
public:
    IterationSchedulerRuntime(
        EngineKvBackend& kv_backend,
        GenerationModelRunner& model_runner,
        IterationSchedulerConfig config,
        GenerationTokenSink* token_sink = nullptr
    );

    ~IterationSchedulerRuntime();

    IterationSchedulerRuntime(IterationSchedulerRuntime const&) = delete;
    IterationSchedulerRuntime& operator=(
        IterationSchedulerRuntime const&) = delete;

    [[nodiscard]] SchedulerAdmissionResult submit(
        GenerationRequest request
    );

    [[nodiscard]] bool cancel(RequestId request_id) noexcept;

    [[nodiscard]] SchedulerIterationResult runIteration();

    // Runs until every accepted request reaches exactly one terminal. The
    // returned terminals include any that were completed before this call.
    [[nodiscard]] std::vector<GenerationTerminal> drain();

    // Terminal delivery is destructive; a terminal is never returned twice.
    [[nodiscard]] std::vector<GenerationTerminal> takeTerminals();

    // Release historical request bookkeeping after its terminal was consumed.
    // Active requests and terminals still waiting in the delivery queue cannot
    // be forgotten.
    [[nodiscard]] bool forgetRequest(RequestId request_id);

    [[nodiscard]] std::optional<SchedulerRequestState> requestState(
        RequestId request_id
    ) const;

    [[nodiscard]] IterationSchedulerSnapshot snapshot() const;
    [[nodiscard]] bool idle() const noexcept;

    void stop() noexcept;
    [[nodiscard]] bool stopped() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kimkvcache
