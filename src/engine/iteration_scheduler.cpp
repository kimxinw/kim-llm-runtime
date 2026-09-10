#include "kim-kv/engine/iteration_scheduler.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <deque>
#include <limits>
#include <new>
#include <unordered_map>
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

[[nodiscard]] std::uint64_t sequenceTokenBudget(
    GenerationRequest const& request) noexcept
{
    return static_cast<std::uint64_t>(request.prompt_token_ids.size())
        + static_cast<std::uint64_t>(request.sampling.max_new_tokens) - 1;
}

} // namespace

class IterationSchedulerRuntime::Impl final {
public:
    struct Record final {
        GenerationRequest request{};
        SchedulerRequestState state{SchedulerRequestState::Created};
        GenerationTerminal terminal{};
        std::uint64_t kv_token_budget{0};
        std::uint32_t prompt_position{0};
        bool request_created{false};
        bool cancel_requested{false};
        bool emitted_token{false};
        Clock::time_point started{};
        Clock::time_point first_token_time{};
        Clock::time_point last_token_time{};
    };

    Impl(
        EngineKvBackend& kv_backend,
        GenerationModelRunner& model_runner,
        IterationSchedulerConfig scheduler_config) noexcept
        : kv(&kv_backend)
        , runner(&model_runner)
        , config(scheduler_config)
    {
    }

    [[nodiscard]] SchedulerAdmissionResult reject(
        RequestId request_id,
        SchedulerAdmissionError error,
        std::string detail,
        bool remember)
    {
        if (remember && request_id != kInvalidRequestId) {
            try {
                states.emplace(request_id, SchedulerRequestState::Rejected);
            } catch (std::bad_alloc const&) {
                return {
                    false,
                    SchedulerAdmissionError::InternalError,
                    "failed to record rejected request",
                };
            }
        }
        return {false, error, std::move(detail)};
    }

    [[nodiscard]] std::optional<std::string> validate(
        GenerationRequest const& request) const
    {
        TinyLlamaConfig const model_config = runner->generationConfig();
        if (!config.valid()) {
            return "scheduler configuration is invalid";
        }
        if (!model_config.valid() || request.request_id == kInvalidRequestId
            || request.prompt_token_ids.empty()
            || request.sampling.max_new_tokens == 0) {
            return "request id, prompt, model config, and max_new_tokens are required";
        }
        if (request.prompt_token_ids.size()
                > model_config.max_position_embeddings
            || request.sampling.max_new_tokens
                > model_config.max_position_embeddings
                    - request.prompt_token_ids.size() + 1) {
            return "prompt and completion exceed the model position limit";
        }
        std::uint32_t const eos = request.sampling.eos_token_id.value_or(
            model_config.eos_token_id
        );
        if (eos >= model_config.vocabulary_size) {
            return "EOS token is outside the model vocabulary";
        }
        for (std::uint32_t token : request.prompt_token_ids) {
            if (token >= model_config.vocabulary_size) {
                return "prompt token is outside the model vocabulary";
            }
        }
        return std::nullopt;
    }

    void finish(
        Record& record,
        GenerationTerminalReason reason,
        GenerationError error,
        std::string detail)
    {
        if (record.state == SchedulerRequestState::Terminal) {
            return;
        }

        record.terminal.reason = reason;
        record.terminal.error = error;
        record.terminal.detail = std::move(detail);
        record.terminal.usage.completion_tokens =
            static_cast<std::uint32_t>(
                record.terminal.output_token_ids.size()
            );
        record.terminal.usage.total_tokens =
            record.terminal.usage.prompt_tokens
            + record.terminal.usage.completion_tokens;

        if (record.request_created) {
            EngineKvStatus const released = kv->releaseRequest(
                record.request.request_id
            );
            record.request_created = false;
            if (!released.ok()) {
                record.terminal.reason = GenerationTerminalReason::Failed;
                record.terminal.error = GenerationError::KvBackendFailed;
                record.terminal.detail = "release request failed: ";
                record.terminal.detail += toString(released.error);
            }
        }

        Clock::time_point const finished = Clock::now();
        record.terminal.metrics.e2e_ns = elapsedNanoseconds(
            record.started, finished
        );
        if (record.emitted_token) {
            record.terminal.metrics.ttft_ns = elapsedNanoseconds(
                record.started, record.first_token_time
            );
            if (record.terminal.output_token_ids.size() > 1) {
                record.terminal.metrics.tpot_ns = elapsedNanoseconds(
                    record.first_token_time, record.last_token_time
                ) / (record.terminal.output_token_ids.size() - 1);
            }
        }

        record.state = SchedulerRequestState::Terminal;
        auto const state = states.find(record.request.request_id);
        if (state != states.end()) {
            state->second = SchedulerRequestState::Terminal;
        }
        reserved_kv_tokens -= record.kv_token_budget;
        try {
            terminal_queue.push_back(record.terminal);
        } catch (std::bad_alloc const&) {
            // State and resources are already terminal/reclaimed. Delivery can
            // be retried from the retained record only in a future API.
        }
    }

    void stopRecord(Record& record)
    {
        finish(
            record,
            GenerationTerminalReason::RuntimeStopped,
            GenerationError::RuntimeStopped,
            "runtime stopped at iteration boundary"
        );
    }

    void stopAll()
    {
        while (!schedule_queue.empty()) {
            RequestId const request_id = schedule_queue.front();
            schedule_queue.pop_front();
            auto const found = records.find(request_id);
            if (found != records.end()
                && found->second.state != SchedulerRequestState::Terminal) {
                stopRecord(found->second);
            }
        }
    }

    [[nodiscard]] bool prepareChunk(
        Record& record,
        std::uint32_t max_tokens,
        GenerationChunkItem& item,
        std::uint32_t& prefill_token_count)
    {
        if (record.cancel_requested
            || (record.request.cancellation != nullptr
                && record.request.cancellation->cancelled())) {
            finish(
                record,
                GenerationTerminalReason::Cancelled,
                GenerationError::None,
                "request cancelled at iteration boundary"
            );
            return false;
        }
        if (stopped.load(std::memory_order_acquire)) {
            stopRecord(record);
            return false;
        }

        if (!record.request_created) {
            EngineKvStatus const created = kv->createRequest(
                record.request.request_id
            );
            if (!created.ok()) {
                finish(
                    record,
                    GenerationTerminalReason::Failed,
                    GenerationError::KvBackendFailed,
                    std::string("create request failed: ")
                        + std::string(toString(created.error))
                );
                return false;
            }
            record.request_created = true;
            record.state = SchedulerRequestState::Running;
            auto const state = states.find(record.request.request_id);
            if (state != states.end()) {
                state->second = SchedulerRequestState::Running;
            }
        }

        bool const prefill = record.prompt_position
            < record.request.prompt_token_ids.size();
        std::uint32_t position = 0;
        std::vector<std::uint32_t> tokens;
        if (prefill) {
            position = record.prompt_position;
            std::uint32_t const remaining = static_cast<std::uint32_t>(
                record.request.prompt_token_ids.size()
            ) - position;
            prefill_token_count = std::min({
                remaining, config.prefill_chunk_size, max_tokens,
            });
            if (prefill_token_count == 0) {
                return false;
            }
            tokens.assign(
                record.request.prompt_token_ids.begin() + position,
                record.request.prompt_token_ids.begin()
                    + position + prefill_token_count
            );
        } else {
            prefill_token_count = 0;
            position = record.terminal.usage.prompt_tokens
                + static_cast<std::uint32_t>(
                    record.terminal.output_token_ids.size()
                ) - 1;
            tokens.push_back(record.terminal.output_token_ids.back());
        }

        item = GenerationChunkItem{
            record.request.request_id, std::move(tokens), position,
        };
        return true;
    }

    void applyChunk(
        Record& record,
        GenerationStepResult step,
        std::uint32_t prefill_token_count)
    {
        if (!step.success) {
            finish(
                record,
                GenerationTerminalReason::Failed,
                GenerationError::ModelFailed,
                std::move(step.detail)
            );
            return;
        }
        TinyLlamaConfig const model_config = runner->generationConfig();
        if (step.greedy_token_id >= model_config.vocabulary_size) {
            finish(
                record,
                GenerationTerminalReason::Failed,
                GenerationError::ModelFailed,
                "model returned a token outside the vocabulary"
            );
            return;
        }
        record.prompt_position += prefill_token_count;

        if (stopped.load(std::memory_order_acquire)) {
            stopRecord(record);
            return;
        }
        if (record.cancel_requested
            || (record.request.cancellation != nullptr
                && record.request.cancellation->cancelled())) {
            finish(
                record,
                GenerationTerminalReason::Cancelled,
                GenerationError::None,
                "request cancelled at iteration boundary"
            );
            return;
        }

        if (record.prompt_position
            != record.request.prompt_token_ids.size()) {
            return;
        }

        try {
            record.terminal.output_token_ids.push_back(
                step.greedy_token_id
            );
        } catch (std::bad_alloc const&) {
            finish(
                record,
                GenerationTerminalReason::Failed,
                GenerationError::InternalError,
                "completion token allocation failed"
            );
            return;
        }
        record.last_token_time = Clock::now();
        if (!record.emitted_token) {
            record.first_token_time = record.last_token_time;
            record.emitted_token = true;
        }

        std::uint32_t const eos = record.request.sampling.eos_token_id.value_or(
            model_config.eos_token_id
        );
        if (step.greedy_token_id == eos) {
            finish(
                record,
                GenerationTerminalReason::EosToken,
                GenerationError::None,
                "EOS token generated"
            );
        } else if (record.terminal.output_token_ids.size()
            == record.request.sampling.max_new_tokens) {
            finish(
                record,
                GenerationTerminalReason::MaxNewTokens,
                GenerationError::None,
                "max_new_tokens reached"
            );
        }
    }

    EngineKvBackend* kv{nullptr};
    GenerationModelRunner* runner{nullptr};
    IterationSchedulerConfig config{};
    std::atomic<bool> stopped{false};
    std::unordered_map<RequestId, Record> records{};
    std::unordered_map<RequestId, SchedulerRequestState> states{};
    std::deque<RequestId> schedule_queue{};
    std::vector<GenerationTerminal> terminal_queue{};
    std::uint64_t iteration_count{0};
    std::uint64_t reserved_kv_tokens{0};
    std::uint64_t model_forward_tokens{0};
    std::uint64_t model_forward_batches{0};
    std::uint64_t prefill_tokens{0};
    std::uint64_t decode_tokens{0};
};

IterationSchedulerRuntime::IterationSchedulerRuntime(
    EngineKvBackend& kv_backend,
    GenerationModelRunner& model_runner,
    IterationSchedulerConfig config)
    : impl_(std::make_unique<Impl>(kv_backend, model_runner, config))
{
}

IterationSchedulerRuntime::~IterationSchedulerRuntime()
{
    if (impl_ != nullptr) {
        impl_->stopped.store(true, std::memory_order_release);
        impl_->stopAll();
    }
}

SchedulerAdmissionResult IterationSchedulerRuntime::submit(
    GenerationRequest request)
{
    if (stopped()) {
        return impl_->reject(
            request.request_id,
            SchedulerAdmissionError::RuntimeStopped,
            "runtime is stopped",
            true
        );
    }
    if (impl_->states.find(request.request_id) != impl_->states.end()) {
        return impl_->reject(
            request.request_id,
            SchedulerAdmissionError::DuplicateRequest,
            "request id has already been submitted",
            false
        );
    }
    if (std::optional<std::string> invalid = impl_->validate(request)) {
        return impl_->reject(
            request.request_id,
            SchedulerAdmissionError::InvalidArgument,
            std::move(*invalid),
            true
        );
    }

    IterationSchedulerSnapshot const current = snapshot();
    if (current.activeCount() >= impl_->config.max_active_requests) {
        return impl_->reject(
            request.request_id,
            SchedulerAdmissionError::MaxActiveRequests,
            "max_active_requests admission gate rejected request",
            true
        );
    }
    std::uint64_t const kv_budget = sequenceTokenBudget(request);
    if (kv_budget > impl_->config.max_kv_tokens
        || impl_->reserved_kv_tokens
            > impl_->config.max_kv_tokens - kv_budget) {
        return impl_->reject(
            request.request_id,
            SchedulerAdmissionError::KvTokenBudget,
            "KV token budget admission gate rejected request",
            true
        );
    }

    try {
        Impl::Record record;
        record.request = std::move(request);
        record.state = SchedulerRequestState::Waiting;
        record.terminal.request_id = record.request.request_id;
        record.terminal.usage.prompt_tokens = static_cast<std::uint32_t>(
            record.request.prompt_token_ids.size()
        );
        record.kv_token_budget = kv_budget;
        record.started = Clock::now();
        RequestId const request_id = record.request.request_id;
        impl_->records.emplace(request_id, std::move(record));
        try {
            impl_->states.emplace(request_id, SchedulerRequestState::Waiting);
            impl_->schedule_queue.push_back(request_id);
        } catch (...) {
            impl_->states.erase(request_id);
            impl_->records.erase(request_id);
            throw;
        }
        impl_->reserved_kv_tokens += kv_budget;
    } catch (std::bad_alloc const&) {
        return {
            false,
            SchedulerAdmissionError::InternalError,
            "scheduler admission allocation failed",
        };
    }
    return {true, SchedulerAdmissionError::None, {}};
}

bool IterationSchedulerRuntime::cancel(RequestId request_id) noexcept
{
    auto const found = impl_->records.find(request_id);
    if (found == impl_->records.end()
        || found->second.state == SchedulerRequestState::Terminal) {
        return false;
    }
    found->second.cancel_requested = true;
    found->second.state = SchedulerRequestState::Cancelling;
    auto const state = impl_->states.find(request_id);
    if (state != impl_->states.end()) {
        state->second = SchedulerRequestState::Cancelling;
    }
    return true;
}

SchedulerIterationResult IterationSchedulerRuntime::runIteration()
{
    SchedulerIterationResult result;
    result.iteration = ++impl_->iteration_count;
    std::size_t const terminals_before = impl_->terminal_queue.size();

    if (stopped()) {
        impl_->stopAll();
        result.terminals_produced = static_cast<std::uint32_t>(
            impl_->terminal_queue.size() - terminals_before
        );
        return result;
    }

    struct PlannedStep final {
        Impl::Record* record{nullptr};
        GenerationChunkItem item{};
        std::uint32_t prefill_token_count{0};
    };

    std::size_t const scheduled = std::min<std::size_t>(
        impl_->schedule_queue.size(), impl_->config.max_batched_tokens
    );
    std::vector<Impl::Record*> selected;
    try {
        selected.reserve(scheduled);
        for (std::size_t index = 0; index < scheduled; ++index) {
            RequestId const request_id = impl_->schedule_queue.front();
            impl_->schedule_queue.pop_front();
            auto const found = impl_->records.find(request_id);
            if (found == impl_->records.end()
                || found->second.state == SchedulerRequestState::Terminal) {
                continue;
            }
            Impl::Record& record = found->second;
            selected.push_back(&record);
        }
    } catch (std::bad_alloc const&) {
        for (Impl::Record* record : selected) {
            if (record->state != SchedulerRequestState::Terminal) {
                impl_->finish(
                    *record,
                    GenerationTerminalReason::Failed,
                    GenerationError::InternalError,
                    "scheduler batch planning allocation failed"
                );
            }
        }
    }

    std::uint32_t remaining_budget = impl_->config.max_batched_tokens;
    std::uint32_t const runner_batch_size =
        impl_->runner->generationMaxBatchSize();
    if (runner_batch_size == 0) {
        for (Impl::Record* record : selected) {
            if (record->state != SchedulerRequestState::Terminal) {
                impl_->finish(
                    *record,
                    GenerationTerminalReason::Failed,
                    GenerationError::ModelFailed,
                    "model runner reported a zero maximum batch size"
                );
            }
        }
    }
    std::vector<PlannedStep> plan;
    try {
        plan.reserve(selected.size());
        std::size_t remaining_candidates = selected.size();
        for (Impl::Record* record : selected) {
            if (remaining_budget == 0
                || record->state == SchedulerRequestState::Terminal) {
                --remaining_candidates;
                continue;
            }
            GenerationChunkItem item;
            std::uint32_t prefill_token_count = 0;
            std::uint32_t const reserved_for_rest =
                static_cast<std::uint32_t>(remaining_candidates - 1);
            std::uint32_t const fair_budget = remaining_budget
                > reserved_for_rest
                ? remaining_budget - reserved_for_rest : 1;
            std::uint32_t const chunk_limit = std::min(
                fair_budget, runner_batch_size
            );
            if (impl_->prepareChunk(
                    *record, chunk_limit, item, prefill_token_count)) {
                std::uint32_t const token_count = static_cast<std::uint32_t>(
                    item.token_ids.size()
                );
                plan.push_back(PlannedStep{
                    record, std::move(item), prefill_token_count,
                });
                remaining_budget -= token_count;
            }
            --remaining_candidates;
        }
    } catch (std::bad_alloc const&) {
        for (Impl::Record* record : selected) {
            if (record->state != SchedulerRequestState::Terminal) {
                impl_->finish(
                    *record,
                    GenerationTerminalReason::Failed,
                    GenerationError::InternalError,
                    "scheduler model chunk allocation failed"
                );
            }
        }
    }

    for (std::size_t begin = 0; begin < plan.size() && !stopped();) {
        std::size_t end = begin;
        std::uint32_t batch_tokens = 0;
        while (end < plan.size()) {
            std::uint32_t const chunk_tokens = static_cast<std::uint32_t>(
                plan[end].item.token_ids.size()
            );
            if (batch_tokens != 0
                && chunk_tokens > runner_batch_size - batch_tokens) {
                break;
            }
            batch_tokens += chunk_tokens;
            ++end;
        }
        std::vector<GenerationChunkItem> batch;
        try {
            batch.reserve(end - begin);
            for (std::size_t index = begin; index < end; ++index) {
                batch.push_back(plan[index].item);
            }
        } catch (std::bad_alloc const&) {
            for (std::size_t index = begin; index < end; ++index) {
                impl_->finish(
                    *plan[index].record,
                    GenerationTerminalReason::Failed,
                    GenerationError::InternalError,
                    "scheduler model chunk batch allocation failed"
                );
            }
            begin = end;
            continue;
        }

        GenerationBatchResult batch_result =
            impl_->runner->generationForwardChunks(batch);
        ++result.model_forward_batches;
        result.model_forward_tokens += batch_tokens;
        for (std::size_t index = begin; index < end; ++index) {
            result.prefill_tokens += plan[index].prefill_token_count;
            if (plan[index].prefill_token_count == 0) {
                ++result.decode_tokens;
            }
        }
        if (!batch_result.success
            || batch_result.steps.size() != batch.size()) {
            std::string detail = batch_result.detail.empty()
                ? "model returned an invalid chunk batch result"
                : std::move(batch_result.detail);
            for (std::size_t index = begin; index < end; ++index) {
                impl_->finish(
                    *plan[index].record,
                    GenerationTerminalReason::Failed,
                    GenerationError::ModelFailed,
                    detail
                );
            }
        } else {
            for (std::size_t index = begin; index < end; ++index) {
                impl_->applyChunk(
                    *plan[index].record,
                    std::move(batch_result.steps[index - begin]),
                    plan[index].prefill_token_count
                );
            }
        }
        begin = end;
    }

    impl_->model_forward_tokens += result.model_forward_tokens;
    impl_->model_forward_batches += result.model_forward_batches;
    impl_->prefill_tokens += result.prefill_tokens;
    impl_->decode_tokens += result.decode_tokens;

    for (Impl::Record* record : selected) {
        if (record->state != SchedulerRequestState::Terminal) {
            if (stopped()) {
                impl_->stopRecord(*record);
            } else {
                impl_->schedule_queue.push_back(
                    record->request.request_id
                );
            }
        }
    }
    if (stopped()) {
        impl_->stopAll();
    }
    result.terminals_produced = static_cast<std::uint32_t>(
        impl_->terminal_queue.size() - terminals_before
    );
    return result;
}

std::vector<GenerationTerminal> IterationSchedulerRuntime::drain()
{
    while (!idle()) {
        SchedulerIterationResult const iteration = runIteration();
        if (iteration.model_forward_tokens == 0
            && iteration.terminals_produced == 0 && !stopped()) {
            break;
        }
    }
    return takeTerminals();
}

std::vector<GenerationTerminal> IterationSchedulerRuntime::takeTerminals()
{
    std::vector<GenerationTerminal> result;
    result.swap(impl_->terminal_queue);
    return result;
}

std::optional<SchedulerRequestState> IterationSchedulerRuntime::requestState(
    RequestId request_id) const
{
    auto const found = impl_->states.find(request_id);
    if (found == impl_->states.end()) {
        return std::nullopt;
    }
    return found->second;
}

IterationSchedulerSnapshot IterationSchedulerRuntime::snapshot() const
{
    IterationSchedulerSnapshot result;
    result.iteration_count = impl_->iteration_count;
    result.accepted_count = impl_->records.size();
    result.reserved_kv_tokens = impl_->reserved_kv_tokens;
    result.model_forward_tokens = impl_->model_forward_tokens;
    result.model_forward_batches = impl_->model_forward_batches;
    result.prefill_tokens = impl_->prefill_tokens;
    result.decode_tokens = impl_->decode_tokens;
    result.stopped = stopped();
    for (auto const& entry : impl_->states) {
        switch (entry.second) {
        case SchedulerRequestState::Created:
            break;
        case SchedulerRequestState::Waiting:
            ++result.waiting_count;
            break;
        case SchedulerRequestState::Running:
            ++result.running_count;
            break;
        case SchedulerRequestState::Cancelling:
            ++result.cancelling_count;
            break;
        case SchedulerRequestState::Terminal:
            ++result.terminal_count;
            break;
        case SchedulerRequestState::Rejected:
            ++result.rejected_count;
            break;
        }
    }
    return result;
}

bool IterationSchedulerRuntime::idle() const noexcept
{
    return impl_->schedule_queue.empty();
}

void IterationSchedulerRuntime::stop() noexcept
{
    impl_->stopped.store(true, std::memory_order_release);
}

bool IterationSchedulerRuntime::stopped() const noexcept
{
    return impl_->stopped.load(std::memory_order_acquire);
}

} // namespace kimkvcache
