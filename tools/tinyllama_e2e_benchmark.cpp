#include "kim-kv/cuda/cuda_engine_kv_backend.h"
#include "kim-kv/cuda/cuda_model_runner.h"
#include "kim-kv/engine/iteration_scheduler.h"
#include "kim-kv/model/weight_manifest.h"

#include <cuda_runtime_api.h>

#if defined(KIM_KV_HAS_NVTX)
#include <nvtx3/nvToolsExt.h>
#endif

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace kimkvcache;
using Clock = std::chrono::steady_clock;

enum class Variant : std::uint8_t {
    Heterogeneous,
    Fixed8,
    Fixed16,
    Fixed32,
    Fixed64,
};

struct Options final {
    std::string manifest{};
    std::string weights{};
    std::string output{};
    std::string git_commit{"unknown"};
    Variant variant{Variant::Heterogeneous};
    std::optional<std::string> case_name{};
    bool profile{false};
    std::uint32_t warmup{1};
    std::uint32_t iterations{3};
    std::uint32_t kv_capacity_tokens{8192};
    std::uint32_t capacity_probe_tokens{512};
};

struct CaseSpec final {
    std::string name{};
    std::uint32_t concurrency{1};
    std::vector<std::uint32_t> prompt_lengths{};
    std::uint32_t output_length{32};
    bool fault{false};
    bool capacity{false};
};

struct CaseSelection final {
    std::vector<CaseSpec> performance{};
    bool capacity{false};

    [[nodiscard]] bool empty() const noexcept
    {
        return performance.empty() && !capacity;
    }
};

struct RequestEvidence final {
    RequestId request_id{kInvalidRequestId};
    std::vector<std::uint32_t> input{};
    std::vector<std::uint32_t> output{};
    GenerationTerminalReason reason{GenerationTerminalReason::Failed};
    GenerationError error{GenerationError::InternalError};
    std::uint64_t ttft_ns{0};
    std::uint64_t tpot_ns{0};
    std::uint64_t e2e_ns{0};
};

struct RunEvidence final {
    std::uint64_t elapsed_ns{0};
    std::uint64_t output_tokens{0};
    std::uint64_t model_forward_tokens{0};
    std::uint64_t model_forward_batches{0};
    std::uint64_t batched_attention_submissions{0};
    std::uint64_t batched_attention_lanes{0};
    std::uint64_t automatic_promotion_attempts{0};
    std::uint64_t automatic_promotion_successes{0};
    std::uint64_t automatic_promotion_skips{0};
    std::uint64_t automatic_promotion_failures{0};
    std::uint64_t prefill_tokens{0};
    std::uint64_t decode_tokens{0};
    std::uint32_t accepted{0};
    std::uint32_t rejected{0};
    std::uint32_t completed{0};
    std::uint32_t failed{0};
    bool resources_reclaimed{false};
    bool expected_outcome{false};
    std::vector<RequestEvidence> requests{};
};

struct ResourceProbe final {
    std::uint64_t peak_committed_tokens{0};
    std::uint64_t peak_reserved_tokens{0};
    std::uint64_t peak_fragmentation_tokens{0};
    std::uint32_t peak_primary_pages{0};
    std::uint32_t peak_secondary_pages{0};
    std::uint64_t primary_allocations{0};
    std::uint64_t secondary_allocations{0};
    std::uint64_t failed_primary_allocations{0};
    std::uint64_t failed_secondary_allocations{0};
    std::uint64_t peak_dynamic_gpu_bytes{0};
};

struct CaseEvidence final {
    CaseSpec spec{};
    bool passed{false};
    bool outputs_consistent{true};
    std::vector<RunEvidence> runs{};
    ResourceProbe resources{};
    std::uint64_t ttft_p50_ns{0};
    std::uint64_t ttft_p95_ns{0};
    std::uint64_t ttft_p99_ns{0};
    std::uint64_t tpot_p50_ns{0};
    std::uint64_t tpot_p95_ns{0};
    std::uint64_t tpot_p99_ns{0};
    std::uint64_t e2e_p50_ns{0};
    std::uint64_t e2e_p95_ns{0};
    std::uint64_t e2e_p99_ns{0};
    double requests_per_second{0.0};
    double output_tokens_per_second{0.0};
    double goodput_requests_per_second{0.0};
};

struct SuiteContext final {
    cudaStream_t stream{nullptr};
    std::unique_ptr<EngineKvBackend> backend{};
    CudaModelRunnerCreateResult model{};
    std::uint64_t storage_bytes{0};
    std::uint64_t weight_bytes{0};
    std::uint64_t workspace_bytes{0};
    std::uint64_t loaded_gpu_bytes{0};
    std::uint32_t capacity_tokens{0};
};

class NvtxRange final {
public:
    NvtxRange(bool enabled, std::string const& name) noexcept
        : active_(enabled)
    {
#if defined(KIM_KV_HAS_NVTX)
        if (active_) {
            static_cast<void>(nvtxRangePushA(name.c_str()));
        }
#else
        static_cast<void>(name);
#endif
    }

    ~NvtxRange()
    {
#if defined(KIM_KV_HAS_NVTX)
        if (active_) {
            static_cast<void>(nvtxRangePop());
        }
#endif
    }

    NvtxRange(NvtxRange const&) = delete;
    NvtxRange& operator=(NvtxRange const&) = delete;

private:
    bool active_{false};
};

class ProfilingModelRunner final : public GenerationModelRunner {
public:
    ProfilingModelRunner(
        GenerationModelRunner& delegate,
        std::map<RequestId, std::uint32_t> prompt_lengths)
        : delegate_(&delegate), prompt_lengths_(std::move(prompt_lengths))
    {
    }

    [[nodiscard]] TinyLlamaConfig generationConfig() const noexcept override
    {
        return delegate_->generationConfig();
    }

    [[nodiscard]] GenerationStepResult generationForwardToken(
        RequestId request_id,
        std::uint32_t token_id,
        std::uint32_t expected_position) override
    {
        NvtxRange const range(
            true, phaseName({{request_id, expected_position}})
        );
        return delegate_->generationForwardToken(
            request_id, token_id, expected_position
        );
    }

    [[nodiscard]] GenerationBatchResult generationForwardBatch(
        std::vector<GenerationBatchItem> const& batch) override
    {
        std::vector<RequestPosition> positions;
        positions.reserve(batch.size());
        for (GenerationBatchItem const& item : batch) {
            positions.push_back({item.request_id, item.expected_position});
        }
        NvtxRange const range(true, phaseName(positions));
        return delegate_->generationForwardBatch(batch);
    }

    [[nodiscard]] GenerationBatchResult generationForwardChunks(
        std::vector<GenerationChunkItem> const& chunks) override
    {
        std::vector<RequestPosition> positions;
        positions.reserve(chunks.size());
        for (GenerationChunkItem const& item : chunks) {
            positions.push_back({item.request_id, item.expected_position});
        }
        NvtxRange const range(true, phaseName(positions));
        return delegate_->generationForwardChunks(chunks);
    }

    [[nodiscard]] std::uint32_t generationMaxBatchSize() const noexcept override
    {
        return delegate_->generationMaxBatchSize();
    }

private:
    struct RequestPosition final {
        RequestId request_id{kInvalidRequestId};
        std::uint32_t expected_position{0};
    };

    [[nodiscard]] std::string phaseName(
        std::vector<RequestPosition> const& positions) const
    {
        bool has_prefill = false;
        bool has_decode = false;
        bool has_unknown = false;
        for (RequestPosition const& position : positions) {
            auto const found = prompt_lengths_.find(position.request_id);
            if (found == prompt_lengths_.end()) {
                has_unknown = true;
            } else if (position.expected_position < found->second) {
                has_prefill = true;
            } else {
                has_decode = true;
            }
        }
        if (has_unknown) {
            return "unclassified";
        }
        if (has_prefill && has_decode) {
            return "mixed";
        }
        if (has_prefill) {
            return "prefill";
        }
        if (has_decode) {
            return "decode";
        }
        return "empty";
    }

    GenerationModelRunner* delegate_{nullptr};
    std::map<RequestId, std::uint32_t> prompt_lengths_{};
};

[[nodiscard]] std::string_view variantName(Variant variant) noexcept
{
    switch (variant) {
    case Variant::Heterogeneous:
        return "hetero_8_64";
    case Variant::Fixed8:
        return "fixed_8";
    case Variant::Fixed16:
        return "fixed_16";
    case Variant::Fixed32:
        return "fixed_32";
    case Variant::Fixed64:
        return "fixed_64";
    }
    return "unknown";
}

[[nodiscard]] std::optional<Variant> parseVariant(
    std::string_view value) noexcept
{
    if (value == "hetero" || value == "hetero_8_64") {
        return Variant::Heterogeneous;
    }
    if (value == "fixed_8") {
        return Variant::Fixed8;
    }
    if (value == "fixed_16") {
        return Variant::Fixed16;
    }
    if (value == "fixed_32") {
        return Variant::Fixed32;
    }
    if (value == "fixed_64") {
        return Variant::Fixed64;
    }
    return std::nullopt;
}

[[nodiscard]] std::uint16_t fixedPageTokens(Variant variant) noexcept
{
    switch (variant) {
    case Variant::Fixed8:
        return 8;
    case Variant::Fixed16:
        return 16;
    case Variant::Fixed32:
        return 32;
    case Variant::Fixed64:
        return 64;
    case Variant::Heterogeneous:
        return 0;
    }
    return 0;
}

[[nodiscard]] bool parseUnsigned(
    std::string_view encoded,
    std::uint32_t& value) noexcept
{
    auto const parsed = std::from_chars(
        encoded.data(), encoded.data() + encoded.size(), value
    );
    return !encoded.empty() && parsed.ec == std::errc{}
        && parsed.ptr == encoded.data() + encoded.size();
}

[[nodiscard]] bool parseOptions(int argc, char** argv, Options& options)
{
    for (int index = 1; index < argc;) {
        std::string const key = argv[index];
        if (key == "--profile") {
            options.profile = true;
            ++index;
            continue;
        }
        if (index + 1 >= argc) {
            return false;
        }
        std::string const value = argv[index + 1];
        if (key == "--manifest") {
            options.manifest = value;
        } else if (key == "--weights") {
            options.weights = value;
        } else if (key == "--output") {
            options.output = value;
        } else if (key == "--git-commit") {
            options.git_commit = value;
        } else if (key == "--variant") {
            std::optional<Variant> const parsed = parseVariant(value);
            if (!parsed.has_value()) {
                return false;
            }
            options.variant = *parsed;
        } else if (key == "--case") {
            if (value.empty()) {
                return false;
            }
            options.case_name = value;
        } else if (key == "--warmup") {
            if (!parseUnsigned(value, options.warmup)) {
                return false;
            }
        } else if (key == "--iterations") {
            if (!parseUnsigned(value, options.iterations)) {
                return false;
            }
        } else if (key == "--kv-capacity-tokens") {
            if (!parseUnsigned(value, options.kv_capacity_tokens)) {
                return false;
            }
        } else if (key == "--capacity-probe-tokens") {
            if (!parseUnsigned(value, options.capacity_probe_tokens)) {
                return false;
            }
        } else {
            return false;
        }
        index += 2;
    }
    bool const valid_iterations = options.profile
        ? options.iterations >= 1
        : options.iterations >= 3;
    bool const valid_profile_selection =
        !options.profile || options.case_name.has_value();
    return !options.manifest.empty() && !options.weights.empty()
        && !options.output.empty() && valid_iterations
        && valid_profile_selection
        && options.kv_capacity_tokens >= 128
        && options.kv_capacity_tokens % 128 == 0
        && options.capacity_probe_tokens >= 128
        && options.capacity_probe_tokens % 128 == 0;
}

[[nodiscard]] std::uint64_t elapsedNanoseconds(
    Clock::time_point begin,
    Clock::time_point end) noexcept
{
    auto const elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end - begin
    ).count();
    return elapsed > 0 ? static_cast<std::uint64_t>(elapsed) : 0;
}

[[nodiscard]] std::uint64_t percentile(
    std::vector<std::uint64_t> values,
    double quantile)
{
    if (values.empty()) {
        return 0;
    }
    std::sort(values.begin(), values.end());
    std::size_t const index = std::min(
        values.size() - 1,
        static_cast<std::size_t>(std::ceil(
            quantile * static_cast<double>(values.size())
        )) - 1
    );
    return values[index];
}

[[nodiscard]] std::vector<std::uint32_t> makePrompt(
    std::uint32_t length,
    std::uint32_t request_index,
    std::uint32_t vocabulary_size)
{
    std::vector<std::uint32_t> result(length);
    for (std::uint32_t index = 0; index < length; ++index) {
        result[index] = (
            index * 37U + request_index * 17U + 1U
        ) % vocabulary_size;
    }
    return result;
}

[[nodiscard]] std::vector<CaseSpec> performanceCases()
{
    std::vector<CaseSpec> result;
    for (std::uint32_t concurrency : {1U, 2U, 4U}) {
        result.push_back(CaseSpec{
            "decode_short_c" + std::to_string(concurrency),
            concurrency,
            std::vector<std::uint32_t>(concurrency, 32),
            32,
            false,
            false,
        });
        result.push_back(CaseSpec{
            "decode_long_prompt_c" + std::to_string(concurrency),
            concurrency,
            std::vector<std::uint32_t>(concurrency, 128),
            32,
            false,
            false,
        });
    }
    result.push_back(CaseSpec{
        "mixed_c4", 4, {32, 128, 32, 128}, 32, false, false,
    });
    result.push_back(CaseSpec{
        "fault_c4", 4, {8, 8, 8, 8}, 8, true, false,
    });
    return result;
}

[[nodiscard]] CaseSpec capacityCase(std::uint32_t capacity_tokens)
{
    std::uint32_t const requests = capacity_tokens / 32 + 1;
    return CaseSpec{
        "capacity",
        requests,
        std::vector<std::uint32_t>(requests, 32),
        1,
        false,
        true,
    };
}

[[nodiscard]] CaseSelection selectCases(
    std::optional<std::string> const& selected_case)
{
    CaseSelection result;
    for (CaseSpec& spec : performanceCases()) {
        if (!selected_case.has_value() || spec.name == *selected_case) {
            result.performance.push_back(std::move(spec));
        }
    }
    result.capacity = !selected_case.has_value()
        || *selected_case == "capacity";
    return result;
}

[[nodiscard]] std::unique_ptr<EngineKvBackend> createBackend(
    Variant variant,
    EngineKvConfig config,
    std::uint32_t capacity_tokens)
{
    if (variant == Variant::Heterogeneous) {
        std::uint32_t const micro_pages = capacity_tokens / 16;
        std::uint32_t const extent_pages = capacity_tokens / 128;
        return createHeterogeneousCudaEngineKvBackend(
            config, micro_pages, extent_pages
        );
    }
    std::uint16_t const page_tokens = fixedPageTokens(variant);
    std::uint32_t const page_capacity =
        (capacity_tokens + page_tokens - 1) / page_tokens;
    return createFixedCudaEngineKvBackend(
        config, page_tokens, page_capacity
    );
}

[[nodiscard]] bool initializeContext(
    Options const& options,
    WeightManifest const& manifest,
    std::uint32_t capacity_tokens,
    SuiteContext& context,
    std::string& error)
{
    std::size_t free_before = 0;
    std::size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_before, &total_bytes) != cudaSuccess) {
        error = "cudaMemGetInfo before context failed";
        return false;
    }
    if (cudaStreamCreate(&context.stream) != cudaSuccess) {
        error = "cudaStreamCreate failed";
        return false;
    }
    TinyLlamaConfig const config = manifest.config;
    std::size_t const attention_workspace =
        static_cast<std::size_t>(config.attention_head_count)
        * config.max_position_embeddings * sizeof(float);
    EngineKvConfig const kv_config{
        KvLayout{
            config.layer_count,
            config.kv_head_count,
            config.head_dimension,
        },
        config.attention_head_count,
        attention_workspace,
    };
    context.backend = createBackend(
        options.variant, kv_config, capacity_tokens
    );
    if (context.backend == nullptr) {
        error = "create Engine KV backend failed";
        return false;
    }
    context.model = createCudaTinyLlamaModelRunner(
        options.manifest,
        options.weights,
        *context.backend,
        reinterpret_cast<EngineStream>(context.stream)
    );
    if (!context.model.ok()) {
        error = context.model.status.detail;
        return false;
    }
    EngineKvBackendSnapshot const snapshot = context.backend->snapshot();
    context.storage_bytes = snapshot.storage_reserved_bytes;
    context.weight_bytes = context.model.runner->deviceWeightBytes();
    context.workspace_bytes = context.model.runner->deviceWorkspaceBytes();
    std::size_t free_after = 0;
    if (cudaMemGetInfo(&free_after, &total_bytes) != cudaSuccess) {
        error = "cudaMemGetInfo after context failed";
        return false;
    }
    context.loaded_gpu_bytes = free_before >= free_after
        ? free_before - free_after
        : 0;
    context.capacity_tokens = capacity_tokens;
    return true;
}

void destroyContext(SuiteContext& context) noexcept
{
    context.model.runner.reset();
    context.backend.reset();
    if (context.stream != nullptr) {
        static_cast<void>(cudaStreamDestroy(context.stream));
        context.stream = nullptr;
    }
}

[[nodiscard]] RunEvidence runOnce(
    SuiteContext& context,
    CaseSpec const& spec,
    std::uint64_t id_base,
    bool collect_resources,
    ResourceProbe* resource_probe,
    bool profile)
{
    RunEvidence result;
    TinyLlamaConfig const config = context.model.runner->generationConfig();
    std::map<RequestId, std::uint32_t> prompt_lengths;
    for (std::uint32_t index = 0; index < spec.concurrency; ++index) {
        prompt_lengths.emplace(
            id_base + index + 1,
            spec.prompt_lengths[index % spec.prompt_lengths.size()]
        );
    }
    ProfilingModelRunner profiling_runner(
        *context.model.runner, std::move(prompt_lengths)
    );
    GenerationModelRunner& scheduler_runner = profile
        ? static_cast<GenerationModelRunner&>(profiling_runner)
        : static_cast<GenerationModelRunner&>(*context.model.runner);
    IterationSchedulerRuntime scheduler(
        *context.backend,
        scheduler_runner,
        IterationSchedulerConfig{
            spec.concurrency,
            spec.concurrency * 16,
            context.capacity_tokens,
            16,
        }
    );
    Clock::time_point const started = Clock::now();
    for (std::uint32_t index = 0; index < spec.concurrency; ++index) {
        std::uint32_t const length = spec.prompt_lengths[
            index % spec.prompt_lengths.size()
        ];
        GenerationRequest request{
            id_base + index + 1,
            makePrompt(length, index, config.vocabulary_size),
            SamplingConfig{spec.output_length, config.eos_token_id},
            {},
        };
        SchedulerAdmissionResult const admitted = scheduler.submit(
            std::move(request)
        );
        if (admitted.ok()) {
            ++result.accepted;
        } else {
            ++result.rejected;
        }
    }
    if (spec.fault) {
        static_cast<void>(injectCudaEngineFailureOnce(
            *context.backend, CudaFailurePoint::Submission
        ));
    }

    std::size_t baseline_free = 0;
    std::size_t total_bytes = 0;
    std::size_t minimum_free = std::numeric_limits<std::size_t>::max();
    EngineKvBackendSnapshot before = context.backend->snapshot();
    if (collect_resources) {
        static_cast<void>(cudaMemGetInfo(&baseline_free, &total_bytes));
        minimum_free = baseline_free;
    }
    while (!scheduler.idle()) {
        static_cast<void>(scheduler.runIteration());
        if (collect_resources && resource_probe != nullptr) {
            EngineKvBackendSnapshot const snapshot =
                context.backend->snapshot();
            resource_probe->peak_committed_tokens = std::max(
                resource_probe->peak_committed_tokens,
                snapshot.committed_token_count
            );
            std::uint64_t const reserved =
                static_cast<std::uint64_t>(
                    snapshot.allocated_primary_pages
                ) * snapshot.primary_page_tokens
                + static_cast<std::uint64_t>(
                    snapshot.allocated_secondary_pages
                ) * snapshot.secondary_page_tokens;
            resource_probe->peak_reserved_tokens = std::max(
                resource_probe->peak_reserved_tokens, reserved
            );
            resource_probe->peak_fragmentation_tokens = std::max(
                resource_probe->peak_fragmentation_tokens,
                reserved >= snapshot.committed_token_count
                    ? reserved - snapshot.committed_token_count
                    : 0
            );
            resource_probe->peak_primary_pages = std::max(
                resource_probe->peak_primary_pages,
                snapshot.allocated_primary_pages
            );
            resource_probe->peak_secondary_pages = std::max(
                resource_probe->peak_secondary_pages,
                snapshot.allocated_secondary_pages
            );
            std::size_t free_bytes = 0;
            if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
                minimum_free = std::min(minimum_free, free_bytes);
            }
        }
    }
    std::vector<GenerationTerminal> terminals = scheduler.takeTerminals();
    Clock::time_point const finished = Clock::now();
    result.elapsed_ns = elapsedNanoseconds(started, finished);
    IterationSchedulerSnapshot const scheduler_state = scheduler.snapshot();
    result.model_forward_tokens = scheduler_state.model_forward_tokens;
    result.model_forward_batches = scheduler_state.model_forward_batches;
    result.prefill_tokens = scheduler_state.prefill_tokens;
    result.decode_tokens = scheduler_state.decode_tokens;
    result.requests.reserve(terminals.size());
    for (GenerationTerminal const& terminal : terminals) {
        RequestEvidence evidence;
        evidence.request_id = terminal.request_id;
        std::uint32_t const index = static_cast<std::uint32_t>(
            terminal.request_id - id_base - 1
        );
        std::uint32_t const length = spec.prompt_lengths[
            index % spec.prompt_lengths.size()
        ];
        evidence.input = makePrompt(
            length, index, config.vocabulary_size
        );
        evidence.output = terminal.output_token_ids;
        evidence.reason = terminal.reason;
        evidence.error = terminal.error;
        evidence.ttft_ns = terminal.metrics.ttft_ns;
        evidence.tpot_ns = terminal.metrics.tpot_ns;
        evidence.e2e_ns = terminal.metrics.e2e_ns;
        result.output_tokens += terminal.output_token_ids.size();
        if (terminal.ok()) {
            ++result.completed;
        } else {
            ++result.failed;
        }
        result.requests.push_back(std::move(evidence));
    }
    std::sort(
        result.requests.begin(), result.requests.end(),
        [](RequestEvidence const& left, RequestEvidence const& right) {
            return left.request_id < right.request_id;
        }
    );
    EngineKvBackendSnapshot const after = context.backend->snapshot();
    result.batched_attention_submissions =
        after.batched_attention_submissions
        - before.batched_attention_submissions;
    result.batched_attention_lanes = after.batched_attention_lanes
        - before.batched_attention_lanes;
    result.automatic_promotion_attempts =
        after.automatic_promotion_attempts
        - before.automatic_promotion_attempts;
    result.automatic_promotion_successes =
        after.automatic_promotion_successes
        - before.automatic_promotion_successes;
    result.automatic_promotion_skips =
        after.automatic_promotion_skips
        - before.automatic_promotion_skips;
    result.automatic_promotion_failures =
        after.automatic_promotion_failures
        - before.automatic_promotion_failures;
    result.resources_reclaimed = scheduler_state.activeCount() == 0
        && scheduler_state.reserved_kv_tokens == 0
        && after.request_count == 0
        && after.active_transaction_count == 0
        && after.committed_token_count == 0
        && after.allocated_primary_pages == 0
        && after.allocated_secondary_pages == 0
        && context.backend->checkInvariants();
    if (spec.fault) {
        result.expected_outcome = result.failed == 1
            && result.completed + result.failed == result.accepted;
    } else if (spec.capacity) {
        result.expected_outcome = result.rejected + result.failed >= 1
            && result.completed + result.failed == result.accepted;
    } else {
        result.expected_outcome = result.rejected == 0
            && result.failed == 0
            && result.completed == spec.concurrency;
    }
    result.expected_outcome = result.expected_outcome
        && result.resources_reclaimed;

    if (collect_resources && resource_probe != nullptr) {
        resource_probe->primary_allocations =
            after.successful_primary_allocations
            - before.successful_primary_allocations;
        resource_probe->secondary_allocations =
            after.successful_secondary_allocations
            - before.successful_secondary_allocations;
        resource_probe->failed_primary_allocations =
            after.failed_primary_allocations - before.failed_primary_allocations;
        resource_probe->failed_secondary_allocations =
            after.failed_secondary_allocations
            - before.failed_secondary_allocations;
        if (minimum_free != std::numeric_limits<std::size_t>::max()
            && baseline_free >= minimum_free) {
            resource_probe->peak_dynamic_gpu_bytes = baseline_free
                - minimum_free;
        }
    }
    return result;
}

[[nodiscard]] bool sameOutputs(
    RunEvidence const& left,
    RunEvidence const& right)
{
    if (left.requests.size() != right.requests.size()) {
        return false;
    }
    for (std::size_t index = 0; index < left.requests.size(); ++index) {
        if (left.requests[index].input != right.requests[index].input
            || left.requests[index].output != right.requests[index].output
            || left.requests[index].reason != right.requests[index].reason
            || left.requests[index].error != right.requests[index].error) {
            return false;
        }
    }
    return true;
}

void summarize(CaseEvidence& result)
{
    std::vector<std::uint64_t> ttft;
    std::vector<std::uint64_t> tpot;
    std::vector<std::uint64_t> e2e;
    double requests_rate = 0.0;
    double tokens_rate = 0.0;
    double goodput_rate = 0.0;
    for (RunEvidence const& run : result.runs) {
        double const seconds = static_cast<double>(run.elapsed_ns) / 1.0e9;
        if (seconds > 0.0) {
            requests_rate += static_cast<double>(run.accepted) / seconds;
            tokens_rate += static_cast<double>(run.output_tokens) / seconds;
            goodput_rate += static_cast<double>(run.completed) / seconds;
        }
        for (RequestEvidence const& request : run.requests) {
            if (request.error == GenerationError::None) {
                ttft.push_back(request.ttft_ns);
                tpot.push_back(request.tpot_ns);
                e2e.push_back(request.e2e_ns);
            }
        }
    }
    double const divisor = static_cast<double>(result.runs.size());
    result.requests_per_second = requests_rate / divisor;
    result.output_tokens_per_second = tokens_rate / divisor;
    result.goodput_requests_per_second = goodput_rate / divisor;
    result.ttft_p50_ns = percentile(ttft, 0.50);
    result.ttft_p95_ns = percentile(ttft, 0.95);
    result.ttft_p99_ns = percentile(ttft, 0.99);
    result.tpot_p50_ns = percentile(tpot, 0.50);
    result.tpot_p95_ns = percentile(tpot, 0.95);
    result.tpot_p99_ns = percentile(tpot, 0.99);
    result.e2e_p50_ns = percentile(e2e, 0.50);
    result.e2e_p95_ns = percentile(e2e, 0.95);
    result.e2e_p99_ns = percentile(e2e, 0.99);
}

[[nodiscard]] CaseEvidence runCase(
    SuiteContext& context,
    CaseSpec spec,
    std::uint32_t warmup,
    std::uint32_t iterations,
    std::uint64_t id_namespace,
    bool profile)
{
    NvtxRange const case_range(profile, "case/" + spec.name);
    for (std::uint32_t index = 0; index < warmup; ++index) {
        NvtxRange const warmup_range(
            profile, "warmup/" + std::to_string(index)
        );
        static_cast<void>(runOnce(
            context,
            spec,
            id_namespace + static_cast<std::uint64_t>(index) * 10'000,
            false,
            nullptr,
            profile
        ));
    }

    CaseEvidence result;
    result.spec = std::move(spec);
    for (std::uint32_t index = 0; index < iterations; ++index) {
        NvtxRange const iteration_range(
            profile, "iteration/" + std::to_string(index)
        );
        result.runs.push_back(runOnce(
            context,
            result.spec,
            id_namespace + 1'000'000
                + static_cast<std::uint64_t>(index) * 10'000,
            false,
            nullptr,
            profile
        ));
    }
    RunEvidence const reference = result.runs.front();
    for (RunEvidence const& run : result.runs) {
        result.outputs_consistent = result.outputs_consistent
            && sameOutputs(reference, run);
        result.passed = result.passed || run.expected_outcome;
    }
    result.passed = result.outputs_consistent
        && std::all_of(
            result.runs.begin(), result.runs.end(),
            [](RunEvidence const& run) { return run.expected_outcome; }
        );
    if (!profile) {
        static_cast<void>(runOnce(
            context,
            result.spec,
            id_namespace + 9'000'000,
            true,
            &result.resources,
            false
        ));
    }
    summarize(result);
    return result;
}

void writeNumbers(
    std::ostream& output,
    std::vector<std::uint32_t> const& values)
{
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << values[index];
    }
    output << ']';
}

void writeRequest(std::ostream& output, RequestEvidence const& request)
{
    output << "{\"request_id\":" << request.request_id
        << ",\"input_tokens\":";
    writeNumbers(output, request.input);
    output << ",\"output_tokens\":";
    writeNumbers(output, request.output);
    output << ",\"terminal_reason\":\"" << toString(request.reason)
        << "\",\"error\":\"" << toString(request.error)
        << "\",\"ttft_ns\":" << request.ttft_ns
        << ",\"tpot_ns\":" << request.tpot_ns
        << ",\"e2e_ns\":" << request.e2e_ns << '}';
}

void writeCase(std::ostream& output, CaseEvidence const& value)
{
    output << "    {\"name\":\"" << value.spec.name
        << "\",\"concurrency\":" << value.spec.concurrency
        << ",\"prompt_lengths\":";
    writeNumbers(output, value.spec.prompt_lengths);
    output << ",\"output_length\":" << value.spec.output_length
        << ",\"fault_workload\":"
        << (value.spec.fault ? "true" : "false")
        << ",\"capacity_workload\":"
        << (value.spec.capacity ? "true" : "false")
        << ",\"passed\":" << (value.passed ? "true" : "false")
        << ",\"outputs_consistent\":"
        << (value.outputs_consistent ? "true" : "false")
        << ",\"summary\":{"
        << "\"ttft_p50_ns\":" << value.ttft_p50_ns
        << ",\"ttft_p95_ns\":" << value.ttft_p95_ns
        << ",\"ttft_p99_ns\":" << value.ttft_p99_ns
        << ",\"tpot_p50_ns\":" << value.tpot_p50_ns
        << ",\"tpot_p95_ns\":" << value.tpot_p95_ns
        << ",\"tpot_p99_ns\":" << value.tpot_p99_ns
        << ",\"e2e_p50_ns\":" << value.e2e_p50_ns
        << ",\"e2e_p95_ns\":" << value.e2e_p95_ns
        << ",\"e2e_p99_ns\":" << value.e2e_p99_ns
        << ",\"requests_per_second\":" << value.requests_per_second
        << ",\"output_tokens_per_second\":"
        << value.output_tokens_per_second
        << ",\"goodput_requests_per_second\":"
        << value.goodput_requests_per_second << "},\"resources\":{"
        << "\"peak_committed_tokens\":"
        << value.resources.peak_committed_tokens
        << ",\"peak_reserved_tokens\":"
        << value.resources.peak_reserved_tokens
        << ",\"peak_fragmentation_tokens\":"
        << value.resources.peak_fragmentation_tokens
        << ",\"peak_primary_pages\":"
        << value.resources.peak_primary_pages
        << ",\"peak_secondary_pages\":"
        << value.resources.peak_secondary_pages
        << ",\"primary_allocations\":"
        << value.resources.primary_allocations
        << ",\"secondary_allocations\":"
        << value.resources.secondary_allocations
        << ",\"failed_primary_allocations\":"
        << value.resources.failed_primary_allocations
        << ",\"failed_secondary_allocations\":"
        << value.resources.failed_secondary_allocations
        << ",\"peak_dynamic_gpu_bytes\":"
        << value.resources.peak_dynamic_gpu_bytes << "},\"runs\":[";
    for (std::size_t run_index = 0;
         run_index < value.runs.size();
         ++run_index) {
        RunEvidence const& run = value.runs[run_index];
        output << "{\"elapsed_ns\":" << run.elapsed_ns
            << ",\"output_tokens\":" << run.output_tokens
            << ",\"model_forward_tokens\":" << run.model_forward_tokens
            << ",\"model_forward_batches\":" << run.model_forward_batches
            << ",\"batched_attention_submissions\":"
            << run.batched_attention_submissions
            << ",\"batched_attention_lanes\":"
            << run.batched_attention_lanes
            << ",\"automatic_promotion_attempts\":"
            << run.automatic_promotion_attempts
            << ",\"automatic_promotion_successes\":"
            << run.automatic_promotion_successes
            << ",\"automatic_promotion_skips\":"
            << run.automatic_promotion_skips
            << ",\"automatic_promotion_failures\":"
            << run.automatic_promotion_failures
            << ",\"average_attention_batch_size\":"
            << (run.batched_attention_submissions == 0 ? 0.0
                : static_cast<double>(run.batched_attention_lanes)
                    / static_cast<double>(
                        run.batched_attention_submissions
                    ))
            << ",\"average_batch_size\":"
            << (run.model_forward_batches == 0 ? 0.0
                : static_cast<double>(run.model_forward_tokens)
                    / static_cast<double>(run.model_forward_batches))
            << ",\"prefill_tokens\":" << run.prefill_tokens
            << ",\"decode_tokens\":" << run.decode_tokens
            << ",\"accepted\":" << run.accepted
            << ",\"rejected\":" << run.rejected
            << ",\"completed\":" << run.completed
            << ",\"failed\":" << run.failed
            << ",\"resources_reclaimed\":"
            << (run.resources_reclaimed ? "true" : "false")
            << ",\"expected_outcome\":"
            << (run.expected_outcome ? "true" : "false")
            << ",\"requests\":[";
        for (std::size_t request_index = 0;
             request_index < run.requests.size();
             ++request_index) {
            writeRequest(output, run.requests[request_index]);
            if (request_index + 1 != run.requests.size()) {
                output << ',';
            }
        }
        output << "]}";
        if (run_index + 1 != value.runs.size()) {
            output << ',';
        }
    }
    output << "]}";
}

[[nodiscard]] int fail(std::string const& message)
{
    std::cerr << "[FAILED] " << message << '\n';
    return 1;
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    if (!parseOptions(argc, argv, options)) {
        return fail("usage: --manifest PATH --weights PATH --variant "
            "hetero|fixed_8|fixed_16|fixed_32|fixed_64 --output PATH "
            "[--warmup N] --iterations N>=3 [--git-commit SHA] "
            "[--kv-capacity-tokens N] [--capacity-probe-tokens N] "
            "[--case NAME] [--profile (allows iterations>=1 and "
            "requires --case)]");
    }
#if !defined(KIM_KV_HAS_NVTX)
    if (options.profile) {
        return fail("--profile requires an NVTX-enabled build");
    }
#endif
    CaseSelection const selection = selectCases(options.case_name);
    if (selection.empty()) {
        return fail("unknown case: " + *options.case_name);
    }
    WeightManifestLoadResult loaded = loadWeightManifest(options.manifest);
    if (!loaded.ok()) {
        return fail("manifest: " + loaded.status.detail);
    }

    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) {
        return fail("cudaGetDeviceProperties");
    }
    int runtime_version = 0;
    static_cast<void>(cudaRuntimeGetVersion(&runtime_version));

    std::string error;
    std::vector<CaseEvidence> cases;
    std::uint64_t id_namespace = 100'000'000;
    std::uint64_t performance_storage_bytes = 0;
    std::uint64_t capacity_storage_bytes = 0;
    std::uint64_t weight_bytes = 0;
    std::uint64_t workspace_bytes = 0;
    std::uint64_t loaded_gpu_bytes = 0;

    if (!selection.performance.empty()) {
        SuiteContext performance;
        if (!initializeContext(
                options,
                loaded.manifest,
                options.kv_capacity_tokens,
                performance,
                error)) {
            destroyContext(performance);
            return fail(error);
        }
        for (CaseSpec const& spec : selection.performance) {
            cases.push_back(runCase(
                performance,
                spec,
                spec.fault ? 0 : options.warmup,
                options.iterations,
                id_namespace,
                options.profile
            ));
            id_namespace += 20'000'000;
        }
        performance_storage_bytes = performance.storage_bytes;
        weight_bytes = performance.weight_bytes;
        workspace_bytes = performance.workspace_bytes;
        loaded_gpu_bytes = performance.loaded_gpu_bytes;
        destroyContext(performance);
    }

    if (selection.capacity) {
        SuiteContext capacity;
        if (!initializeContext(
                options,
                loaded.manifest,
                options.capacity_probe_tokens,
                capacity,
                error)) {
            destroyContext(capacity);
            return fail(error);
        }
        cases.push_back(runCase(
            capacity,
            capacityCase(options.capacity_probe_tokens),
            0,
            options.iterations,
            id_namespace,
            options.profile
        ));
        capacity_storage_bytes = capacity.storage_bytes;
        if (selection.performance.empty()) {
            weight_bytes = capacity.weight_bytes;
            workspace_bytes = capacity.workspace_bytes;
            loaded_gpu_bytes = capacity.loaded_gpu_bytes;
        }
        destroyContext(capacity);
    }

    bool const passed = std::all_of(
        cases.begin(), cases.end(),
        [](CaseEvidence const& value) { return value.passed; }
    );
    std::ofstream output(options.output);
    if (!output) {
        return fail("open output JSON");
    }
    output << std::setprecision(10)
        << "{\n  \"schema_version\":1,\n"
        << "  \"suite\":\"tinyllama_e5_end_to_end\",\n"
        << "  \"successful\":" << (passed ? "true" : "false")
        << ",\n  \"variant\":\"" << variantName(options.variant)
        << "\",\n  \"git_commit\":\"" << options.git_commit
        << "\",\n  \"checkpoint\":\"" << loaded.manifest.checkpoint
        << "\",\n  \"checkpoint_revision\":\""
        << loaded.manifest.checkpoint_revision
        << "\",\n  \"environment\":{\"gpu\":\""
        << properties.name << "\",\"cuda_runtime\":" << runtime_version
        << "},\n  \"config\":{\"warmup\":" << options.warmup
        << ",\"iterations\":" << options.iterations
        << ",\"profile\":" << (options.profile ? "true" : "false")
        << ",\"selected_case\":\""
        << (options.case_name.has_value() ? *options.case_name : "all")
        << "\""
        << ",\"kv_capacity_tokens\":" << options.kv_capacity_tokens
        << ",\"capacity_probe_tokens\":"
        << options.capacity_probe_tokens
        << ",\"performance_storage_bytes\":"
        << performance_storage_bytes
        << ",\"capacity_storage_bytes\":" << capacity_storage_bytes
        << ",\"model_weight_bytes\":" << weight_bytes
        << ",\"model_workspace_bytes\":" << workspace_bytes
        << ",\"logical_static_gpu_bytes\":"
        << performance_storage_bytes + weight_bytes + workspace_bytes
        << ",\"loaded_gpu_bytes\":" << loaded_gpu_bytes
        << "},\n  \"cases\":[\n";
    for (std::size_t index = 0; index < cases.size(); ++index) {
        writeCase(output, cases[index]);
        output << (index + 1 == cases.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    output.close();
    if (!passed) {
        return fail("one or more E5 cases failed");
    }
    std::cout << "E5 report: " << options.output << '\n';
    return 0;
}
