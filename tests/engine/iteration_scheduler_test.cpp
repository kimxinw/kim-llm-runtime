#include "kim-kv/engine/iteration_scheduler.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using namespace kimkvcache;

int failures = 0;

void expect(bool condition, std::string const& message)
{
    if (!condition) {
        std::cerr << "[FAILED] " << message << '\n';
        ++failures;
    }
}

class FakeKvBackend final : public EngineKvBackend {
public:
    std::unordered_set<RequestId> fail_create{};

    [[nodiscard]] EngineKvBackendKind kind() const noexcept override
    {
        return EngineKvBackendKind::Heterogeneous;
    }

    [[nodiscard]] EngineKvConfig config() const noexcept override
    {
        return {};
    }

    [[nodiscard]] EngineKvStatus createRequest(RequestId request_id) override
    {
        if (fail_create.count(request_id) != 0) {
            return {EngineKvError::ResourceExhausted};
        }
        return requests_.insert(request_id).second
            ? EngineKvStatus{}
            : EngineKvStatus{EngineKvError::RequestAlreadyExists};
    }

    [[nodiscard]] EngineKvStatus forkRequest(RequestId, RequestId) override
    {
        return {EngineKvError::InvalidState};
    }

    [[nodiscard]] EngineKvStatus releaseRequest(RequestId request_id) override
    {
        return requests_.erase(request_id) == 1
            ? EngineKvStatus{}
            : EngineKvStatus{EngineKvError::RequestNotFound};
    }

    [[nodiscard]] TokenReserveResult reserveToken(
        ReserveTokenRequest const&) override
    {
        TokenReserveResult result;
        result.status = {EngineKvError::InvalidState};
        return result;
    }

    [[nodiscard]] EngineKvBackendSnapshot snapshot() const override
    {
        return {requests_.size(), 0, 0};
    }

    [[nodiscard]] bool checkInvariants() const override
    {
        return true;
    }

private:
    std::unordered_set<RequestId> requests_{};
};

class FakeModelRunner final : public GenerationModelRunner {
public:
    TinyLlamaConfig model_config{
        8, 16, 2, 4, 2, 2, 256, 256, 1, 255,
        1.0e-5F, 10000.0F, false,
    };
    std::unordered_set<RequestId> fail_requests{};
    std::unordered_map<RequestId, std::uint64_t> calls{};
    std::vector<std::size_t> batch_sizes{};
    std::vector<std::vector<std::size_t>> chunk_sizes{};
    IterationSchedulerRuntime* stop_scheduler{nullptr};
    RequestId stop_request{kInvalidRequestId};

    [[nodiscard]] TinyLlamaConfig generationConfig() const noexcept override
    {
        return model_config;
    }

    [[nodiscard]] GenerationStepResult generationForwardToken(
        RequestId request_id,
        std::uint32_t token_id,
        std::uint32_t expected_position) override
    {
        ++calls[request_id];
        if (request_id == stop_request && stop_scheduler != nullptr) {
            stop_scheduler->stop();
        }
        if (fail_requests.count(request_id) != 0) {
            return {false, 0, "injected request-local failure"};
        }
        return {
            true,
            static_cast<std::uint32_t>(
                (token_id * 7 + expected_position * 3 + 1) % 251
            ),
            {},
        };
    }

    [[nodiscard]] GenerationBatchResult generationForwardBatch(
        std::vector<GenerationBatchItem> const& batch) override
    {
        batch_sizes.push_back(batch.size());
        return GenerationModelRunner::generationForwardBatch(batch);
    }

    [[nodiscard]] GenerationBatchResult generationForwardChunks(
        std::vector<GenerationChunkItem> const& chunks) override
    {
        std::vector<std::size_t> sizes;
        sizes.reserve(chunks.size());
        for (GenerationChunkItem const& chunk : chunks) {
            sizes.push_back(chunk.token_ids.size());
        }
        chunk_sizes.push_back(std::move(sizes));
        return GenerationModelRunner::generationForwardChunks(chunks);
    }
};

std::vector<std::uint32_t> prompt(std::uint32_t length)
{
    std::vector<std::uint32_t> tokens(length);
    for (std::uint32_t index = 0; index < length; ++index) {
        tokens[index] = (index * 11 + 3) % 251;
    }
    return tokens;
}

GenerationRequest request(
    RequestId request_id,
    std::uint32_t input_length,
    std::uint32_t output_length)
{
    return {
        request_id,
        prompt(input_length),
        SamplingConfig{output_length, 255},
        {},
    };
}

std::vector<std::uint32_t> referenceGreedy(
    std::vector<std::uint32_t> const& input,
    std::uint32_t output_length)
{
    std::uint32_t next = 0;
    for (std::uint32_t position = 0; position < input.size(); ++position) {
        next = (input[position] * 7 + position * 3 + 1) % 251;
    }
    std::vector<std::uint32_t> output;
    for (std::uint32_t generated = 0; generated < output_length; ++generated) {
        output.push_back(next);
        if (generated + 1 != output_length) {
            std::uint32_t const position = static_cast<std::uint32_t>(
                input.size()
            ) + generated;
            next = (next * 7 + position * 3 + 1) % 251;
        }
    }
    return output;
}

std::unordered_map<RequestId, GenerationTerminal> byId(
    std::vector<GenerationTerminal> terminals)
{
    std::unordered_map<RequestId, GenerationTerminal> result;
    for (GenerationTerminal& terminal : terminals) {
        result.emplace(terminal.request_id, std::move(terminal));
    }
    return result;
}

void expectReclaimed(
    FakeKvBackend const& backend,
    IterationSchedulerRuntime const& scheduler,
    std::string const& context)
{
    EngineKvBackendSnapshot const kv = backend.snapshot();
    IterationSchedulerSnapshot const state = scheduler.snapshot();
    expect(kv.request_count == 0, context + " releases KV requests");
    expect(kv.active_transaction_count == 0
        && kv.committed_token_count == 0,
        context + " restores KV counters");
    expect(state.activeCount() == 0 && state.reserved_kv_tokens == 0,
        context + " restores scheduler budgets");
    expect(backend.checkInvariants(), context + " preserves invariants");
}

void testConcurrencyAndReference()
{
    for (std::uint32_t concurrency : {1U, 2U, 4U}) {
        FakeKvBackend backend;
        FakeModelRunner runner;
        IterationSchedulerRuntime scheduler(
            backend, runner, {concurrency, concurrency, 4096}
        );
        std::unordered_map<RequestId, GenerationRequest> requests;
        for (std::uint32_t index = 0; index < concurrency; ++index) {
            GenerationRequest value = request(
                100 + index, 2 + index, 4 + index
            );
            requests.emplace(value.request_id, value);
            expect(scheduler.submit(std::move(value)).ok(),
                "c1/c2/c4 request is accepted");
        }
        auto terminals = byId(scheduler.drain());
        expect(terminals.size() == concurrency,
            "c1/c2/c4 produces one terminal per request");
        for (auto const& entry : requests) {
            auto const found = terminals.find(entry.first);
            expect(found != terminals.end() && found->second.ok(),
                "c1/c2/c4 terminal succeeds");
            if (found != terminals.end()) {
                expect(found->second.output_token_ids == referenceGreedy(
                    entry.second.prompt_token_ids,
                    entry.second.sampling.max_new_tokens
                ), "c1/c2/c4 output matches independent reference");
            }
        }
        expect(scheduler.takeTerminals().empty(),
            "terminal delivery is unique");
        expect(!runner.batch_sizes.empty()
            && runner.batch_sizes.front() == concurrency,
            "c1/c2/c4 is submitted as one model batch");
        expectReclaimed(backend, scheduler, "c1/c2/c4");
    }
}

void testDynamicJoinFifoAndShortExit()
{
    FakeKvBackend backend;
    FakeModelRunner runner;
    IterationSchedulerRuntime scheduler(backend, runner, {4, 2, 4096});
    expect(scheduler.submit(request(1, 6, 2)).ok(), "long request accepted");
    SchedulerIterationResult first = scheduler.runIteration();
    expect(first.model_forward_tokens == 2 && first.prefill_tokens == 2,
        "first iteration advances a two-token prefill chunk");
    expect(scheduler.submit(request(2, 1, 1)).ok(),
        "request dynamically joins at iteration boundary");
    expect(scheduler.submit(request(3, 2, 2)).ok(),
        "third request joins waiting queue");

    SchedulerIterationResult second = scheduler.runIteration();
    expect(second.model_forward_tokens <= 2,
        "max_batched_tokens gates each iteration");
    std::vector<GenerationTerminal> early = scheduler.takeTerminals();
    expect(early.size() == 1 && early.front().request_id == 2,
        "short request exits before long request");
    expect(scheduler.requestState(1) == SchedulerRequestState::Running,
        "long request remains independently running");

    std::vector<GenerationTerminal> rest = scheduler.drain();
    expect(rest.size() == 2, "remaining requests terminate once");
    expectReclaimed(backend, scheduler, "dynamic FIFO");
}

void testChunkedPrefillBudgetAndBatchAccounting()
{
    FakeKvBackend backend;
    FakeModelRunner runner;
    IterationSchedulerRuntime scheduler(backend, runner, {2, 6, 4096, 3});
    expect(scheduler.submit(request(40, 5, 2)).ok(),
        "first chunked-prefill request accepted");
    expect(scheduler.submit(request(41, 5, 2)).ok(),
        "second chunked-prefill request accepted");

    SchedulerIterationResult first = scheduler.runIteration();
    expect(first.model_forward_tokens == 6
        && first.model_forward_batches == 1,
        "chunked prefill consumes one multi-token model batch");
    expect(first.prefill_tokens == 6 && first.decode_tokens == 0,
        "chunked prefill accounts prompt positions separately");
    expect(runner.calls[40] == 3 && runner.calls[41] == 3,
        "prefill chunk advances each request by configured quota");
    expect(runner.batch_sizes.size() == 3
        && std::all_of(
            runner.batch_sizes.begin(), runner.batch_sizes.end(),
            [](std::size_t size) { return size == 2; }
        ), "each prefill wave is a real two-request model batch");
    expect(!runner.chunk_sizes.empty()
        && runner.chunk_sizes.front() == std::vector<std::size_t>({3, 3}),
        "scheduler submits one three-token chunk per request");

    std::vector<GenerationTerminal> terminals = scheduler.drain();
    expect(terminals.size() == 2
        && std::all_of(
            terminals.begin(), terminals.end(),
            [](GenerationTerminal const& terminal) {
                return terminal.ok();
            }
        ), "chunked-prefill requests complete successfully");
    IterationSchedulerSnapshot const snapshot = scheduler.snapshot();
    expect(snapshot.prefill_tokens == 10 && snapshot.decode_tokens == 2,
        "snapshot separates all prefill and decode work");
    expect(snapshot.model_forward_tokens == 12
        && snapshot.model_forward_batches == 3,
        "snapshot exposes cumulative batch utilization");
    expectReclaimed(backend, scheduler, "chunked prefill");
}

void testAdmissionGates()
{
    FakeKvBackend backend;
    FakeModelRunner runner;
    IterationSchedulerRuntime scheduler(backend, runner, {2, 1, 8});

    expect(scheduler.submit(request(10, 2, 2)).ok(),
        "first admission succeeds");
    SchedulerAdmissionResult duplicate = scheduler.submit(request(10, 1, 1));
    expect(duplicate.error == SchedulerAdmissionError::DuplicateRequest,
        "duplicate id is rejected");
    expect(scheduler.submit(request(11, 2, 2)).ok(),
        "second active request succeeds");
    SchedulerAdmissionResult active = scheduler.submit(request(12, 1, 1));
    expect(active.error == SchedulerAdmissionError::MaxActiveRequests,
        "max active admission gate rejects excess request");

    (void)scheduler.drain();
    SchedulerAdmissionResult kv = scheduler.submit(request(13, 8, 2));
    expect(kv.error == SchedulerAdmissionError::KvTokenBudget,
        "KV budget rejects oversized request");
    GenerationRequest invalid = request(14, 1, 1);
    invalid.prompt_token_ids.clear();
    expect(scheduler.submit(std::move(invalid)).error
        == SchedulerAdmissionError::InvalidArgument,
        "invalid request is rejected before acceptance");
    expect(scheduler.requestState(12) == SchedulerRequestState::Rejected
        && scheduler.requestState(13) == SchedulerRequestState::Rejected,
        "rejected requests have explicit state");
    expect(scheduler.snapshot().accepted_count == 2,
        "rejected requests never enter accepted set");
    expectReclaimed(backend, scheduler, "admission gates");
}

void testCancellationFailureAndOomIsolation()
{
    FakeKvBackend backend;
    FakeModelRunner runner;
    runner.fail_requests.insert(22);
    backend.fail_create.insert(23);
    IterationSchedulerRuntime scheduler(backend, runner, {4, 4, 4096});
    for (RequestId id : {20U, 21U, 22U, 23U}) {
        expect(scheduler.submit(request(id, 2, 2)).ok(),
            "isolation request accepted");
    }
    expect(scheduler.cancel(21), "accepted request enters cancelling");
    expect(scheduler.requestState(21) == SchedulerRequestState::Cancelling,
        "cancelling state is observable");

    auto terminals = byId(scheduler.drain());
    expect(terminals.size() == 4, "all isolated requests terminate once");
    expect(terminals[20].ok(), "healthy request survives peer failures");
    expect(terminals[21].reason == GenerationTerminalReason::Cancelled,
        "cancelled request has cancelled terminal");
    expect(terminals[22].error == GenerationError::ModelFailed,
        "model failure remains request-local");
    expect(terminals[23].error == GenerationError::KvBackendFailed,
        "backend OOM remains request-local");
    expectReclaimed(backend, scheduler, "failure isolation");
}

void testRuntimeStopAndPressureStability()
{
    {
        FakeKvBackend backend;
        FakeModelRunner runner;
        IterationSchedulerRuntime scheduler(backend, runner, {4, 4, 4096});
        runner.stop_scheduler = &scheduler;
        runner.stop_request = 30;
        expect(scheduler.submit(request(30, 4, 4)).ok(),
            "stop trigger accepted");
        expect(scheduler.submit(request(31, 4, 4)).ok(),
            "stop peer accepted");
        auto terminals = scheduler.drain();
        expect(terminals.size() == 2,
            "stop drains every accepted request exactly once");
        expect(std::all_of(terminals.begin(), terminals.end(),
            [](GenerationTerminal const& terminal) {
                return terminal.reason
                    == GenerationTerminalReason::RuntimeStopped;
            }), "stop gives all active requests stopped terminals");
        expect(scheduler.submit(request(32, 1, 1)).error
            == SchedulerAdmissionError::RuntimeStopped,
            "stopped runtime rejects new requests");
        expectReclaimed(backend, scheduler, "runtime stop");
    }

    FakeKvBackend backend;
    FakeModelRunner runner;
    IterationSchedulerRuntime scheduler(backend, runner, {4, 4, 4096});
    for (std::uint32_t wave = 0; wave < 25; ++wave) {
        for (std::uint32_t index = 0; index < 4; ++index) {
            expect(scheduler.submit(request(1000 + wave * 4 + index, 2, 2)).ok(),
                "pressure request accepted");
        }
        expect(scheduler.drain().size() == 4,
            "pressure wave returns four unique terminals");
        expectReclaimed(backend, scheduler, "pressure wave");
    }
    expect(scheduler.snapshot().accepted_count == 100,
        "pressure test accounts for all accepted requests");
}

void testDestructorReclaimsActiveRequests()
{
    FakeKvBackend backend;
    FakeModelRunner runner;
    {
        IterationSchedulerRuntime scheduler(backend, runner, {2, 2, 64});
        expect(scheduler.submit(request(2000, 4, 4)).ok(),
            "destructor request accepted");
        (void)scheduler.runIteration();
        expect(backend.snapshot().request_count == 1,
            "request is active before scheduler destruction");
    }
    expect(backend.snapshot().request_count == 0,
        "scheduler destructor releases active KV request");
    expect(backend.checkInvariants(),
        "scheduler destructor preserves backend invariants");
}

} // namespace

int main()
{
    testConcurrencyAndReference();
    testDynamicJoinFifoAndShortExit();
    testChunkedPrefillBudgetAndBatchAccounting();
    testAdmissionGates();
    testCancellationFailureAndOomIsolation();
    testRuntimeStopAndPressureStability();
    testDestructorReclaimsActiveRequests();
    if (failures == 0) {
        std::cout << "Iteration scheduler contract passed\n";
    }
    return failures == 0 ? 0 : 1;
}
