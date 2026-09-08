#include "kim-kv/engine/engine_kv.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace kimkvcache;

static_assert(!std::is_copy_constructible_v<TokenTransaction>);
static_assert(!std::is_copy_assignable_v<TokenTransaction>);
static_assert(std::is_nothrow_move_constructible_v<TokenTransaction>);
static_assert(std::is_nothrow_move_assignable_v<TokenTransaction>);

int failures = 0;

void expect(bool condition, std::string const& message)
{
    if (!condition) {
        std::cerr << "[FAILED] " << message << '\n';
        ++failures;
    }
}

struct TransactionTrace final {
    std::vector<std::string> events{};
    EngineKvError write_failure{EngineKvError::None};
    EngineKvError attention_failure{EngineKvError::None};
    EngineKvError commit_failure{EngineKvError::None};
    std::uint32_t failure_layer{0};
    std::uint32_t rollback_count{0};
    std::uint32_t attention_batch_count{0};
    EngineStream expected_stream{nullptr};
};

class FakeTransactionBackend final : public TokenTransactionBackend {
public:
    explicit FakeTransactionBackend(
        std::shared_ptr<TransactionTrace> trace)
        : trace_(std::move(trace))
    {
    }

    [[nodiscard]] EngineKvStatus writeLayer(
        LayerKvWrite const& write,
        EngineStream stream) override
    {
        expect(stream == trace_->expected_stream, "write must use bound stream");
        trace_->events.push_back("write:" + std::to_string(write.layer));
        if (write.layer == trace_->failure_layer
            && trace_->write_failure != EngineKvError::None) {
            return EngineKvStatus{trace_->write_failure};
        }
        return {};
    }

    [[nodiscard]] EngineKvStatus attendLayer(
        PagedDecodeRequest const& request,
        EngineStream stream) override
    {
        expect(stream == trace_->expected_stream, "attention must use bound stream");
        trace_->events.push_back(
            "attend:" + std::to_string(request.layer)
        );
        if (request.layer == trace_->failure_layer
            && trace_->attention_failure != EngineKvError::None) {
            return EngineKvStatus{trace_->attention_failure};
        }
        return {};
    }

    void attendLayerBatch(PagedDecodeBatch& batch) override
    {
        ++trace_->attention_batch_count;
        TokenTransactionBackend::attendLayerBatch(batch);
    }

    [[nodiscard]] EngineKvStatus commit(
        EngineStream stream) override
    {
        expect(stream == trace_->expected_stream, "commit must use bound stream");
        trace_->events.push_back("commit");
        return EngineKvStatus{trace_->commit_failure};
    }

    void rollback(EngineStream stream) noexcept override
    {
        expect(stream == trace_->expected_stream, "rollback must use bound stream");
        trace_->events.push_back("rollback");
        ++trace_->rollback_count;
    }

private:
    std::shared_ptr<TransactionTrace> trace_;
};

struct DeviceInputs final {
    KvScalar key{1};
    KvScalar value{2};
    KvScalar query{3};
    KvScalar output{0};
    std::array<std::byte, 64> workspace{};
};

[[nodiscard]] TokenTransaction makeTransaction(
    std::shared_ptr<TransactionTrace> const& trace,
    std::uint32_t layer_count = 3,
    TokenTransactionId transaction_id = 7)
{
    return TokenTransaction(
        std::make_unique<FakeTransactionBackend>(trace),
        transaction_id,
        42,
        9,
        layer_count,
        trace->expected_stream
    );
}

[[nodiscard]] LayerKvWrite makeWrite(
    DeviceInputs const& inputs,
    std::uint32_t layer)
{
    return LayerKvWrite{layer, &inputs.key, &inputs.value};
}

[[nodiscard]] PagedDecodeRequest makeAttention(
    DeviceInputs& inputs,
    std::uint32_t layer)
{
    return PagedDecodeRequest{
        layer,
        &inputs.query,
        &inputs.output,
        inputs.workspace.data(),
        inputs.workspace.size(),
        0.125F,
    };
}

void testConfigurationAndDescriptorContract()
{
    EngineKvConfig const gqa_config{
        KvLayout{22, 4, 64},
        32,
        4096,
    };
    expect(gqa_config.valid(), "valid GQA config must be accepted");

    EngineKvConfig invalid_gqa = gqa_config;
    invalid_gqa.query_head_count = 30;
    expect(
        !invalid_gqa.valid(),
        "query heads must be an integer multiple of KV heads"
    );

    EngineKvConfig invalid_promotion = gqa_config;
    invalid_promotion.promotion_policy =
        static_cast<EngineKvPromotionPolicy>(255);
    expect(!invalid_promotion.valid(),
        "unknown promotion policies must be rejected");

    DeviceBlockDescriptor descriptor{
        3,
        11,
        64,
        8,
        8,
        PageKind::Micro,
    };
    expect(descriptor.valid(), "complete device descriptor must be valid");

    DeviceBlockTableView const view{
        &descriptor,
        1,
        72,
        72,
    };
    expect(view.valid(), "descriptor view must bind the current token position");

    DeviceBlockTableView stale_view = view;
    stale_view.current_token_position = 71;
    expect(
        !stale_view.valid(),
        "descriptor view must reject Scheduler/KV length drift"
    );
}

void testNormalLayerSequenceAndCommit()
{
    int stream_token = 0;
    auto trace = std::make_shared<TransactionTrace>();
    trace->expected_stream = &stream_token;
    TokenTransaction transaction = makeTransaction(trace);
    DeviceInputs inputs;

    TokenTransactionSnapshot snapshot = transaction.snapshot();
    expect(transaction.valid(), "new transaction must be active");
    expect(snapshot.transaction_id == 7, "transaction id must be preserved");
    expect(snapshot.request_id == 42, "request id must be preserved");
    expect(snapshot.logical_token_position == 9, "token position must be preserved");
    expect(snapshot.next_layer == 0, "layer sequence must begin at zero");
    expect(
        snapshot.phase == TokenTransactionPhase::AwaitingLayerWrite,
        "new transaction must await layer write"
    );

    expect(
        transaction.attendLayer(makeAttention(inputs, 0)).error
            == EngineKvError::LayerOutOfOrder,
        "attention before write must be rejected"
    );
    expect(
        transaction.writeLayer(makeWrite(inputs, 1)).error
            == EngineKvError::LayerOutOfOrder,
        "skipping the first layer must be rejected"
    );
    expect(trace->events.empty(), "rejected ordering must not reach backend");

    for (std::uint32_t layer = 0; layer < 3; ++layer) {
        expect(
            transaction.writeLayer(makeWrite(inputs, layer)).ok(),
            "ordered layer write must succeed"
        );
        expect(
            transaction.writeLayer(makeWrite(inputs, layer)).error
                == EngineKvError::LayerOutOfOrder,
            "duplicate layer write must be rejected"
        );
        expect(
            transaction.attendLayer(makeAttention(inputs, layer)).ok(),
            "same-layer attention must succeed after write"
        );

        if (layer != 2) {
            expect(
                transaction.commit().error == EngineKvError::InvalidState,
                "commit before all layers must be rejected"
            );
        }
    }

    snapshot = transaction.snapshot();
    expect(
        snapshot.phase == TokenTransactionPhase::ReadyToCommit,
        "last attention must make transaction commit-ready"
    );
    expect(snapshot.next_layer == 3, "all layers must be completed once");
    expect(transaction.commit().ok(), "complete transaction must commit");
    expect(!transaction.valid(), "committed transaction must become terminal");
    expect(
        transaction.snapshot().phase == TokenTransactionPhase::Committed,
        "terminal state must record Commit"
    );
    expect(
        transaction.rollback().error == EngineKvError::InvalidState,
        "committed transaction must reject Rollback"
    );

    std::vector<std::string> const expected{
        "write:0", "attend:0",
        "write:1", "attend:1",
        "write:2", "attend:2",
        "commit",
    };
    expect(trace->events == expected, "backend call sequence must be exact");
    expect(trace->rollback_count == 0, "Commit must not Rollback");
}

void testArgumentValidationDoesNotSubmit()
{
    auto trace = std::make_shared<TransactionTrace>();
    TokenTransaction transaction = makeTransaction(trace, 1);
    DeviceInputs inputs;

    LayerKvWrite invalid_write = makeWrite(inputs, 0);
    invalid_write.device_key = nullptr;
    expect(
        transaction.writeLayer(invalid_write).error
            == EngineKvError::InvalidArgument,
        "null K pointer must be rejected"
    );
    expect(trace->events.empty(), "invalid write must not reach backend");

    expect(
        transaction.writeLayer(makeWrite(inputs, 0)).ok(),
        "valid write must remain possible after argument rejection"
    );
    PagedDecodeRequest invalid_attention = makeAttention(inputs, 0);
    invalid_attention.workspace_bytes = 0;
    expect(
        transaction.attendLayer(invalid_attention).error
            == EngineKvError::InvalidArgument,
        "empty workspace must be rejected"
    );
    expect(
        trace->events.size() == 1,
        "invalid attention must not reach backend"
    );
    expect(transaction.rollback().ok(), "explicit Rollback must succeed");
}

void testBatchedAttentionAdvancesAndFailsLanesIndependently()
{
    int stream_token = 0;
    auto healthy_trace = std::make_shared<TransactionTrace>();
    auto failed_trace = std::make_shared<TransactionTrace>();
    healthy_trace->expected_stream = &stream_token;
    failed_trace->expected_stream = &stream_token;
    failed_trace->attention_failure = EngineKvError::SubmissionFailed;
    failed_trace->failure_layer = 0;

    TokenTransaction healthy = makeTransaction(healthy_trace, 1, 20);
    TokenTransaction failed = makeTransaction(failed_trace, 1, 21);
    DeviceInputs healthy_inputs;
    DeviceInputs failed_inputs;
    expect(healthy.writeLayer(makeWrite(healthy_inputs, 0)).ok(),
        "healthy batch lane writes its KV");
    expect(failed.writeLayer(makeWrite(failed_inputs, 0)).ok(),
        "failing batch lane writes its KV");

    std::array<PagedDecodeBatchItem, 2> items{{
        {&healthy, makeAttention(healthy_inputs, 0)},
        {&failed, makeAttention(failed_inputs, 0)},
    }};
    std::array<DevicePagedDecodeBatchItem, 2> host_items{};
    std::array<DevicePagedDecodeBatchItem, 2> device_items{};
    PagedDecodeBatch batch{
        items.data(),
        items.size(),
        host_items.data(),
        device_items.data(),
        host_items.size(),
    };
    attendLayerBatch(batch);

    expect(items[0].status.ok(), "healthy batch lane succeeds");
    expect(items[1].status.error == EngineKvError::SubmissionFailed,
        "failed batch lane reports its own backend error");
    expect(healthy.snapshot().phase == TokenTransactionPhase::ReadyToCommit,
        "healthy batch lane advances to commit-ready");
    expect(failed.snapshot().phase == TokenTransactionPhase::Failed,
        "failed batch lane becomes fail-closed");
    expect(healthy_trace->attention_batch_count == 1
            && failed_trace->attention_batch_count == 0,
        "one backend dispatch coordinates the batch");
    expect(healthy.commit().ok(), "healthy batch lane commits independently");
    expect(failed.rollback().ok(), "failed batch lane rolls back independently");
}

void testDestructorAndMoveOwnership()
{
    auto invalid_trace = std::make_shared<TransactionTrace>();
    TokenTransaction invalid = makeTransaction(invalid_trace, 0);
    expect(!invalid.valid(), "zero-layer transaction must be rejected");
    expect(
        invalid_trace->rollback_count == 1,
        "invalid reservation must release backend resources"
    );

    auto destructor_trace = std::make_shared<TransactionTrace>();
    {
        TokenTransaction transaction = makeTransaction(destructor_trace, 1);
        expect(transaction.valid(), "scoped transaction must be active");
    }
    expect(
        destructor_trace->rollback_count == 1,
        "uncommitted destructor must Rollback exactly once"
    );

    auto source_trace = std::make_shared<TransactionTrace>();
    auto target_trace = std::make_shared<TransactionTrace>();
    {
        TokenTransaction source = makeTransaction(source_trace, 1, 10);
        TokenTransaction target = makeTransaction(target_trace, 1, 11);
        target = std::move(source);

        expect(!source.valid(), "moved-from transaction must be empty");
        expect(
            source.snapshot().phase == TokenTransactionPhase::Empty,
            "moved-from transaction must expose Empty state"
        );
        expect(target.valid(), "move target must own source transaction");
        expect(
            target_trace->rollback_count == 1,
            "move assignment must Rollback overwritten transaction"
        );
    }
    expect(
        source_trace->rollback_count == 1,
        "moved transaction must still Rollback exactly once"
    );
    expect(
        target_trace->rollback_count == 1,
        "overwritten transaction must not Rollback twice"
    );
}

void testSubmissionAndCommitFailuresRollback()
{
    DeviceInputs inputs;

    auto submission_trace = std::make_shared<TransactionTrace>();
    submission_trace->attention_failure = EngineKvError::SubmissionFailed;
    submission_trace->failure_layer = 0;
    TokenTransaction submission = makeTransaction(submission_trace, 1);

    expect(
        submission.writeLayer(makeWrite(inputs, 0)).ok(),
        "write before injected attention failure must succeed"
    );
    expect(
        submission.attendLayer(makeAttention(inputs, 0)).error
            == EngineKvError::SubmissionFailed,
        "submission failure must be returned"
    );
    expect(
        submission.snapshot().phase == TokenTransactionPhase::Failed,
        "submission failure must make transaction fail-closed"
    );
    expect(
        submission.commit().error == EngineKvError::InvalidState,
        "failed transaction must reject Commit"
    );
    expect(submission.rollback().ok(), "failed transaction must allow Rollback");
    expect(
        submission_trace->rollback_count == 1,
        "submission failure must release resources once"
    );

    auto completion_trace = std::make_shared<TransactionTrace>();
    completion_trace->commit_failure = EngineKvError::ExecutionFailed;
    TokenTransaction completion = makeTransaction(completion_trace, 1);
    expect(
        completion.writeLayer(makeWrite(inputs, 0)).ok()
            && completion.attendLayer(makeAttention(inputs, 0)).ok(),
        "transaction must reach commit failure point"
    );
    expect(
        completion.commit().error == EngineKvError::ExecutionFailed,
        "completion failure must be returned from Commit"
    );
    expect(
        completion.snapshot().phase == TokenTransactionPhase::RolledBack,
        "failed completion must atomically Rollback"
    );
    expect(
        completion_trace->rollback_count == 1,
        "failed completion must Rollback exactly once"
    );
}

class FakeEngineBackend final : public EngineKvBackend {
public:
    explicit FakeEngineBackend(EngineKvBackendKind kind)
        : kind_(kind)
    {
    }

    [[nodiscard]] EngineKvBackendKind kind() const noexcept override
    {
        return kind_;
    }

    [[nodiscard]] EngineKvConfig config() const noexcept override
    {
        return EngineKvConfig{KvLayout{2, 2, 8}, 4, 64};
    }

    [[nodiscard]] EngineKvStatus createRequest(RequestId) override
    {
        ++snapshot_.request_count;
        return {};
    }

    [[nodiscard]] EngineKvStatus forkRequest(RequestId, RequestId) override
    {
        ++snapshot_.request_count;
        return {};
    }

    [[nodiscard]] EngineKvStatus releaseRequest(RequestId) override
    {
        if (snapshot_.request_count == 0) {
            return EngineKvStatus{EngineKvError::RequestNotFound};
        }
        --snapshot_.request_count;
        return {};
    }

    [[nodiscard]] TokenReserveResult reserveToken(
        ReserveTokenRequest const& request) override
    {
        TokenReserveResult result;
        if (request.request_id == kInvalidRequestId) {
            result.status = EngineKvStatus{EngineKvError::InvalidArgument};
            return result;
        }

        last_trace_ = std::make_shared<TransactionTrace>();
        last_trace_->expected_stream = request.stream;
        result.transaction = TokenTransaction(
            std::make_unique<FakeTransactionBackend>(last_trace_),
            next_transaction_id_++,
            request.request_id,
            request.expected_committed_tokens,
            config().kv_layout.layer_count,
            request.stream
        );
        ++snapshot_.active_transaction_count;
        return result;
    }

    [[nodiscard]] EngineKvBackendSnapshot snapshot() const override
    {
        return snapshot_;
    }

    [[nodiscard]] bool checkInvariants() const override
    {
        return config().valid();
    }

private:
    EngineKvBackendKind kind_;
    TokenTransactionId next_transaction_id_{1};
    EngineKvBackendSnapshot snapshot_{};
    std::shared_ptr<TransactionTrace> last_trace_{};
};

void testFixedAndHeterogeneousShareOneInterface()
{
    for (EngineKvBackendKind const kind : {
             EngineKvBackendKind::Heterogeneous,
             EngineKvBackendKind::Fixed,
         }) {
        std::unique_ptr<EngineKvBackend> backend =
            std::make_unique<FakeEngineBackend>(kind);
        expect(backend->kind() == kind, "backend kind must be observable");
        expect(backend->config().valid(), "backend config must be valid");
        expect(backend->createRequest(8).ok(), "request creation must use common API");

        TokenReserveResult reserved = backend->reserveToken(
            ReserveTokenRequest{8, 17, nullptr}
        );
        expect(reserved.ok(), "both page strategies must reserve through common API");
        expect(
            reserved.transaction.snapshot().logical_token_position == 17,
            "reservation must preserve expected committed length"
        );
        expect(
            reserved.transaction.rollback().ok(),
            "both page strategies must share rollback semantics"
        );
        expect(backend->releaseRequest(8).ok(), "request release must use common API");
        expect(backend->checkInvariants(), "backend invariants must remain valid");
    }
}

} // namespace

int main()
{
    testConfigurationAndDescriptorContract();
    testNormalLayerSequenceAndCommit();
    testArgumentValidationDoesNotSubmit();
    testBatchedAttentionAdvancesAndFailsLanesIndependently();
    testDestructorAndMoveOwnership();
    testSubmissionAndCommitFailuresRollback();
    testFixedAndHeterogeneousShareOneInterface();

    if (failures != 0) {
        std::cerr << failures << " engine KV contract checks failed\n";
        return 1;
    }

    std::cout << "engine KV contract checks passed\n";
    return 0;
}
