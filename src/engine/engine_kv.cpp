#include "kim-kv/engine/engine_kv.h"

#include <utility>

namespace kimkvcache {
namespace {

[[nodiscard]] constexpr EngineKvStatus status(
    EngineKvError error) noexcept
{
    return EngineKvStatus{error};
}

} // namespace

TokenTransactionBackend* TokenTransactionBackend::batchBackend(
    LayerKvWriteBatchItem const& item) noexcept
{
    return item.transaction != nullptr
        ? item.transaction->backend_.get()
        : nullptr;
}

EngineStream TokenTransactionBackend::batchStream(
    LayerKvWriteBatchItem const& item) noexcept
{
    return item.transaction != nullptr
        ? item.transaction->stream_
        : nullptr;
}

TokenTransactionBackend* TokenTransactionBackend::batchBackend(
    PagedDecodeBatchItem const& item) noexcept
{
    return item.transaction != nullptr
        ? item.transaction->backend_.get()
        : nullptr;
}

EngineStream TokenTransactionBackend::batchStream(
    PagedDecodeBatchItem const& item) noexcept
{
    return item.transaction != nullptr
        ? item.transaction->stream_
        : nullptr;
}

void TokenTransactionBackend::writeLayerBatch(LayerKvWriteBatch& batch)
{
    for (std::size_t index = 0; index < batch.item_count; ++index) {
        LayerKvWriteBatchItem& item = batch.items[index];
        if (!item.status.ok()) {
            continue;
        }
        TokenTransactionBackend* const backend = batchBackend(item);
        if (backend == nullptr) {
            item.status = status(EngineKvError::InvalidState);
            continue;
        }
        item.status = backend->writeLayer(item.write, batchStream(item));
    }
}

void TokenTransactionBackend::attendLayerBatch(PagedDecodeBatch& batch)
{
    for (std::size_t index = 0; index < batch.item_count; ++index) {
        PagedDecodeBatchItem& item = batch.items[index];
        if (!item.status.ok()) {
            continue;
        }
        TokenTransactionBackend* const backend = batchBackend(item);
        if (backend == nullptr) {
            item.status = status(EngineKvError::InvalidState);
            continue;
        }
        item.status = backend->attendLayer(item.request, batchStream(item));
    }
}

TokenTransaction::TokenTransaction(
    std::unique_ptr<TokenTransactionBackend> backend,
    TokenTransactionId transaction_id,
    RequestId request_id,
    std::uint32_t logical_token_position,
    std::uint32_t layer_count,
    EngineStream stream) noexcept
    : backend_(std::move(backend))
    , transaction_id_(transaction_id)
    , request_id_(request_id)
    , logical_token_position_(logical_token_position)
    , layer_count_(layer_count)
    , phase_(TokenTransactionPhase::AwaitingLayerWrite)
    , stream_(stream)
{
    if (backend_ == nullptr
        || transaction_id_ == kInvalidTokenTransactionId
        || request_id_ == kInvalidRequestId
        || layer_count_ == 0) {
        if (backend_ != nullptr) {
            backend_->rollback(stream_);
        }
        backend_.reset();
        resetMovedFrom();
    }
}

TokenTransaction::~TokenTransaction()
{
    rollbackNoexcept();
}

TokenTransaction::TokenTransaction(TokenTransaction&& other) noexcept
    : backend_(std::move(other.backend_))
    , transaction_id_(other.transaction_id_)
    , request_id_(other.request_id_)
    , logical_token_position_(other.logical_token_position_)
    , layer_count_(other.layer_count_)
    , next_layer_(other.next_layer_)
    , phase_(other.phase_)
    , stream_(other.stream_)
{
    other.resetMovedFrom();
}

TokenTransaction& TokenTransaction::operator=(
    TokenTransaction&& other) noexcept
{
    if (this == &other) {
        return *this;
    }

    rollbackNoexcept();

    backend_ = std::move(other.backend_);
    transaction_id_ = other.transaction_id_;
    request_id_ = other.request_id_;
    logical_token_position_ = other.logical_token_position_;
    layer_count_ = other.layer_count_;
    next_layer_ = other.next_layer_;
    phase_ = other.phase_;
    stream_ = other.stream_;

    other.resetMovedFrom();
    return *this;
}

bool TokenTransaction::active() const noexcept
{
    switch (phase_) {
    case TokenTransactionPhase::AwaitingLayerWrite:
    case TokenTransactionPhase::AwaitingLayerAttention:
    case TokenTransactionPhase::ReadyToCommit:
    case TokenTransactionPhase::Failed:
        return backend_ != nullptr;
    case TokenTransactionPhase::Empty:
    case TokenTransactionPhase::Committed:
    case TokenTransactionPhase::RolledBack:
        return false;
    }

    return false;
}

bool TokenTransaction::valid() const noexcept
{
    return active();
}

TokenTransactionSnapshot TokenTransaction::snapshot() const noexcept
{
    return TokenTransactionSnapshot{
        transaction_id_,
        request_id_,
        logical_token_position_,
        layer_count_,
        next_layer_,
        phase_,
    };
}

EngineKvStatus TokenTransaction::writeLayer(
    LayerKvWrite const& write)
{
    if (!active()) {
        return status(EngineKvError::InvalidState);
    }
    if (phase_ != TokenTransactionPhase::AwaitingLayerWrite
        || write.layer != next_layer_) {
        return status(EngineKvError::LayerOutOfOrder);
    }
    if (write.device_key == nullptr || write.device_value == nullptr) {
        return status(EngineKvError::InvalidArgument);
    }

    EngineKvStatus const result = backend_->writeLayer(write, stream_);
    if (!result.ok()) {
        phase_ = TokenTransactionPhase::Failed;
        return result;
    }

    phase_ = TokenTransactionPhase::AwaitingLayerAttention;
    return {};
}

EngineKvStatus TokenTransaction::attendLayer(
    PagedDecodeRequest const& request)
{
    if (!active()) {
        return status(EngineKvError::InvalidState);
    }
    if (phase_ != TokenTransactionPhase::AwaitingLayerAttention
        || request.layer != next_layer_) {
        return status(EngineKvError::LayerOutOfOrder);
    }
    if (request.device_query == nullptr
        || request.device_output == nullptr
        || request.device_workspace == nullptr
        || request.workspace_bytes == 0
        || !(request.attention_scale > 0.0F)) {
        return status(EngineKvError::InvalidArgument);
    }

    EngineKvStatus const result = backend_->attendLayer(request, stream_);
    if (!result.ok()) {
        phase_ = TokenTransactionPhase::Failed;
        return result;
    }

    ++next_layer_;
    phase_ = next_layer_ == layer_count_
        ? TokenTransactionPhase::ReadyToCommit
        : TokenTransactionPhase::AwaitingLayerWrite;
    return {};
}

EngineKvStatus TokenTransaction::commit()
{
    if (!active() || phase_ != TokenTransactionPhase::ReadyToCommit) {
        return status(EngineKvError::InvalidState);
    }

    EngineKvStatus const result = backend_->commit(stream_);
    if (!result.ok()) {
        backend_->rollback(stream_);
        backend_.reset();
        phase_ = TokenTransactionPhase::RolledBack;
        return result;
    }

    backend_.reset();
    phase_ = TokenTransactionPhase::Committed;
    return {};
}

EngineKvStatus TokenTransaction::rollback() noexcept
{
    if (!active()) {
        return status(EngineKvError::InvalidState);
    }

    rollbackNoexcept();
    return {};
}

void TokenTransaction::rollbackNoexcept() noexcept
{
    if (!active()) {
        return;
    }

    backend_->rollback(stream_);
    backend_.reset();
    phase_ = TokenTransactionPhase::RolledBack;
}

void TokenTransaction::resetMovedFrom() noexcept
{
    transaction_id_ = kInvalidTokenTransactionId;
    request_id_ = kInvalidRequestId;
    logical_token_position_ = 0;
    layer_count_ = 0;
    next_layer_ = 0;
    phase_ = TokenTransactionPhase::Empty;
    stream_ = nullptr;
}

void writeLayerBatch(LayerKvWriteBatch& batch)
{
    if (batch.items == nullptr || batch.item_count == 0) {
        return;
    }

    bool const scratch_valid = batch.host_items != nullptr
        && batch.device_items != nullptr
        && batch.item_capacity >= batch.item_count;
    TokenTransactionBackend* dispatcher = nullptr;

    for (std::size_t index = 0; index < batch.item_count; ++index) {
        LayerKvWriteBatchItem& item = batch.items[index];
        item.status = {};
        item.dispatched = false;
        TokenTransaction* const transaction = item.transaction;
        if (!scratch_valid) {
            item.status = status(EngineKvError::InvalidArgument);
        } else if (transaction == nullptr || !transaction->active()) {
            item.status = status(EngineKvError::InvalidState);
        } else if (transaction->phase_
                != TokenTransactionPhase::AwaitingLayerWrite
            || item.write.layer != transaction->next_layer_) {
            item.status = status(EngineKvError::LayerOutOfOrder);
        } else if (item.write.device_key == nullptr
            || item.write.device_value == nullptr) {
            item.status = status(EngineKvError::InvalidArgument);
        } else {
            item.dispatched = true;
            if (dispatcher == nullptr) {
                dispatcher = transaction->backend_.get();
            }
        }
    }

    if (dispatcher == nullptr) {
        return;
    }
    dispatcher->writeLayerBatch(batch);

    for (std::size_t index = 0; index < batch.item_count; ++index) {
        LayerKvWriteBatchItem& item = batch.items[index];
        TokenTransaction* const transaction = item.transaction;
        if (!item.dispatched || transaction == nullptr
            || transaction->phase_
                != TokenTransactionPhase::AwaitingLayerWrite
            || item.write.layer != transaction->next_layer_) {
            continue;
        }
        if (!item.status.ok()) {
            transaction->phase_ = TokenTransactionPhase::Failed;
            continue;
        }
        transaction->phase_ = TokenTransactionPhase::AwaitingLayerAttention;
    }
}

void attendLayerBatch(PagedDecodeBatch& batch)
{
    if (batch.items == nullptr || batch.item_count == 0) {
        return;
    }

    bool const scratch_valid = batch.host_items != nullptr
        && batch.device_items != nullptr
        && batch.item_capacity >= batch.item_count;
    TokenTransactionBackend* dispatcher = nullptr;

    for (std::size_t index = 0; index < batch.item_count; ++index) {
        PagedDecodeBatchItem& item = batch.items[index];
        item.status = {};
        item.dispatched = false;
        TokenTransaction* const transaction = item.transaction;
        if (!scratch_valid) {
            item.status = status(EngineKvError::InvalidArgument);
        } else if (transaction == nullptr || !transaction->active()) {
            item.status = status(EngineKvError::InvalidState);
        } else if (transaction->phase_
                != TokenTransactionPhase::AwaitingLayerAttention
            || item.request.layer != transaction->next_layer_) {
            item.status = status(EngineKvError::LayerOutOfOrder);
        } else if (item.request.device_query == nullptr
            || item.request.device_output == nullptr
            || item.request.device_workspace == nullptr
            || item.request.workspace_bytes == 0
            || !(item.request.attention_scale > 0.0F)) {
            item.status = status(EngineKvError::InvalidArgument);
        } else {
            item.dispatched = true;
            if (dispatcher == nullptr) {
                dispatcher = transaction->backend_.get();
            }
        }
    }

    if (dispatcher == nullptr) {
        return;
    }
    dispatcher->attendLayerBatch(batch);

    for (std::size_t index = 0; index < batch.item_count; ++index) {
        PagedDecodeBatchItem& item = batch.items[index];
        TokenTransaction* const transaction = item.transaction;
        if (!item.dispatched || transaction == nullptr
            || transaction->phase_
                != TokenTransactionPhase::AwaitingLayerAttention
            || item.request.layer != transaction->next_layer_) {
            continue;
        }
        if (!item.status.ok()) {
            transaction->phase_ = TokenTransactionPhase::Failed;
            continue;
        }

        ++transaction->next_layer_;
        transaction->phase_ = transaction->next_layer_
                == transaction->layer_count_
            ? TokenTransactionPhase::ReadyToCommit
            : TokenTransactionPhase::AwaitingLayerWrite;
    }
}

} // namespace kimkvcache
