#pragma once

#include "kim-kv/core/page_handle.h"
#include "kim-kv/core/kv_layout.h"
#include "kim-kv/runtime/promotion.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <type_traits>

namespace kimkvcache {

using EngineStream = void*;
using TokenTransactionId = std::uint64_t;

constexpr TokenTransactionId kInvalidTokenTransactionId = 0;

enum class EngineKvBackendKind : std::uint8_t {
    Heterogeneous,
    Fixed,
};

// Promotion is a storage-layout optimization and must never change token
// commit semantics. Eager promotes each newly completed aligned 64-token run;
// Disabled is retained for controlled A/B measurements.
enum class EngineKvPromotionPolicy : std::uint8_t {
    Disabled,
    Eager,
};

[[nodiscard]] constexpr bool isKnownEngineKvPromotionPolicy(
    EngineKvPromotionPolicy policy) noexcept
{
    return policy == EngineKvPromotionPolicy::Disabled
        || policy == EngineKvPromotionPolicy::Eager;
}

enum class EngineKvError : std::uint8_t {
    None,
    InvalidArgument,
    InvalidState,
    RequestNotFound,
    RequestAlreadyExists,
    RequestConflict,
    ResourceExhausted,
    LayerOutOfOrder,
    SubmissionFailed,
    ExecutionFailed,
    InternalError,
};

[[nodiscard]] constexpr std::string_view toString(
    EngineKvError error) noexcept
{
    switch (error) {
    case EngineKvError::None:
        return "none";
    case EngineKvError::InvalidArgument:
        return "invalid_argument";
    case EngineKvError::InvalidState:
        return "invalid_state";
    case EngineKvError::RequestNotFound:
        return "request_not_found";
    case EngineKvError::RequestAlreadyExists:
        return "request_already_exists";
    case EngineKvError::RequestConflict:
        return "request_conflict";
    case EngineKvError::ResourceExhausted:
        return "resource_exhausted";
    case EngineKvError::LayerOutOfOrder:
        return "layer_out_of_order";
    case EngineKvError::SubmissionFailed:
        return "submission_failed";
    case EngineKvError::ExecutionFailed:
        return "execution_failed";
    case EngineKvError::InternalError:
        return "internal_error";
    }

    return "unknown";
}

struct EngineKvStatus final {
    EngineKvError error{EngineKvError::None};

    [[nodiscard]] constexpr bool ok() const noexcept
    {
        return error == EngineKvError::None;
    }

    [[nodiscard]] explicit constexpr operator bool() const noexcept
    {
        return ok();
    }
};

struct EngineKvConfig final {
    KvLayout kv_layout{};
    std::uint32_t query_head_count{0};
    std::size_t attention_workspace_bytes{0};
    EngineKvPromotionPolicy promotion_policy{
        EngineKvPromotionPolicy::Eager
    };

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return kv_layout.valid()
            && query_head_count >= kv_layout.kv_head_count
            && query_head_count % kv_layout.kv_head_count == 0
            && attention_workspace_bytes != 0
            && isKnownEngineKvPromotionPolicy(promotion_policy);
    }
};

// Host 端在持有 PageLease 时构造并上传该描述符。Device 指针只能在对应
// Lease 和提交 Stream 的工作完成前使用，不能跨 BlockTable 版本缓存裸指针。
struct DeviceBlockDescriptor final {
    std::uint32_t slot{PageHandle::kInvalidSlot};
    std::uint32_t generation{PageHandle::kInvalidGeneration};
    std::uint32_t logical_token_begin{0};
    std::uint16_t valid_tokens{0};
    std::uint16_t page_token_capacity{0};
    PageKind kind{PageKind::Micro};

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return slot != PageHandle::kInvalidSlot
            && generation != PageHandle::kInvalidGeneration
            && valid_tokens != 0
            && page_token_capacity != 0
            && valid_tokens <= page_token_capacity
            && isKnownPageKind(kind);
    }
};

static_assert(
    std::is_trivially_copyable_v<DeviceBlockDescriptor>,
    "DeviceBlockDescriptor must remain trivially copyable"
);

static_assert(
    std::is_standard_layout_v<DeviceBlockDescriptor>,
    "DeviceBlockDescriptor must remain standard-layout"
);

struct DeviceBlockTableView final {
    DeviceBlockDescriptor const* device_descriptors{nullptr};
    std::uint32_t descriptor_count{0};
    std::uint32_t committed_token_count{0};
    std::uint32_t current_token_position{0};

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return device_descriptors != nullptr
            && descriptor_count != 0
            && current_token_position == committed_token_count;
    }
};

struct LayerKvWrite final {
    std::uint32_t layer{0};
    KvScalar const* device_key{nullptr};
    KvScalar const* device_value{nullptr};
};

class TokenTransaction;

// One lane of a batched layer KV write. Each lane owns an independent token
// transaction and may resolve to a different target page/COW source.
struct LayerKvWriteBatchItem final {
    TokenTransaction* transaction{nullptr};
    LayerKvWrite write{};
    EngineKvStatus status{EngineKvError::InvalidState};
    bool dispatched{false};
};

// Device-side metadata for one KV-write lane. copy_source/copy_token_count are
// populated only for partial-tail COW; every valid lane writes one new token.
struct DeviceLayerKvWriteBatchItem final {
    KvScalar const* device_key{nullptr};
    KvScalar const* device_value{nullptr};
    KvScalar const* copy_source{nullptr};
    KvScalar* target{nullptr};
    std::uint32_t source_capacity{0};
    std::uint32_t target_capacity{0};
    std::uint32_t copy_token_count{0};
    std::uint32_t target_token{0};
    std::uint32_t layer{0};
};

static_assert(
    std::is_trivially_copyable_v<DeviceLayerKvWriteBatchItem>,
    "DeviceLayerKvWriteBatchItem must remain trivially copyable"
);

static_assert(
    std::is_standard_layout_v<DeviceLayerKvWriteBatchItem>,
    "DeviceLayerKvWriteBatchItem must remain standard-layout"
);

// host_items and device_items are caller-owned preallocated scratch. CUDA
// backends upload the per-lane targets once and issue one layer-write kernel.
struct LayerKvWriteBatch final {
    LayerKvWriteBatchItem* items{nullptr};
    std::size_t item_count{0};
    DeviceLayerKvWriteBatchItem* host_items{nullptr};
    DeviceLayerKvWriteBatchItem* device_items{nullptr};
    std::size_t item_capacity{0};
};

struct PagedDecodeRequest final {
    std::uint32_t layer{0};
    KvScalar const* device_query{nullptr};
    KvScalar* device_output{nullptr};
    void* device_workspace{nullptr};
    std::size_t workspace_bytes{0};
    float attention_scale{0.0F};
};

// One lane of a ragged paged-attention launch. The caller owns the transaction
// and all query/output/workspace storage. status is populated independently so
// one rejected or failed lane does not prevent healthy lanes from advancing.
struct PagedDecodeBatchItem final {
    TokenTransaction* transaction{nullptr};
    PagedDecodeRequest request{};
    EngineKvStatus status{EngineKvError::InvalidState};
    bool dispatched{false};
};

// Device-side lane metadata uploaded once per layer. Every lane may reference
// an independent block table, token count, query, score workspace, and output.
struct DevicePagedDecodeBatchItem final {
    DeviceBlockDescriptor const* device_descriptors{nullptr};
    std::uint32_t descriptor_count{0};
    std::uint32_t token_count{0};
    std::uint32_t layer{0};
    KvScalar const* device_query{nullptr};
    float* device_scores{nullptr};
    KvScalar* device_output{nullptr};
    float attention_scale{0.0F};
};

static_assert(
    std::is_trivially_copyable_v<DevicePagedDecodeBatchItem>,
    "DevicePagedDecodeBatchItem must remain trivially copyable"
);

static_assert(
    std::is_standard_layout_v<DevicePagedDecodeBatchItem>,
    "DevicePagedDecodeBatchItem must remain standard-layout"
);

// host_items and device_items are caller-owned, preallocated scratch with at
// least item_capacity entries. CUDA backends use them to avoid per-layer
// allocation; non-CUDA backends may ignore the scratch and use scalar fallback.
struct PagedDecodeBatch final {
    PagedDecodeBatchItem* items{nullptr};
    std::size_t item_count{0};
    DevicePagedDecodeBatchItem* host_items{nullptr};
    DevicePagedDecodeBatchItem* device_items{nullptr};
    std::size_t item_capacity{0};
};

struct ReserveTokenRequest final {
    RequestId request_id{kInvalidRequestId};
    std::uint32_t expected_committed_tokens{0};
    EngineStream stream{nullptr};
};

enum class TokenTransactionPhase : std::uint8_t {
    Empty,
    AwaitingLayerWrite,
    AwaitingLayerAttention,
    ReadyToCommit,
    Failed,
    Committed,
    RolledBack,
};

[[nodiscard]] constexpr std::string_view toString(
    TokenTransactionPhase phase) noexcept
{
    switch (phase) {
    case TokenTransactionPhase::Empty:
        return "empty";
    case TokenTransactionPhase::AwaitingLayerWrite:
        return "awaiting_layer_write";
    case TokenTransactionPhase::AwaitingLayerAttention:
        return "awaiting_layer_attention";
    case TokenTransactionPhase::ReadyToCommit:
        return "ready_to_commit";
    case TokenTransactionPhase::Failed:
        return "failed";
    case TokenTransactionPhase::Committed:
        return "committed";
    case TokenTransactionPhase::RolledBack:
        return "rolled_back";
    }

    return "unknown";
}

struct TokenTransactionSnapshot final {
    TokenTransactionId transaction_id{kInvalidTokenTransactionId};
    RequestId request_id{kInvalidRequestId};
    std::uint32_t logical_token_position{0};
    std::uint32_t layer_count{0};
    std::uint32_t next_layer{0};
    TokenTransactionPhase phase{TokenTransactionPhase::Empty};
};

// Heterogeneous 与 Fixed 后端实现这个窄 SPI。write/attend 只提交到绑定的
// Stream，不做 Host Wait；commit 是 Token 边界唯一允许的完成检查点。
// 任意失败后 rollback 必须等待在途访问安全结束，再释放 Descriptor/PageLease。
class TokenTransactionBackend {
public:
    virtual ~TokenTransactionBackend() = default;

    TokenTransactionBackend(TokenTransactionBackend const&) = delete;
    TokenTransactionBackend& operator=(TokenTransactionBackend const&) = delete;

    [[nodiscard]] virtual EngineKvStatus writeLayer(
        LayerKvWrite const& write,
        EngineStream stream
    ) = 0;

    // The default implementation preserves compatibility through scalar
    // submissions. CUDA backends override this when lanes share storage/stream.
    virtual void writeLayerBatch(LayerKvWriteBatch& batch);

    [[nodiscard]] virtual EngineKvStatus attendLayer(
        PagedDecodeRequest const& request,
        EngineStream stream
    ) = 0;

    // The default implementation preserves compatibility by submitting every
    // valid lane independently. CUDA backends override this with one ragged
    // batch launch when all participating transactions share storage/stream.
    virtual void attendLayerBatch(PagedDecodeBatch& batch);

    [[nodiscard]] virtual EngineKvStatus commit(
        EngineStream stream
    ) = 0;

    virtual void rollback(EngineStream stream) noexcept = 0;

protected:
    TokenTransactionBackend() = default;

    [[nodiscard]] static TokenTransactionBackend* batchBackend(
        LayerKvWriteBatchItem const& item
    ) noexcept;

    [[nodiscard]] static EngineStream batchStream(
        LayerKvWriteBatchItem const& item
    ) noexcept;

    [[nodiscard]] static TokenTransactionBackend* batchBackend(
        PagedDecodeBatchItem const& item
    ) noexcept;

    [[nodiscard]] static EngineStream batchStream(
        PagedDecodeBatchItem const& item
    ) noexcept;
};

// 一个 Token 的 move-only RAII 事务。正常顺序固定为每层
// WriteLayer -> AttendLayer，全部层完成后才能 Commit。未 Commit 的对象在
// 析构或被 move-assign 覆盖时自动 Rollback。
class TokenTransaction final {
public:
    TokenTransaction() noexcept = default;

    TokenTransaction(
        std::unique_ptr<TokenTransactionBackend> backend,
        TokenTransactionId transaction_id,
        RequestId request_id,
        std::uint32_t logical_token_position,
        std::uint32_t layer_count,
        EngineStream stream
    ) noexcept;

    ~TokenTransaction();

    TokenTransaction(TokenTransaction const&) = delete;
    TokenTransaction& operator=(TokenTransaction const&) = delete;

    TokenTransaction(TokenTransaction&& other) noexcept;
    TokenTransaction& operator=(TokenTransaction&& other) noexcept;

    [[nodiscard]] bool valid() const noexcept;

    [[nodiscard]] TokenTransactionSnapshot snapshot() const noexcept;

    [[nodiscard]] EngineKvStatus writeLayer(
        LayerKvWrite const& write
    );

    [[nodiscard]] EngineKvStatus attendLayer(
        PagedDecodeRequest const& request
    );

    [[nodiscard]] EngineKvStatus commit();

    [[nodiscard]] EngineKvStatus rollback() noexcept;

private:
    [[nodiscard]] bool active() const noexcept;
    void rollbackNoexcept() noexcept;
    void resetMovedFrom() noexcept;

    std::unique_ptr<TokenTransactionBackend> backend_{};
    TokenTransactionId transaction_id_{kInvalidTokenTransactionId};
    RequestId request_id_{kInvalidRequestId};
    std::uint32_t logical_token_position_{0};
    std::uint32_t layer_count_{0};
    std::uint32_t next_layer_{0};
    TokenTransactionPhase phase_{TokenTransactionPhase::Empty};
    EngineStream stream_{nullptr};

    friend class TokenTransactionBackend;
    friend void writeLayerBatch(LayerKvWriteBatch& batch);
    friend void attendLayerBatch(PagedDecodeBatch& batch);
};

// Validates every lane against the scalar transaction state machine, dispatches
// one backend batch operation, then advances/fails lanes independently.
void writeLayerBatch(LayerKvWriteBatch& batch);

// Validates every lane against the scalar TokenTransaction state machine,
// dispatches the backend batch operation, then advances/fails lanes separately.
void attendLayerBatch(PagedDecodeBatch& batch);

struct TokenReserveResult final {
    EngineKvStatus status{};
    TokenTransaction transaction{};

    [[nodiscard]] bool ok() const noexcept
    {
        return status.ok() && transaction.valid();
    }

    [[nodiscard]] explicit operator bool() const noexcept
    {
        return ok();
    }
};

struct EngineKvBackendSnapshot final {
    std::uint64_t request_count{0};
    std::uint64_t active_transaction_count{0};
    std::uint64_t committed_token_count{0};
    std::uint64_t batched_kv_write_submissions{0};
    std::uint64_t batched_kv_write_lanes{0};
    std::uint64_t batched_attention_submissions{0};
    std::uint64_t batched_attention_lanes{0};
    // Policy-neutral page telemetry. Fixed backends only populate primary;
    // Heterogeneous backends expose Micro as primary and Extent as secondary.
    std::uint16_t primary_page_tokens{0};
    std::uint16_t secondary_page_tokens{0};
    std::uint32_t primary_page_capacity{0};
    std::uint32_t secondary_page_capacity{0};
    std::uint32_t allocated_primary_pages{0};
    std::uint32_t allocated_secondary_pages{0};
    std::uint64_t successful_primary_allocations{0};
    std::uint64_t successful_secondary_allocations{0};
    std::uint64_t failed_primary_allocations{0};
    std::uint64_t failed_secondary_allocations{0};
    std::uint64_t storage_reserved_bytes{0};
    // Eager promotion is best-effort after a token commit. A failure means
    // the committed Micro layout was retained; it does not fail generation.
    std::uint64_t automatic_promotion_attempts{0};
    std::uint64_t automatic_promotion_successes{0};
    std::uint64_t automatic_promotion_skips{0};
    std::uint64_t automatic_promotion_failures{0};
};

// ModelRunner/Scheduler 只依赖该接口。Fixed 与 Heterogeneous 后端必须保持
// 相同模型布局、Stream、Workspace 和计时边界，唯一变量只能是分页策略。
class EngineKvBackend {
public:
    virtual ~EngineKvBackend() = default;

    EngineKvBackend(EngineKvBackend const&) = delete;
    EngineKvBackend& operator=(EngineKvBackend const&) = delete;

    [[nodiscard]] virtual EngineKvBackendKind kind() const noexcept = 0;
    [[nodiscard]] virtual EngineKvConfig config() const noexcept = 0;

    [[nodiscard]] virtual EngineKvStatus createRequest(
        RequestId request_id
    ) = 0;

    [[nodiscard]] virtual EngineKvStatus forkRequest(
        RequestId source_request_id,
        RequestId child_request_id
    ) = 0;

    [[nodiscard]] virtual EngineKvStatus releaseRequest(
        RequestId request_id
    ) = 0;

    [[nodiscard]] virtual TokenReserveResult reserveToken(
        ReserveTokenRequest const& request
    ) = 0;

    [[nodiscard]] virtual EngineKvBackendSnapshot snapshot() const = 0;
    [[nodiscard]] virtual bool checkInvariants() const = 0;

protected:
    EngineKvBackend() = default;
};

} // namespace kimkvcache
