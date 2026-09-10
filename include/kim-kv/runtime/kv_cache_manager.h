#pragma once

#include "kim-kv/core/block_table.h"
#include "kim-kv/core/page_pool.h"
#include "kim-kv/core/page_state.h"
#include "kim-kv/runtime/page_lease.h"
#include "kim-kv/runtime/promotion.h"
#include "kim-kv/runtime/token_reservation.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace kimkvcache {

struct PageMetadataSnapshot final {
    PageHandle handle{PageHandle::invalid()};
    PageState state{PageState::Free};
    std::uint16_t valid_tokens{0};
    std::uint32_t ref_count{0};
    std::uint32_t promotion_pins{0};
    std::uint32_t inflight_readers{0};
    RequestId mutable_owner{kInvalidRequestId};

    [[nodiscard]] constexpr bool shared() const noexcept
    {
        return state == PageState::Sealed && ref_count > 1;
    }
};

struct KvCacheManagerSnapshot final {
    std::uint64_t request_count{0};
    std::uint64_t promotion_count{0};
    std::uint64_t page_lease_count{0};
    std::uint64_t token_reservation_count{0};
    PagePoolSnapshot micro_pool{};
    PagePoolSnapshot extent_pool{};
};

class KvCacheManager final {
public:
    KvCacheManager(
        std::uint32_t micro_capacity,
        std::uint32_t extent_capacity
    );

    ~KvCacheManager() = default;

    KvCacheManager(KvCacheManager const&) = delete;
    KvCacheManager& operator=(KvCacheManager const&) = delete;

    KvCacheManager(KvCacheManager&&) = delete;
    KvCacheManager& operator=(KvCacheManager&&) = delete;

    [[nodiscard]] KvCacheError createRequest(RequestId request_id);

    [[nodiscard]] KvCacheError append(
        RequestId request_id,
        std::uint32_t token_count
    );

    [[nodiscard]] TokenReservationResult reserveToken(
        RequestId request_id,
        std::uint32_t expected_committed_tokens
    );

    [[nodiscard]] TokenReservationResult reserveTokens(
        RequestId request_id,
        std::uint32_t expected_committed_tokens,
        std::uint32_t token_count
    );

    [[nodiscard]] KvCacheError commitTokenReservation(
        KvTokenReservationId reservation_id
    );

    [[nodiscard]] KvCacheError rollbackTokenReservation(
        KvTokenReservationId reservation_id
    );

    [[nodiscard]] KvCacheError sealTail(RequestId request_id);

    [[nodiscard]] KvCacheError forkRequest(
        RequestId source_request_id,
        RequestId child_request_id
    );

    [[nodiscard]] KvCacheError releaseRequest(RequestId request_id);

    // Prepare 只固定元数据和物理页生命周期，不执行数据复制。
    // 调用方应在异步 Copy 成功后调用 commitPromotion；任何失败路径
    // 都必须调用 rollbackPromotion。K4 会在这两个调用之间接入 CUDA。
    [[nodiscard]] PromotionPrepareResult preparePromotion(
        RequestId request_id,
        std::uint32_t logical_token_begin
    );

    [[nodiscard]] KvCacheError commitPromotion(
        PromotionId promotion_id
    );

    [[nodiscard]] KvCacheError rollbackPromotion(
        PromotionId promotion_id
    );

    // 原子快照当前 BlockTable 并为全部可见 Page 增加 inflight_readers。
    // 空请求同样返回一个有效租约，以便调用方统一处理生命周期。
    [[nodiscard]] PageLeaseAcquireResult acquireRequestReadLease(
        RequestId request_id
    );

    // 固定一个已 Prepare Promotion 的 8 个 Source 和 1 个 Target。
    // 该租约必须覆盖 CUDA Copy 从提交到 Event 完成的整个区间。
    [[nodiscard]] PageLeaseAcquireResult acquirePromotionIoLease(
        PromotionId promotion_id
    );

    [[nodiscard]] PageLeaseAcquireResult acquireTokenReservationLease(
        KvTokenReservationId reservation_id
    );

    [[nodiscard]] KvCacheError releasePageLease(
        PageLeaseId lease_id
    );

    [[nodiscard]] std::optional<BlockTable> blockTable(
        RequestId request_id) const;

    [[nodiscard]] std::optional<PageMetadataSnapshot> pageMetadata(
        PageHandle handle) const;

    [[nodiscard]] std::optional<PromotionTransactionSnapshot>
    promotionTransaction(PromotionId promotion_id) const;

    [[nodiscard]] KvCacheManagerSnapshot snapshot() const;

    [[nodiscard]] bool checkInvariants() const;

private:
    struct InvariantCounters;

    struct RuntimeSlot final {
        PageState state{PageState::Free};
        std::uint32_t generation{0};
        std::uint16_t valid_tokens{0};
        std::uint32_t ref_count{0};
        std::uint32_t promotion_pins{0};
        std::uint32_t inflight_readers{0};
        RequestId mutable_owner{kInvalidRequestId};
    };

    struct RequestState final {
        BlockTable table;
    };

    struct StagedPage final {
        PageHandle handle{PageHandle::invalid()};
        std::size_t entry_index{0};
    };

    struct PromotionTransaction final {
        PromotionId promotion_id{kInvalidPromotionId};
        RequestId request_id{kInvalidRequestId};
        std::uint64_t prepared_table_version{0};
        std::uint32_t logical_token_begin{0};
        std::size_t source_entry_index{0};
        std::array<PageHandle, kPromotionSourcePageCount> source_handles{};
        PageHandle target_handle{PageHandle::invalid()};
    };

    struct TokenReservation final {
        KvTokenReservationId reservation_id{kInvalidKvTokenReservationId};
        RequestId request_id{kInvalidRequestId};
        std::uint64_t prepared_table_version{0};
        std::uint32_t token_count{0};
        BlockTable candidate{};
        std::vector<StagedPage> staged_pages{};
        PageHandle existing_mutable{PageHandle::invalid()};
        PageHandle replaced_sealed_tail{PageHandle::invalid()};
    };

    struct PageLease final {
        PageLeaseId lease_id{kInvalidPageLeaseId};
        std::vector<PageHandle> handles{};
    };

    [[nodiscard]] RuntimeSlot* runtimeSlotLocked(
        PageHandle handle) noexcept;

    [[nodiscard]] RuntimeSlot const* runtimeSlotLocked(
        PageHandle handle) const noexcept;

    [[nodiscard]] KvCacheError allocateStagedMicroLocked(
        std::uint16_t initial_valid_tokens,
        PageHandle& handle
    );

    [[nodiscard]] KvCacheError allocatePromotionTargetLocked(
        PageHandle& handle
    );

    void rollbackStagedPagesLocked(
        std::vector<StagedPage> const& staged_pages) noexcept;

    [[nodiscard]] KvCacheError decrementReferenceLocked(
        PageHandle handle
    );

    [[nodiscard]] KvCacheError freePageLocked(PageHandle handle);

    [[nodiscard]] PromotionId allocatePromotionIdLocked();

    [[nodiscard]] KvTokenReservationId
    allocateTokenReservationIdLocked();

    [[nodiscard]] PageLeaseId allocatePageLeaseIdLocked();

    [[nodiscard]] PageLeaseAcquireResult acquirePageLeaseLocked(
        std::vector<PageHandle> handles,
        BlockTable table
    );

    [[nodiscard]] KvCacheError rollbackPromotionLocked(
        PromotionId promotion_id
    );

    [[nodiscard]] KvCacheError rollbackTokenReservationLocked(
        KvTokenReservationId reservation_id
    );

    [[nodiscard]] bool hasTokenReservationLocked(
        RequestId request_id
    ) const noexcept;

    [[nodiscard]] KvCacheError rollbackPromotionsForRequestLocked(
        RequestId request_id
    );

    void resetFreedRuntimeLocked(RuntimeSlot& runtime) noexcept;

    [[nodiscard]] bool checkInvariantsLocked() const;

    [[nodiscard]] bool checkRequestInvariantsLocked(
        InvariantCounters& expected) const;

    [[nodiscard]] bool checkPromotionInvariantsLocked(
        InvariantCounters& expected) const;

    [[nodiscard]] bool checkTokenReservationInvariantsLocked(
        InvariantCounters& expected) const;

    [[nodiscard]] bool checkLeaseInvariantsLocked(
        InvariantCounters& expected) const;

    [[nodiscard]] bool checkPoolInvariantsLocked(
        PageKind kind,
        PagePool const& pool,
        std::vector<RuntimeSlot> const& runtime_slots,
        InvariantCounters const& expected) const;

    PagePool micro_pool_;
    PagePool extent_pool_;

    std::vector<RuntimeSlot> micro_runtime_;
    std::vector<RuntimeSlot> extent_runtime_;

    std::unordered_map<RequestId, RequestState> requests_;
    std::unordered_map<PromotionId, PromotionTransaction> promotions_;
    std::unordered_map<KvTokenReservationId, TokenReservation>
        token_reservations_;
    std::unordered_map<RequestId, KvTokenReservationId>
        request_token_reservations_;
    std::unordered_map<PageLeaseId, PageLease> page_leases_;

    PromotionId next_promotion_id_{1};
    KvTokenReservationId next_token_reservation_id_{1};
    PageLeaseId next_page_lease_id_{1};

    mutable std::mutex mutex_;
};

} // namespace kimkvcache
