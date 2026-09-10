#include "kim-kv/runtime/kv_cache_manager.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace kimkvcache {

struct KvCacheManager::InvariantCounters final {
    std::vector<std::uint32_t> micro_refs;
    std::vector<std::uint32_t> extent_refs;
    std::vector<std::uint32_t> micro_pins;
    std::vector<std::uint32_t> extent_pins;
    std::vector<std::uint32_t> micro_readers;
    std::vector<std::uint32_t> extent_readers;
    std::vector<std::uint8_t> micro_targets;
    std::vector<std::uint8_t> extent_targets;

    InvariantCounters(std::size_t micro_size, std::size_t extent_size)
        : micro_refs(micro_size, 0),
          extent_refs(extent_size, 0),
          micro_pins(micro_size, 0),
          extent_pins(extent_size, 0),
          micro_readers(micro_size, 0),
          extent_readers(extent_size, 0),
          micro_targets(micro_size, 0),
          extent_targets(extent_size, 0)
    {
    }

    [[nodiscard]] std::vector<std::uint32_t>& refs(PageKind kind)
    {
        return kind == PageKind::Micro ? micro_refs : extent_refs;
    }

    [[nodiscard]] std::vector<std::uint32_t> const& refs(
        PageKind kind) const
    {
        return kind == PageKind::Micro ? micro_refs : extent_refs;
    }

    [[nodiscard]] std::vector<std::uint32_t>& pins(PageKind kind)
    {
        return kind == PageKind::Micro ? micro_pins : extent_pins;
    }

    [[nodiscard]] std::vector<std::uint32_t> const& pins(
        PageKind kind) const
    {
        return kind == PageKind::Micro ? micro_pins : extent_pins;
    }

    [[nodiscard]] std::vector<std::uint32_t>& readers(PageKind kind)
    {
        return kind == PageKind::Micro
            ? micro_readers
            : extent_readers;
    }

    [[nodiscard]] std::vector<std::uint32_t> const& readers(
        PageKind kind) const
    {
        return kind == PageKind::Micro
            ? micro_readers
            : extent_readers;
    }

    [[nodiscard]] std::vector<std::uint8_t>& targets(PageKind kind)
    {
        return kind == PageKind::Micro ? micro_targets : extent_targets;
    }

    [[nodiscard]] std::vector<std::uint8_t> const& targets(
        PageKind kind) const
    {
        return kind == PageKind::Micro ? micro_targets : extent_targets;
    }
};

bool KvCacheManager::checkInvariants() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return checkInvariantsLocked();
}

bool KvCacheManager::checkInvariantsLocked() const
{
    if (!micro_pool_.checkInvariants()
        || !extent_pool_.checkInvariants()) {
        return false;
    }

    InvariantCounters expected(
        micro_runtime_.size(),
        extent_runtime_.size()
    );
    return checkRequestInvariantsLocked(expected)
        && checkPromotionInvariantsLocked(expected)
        && checkTokenReservationInvariantsLocked(expected)
        && checkLeaseInvariantsLocked(expected)
        && checkPoolInvariantsLocked(
            PageKind::Micro,
            micro_pool_,
            micro_runtime_,
            expected
        )
        && checkPoolInvariantsLocked(
            PageKind::Extent,
            extent_pool_,
            extent_runtime_,
            expected
        );
}

bool KvCacheManager::checkTokenReservationInvariantsLocked(
    InvariantCounters& expected) const
{
    if (request_token_reservations_.size() != token_reservations_.size()) {
        return false;
    }
    for (auto const& [reservation_id, transaction] : token_reservations_) {
        if (reservation_id == kInvalidKvTokenReservationId
            || transaction.reservation_id != reservation_id
            || transaction.request_id == kInvalidRequestId
            || !transaction.candidate.checkInvariants()) {
            return false;
        }
        auto const request_iterator = requests_.find(transaction.request_id);
        auto const request_reservation = request_token_reservations_.find(
            transaction.request_id
        );
        if (request_iterator == requests_.end()
            || request_reservation == request_token_reservations_.end()
            || request_reservation->second != reservation_id) {
            return false;
        }
        BlockTable const& before = request_iterator->second.table;
        if (before.version() != transaction.prepared_table_version
            || transaction.candidate.version() != before.version() + 1
            || transaction.candidate.tokenCount()
                != before.tokenCount() + transaction.token_count
            || transaction.token_count == 0) {
            return false;
        }

        if (transaction.staged_pages.empty()
            && !transaction.existing_mutable.isStructurallyValid()) {
            return false;
        }
        for (StagedPage const& staged : transaction.staged_pages) {
            RuntimeSlot const* target = runtimeSlotLocked(staged.handle);
            auto& targets = expected.targets(PageKind::Micro);
            if (target == nullptr
                || target->state != PageState::CopyTarget
                || target->ref_count != 0
                || target->promotion_pins != 0
                || target->mutable_owner != kInvalidRequestId
                || staged.entry_index >= transaction.candidate.entries().size()
                || transaction.candidate.entries()[staged.entry_index].handle
                    != staged.handle
                || staged.handle.slot >= targets.size()
                || targets[staged.handle.slot] != 0) {
                return false;
            }
            std::uint16_t const expected_staged_tokens =
                transaction.replaced_sealed_tail.isStructurallyValid()
                    && !before.entries().empty()
                    && staged.entry_index + 1 == before.entries().size()
                ? before.entries().back().valid_tokens
                : 0;
            if (target->valid_tokens != expected_staged_tokens) {
                return false;
            }
            targets[staged.handle.slot] = 1;
        }
        if (transaction.existing_mutable.isStructurallyValid()) {
            RuntimeSlot const* existing = runtimeSlotLocked(
                transaction.existing_mutable
            );
            if (existing == nullptr
                || existing->state != PageState::Mutable
                || existing->ref_count != 1
                || existing->mutable_owner != transaction.request_id
                || before.entries().empty()
                || before.entries().back().handle
                    != transaction.existing_mutable) {
                return false;
            }
        }
        if (transaction.replaced_sealed_tail.isStructurallyValid()) {
            RuntimeSlot const* replaced = runtimeSlotLocked(
                transaction.replaced_sealed_tail
            );
            if (replaced == nullptr
                || replaced->state != PageState::Sealed
                || replaced->ref_count == 0
                || before.entries().empty()
                || before.entries().back().handle
                    != transaction.replaced_sealed_tail) {
                return false;
            }
        }
    }
    return true;
}

bool KvCacheManager::checkRequestInvariantsLocked(
    InvariantCounters& expected) const
{
    for (auto const& request_pair : requests_) {
        RequestId const request_id = request_pair.first;
        BlockTable const& table = request_pair.second.table;
        if (request_id == kInvalidRequestId
            || !table.checkInvariants()) {
            return false;
        }

        for (std::size_t index = 0;
             index < table.entries_.size();
             ++index) {
            MappingEntry const& entry = table.entries_[index];
            RuntimeSlot const* runtime = runtimeSlotLocked(entry.handle);
            if (runtime == nullptr
                || runtime->state == PageState::Free
                || runtime->state == PageState::CopyTarget
                || runtime->state == PageState::Retiring
                || runtime->valid_tokens != entry.valid_tokens
                || runtime->ref_count == 0) {
                return false;
            }

            if (runtime->state == PageState::Mutable) {
                bool const is_tail = index + 1 == table.entries_.size();
                if (!is_tail
                    || entry.kind != PageKind::Micro
                    || runtime->ref_count != 1
                    || runtime->mutable_owner != request_id
                    || runtime->valid_tokens == 0
                    || runtime->valid_tokens >= kMicroPageTokenCapacity) {
                    return false;
                }
            } else if (runtime->state == PageState::Sealed) {
                if (runtime->mutable_owner != kInvalidRequestId) {
                    return false;
                }
            } else {
                return false;
            }

            auto& refs = expected.refs(entry.kind);
            if (entry.handle.slot >= refs.size()
                || refs[entry.handle.slot]
                    == std::numeric_limits<std::uint32_t>::max()) {
                return false;
            }
            ++refs[entry.handle.slot];
        }
    }
    return true;
}

bool KvCacheManager::checkPromotionInvariantsLocked(
    InvariantCounters& expected) const
{
    for (auto const& promotion_pair : promotions_) {
        PromotionId const promotion_id = promotion_pair.first;
        PromotionTransaction const& transaction = promotion_pair.second;
        if (promotion_id == kInvalidPromotionId
            || transaction.promotion_id != promotion_id
            || transaction.request_id == kInvalidRequestId
            || transaction.target_handle.kind != PageKind::Extent) {
            return false;
        }

        auto const request_iterator = requests_.find(transaction.request_id);
        if (request_iterator == requests_.end()) {
            return false;
        }
        BlockTable const& table = request_iterator->second.table;
        if (transaction.prepared_table_version > table.version_) {
            return false;
        }

        RuntimeSlot const* target =
            runtimeSlotLocked(transaction.target_handle);
        auto& extent_targets = expected.targets(PageKind::Extent);
        if (target == nullptr
            || target->state != PageState::CopyTarget
            || target->valid_tokens != kExtentPageTokenCapacity
            || target->ref_count != 0
            || target->promotion_pins != 0
            || target->mutable_owner != kInvalidRequestId
            || transaction.target_handle.slot >= extent_targets.size()
            || extent_targets[transaction.target_handle.slot] != 0) {
            return false;
        }
        extent_targets[transaction.target_handle.slot] = 1;

        auto const first_source = std::find_if(
            table.entries_.begin(),
            table.entries_.end(),
            [&transaction](MappingEntry const& entry) {
                return entry.handle == transaction.source_handles[0];
            }
        );
        if (first_source == table.entries_.end()) {
            return false;
        }

        std::size_t const current_source_index =
            static_cast<std::size_t>(
                std::distance(table.entries_.begin(), first_source)
            );
        if (table.entries_.size() - current_source_index
            < kPromotionSourcePageCount) {
            return false;
        }

        auto& micro_pins = expected.pins(PageKind::Micro);
        for (std::size_t offset = 0;
             offset < kPromotionSourcePageCount;
             ++offset) {
            PageHandle const source_handle =
                transaction.source_handles[offset];
            MappingEntry const& entry =
                table.entries_[current_source_index + offset];
            RuntimeSlot const* source = runtimeSlotLocked(source_handle);
            std::uint64_t const expected_begin =
                static_cast<std::uint64_t>(
                    transaction.logical_token_begin
                ) + offset * kMicroPageTokenCapacity;

            if (expected_begin
                    > std::numeric_limits<std::uint32_t>::max()
                || source_handle.kind != PageKind::Micro
                || entry.handle != source_handle
                || entry.kind != PageKind::Micro
                || entry.logical_token_begin != expected_begin
                || entry.valid_tokens != kMicroPageTokenCapacity
                || source == nullptr
                || source->state != PageState::Sealed
                || source->valid_tokens != kMicroPageTokenCapacity
                || source->ref_count == 0
                || source->promotion_pins == 0
                || source->mutable_owner != kInvalidRequestId
                || source_handle.slot >= micro_pins.size()
                || micro_pins[source_handle.slot] != 0) {
                return false;
            }
            micro_pins[source_handle.slot] = 1;
        }
    }
    return true;
}

bool KvCacheManager::checkLeaseInvariantsLocked(
    InvariantCounters& expected) const
{
    for (auto const& lease_pair : page_leases_) {
        PageLeaseId const lease_id = lease_pair.first;
        PageLease const& lease = lease_pair.second;
        if (lease_id == kInvalidPageLeaseId
            || lease.lease_id != lease_id) {
            return false;
        }

        for (PageHandle const handle : lease.handles) {
            RuntimeSlot const* runtime = runtimeSlotLocked(handle);
            auto& readers = expected.readers(handle.kind);
            if (runtime == nullptr
                || runtime->state == PageState::Free
                || runtime->inflight_readers == 0
                || handle.slot >= readers.size()
                || readers[handle.slot]
                    == std::numeric_limits<std::uint32_t>::max()) {
                return false;
            }
            ++readers[handle.slot];
        }
    }
    return true;
}

bool KvCacheManager::checkPoolInvariantsLocked(
    PageKind kind,
    PagePool const& pool,
    std::vector<RuntimeSlot> const& runtime_slots,
    InvariantCounters const& expected) const
{
    auto const& refs = expected.refs(kind);
    auto const& pins = expected.pins(kind);
    auto const& readers = expected.readers(kind);
    auto const& targets = expected.targets(kind);
    if (runtime_slots.size() != refs.size()
        || runtime_slots.size() != pins.size()
        || runtime_slots.size() != readers.size()
        || runtime_slots.size() != targets.size()) {
        return false;
    }

    std::uint32_t observed_allocated = 0;
    for (std::size_t index = 0;
         index < runtime_slots.size();
         ++index) {
        RuntimeSlot const& runtime = runtime_slots[index];
        if (runtime.state == PageState::Free) {
            if (runtime.valid_tokens != 0
                || runtime.ref_count != 0
                || runtime.promotion_pins != 0
                || runtime.inflight_readers != 0
                || runtime.mutable_owner != kInvalidRequestId
                || refs[index] != 0
                || pins[index] != 0
                || readers[index] != 0
                || targets[index] != 0) {
                return false;
            }
            continue;
        }

        ++observed_allocated;
        if (runtime.generation == PageHandle::kInvalidGeneration) {
            return false;
        }

        PageHandle const handle{
            kind,
            static_cast<std::uint32_t>(index),
            runtime.generation,
        };
        if (pool.validate(handle) != PagePoolError::None
            || runtime.ref_count != refs[index]
            || runtime.promotion_pins != pins[index]
            || runtime.inflight_readers != readers[index]) {
            return false;
        }

        std::uint16_t const capacity = pageTokenCapacity(kind);
        bool const is_expected_target = targets[index] != 0;
        switch (runtime.state) {
        case PageState::Free:
            return false;
        case PageState::Mutable:
            if (kind != PageKind::Micro
                || runtime.ref_count != 1
                || runtime.valid_tokens == 0
                || runtime.valid_tokens >= capacity
                || runtime.mutable_owner == kInvalidRequestId
                || runtime.promotion_pins != 0
                || is_expected_target) {
                return false;
            }
            break;
        case PageState::Sealed:
            if (runtime.ref_count == 0
                || runtime.valid_tokens == 0
                || runtime.valid_tokens > capacity
                || runtime.mutable_owner != kInvalidRequestId
                || is_expected_target) {
                return false;
            }
            break;
        case PageState::CopyTarget:
            if (!is_expected_target
                || runtime.ref_count != 0
                || runtime.promotion_pins != 0
                || runtime.mutable_owner != kInvalidRequestId) {
                return false;
            }
            if ((kind == PageKind::Extent
                    && runtime.valid_tokens != capacity)
                || (kind == PageKind::Micro
                    && runtime.valid_tokens >= capacity)) {
                return false;
            }
            break;
        case PageState::Retiring:
            if (runtime.ref_count != 0
                || runtime.mutable_owner != kInvalidRequestId
                || is_expected_target
                || (runtime.promotion_pins == 0
                    && runtime.inflight_readers == 0)) {
                return false;
            }
            break;
        }
    }

    PagePoolSnapshot const pool_snapshot = pool.snapshot();
    return pool_snapshot.allocated_slots == observed_allocated
        && pool_snapshot.capacityBalanced();
}

} // namespace kimkvcache
