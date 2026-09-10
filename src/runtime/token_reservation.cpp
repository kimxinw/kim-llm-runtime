#include "kim-kv/runtime/kv_cache_manager.h"

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

namespace kimkvcache {

bool KvCacheManager::hasTokenReservationLocked(
    RequestId request_id) const noexcept
{
    return request_token_reservations_.find(request_id)
        != request_token_reservations_.end();
}

KvTokenReservationId KvCacheManager::allocateTokenReservationIdLocked()
{
    std::size_t const maximum_attempts = token_reservations_.size() + 1;
    for (std::size_t attempt = 0; attempt < maximum_attempts; ++attempt) {
        KvTokenReservationId const candidate = next_token_reservation_id_++;
        if (next_token_reservation_id_ == kInvalidKvTokenReservationId) {
            next_token_reservation_id_ = 1;
        }
        if (candidate != kInvalidKvTokenReservationId
            && token_reservations_.find(candidate)
                == token_reservations_.end()) {
            return candidate;
        }
    }
    return kInvalidKvTokenReservationId;
}

TokenReservationResult KvCacheManager::reserveToken(
    RequestId request_id,
    std::uint32_t expected_committed_tokens)
{
    return reserveTokens(request_id, expected_committed_tokens, 1);
}

TokenReservationResult KvCacheManager::reserveTokens(
    RequestId request_id,
    std::uint32_t expected_committed_tokens,
    std::uint32_t token_count)
{
    TokenReservationResult result;
    result.request_id = request_id;
    result.logical_token_position = expected_committed_tokens;
    result.token_count = token_count;
    if (request_id == kInvalidRequestId || token_count == 0) {
        result.error = KvCacheError::InvalidArgument;
        return result;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto const request_iterator = requests_.find(request_id);
    if (request_iterator == requests_.end()) {
        result.error = KvCacheError::RequestNotFound;
        return result;
    }
    if (hasTokenReservationLocked(request_id)) {
        result.error = KvCacheError::RequestConflict;
        return result;
    }
    if (std::any_of(
            promotions_.begin(), promotions_.end(),
            [request_id](auto const& entry) {
                return entry.second.request_id == request_id;
            })) {
        result.error = KvCacheError::RequestConflict;
        return result;
    }

    BlockTable const& before = request_iterator->second.table;
    if (before.tokenCount() != expected_committed_tokens
        || before.version_ == std::numeric_limits<std::uint64_t>::max()
        || token_count > std::numeric_limits<std::uint32_t>::max()
            - expected_committed_tokens) {
        result.error = KvCacheError::InvalidState;
        return result;
    }

    BlockTable candidate = before;
    std::size_t const maximum_new_pages =
        (static_cast<std::size_t>(token_count)
            + kMicroPageTokenCapacity - 1)
        / kMicroPageTokenCapacity + 1;
    candidate.entries_.reserve(candidate.entries_.size() + maximum_new_pages);
    std::vector<StagedPage> staged_pages;
    staged_pages.reserve(maximum_new_pages);
    constexpr std::size_t kNoActiveEntry =
        std::numeric_limits<std::size_t>::max();
    std::size_t active_entry_index = kNoActiveEntry;
    PageHandle existing_mutable = PageHandle::invalid();
    PageHandle replaced_sealed_tail = PageHandle::invalid();

    if (!candidate.entries_.empty()) {
        std::size_t const tail_index = candidate.entries_.size() - 1;
        MappingEntry& tail = candidate.entries_[tail_index];
        RuntimeSlot* runtime = runtimeSlotLocked(tail.handle);
        if (runtime == nullptr || runtime->valid_tokens != tail.valid_tokens) {
            result.error = KvCacheError::InternalInvariantViolation;
            return result;
        }
        if (tail.kind == PageKind::Micro
            && tail.valid_tokens < kMicroPageTokenCapacity
            && runtime->state == PageState::Mutable) {
            if (runtime->ref_count != 1
                || runtime->mutable_owner != request_id) {
                result.error = KvCacheError::InternalInvariantViolation;
                return result;
            }
            existing_mutable = tail.handle;
            active_entry_index = tail_index;
        } else if (tail.kind == PageKind::Micro
            && tail.valid_tokens < kMicroPageTokenCapacity
            && runtime->state == PageState::Sealed) {
            PageHandle target = PageHandle::invalid();
            KvCacheError const allocation_error =
                allocateStagedMicroLocked(tail.valid_tokens, target);
            if (allocation_error != KvCacheError::None) {
                result.error = allocation_error;
                return result;
            }
            staged_pages.push_back(StagedPage{target, tail_index});
            replaced_sealed_tail = tail.handle;
            tail.handle = target;
            active_entry_index = tail_index;
        } else if (runtime->state != PageState::Sealed) {
            result.error = KvCacheError::InternalInvariantViolation;
            return result;
        }
    }

    std::uint32_t remaining_tokens = token_count;
    std::uint32_t logical_end = expected_committed_tokens;
    while (remaining_tokens != 0) {
        if (active_entry_index == kNoActiveEntry) {
            PageHandle target = PageHandle::invalid();
            KvCacheError const allocation_error =
                allocateStagedMicroLocked(0, target);
            if (allocation_error != KvCacheError::None) {
                rollbackStagedPagesLocked(staged_pages);
                result.error = allocation_error;
                return result;
            }
            std::size_t const entry_index = candidate.entries_.size();
            candidate.entries_.push_back(MappingEntry{
                logical_end, 0, PageKind::Micro, target,
            });
            staged_pages.push_back(StagedPage{target, entry_index});
            active_entry_index = entry_index;
        }
        MappingEntry& active = candidate.entries_[active_entry_index];
        std::uint32_t const appended = std::min(
            remaining_tokens,
            static_cast<std::uint32_t>(
                kMicroPageTokenCapacity - active.valid_tokens)
        );
        active.valid_tokens = static_cast<std::uint16_t>(
            active.valid_tokens + appended
        );
        logical_end += appended;
        remaining_tokens -= appended;
        if (active.valid_tokens == kMicroPageTokenCapacity) {
            active_entry_index = kNoActiveEntry;
        }
    }

    candidate.version_ = before.version_ + 1;
    if (!candidate.checkInvariants()) {
        rollbackStagedPagesLocked(staged_pages);
        result.error = KvCacheError::InternalInvariantViolation;
        return result;
    }

    KvTokenReservationId const reservation_id =
        allocateTokenReservationIdLocked();
    if (reservation_id == kInvalidKvTokenReservationId) {
        rollbackStagedPagesLocked(staged_pages);
        result.error = KvCacheError::InternalInvariantViolation;
        return result;
    }

    auto const reservation_insertion = token_reservations_.emplace(
        reservation_id,
        TokenReservation{
            reservation_id,
            request_id,
            before.version_,
            token_count,
            candidate,
            staged_pages,
            existing_mutable,
            replaced_sealed_tail,
        }
    );
    if (!reservation_insertion.second) {
        rollbackStagedPagesLocked(staged_pages);
        result.error = KvCacheError::InternalInvariantViolation;
        return result;
    }
    auto const request_insertion = request_token_reservations_.emplace(
        request_id,
        reservation_id
    );
    if (!request_insertion.second) {
        rollbackStagedPagesLocked(
            reservation_insertion.first->second.staged_pages
        );
        token_reservations_.erase(reservation_insertion.first);
        result.error = KvCacheError::InternalInvariantViolation;
        return result;
    }

    result.error = KvCacheError::None;
    result.reservation_id = reservation_id;
    result.before = before;
    result.reserved = std::move(candidate);
    return result;
}

KvCacheError KvCacheManager::commitTokenReservation(
    KvTokenReservationId reservation_id)
{
    if (reservation_id == kInvalidKvTokenReservationId) {
        return KvCacheError::InvalidArgument;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto transaction_iterator = token_reservations_.find(reservation_id);
    if (transaction_iterator == token_reservations_.end()) {
        return KvCacheError::TokenReservationNotFound;
    }
    TokenReservation& transaction = transaction_iterator->second;
    auto request_iterator = requests_.find(transaction.request_id);
    if (request_iterator == requests_.end()
        || request_iterator->second.table.version_
            != transaction.prepared_table_version) {
        return KvCacheError::RequestConflict;
    }

    RuntimeSlot* existing = transaction.existing_mutable.isStructurallyValid()
        ? runtimeSlotLocked(transaction.existing_mutable)
        : nullptr;
    RuntimeSlot* replaced =
        transaction.replaced_sealed_tail.isStructurallyValid()
        ? runtimeSlotLocked(transaction.replaced_sealed_tail)
        : nullptr;

    if ((existing != nullptr
            && (existing->state != PageState::Mutable
                || existing->ref_count != 1
                || existing->mutable_owner != transaction.request_id))
        || (replaced != nullptr
            && (replaced->state != PageState::Sealed
                || replaced->ref_count == 0))) {
        return KvCacheError::InternalInvariantViolation;
    }
    for (StagedPage const& staged : transaction.staged_pages) {
        RuntimeSlot* runtime = runtimeSlotLocked(staged.handle);
        if (runtime == nullptr || runtime->state != PageState::CopyTarget
            || runtime->ref_count != 0
            || staged.entry_index >= transaction.candidate.entries_.size()
            || transaction.candidate.entries_[staged.entry_index].handle
                != staged.handle) {
            return KvCacheError::InternalInvariantViolation;
        }
    }

    if (transaction.replaced_sealed_tail.isStructurallyValid()) {
        KvCacheError const decrement_error = decrementReferenceLocked(
            transaction.replaced_sealed_tail
        );
        if (decrement_error != KvCacheError::None) {
            return decrement_error;
        }
    }
    if (existing != nullptr) {
        for (MappingEntry const& entry : transaction.candidate.entries_) {
            if (entry.handle == transaction.existing_mutable) {
                existing->valid_tokens = entry.valid_tokens;
                if (entry.valid_tokens == kMicroPageTokenCapacity) {
                    existing->state = PageState::Sealed;
                    existing->mutable_owner = kInvalidRequestId;
                }
                break;
            }
        }
    }
    for (StagedPage const& staged : transaction.staged_pages) {
        RuntimeSlot* runtime = runtimeSlotLocked(staged.handle);
        MappingEntry const& entry =
            transaction.candidate.entries_[staged.entry_index];
        runtime->valid_tokens = entry.valid_tokens;
        runtime->ref_count = 1;
        if (entry.valid_tokens == kMicroPageTokenCapacity) {
            runtime->state = PageState::Sealed;
            runtime->mutable_owner = kInvalidRequestId;
        } else {
            runtime->state = PageState::Mutable;
            runtime->mutable_owner = transaction.request_id;
        }
    }

    request_iterator->second.table = std::move(transaction.candidate);
    request_token_reservations_.erase(transaction.request_id);
    token_reservations_.erase(transaction_iterator);
    return KvCacheError::None;
}

KvCacheError KvCacheManager::rollbackTokenReservationLocked(
    KvTokenReservationId reservation_id)
{
    auto const iterator = token_reservations_.find(reservation_id);
    if (iterator == token_reservations_.end()) {
        return KvCacheError::TokenReservationNotFound;
    }
    TokenReservation const& transaction = iterator->second;
    rollbackStagedPagesLocked(transaction.staged_pages);
    request_token_reservations_.erase(transaction.request_id);
    token_reservations_.erase(iterator);
    return KvCacheError::None;
}

KvCacheError KvCacheManager::rollbackTokenReservation(
    KvTokenReservationId reservation_id)
{
    if (reservation_id == kInvalidKvTokenReservationId) {
        return KvCacheError::InvalidArgument;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return rollbackTokenReservationLocked(reservation_id);
}

PageLeaseAcquireResult KvCacheManager::acquireTokenReservationLease(
    KvTokenReservationId reservation_id)
{
    PageLeaseAcquireResult result;
    if (reservation_id == kInvalidKvTokenReservationId) {
        result.error = KvCacheError::InvalidArgument;
        return result;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto const iterator = token_reservations_.find(reservation_id);
    if (iterator == token_reservations_.end()) {
        result.error = KvCacheError::TokenReservationNotFound;
        return result;
    }
    TokenReservation const& transaction = iterator->second;
    std::vector<PageHandle> handles;
    handles.reserve(transaction.candidate.entries_.size() + 1);
    for (MappingEntry const& entry : transaction.candidate.entries_) {
        handles.push_back(entry.handle);
    }
    if (transaction.replaced_sealed_tail.isStructurallyValid()) {
        handles.push_back(transaction.replaced_sealed_tail);
    }
    return acquirePageLeaseLocked(
        std::move(handles),
        transaction.candidate
    );
}

} // namespace kimkvcache
