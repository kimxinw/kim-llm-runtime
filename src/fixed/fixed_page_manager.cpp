#include "kim-kv/fixed/fixed_page_manager.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kimkvcache {
namespace {

[[nodiscard]] std::uint16_t validateTokensPerPage(
    std::uint16_t tokens_per_page)
{
    if (tokens_per_page == 0) {
        throw std::invalid_argument(
            "fixed page token capacity must be greater than zero"
        );
    }
    return tokens_per_page;
}

} // namespace

FixedPageManager::FixedPageManager(
    std::uint16_t tokens_per_page,
    std::uint32_t page_capacity)
    : tokens_per_page_(validateTokensPerPage(tokens_per_page))
    , pool_(PageKind::Micro, page_capacity)
    , runtime_(page_capacity)
{
}

std::uint16_t FixedPageManager::tokensPerPage() const noexcept
{
    return tokens_per_page_;
}

KvCacheError FixedPageManager::createRequest(RequestId request_id)
{
    if (request_id == kInvalidRequestId) {
        return KvCacheError::InvalidArgument;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto const result = requests_.emplace(
        request_id,
        RequestState{}
    );

    if (!result.second) {
        return KvCacheError::RequestAlreadyExists;
    }

    return KvCacheError::None;
}

FixedPageManager::RuntimeSlot* FixedPageManager::runtimeSlotLocked(
    PageHandle handle) noexcept
{
    if (!handle.isStructurallyValid()
        || handle.kind != PageKind::Micro) {
        return nullptr;
    }

    if (handle.slot >= runtime_.size()
        || runtime_[handle.slot].generation != handle.generation) {
        return nullptr;
    }

    return &runtime_[handle.slot];
}

FixedPageManager::RuntimeSlot const* FixedPageManager::runtimeSlotLocked(
    PageHandle handle) const noexcept
{
    if (!handle.isStructurallyValid()
        || handle.kind != PageKind::Micro) {
        return nullptr;
    }

    if (handle.slot >= runtime_.size()
        || runtime_[handle.slot].generation != handle.generation) {
        return nullptr;
    }

    return &runtime_[handle.slot];
}

KvCacheError FixedPageManager::allocateStagedLocked(
    std::uint16_t initial_valid_tokens,
    PageHandle& handle)
{
    if (initial_valid_tokens >= tokens_per_page_) {
        return KvCacheError::InvalidArgument;
    }

    PageAllocationResult const allocation = pool_.allocate();

    if (!allocation.ok()) {
        return allocation.error == PagePoolError::Exhausted
            ? KvCacheError::ResourceExhausted
            : KvCacheError::InternalInvariantViolation;
    }

    handle = allocation.handle;

    peak_allocated_pages_ = std::max<std::uint64_t>(
        peak_allocated_pages_,
        pool_.snapshot().allocated_slots
    );

    if (handle.slot >= runtime_.size()) {
        static_cast<void>(pool_.release(handle));
        handle = PageHandle::invalid();
        return KvCacheError::InternalInvariantViolation;
    }

    RuntimeSlot& runtime = runtime_[handle.slot];

    if (runtime.state != PageState::Free || runtime.ref_count != 0) {
        static_cast<void>(pool_.release(handle));
        handle = PageHandle::invalid();
        return KvCacheError::InternalInvariantViolation;
    }

    // 与 Hetero 运行时一致：COW 目标和新页在 Commit 前保持 CopyTarget。
    runtime.state = PageState::CopyTarget;
    runtime.generation = handle.generation;
    runtime.valid_tokens = initial_valid_tokens;
    runtime.ref_count = 0;
    runtime.mutable_owner = kInvalidRequestId;

    return KvCacheError::None;
}

void FixedPageManager::resetFreedRuntimeLocked(
    RuntimeSlot& runtime) noexcept
{
    // generation 保留，用于识别当前 Free Slot 的最后一代 Handle。
    runtime.state = PageState::Free;
    runtime.valid_tokens = 0;
    runtime.ref_count = 0;
    runtime.mutable_owner = kInvalidRequestId;
}

void FixedPageManager::rollbackStagedPagesLocked(
    std::vector<StagedPage> const& staged_pages) noexcept
{
    for (auto iterator = staged_pages.rbegin();
         iterator != staged_pages.rend();
         ++iterator) {
        RuntimeSlot* runtime = runtimeSlotLocked(iterator->handle);

        if (runtime == nullptr
            || runtime->state != PageState::CopyTarget
            || runtime->ref_count != 0) {
            continue;
        }

        PagePoolError const release_error =
            pool_.release(iterator->handle);

        if (release_error == PagePoolError::None) {
            resetFreedRuntimeLocked(*runtime);
        }
    }
}

KvCacheError FixedPageManager::freePageLocked(PageHandle handle)
{
    RuntimeSlot* runtime = runtimeSlotLocked(handle);

    if (runtime == nullptr || runtime->state == PageState::Free) {
        return KvCacheError::InternalInvariantViolation;
    }

    if (runtime->ref_count != 0) {
        return KvCacheError::InvalidState;
    }

    runtime->mutable_owner = kInvalidRequestId;

    if (pool_.validate(handle) != PagePoolError::None
        || pool_.release(handle) != PagePoolError::None) {
        return KvCacheError::InternalInvariantViolation;
    }

    resetFreedRuntimeLocked(*runtime);

    return KvCacheError::None;
}

KvCacheError FixedPageManager::decrementReferenceLocked(
    PageHandle handle)
{
    RuntimeSlot* runtime = runtimeSlotLocked(handle);

    if (runtime == nullptr
        || runtime->state == PageState::Free
        || runtime->state == PageState::CopyTarget
        || runtime->ref_count == 0) {
        return KvCacheError::InternalInvariantViolation;
    }

    --runtime->ref_count;

    if (runtime->ref_count != 0) {
        return KvCacheError::None;
    }

    return freePageLocked(handle);
}

KvCacheError FixedPageManager::append(
    RequestId request_id,
    std::uint32_t token_count)
{
    if (request_id == kInvalidRequestId || token_count == 0) {
        return KvCacheError::InvalidArgument;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto request_iterator = requests_.find(request_id);

    if (request_iterator == requests_.end()) {
        return KvCacheError::RequestNotFound;
    }

    if (hasTokenReservationLocked(request_id)) {
        return KvCacheError::RequestConflict;
    }

    RequestState& request = request_iterator->second;
    std::uint32_t const old_token_count = request.table.tokenCount();

    if (token_count >
        std::numeric_limits<std::uint32_t>::max() - old_token_count) {
        return KvCacheError::InvalidArgument;
    }

    BlockTable candidate = request.table;

    std::size_t const maximum_new_pages =
        (static_cast<std::size_t>(token_count)
            + tokens_per_page_ - 1)
        / tokens_per_page_
        + 1;

    candidate.entries_.reserve(
        candidate.entries_.size() + maximum_new_pages
    );

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
        RuntimeSlot* tail_runtime = runtimeSlotLocked(tail.handle);

        if (tail_runtime == nullptr
            || tail_runtime->valid_tokens != tail.valid_tokens) {
            return KvCacheError::InternalInvariantViolation;
        }

        if (tail.valid_tokens < tokens_per_page_) {
            if (tail_runtime->state == PageState::Mutable) {
                if (tail_runtime->ref_count != 1
                    || tail_runtime->mutable_owner != request_id) {
                    return KvCacheError::InternalInvariantViolation;
                }

                existing_mutable = tail.handle;
                active_entry_index = tail_index;
            } else if (tail_runtime->state == PageState::Sealed) {
                // Sealed 尾页写入即 Partial Tail COW。
                PageHandle copy_target = PageHandle::invalid();

                KvCacheError const allocation_error =
                    allocateStagedLocked(
                        tail.valid_tokens,
                        copy_target
                    );

                if (allocation_error != KvCacheError::None) {
                    return allocation_error;
                }

                staged_pages.push_back(
                    StagedPage{copy_target, tail_index}
                );

                replaced_sealed_tail = tail.handle;
                tail.handle = copy_target;
                active_entry_index = tail_index;
            } else {
                return KvCacheError::InvalidState;
            }
        } else if (tail_runtime->state != PageState::Sealed) {
            return KvCacheError::InternalInvariantViolation;
        }
    }

    std::uint32_t remaining_tokens = token_count;
    std::uint32_t logical_end = old_token_count;

    while (remaining_tokens != 0) {
        if (active_entry_index == kNoActiveEntry) {
            PageHandle new_page = PageHandle::invalid();

            KvCacheError const allocation_error =
                allocateStagedLocked(0, new_page);

            if (allocation_error != KvCacheError::None) {
                rollbackStagedPagesLocked(staged_pages);
                return allocation_error;
            }

            std::size_t const new_index = candidate.entries_.size();

            candidate.entries_.push_back(
                MappingEntry{
                    logical_end,
                    0,
                    PageKind::Micro,
                    new_page,
                }
            );

            staged_pages.push_back(
                StagedPage{new_page, new_index}
            );

            active_entry_index = new_index;
        }

        MappingEntry& active = candidate.entries_[active_entry_index];

        std::uint32_t const available =
            tokens_per_page_ - active.valid_tokens;

        std::uint32_t const appended =
            std::min(remaining_tokens, available);

        active.valid_tokens = static_cast<std::uint16_t>(
            active.valid_tokens + appended
        );

        logical_end += appended;
        remaining_tokens -= appended;

        if (active.valid_tokens == tokens_per_page_) {
            active_entry_index = kNoActiveEntry;
        }
    }

    candidate.version_ = request.table.version_ + 1;

    if (!candidate.checkInvariants(tokens_per_page_)) {
        rollbackStagedPagesLocked(staged_pages);
        return KvCacheError::InternalInvariantViolation;
    }

    // 从这里开始进入无异常 Commit 区，语义与 Hetero append 一致。
    if (replaced_sealed_tail.isStructurallyValid()) {
        KvCacheError const decrement_error =
            decrementReferenceLocked(replaced_sealed_tail);

        if (decrement_error != KvCacheError::None) {
            rollbackStagedPagesLocked(staged_pages);
            return decrement_error;
        }
    }

    if (existing_mutable.isStructurallyValid()) {
        RuntimeSlot* runtime = runtimeSlotLocked(existing_mutable);

        for (MappingEntry const& entry : candidate.entries_) {
            if (entry.handle != existing_mutable) {
                continue;
            }

            runtime->valid_tokens = entry.valid_tokens;

            if (entry.valid_tokens == tokens_per_page_) {
                runtime->state = PageState::Sealed;
                runtime->mutable_owner = kInvalidRequestId;
            }

            break;
        }
    }

    for (StagedPage const& staged : staged_pages) {
        RuntimeSlot* runtime = runtimeSlotLocked(staged.handle);
        MappingEntry const& entry =
            candidate.entries_[staged.entry_index];

        runtime->valid_tokens = entry.valid_tokens;
        runtime->ref_count = 1;

        if (entry.valid_tokens == tokens_per_page_) {
            runtime->state = PageState::Sealed;
            runtime->mutable_owner = kInvalidRequestId;
        } else {
            runtime->state = PageState::Mutable;
            runtime->mutable_owner = request_id;
        }
    }

    request.table = std::move(candidate);

    return KvCacheError::None;
}

KvCacheError FixedPageManager::sealTail(RequestId request_id)
{
    if (request_id == kInvalidRequestId) {
        return KvCacheError::InvalidArgument;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto request_iterator = requests_.find(request_id);

    if (request_iterator == requests_.end()) {
        return KvCacheError::RequestNotFound;
    }

    if (hasTokenReservationLocked(request_id)) {
        return KvCacheError::RequestConflict;
    }

    BlockTable& table = request_iterator->second.table;

    if (table.entries_.empty()) {
        return KvCacheError::InvalidState;
    }

    MappingEntry const& tail = table.entries_.back();
    RuntimeSlot* runtime = runtimeSlotLocked(tail.handle);

    if (runtime == nullptr
        || runtime->valid_tokens != tail.valid_tokens) {
        return KvCacheError::InternalInvariantViolation;
    }

    if (runtime->state == PageState::Sealed) {
        return KvCacheError::None;
    }

    if (runtime->state != PageState::Mutable
        || runtime->ref_count != 1
        || runtime->mutable_owner != request_id) {
        return KvCacheError::InvalidState;
    }

    runtime->state = PageState::Sealed;
    runtime->mutable_owner = kInvalidRequestId;

    // Mapping 没有变化，因此 BlockTable Version 不增加。
    return KvCacheError::None;
}

KvCacheError FixedPageManager::forkRequest(
    RequestId source_request_id,
    RequestId child_request_id)
{
    if (source_request_id == kInvalidRequestId
        || child_request_id == kInvalidRequestId
        || source_request_id == child_request_id) {
        return KvCacheError::InvalidArgument;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto source_iterator = requests_.find(source_request_id);

    if (source_iterator == requests_.end()) {
        return KvCacheError::RequestNotFound;
    }

    if (hasTokenReservationLocked(source_request_id)) {
        return KvCacheError::RequestConflict;
    }

    if (requests_.find(child_request_id) != requests_.end()) {
        return KvCacheError::RequestAlreadyExists;
    }

    BlockTable const& source_table = source_iterator->second.table;

    for (std::size_t index = 0;
         index < source_table.entries_.size();
         ++index) {
        MappingEntry const& entry = source_table.entries_[index];
        RuntimeSlot* runtime = runtimeSlotLocked(entry.handle);

        if (runtime == nullptr
            || runtime->valid_tokens != entry.valid_tokens
            || runtime->ref_count == 0
            || runtime->ref_count ==
                std::numeric_limits<std::uint32_t>::max()) {
            return KvCacheError::InternalInvariantViolation;
        }

        if (runtime->state == PageState::Mutable) {
            bool const is_tail =
                index + 1 == source_table.entries_.size();

            if (!is_tail
                || runtime->ref_count != 1
                || runtime->mutable_owner != source_request_id) {
                return KvCacheError::InternalInvariantViolation;
            }
        } else if (runtime->state != PageState::Sealed) {
            return KvCacheError::InvalidState;
        }
    }

    BlockTable child_table = source_table;

    // Child Version 是独立的，不继承 Source 的历史版本号。
    child_table.version_ = child_table.entries_.empty() ? 0 : 1;

    auto const insertion = requests_.emplace(
        child_request_id,
        RequestState{std::move(child_table)}
    );

    if (!insertion.second) {
        return KvCacheError::RequestAlreadyExists;
    }

    for (MappingEntry const& entry : source_table.entries_) {
        RuntimeSlot* runtime = runtimeSlotLocked(entry.handle);

        if (runtime->state == PageState::Mutable) {
            runtime->state = PageState::Sealed;
            runtime->mutable_owner = kInvalidRequestId;
        }

        ++runtime->ref_count;
    }

    return KvCacheError::None;
}

KvCacheError FixedPageManager::releaseRequest(RequestId request_id)
{
    if (request_id == kInvalidRequestId) {
        return KvCacheError::InvalidArgument;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto request_iterator = requests_.find(request_id);

    if (request_iterator == requests_.end()) {
        return KvCacheError::RequestNotFound;
    }

    if (hasTokenReservationLocked(request_id)) {
        return KvCacheError::RequestConflict;
    }

    BlockTable const& table = request_iterator->second.table;

    for (MappingEntry const& entry : table.entries_) {
        RuntimeSlot const* runtime = runtimeSlotLocked(entry.handle);

        if (runtime == nullptr
            || runtime->state == PageState::Free
            || runtime->state == PageState::CopyTarget
            || runtime->ref_count == 0) {
            return KvCacheError::InternalInvariantViolation;
        }
    }

    for (MappingEntry const& entry : table.entries_) {
        KvCacheError const decrement_error =
            decrementReferenceLocked(entry.handle);

        if (decrement_error != KvCacheError::None) {
            return decrement_error;
        }
    }

    requests_.erase(request_iterator);

    return KvCacheError::None;
}

std::optional<BlockTable> FixedPageManager::blockTable(
    RequestId request_id) const
{
    std::lock_guard<std::mutex> lock(mutex_);

    auto request_iterator = requests_.find(request_id);

    if (request_iterator == requests_.end()) {
        return std::nullopt;
    }

    return request_iterator->second.table;
}

FixedPageManagerSnapshot FixedPageManager::snapshot() const
{
    std::lock_guard<std::mutex> lock(mutex_);

    FixedPageManagerSnapshot result{};

    result.request_count = requests_.size();
    result.pool = pool_.snapshot();
    result.successful_allocations = result.pool.successful_allocations;
    result.peak_allocated_pages = peak_allocated_pages_;
    result.token_reservation_count = token_reservations_.size();

    return result;
}

bool FixedPageManager::checkInvariants() const
{
    std::lock_guard<std::mutex> lock(mutex_);

    return checkInvariantsLocked();
}

bool FixedPageManager::checkInvariantsLocked() const
{
    if (tokens_per_page_ == 0 || !pool_.checkInvariants()) {
        return false;
    }

    if (token_reservations_.size()
            != request_token_reservations_.size()) {
        return false;
    }

    std::vector<std::uint32_t> expected_references(runtime_.size(), 0);

    for (auto const& [request_id, request] : requests_) {
        BlockTable const& table = request.table;

        if (request_id == kInvalidRequestId
            || !table.checkInvariants(tokens_per_page_)) {
            return false;
        }

        for (std::size_t index = 0;
             index < table.entries_.size();
             ++index) {
            MappingEntry const& entry = table.entries_[index];

            if (entry.kind != PageKind::Micro
                || entry.valid_tokens == 0
                || entry.valid_tokens > tokens_per_page_
                || entry.handle.slot >= expected_references.size()
                || expected_references[entry.handle.slot]
                    == std::numeric_limits<std::uint32_t>::max()) {
                return false;
            }

            RuntimeSlot const* runtime = runtimeSlotLocked(entry.handle);

            if (runtime == nullptr
                || runtime->valid_tokens != entry.valid_tokens) {
                return false;
            }

            bool const is_tail = index + 1 == table.entries_.size();

            if (runtime->state == PageState::Mutable) {
                if (!is_tail
                    || runtime->ref_count != 1
                    || runtime->mutable_owner != request_id) {
                    return false;
                }
            } else if (runtime->state != PageState::Sealed) {
                return false;
            }

            ++expected_references[entry.handle.slot];
        }
    }

    std::vector<std::uint8_t> expected_targets(runtime_.size(), 0);
    for (auto const& [reservation_id, transaction] : token_reservations_) {
        auto const request_iterator = requests_.find(transaction.request_id);
        auto const request_reservation = request_token_reservations_.find(
            transaction.request_id
        );
        if (reservation_id == kInvalidKvTokenReservationId
            || transaction.reservation_id != reservation_id
            || request_iterator == requests_.end()
            || request_reservation == request_token_reservations_.end()
            || request_reservation->second != reservation_id
            || !transaction.candidate.checkInvariants(tokens_per_page_)
            || transaction.candidate.tokenCount()
                != request_iterator->second.table.tokenCount()
                    + transaction.token_count
            || transaction.token_count == 0
            || transaction.candidate.version()
                != request_iterator->second.table.version() + 1) {
            return false;
        }
        if (transaction.staged_pages.empty()
            && !transaction.existing_mutable.isStructurallyValid()) {
            return false;
        }
        for (StagedPage const& staged : transaction.staged_pages) {
            RuntimeSlot const* target = runtimeSlotLocked(staged.handle);
            if (target == nullptr
                || target->state != PageState::CopyTarget
                || target->ref_count != 0
                || staged.entry_index >= transaction.candidate.entries().size()
                || transaction.candidate.entries()[staged.entry_index].handle
                    != staged.handle
                || staged.handle.slot >= expected_targets.size()
                || expected_targets[staged.handle.slot] != 0) {
                return false;
            }
            std::uint16_t expected_tokens = 0;
            BlockTable const& before = request_iterator->second.table;
            if (transaction.replaced_sealed_tail.isStructurallyValid()
                && !before.entries().empty()
                && staged.entry_index + 1 == before.entries().size()) {
                expected_tokens = before.entries().back().valid_tokens;
            }
            if (target->valid_tokens != expected_tokens) {
                return false;
            }
            expected_targets[staged.handle.slot] = 1;
        }
        if (transaction.existing_mutable.isStructurallyValid()) {
            RuntimeSlot const* existing = runtimeSlotLocked(
                transaction.existing_mutable
            );
            if (existing == nullptr
                || existing->state != PageState::Mutable
                || existing->ref_count != 1
                || existing->mutable_owner != transaction.request_id
                || request_iterator->second.table.entries().empty()
                || request_iterator->second.table.entries().back().handle
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
                || request_iterator->second.table.entries().empty()
                || request_iterator->second.table.entries().back().handle
                    != transaction.replaced_sealed_tail) {
                return false;
            }
        }
    }

    std::size_t allocated_slots = 0;

    for (std::size_t slot_index = 0;
         slot_index < runtime_.size();
         ++slot_index) {
        RuntimeSlot const& runtime = runtime_[slot_index];

        switch (runtime.state) {
        case PageState::Free:
            if (runtime.valid_tokens != 0
                || runtime.ref_count != 0
                || runtime.mutable_owner != kInvalidRequestId
                || expected_references[slot_index] != 0) {
                return false;
            }
            continue;
        case PageState::Mutable:
            if (runtime.ref_count != 1
                || runtime.mutable_owner == kInvalidRequestId) {
                return false;
            }
            break;
        case PageState::Sealed:
            if (runtime.ref_count == 0
                || runtime.mutable_owner != kInvalidRequestId) {
                return false;
            }
            break;
        case PageState::CopyTarget:
            if (runtime.ref_count != 0
                || runtime.mutable_owner != kInvalidRequestId
                || expected_targets[slot_index] == 0
                || runtime.valid_tokens >= tokens_per_page_) {
                return false;
            }
            break;
        default:
            return false;
        }

        if (runtime.valid_tokens > tokens_per_page_
            || (runtime.valid_tokens == 0
                && runtime.state != PageState::CopyTarget)) {
            return false;
        }

        if (runtime.generation == PageHandle::kInvalidGeneration
            || runtime.ref_count != expected_references[slot_index]) {
            return false;
        }

        PageHandle const handle{
            PageKind::Micro,
            static_cast<std::uint32_t>(slot_index),
            runtime.generation,
        };

        if (pool_.validate(handle) != PagePoolError::None) {
            return false;
        }

        ++allocated_slots;
    }

    PagePoolSnapshot const pool = pool_.snapshot();

    if (pool.allocated_slots != allocated_slots
        || !pool.capacityBalanced()) {
        return false;
    }

    return true;
}

} // namespace kimkvcache
