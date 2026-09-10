#include "kim-kv/fixed/fixed_page_manager.h"
#include "kim-kv/runtime/kv_cache_manager.h"

#include <iostream>
#include <string>

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

void testHeterogeneousReserveCommitRollback()
{
    KvCacheManager manager(8, 1);
    expect(manager.createRequest(1) == KvCacheError::None, "create request");
    KvCacheManagerSnapshot const baseline = manager.snapshot();

    TokenReservationResult reserved = manager.reserveToken(1, 0);
    expect(reserved.ok(), "empty request token reserve");
    expect(reserved.before.tokenCount() == 0, "before remains empty");
    expect(reserved.reserved.tokenCount() == 1, "candidate includes token");
    expect(manager.blockTable(1)->tokenCount() == 0, "reserve is invisible");
    expect(manager.snapshot().token_reservation_count == 1, "active count");
    expect(manager.checkInvariants(), "reserve invariants");

    expect(
        manager.reserveToken(1, 0).error == KvCacheError::RequestConflict,
        "second reserve conflicts"
    );
    expect(
        manager.append(1, 1) == KvCacheError::RequestConflict,
        "legacy append conflicts"
    );
    expect(
        manager.forkRequest(1, 2) == KvCacheError::RequestConflict,
        "fork conflicts"
    );
    expect(
        manager.releaseRequest(1) == KvCacheError::RequestConflict,
        "release conflicts"
    );
    expect(
        manager.preparePromotion(1, 0).error
            == KvCacheError::RequestConflict,
        "promotion conflicts"
    );

    PageLeaseAcquireResult lease = manager.acquireTokenReservationLease(
        reserved.reservation_id
    );
    expect(lease.ok(), "reservation lease");
    expect(lease.table.tokenCount() == 1, "lease exposes candidate");
    expect(manager.checkInvariants(), "lease invariants");
    expect(
        manager.releasePageLease(lease.lease_id) == KvCacheError::None,
        "release reservation lease"
    );
    expect(
        manager.rollbackTokenReservation(reserved.reservation_id)
            == KvCacheError::None,
        "rollback reservation"
    );
    expect(manager.blockTable(1)->tokenCount() == 0, "rollback invisible");
    expect(
        manager.snapshot().micro_pool.allocated_slots
            == baseline.micro_pool.allocated_slots,
        "rollback returns staged page"
    );
    expect(manager.checkInvariants(), "rollback invariants");

    reserved = manager.reserveToken(1, 0);
    expect(reserved.ok(), "reserve again");
    lease = manager.acquireTokenReservationLease(reserved.reservation_id);
    expect(lease.ok(), "second lease");
    expect(
        manager.releasePageLease(lease.lease_id) == KvCacheError::None,
        "release before commit"
    );
    expect(
        manager.commitTokenReservation(reserved.reservation_id)
            == KvCacheError::None,
        "commit reservation"
    );
    expect(manager.blockTable(1)->tokenCount() == 1, "commit publishes token");
    expect(manager.checkInvariants(), "commit invariants");

    expect(
        manager.reserveToken(1, 0).error == KvCacheError::InvalidState,
        "stale expected length rejected"
    );
    TokenReservationResult mutable_tail = manager.reserveToken(1, 1);
    expect(mutable_tail.ok(), "reserve on mutable tail");
    expect(
        mutable_tail.before.entries().back().handle
            == mutable_tail.reserved.entries().back().handle,
        "exclusive mutable tail is reused"
    );
    expect(manager.checkInvariants(), "mutable reservation invariants");
    expect(
        manager.rollbackTokenReservation(mutable_tail.reservation_id)
            == KvCacheError::None,
        "mutable rollback"
    );
    expect(manager.blockTable(1)->tokenCount() == 1, "mutable rollback length");
}

void testHeterogeneousCowAndOom()
{
    KvCacheManager manager(4, 1);
    expect(manager.createRequest(10) == KvCacheError::None, "create source");
    expect(manager.append(10, 3) == KvCacheError::None, "append source");
    expect(manager.forkRequest(10, 11) == KvCacheError::None, "fork source");
    std::uint32_t const pages_before = manager.snapshot().micro_pool.allocated_slots;

    TokenReservationResult cow = manager.reserveToken(11, 3);
    expect(cow.ok(), "reserve shared partial tail");
    expect(
        cow.before.entries().back().handle
            != cow.reserved.entries().back().handle,
        "shared sealed tail uses COW"
    );
    expect(manager.checkInvariants(), "COW reservation invariants");
    expect(
        manager.rollbackTokenReservation(cow.reservation_id)
            == KvCacheError::None,
        "COW rollback"
    );
    expect(
        manager.snapshot().micro_pool.allocated_slots == pages_before,
        "COW rollback releases target"
    );

    KvCacheManager exhausted(1, 1);
    expect(exhausted.createRequest(19) == KvCacheError::None, "create holder");
    expect(exhausted.append(19, 1) == KvCacheError::None, "fill micro pool");
    expect(exhausted.createRequest(20) == KvCacheError::None, "create OOM req");
    expect(
        exhausted.reserveToken(20, 0).error
            == KvCacheError::ResourceExhausted,
        "reserve OOM"
    );
    expect(exhausted.checkInvariants(), "OOM invariants");
}

void testFixedReserveCommitRollback()
{
    FixedPageManager manager(16, 4);
    expect(manager.createRequest(30) == KvCacheError::None, "fixed create");
    FixedPageManagerSnapshot const baseline = manager.snapshot();
    TokenReservationResult reserved = manager.reserveToken(30, 0);
    expect(reserved.ok(), "fixed reserve");
    expect(manager.blockTable(30)->tokenCount() == 0, "fixed invisible");
    expect(manager.checkInvariants(), "fixed reserve invariants");
    expect(
        manager.releaseRequest(30) == KvCacheError::RequestConflict,
        "fixed release conflict"
    );
    expect(
        manager.rollbackTokenReservation(reserved.reservation_id)
            == KvCacheError::None,
        "fixed rollback"
    );
    expect(
        manager.snapshot().pool.allocated_slots
            == baseline.pool.allocated_slots,
        "fixed rollback returns page"
    );

    reserved = manager.reserveToken(30, 0);
    expect(
        manager.commitTokenReservation(reserved.reservation_id)
            == KvCacheError::None,
        "fixed commit"
    );
    expect(manager.blockTable(30)->tokenCount() == 1, "fixed commit visible");
    expect(manager.checkInvariants(), "fixed commit invariants");

    FixedPageManager exhausted(8, 1);
    expect(exhausted.createRequest(39) == KvCacheError::None, "fixed holder");
    expect(exhausted.append(39, 1) == KvCacheError::None, "fill fixed pool");
    expect(exhausted.createRequest(40) == KvCacheError::None, "fixed OOM create");
    expect(
        exhausted.reserveToken(40, 0).error
            == KvCacheError::ResourceExhausted,
        "fixed reserve OOM"
    );
    expect(exhausted.checkInvariants(), "fixed OOM invariants");
}

void testContiguousMultiTokenReservations()
{
    KvCacheManager heterogeneous(8, 1);
    expect(heterogeneous.createRequest(50) == KvCacheError::None,
        "multi-token heterogeneous create");
    TokenReservationResult multi = heterogeneous.reserveTokens(50, 0, 20);
    expect(multi.ok() && multi.token_count == 20,
        "heterogeneous reserves one contiguous token segment");
    expect(multi.reserved.tokenCount() == 20
        && multi.reserved.entries().size() == 3,
        "heterogeneous segment spans three micro pages");
    expect(heterogeneous.blockTable(50)->tokenCount() == 0,
        "multi-token reservation remains invisible before commit");
    expect(heterogeneous.checkInvariants(),
        "multi-token heterogeneous reserve invariants");
    expect(heterogeneous.commitTokenReservation(multi.reservation_id)
            == KvCacheError::None,
        "multi-token heterogeneous commit");
    expect(heterogeneous.blockTable(50)->tokenCount() == 20,
        "multi-token heterogeneous commit publishes atomically");

    expect(heterogeneous.forkRequest(50, 51) == KvCacheError::None,
        "fork seals partial tail for chunk COW");
    std::uint32_t const pages_before =
        heterogeneous.snapshot().micro_pool.allocated_slots;
    multi = heterogeneous.reserveTokens(51, 20, 10);
    expect(multi.ok() && multi.reserved.tokenCount() == 30
        && multi.reserved.entries().size() == 4,
        "chunk COW can fill the sealed tail and cross a page boundary");
    expect(heterogeneous.rollbackTokenReservation(multi.reservation_id)
            == KvCacheError::None,
        "chunk COW rollback succeeds");
    expect(heterogeneous.blockTable(51)->tokenCount() == 20
        && heterogeneous.snapshot().micro_pool.allocated_slots == pages_before,
        "chunk COW rollback restores mapping and page accounting");
    expect(heterogeneous.checkInvariants(),
        "chunk COW rollback invariants");

    FixedPageManager fixed(8, 8);
    expect(fixed.createRequest(60) == KvCacheError::None,
        "multi-token fixed create");
    multi = fixed.reserveTokens(60, 0, 19);
    expect(multi.ok() && multi.reserved.tokenCount() == 19
        && multi.reserved.entries().size() == 3,
        "fixed backend reserves a cross-page token segment");
    expect(fixed.commitTokenReservation(multi.reservation_id)
            == KvCacheError::None,
        "multi-token fixed commit");
    expect(fixed.blockTable(60)->tokenCount() == 19
        && fixed.checkInvariants(),
        "multi-token fixed commit preserves invariants");
}

} // namespace

int main()
{
    testHeterogeneousReserveCommitRollback();
    testHeterogeneousCowAndOom();
    testFixedReserveCommitRollback();
    testContiguousMultiTokenReservations();
    if (failures != 0) {
        std::cerr << failures << " token reservation checks failed\n";
        return 1;
    }
    std::cout << "token reservation checks passed\n";
    return 0;
}
