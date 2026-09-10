#pragma once

#include "kim-kv/core/block_table.h"
#include "kim-kv/core/page_state.h"
#include "kim-kv/runtime/promotion.h"

#include <cstdint>

namespace kimkvcache {

using KvTokenReservationId = std::uint64_t;

constexpr KvTokenReservationId kInvalidKvTokenReservationId = 0;

// Metadata-only contiguous-token transaction result. `before` remains globally
// visible until commit; `reserved` is visible only to the owning Engine
// transaction and includes every uncommitted token in the chunk.
struct TokenReservationResult final {
    KvCacheError error{KvCacheError::None};
    KvTokenReservationId reservation_id{kInvalidKvTokenReservationId};
    RequestId request_id{kInvalidRequestId};
    std::uint32_t logical_token_position{0};
    std::uint32_t token_count{0};
    BlockTable before{};
    BlockTable reserved{};

    [[nodiscard]] constexpr bool ok() const noexcept
    {
        return error == KvCacheError::None
            && reservation_id != kInvalidKvTokenReservationId
            && token_count != 0;
    }

    [[nodiscard]] explicit constexpr operator bool() const noexcept
    {
        return ok();
    }
};

} // namespace kimkvcache
