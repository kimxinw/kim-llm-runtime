#pragma once

#include "kim-kv/core/block_table.h"
#include "kim-kv/cuda/cuda_status.h"
#include "kim-kv/engine/engine_kv.h"
#include "kim-kv/core/kv_layout.h"
#include "kim-kv/runtime/promotion.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace kimkvcache {

// Owns one Reserve -> per-layer Write/Attend -> token-boundary completion
// sequence. Descriptor upload happens once at construction; layer calls only
// enqueue work on the bound stream and never wait on the host.
class CudaEngineTransaction final {
public:
    struct Impl;

    struct AttentionBatchItem final {
        CudaEngineTransaction* transaction{nullptr};
        PagedDecodeRequest request{};
        CudaStatus status{CudaError::InvalidArgument, 0};
    };

    struct WriteBatchItem final {
        CudaEngineTransaction* transaction{nullptr};
        LayerKvWrite write{};
        CudaStatus status{CudaError::InvalidArgument, 0};
    };

    ~CudaEngineTransaction();
    CudaEngineTransaction(CudaEngineTransaction const&) = delete;
    CudaEngineTransaction& operator=(CudaEngineTransaction const&) = delete;
    CudaEngineTransaction(CudaEngineTransaction&&) noexcept;
    CudaEngineTransaction& operator=(CudaEngineTransaction&&) noexcept;

    [[nodiscard]] CudaStatus status() const noexcept;
    [[nodiscard]] CudaStatus writeLayer(
        LayerKvWrite const& write
    ) noexcept;
    [[nodiscard]] CudaStatus attendLayer(
        PagedDecodeRequest const& request
    ) noexcept;

    static void writeLayerBatch(
        WriteBatchItem* items,
        std::size_t item_count,
        DeviceLayerKvWriteBatchItem* host_items,
        DeviceLayerKvWriteBatchItem* device_items,
        std::size_t item_capacity
    ) noexcept;

    static void attendLayerBatch(
        AttentionBatchItem* items,
        std::size_t item_count,
        DevicePagedDecodeBatchItem* host_items,
        DevicePagedDecodeBatchItem* device_items,
        std::size_t item_capacity
    ) noexcept;
    [[nodiscard]] CudaStatus finish() noexcept;

private:
    explicit CudaEngineTransaction(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
    friend class CudaKvStorage;
};

struct CudaEngineTransactionBeginResult final {
    CudaStatus status{};
    std::unique_ptr<CudaEngineTransaction> transaction{};

    [[nodiscard]] bool ok() const noexcept
    {
        return status.ok() && transaction != nullptr;
    }
};

class CudaSubmission final {
public:
    struct Impl;

    CudaSubmission() noexcept;
    ~CudaSubmission();

    CudaSubmission(CudaSubmission const&) = delete;
    CudaSubmission& operator=(CudaSubmission const&) = delete;

    CudaSubmission(CudaSubmission&&) noexcept;
    CudaSubmission& operator=(CudaSubmission&&) noexcept;

    [[nodiscard]] bool submitted() const noexcept;
    [[nodiscard]] CudaStatus submissionStatus() const noexcept;

    // query 不阻塞；未完成时返回 CudaError::NotReady。
    [[nodiscard]] CudaStatus query() noexcept;

    // wait 可重复调用，首次调用同步 Event 并缓存最终状态。
    [[nodiscard]] CudaStatus wait() noexcept;

private:
    explicit CudaSubmission(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> impl_;

    friend class CudaKvStorage;
};

class CudaKvStorage final {
public:
    struct Impl;

    CudaKvStorage(
        KvLayout layout,
        std::uint32_t micro_capacity,
        std::uint32_t extent_capacity,
        std::uint16_t micro_page_tokens = kMicroPageTokenCapacity,
        std::uint16_t extent_page_tokens = kExtentPageTokenCapacity
    ) noexcept;

    ~CudaKvStorage();

    CudaKvStorage(CudaKvStorage const&) = delete;
    CudaKvStorage& operator=(CudaKvStorage const&) = delete;

    CudaKvStorage(CudaKvStorage&&) = delete;
    CudaKvStorage& operator=(CudaKvStorage&&) = delete;

    [[nodiscard]] CudaStatus status() const noexcept;
    [[nodiscard]] KvLayout layout() const noexcept;
    [[nodiscard]] CudaStorageSnapshot snapshot() const noexcept;

    // 下列 API 中的 Handle 必须由调用方持有有效 PageLease；低层 Storage
    // 只解析稳定 Slot 地址，不拥有 Host Generation/RefCount 元数据。
    // before/after 用于识别 Partial Tail COW；device_input 使用连续布局
    // [layer][K/V][new_token][head][dim]。
    [[nodiscard]] CudaSubmission appendAsync(
        BlockTable const& before,
        BlockTable const& after,
        std::uint32_t append_token_begin,
        std::uint32_t token_count,
        KvScalar const* device_input,
        CudaStream stream = nullptr
    );

    [[nodiscard]] CudaSubmission gatherAsync(
        BlockTable const& table,
        KvScalar* device_output,
        CudaStream stream = nullptr
    );

    [[nodiscard]] CudaSubmission promoteAsync(
        PromotionPrepareResult const& promotion,
        CudaStream stream = nullptr
    );

    // 先按 BlockTable Gather 到内部连续临时区，再运行 FP32 累加的
    // Reference Attention。query/output 均为 [layer][head][dim]。
    [[nodiscard]] CudaSubmission referenceAttentionAsync(
        BlockTable const& table,
        float const* device_query,
        float* device_output,
        CudaStream stream = nullptr
    );

    [[nodiscard]] CudaEngineTransactionBeginResult beginEngineTransaction(
        BlockTable const& before,
        BlockTable const& reserved,
        std::uint32_t query_head_count,
        CudaStream stream = nullptr
    );

    // 每次只消费一个故障点，用于验证同步提交失败与 Event 完成失败。
    void injectFailureOnce(CudaFailurePoint point) noexcept;

private:
    std::shared_ptr<Impl> impl_;
};

} // namespace kimkvcache
