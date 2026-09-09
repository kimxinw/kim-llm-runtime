#pragma once

#include "storage.h"
#include "kernels.cuh"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace kimkvcache {

struct CudaKvStorage::Impl final {
    KvLayout layout{};
    std::uint32_t micro_capacity{0};
    std::uint32_t extent_capacity{0};
    std::uint16_t micro_page_tokens{kMicroPageTokenCapacity};
    std::uint16_t extent_page_tokens{kExtentPageTokenCapacity};
    std::size_t micro_page_elements{0};
    std::size_t extent_page_elements{0};
    std::size_t micro_reserved_bytes{0};
    std::size_t extent_reserved_bytes{0};
    KvScalar* micro_data{nullptr};
    KvScalar* extent_data{nullptr};
    CudaStatus initialization_status{};
    bool fail_submission_once{false};
    bool fail_completion_once{false};
    mutable std::mutex fault_mutex;

    ~Impl();

    [[nodiscard]] bool consumeFailure(
        CudaFailurePoint point) noexcept;

    [[nodiscard]] KvScalar* pagePointer(
        PageHandle handle) const noexcept;

    [[nodiscard]] std::uint16_t pageTokenCapacity(
        PageKind kind) const noexcept;

    [[nodiscard]] bool validTable(
        BlockTable const& table) const noexcept;
};

struct CudaSubmission::Impl final {
    std::shared_ptr<void> storage_owner{};
    cudaEvent_t event{nullptr};
    cudaStream_t stream{nullptr};
    std::vector<void*> temporaries{};
    CudaStatus submission_status{};
    CudaStatus final_status{};
    bool inject_completion_failure{false};
    bool finished{false};

    void cleanup() noexcept;
    [[nodiscard]] CudaStatus complete(bool wait) noexcept;
    ~Impl();
};

struct CudaEngineTransaction::Impl final {
    std::shared_ptr<CudaKvStorage::Impl> storage{};
    BlockTable before{};
    BlockTable reserved{};
    ::kimkvcache::DeviceBlockDescriptor* device_descriptors{nullptr};
    std::uint32_t descriptor_count{0};
    std::uint32_t query_head_count{0};
    cudaStream_t stream{nullptr};
    cudaEvent_t completion_event{nullptr};
    CudaStatus submission_status{};
    CudaStatus final_status{};
    bool finished{false};

    ~Impl();
    [[nodiscard]] CudaStatus failSubmission(cudaError_t error) noexcept;
    [[nodiscard]] CudaStatus prepareWrite(
        LayerKvWrite const& write,
        DeviceLayerKvWriteBatchItem& item
    ) noexcept;
    [[nodiscard]] CudaStatus prepareAttention(
        PagedDecodeRequest const& request,
        DevicePagedDecodeBatchItem& item
    ) noexcept;
    [[nodiscard]] CudaStatus complete() noexcept;
};

namespace cuda_storage_detail {

using cuda_detail::DeviceLayout;

[[nodiscard]] inline cudaStream_t nativeStream(
    CudaStream stream) noexcept
{
    return reinterpret_cast<cudaStream_t>(stream);
}

[[nodiscard]] inline CudaStatus mapInitializationError(
    cudaError_t error) noexcept
{
    if (error == cudaSuccess) {
        return CudaStatus{};
    }

    CudaError mapped = CudaError::RuntimeUnavailable;
    if (error == cudaErrorMemoryAllocation) {
        mapped = CudaError::AllocationFailed;
    }
    return CudaStatus{mapped, static_cast<int>(error)};
}

[[nodiscard]] inline CudaStatus mapSubmissionError(
    cudaError_t error) noexcept
{
    if (error == cudaErrorMemoryAllocation) {
        return CudaStatus{
            CudaError::AllocationFailed,
            static_cast<int>(error),
        };
    }
    return error == cudaSuccess
        ? CudaStatus{}
        : CudaStatus{
            CudaError::SubmissionFailed,
            static_cast<int>(error),
        };
}

[[nodiscard]] inline std::size_t operationElements(
    KvLayout const& layout,
    std::uint32_t token_count) noexcept
{
    std::size_t result = 0;
    return layout.elementsForTokens(token_count, result) ? result : 0;
}

[[nodiscard]] inline DeviceLayout deviceLayout(
    KvLayout const& layout) noexcept
{
    return DeviceLayout{
        layout.layer_count,
        layout.kv_head_count,
        layout.head_dimension,
    };
}

[[nodiscard]] std::unique_ptr<CudaSubmission::Impl> beginOperation(
    std::shared_ptr<CudaKvStorage::Impl> const& storage,
    CudaStream stream);

[[nodiscard]] CudaStatus recordOperation(
    CudaSubmission::Impl& operation) noexcept;

[[nodiscard]] bool addTemporary(
    CudaSubmission::Impl& operation,
    void* allocation);

} // namespace cuda_storage_detail
} // namespace kimkvcache
