#include "../kv/storage_internal.cuh"

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace kimkvcache {
namespace {

[[nodiscard]] CudaStatus invalidArgument() noexcept
{
    return CudaStatus{CudaError::InvalidArgument, 0};
}

[[nodiscard]] CudaStatus injectedSubmissionFailure() noexcept
{
    return CudaStatus{CudaError::SubmissionFailed, 0};
}

} // namespace

CudaEngineTransaction::Impl::~Impl()
{
    static_cast<void>(complete());
    if (device_descriptors != nullptr) {
        static_cast<void>(cudaFree(device_descriptors));
    }
    if (completion_event != nullptr) {
        static_cast<void>(cudaEventDestroy(completion_event));
    }
}

CudaStatus CudaEngineTransaction::Impl::failSubmission(
    cudaError_t error) noexcept
{
    submission_status = cuda_storage_detail::mapSubmissionError(error);
    final_status = submission_status;
    return submission_status;
}

CudaStatus CudaEngineTransaction::Impl::prepareAttention(
    PagedDecodeRequest const& request,
    DevicePagedDecodeBatchItem& item) noexcept
{
    item = {};
    if (finished || !submission_status.ok()
        || request.layer >= storage->layout.layer_count
        || request.device_query == nullptr
        || request.device_output == nullptr
        || request.device_workspace == nullptr
        || !(request.attention_scale > 0.0F)
        || request.query_token_count != 1
        || reserved.tokenCount() != before.tokenCount() + 1) {
        return !submission_status.ok() ? submission_status : invalidArgument();
    }
    std::uint32_t const token_count = reserved.tokenCount();
    std::size_t const score_count =
        static_cast<std::size_t>(query_head_count) * token_count;
    if (token_count == 0
        || score_count > std::numeric_limits<std::size_t>::max()
            / sizeof(float)
        || request.workspace_bytes < score_count * sizeof(float)) {
        return invalidArgument();
    }
    if (storage->consumeFailure(CudaFailurePoint::Submission)) {
        submission_status = injectedSubmissionFailure();
        final_status = submission_status;
        return submission_status;
    }

    item = DevicePagedDecodeBatchItem{
        device_descriptors,
        descriptor_count,
        token_count,
        request.layer,
        request.device_query,
        static_cast<float*>(request.device_workspace),
        request.device_output,
        request.attention_scale,
    };
    return {};
}

CudaStatus CudaEngineTransaction::Impl::complete() noexcept
{
    if (finished) {
        return final_status;
    }
    if (!submission_status.ok()) {
        static_cast<void>(cudaStreamSynchronize(stream));
        final_status = submission_status;
        finished = true;
        return final_status;
    }
    if (completion_event == nullptr) {
        final_status = CudaStatus{CudaError::InternalError, 0};
        finished = true;
        return final_status;
    }

    cudaError_t error = cudaEventRecord(completion_event, stream);
    if (error == cudaSuccess) {
        error = cudaEventSynchronize(completion_event);
    }
    if (error != cudaSuccess) {
        final_status = CudaStatus{
            CudaError::ExecutionFailed,
            static_cast<int>(error),
        };
    } else if (storage->consumeFailure(CudaFailurePoint::Completion)) {
        final_status = CudaStatus{
            CudaError::ExecutionFailed,
            static_cast<int>(cudaErrorUnknown),
        };
    } else {
        final_status = {};
    }
    finished = true;
    return final_status;
}

CudaEngineTransaction::CudaEngineTransaction(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl))
{
}

CudaEngineTransaction::~CudaEngineTransaction() = default;
CudaEngineTransaction::CudaEngineTransaction(
    CudaEngineTransaction&&) noexcept = default;
CudaEngineTransaction& CudaEngineTransaction::operator=(
    CudaEngineTransaction&&) noexcept = default;

CudaStatus CudaEngineTransaction::status() const noexcept
{
    return impl_ != nullptr ? impl_->submission_status : invalidArgument();
}

CudaStatus CudaEngineTransaction::Impl::prepareWrite(
    LayerKvWrite const& write,
    DeviceLayerKvWriteBatchItem& item) noexcept
{
    item = {};
    if (finished
        || !submission_status.ok()
        || write.layer >= storage->layout.layer_count
        || write.device_key == nullptr
        || write.device_value == nullptr
        || write.token_count != 1
        || reserved.tokenCount() != before.tokenCount() + 1) {
        return !submission_status.ok()
            ? submission_status
            : invalidArgument();
    }
    if (storage->consumeFailure(CudaFailurePoint::Submission)) {
        submission_status = injectedSubmissionFailure();
        final_status = submission_status;
        return submission_status;
    }

    MappingEntry const& target_entry = reserved.entries().back();
    KvScalar* target = storage->pagePointer(target_entry.handle);
    std::uint16_t const target_capacity =
        storage->pageTokenCapacity(target_entry.kind);
    if (target == nullptr
        || reserved.tokenCount() == 0
        || reserved.tokenCount() - 1
            < target_entry.logical_token_begin) {
        return failSubmission(cudaErrorInvalidValue);
    }

    KvScalar const* copy_source = nullptr;
    std::uint32_t source_capacity = 0;
    std::uint32_t copy_token_count = 0;
    if (!before.entries().empty()) {
        MappingEntry const& old_tail = before.entries().back();
        if (old_tail.logical_token_begin
                == target_entry.logical_token_begin
            && old_tail.handle != target_entry.handle) {
            copy_source = storage->pagePointer(old_tail.handle);
            if (copy_source == nullptr) {
                return failSubmission(cudaErrorInvalidValue);
            }
            source_capacity = storage->pageTokenCapacity(old_tail.kind);
            copy_token_count = old_tail.valid_tokens;
        }
    }

    std::uint32_t const page_token = reserved.tokenCount() - 1
        - target_entry.logical_token_begin;
    item = DeviceLayerKvWriteBatchItem{
        write.device_key,
        write.device_value,
        copy_source,
        target,
        source_capacity,
        target_capacity,
        copy_token_count,
        page_token,
        write.layer,
    };
    return {};
}

CudaStatus CudaEngineTransaction::writeLayer(
    LayerKvWrite const& write) noexcept
{
    if (impl_ == nullptr) {
        return invalidArgument();
    }
    std::uint32_t const transaction_token_count =
        impl_->reserved.tokenCount() - impl_->before.tokenCount();
    if (write.token_count != transaction_token_count
        || write.token_count == 0) {
        return invalidArgument();
    }
    if (write.token_count > 1) {
        if (impl_->finished || !impl_->submission_status.ok()
            || write.layer >= impl_->storage->layout.layer_count
            || write.device_key == nullptr || write.device_value == nullptr) {
            return !impl_->submission_status.ok()
                ? impl_->submission_status : invalidArgument();
        }
        if (impl_->storage->consumeFailure(CudaFailurePoint::Submission)) {
            impl_->submission_status = injectedSubmissionFailure();
            impl_->final_status = impl_->submission_status;
            return impl_->submission_status;
        }
        if (!impl_->before.entries().empty()) {
            MappingEntry const& old_tail = impl_->before.entries().back();
            MappingEntry const& new_tail = impl_->reserved.entries()[
                impl_->before.entries().size() - 1
            ];
            if (old_tail.logical_token_begin
                    == new_tail.logical_token_begin
                && old_tail.handle != new_tail.handle) {
                KvScalar const* const source =
                    impl_->storage->pagePointer(old_tail.handle);
                KvScalar* const target =
                    impl_->storage->pagePointer(new_tail.handle);
                if (source == nullptr || target == nullptr) {
                    return impl_->failSubmission(cudaErrorInvalidValue);
                }
                cuda_detail::launchCopyLayerTokens(
                    source,
                    impl_->storage->pageTokenCapacity(old_tail.kind),
                    target,
                    impl_->storage->pageTokenCapacity(new_tail.kind),
                    old_tail.valid_tokens,
                    write.layer,
                    cuda_storage_detail::deviceLayout(
                        impl_->storage->layout
                    ),
                    impl_->stream
                );
            }
        }
        cuda_detail::launchWriteLayerTokens(
            impl_->device_descriptors,
            impl_->descriptor_count,
            impl_->storage->micro_data,
            impl_->storage->micro_page_elements,
            impl_->storage->extent_data,
            impl_->storage->extent_page_elements,
            impl_->before.tokenCount(),
            write.token_count,
            write.layer,
            write.device_key,
            write.device_value,
            cuda_storage_detail::deviceLayout(impl_->storage->layout),
            impl_->stream
        );
        cudaError_t const error = cudaGetLastError();
        return error == cudaSuccess
            ? CudaStatus{} : impl_->failSubmission(error);
    }
    DeviceLayerKvWriteBatchItem item;
    CudaStatus const prepared = impl_->prepareWrite(write, item);
    if (!prepared.ok()) {
        return prepared;
    }
    if (item.copy_token_count != 0) {
        cuda_detail::launchCopyLayerTokens(
            item.copy_source,
            item.source_capacity,
            item.target,
            item.target_capacity,
            item.copy_token_count,
            item.layer,
            cuda_storage_detail::deviceLayout(impl_->storage->layout),
            impl_->stream
        );
    }
    cuda_detail::launchWriteLayerToken(
        item.device_key,
        item.device_value,
        item.target,
        item.target_capacity,
        item.target_token,
        item.layer,
        cuda_storage_detail::deviceLayout(impl_->storage->layout),
        impl_->stream
    );
    cudaError_t const error = cudaGetLastError();
    return error == cudaSuccess
        ? CudaStatus{}
        : impl_->failSubmission(error);
}

void CudaEngineTransaction::writeLayerBatch(
    WriteBatchItem* items,
    std::size_t item_count,
    DeviceLayerKvWriteBatchItem* host_items,
    DeviceLayerKvWriteBatchItem* device_items,
    std::size_t item_capacity) noexcept
{
    if (items == nullptr || item_count == 0 || host_items == nullptr
        || device_items == nullptr || item_capacity < item_count
        || item_count > std::numeric_limits<std::uint32_t>::max()
        || item_count > std::numeric_limits<std::size_t>::max()
            / sizeof(DeviceLayerKvWriteBatchItem)) {
        if (items != nullptr) {
            for (std::size_t index = 0; index < item_count; ++index) {
                items[index].status = invalidArgument();
            }
        }
        return;
    }

    std::shared_ptr<CudaKvStorage::Impl> storage;
    cudaStream_t stream = nullptr;
    std::uint32_t max_copy_token_count = 0;
    std::size_t ready_count = 0;
    for (std::size_t index = 0; index < item_count; ++index) {
        host_items[index] = {};
        WriteBatchItem& item = items[index];
        if (item.transaction == nullptr || item.transaction->impl_ == nullptr) {
            item.status = invalidArgument();
            continue;
        }
        Impl& impl = *item.transaction->impl_;
        if (storage == nullptr) {
            storage = impl.storage;
            stream = impl.stream;
        } else if (impl.storage.get() != storage.get()
            || impl.stream != stream) {
            item.status = invalidArgument();
            continue;
        }
        item.status = impl.prepareWrite(item.write, host_items[index]);
        if (item.status.ok()) {
            ++ready_count;
            max_copy_token_count = std::max(
                max_copy_token_count, host_items[index].copy_token_count
            );
        }
    }
    if (ready_count == 0 || storage == nullptr) {
        return;
    }

    std::size_t const metadata_bytes =
        item_count * sizeof(DeviceLayerKvWriteBatchItem);
    cudaError_t error = cudaMemcpyAsync(
        device_items,
        host_items,
        metadata_bytes,
        cudaMemcpyHostToDevice,
        stream
    );
    if (error == cudaSuccess) {
        cuda_detail::launchWriteLayerTokenBatch(
            device_items,
            static_cast<std::uint32_t>(item_count),
            max_copy_token_count,
            cuda_storage_detail::deviceLayout(storage->layout),
            stream
        );
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        return;
    }
    for (std::size_t index = 0; index < item_count; ++index) {
        if (items[index].status.ok()
            && items[index].transaction != nullptr
            && items[index].transaction->impl_ != nullptr) {
            items[index].status =
                items[index].transaction->impl_->failSubmission(error);
        }
    }
}

CudaStatus CudaEngineTransaction::attendLayer(
    PagedDecodeRequest const& request) noexcept
{
    if (impl_ == nullptr) {
        return invalidArgument();
    }
    std::uint32_t const transaction_token_count =
        impl_->reserved.tokenCount() - impl_->before.tokenCount();
    if (request.query_token_count != transaction_token_count
        || request.query_token_count == 0) {
        return invalidArgument();
    }
    if (request.query_token_count > 1) {
        if (impl_->finished || !impl_->submission_status.ok()
            || request.layer >= impl_->storage->layout.layer_count
            || request.device_query == nullptr
            || request.device_output == nullptr
            || request.device_workspace == nullptr
            || !(request.attention_scale > 0.0F)) {
            return !impl_->submission_status.ok()
                ? impl_->submission_status : invalidArgument();
        }
        std::size_t score_count = request.query_token_count;
        if (score_count > std::numeric_limits<std::size_t>::max()
                / impl_->query_head_count) {
            return invalidArgument();
        }
        score_count *= impl_->query_head_count;
        if (score_count > std::numeric_limits<std::size_t>::max()
                / impl_->reserved.tokenCount()) {
            return invalidArgument();
        }
        score_count *= impl_->reserved.tokenCount();
        if (score_count > std::numeric_limits<std::size_t>::max()
                / sizeof(float)
            || request.workspace_bytes < score_count * sizeof(float)) {
            return invalidArgument();
        }
        if (impl_->storage->consumeFailure(CudaFailurePoint::Submission)) {
            impl_->submission_status = injectedSubmissionFailure();
            impl_->final_status = impl_->submission_status;
            return impl_->submission_status;
        }
        cuda_detail::launchPagedPrefillAttention(
            impl_->device_descriptors,
            impl_->descriptor_count,
            impl_->storage->micro_data,
            impl_->storage->micro_page_elements,
            impl_->storage->extent_data,
            impl_->storage->extent_page_elements,
            impl_->reserved.tokenCount(),
            request.query_token_count,
            request.layer,
            impl_->query_head_count,
            request.device_query,
            static_cast<float*>(request.device_workspace),
            request.device_output,
            request.attention_scale,
            cuda_storage_detail::deviceLayout(impl_->storage->layout),
            impl_->stream
        );
        cudaError_t const error = cudaGetLastError();
        return error == cudaSuccess
            ? CudaStatus{} : impl_->failSubmission(error);
    }
    DevicePagedDecodeBatchItem item;
    CudaStatus const prepared = impl_->prepareAttention(request, item);
    if (!prepared.ok()) {
        return prepared;
    }

    cuda_detail::launchPagedDecodeAttention(
        item.device_descriptors,
        item.descriptor_count,
        impl_->storage->micro_data,
        impl_->storage->micro_page_elements,
        impl_->storage->extent_data,
        impl_->storage->extent_page_elements,
        item.token_count,
        item.layer,
        impl_->query_head_count,
        item.device_query,
        item.device_scores,
        item.device_output,
        item.attention_scale,
        cuda_storage_detail::deviceLayout(impl_->storage->layout),
        impl_->stream
    );
    cudaError_t const error = cudaGetLastError();
    return error == cudaSuccess
        ? CudaStatus{}
        : impl_->failSubmission(error);
}

void CudaEngineTransaction::attendLayerBatch(
    AttentionBatchItem* items,
    std::size_t item_count,
    DevicePagedDecodeBatchItem* host_items,
    DevicePagedDecodeBatchItem* device_items,
    std::size_t item_capacity) noexcept
{
    if (items == nullptr || item_count == 0 || host_items == nullptr
        || device_items == nullptr || item_capacity < item_count
        || item_count > std::numeric_limits<std::uint32_t>::max()
        || item_count > std::numeric_limits<std::size_t>::max()
            / sizeof(DevicePagedDecodeBatchItem)) {
        if (items != nullptr) {
            for (std::size_t index = 0; index < item_count; ++index) {
                items[index].status = invalidArgument();
            }
        }
        return;
    }

    std::shared_ptr<CudaKvStorage::Impl> storage;
    cudaStream_t stream = nullptr;
    std::uint32_t query_head_count = 0;
    std::size_t ready_count = 0;
    for (std::size_t index = 0; index < item_count; ++index) {
        host_items[index] = {};
        AttentionBatchItem& item = items[index];
        if (item.transaction == nullptr || item.transaction->impl_ == nullptr) {
            item.status = invalidArgument();
            continue;
        }
        Impl& impl = *item.transaction->impl_;
        if (storage == nullptr) {
            storage = impl.storage;
            stream = impl.stream;
            query_head_count = impl.query_head_count;
        } else if (impl.storage.get() != storage.get()
            || impl.stream != stream
            || impl.query_head_count != query_head_count) {
            item.status = invalidArgument();
            continue;
        }
        item.status = impl.prepareAttention(item.request, host_items[index]);
        ready_count += item.status.ok() ? 1U : 0U;
    }
    if (ready_count == 0 || storage == nullptr) {
        return;
    }

    std::size_t const metadata_bytes =
        item_count * sizeof(DevicePagedDecodeBatchItem);
    cudaError_t error = cudaMemcpyAsync(
        device_items,
        host_items,
        metadata_bytes,
        cudaMemcpyHostToDevice,
        stream
    );
    if (error == cudaSuccess) {
        cuda_detail::launchPagedDecodeAttentionBatch(
            device_items,
            static_cast<std::uint32_t>(item_count),
            storage->micro_data,
            storage->micro_page_elements,
            storage->extent_data,
            storage->extent_page_elements,
            query_head_count,
            cuda_storage_detail::deviceLayout(storage->layout),
            stream
        );
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        return;
    }
    for (std::size_t index = 0; index < item_count; ++index) {
        if (items[index].status.ok()
            && items[index].transaction != nullptr
            && items[index].transaction->impl_ != nullptr) {
            items[index].status =
                items[index].transaction->impl_->failSubmission(error);
        }
    }
}

CudaStatus CudaEngineTransaction::finish() noexcept
{
    return impl_ != nullptr ? impl_->complete() : invalidArgument();
}

CudaEngineTransactionBeginResult CudaKvStorage::beginEngineTransaction(
    BlockTable const& before,
    BlockTable const& reserved,
    std::uint32_t query_head_count,
    CudaStream stream)
{
    CudaEngineTransactionBeginResult result;
    if (impl_ == nullptr || !impl_->initialization_status.ok()) {
        result.status = impl_ != nullptr
            ? impl_->initialization_status
            : CudaStatus{CudaError::AllocationFailed, 0};
        return result;
    }
    if (!impl_->validTable(before)
        || !impl_->validTable(reserved)
        || reserved.tokenCount() <= before.tokenCount()
        || query_head_count < impl_->layout.kv_head_count
        || query_head_count % impl_->layout.kv_head_count != 0
        || reserved.entries().empty()) {
        result.status = invalidArgument();
        return result;
    }
    if (impl_->consumeFailure(CudaFailurePoint::Submission)) {
        result.status = injectedSubmissionFailure();
        return result;
    }

    std::unique_ptr<CudaEngineTransaction::Impl> transaction;
    std::vector<::kimkvcache::DeviceBlockDescriptor> descriptors;
    try {
        transaction = std::make_unique<CudaEngineTransaction::Impl>();
        transaction->storage = impl_;
        transaction->before = before;
        transaction->reserved = reserved;
        transaction->descriptor_count = static_cast<std::uint32_t>(
            reserved.entries().size()
        );
        transaction->query_head_count = query_head_count;
        transaction->stream = cuda_storage_detail::nativeStream(stream);
        descriptors.reserve(reserved.entries().size());
        for (MappingEntry const& entry : reserved.entries()) {
            descriptors.push_back(::kimkvcache::DeviceBlockDescriptor{
                entry.handle.slot,
                entry.handle.generation,
                entry.logical_token_begin,
                entry.valid_tokens,
                impl_->pageTokenCapacity(entry.kind),
                entry.kind,
            });
        }
    } catch (...) {
        result.status = CudaStatus{CudaError::AllocationFailed, 0};
        return result;
    }

    cudaError_t error = cudaEventCreateWithFlags(
        &transaction->completion_event,
        cudaEventDisableTiming
    );
    std::size_t const descriptor_bytes = descriptors.size()
        * sizeof(::kimkvcache::DeviceBlockDescriptor);
    if (error == cudaSuccess) {
        error = cudaMalloc(
            reinterpret_cast<void**>(&transaction->device_descriptors),
            descriptor_bytes
        );
    }
    if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            transaction->device_descriptors,
            descriptors.data(),
            descriptor_bytes,
            cudaMemcpyHostToDevice,
            transaction->stream
        );
    }
    if (error != cudaSuccess) {
        result.status = cuda_storage_detail::mapSubmissionError(error);
        return result;
    }

    transaction->submission_status = {};
    result.transaction = std::unique_ptr<CudaEngineTransaction>(
        new CudaEngineTransaction(std::move(transaction))
    );
    result.status = {};
    return result;
}

} // namespace kimkvcache
