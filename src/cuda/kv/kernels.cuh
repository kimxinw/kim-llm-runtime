#pragma once

#include "storage.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace kimkvcache::cuda_detail {

inline constexpr unsigned int kThreadsPerBlock = 256;

struct DeviceLayout final {
    std::uint32_t layers;
    std::uint32_t heads;
    std::uint32_t dimensions;
};

struct DeviceBlockDescriptor final {
    KvScalar const* data;
    std::uint32_t logical_token_begin;
    std::uint16_t valid_tokens;
    std::uint16_t token_capacity;
};

struct PromotionSlots final {
    std::uint32_t slots[kPromotionSourcePageCount];
};

void launchCopyPageTokens(
    KvScalar const* source,
    std::uint32_t source_capacity,
    KvScalar* target,
    std::uint32_t target_capacity,
    std::uint32_t token_count,
    DeviceLayout layout,
    std::size_t element_count,
    cudaStream_t stream
);

void launchAppendTokens(
    KvScalar const* source,
    std::uint32_t source_token_capacity,
    std::uint32_t source_token_begin,
    KvScalar* target,
    std::uint32_t target_token_capacity,
    std::uint32_t target_token_begin,
    std::uint32_t segment_token_count,
    DeviceLayout layout,
    std::size_t element_count,
    cudaStream_t stream
);

void launchPromotion(
    KvScalar const* micro_pool,
    std::size_t micro_page_elements,
    PromotionSlots source_slots,
    KvScalar* target,
    DeviceLayout layout,
    std::size_t element_count,
    cudaStream_t stream
);

void launchGather(
    DeviceBlockDescriptor const* descriptors,
    std::uint32_t descriptor_count,
    KvScalar* output,
    std::uint32_t total_tokens,
    DeviceLayout layout,
    std::size_t element_count,
    cudaStream_t stream
);

void launchAttentionScores(
    KvScalar const* contiguous_kv,
    std::uint32_t token_count,
    float const* query,
    float* scores,
    DeviceLayout layout,
    cudaStream_t stream
);

void launchAttentionOutput(
    KvScalar const* contiguous_kv,
    std::uint32_t token_count,
    float const* scores,
    float* output,
    DeviceLayout layout,
    cudaStream_t stream
);

void launchCopyLayerTokens(
    KvScalar const* source,
    std::uint32_t source_capacity,
    KvScalar* target,
    std::uint32_t target_capacity,
    std::uint32_t token_count,
    std::uint32_t layer,
    DeviceLayout layout,
    cudaStream_t stream
);

void launchWriteLayerToken(
    KvScalar const* key,
    KvScalar const* value,
    KvScalar* target,
    std::uint32_t target_capacity,
    std::uint32_t target_token,
    std::uint32_t layer,
    DeviceLayout layout,
    cudaStream_t stream
);

// Writes a contiguous [token][kv_head][dimension] K/V chunk directly into
// the paged mapping. The chunk may span any number of page descriptors.
void launchWriteLayerTokens(
    ::kimkvcache::DeviceBlockDescriptor const* descriptors,
    std::uint32_t descriptor_count,
    KvScalar* micro_pool,
    std::size_t micro_page_elements,
    KvScalar* extent_pool,
    std::size_t extent_page_elements,
    std::uint32_t logical_token_begin,
    std::uint32_t token_count,
    std::uint32_t layer,
    KvScalar const* key,
    KvScalar const* value,
    DeviceLayout layout,
    cudaStream_t stream
);

void launchWriteLayerTokenBatch(
    ::kimkvcache::DeviceLayerKvWriteBatchItem const* items,
    std::uint32_t item_count,
    std::uint32_t max_copy_token_count,
    DeviceLayout layout,
    cudaStream_t stream
);

void launchPagedDecodeAttention(
    ::kimkvcache::DeviceBlockDescriptor const* descriptors,
    std::uint32_t descriptor_count,
    KvScalar const* micro_pool,
    std::size_t micro_page_elements,
    KvScalar const* extent_pool,
    std::size_t extent_page_elements,
    std::uint32_t token_count,
    std::uint32_t layer,
    std::uint32_t query_head_count,
    KvScalar const* query,
    float* scores,
    KvScalar* output,
    float attention_scale,
    DeviceLayout layout,
    cudaStream_t stream
);

// Causal paged attention for a contiguous query chunk at the end of the
// reserved sequence. Query i observes history plus chunk positions [0, i].
void launchPagedPrefillAttention(
    ::kimkvcache::DeviceBlockDescriptor const* descriptors,
    std::uint32_t descriptor_count,
    KvScalar const* micro_pool,
    std::size_t micro_page_elements,
    KvScalar const* extent_pool,
    std::size_t extent_page_elements,
    std::uint32_t total_token_count,
    std::uint32_t query_token_count,
    std::uint32_t layer,
    std::uint32_t query_head_count,
    KvScalar const* query,
    float* scores,
    KvScalar* output,
    float attention_scale,
    DeviceLayout layout,
    cudaStream_t stream
);

void launchPagedDecodeAttentionBatch(
    ::kimkvcache::DevicePagedDecodeBatchItem const* items,
    std::uint32_t item_count,
    KvScalar const* micro_pool,
    std::size_t micro_page_elements,
    KvScalar const* extent_pool,
    std::size_t extent_page_elements,
    std::uint32_t query_head_count,
    DeviceLayout layout,
    cudaStream_t stream
);

} // namespace kimkvcache::cuda_detail
