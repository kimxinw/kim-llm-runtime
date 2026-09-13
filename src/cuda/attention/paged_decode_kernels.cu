#include "../kv/kernels.cuh"

#include <cuda_fp16.h>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace kimkvcache::cuda_detail {
namespace {

__device__ std::size_t tensorOffset(
    DeviceLayout layout,
    std::uint32_t layer,
    std::uint32_t component,
    std::uint32_t token,
    std::uint32_t head,
    std::uint32_t dimension,
    std::uint32_t token_capacity)
{
    return (((static_cast<std::size_t>(layer) * 2 + component)
                * token_capacity + token)
                * layout.heads + head)
                * layout.dimensions
        + dimension;
}

__global__ void copyLayerTokensKernel(
    KvScalar const* source,
    std::uint32_t source_capacity,
    KvScalar* target,
    std::uint32_t target_capacity,
    std::uint32_t token_count,
    std::uint32_t layer,
    DeviceLayout layout,
    std::size_t element_count)
{
    std::size_t const index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= element_count) {
        return;
    }
    std::uint32_t dimension = static_cast<std::uint32_t>(
        index % layout.dimensions
    );
    std::size_t decoded = index / layout.dimensions;
    std::uint32_t head = static_cast<std::uint32_t>(
        decoded % layout.heads
    );
    decoded /= layout.heads;
    std::uint32_t token = static_cast<std::uint32_t>(
        decoded % token_count
    );
    std::uint32_t component = static_cast<std::uint32_t>(
        decoded / token_count
    );
    target[tensorOffset(
        layout, layer, component, token, head, dimension, target_capacity
    )] = source[tensorOffset(
        layout, layer, component, token, head, dimension, source_capacity
    )];
}

__global__ void writeLayerTokenKernel(
    KvScalar const* key,
    KvScalar const* value,
    KvScalar* target,
    std::uint32_t target_capacity,
    std::uint32_t target_token,
    std::uint32_t layer,
    DeviceLayout layout,
    std::size_t element_count)
{
    std::size_t const index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= element_count) {
        return;
    }
    std::uint32_t dimension = static_cast<std::uint32_t>(
        index % layout.dimensions
    );
    std::size_t decoded = index / layout.dimensions;
    std::uint32_t head = static_cast<std::uint32_t>(decoded % layout.heads);
    std::uint32_t component = static_cast<std::uint32_t>(
        decoded / layout.heads
    );
    KvScalar const* source = component == 0 ? key : value;
    target[tensorOffset(
        layout,
        layer,
        component,
        target_token,
        head,
        dimension,
        target_capacity
    )] = source[static_cast<std::size_t>(head) * layout.dimensions
        + dimension];
}

__device__ ::kimkvcache::DeviceBlockDescriptor findDescriptor(
    ::kimkvcache::DeviceBlockDescriptor const* descriptors,
    std::uint32_t descriptor_count,
    std::uint32_t token);

__device__ KvScalar* descriptorMutablePage(
    ::kimkvcache::DeviceBlockDescriptor descriptor,
    KvScalar* micro_pool,
    std::size_t micro_page_elements,
    KvScalar* extent_pool,
    std::size_t extent_page_elements)
{
    return descriptor.kind == PageKind::Micro
        ? micro_pool + static_cast<std::size_t>(descriptor.slot)
            * micro_page_elements
        : extent_pool + static_cast<std::size_t>(descriptor.slot)
            * extent_page_elements;
}

__global__ void writeLayerTokensKernel(
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
    std::size_t element_count)
{
    std::size_t const index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= element_count) {
        return;
    }
    std::uint32_t const dimension = static_cast<std::uint32_t>(
        index % layout.dimensions
    );
    std::size_t decoded = index / layout.dimensions;
    std::uint32_t const head = static_cast<std::uint32_t>(
        decoded % layout.heads
    );
    decoded /= layout.heads;
    std::uint32_t const token = static_cast<std::uint32_t>(
        decoded % token_count
    );
    std::uint32_t const component = static_cast<std::uint32_t>(
        decoded / token_count
    );
    std::uint32_t const logical_token = logical_token_begin + token;
    ::kimkvcache::DeviceBlockDescriptor const descriptor = findDescriptor(
        descriptors, descriptor_count, logical_token
    );
    KvScalar* const target = descriptorMutablePage(
        descriptor,
        micro_pool,
        micro_page_elements,
        extent_pool,
        extent_page_elements
    );
    KvScalar const* const source = component == 0 ? key : value;
    std::size_t const source_offset =
        (static_cast<std::size_t>(token) * layout.heads + head)
            * layout.dimensions + dimension;
    target[tensorOffset(
        layout,
        layer,
        component,
        logical_token - descriptor.logical_token_begin,
        head,
        dimension,
        descriptor.page_token_capacity
    )] = source[source_offset];
}

__global__ void writeLayerTokenBatchKernel(
    ::kimkvcache::DeviceLayerKvWriteBatchItem const* items,
    DeviceLayout layout,
    std::size_t max_element_count)
{
    std::size_t const index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= max_element_count) {
        return;
    }
    ::kimkvcache::DeviceLayerKvWriteBatchItem const item = items[blockIdx.y];
    if (item.target == nullptr || item.device_key == nullptr
        || item.device_value == nullptr || item.target_capacity == 0) {
        return;
    }

    std::size_t const token_elements = static_cast<std::size_t>(2)
        * layout.heads * layout.dimensions;
    std::size_t const copy_elements = static_cast<std::size_t>(
        item.copy_token_count
    ) * token_elements;
    if (index < copy_elements) {
        std::uint32_t const dimension = static_cast<std::uint32_t>(
            index % layout.dimensions
        );
        std::size_t decoded = index / layout.dimensions;
        std::uint32_t const head = static_cast<std::uint32_t>(
            decoded % layout.heads
        );
        decoded /= layout.heads;
        std::uint32_t const token = static_cast<std::uint32_t>(
            decoded % item.copy_token_count
        );
        std::uint32_t const component = static_cast<std::uint32_t>(
            decoded / item.copy_token_count
        );
        item.target[tensorOffset(
            layout,
            item.layer,
            component,
            token,
            head,
            dimension,
            item.target_capacity
        )] = item.copy_source[tensorOffset(
            layout,
            item.layer,
            component,
            token,
            head,
            dimension,
            item.source_capacity
        )];
        return;
    }

    std::size_t const write_index = index - copy_elements;
    if (write_index >= token_elements) {
        return;
    }
    std::uint32_t const dimension = static_cast<std::uint32_t>(
        write_index % layout.dimensions
    );
    std::size_t const decoded = write_index / layout.dimensions;
    std::uint32_t const head = static_cast<std::uint32_t>(
        decoded % layout.heads
    );
    std::uint32_t const component = static_cast<std::uint32_t>(
        decoded / layout.heads
    );
    KvScalar const* source = component == 0
        ? item.device_key : item.device_value;
    item.target[tensorOffset(
        layout,
        item.layer,
        component,
        item.target_token,
        head,
        dimension,
        item.target_capacity
    )] = source[static_cast<std::size_t>(head) * layout.dimensions
        + dimension];
}

__device__ KvScalar const* descriptorPage(
    ::kimkvcache::DeviceBlockDescriptor descriptor,
    KvScalar const* micro_pool,
    std::size_t micro_page_elements,
    KvScalar const* extent_pool,
    std::size_t extent_page_elements)
{
    return descriptor.kind == PageKind::Micro
        ? micro_pool + static_cast<std::size_t>(descriptor.slot)
            * micro_page_elements
        : extent_pool + static_cast<std::size_t>(descriptor.slot)
            * extent_page_elements;
}

__device__ ::kimkvcache::DeviceBlockDescriptor findDescriptor(
    ::kimkvcache::DeviceBlockDescriptor const* descriptors,
    std::uint32_t descriptor_count,
    std::uint32_t token)
{
    for (std::uint32_t index = 0; index < descriptor_count; ++index) {
        ::kimkvcache::DeviceBlockDescriptor const descriptor =
            descriptors[index];
        if (token >= descriptor.logical_token_begin
            && token < descriptor.logical_token_begin
                + descriptor.valid_tokens) {
            return descriptor;
        }
    }
    return {};
}

__global__ void pagedAttentionScoresKernel(
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
    float attention_scale,
    DeviceLayout layout)
{
    extern __shared__ float reduction[];
    std::uint32_t const query_head = blockIdx.x;
    std::uint32_t const group_size = query_head_count / layout.heads;
    std::uint32_t const kv_head = query_head / group_size;
    std::size_t const query_base =
        static_cast<std::size_t>(query_head) * layout.dimensions;

    for (std::uint32_t token = 0; token < token_count; ++token) {
        ::kimkvcache::DeviceBlockDescriptor const descriptor =
            findDescriptor(descriptors, descriptor_count, token);
        KvScalar const* page = descriptorPage(
            descriptor,
            micro_pool,
            micro_page_elements,
            extent_pool,
            extent_page_elements
        );
        std::uint32_t const page_token =
            token - descriptor.logical_token_begin;
        float partial = 0.0F;
        for (std::uint32_t dimension = threadIdx.x;
             dimension < layout.dimensions;
             dimension += blockDim.x) {
            __half const q = reinterpret_cast<__half const*>(query)[
                query_base + dimension
            ];
            __half const key = reinterpret_cast<__half const*>(page)[
                tensorOffset(
                    layout,
                    layer,
                    0,
                    page_token,
                    kv_head,
                    dimension,
                    descriptor.page_token_capacity
                )
            ];
            partial += __half2float(q) * __half2float(key);
        }
        reduction[threadIdx.x] = partial;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2;
             stride != 0;
             stride /= 2) {
            if (threadIdx.x < stride) {
                reduction[threadIdx.x] += reduction[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            scores[static_cast<std::size_t>(query_head) * token_count
                + token] = reduction[0] * attention_scale;
        }
        __syncthreads();
    }
}

__global__ void pagedAttentionOutputKernel(
    ::kimkvcache::DeviceBlockDescriptor const* descriptors,
    std::uint32_t descriptor_count,
    KvScalar const* micro_pool,
    std::size_t micro_page_elements,
    KvScalar const* extent_pool,
    std::size_t extent_page_elements,
    std::uint32_t token_count,
    std::uint32_t layer,
    std::uint32_t query_head_count,
    float const* scores,
    KvScalar* output,
    DeviceLayout layout)
{
    extern __shared__ float reduction[];
    std::uint32_t const query_head = blockIdx.x;
    std::uint32_t const group_size = query_head_count / layout.heads;
    std::uint32_t const kv_head = query_head / group_size;
    float const* head_scores = scores
        + static_cast<std::size_t>(query_head) * token_count;

    float local_maximum = -FLT_MAX;
    for (std::uint32_t token = threadIdx.x;
         token < token_count;
         token += blockDim.x) {
        local_maximum = fmaxf(local_maximum, head_scores[token]);
    }
    reduction[threadIdx.x] = local_maximum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2;
         stride != 0;
         stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] = fmaxf(
                reduction[threadIdx.x], reduction[threadIdx.x + stride]
            );
        }
        __syncthreads();
    }
    float const maximum = reduction[0];
    __syncthreads();
    float local_sum = 0.0F;
    for (std::uint32_t token = threadIdx.x;
         token < token_count;
         token += blockDim.x) {
        local_sum += expf(head_scores[token] - maximum);
    }
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2;
         stride != 0;
         stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float const denominator = reduction[0];

    for (std::uint32_t dimension = threadIdx.x;
         dimension < layout.dimensions;
         dimension += blockDim.x) {
        float weighted_value = 0.0F;
        for (std::uint32_t token = 0; token < token_count; ++token) {
            ::kimkvcache::DeviceBlockDescriptor const descriptor =
                findDescriptor(descriptors, descriptor_count, token);
            KvScalar const* page = descriptorPage(
                descriptor,
                micro_pool,
                micro_page_elements,
                extent_pool,
                extent_page_elements
            );
            std::uint32_t const page_token =
                token - descriptor.logical_token_begin;
            __half const value = reinterpret_cast<__half const*>(page)[
                tensorOffset(
                    layout,
                    layer,
                    1,
                    page_token,
                    kv_head,
                    dimension,
                    descriptor.page_token_capacity
                )
            ];
            weighted_value += expf(head_scores[token] - maximum)
                * __half2float(value);
        }
        reinterpret_cast<__half*>(output)[
            static_cast<std::size_t>(query_head) * layout.dimensions
                + dimension
        ] = __float2half(weighted_value / denominator);
    }
}

__global__ void pagedPrefillScoresKernel(
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
    float attention_scale,
    DeviceLayout layout)
{
    extern __shared__ float reduction[];
    std::uint32_t const query_head = blockIdx.x;
    std::uint32_t const query_token = blockIdx.y;
    std::uint32_t const visible_tokens =
        total_token_count - query_token_count + query_token + 1;
    std::uint32_t const group_size = query_head_count / layout.heads;
    std::uint32_t const kv_head = query_head / group_size;
    std::size_t const query_base =
        (static_cast<std::size_t>(query_token) * query_head_count
            + query_head) * layout.dimensions;
    float* const head_scores = scores
        + (static_cast<std::size_t>(query_token) * query_head_count
            + query_head) * total_token_count;

    for (std::uint32_t token = 0; token < visible_tokens; ++token) {
        ::kimkvcache::DeviceBlockDescriptor const descriptor =
            findDescriptor(descriptors, descriptor_count, token);
        KvScalar const* const page = descriptorPage(
            descriptor,
            micro_pool,
            micro_page_elements,
            extent_pool,
            extent_page_elements
        );
        float partial = 0.0F;
        for (std::uint32_t dimension = threadIdx.x;
             dimension < layout.dimensions;
             dimension += blockDim.x) {
            __half const q = reinterpret_cast<__half const*>(query)[
                query_base + dimension
            ];
            __half const key = reinterpret_cast<__half const*>(page)[
                tensorOffset(
                    layout,
                    layer,
                    0,
                    token - descriptor.logical_token_begin,
                    kv_head,
                    dimension,
                    descriptor.page_token_capacity
                )
            ];
            partial += __half2float(q) * __half2float(key);
        }
        reduction[threadIdx.x] = partial;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2;
             stride != 0;
             stride /= 2) {
            if (threadIdx.x < stride) {
                reduction[threadIdx.x] += reduction[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            head_scores[token] = reduction[0] * attention_scale;
        }
        __syncthreads();
    }
}

__global__ void pagedPrefillOutputKernel(
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
    float const* scores,
    KvScalar* output,
    DeviceLayout layout)
{
    extern __shared__ float reduction[];
    std::uint32_t const query_head = blockIdx.x;
    std::uint32_t const query_token = blockIdx.y;
    std::uint32_t const visible_tokens =
        total_token_count - query_token_count + query_token + 1;
    std::uint32_t const group_size = query_head_count / layout.heads;
    std::uint32_t const kv_head = query_head / group_size;
    float const* const head_scores = scores
        + (static_cast<std::size_t>(query_token) * query_head_count
            + query_head) * total_token_count;

    float local_maximum = -FLT_MAX;
    for (std::uint32_t token = threadIdx.x;
         token < visible_tokens;
         token += blockDim.x) {
        local_maximum = fmaxf(local_maximum, head_scores[token]);
    }
    reduction[threadIdx.x] = local_maximum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2;
         stride != 0;
         stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] = fmaxf(
                reduction[threadIdx.x], reduction[threadIdx.x + stride]
            );
        }
        __syncthreads();
    }
    float const maximum = reduction[0];
    __syncthreads();
    float local_sum = 0.0F;
    for (std::uint32_t token = threadIdx.x;
         token < visible_tokens;
         token += blockDim.x) {
        local_sum += expf(head_scores[token] - maximum);
    }
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2;
         stride != 0;
         stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float const denominator = reduction[0];

    for (std::uint32_t dimension = threadIdx.x;
         dimension < layout.dimensions;
         dimension += blockDim.x) {
        float weighted_value = 0.0F;
        for (std::uint32_t token = 0; token < visible_tokens; ++token) {
            ::kimkvcache::DeviceBlockDescriptor const descriptor =
                findDescriptor(descriptors, descriptor_count, token);
            KvScalar const* const page = descriptorPage(
                descriptor,
                micro_pool,
                micro_page_elements,
                extent_pool,
                extent_page_elements
            );
            __half const value = reinterpret_cast<__half const*>(page)[
                tensorOffset(
                    layout,
                    layer,
                    1,
                    token - descriptor.logical_token_begin,
                    kv_head,
                    dimension,
                    descriptor.page_token_capacity
                )
            ];
            weighted_value += expf(head_scores[token] - maximum)
                * __half2float(value);
        }
        reinterpret_cast<__half*>(output)[
            (static_cast<std::size_t>(query_token) * query_head_count
                + query_head) * layout.dimensions + dimension
        ] = __float2half(weighted_value / denominator);
    }
}

__global__ void pagedAttentionScoresBatchKernel(
    ::kimkvcache::DevicePagedDecodeBatchItem const* items,
    KvScalar const* micro_pool,
    std::size_t micro_page_elements,
    KvScalar const* extent_pool,
    std::size_t extent_page_elements,
    std::uint32_t query_head_count,
    DeviceLayout layout)
{
    extern __shared__ float reduction[];
    ::kimkvcache::DevicePagedDecodeBatchItem const item = items[blockIdx.y];
    if (item.device_descriptors == nullptr || item.descriptor_count == 0
        || item.token_count == 0 || item.device_query == nullptr
        || item.device_scores == nullptr) {
        return;
    }

    std::uint32_t const query_head = blockIdx.x;
    std::uint32_t const group_size = query_head_count / layout.heads;
    std::uint32_t const kv_head = query_head / group_size;
    std::size_t const query_base =
        static_cast<std::size_t>(query_head) * layout.dimensions;

    for (std::uint32_t token = 0; token < item.token_count; ++token) {
        ::kimkvcache::DeviceBlockDescriptor const descriptor = findDescriptor(
            item.device_descriptors, item.descriptor_count, token
        );
        KvScalar const* page = descriptorPage(
            descriptor,
            micro_pool,
            micro_page_elements,
            extent_pool,
            extent_page_elements
        );
        std::uint32_t const page_token =
            token - descriptor.logical_token_begin;
        float partial = 0.0F;
        for (std::uint32_t dimension = threadIdx.x;
             dimension < layout.dimensions;
             dimension += blockDim.x) {
            __half const q = reinterpret_cast<__half const*>(
                item.device_query
            )[query_base + dimension];
            __half const key = reinterpret_cast<__half const*>(page)[
                tensorOffset(
                    layout,
                    item.layer,
                    0,
                    page_token,
                    kv_head,
                    dimension,
                    descriptor.page_token_capacity
                )
            ];
            partial += __half2float(q) * __half2float(key);
        }
        reduction[threadIdx.x] = partial;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2;
             stride != 0;
             stride /= 2) {
            if (threadIdx.x < stride) {
                reduction[threadIdx.x] += reduction[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            item.device_scores[
                static_cast<std::size_t>(query_head) * item.token_count
                    + token
            ] = reduction[0] * item.attention_scale;
        }
        __syncthreads();
    }
}

__global__ void pagedAttentionOutputBatchKernel(
    ::kimkvcache::DevicePagedDecodeBatchItem const* items,
    KvScalar const* micro_pool,
    std::size_t micro_page_elements,
    KvScalar const* extent_pool,
    std::size_t extent_page_elements,
    std::uint32_t query_head_count,
    DeviceLayout layout)
{
    extern __shared__ float reduction[];
    ::kimkvcache::DevicePagedDecodeBatchItem const item = items[blockIdx.y];
    if (item.device_descriptors == nullptr || item.descriptor_count == 0
        || item.token_count == 0 || item.device_scores == nullptr
        || item.device_output == nullptr) {
        return;
    }

    std::uint32_t const query_head = blockIdx.x;
    std::uint32_t const group_size = query_head_count / layout.heads;
    std::uint32_t const kv_head = query_head / group_size;
    float const* head_scores = item.device_scores
        + static_cast<std::size_t>(query_head) * item.token_count;

    float local_maximum = -FLT_MAX;
    for (std::uint32_t token = threadIdx.x;
         token < item.token_count;
         token += blockDim.x) {
        local_maximum = fmaxf(local_maximum, head_scores[token]);
    }
    reduction[threadIdx.x] = local_maximum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2;
         stride != 0;
         stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] = fmaxf(
                reduction[threadIdx.x], reduction[threadIdx.x + stride]
            );
        }
        __syncthreads();
    }
    float const maximum = reduction[0];
    __syncthreads();
    float local_sum = 0.0F;
    for (std::uint32_t token = threadIdx.x;
         token < item.token_count;
         token += blockDim.x) {
        local_sum += expf(head_scores[token] - maximum);
    }
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2;
         stride != 0;
         stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float const denominator = reduction[0];

    for (std::uint32_t dimension = threadIdx.x;
         dimension < layout.dimensions;
         dimension += blockDim.x) {
        float weighted_value = 0.0F;
        for (std::uint32_t token = 0; token < item.token_count; ++token) {
            ::kimkvcache::DeviceBlockDescriptor const descriptor =
                findDescriptor(
                    item.device_descriptors, item.descriptor_count, token
                );
            KvScalar const* page = descriptorPage(
                descriptor,
                micro_pool,
                micro_page_elements,
                extent_pool,
                extent_page_elements
            );
            std::uint32_t const page_token =
                token - descriptor.logical_token_begin;
            __half const value = reinterpret_cast<__half const*>(page)[
                tensorOffset(
                    layout,
                    item.layer,
                    1,
                    page_token,
                    kv_head,
                    dimension,
                    descriptor.page_token_capacity
                )
            ];
            weighted_value += expf(head_scores[token] - maximum)
                * __half2float(value);
        }
        reinterpret_cast<__half*>(item.device_output)[
            static_cast<std::size_t>(query_head) * layout.dimensions
                + dimension
        ] = __float2half(weighted_value / denominator);
    }
}

#if defined(KIM_KV_ENABLE_FUSED_ATTENTION)
// Four warps own disjoint sequence ranges. Each warp keeps a stable online
// softmax and its weighted V vector in registers; one block barrier merges
// the four partials. No score tensor is written to global memory.
inline constexpr unsigned int kFusedWarps = 4;
inline constexpr unsigned int kFusedThreads = kFusedWarps * 32;
inline constexpr unsigned int kFusedMaxDimensions = 128;

__device__ void fusedPagedAttention(
    ::kimkvcache::DeviceBlockDescriptor const* descriptors,
    std::uint32_t descriptor_count,
    KvScalar const* micro_pool,
    std::size_t micro_page_elements,
    KvScalar const* extent_pool,
    std::size_t extent_page_elements,
    std::uint32_t visible_tokens,
    std::uint32_t layer,
    std::uint32_t query_head_count,
    KvScalar const* query,
    KvScalar* output,
    float attention_scale,
    DeviceLayout layout)
{
    extern __shared__ float partials[];
    unsigned int const lane = threadIdx.x % 32;
    unsigned int const warp = threadIdx.x / 32;
    unsigned int const query_head = blockIdx.x;
    unsigned int const kv_head = query_head / (query_head_count / layout.heads);
    float q[kFusedMaxDimensions / 32];
    float weighted[kFusedMaxDimensions / 32] = {};
    for (unsigned int i = 0; i < kFusedMaxDimensions / 32; ++i) {
        unsigned int const dimension = lane + i * 32;
        q[i] = dimension < layout.dimensions
            ? __half2float(reinterpret_cast<__half const*>(query)[
                static_cast<std::size_t>(query_head) * layout.dimensions
                    + dimension]) : 0.0F;
    }
    float maximum = -FLT_MAX;
    float denominator = 0.0F;
    unsigned int const range = (visible_tokens + kFusedWarps - 1) / kFusedWarps;
    unsigned int const begin = warp * range;
    unsigned int const end = min(begin + range, visible_tokens);
    for (unsigned int token = begin; token < end; ++token) {
        auto const descriptor = findDescriptor(descriptors, descriptor_count, token);
        KvScalar const* const page = descriptorPage(
            descriptor, micro_pool, micro_page_elements,
            extent_pool, extent_page_elements);
        unsigned int const page_token = token - descriptor.logical_token_begin;
        float dot = 0.0F;
        for (unsigned int i = 0; i < kFusedMaxDimensions / 32; ++i) {
            unsigned int const dimension = lane + i * 32;
            if (dimension < layout.dimensions) {
                float const key = __half2float(reinterpret_cast<__half const*>(page)[
                    tensorOffset(layout, layer, 0, page_token, kv_head,
                        dimension, descriptor.page_token_capacity)]);
                dot += q[i] * key;
            }
        }
        for (unsigned int offset = 16; offset != 0; offset /= 2) {
            dot += __shfl_down_sync(0xffffffffU, dot, offset);
        }
        float const score = __shfl_sync(0xffffffffU, dot, 0) * attention_scale;
        float const next_maximum = fmaxf(maximum, score);
        float const rescale = expf(maximum - next_maximum);
        float const weight = expf(score - next_maximum);
        denominator = denominator * rescale + weight;
        for (unsigned int i = 0; i < kFusedMaxDimensions / 32; ++i) {
            unsigned int const dimension = lane + i * 32;
            if (dimension < layout.dimensions) {
                float const value = __half2float(reinterpret_cast<__half const*>(page)[
                    tensorOffset(layout, layer, 1, page_token, kv_head,
                        dimension, descriptor.page_token_capacity)]);
                weighted[i] = weighted[i] * rescale + weight * value;
            }
        }
        maximum = next_maximum;
    }
    if (lane == 0) {
        partials[warp] = maximum;
        partials[kFusedWarps + warp] = denominator;
    }
    for (unsigned int i = 0; i < kFusedMaxDimensions / 32; ++i) {
        unsigned int const dimension = lane + i * 32;
        if (dimension < layout.dimensions) {
            partials[2 * kFusedWarps + warp * layout.dimensions + dimension]
                = weighted[i];
        }
    }
    __syncthreads();
    float merged_maximum = -FLT_MAX;
    for (unsigned int i = 0; i < kFusedWarps; ++i) {
        merged_maximum = fmaxf(merged_maximum, partials[i]);
    }
    float scales[kFusedWarps];
    float merged_denominator = 0.0F;
    for (unsigned int i = 0; i < kFusedWarps; ++i) {
        scales[i] = expf(partials[i] - merged_maximum);
        merged_denominator += scales[i] * partials[kFusedWarps + i];
    }
    for (unsigned int dimension = threadIdx.x;
         dimension < layout.dimensions; dimension += blockDim.x) {
        float value = 0.0F;
        for (unsigned int i = 0; i < kFusedWarps; ++i) {
            value += scales[i]
                * partials[2 * kFusedWarps + i * layout.dimensions + dimension];
        }
        reinterpret_cast<__half*>(output)[
            static_cast<std::size_t>(query_head) * layout.dimensions + dimension]
            = __float2half(value / merged_denominator);
    }
}

__global__ void pagedAttentionFusedKernel(
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
    KvScalar* output,
    float attention_scale,
    DeviceLayout layout)
{
    unsigned int const query_token = blockIdx.y;
    std::size_t const offset = static_cast<std::size_t>(query_token)
        * query_head_count * layout.dimensions;
    fusedPagedAttention(descriptors, descriptor_count,
        micro_pool, micro_page_elements, extent_pool, extent_page_elements,
        total_token_count - query_token_count + query_token + 1,
        layer, query_head_count, query + offset, output + offset,
        attention_scale, layout);
}

__global__ void pagedAttentionFusedBatchKernel(
    ::kimkvcache::DevicePagedDecodeBatchItem const* items,
    KvScalar const* micro_pool,
    std::size_t micro_page_elements,
    KvScalar const* extent_pool,
    std::size_t extent_page_elements,
    std::uint32_t query_head_count,
    DeviceLayout layout)
{
    auto const item = items[blockIdx.y];
    if (item.device_descriptors == nullptr || item.descriptor_count == 0
        || item.token_count == 0 || item.device_query == nullptr
        || item.device_output == nullptr || item.device_scores == nullptr) {
        return;
    }
    fusedPagedAttention(item.device_descriptors, item.descriptor_count,
        micro_pool, micro_page_elements, extent_pool, extent_page_elements,
        item.token_count, item.layer, query_head_count,
        item.device_query, item.device_output, item.attention_scale, layout);
}

#endif

[[nodiscard]] dim3 gridFor(std::size_t element_count) noexcept
{
    std::size_t const blocks =
        (element_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    return dim3(static_cast<unsigned int>(blocks));
}

} // namespace

void launchCopyLayerTokens(
    KvScalar const* source,
    std::uint32_t source_capacity,
    KvScalar* target,
    std::uint32_t target_capacity,
    std::uint32_t token_count,
    std::uint32_t layer,
    DeviceLayout layout,
    cudaStream_t stream)
{
    std::size_t const elements = static_cast<std::size_t>(2)
        * token_count * layout.heads * layout.dimensions;
    copyLayerTokensKernel<<<gridFor(elements), kThreadsPerBlock, 0, stream>>>(
        source,
        source_capacity,
        target,
        target_capacity,
        token_count,
        layer,
        layout,
        elements
    );
}

void launchWriteLayerToken(
    KvScalar const* key,
    KvScalar const* value,
    KvScalar* target,
    std::uint32_t target_capacity,
    std::uint32_t target_token,
    std::uint32_t layer,
    DeviceLayout layout,
    cudaStream_t stream)
{
    std::size_t const elements = static_cast<std::size_t>(2)
        * layout.heads * layout.dimensions;
    writeLayerTokenKernel<<<gridFor(elements), kThreadsPerBlock, 0, stream>>>(
        key,
        value,
        target,
        target_capacity,
        target_token,
        layer,
        layout,
        elements
    );
}

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
    cudaStream_t stream)
{
    std::size_t const elements = static_cast<std::size_t>(2)
        * token_count * layout.heads * layout.dimensions;
    writeLayerTokensKernel<<<
        gridFor(elements), kThreadsPerBlock, 0, stream>>>(
        descriptors,
        descriptor_count,
        micro_pool,
        micro_page_elements,
        extent_pool,
        extent_page_elements,
        logical_token_begin,
        token_count,
        layer,
        key,
        value,
        layout,
        elements
    );
}

void launchWriteLayerTokenBatch(
    ::kimkvcache::DeviceLayerKvWriteBatchItem const* items,
    std::uint32_t item_count,
    std::uint32_t max_copy_token_count,
    DeviceLayout layout,
    cudaStream_t stream)
{
    std::size_t const elements = static_cast<std::size_t>(
        max_copy_token_count + 1
    ) * 2 * layout.heads * layout.dimensions;
    dim3 const scalar_grid = gridFor(elements);
    dim3 const grid(scalar_grid.x, item_count);
    writeLayerTokenBatchKernel<<<grid, kThreadsPerBlock, 0, stream>>>(
        items, layout, elements
    );
}

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
    cudaStream_t stream)
{
#if defined(KIM_KV_ENABLE_FUSED_ATTENTION)
    if (layout.dimensions <= kFusedMaxDimensions) {
        std::size_t const shared = kFusedWarps * (layout.dimensions + 2)
            * sizeof(float);
        pagedAttentionFusedKernel<<<query_head_count, kFusedThreads, shared, stream>>>(
            descriptors, descriptor_count, micro_pool, micro_page_elements,
            extent_pool, extent_page_elements, token_count, 1, layer,
            query_head_count, query, output, attention_scale, layout);
        return;
    }
#endif
    std::size_t const shared_bytes = kThreadsPerBlock * sizeof(float);
    pagedAttentionScoresKernel<<<
        query_head_count, kThreadsPerBlock, shared_bytes, stream>>>(
        descriptors,
        descriptor_count,
        micro_pool,
        micro_page_elements,
        extent_pool,
        extent_page_elements,
        token_count,
        layer,
        query_head_count,
        query,
        scores,
        attention_scale,
        layout
    );
    pagedAttentionOutputKernel<<<
        query_head_count, kThreadsPerBlock, shared_bytes, stream>>>(
        descriptors,
        descriptor_count,
        micro_pool,
        micro_page_elements,
        extent_pool,
        extent_page_elements,
        token_count,
        layer,
        query_head_count,
        scores,
        output,
        layout
    );
}

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
    cudaStream_t stream)
{
#if defined(KIM_KV_ENABLE_FUSED_ATTENTION)
    if (layout.dimensions <= kFusedMaxDimensions) {
        std::size_t const shared = kFusedWarps * (layout.dimensions + 2)
            * sizeof(float);
        dim3 const fused_grid(query_head_count, query_token_count);
        pagedAttentionFusedKernel<<<fused_grid, kFusedThreads, shared, stream>>>(
            descriptors, descriptor_count, micro_pool, micro_page_elements,
            extent_pool, extent_page_elements, total_token_count, query_token_count,
            layer, query_head_count, query, output, attention_scale, layout);
        return;
    }
#endif
    std::size_t const shared_bytes = kThreadsPerBlock * sizeof(float);
    dim3 const grid(query_head_count, query_token_count);
    pagedPrefillScoresKernel<<<
        grid, kThreadsPerBlock, shared_bytes, stream>>>(
        descriptors,
        descriptor_count,
        micro_pool,
        micro_page_elements,
        extent_pool,
        extent_page_elements,
        total_token_count,
        query_token_count,
        layer,
        query_head_count,
        query,
        scores,
        attention_scale,
        layout
    );
    pagedPrefillOutputKernel<<<
        grid, kThreadsPerBlock, shared_bytes, stream>>>(
        descriptors,
        descriptor_count,
        micro_pool,
        micro_page_elements,
        extent_pool,
        extent_page_elements,
        total_token_count,
        query_token_count,
        layer,
        query_head_count,
        scores,
        output,
        layout
    );
}

void launchPagedDecodeAttentionBatch(
    ::kimkvcache::DevicePagedDecodeBatchItem const* items,
    std::uint32_t item_count,
    KvScalar const* micro_pool,
    std::size_t micro_page_elements,
    KvScalar const* extent_pool,
    std::size_t extent_page_elements,
    std::uint32_t query_head_count,
    DeviceLayout layout,
    cudaStream_t stream)
{
#if defined(KIM_KV_ENABLE_FUSED_ATTENTION)
    if (layout.dimensions <= kFusedMaxDimensions) {
        std::size_t const shared = kFusedWarps * (layout.dimensions + 2)
            * sizeof(float);
        dim3 const fused_grid(query_head_count, item_count);
        pagedAttentionFusedBatchKernel<<<fused_grid, kFusedThreads, shared, stream>>>(
            items, micro_pool, micro_page_elements, extent_pool,
            extent_page_elements, query_head_count, layout);
        return;
    }
#endif
    std::size_t const shared_bytes = kThreadsPerBlock * sizeof(float);
    dim3 const grid(query_head_count, item_count);
    pagedAttentionScoresBatchKernel<<<
        grid, kThreadsPerBlock, shared_bytes, stream>>>(
        items,
        micro_pool,
        micro_page_elements,
        extent_pool,
        extent_page_elements,
        query_head_count,
        layout
    );
    pagedAttentionOutputBatchKernel<<<
        grid, kThreadsPerBlock, shared_bytes, stream>>>(
        items,
        micro_pool,
        micro_page_elements,
        extent_pool,
        extent_page_elements,
        query_head_count,
        layout
    );
}

} // namespace kimkvcache::cuda_detail
