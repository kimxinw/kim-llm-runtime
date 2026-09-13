#include "kernels.cuh"
#include "kim-kv/engine/engine_kv.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace kimkvcache;
std::ofstream benchmark;

void check(cudaError_t status)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

template<class Launch> double measure(Launch launch)
{
    launch();
    check(cudaGetLastError());
    if (!benchmark.is_open()) {
        return 0.0;
    }
    for (int i = 0; i < 5; ++i) { launch(); }
    check(cudaDeviceSynchronize());
    cudaEvent_t start, end;
    check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
    std::vector<double> samples;
    for (int repeat = 0; repeat < 10; ++repeat) {
        check(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) { launch(); }
        check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
        float elapsed;
        check(cudaEventElapsedTime(&elapsed, start, end));
        samples.push_back(elapsed * 1000.0 / 20);
    }
    check(cudaEventDestroy(start)); check(cudaEventDestroy(end));
    std::sort(samples.begin(), samples.end());
    return (samples[4] + samples[5]) / 2;
}

KvScalar encode(float value)
{
    __half half = __float2half(value);
    KvScalar bits;
    std::memcpy(&bits, &half, sizeof(bits));
    return bits;
}

float decode(KvScalar bits)
{
    return __half2float(__ushort_as_half(bits));
}

template<class T> struct Buffer {
    T* data = nullptr;
    explicit Buffer(std::size_t count)
    {
        check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~Buffer() { cudaFree(data); }
    Buffer(Buffer const&) = delete;
    Buffer& operator=(Buffer const&) = delete;
    void upload(std::vector<T> const& values)
    {
        check(cudaMemcpy(data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
};

struct Fixture {
    cuda_detail::DeviceLayout layout;
    unsigned int heads;
    unsigned int queries;
    std::vector<unsigned int> lengths;
    std::vector<std::vector<KvScalar>> keys, values;
    std::vector<KvScalar> query, micro, extent;
    std::vector<DeviceBlockDescriptor> descriptors;
    std::vector<unsigned int> descriptor_offsets, descriptor_counts;
    std::size_t micro_elements, extent_elements;
    float scale;

    Fixture(unsigned int dimensions, unsigned int query_count,
        std::vector<unsigned int> token_counts, bool extreme = false)
        : layout{2, 4, dimensions}, heads(32), queries(query_count),
          lengths(std::move(token_counts)),
          micro_elements(4ULL * 8 * layout.heads * dimensions),
          extent_elements(4ULL * 64 * layout.heads * dimensions),
          scale(1.0F / std::sqrt(static_cast<float>(dimensions)))
    {
        unsigned int micro_slots = 0, extent_slots = 0;
        for (unsigned int request = 0; request < lengths.size(); ++request) {
            unsigned int length = lengths[request];
            keys.emplace_back(length * layout.heads * dimensions);
            values.emplace_back(length * layout.heads * dimensions);
            for (std::size_t i = 0; i < keys.back().size(); ++i) {
                keys.back()[i] = encode((extreme ? 10.0F : 0.8F)
                    * std::sin(static_cast<float>(i + request * 19 + 1) * 0.071F));
                values.back()[i] = encode(std::cos(static_cast<float>(i + request * 7) * 0.033F));
            }
            for (unsigned int q = 0; q < queries; ++q) {
                for (unsigned int h = 0; h < heads; ++h) {
                    for (unsigned int d = 0; d < dimensions; ++d) {
                        query.push_back(encode((extreme ? 80.0F : 0.7F)
                            * std::cos(static_cast<float>(q * 37 + h * 11 + d + request) * 0.053F)));
                    }
                }
            }
            descriptor_offsets.push_back(static_cast<unsigned int>(descriptors.size()));
            unsigned int token = 0, page_index = 0;
            while (token < length) {
                bool const small = page_index++ % 2 == 0;
                unsigned int capacity = small ? 8 : 64;
                unsigned int valid = std::min(capacity, length - token);
                unsigned int slot = small ? micro_slots++ : extent_slots++;
                auto& pool = small ? micro : extent;
                auto const page_elements = small ? micro_elements : extent_elements;
                pool.resize((slot + 1) * page_elements);
                descriptors.push_back(DeviceBlockDescriptor{slot, 1, token,
                    static_cast<std::uint16_t>(valid), static_cast<std::uint16_t>(capacity),
                    small ? PageKind::Micro : PageKind::Extent});
                for (unsigned int t = 0; t < valid; ++t) {
                    for (unsigned int h = 0; h < layout.heads; ++h) {
                        for (unsigned int d = 0; d < dimensions; ++d) {
                            auto const dense = ((token + t) * layout.heads + h) * dimensions + d;
                            for (unsigned int component = 0; component < 2; ++component) {
                                auto const paged = slot * page_elements
                                    + (((2 + component) * capacity + t) * layout.heads + h) * dimensions + d;
                                pool[paged] = component == 0 ? keys.back()[dense] : values.back()[dense];
                            }
                        }
                    }
                }
                token += valid;
            }
            descriptor_counts.push_back(static_cast<unsigned int>(descriptors.size()) - descriptor_offsets.back());
        }
        micro.resize(std::max<std::size_t>(micro.size(), 1));
        extent.resize(std::max<std::size_t>(extent.size(), 1));
    }

    double validate(std::vector<KvScalar> const& output) const
    {
        double max_error = 0.0;
        for (unsigned int request = 0; request < lengths.size(); ++request) {
            unsigned int const length = lengths[request];
            for (unsigned int q = 0; q < queries; ++q) {
                unsigned int const visible = length - queries + q + 1;
                for (unsigned int h = 0; h < heads; ++h) {
                    unsigned int const kv_head = h / (heads / layout.heads);
                    std::vector<double> scores(visible);
                    auto const qbase = ((request * queries + q) * heads + h) * layout.dimensions;
                    for (unsigned int token = 0; token < visible; ++token) {
                        double dot = 0.0;
                        for (unsigned int d = 0; d < layout.dimensions; ++d) {
                            auto const index = (token * layout.heads + kv_head) * layout.dimensions + d;
                            dot += static_cast<double>(decode(query[qbase + d])) * decode(keys[request][index]);
                        }
                        scores[token] = dot * scale;
                    }
                    double const maximum = *std::max_element(scores.begin(), scores.end());
                    double denominator = 0.0;
                    for (double& score : scores) {
                        score = std::exp(score - maximum);
                        denominator += score;
                    }
                    for (unsigned int d = 0; d < layout.dimensions; ++d) {
                        double weighted = 0.0;
                        for (unsigned int token = 0; token < visible; ++token) {
                            auto const index = (token * layout.heads + kv_head) * layout.dimensions + d;
                            weighted += scores[token] * decode(values[request][index]);
                        }
                        double const expected = weighted / denominator;
                        double const actual = decode(output[qbase + d]);
                        double const error = std::abs(actual - expected);
                        max_error = std::max(max_error, error);
                        if (!std::isfinite(actual) || error > 0.004 + 0.002 * std::abs(expected)) {
                            throw std::runtime_error("attention differs from causal dense CPU reference; error=" + std::to_string(error));
                        }
                    }
                }
            }
        }
        return max_error;
    }
};

void run(unsigned int dimensions, unsigned int queries,
    std::vector<unsigned int> lengths, bool extreme = false)
{
    Fixture f(dimensions, queries, std::move(lengths), extreme);
    Buffer<KvScalar> micro(f.micro.size()), extent(f.extent.size()), query(f.query.size()), output(f.query.size());
    Buffer<DeviceBlockDescriptor> descriptors(f.descriptors.size());
    Buffer<float> scores(f.lengths.size() * queries * f.heads * f.lengths.front());
    micro.upload(f.micro); extent.upload(f.extent); query.upload(f.query); descriptors.upload(f.descriptors);
    double gpu_us = 0.0;
    if (queries == 1 && f.lengths.size() > 1) {
        std::vector<DevicePagedDecodeBatchItem> items;
        for (unsigned int i = 0; i < f.lengths.size(); ++i) {
            auto const offset = i * f.heads * dimensions;
            items.push_back(DevicePagedDecodeBatchItem{descriptors.data + f.descriptor_offsets[i],
                f.descriptor_counts[i], f.lengths[i], 1, query.data + offset,
                scores.data + i * f.heads * f.lengths.front(), output.data + offset, f.scale});
        }
        // A disabled lane must not dereference any pointer or alter valid lanes.
        items.push_back(DevicePagedDecodeBatchItem{});
        Buffer<DevicePagedDecodeBatchItem> device_items(items.size());
        device_items.upload(items);
        gpu_us = measure([&] { cuda_detail::launchPagedDecodeAttentionBatch(device_items.data,
            static_cast<unsigned int>(items.size()), micro.data, f.micro_elements,
            extent.data, f.extent_elements, f.heads, f.layout, nullptr); });
        check(cudaGetLastError()); check(cudaDeviceSynchronize());
    } else if (queries == 1) {
        gpu_us = measure([&] { cuda_detail::launchPagedDecodeAttention(descriptors.data, f.descriptor_counts[0],
            micro.data, f.micro_elements, extent.data, f.extent_elements, f.lengths[0],
            1, f.heads, query.data, scores.data, output.data, f.scale, f.layout, nullptr); });
    } else {
        gpu_us = measure([&] { cuda_detail::launchPagedPrefillAttention(descriptors.data, f.descriptor_counts[0],
            micro.data, f.micro_elements, extent.data, f.extent_elements, f.lengths[0],
            queries, 1, f.heads, query.data, scores.data, output.data, f.scale, f.layout, nullptr); });
    }
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    std::vector<KvScalar> result(f.query.size());
    check(cudaMemcpy(result.data(), output.data, result.size() * sizeof(KvScalar), cudaMemcpyDeviceToHost));
    double const error = f.validate(result);
    std::cout << "dim=" << dimensions << " queries=" << queries << " batch=" << f.lengths.size()
        << " tokens=" << f.lengths[0] << " max_error=" << error << '\n';
    if (benchmark.is_open()) {
        benchmark << dimensions << ',' << queries << ',' << f.lengths.size() << ','
            << f.lengths[0] << ',' << extreme << ',' << gpu_us << ',' << error << '\n';
    }
}
} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc == 3 && std::string(argv[1]) == "--benchmark") {
            benchmark.open(argv[2]);
            if (!benchmark) { throw std::runtime_error("cannot open benchmark CSV"); }
            benchmark << "dimensions,queries,batch,tokens,extreme,gpu_us,max_error\n";
        } else if (argc != 1) {
            throw std::runtime_error("usage: paged_attention_fusion_test [--benchmark CSV]");
        }
        for (unsigned int dimension : {8U, 33U, 64U, 128U, 192U}) {
            run(dimension, 1, {1});
            run(dimension, 1, {65, 9, 64, 33});
            run(dimension, 16, {129});
        }
        run(64, 1, {1024, 512, 129, 65, 64, 33, 9, 1});
        run(64, 8, {129}, true);
        std::cout << "Paged attention dense-reference contracts PASS\n";
        return 0;
    } catch (std::exception const& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
