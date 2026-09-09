#include "runner_internal.cuh"

#include <algorithm>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <vector>

namespace kimkvcache {

CudaModelRunnerCreateResult createCudaTinyLlamaModelRunner(
    std::string const& manifest_path,
    std::string const& data_path,
    EngineKvBackend& kv_backend,
    EngineStream stream,
    CudaModelRunnerOptions options)
{
    using namespace cuda_model_runner_detail;

    CudaModelRunnerCreateResult result;
    if (manifest_path.empty() || data_path.empty() || stream == nullptr
        || !options.valid()
        || options.max_batch_size
            > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        result.status = failure(
            CudaModelRunnerError::InvalidArgument,
            0,
            "manifest, data path, and stream are required"
        );
        return result;
    }
    WeightManifestLoadResult loaded = loadWeightManifest(manifest_path);
    if (!loaded.ok()) {
        result.status = failure(
            CudaModelRunnerError::ManifestError,
            static_cast<int>(loaded.status.error),
            loaded.status.detail
        );
        return result;
    }
    WeightManifestStatus const data_status = validateWeightDataFile(
        loaded.manifest, data_path
    );
    if (!data_status.ok()) {
        result.status = failure(
            CudaModelRunnerError::DataFileError,
            static_cast<int>(data_status.error),
            data_status.detail
        );
        return result;
    }

    TinyLlamaConfig const config = loaded.manifest.config;
    EngineKvConfig const kv_config = kv_backend.config();
    std::size_t const required_attention_workspace =
        static_cast<std::size_t>(config.attention_head_count)
        * config.max_position_embeddings * sizeof(float);
    if (kv_config.kv_layout.layer_count != config.layer_count
        || kv_config.kv_layout.kv_head_count != config.kv_head_count
        || kv_config.kv_layout.head_dimension != config.head_dimension
        || kv_config.query_head_count != config.attention_head_count
        || kv_config.attention_workspace_bytes
            < required_attention_workspace) {
        result.status = failure(
            CudaModelRunnerError::InvalidArgument,
            0,
            "KV backend layout does not match model manifest"
        );
        return result;
    }

    std::unique_ptr<CudaTinyLlamaModelRunner::Impl> impl;
    try {
        impl = std::make_unique<CudaTinyLlamaModelRunner::Impl>();
        impl->manifest = std::move(loaded.manifest);
    } catch (std::bad_alloc const&) {
        result.status = failure(
            CudaModelRunnerError::AllocationFailed,
            0,
            "runner metadata allocation failed"
        );
        return result;
    }
    impl->backend = &kv_backend;
    impl->stream = reinterpret_cast<cudaStream_t>(stream);
    impl->max_batch_size = options.max_batch_size;
    try {
        impl->host_kv_write_batch_items.resize(impl->max_batch_size);
        impl->host_attention_batch_items.resize(impl->max_batch_size);
    } catch (std::bad_alloc const&) {
        result.status = failure(
            CudaModelRunnerError::AllocationFailed,
            0,
            "batch attention host workspace allocation failed"
        );
        return result;
    }
    if (!makeWorkspaceLayout(
            config, impl->max_batch_size, impl->workspace_layout)) {
        result.status = failure(
            CudaModelRunnerError::AllocationFailed,
            0,
            "workspace size overflow"
        );
        return result;
    }

    cudaError_t cuda_status = cudaMalloc(
        reinterpret_cast<void**>(&impl->device_weights),
        static_cast<std::size_t>(impl->manifest.data_bytes)
    );
    if (cuda_status != cudaSuccess) {
        result.status = cudaFailure(cuda_status, "allocate model weights");
        return result;
    }
    cuda_status = cudaMalloc(
        reinterpret_cast<void**>(&impl->device_workspace),
        impl->workspace_layout.bytes
    );
    if (cuda_status != cudaSuccess) {
        result.status = cudaFailure(cuda_status, "allocate model workspace");
        return result;
    }

    std::ifstream weights(data_path, std::ios::binary);
    std::vector<std::uint8_t> staging;
    try {
        staging.resize(16 * 1024 * 1024);
    } catch (std::bad_alloc const&) {
        result.status = failure(
            CudaModelRunnerError::AllocationFailed,
            0,
            "weight staging buffer allocation failed"
        );
        return result;
    }
    for (WeightTensorDescriptor const& tensor : impl->manifest.tensors) {
        weights.clear();
        weights.seekg(static_cast<std::streamoff>(tensor.offset));
        std::uint64_t position = 0;
        while (position < tensor.byte_size) {
            std::size_t const bytes = static_cast<std::size_t>(
                std::min<std::uint64_t>(
                    tensor.byte_size - position, staging.size()
                )
            );
            weights.read(
                reinterpret_cast<char*>(staging.data()),
                static_cast<std::streamsize>(bytes)
            );
            if (weights.gcount() != static_cast<std::streamsize>(bytes)) {
                result.status = failure(
                    CudaModelRunnerError::DataFileError,
                    0,
                    "short read while uploading " + tensor.name
                );
                return result;
            }
            cuda_status = cudaMemcpy(
                impl->device_weights + tensor.offset + position,
                staging.data(),
                bytes,
                cudaMemcpyHostToDevice
            );
            if (cuda_status != cudaSuccess) {
                result.status = cudaFailure(
                    cuda_status, "upload " + tensor.name
                );
                return result;
            }
            position += bytes;
        }
    }

    cublasStatus_t cublas_status = cublasCreate(&impl->cublas);
    if (cublas_status == CUBLAS_STATUS_SUCCESS) {
        cublas_status = cublasSetStream(impl->cublas, impl->stream);
    }
    if (cublas_status == CUBLAS_STATUS_SUCCESS) {
        cublas_status = cublasSetAtomicsMode(
            impl->cublas, CUBLAS_ATOMICS_NOT_ALLOWED
        );
    }
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        result.status = cublasFailure(cublas_status, "initialize cuBLAS");
        return result;
    }

    result.runner = std::unique_ptr<CudaTinyLlamaModelRunner>(
        new CudaTinyLlamaModelRunner(std::move(impl))
    );
    result.status = {};
    return result;
}

} // namespace kimkvcache
