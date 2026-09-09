#include "kernels.cuh"
#include "runner_internal.cuh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kimkvcache {

using namespace cuda_model_runner_detail;

CudaTinyLlamaModelRunner::CudaTinyLlamaModelRunner(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl))
{
}

CudaTinyLlamaModelRunner::~CudaTinyLlamaModelRunner() = default;
CudaTinyLlamaModelRunner::CudaTinyLlamaModelRunner(
    CudaTinyLlamaModelRunner&&) noexcept = default;
CudaTinyLlamaModelRunner& CudaTinyLlamaModelRunner::operator=(
    CudaTinyLlamaModelRunner&&) noexcept = default;

TinyLlamaConfig CudaTinyLlamaModelRunner::config() const noexcept
{
    return impl_ != nullptr ? impl_->manifest.config : TinyLlamaConfig{};
}

std::uint64_t CudaTinyLlamaModelRunner::deviceWeightBytes() const noexcept
{
    return impl_ != nullptr ? impl_->manifest.data_bytes : 0;
}

std::uint64_t CudaTinyLlamaModelRunner::deviceWorkspaceBytes() const noexcept
{
    return impl_ != nullptr ? impl_->workspace_layout.bytes : 0;
}

std::uint32_t CudaTinyLlamaModelRunner::maxBatchSize() const noexcept
{
    return impl_ != nullptr ? impl_->max_batch_size : 0;
}

ModelTokenResult CudaTinyLlamaModelRunner::forwardToken(
    RequestId request_id,
    std::uint32_t token_id,
    std::uint32_t expected_position,
    ModelRunnerDebugCapture* debug_capture,
    ModelRunnerOutputOptions output_options)
{
    ModelTokenResult result;
    if (impl_ == nullptr || impl_->backend == nullptr) {
        result.status = failure(
            CudaModelRunnerError::InvalidState, 0, "runner is empty"
        );
        return result;
    }
    TinyLlamaConfig const& config = impl_->manifest.config;
    if (request_id == kInvalidRequestId
        || token_id >= config.vocabulary_size
        || expected_position >= config.max_position_embeddings) {
        result.status = failure(
            CudaModelRunnerError::InvalidArgument,
            0,
            "request, token, or position is out of range"
        );
        return result;
    }

    std::unordered_map<std::uint32_t, std::size_t> capture_indices;
    try {
        if (output_options.copy_logits) {
            result.logits.resize(config.vocabulary_size);
        }
        if (debug_capture != nullptr) {
            debug_capture->embedding.resize(config.hidden_size);
            debug_capture->final_hidden_state.resize(config.hidden_size);
            debug_capture->layer_hidden_states.clear();
            debug_capture->layer_hidden_states.resize(
                debug_capture->layer_indices.size(),
                std::vector<KvScalar>(config.hidden_size)
            );
            for (std::size_t index = 0;
                 index < debug_capture->layer_indices.size();
                 ++index) {
                std::uint32_t const layer =
                    debug_capture->layer_indices[index];
                if (layer >= config.layer_count
                    || !capture_indices.emplace(layer, index).second) {
                    result.status = failure(
                        CudaModelRunnerError::InvalidArgument,
                        0,
                        "debug layer is duplicated or out of range"
                    );
                    return result;
                }
            }
        }
    } catch (std::bad_alloc const&) {
        result.status = failure(
            CudaModelRunnerError::AllocationFailed,
            0,
            "host output allocation failed"
        );
        return result;
    }

    TokenReserveResult reserved = impl_->backend->reserveToken(
        ReserveTokenRequest{
            request_id,
            expected_position,
            reinterpret_cast<EngineStream>(impl_->stream),
        }
    );
    if (!reserved.ok()) {
        result.status = kvFailure(reserved.status, "reserve token");
        return result;
    }

    WorkspaceLayout const& offsets = impl_->workspace_layout;
    KvScalar* const hidden = impl_->workspace<KvScalar>(offsets.hidden);
    KvScalar* const normalized =
        impl_->workspace<KvScalar>(offsets.normalized);
    KvScalar* const query = impl_->workspace<KvScalar>(offsets.query);
    KvScalar* const key = impl_->workspace<KvScalar>(offsets.key);
    KvScalar* const value = impl_->workspace<KvScalar>(offsets.value);
    KvScalar* const attention =
        impl_->workspace<KvScalar>(offsets.attention);
    KvScalar* const projection =
        impl_->workspace<KvScalar>(offsets.projection);
    KvScalar* const gate = impl_->workspace<KvScalar>(offsets.gate);
    KvScalar* const up = impl_->workspace<KvScalar>(offsets.up);
    KvScalar* const activated =
        impl_->workspace<KvScalar>(offsets.activated);
    float* const logits = impl_->workspace<float>(offsets.logits);
    std::uint32_t* const greedy_token =
        impl_->workspace<std::uint32_t>(offsets.greedy_token);
    float* const attention_scores =
        impl_->workspace<float>(offsets.attention_scores);

    cuda_model_detail::launchEmbedding(
        impl_->weight("model.embed_tokens.weight"),
        token_id,
        hidden,
        config.hidden_size,
        impl_->stream
    );
    result.status = impl_->checkLaunch("embedding lookup");
    if (!result.status.ok()) {
        return result;
    }
    if (debug_capture != nullptr) {
        cudaError_t const copied = cudaMemcpyAsync(
            debug_capture->embedding.data(),
            hidden,
            config.hidden_size * sizeof(KvScalar),
            cudaMemcpyDeviceToHost,
            impl_->stream
        );
        if (copied != cudaSuccess) {
            result.status = cudaFailure(copied, "capture embedding");
            return result;
        }
    }

    std::uint32_t const kv_size =
        config.kv_head_count * config.head_dimension;
    float const attention_scale =
        1.0F / std::sqrt(static_cast<float>(config.head_dimension));
    std::size_t const score_bytes =
        static_cast<std::size_t>(config.attention_head_count)
        * config.max_position_embeddings * sizeof(float);
    for (std::uint32_t layer = 0; layer < config.layer_count; ++layer) {
        std::string const prefix = "model.layers."
            + std::to_string(layer) + ".";
        cuda_model_detail::launchRmsNorm(
            hidden,
            impl_->weight(prefix + "input_layernorm.weight"),
            normalized,
            config.hidden_size,
            config.rms_norm_epsilon,
            impl_->stream
        );
        result.status = impl_->checkLaunch("input rmsnorm");
        if (!result.status.ok()) {
            return result;
        }
        result.status = impl_->gemv(
            impl_->weight(prefix + "self_attn.q_proj.weight"),
            normalized, query, config.hidden_size, config.hidden_size,
            "q projection"
        );
        if (!result.status.ok()) {
            return result;
        }
        result.status = impl_->gemv(
            impl_->weight(prefix + "self_attn.k_proj.weight"),
            normalized, key, kv_size, config.hidden_size, "k projection"
        );
        if (!result.status.ok()) {
            return result;
        }
        result.status = impl_->gemv(
            impl_->weight(prefix + "self_attn.v_proj.weight"),
            normalized, value, kv_size, config.hidden_size, "v projection"
        );
        if (!result.status.ok()) {
            return result;
        }
        cuda_model_detail::launchRope(
            query,
            key,
            config.attention_head_count,
            config.kv_head_count,
            config.head_dimension,
            expected_position,
            config.rope_theta,
            impl_->stream
        );
        result.status = impl_->checkLaunch("rope");
        if (!result.status.ok()) {
            return result;
        }

        EngineKvStatus const write = reserved.transaction.writeLayer(
            LayerKvWrite{layer, key, value}
        );
        if (!write.ok()) {
            result.status = kvFailure(write, "write layer kv");
            return result;
        }
        EngineKvStatus const attended = reserved.transaction.attendLayer(
            PagedDecodeRequest{
                layer,
                query,
                attention,
                attention_scores,
                score_bytes,
                attention_scale,
            }
        );
        if (!attended.ok()) {
            result.status = kvFailure(attended, "paged decode attention");
            return result;
        }
        result.status = impl_->gemv(
            impl_->weight(prefix + "self_attn.o_proj.weight"),
            attention, projection, config.hidden_size, config.hidden_size,
            "attention output projection"
        );
        if (!result.status.ok()) {
            return result;
        }
        cuda_model_detail::launchResidualAdd(
            hidden, projection, hidden, config.hidden_size, impl_->stream
        );
        result.status = impl_->checkLaunch("attention residual");
        if (!result.status.ok()) {
            return result;
        }
        cuda_model_detail::launchRmsNorm(
            hidden,
            impl_->weight(prefix + "post_attention_layernorm.weight"),
            normalized,
            config.hidden_size,
            config.rms_norm_epsilon,
            impl_->stream
        );
        result.status = impl_->checkLaunch("post attention rmsnorm");
        if (!result.status.ok()) {
            return result;
        }
        result.status = impl_->gemv(
            impl_->weight(prefix + "mlp.gate_proj.weight"),
            normalized, gate, config.intermediate_size, config.hidden_size,
            "mlp gate projection"
        );
        if (!result.status.ok()) {
            return result;
        }
        result.status = impl_->gemv(
            impl_->weight(prefix + "mlp.up_proj.weight"),
            normalized, up, config.intermediate_size, config.hidden_size,
            "mlp up projection"
        );
        if (!result.status.ok()) {
            return result;
        }
        cuda_model_detail::launchSwiGlu(
            gate, up, activated, config.intermediate_size, impl_->stream
        );
        result.status = impl_->checkLaunch("swiglu");
        if (!result.status.ok()) {
            return result;
        }
        result.status = impl_->gemv(
            impl_->weight(prefix + "mlp.down_proj.weight"),
            activated, projection, config.hidden_size,
            config.intermediate_size, "mlp down projection"
        );
        if (!result.status.ok()) {
            return result;
        }
        cuda_model_detail::launchResidualAdd(
            hidden, projection, hidden, config.hidden_size, impl_->stream
        );
        result.status = impl_->checkLaunch("mlp residual");
        if (!result.status.ok()) {
            return result;
        }

        auto const capture = capture_indices.find(layer);
        if (capture != capture_indices.end()) {
            cudaError_t const copied = cudaMemcpyAsync(
                debug_capture->layer_hidden_states[capture->second].data(),
                hidden,
                config.hidden_size * sizeof(KvScalar),
                cudaMemcpyDeviceToHost,
                impl_->stream
            );
            if (copied != cudaSuccess) {
                result.status = cudaFailure(copied, "capture layer hidden");
                return result;
            }
        }
    }

    cuda_model_detail::launchRmsNorm(
        hidden,
        impl_->weight("model.norm.weight"),
        normalized,
        config.hidden_size,
        config.rms_norm_epsilon,
        impl_->stream
    );
    result.status = impl_->checkLaunch("final rmsnorm");
    if (!result.status.ok()) {
        return result;
    }
    result.status = impl_->logitsGemv(
        impl_->weight("lm_head.weight"), normalized, logits
    );
    if (!result.status.ok()) {
        return result;
    }
    cuda_model_detail::launchArgmax(
        logits, config.vocabulary_size, greedy_token, impl_->stream
    );
    result.status = impl_->checkLaunch("greedy argmax");
    if (!result.status.ok()) {
        return result;
    }

    cudaError_t copied = cudaSuccess;
    if (output_options.copy_logits) {
        copied = cudaMemcpyAsync(
            result.logits.data(),
            logits,
            result.logits.size() * sizeof(float),
            cudaMemcpyDeviceToHost,
            impl_->stream
        );
    }
    if (copied == cudaSuccess) {
        copied = cudaMemcpyAsync(
            &result.greedy_token_id,
            greedy_token,
            sizeof(result.greedy_token_id),
            cudaMemcpyDeviceToHost,
            impl_->stream
        );
    }
    if (copied == cudaSuccess && debug_capture != nullptr) {
        copied = cudaMemcpyAsync(
            debug_capture->final_hidden_state.data(),
            normalized,
            config.hidden_size * sizeof(KvScalar),
            cudaMemcpyDeviceToHost,
            impl_->stream
        );
    }
    if (copied != cudaSuccess) {
        result.status = cudaFailure(copied, "copy model output");
        return result;
    }

    EngineKvStatus const committed = reserved.transaction.commit();
    if (!committed.ok()) {
        result.status = kvFailure(committed, "commit model token");
        return result;
    }
    result.status = {};
    return result;
}

TinyLlamaConfig CudaTinyLlamaModelRunner::generationConfig() const noexcept
{
    return config();
}

GenerationBatchResult CudaTinyLlamaModelRunner::generationForwardBatch(
    std::vector<GenerationBatchItem> const& batch)
{
    GenerationBatchResult result;
    if (impl_ == nullptr || impl_->backend == nullptr) {
        result.detail = "runner is empty";
        return result;
    }
    if (batch.empty() || batch.size() > impl_->max_batch_size) {
        result.detail = "generation batch is empty or exceeds max batch size";
        return result;
    }

    TinyLlamaConfig const& config = impl_->manifest.config;
    std::vector<std::size_t> active_indices;
    std::vector<std::uint32_t> token_ids;
    std::vector<std::uint32_t> positions;
    std::vector<TokenTransaction> transactions;
    std::vector<LayerKvWriteBatchItem> kv_write_batch_items;
    std::vector<PagedDecodeBatchItem> attention_batch_items;
    try {
        result.steps.resize(batch.size());
        kv_write_batch_items.resize(batch.size());
        attention_batch_items.resize(batch.size());
        active_indices.reserve(batch.size());
        token_ids.reserve(batch.size());
        positions.reserve(batch.size());
        transactions.reserve(batch.size());
    } catch (std::bad_alloc const&) {
        result.steps.clear();
        result.detail = "batch host workspace allocation failed";
        return result;
    }

    for (std::size_t index = 0; index < batch.size(); ++index) {
        GenerationBatchItem const& item = batch[index];
        if (item.request_id == kInvalidRequestId
            || item.token_id >= config.vocabulary_size
            || item.expected_position >= config.max_position_embeddings) {
            result.steps[index] = {
                false, 0, "request, token, or position is out of range",
            };
            continue;
        }
        TokenReserveResult reserved = impl_->backend->reserveToken(
            ReserveTokenRequest{
                item.request_id,
                item.expected_position,
                reinterpret_cast<EngineStream>(impl_->stream),
            }
        );
        if (!reserved.ok()) {
            CudaModelRunnerStatus const status = kvFailure(
                reserved.status, "reserve token"
            );
            result.steps[index] = {false, 0, status.detail};
            continue;
        }
        try {
            active_indices.push_back(index);
            token_ids.push_back(item.token_id);
            positions.push_back(item.expected_position);
            transactions.push_back(std::move(reserved.transaction));
        } catch (std::bad_alloc const&) {
            result.steps[index] = {
                false, 0, "batch transaction allocation failed",
            };
        }
    }
    result.success = true;
    if (active_indices.empty()) {
        return result;
    }

    std::uint32_t const batch_size = static_cast<std::uint32_t>(
        active_indices.size()
    );
    std::vector<bool> active(batch_size, true);
    std::vector<std::uint32_t> greedy_tokens;
    try {
        greedy_tokens.resize(batch_size);
    } catch (std::bad_alloc const&) {
        for (std::size_t index : active_indices) {
            result.steps[index] = {
                false, 0, "batch output allocation failed",
            };
        }
        return result;
    }

    auto failActive = [&](std::string const& detail) {
        for (std::uint32_t lane = 0; lane < batch_size; ++lane) {
            if (active[lane]) {
                result.steps[active_indices[lane]] = {false, 0, detail};
                active[lane] = false;
            }
        }
    };
    auto checkLaunch = [&](std::string const& operation) {
        CudaModelRunnerStatus const status = impl_->checkLaunch(operation);
        if (!status.ok()) {
            failActive(status.detail);
            return false;
        }
        return true;
    };
    auto checkDense = [&](CudaModelRunnerStatus const& status) {
        if (!status.ok()) {
            failActive(status.detail);
            return false;
        }
        return true;
    };

    WorkspaceLayout const& offsets = impl_->workspace_layout;
    std::uint32_t* const device_token_ids =
        impl_->workspace<std::uint32_t>(offsets.token_ids);
    std::uint32_t* const device_positions =
        impl_->workspace<std::uint32_t>(offsets.positions);
    KvScalar* const hidden = impl_->workspace<KvScalar>(offsets.hidden);
    KvScalar* const normalized =
        impl_->workspace<KvScalar>(offsets.normalized);
    KvScalar* const query = impl_->workspace<KvScalar>(offsets.query);
    KvScalar* const key = impl_->workspace<KvScalar>(offsets.key);
    KvScalar* const value = impl_->workspace<KvScalar>(offsets.value);
    KvScalar* const attention =
        impl_->workspace<KvScalar>(offsets.attention);
    KvScalar* const projection =
        impl_->workspace<KvScalar>(offsets.projection);
    KvScalar* const gate = impl_->workspace<KvScalar>(offsets.gate);
    KvScalar* const up = impl_->workspace<KvScalar>(offsets.up);
    KvScalar* const activated =
        impl_->workspace<KvScalar>(offsets.activated);
    float* const logits = impl_->workspace<float>(offsets.logits);
    std::uint32_t* const device_greedy_tokens =
        impl_->workspace<std::uint32_t>(offsets.greedy_token);
    float* const attention_scores =
        impl_->workspace<float>(offsets.attention_scores);
    DeviceLayerKvWriteBatchItem* const device_kv_write_batch_items =
        impl_->workspace<DeviceLayerKvWriteBatchItem>(
            offsets.kv_write_batch_items
        );
    DevicePagedDecodeBatchItem* const device_attention_batch_items =
        impl_->workspace<DevicePagedDecodeBatchItem>(
            offsets.attention_batch_items
        );

    std::size_t const index_bytes = static_cast<std::size_t>(batch_size)
        * sizeof(std::uint32_t);
    cudaError_t copied = cudaMemcpyAsync(
        device_token_ids,
        token_ids.data(),
        index_bytes,
        cudaMemcpyHostToDevice,
        impl_->stream
    );
    if (copied == cudaSuccess) {
        copied = cudaMemcpyAsync(
            device_positions,
            positions.data(),
            index_bytes,
            cudaMemcpyHostToDevice,
            impl_->stream
        );
    }
    if (copied != cudaSuccess) {
        failActive(cudaFailure(copied, "copy batch inputs").detail);
        return result;
    }

    cuda_model_detail::launchEmbeddingBatch(
        impl_->weight("model.embed_tokens.weight"),
        device_token_ids,
        hidden,
        config.hidden_size,
        batch_size,
        impl_->stream
    );
    if (!checkLaunch("batched embedding lookup")) {
        return result;
    }

    std::uint32_t const kv_size =
        config.kv_head_count * config.head_dimension;
    float const attention_scale =
        1.0F / std::sqrt(static_cast<float>(config.head_dimension));
    std::size_t const score_bytes_per_lane =
        static_cast<std::size_t>(config.attention_head_count)
        * config.max_position_embeddings * sizeof(float);
    for (std::uint32_t layer = 0; layer < config.layer_count; ++layer) {
        std::string const prefix = "model.layers."
            + std::to_string(layer) + ".";
        cuda_model_detail::launchRmsNormBatch(
            hidden,
            impl_->weight(prefix + "input_layernorm.weight"),
            normalized,
            config.hidden_size,
            batch_size,
            config.rms_norm_epsilon,
            impl_->stream
        );
        if (!checkLaunch("batched input rmsnorm")
            || !checkDense(impl_->gemmBatch(
                impl_->weight(prefix + "self_attn.q_proj.weight"),
                normalized, query, config.hidden_size, config.hidden_size,
                batch_size, "batched q projection"
            ))
            || !checkDense(impl_->gemmBatch(
                impl_->weight(prefix + "self_attn.k_proj.weight"),
                normalized, key, kv_size, config.hidden_size,
                batch_size, "batched k projection"
            ))
            || !checkDense(impl_->gemmBatch(
                impl_->weight(prefix + "self_attn.v_proj.weight"),
                normalized, value, kv_size, config.hidden_size,
                batch_size, "batched v projection"
            ))) {
            return result;
        }
        cuda_model_detail::launchRopeBatch(
            query,
            key,
            config.attention_head_count,
            config.kv_head_count,
            config.head_dimension,
            device_positions,
            batch_size,
            config.rope_theta,
            impl_->stream
        );
        if (!checkLaunch("batched rope")) {
            return result;
        }

        auto kvWrite = [&](std::uint32_t lane) {
            return LayerKvWrite{
                layer,
                key + static_cast<std::size_t>(lane) * kv_size,
                value + static_cast<std::size_t>(lane) * kv_size,
            };
        };
        std::size_t const write_lane_count = static_cast<std::size_t>(
            std::count(active.begin(), active.end(), true)
        );
        if (write_lane_count == 1) {
            for (std::uint32_t lane = 0; lane < batch_size; ++lane) {
                if (!active[lane]) {
                    continue;
                }
                EngineKvStatus const write =
                    transactions[lane].writeLayer(kvWrite(lane));
                if (!write.ok()) {
                    result.steps[active_indices[lane]] = {
                        false, 0,
                        kvFailure(write, "single-lane layer kv write").detail,
                    };
                    active[lane] = false;
                }
            }
        } else {
            for (std::uint32_t lane = 0; lane < batch_size; ++lane) {
                kv_write_batch_items[lane] = {};
                if (active[lane]) {
                    kv_write_batch_items[lane].transaction =
                        &transactions[lane];
                    kv_write_batch_items[lane].write = kvWrite(lane);
                }
            }
            LayerKvWriteBatch write_batch{
                kv_write_batch_items.data(),
                batch_size,
                impl_->host_kv_write_batch_items.data(),
                device_kv_write_batch_items,
                impl_->host_kv_write_batch_items.size(),
            };
            writeLayerBatch(write_batch);
            for (std::uint32_t lane = 0; lane < batch_size; ++lane) {
                if (!active[lane] || kv_write_batch_items[lane].status.ok()) {
                    continue;
                }
                result.steps[active_indices[lane]] = {
                    false,
                    0,
                    kvFailure(
                        kv_write_batch_items[lane].status,
                        "batched layer kv write"
                    ).detail,
                };
                active[lane] = false;
            }
        }
        std::size_t const attention_lane_count = static_cast<std::size_t>(
            std::count(active.begin(), active.end(), true)
        );
        if (attention_lane_count == 0) {
            return result;
        }

        auto attentionRequest = [&](std::uint32_t lane) {
            return PagedDecodeRequest{
                layer,
                query + static_cast<std::size_t>(lane)
                    * config.hidden_size,
                attention + static_cast<std::size_t>(lane)
                    * config.hidden_size,
                attention_scores + static_cast<std::size_t>(lane)
                    * config.attention_head_count
                    * config.max_position_embeddings,
                score_bytes_per_lane,
                attention_scale,
            };
        };
        if (attention_lane_count == 1) {
            for (std::uint32_t lane = 0; lane < batch_size; ++lane) {
                if (!active[lane]) {
                    continue;
                }
                EngineKvStatus const attended =
                    transactions[lane].attendLayer(attentionRequest(lane));
                if (!attended.ok()) {
                    result.steps[active_indices[lane]] = {
                        false,
                        0,
                        kvFailure(
                            attended, "single-lane paged decode attention"
                        ).detail,
                    };
                    active[lane] = false;
                }
            }
        } else {
            for (std::uint32_t lane = 0; lane < batch_size; ++lane) {
                attention_batch_items[lane] = {};
                if (active[lane]) {
                    attention_batch_items[lane].transaction =
                        &transactions[lane];
                    attention_batch_items[lane].request =
                        attentionRequest(lane);
                }
            }
            PagedDecodeBatch attention_batch{
                attention_batch_items.data(),
                batch_size,
                impl_->host_attention_batch_items.data(),
                device_attention_batch_items,
                impl_->host_attention_batch_items.size(),
            };
            attendLayerBatch(attention_batch);
            for (std::uint32_t lane = 0; lane < batch_size; ++lane) {
                if (!active[lane]
                    || attention_batch_items[lane].status.ok()) {
                    continue;
                }
                result.steps[active_indices[lane]] = {
                    false,
                    0,
                    kvFailure(
                        attention_batch_items[lane].status,
                        "batched paged decode attention"
                    ).detail,
                };
                active[lane] = false;
            }
        }
        if (std::none_of(active.begin(), active.end(), [](bool value) {
                return value;
            })) {
            return result;
        }

        if (!checkDense(impl_->gemmBatch(
                impl_->weight(prefix + "self_attn.o_proj.weight"),
                attention, projection, config.hidden_size, config.hidden_size,
                batch_size, "batched attention output projection"
            ))) {
            return result;
        }
        cuda_model_detail::launchResidualAddBatch(
            hidden, projection, hidden, config.hidden_size,
            batch_size, impl_->stream
        );
        if (!checkLaunch("batched attention residual")) {
            return result;
        }
        cuda_model_detail::launchRmsNormBatch(
            hidden,
            impl_->weight(prefix + "post_attention_layernorm.weight"),
            normalized,
            config.hidden_size,
            batch_size,
            config.rms_norm_epsilon,
            impl_->stream
        );
        if (!checkLaunch("batched post attention rmsnorm")
            || !checkDense(impl_->gemmBatch(
                impl_->weight(prefix + "mlp.gate_proj.weight"),
                normalized, gate, config.intermediate_size,
                config.hidden_size, batch_size, "batched mlp gate projection"
            ))
            || !checkDense(impl_->gemmBatch(
                impl_->weight(prefix + "mlp.up_proj.weight"),
                normalized, up, config.intermediate_size,
                config.hidden_size, batch_size, "batched mlp up projection"
            ))) {
            return result;
        }
        cuda_model_detail::launchSwiGluBatch(
            gate, up, activated, config.intermediate_size,
            batch_size, impl_->stream
        );
        if (!checkLaunch("batched swiglu")
            || !checkDense(impl_->gemmBatch(
                impl_->weight(prefix + "mlp.down_proj.weight"),
                activated, projection, config.hidden_size,
                config.intermediate_size, batch_size,
                "batched mlp down projection"
            ))) {
            return result;
        }
        cuda_model_detail::launchResidualAddBatch(
            hidden, projection, hidden, config.hidden_size,
            batch_size, impl_->stream
        );
        if (!checkLaunch("batched mlp residual")) {
            return result;
        }
    }

    cuda_model_detail::launchRmsNormBatch(
        hidden,
        impl_->weight("model.norm.weight"),
        normalized,
        config.hidden_size,
        batch_size,
        config.rms_norm_epsilon,
        impl_->stream
    );
    if (!checkLaunch("batched final rmsnorm")
        || !checkDense(impl_->logitsGemmBatch(
            impl_->weight("lm_head.weight"), normalized, logits, batch_size
        ))) {
        return result;
    }
    cuda_model_detail::launchArgmaxBatch(
        logits,
        config.vocabulary_size,
        device_greedy_tokens,
        batch_size,
        impl_->stream
    );
    if (!checkLaunch("batched greedy argmax")) {
        return result;
    }
    copied = cudaMemcpyAsync(
        greedy_tokens.data(),
        device_greedy_tokens,
        index_bytes,
        cudaMemcpyDeviceToHost,
        impl_->stream
    );
    if (copied != cudaSuccess) {
        failActive(cudaFailure(copied, "copy batched model output").detail);
        return result;
    }

    for (std::uint32_t lane = 0; lane < batch_size; ++lane) {
        if (!active[lane]) {
            continue;
        }
        EngineKvStatus const committed = transactions[lane].commit();
        if (!committed.ok()) {
            result.steps[active_indices[lane]] = {
                false, 0,
                kvFailure(committed, "commit batched model token").detail,
            };
            active[lane] = false;
            continue;
        }
        result.steps[active_indices[lane]] = {
            true, greedy_tokens[lane], {},
        };
    }
    return result;
}

std::uint32_t CudaTinyLlamaModelRunner::generationMaxBatchSize() const noexcept
{
    return maxBatchSize();
}

GenerationStepResult CudaTinyLlamaModelRunner::generationForwardToken(
    RequestId request_id,
    std::uint32_t token_id,
    std::uint32_t expected_position)
{
    try {
        GenerationBatchResult result = generationForwardBatch({
            GenerationBatchItem{request_id, token_id, expected_position},
        });
        if (!result.success || result.steps.size() != 1) {
            return {
                false,
                0,
                result.detail.empty()
                    ? "single-token batch returned an invalid result"
                    : std::move(result.detail),
            };
        }
        return std::move(result.steps.front());
    } catch (std::bad_alloc const&) {
        return {false, 0, "single-token batch allocation failed"};
    }
}


} // namespace kimkvcache
