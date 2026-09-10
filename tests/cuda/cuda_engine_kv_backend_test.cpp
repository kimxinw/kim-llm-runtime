#include "kim-kv/cuda/cuda_engine_kv_backend.h"
#include "storage.h"
#include "kim-kv/runtime/kv_cache_manager.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

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

KvScalar fp16(float value)
{
    __half const half = __float2half(value);
    KvScalar bits = 0;
    static_assert(sizeof(bits) == sizeof(half));
    std::memcpy(&bits, &half, sizeof(bits));
    return bits;
}

float fp32(KvScalar bits)
{
    return __half2float(__ushort_as_half(bits));
}

struct DeviceBuffers final {
    KvScalar* key{nullptr};
    KvScalar* value{nullptr};
    KvScalar* query{nullptr};
    KvScalar* output{nullptr};
    void* workspace{nullptr};
    cudaStream_t stream{nullptr};
    bool owns_stream{true};

    explicit DeviceBuffers(cudaStream_t shared_stream = nullptr)
        : stream(shared_stream)
        , owns_stream(shared_stream == nullptr)
    {
        if (owns_stream) {
            expect(cudaStreamCreate(&stream) == cudaSuccess, "create stream");
        }
        expect(cudaMalloc(reinterpret_cast<void**>(&key), 4) == cudaSuccess,
            "allocate key");
        expect(cudaMalloc(reinterpret_cast<void**>(&value), 4) == cudaSuccess,
            "allocate value");
        expect(cudaMalloc(reinterpret_cast<void**>(&query), 8) == cudaSuccess,
            "allocate query");
        expect(cudaMalloc(reinterpret_cast<void**>(&output), 8) == cudaSuccess,
            "allocate output");
        expect(cudaMalloc(&workspace, 4096) == cudaSuccess,
            "allocate workspace");
    }

    ~DeviceBuffers()
    {
        if (owns_stream) {
            static_cast<void>(cudaStreamSynchronize(stream));
        }
        static_cast<void>(cudaFree(workspace));
        static_cast<void>(cudaFree(output));
        static_cast<void>(cudaFree(query));
        static_cast<void>(cudaFree(value));
        static_cast<void>(cudaFree(key));
        if (owns_stream) {
            static_cast<void>(cudaStreamDestroy(stream));
        }
    }
};

enum class FaultMode {
    None,
    Submission,
    Completion,
    PromotionSubmission,
};

std::vector<float> referenceOutput(
    std::vector<int> const& history,
    int new_token)
{
    std::vector<int> tokens = history;
    tokens.push_back(new_token);
    std::vector<float> result(4, 0.0F);
    float const scale = 1.0F / std::sqrt(2.0F);
    float const queries[2][2]{{0.5F, 0.25F}, {-0.2F, 0.4F}};
    for (std::size_t query_head = 0; query_head < 2; ++query_head) {
        std::vector<float> scores;
        for (int token : tokens) {
            float const key[2]{
                0.1F * static_cast<float>(token + 1) + 0.02F,
                -0.05F * static_cast<float>(token) + 0.01F,
            };
            scores.push_back((queries[query_head][0] * key[0]
                + queries[query_head][1] * key[1]) * scale);
        }
        float const maximum = *std::max_element(scores.begin(), scores.end());
        float denominator = 0.0F;
        for (float score : scores) {
            denominator += std::exp(score - maximum);
        }
        for (std::size_t index = 0; index < tokens.size(); ++index) {
            float const probability = std::exp(scores[index] - maximum)
                / denominator;
            float const token = static_cast<float>(tokens[index] + 1);
            result[query_head * 2] += probability * (token + 0.2F);
            result[query_head * 2 + 1] += probability * (token * 0.5F - 0.1F);
        }
    }
    return result;
}

bool runToken(
    EngineKvBackend& backend,
    RequestId request_id,
    std::vector<int>& history,
    int token,
    DeviceBuffers& device,
    FaultMode fault = FaultMode::None)
{
    TokenReserveResult reserved = backend.reserveToken(ReserveTokenRequest{
        request_id,
        static_cast<std::uint32_t>(history.size()),
        reinterpret_cast<EngineStream>(device.stream),
    });
    expect(reserved.ok(), "reserve engine token");
    if (!reserved.ok()) {
        return false;
    }
    expect(
        backend.snapshot().active_transaction_count == 1,
        "reserve increments active transaction count"
    );

    std::vector<KvScalar> const host_query{
        fp16(0.5F), fp16(0.25F), fp16(-0.2F), fp16(0.4F),
    };
    expect(cudaMemcpyAsync(
        device.query,
        host_query.data(),
        host_query.size() * sizeof(KvScalar),
        cudaMemcpyHostToDevice,
        device.stream
    ) == cudaSuccess, "upload query");

    if (fault == FaultMode::Submission) {
        expect(injectCudaEngineFailureOnce(
            backend, CudaFailurePoint::Submission), "inject submission");
    }

    for (std::uint32_t layer = 0; layer < 2; ++layer) {
        float const layer_offset = layer == 1 ? 0.02F : 0.0F;
        std::vector<KvScalar> const host_key{
            fp16(0.1F * static_cast<float>(token + 1) + layer_offset),
            fp16(-0.05F * static_cast<float>(token) + layer_offset * 0.5F),
        };
        std::vector<KvScalar> const host_value{
            fp16(static_cast<float>(token + 1) + layer_offset * 10.0F),
            fp16(static_cast<float>(token + 1) * 0.5F
                - layer_offset * 5.0F),
        };
        expect(cudaMemcpyAsync(
            device.key,
            host_key.data(),
            host_key.size() * sizeof(KvScalar),
            cudaMemcpyHostToDevice,
            device.stream
        ) == cudaSuccess, "upload layer key");
        expect(cudaMemcpyAsync(
            device.value,
            host_value.data(),
            host_value.size() * sizeof(KvScalar),
            cudaMemcpyHostToDevice,
            device.stream
        ) == cudaSuccess, "upload layer value");

        EngineKvStatus const write = reserved.transaction.writeLayer(
            LayerKvWrite{layer, device.key, device.value}
        );
        if (fault == FaultMode::Submission && layer == 0) {
            expect(
                write.error == EngineKvError::SubmissionFailed,
                "submission failure reaches transaction"
            );
            expect(reserved.transaction.rollback().ok(), "fault rollback");
            expect(backend.checkInvariants(), "submission rollback invariants");
            return false;
        }
        expect(write.ok(), "enqueue layer write");
        EngineKvStatus const attention = reserved.transaction.attendLayer(
            PagedDecodeRequest{
                layer,
                device.query,
                device.output,
                device.workspace,
                4096,
                1.0F / std::sqrt(2.0F),
            }
        );
        expect(attention.ok(), "enqueue direct paged attention");
    }

    if (fault == FaultMode::Completion) {
        expect(injectCudaEngineFailureOnce(
            backend, CudaFailurePoint::Completion), "inject completion");
        expect(
            reserved.transaction.commit().error
                == EngineKvError::ExecutionFailed,
            "completion failure rolls back at token boundary"
        );
        expect(backend.checkInvariants(), "completion rollback invariants");
        return false;
    }

    if (fault == FaultMode::PromotionSubmission) {
        expect(
            (history.size() + 1) % kExtentPageTokenCapacity == 0,
            "promotion fault is injected at an Extent boundary"
        );
        expect(injectCudaEngineFailureOnce(
            backend, CudaFailurePoint::Submission),
            "inject promotion submission failure"
        );
    }

    expect(reserved.transaction.commit().ok(), "commit engine token");
    std::vector<KvScalar> host_output(4);
    expect(cudaMemcpy(
        host_output.data(),
        device.output,
        host_output.size() * sizeof(KvScalar),
        cudaMemcpyDeviceToHost
    ) == cudaSuccess, "download attention output");
    std::vector<float> const expected = referenceOutput(history, token);
    for (std::size_t index = 0; index < expected.size(); ++index) {
        float const actual = fp32(host_output[index]);
        expect(
            std::abs(actual - expected[index]) < 0.035F,
            "paged GQA output matches contiguous reference"
        );
    }
    history.push_back(token);
    expect(backend.checkInvariants(), "post-commit engine invariants");
    expect(
        backend.snapshot().active_transaction_count == 0,
        "commit clears active transaction"
    );
    return true;
}

void testAutomaticPromotion()
{
    EngineKvConfig const config{KvLayout{2, 1, 2}, 2, 4096};
    std::unique_ptr<EngineKvBackend> backend =
        createHeterogeneousCudaEngineKvBackend(config, 16, 2);
    expect(backend != nullptr, "create automatic promotion backend");
    expect(backend->createRequest(80).ok(), "create promotion request");

    DeviceBuffers device;
    std::vector<int> history;
    for (int token = 0; token < 64; ++token) {
        expect(runToken(*backend, 80, history, token, device),
            "commit automatic promotion source token");
    }

    EngineKvBackendSnapshot promoted = backend->snapshot();
    expect(promoted.committed_token_count == 64,
        "promotion preserves committed token count");
    expect(promoted.allocated_primary_pages == 0
            && promoted.allocated_secondary_pages == 1,
        "eight Micro pages are atomically replaced by one Extent");
    expect(promoted.automatic_promotion_attempts == 1
            && promoted.automatic_promotion_successes == 1
            && promoted.automatic_promotion_skips == 0
            && promoted.automatic_promotion_failures == 0,
        "snapshot reports successful automatic promotion");

    expect(runToken(*backend, 80, history, 64, device),
        "attention and append continue across Extent plus Micro tail");
    promoted = backend->snapshot();
    expect(promoted.allocated_primary_pages == 1
            && promoted.allocated_secondary_pages == 1,
        "post-promotion append creates a Micro tail");
    expect(backend->releaseRequest(80).ok(),
        "release automatically promoted request");
    EngineKvBackendSnapshot const empty = backend->snapshot();
    expect(empty.allocated_primary_pages == 0
            && empty.allocated_secondary_pages == 0,
        "automatic promotion pages return to both pools");
    expect(backend->checkInvariants(),
        "automatic promotion preserves backend invariants");
}

void testAutomaticPromotionFailureIsBestEffort()
{
    EngineKvConfig const config{KvLayout{2, 1, 2}, 2, 4096};
    std::unique_ptr<EngineKvBackend> backend =
        createHeterogeneousCudaEngineKvBackend(config, 16, 1);
    expect(backend != nullptr, "create promotion failure backend");
    expect(backend->createRequest(90).ok(),
        "create promotion failure request");

    DeviceBuffers device;
    std::vector<int> history;
    for (int token = 0; token < 63; ++token) {
        expect(runToken(*backend, 90, history, token, device),
            "commit promotion failure source token");
    }
    expect(runToken(
        *backend,
        90,
        history,
        63,
        device,
        FaultMode::PromotionSubmission
    ), "promotion submission failure does not fail token commit");

    EngineKvBackendSnapshot const failed = backend->snapshot();
    expect(failed.committed_token_count == 64,
        "failed promotion retains the committed token");
    expect(failed.allocated_primary_pages == 8
            && failed.allocated_secondary_pages == 0,
        "failed promotion rolls back to the original Micro mapping");
    expect(failed.automatic_promotion_attempts == 1
            && failed.automatic_promotion_successes == 0
            && failed.automatic_promotion_failures == 1,
        "snapshot reports the best-effort promotion failure");
    expect(backend->checkInvariants(),
        "failed automatic promotion preserves invariants");
    expect(backend->releaseRequest(90).ok(),
        "release request after automatic promotion failure");
    EngineKvBackendSnapshot const empty = backend->snapshot();
    expect(empty.allocated_primary_pages == 0
            && empty.allocated_secondary_pages == 0,
        "failed promotion leaves no page leak");
}

void testAutomaticPromotionCanBeDisabled()
{
    EngineKvConfig const config{
        KvLayout{2, 1, 2},
        2,
        4096,
        EngineKvPromotionPolicy::Disabled,
    };
    std::unique_ptr<EngineKvBackend> backend =
        createHeterogeneousCudaEngineKvBackend(config, 8, 1);
    expect(backend != nullptr, "create promotion-disabled backend");
    expect(backend->createRequest(100).ok(),
        "create promotion-disabled request");

    DeviceBuffers device;
    std::vector<int> history;
    for (int token = 0; token < 64; ++token) {
        expect(runToken(*backend, 100, history, token, device),
            "commit promotion-disabled source token");
    }
    EngineKvBackendSnapshot const snapshot = backend->snapshot();
    expect(snapshot.allocated_primary_pages == 8
            && snapshot.allocated_secondary_pages == 0,
        "disabled policy retains the Micro layout");
    expect(snapshot.automatic_promotion_attempts == 0,
        "disabled policy performs no promotion attempt");
    expect(backend->releaseRequest(100).ok(),
        "release promotion-disabled request");
    expect(backend->checkInvariants(),
        "disabled policy preserves invariants");
}

void testAutomaticPromotionExtentExhaustion()
{
    EngineKvConfig const config{KvLayout{2, 1, 2}, 2, 4096};
    std::unique_ptr<EngineKvBackend> backend =
        createHeterogeneousCudaEngineKvBackend(config, 16, 1);
    expect(backend != nullptr, "create extent exhaustion backend");
    expect(backend->createRequest(110).ok(), "create extent owner");
    expect(backend->createRequest(111).ok(), "create extent contender");

    DeviceBuffers device;
    std::vector<int> owner_history;
    std::vector<int> contender_history;
    for (int token = 0; token < 64; ++token) {
        expect(runToken(*backend, 110, owner_history, token, device),
            "commit extent owner token");
    }
    for (int token = 0; token < 64; ++token) {
        expect(runToken(*backend, 111, contender_history, token, device),
            "Extent OOM does not fail contender token");
    }

    EngineKvBackendSnapshot const exhausted = backend->snapshot();
    expect(exhausted.committed_token_count == 128,
        "Extent OOM preserves both committed sequences");
    expect(exhausted.allocated_primary_pages == 8
            && exhausted.allocated_secondary_pages == 1,
        "Extent OOM retains contender Micro pages");
    expect(exhausted.automatic_promotion_attempts == 2
            && exhausted.automatic_promotion_successes == 1
            && exhausted.automatic_promotion_failures == 1,
        "snapshot distinguishes successful and exhausted promotions");
    expect(backend->checkInvariants(),
        "Extent OOM preserves backend invariants");
    expect(backend->releaseRequest(111).ok(),
        "release Extent contender");
    expect(backend->releaseRequest(110).ok(), "release Extent owner");
    EngineKvBackendSnapshot const empty = backend->snapshot();
    expect(empty.allocated_primary_pages == 0
            && empty.allocated_secondary_pages == 0,
        "Extent exhaustion test reclaims both pools");
}

void testBackend(
    EngineKvBackendKind kind,
    std::uint16_t fixed_page_tokens = 8)
{
    EngineKvConfig const config{KvLayout{2, 1, 2}, 2, 4096};
    std::unique_ptr<EngineKvBackend> backend =
        kind == EngineKvBackendKind::Heterogeneous
        ? createHeterogeneousCudaEngineKvBackend(config, 32, 4)
        : createFixedCudaEngineKvBackend(config, fixed_page_tokens, 32);
    expect(backend != nullptr, "create CUDA engine backend");
    expect(backend->kind() == kind, "backend kind");
    EngineKvBackendSnapshot const initial = backend->snapshot();
    expect(initial.primary_page_tokens
            == (kind == EngineKvBackendKind::Heterogeneous
                ? kMicroPageTokenCapacity
                : fixed_page_tokens),
        "snapshot exposes primary page size");
    expect(initial.primary_page_capacity == 32,
        "snapshot exposes primary page capacity");
    expect(initial.secondary_page_capacity
            == (kind == EngineKvBackendKind::Heterogeneous ? 4U : 0U),
        "snapshot distinguishes heterogeneous secondary pool");
    expect(initial.storage_reserved_bytes != 0,
        "snapshot exposes reserved CUDA storage bytes");
    expect(backend->createRequest(1).ok(), "create engine request");

    DeviceBuffers device;
    std::vector<int> parent_history;
    for (int token = 0; token < 10; ++token) {
        expect(runToken(
            *backend, 1, parent_history, token, device),
            "multi-page token"
        );
    }
    expect(
        backend->snapshot().committed_token_count == 10,
        "ten committed tokens"
    );
    expect(backend->snapshot().allocated_primary_pages != 0
        && backend->snapshot().successful_primary_allocations != 0,
        "snapshot exposes live and cumulative page allocations");

    expect(backend->forkRequest(1, 2).ok(), "fork engine request");
    std::vector<int> child_history = parent_history;
    expect(runToken(
        *backend, 2, child_history, 21, device),
        "forked partial-tail COW token"
    );
    expect(
        backend->snapshot().committed_token_count == 21,
        "snapshot sums parent and child lengths"
    );

    std::uint64_t const before_fault =
        backend->snapshot().committed_token_count;
    expect(!runToken(
        *backend,
        1,
        parent_history,
        30,
        device,
        FaultMode::Submission), "submission failure is not committed");
    expect(
        backend->snapshot().committed_token_count == before_fault,
        "submission failure preserves length"
    );
    expect(!runToken(
        *backend,
        1,
        parent_history,
        31,
        device,
        FaultMode::Completion), "completion failure is not committed");
    expect(
        backend->snapshot().committed_token_count == before_fault,
        "completion failure preserves length"
    );

    expect(backend->releaseRequest(2).ok(), "release child");
    expect(backend->releaseRequest(1).ok(), "release parent");
    EngineKvBackendSnapshot const empty = backend->snapshot();
    expect(empty.request_count == 0, "all requests released");
    expect(empty.committed_token_count == 0, "released token count cleared");
    expect(empty.allocated_primary_pages == 0
        && empty.allocated_secondary_pages == 0,
        "released backend reports no allocated pages");
    expect(backend->checkInvariants(), "final engine invariants");
}

void testExtentPagedAttention()
{
    KvLayout const layout{1, 1, 2};
    KvCacheManager manager(9, 1);
    CudaKvStorage storage(layout, 9, 1);
    expect(storage.status().ok(), "extent storage initialized");
    expect(manager.createRequest(50) == KvCacheError::None, "extent request");

    std::vector<KvScalar> host_input(256, fp16(0.0F));
    for (std::uint32_t token = 0; token < 64; ++token) {
        host_input[layout.offset(
            0, KvComponent::Key, token, 0, 0, 64)] = fp16(1.0F);
        host_input[layout.offset(
            0, KvComponent::Key, token, 0, 1, 64)] = fp16(1.0F);
        host_input[layout.offset(
            0, KvComponent::Value, token, 0, 0, 64)] =
            fp16(static_cast<float>(token) / 64.0F);
        host_input[layout.offset(
            0, KvComponent::Value, token, 0, 1, 64)] =
            fp16(-static_cast<float>(token) / 128.0F);
    }
    KvScalar* device_input = nullptr;
    expect(cudaMalloc(
        reinterpret_cast<void**>(&device_input),
        host_input.size() * sizeof(KvScalar)) == cudaSuccess,
        "extent input allocation"
    );
    expect(cudaMemcpy(
        device_input,
        host_input.data(),
        host_input.size() * sizeof(KvScalar),
        cudaMemcpyHostToDevice) == cudaSuccess,
        "extent input upload"
    );

    PageLeaseAcquireResult before = manager.acquireRequestReadLease(50);
    expect(before.ok(), "extent before lease");
    expect(manager.append(50, 64) == KvCacheError::None, "append 64 micro tokens");
    PageLeaseAcquireResult after = manager.acquireRequestReadLease(50);
    expect(after.ok(), "extent after lease");
    CudaSubmission append = storage.appendAsync(
        before.table, after.table, 0, 64, device_input
    );
    expect(append.submissionStatus().ok() && append.wait().ok(),
        "upload 64-token KV");
    expect(manager.releasePageLease(after.lease_id) == KvCacheError::None,
        "release after lease");
    expect(manager.releasePageLease(before.lease_id) == KvCacheError::None,
        "release before lease");

    PromotionPrepareResult promotion = manager.preparePromotion(50, 0);
    expect(promotion.ok(), "prepare extent promotion");
    PageLeaseAcquireResult promotion_lease =
        manager.acquirePromotionIoLease(promotion.promotion_id);
    expect(promotion_lease.ok(), "promotion IO lease");
    CudaSubmission promoted = storage.promoteAsync(promotion);
    expect(promoted.submissionStatus().ok() && promoted.wait().ok(),
        "copy micro pages to extent");
    expect(manager.releasePageLease(promotion_lease.lease_id)
            == KvCacheError::None,
        "release promotion lease");
    expect(manager.commitPromotion(promotion.promotion_id)
            == KvCacheError::None,
        "commit extent promotion");
    expect(
        manager.blockTable(50)->entries().size() == 1
            && manager.blockTable(50)->entries().front().kind
                == PageKind::Extent,
        "history is represented by one extent"
    );

    TokenReservationResult reservation = manager.reserveToken(50, 64);
    expect(reservation.ok(), "reserve token after extent");
    PageLeaseAcquireResult reservation_lease =
        manager.acquireTokenReservationLease(reservation.reservation_id);
    expect(reservation_lease.ok(), "extent transaction lease");
    CudaEngineTransactionBeginResult transaction =
        storage.beginEngineTransaction(
            reservation.before, reservation.reserved, 1, nullptr
        );
    expect(transaction.ok(), "begin extent engine transaction");

    KvScalar* key = nullptr;
    KvScalar* value = nullptr;
    KvScalar* query = nullptr;
    KvScalar* output = nullptr;
    void* workspace = nullptr;
    expect(cudaMalloc(reinterpret_cast<void**>(&key), 4) == cudaSuccess,
        "extent key allocation");
    expect(cudaMalloc(reinterpret_cast<void**>(&value), 4) == cudaSuccess,
        "extent value allocation");
    expect(cudaMalloc(reinterpret_cast<void**>(&query), 4) == cudaSuccess,
        "extent query allocation");
    expect(cudaMalloc(reinterpret_cast<void**>(&output), 4) == cudaSuccess,
        "extent output allocation");
    expect(cudaMalloc(&workspace, 65 * sizeof(float)) == cudaSuccess,
        "extent workspace allocation");
    std::vector<KvScalar> const current_key{fp16(1.0F), fp16(1.0F)};
    std::vector<KvScalar> const current_value{fp16(1.0F), fp16(-1.0F)};
    std::vector<KvScalar> const current_query{fp16(1.0F), fp16(1.0F)};
    expect(cudaMemcpy(key, current_key.data(), 4, cudaMemcpyHostToDevice)
            == cudaSuccess,
        "extent key upload");
    expect(cudaMemcpy(value, current_value.data(), 4, cudaMemcpyHostToDevice)
            == cudaSuccess,
        "extent value upload");
    expect(cudaMemcpy(query, current_query.data(), 4, cudaMemcpyHostToDevice)
            == cudaSuccess,
        "extent query upload");
    expect(transaction.transaction->writeLayer(
            LayerKvWrite{0, key, value}).ok(),
        "write token after extent");
    expect(transaction.transaction->attendLayer(PagedDecodeRequest{
            0,
            query,
            output,
            workspace,
            65 * sizeof(float),
            1.0F / std::sqrt(2.0F),
        }).ok(),
        "attend directly across extent and micro page");
    expect(transaction.transaction->finish().ok(), "finish extent attention");
    std::vector<KvScalar> host_output(2);
    expect(cudaMemcpy(
        host_output.data(), output, 4, cudaMemcpyDeviceToHost)
            == cudaSuccess,
        "extent output download");
    expect(std::abs(fp32(host_output[0]) - 0.5F) < 0.01F,
        "extent value dimension zero");
    expect(std::abs(fp32(host_output[1]) + 0.2577F) < 0.01F,
        "extent value dimension one");
    expect(manager.releasePageLease(reservation_lease.lease_id)
            == KvCacheError::None,
        "release extent transaction lease");
    expect(manager.commitTokenReservation(reservation.reservation_id)
            == KvCacheError::None,
        "commit token after extent");
    expect(manager.checkInvariants(), "extent transaction invariants");

    static_cast<void>(cudaFree(workspace));
    static_cast<void>(cudaFree(output));
    static_cast<void>(cudaFree(query));
    static_cast<void>(cudaFree(value));
    static_cast<void>(cudaFree(key));
    static_cast<void>(cudaFree(device_input));
}

void testBatchedWritePreservesCowHistory(EngineKvBackendKind kind)
{
    EngineKvConfig const config{KvLayout{1, 1, 2}, 2, 4096};
    std::unique_ptr<EngineKvBackend> backend =
        kind == EngineKvBackendKind::Heterogeneous
        ? createHeterogeneousCudaEngineKvBackend(config, 16, 2)
        : createFixedCudaEngineKvBackend(config, 8, 16);
    expect(backend != nullptr, "create batched-write COW backend");
    expect(backend->createRequest(90).ok(), "create COW source request");
    expect(backend->createRequest(92).ok(), "create independent request");

    cudaStream_t stream = nullptr;
    expect(cudaStreamCreate(&stream) == cudaSuccess,
        "create shared batched-write COW stream");
    DeviceBuffers source(stream);
    DeviceBuffers child(stream);
    DeviceBuffers independent(stream);
    std::vector<KvScalar> const query{
        fp16(0.5F), fp16(0.25F), fp16(-0.2F), fp16(0.4F),
    };
    auto uploadToken = [&](DeviceBuffers& device, int token) {
        std::vector<KvScalar> const key{
            fp16(0.1F * static_cast<float>(token + 1) + 0.02F),
            fp16(-0.05F * static_cast<float>(token) + 0.01F),
        };
        std::vector<KvScalar> const value{
            fp16(static_cast<float>(token + 1) + 0.2F),
            fp16(static_cast<float>(token + 1) * 0.5F - 0.1F),
        };
        expect(cudaMemcpyAsync(
            device.query, query.data(), 8, cudaMemcpyHostToDevice, stream
        ) == cudaSuccess, "upload batched-write COW query");
        expect(cudaMemcpyAsync(
            device.key, key.data(), 4, cudaMemcpyHostToDevice, stream
        ) == cudaSuccess, "upload batched-write COW key");
        expect(cudaMemcpyAsync(
            device.value, value.data(), 4, cudaMemcpyHostToDevice, stream
        ) == cudaSuccess, "upload batched-write COW value");
    };

    uploadToken(source, 0);
    TokenReserveResult seeded = backend->reserveToken({
        90, 0, reinterpret_cast<EngineStream>(stream),
    });
    expect(seeded.ok(), "reserve COW source token");
    expect(seeded.transaction.writeLayer({0, source.key, source.value}).ok(),
        "write COW source token");
    expect(seeded.transaction.attendLayer({
        0, source.query, source.output, source.workspace, 4096,
        1.0F / std::sqrt(2.0F),
    }).ok(), "attend COW source token");
    expect(seeded.transaction.commit().ok(), "commit COW source token");
    expect(backend->forkRequest(90, 91).ok(), "fork partial-tail child");

    uploadToken(child, 1);
    uploadToken(independent, 2);
    TokenReserveResult child_reserved = backend->reserveToken({
        91, 1, reinterpret_cast<EngineStream>(stream),
    });
    TokenReserveResult independent_reserved = backend->reserveToken({
        92, 0, reinterpret_cast<EngineStream>(stream),
    });
    expect(child_reserved.ok() && independent_reserved.ok(),
        "reserve COW and independent batch lanes");
    std::array<LayerKvWriteBatchItem, 2> items{{
        {&child_reserved.transaction, {0, child.key, child.value}},
        {&independent_reserved.transaction,
            {0, independent.key, independent.value}},
    }};
    std::array<DeviceLayerKvWriteBatchItem, 2> host_items{};
    DeviceLayerKvWriteBatchItem* device_items = nullptr;
    expect(cudaMalloc(
        reinterpret_cast<void**>(&device_items),
        sizeof(DeviceLayerKvWriteBatchItem) * items.size()
    ) == cudaSuccess, "allocate batched-write COW metadata");
    LayerKvWriteBatch batch{
        items.data(), items.size(), host_items.data(), device_items,
        host_items.size(),
    };
    writeLayerBatch(batch);
    expect(items[0].status.ok() && items[1].status.ok(),
        "COW and independent batch lanes both write");
    expect(child_reserved.transaction.attendLayer({
        0, child.query, child.output, child.workspace, 4096,
        1.0F / std::sqrt(2.0F),
    }).ok(), "COW child attends copied history");
    expect(independent_reserved.transaction.attendLayer({
        0, independent.query, independent.output, independent.workspace, 4096,
        1.0F / std::sqrt(2.0F),
    }).ok(), "independent lane attends its own token");
    expect(child_reserved.transaction.commit().ok(), "commit COW child");
    expect(independent_reserved.transaction.commit().ok(),
        "commit independent batch lane");

    auto checkOutput = [&](DeviceBuffers& device,
                           std::vector<int> const& history,
                           int token,
                           std::string const& message) {
        std::array<KvScalar, 4> actual{};
        expect(cudaMemcpy(
            actual.data(), device.output, 8, cudaMemcpyDeviceToHost
        ) == cudaSuccess, "download batched-write COW output");
        std::vector<float> const expected = referenceOutput(history, token);
        for (std::size_t index = 0; index < expected.size(); ++index) {
            expect(std::abs(fp32(actual[index]) - expected[index]) < 0.035F,
                message);
        }
    };
    checkOutput(child, {0}, 1,
        "batched-write COW preserves shared history");
    checkOutput(independent, {}, 2,
        "batched-write independent lane preserves isolation");
    EngineKvBackendSnapshot const snapshot = backend->snapshot();
    expect(snapshot.batched_kv_write_submissions == 1
            && snapshot.batched_kv_write_lanes == 2,
        "COW batch records one two-lane write submission");
    expect(backend->releaseRequest(92).ok(), "release independent request");
    expect(backend->releaseRequest(91).ok(), "release COW child request");
    expect(backend->releaseRequest(90).ok(), "release COW source request");
    expect(backend->checkInvariants(), "batched-write COW invariants");

    static_cast<void>(cudaFree(device_items));
    static_cast<void>(cudaStreamDestroy(stream));
}

void testBatchedWriteFailureIsolation(EngineKvBackendKind kind)
{
    EngineKvConfig const config{KvLayout{1, 1, 2}, 2, 4096};
    std::unique_ptr<EngineKvBackend> backend =
        kind == EngineKvBackendKind::Heterogeneous
        ? createHeterogeneousCudaEngineKvBackend(config, 16, 2)
        : createFixedCudaEngineKvBackend(config, 8, 16);
    expect(backend != nullptr, "create batched-write isolation backend");
    expect(backend->createRequest(80).ok(),
        "create failing batched-write request");
    expect(backend->createRequest(81).ok(),
        "create healthy batched-write request");

    cudaStream_t stream = nullptr;
    expect(cudaStreamCreate(&stream) == cudaSuccess,
        "create shared batched-write stream");
    DeviceBuffers first(stream);
    DeviceBuffers second(stream);
    TokenReserveResult first_reserved = backend->reserveToken({
        80, 0, reinterpret_cast<EngineStream>(stream),
    });
    TokenReserveResult second_reserved = backend->reserveToken({
        81, 0, reinterpret_cast<EngineStream>(stream),
    });
    expect(first_reserved.ok() && second_reserved.ok(),
        "reserve two independent batched-write lanes");

    std::vector<KvScalar> const query{
        fp16(0.5F), fp16(0.25F), fp16(-0.2F), fp16(0.4F),
    };
    std::vector<KvScalar> const first_key{fp16(0.1F), fp16(0.2F)};
    std::vector<KvScalar> const first_value{fp16(1.0F), fp16(2.0F)};
    std::vector<KvScalar> const second_key{fp16(0.3F), fp16(0.4F)};
    std::vector<KvScalar> const second_value{fp16(3.0F), fp16(4.0F)};
    auto upload = [&](DeviceBuffers& device,
                      std::vector<KvScalar> const& key,
                      std::vector<KvScalar> const& value) {
        expect(cudaMemcpyAsync(
            device.query, query.data(), 8, cudaMemcpyHostToDevice, stream
        ) == cudaSuccess, "upload batched-write query");
        expect(cudaMemcpyAsync(
            device.key, key.data(), 4, cudaMemcpyHostToDevice, stream
        ) == cudaSuccess, "upload batched-write key");
        expect(cudaMemcpyAsync(
            device.value, value.data(), 4, cudaMemcpyHostToDevice, stream
        ) == cudaSuccess, "upload batched-write value");
    };
    upload(first, first_key, first_value);
    upload(second, second_key, second_value);

    expect(injectCudaEngineFailureOnce(
        *backend, CudaFailurePoint::Submission),
        "inject one batched-write submission failure");
    std::array<LayerKvWriteBatchItem, 2> items{{
        {&first_reserved.transaction, {0, first.key, first.value}},
        {&second_reserved.transaction, {0, second.key, second.value}},
    }};
    std::array<DeviceLayerKvWriteBatchItem, 2> host_items{};
    DeviceLayerKvWriteBatchItem* device_items = nullptr;
    expect(cudaMalloc(
        reinterpret_cast<void**>(&device_items),
        sizeof(DeviceLayerKvWriteBatchItem) * items.size()
    ) == cudaSuccess, "allocate batched-write lane metadata");
    LayerKvWriteBatch batch{
        items.data(),
        items.size(),
        host_items.data(),
        device_items,
        host_items.size(),
    };
    writeLayerBatch(batch);
    expect(items[0].status.error == EngineKvError::SubmissionFailed,
        "injected batched-write lane fails closed");
    expect(items[1].status.ok(),
        "healthy batched-write lane survives another lane failure");
    expect(first_reserved.transaction.rollback().ok(),
        "failed batched-write lane rolls back");
    expect(second_reserved.transaction.attendLayer({
        0,
        second.query,
        second.output,
        second.workspace,
        4096,
        1.0F / std::sqrt(2.0F),
    }).ok(), "healthy batched-write lane attends");
    expect(second_reserved.transaction.commit().ok(),
        "healthy batched-write lane commits");

    std::array<KvScalar, 4> output{};
    expect(cudaMemcpy(
        output.data(), second.output, 8, cudaMemcpyDeviceToHost
    ) == cudaSuccess, "download healthy batched-write output");
    expect(std::abs(fp32(output[0]) - 3.0F) < 0.01F
            && std::abs(fp32(output[1]) - 4.0F) < 0.01F
            && std::abs(fp32(output[2]) - 3.0F) < 0.01F
            && std::abs(fp32(output[3]) - 4.0F) < 0.01F,
        "healthy batched-write lane stores independent K/V");
    EngineKvBackendSnapshot const snapshot = backend->snapshot();
    expect(snapshot.committed_token_count == 1
            && snapshot.active_transaction_count == 0,
        "batched-write failure publishes only the healthy lane");
    expect(snapshot.batched_kv_write_submissions == 1
            && snapshot.batched_kv_write_lanes == 1,
        "batched-write telemetry records the surviving lane");
    expect(backend->releaseRequest(81).ok(),
        "release healthy batched-write request");
    expect(backend->releaseRequest(80).ok(),
        "release failed batched-write request");
    expect(backend->checkInvariants(),
        "batched-write failure isolation invariants");

    static_cast<void>(cudaFree(device_items));
    static_cast<void>(cudaStreamDestroy(stream));
}

void testBatchedAttentionFailureIsolation(EngineKvBackendKind kind)
{
    EngineKvConfig const config{KvLayout{1, 1, 2}, 2, 4096};
    std::unique_ptr<EngineKvBackend> backend =
        kind == EngineKvBackendKind::Heterogeneous
        ? createHeterogeneousCudaEngineKvBackend(config, 16, 2)
        : createFixedCudaEngineKvBackend(config, 8, 16);
    expect(backend != nullptr, "create batch isolation backend");
    expect(backend->createRequest(70).ok(), "create failing batch request");
    expect(backend->createRequest(71).ok(), "create healthy batch request");

    cudaStream_t stream = nullptr;
    expect(cudaStreamCreate(&stream) == cudaSuccess,
        "create shared batch stream");
    DeviceBuffers first(stream);
    DeviceBuffers second(stream);
    TokenReserveResult first_reserved = backend->reserveToken({
        70, 0, reinterpret_cast<EngineStream>(stream),
    });
    TokenReserveResult second_reserved = backend->reserveToken({
        71, 0, reinterpret_cast<EngineStream>(stream),
    });
    expect(first_reserved.ok() && second_reserved.ok(),
        "reserve two independent batch lanes");

    std::vector<KvScalar> const query{
        fp16(0.5F), fp16(0.25F), fp16(-0.2F), fp16(0.4F),
    };
    std::vector<KvScalar> const first_key{fp16(0.1F), fp16(0.2F)};
    std::vector<KvScalar> const first_value{fp16(1.0F), fp16(2.0F)};
    std::vector<KvScalar> const second_key{fp16(0.3F), fp16(0.4F)};
    std::vector<KvScalar> const second_value{fp16(3.0F), fp16(4.0F)};
    auto upload = [&](DeviceBuffers& device,
                      std::vector<KvScalar> const& key,
                      std::vector<KvScalar> const& value) {
        expect(cudaMemcpyAsync(
            device.query, query.data(), 8, cudaMemcpyHostToDevice, stream
        ) == cudaSuccess, "upload batch query");
        expect(cudaMemcpyAsync(
            device.key, key.data(), 4, cudaMemcpyHostToDevice, stream
        ) == cudaSuccess, "upload batch key");
        expect(cudaMemcpyAsync(
            device.value, value.data(), 4, cudaMemcpyHostToDevice, stream
        ) == cudaSuccess, "upload batch value");
    };
    upload(first, first_key, first_value);
    upload(second, second_key, second_value);
    expect(first_reserved.transaction.writeLayer(
            {0, first.key, first.value}).ok(),
        "write failing batch lane before attention");
    expect(second_reserved.transaction.writeLayer(
            {0, second.key, second.value}).ok(),
        "write healthy batch lane before attention");

    expect(injectCudaEngineFailureOnce(
        *backend, CudaFailurePoint::Submission),
        "inject one batch attention submission failure");
    std::array<PagedDecodeBatchItem, 2> items{{
        {&first_reserved.transaction,
            {0, first.query, first.output, first.workspace, 4096,
                1.0F / std::sqrt(2.0F)}},
        {&second_reserved.transaction,
            {0, second.query, second.output, second.workspace, 4096,
                1.0F / std::sqrt(2.0F)}},
    }};
    std::array<DevicePagedDecodeBatchItem, 2> host_items{};
    DevicePagedDecodeBatchItem* device_items = nullptr;
    expect(cudaMalloc(
        reinterpret_cast<void**>(&device_items),
        sizeof(DevicePagedDecodeBatchItem) * items.size()
    ) == cudaSuccess, "allocate batch lane metadata");
    PagedDecodeBatch batch{
        items.data(),
        items.size(),
        host_items.data(),
        device_items,
        host_items.size(),
    };
    attendLayerBatch(batch);
    expect(items[0].status.error == EngineKvError::SubmissionFailed,
        "injected batch lane fails closed");
    expect(items[1].status.ok(),
        "healthy batch lane survives another lane failure");
    expect(first_reserved.transaction.rollback().ok(),
        "failed batch lane rolls back");
    expect(second_reserved.transaction.commit().ok(),
        "healthy batch lane commits");

    std::array<KvScalar, 4> output{};
    expect(cudaMemcpy(
        output.data(), second.output, 8, cudaMemcpyDeviceToHost
    ) == cudaSuccess, "download healthy batch output");
    expect(std::abs(fp32(output[0]) - 3.0F) < 0.01F
            && std::abs(fp32(output[1]) - 4.0F) < 0.01F
            && std::abs(fp32(output[2]) - 3.0F) < 0.01F
            && std::abs(fp32(output[3]) - 4.0F) < 0.01F,
        "healthy batch lane computes its independent GQA output");
    EngineKvBackendSnapshot const snapshot = backend->snapshot();
    expect(snapshot.committed_token_count == 1
            && snapshot.active_transaction_count == 0,
        "batch failure publishes only the healthy lane");
    expect(snapshot.batched_attention_submissions == 1
            && snapshot.batched_attention_lanes == 1,
        "batch telemetry records the surviving submitted lane");
    expect(backend->releaseRequest(71).ok(), "release healthy batch request");
    expect(backend->releaseRequest(70).ok(), "release failed batch request");
    expect(backend->checkInvariants(), "batch failure isolation invariants");

    static_cast<void>(cudaFree(device_items));
    static_cast<void>(cudaStreamDestroy(stream));
}

void testMultiTokenCausalPrefill(EngineKvBackendKind kind)
{
    constexpr std::uint32_t kTokenCount = 10;
    EngineKvConfig const config{KvLayout{1, 1, 2}, 2, 4096};
    std::unique_ptr<EngineKvBackend> backend =
        kind == EngineKvBackendKind::Heterogeneous
        ? createHeterogeneousCudaEngineKvBackend(config, 16, 2)
        : createFixedCudaEngineKvBackend(config, 8, 16);
    expect(backend != nullptr, "create multi-token prefill backend");
    expect(backend->createRequest(120).ok(),
        "create multi-token prefill request");

    cudaStream_t stream = nullptr;
    KvScalar* key = nullptr;
    KvScalar* value = nullptr;
    KvScalar* query = nullptr;
    KvScalar* output = nullptr;
    float* scores = nullptr;
    expect(cudaStreamCreate(&stream) == cudaSuccess,
        "create multi-token prefill stream");
    std::size_t const kv_elements = kTokenCount * 2;
    std::size_t const query_elements = kTokenCount * 4;
    std::size_t const score_elements = kTokenCount * 2 * kTokenCount;
    expect(cudaMalloc(reinterpret_cast<void**>(&key),
            kv_elements * sizeof(KvScalar)) == cudaSuccess,
        "allocate multi-token keys");
    expect(cudaMalloc(reinterpret_cast<void**>(&value),
            kv_elements * sizeof(KvScalar)) == cudaSuccess,
        "allocate multi-token values");
    expect(cudaMalloc(reinterpret_cast<void**>(&query),
            query_elements * sizeof(KvScalar)) == cudaSuccess,
        "allocate multi-token queries");
    expect(cudaMalloc(reinterpret_cast<void**>(&output),
            query_elements * sizeof(KvScalar)) == cudaSuccess,
        "allocate multi-token output");
    expect(cudaMalloc(reinterpret_cast<void**>(&scores),
            score_elements * sizeof(float)) == cudaSuccess,
        "allocate multi-token score workspace");

    std::vector<KvScalar> host_key(kv_elements, fp16(0.0F));
    std::vector<KvScalar> host_value(kv_elements);
    std::vector<KvScalar> host_query(query_elements, fp16(1.0F));
    for (std::uint32_t token = 0; token < kTokenCount; ++token) {
        host_value[token * 2] = fp16(static_cast<float>(token + 1));
        host_value[token * 2 + 1] = fp16(
            static_cast<float>((token + 1) * 2)
        );
    }
    expect(cudaMemcpyAsync(key, host_key.data(),
            kv_elements * sizeof(KvScalar), cudaMemcpyHostToDevice, stream)
            == cudaSuccess,
        "upload multi-token keys");
    expect(cudaMemcpyAsync(value, host_value.data(),
            kv_elements * sizeof(KvScalar), cudaMemcpyHostToDevice, stream)
            == cudaSuccess,
        "upload multi-token values");
    expect(cudaMemcpyAsync(query, host_query.data(),
            query_elements * sizeof(KvScalar), cudaMemcpyHostToDevice, stream)
            == cudaSuccess,
        "upload multi-token queries");

    TokenReserveResult reserved = backend->reserveToken({
        120, 0, reinterpret_cast<EngineStream>(stream), kTokenCount,
    });
    expect(reserved.ok()
            && reserved.transaction.snapshot().token_count == kTokenCount,
        "reserve one ten-token engine transaction");
    expect(reserved.transaction.writeLayer({
            0, key, value, kTokenCount}).ok(),
        "write ten-token KV segment across page boundary");
    expect(reserved.transaction.attendLayer({
            0,
            query,
            output,
            scores,
            score_elements * sizeof(float),
            1.0F / std::sqrt(2.0F),
            kTokenCount,
        }).ok(), "run causal paged prefill attention");
    expect(reserved.transaction.commit().ok(),
        "commit ten-token prefill atomically");

    std::vector<KvScalar> host_output(query_elements);
    expect(cudaMemcpy(host_output.data(), output,
            query_elements * sizeof(KvScalar), cudaMemcpyDeviceToHost)
            == cudaSuccess,
        "download multi-token prefill output");
    for (std::uint32_t token = 0; token < kTokenCount; ++token) {
        float const expected_first = static_cast<float>(token + 2) / 2.0F;
        for (std::uint32_t head = 0; head < 2; ++head) {
            std::size_t const base = (token * 2 + head) * 2;
            expect(std::abs(fp32(host_output[base]) - expected_first) < 0.02F
                    && std::abs(fp32(host_output[base + 1])
                        - expected_first * 2.0F) < 0.03F,
                "each prefill query sees only its causal prefix");
        }
    }
    expect(backend->snapshot().committed_token_count == kTokenCount,
        "multi-token commit updates backend length once");
    expect(backend->forkRequest(120, 121).ok(),
        "fork multi-token history before partial-tail COW");
    constexpr std::uint32_t kCowTokens = 3;
    std::vector<KvScalar> cow_value(kCowTokens * 2);
    for (std::uint32_t token = 0; token < kCowTokens; ++token) {
        cow_value[token * 2] = fp16(100.0F);
        cow_value[token * 2 + 1] = fp16(200.0F);
    }
    expect(cudaMemcpyAsync(value, cow_value.data(),
            cow_value.size() * sizeof(KvScalar),
            cudaMemcpyHostToDevice, stream) == cudaSuccess,
        "upload chunk COW values");
    TokenReserveResult cow = backend->reserveToken({
        121, kTokenCount, reinterpret_cast<EngineStream>(stream), kCowTokens,
    });
    expect(cow.ok(), "reserve chunk on shared partial tail");
    expect(cow.transaction.writeLayer({0, key, value, kCowTokens}).ok(),
        "write chunk through partial-tail COW");
    expect(cow.transaction.attendLayer({
            0,
            query,
            output,
            scores,
            score_elements * sizeof(float),
            1.0F / std::sqrt(2.0F),
            kCowTokens,
        }).ok(), "attend over copied history and new COW chunk");
    expect(cow.transaction.commit().ok(), "commit chunk COW atomically");
    std::vector<KvScalar> cow_output(kCowTokens * 4);
    expect(cudaMemcpy(cow_output.data(), output,
            cow_output.size() * sizeof(KvScalar), cudaMemcpyDeviceToHost)
            == cudaSuccess,
        "download chunk COW attention output");
    for (std::uint32_t token = 0; token < kCowTokens; ++token) {
        float const expected_first = (55.0F + 100.0F * (token + 1))
            / static_cast<float>(11 + token);
        for (std::uint32_t head = 0; head < 2; ++head) {
            std::size_t const base = (token * 2 + head) * 2;
            expect(std::abs(fp32(cow_output[base]) - expected_first) < 0.04F
                    && std::abs(fp32(cow_output[base + 1])
                        - expected_first * 2.0F) < 0.08F,
                "chunk COW preserves history for causal attention");
        }
    }
    expect(backend->releaseRequest(121).ok(),
        "release chunk COW child request");
    expect(backend->releaseRequest(120).ok(),
        "release multi-token prefill request");
    expect(backend->checkInvariants(),
        "multi-token prefill preserves backend invariants");

    static_cast<void>(cudaFree(scores));
    static_cast<void>(cudaFree(output));
    static_cast<void>(cudaFree(query));
    static_cast<void>(cudaFree(value));
    static_cast<void>(cudaFree(key));
    static_cast<void>(cudaStreamDestroy(stream));
}

} // namespace

int main()
{
    testBackend(EngineKvBackendKind::Heterogeneous);
    for (std::uint16_t const page_tokens : {8, 16, 32, 64}) {
        testBackend(EngineKvBackendKind::Fixed, page_tokens);
    }
    testExtentPagedAttention();
    testAutomaticPromotion();
    testAutomaticPromotionFailureIsBestEffort();
    testAutomaticPromotionCanBeDisabled();
    testAutomaticPromotionExtentExhaustion();
    testBatchedWritePreservesCowHistory(
        EngineKvBackendKind::Heterogeneous
    );
    testBatchedWritePreservesCowHistory(EngineKvBackendKind::Fixed);
    testBatchedWriteFailureIsolation(EngineKvBackendKind::Heterogeneous);
    testBatchedWriteFailureIsolation(EngineKvBackendKind::Fixed);
    testBatchedAttentionFailureIsolation(
        EngineKvBackendKind::Heterogeneous
    );
    testBatchedAttentionFailureIsolation(EngineKvBackendKind::Fixed);
    testMultiTokenCausalPrefill(EngineKvBackendKind::Heterogeneous);
    testMultiTokenCausalPrefill(EngineKvBackendKind::Fixed);
    if (failures != 0) {
        std::cerr << failures << " CUDA engine KV checks failed\n";
        return 1;
    }
    std::cout << "CUDA engine KV checks passed\n";
    return 0;
}
