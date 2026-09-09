#include "kim-kv/cuda/cuda_engine_kv_backend.h"
#include "kim-kv/cuda/cuda_model_runner.h"
#include "kim-kv/engine/iteration_scheduler.h"
#include "kim-kv/model/weight_manifest.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
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

float rounded(float value)
{
    return __half2float(__float2half(value));
}

KvScalar fp16(float value)
{
    __half const half = __float2half(value);
    KvScalar bits = 0;
    std::memcpy(&bits, &half, sizeof(bits));
    return bits;
}

float fp32(KvScalar value)
{
    return __half2float(__ushort_as_half(value));
}

struct TestArchive final {
    TinyLlamaConfig config{
        8, 16, 2, 4, 2, 2, 16, 256, 1, 2, 1.0e-5F, 10000.0F, false,
    };
    std::filesystem::path root{};
    std::filesystem::path manifest_path{};
    std::filesystem::path data_path{};
    std::unordered_map<std::string, std::vector<float>> weights{};
    WeightManifest manifest{};

    explicit TestArchive(std::filesystem::path value)
        : root(std::move(value))
        , manifest_path(root / "model.manifest")
        , data_path(root / "model.weights")
    {
        std::filesystem::remove_all(root);
        std::filesystem::create_directories(root);
        build();
    }

    ~TestArchive()
    {
        std::filesystem::remove_all(root);
    }

    void addTensor(
        std::ofstream& output,
        std::string name,
        std::vector<std::uint32_t> shape,
        float scale,
        bool norm = false)
    {
        std::size_t elements = 1;
        for (std::uint32_t dimension : shape) {
            elements *= dimension;
        }
        std::size_t phase = 1;
        for (char character : name) {
            phase = (phase * 33 + static_cast<unsigned char>(character)) % 97;
        }
        std::vector<float> values(elements);
        std::vector<KvScalar> encoded(elements);
        for (std::size_t index = 0; index < elements; ++index) {
            float const value = norm
                ? 1.0F + 0.025F * std::sin(
                    static_cast<float>(index + phase) * 0.31F)
                : scale * std::sin(
                    static_cast<float>((index + 1) * (phase + 3)) * 0.071F
                );
            values[index] = rounded(value);
            encoded[index] = fp16(value);
        }
        std::uint64_t const offset = static_cast<std::uint64_t>(output.tellp());
        output.write(
            reinterpret_cast<char const*>(encoded.data()),
            static_cast<std::streamsize>(encoded.size() * sizeof(KvScalar))
        );
        manifest.tensors.push_back(WeightTensorDescriptor{
            name,
            offset,
            encoded.size() * sizeof(KvScalar),
            std::move(shape),
            {},
        });
        weights.emplace(std::move(name), std::move(values));
    }

    void writeManifest() const
    {
        std::ofstream output(manifest_path);
        output << "version=1\n"
            << "checkpoint=contract/tinyllama\n"
            << "checkpoint_revision=0123456789abcdef\n"
            << "tokenizer_revision=0123456789abcdef\n"
            << "config_sha256=" << std::string(64, 'b') << '\n'
            << "dtype=fp16\n"
            << "data_file=" << data_path.filename().string() << '\n'
            << "data_bytes=" << manifest.data_bytes << '\n'
            << "hidden_size=" << config.hidden_size << '\n'
            << "intermediate_size=" << config.intermediate_size << '\n'
            << "layer_count=" << config.layer_count << '\n'
            << "attention_head_count=" << config.attention_head_count << '\n'
            << "kv_head_count=" << config.kv_head_count << '\n'
            << "head_dimension=" << config.head_dimension << '\n'
            << "vocabulary_size=" << config.vocabulary_size << '\n'
            << "max_position_embeddings="
            << config.max_position_embeddings << '\n'
            << "bos_token_id=" << config.bos_token_id << '\n'
            << "eos_token_id=" << config.eos_token_id << '\n'
            << "rms_norm_epsilon=" << config.rms_norm_epsilon << '\n'
            << "rope_theta=" << config.rope_theta << '\n'
            << "tied_word_embeddings=0\n"
            << "tensor_count=" << manifest.tensors.size() << '\n';
        for (WeightTensorDescriptor const& tensor : manifest.tensors) {
            output << "tensor=" << tensor.name << '|' << tensor.offset
                << '|' << tensor.byte_size << '|';
            for (std::size_t index = 0; index < tensor.shape.size(); ++index) {
                if (index != 0) {
                    output << ',';
                }
                output << tensor.shape[index];
            }
            output << '|' << tensor.sha256 << '\n';
        }
    }

    void build()
    {
        manifest.version = 1;
        manifest.checkpoint = "contract/tinyllama";
        manifest.checkpoint_revision = "0123456789abcdef";
        manifest.tokenizer_revision = "0123456789abcdef";
        manifest.config_sha256 = std::string(64, 'b');
        manifest.data_type = "fp16";
        manifest.data_file = data_path.filename().string();
        manifest.config = config;
        std::ofstream output(data_path, std::ios::binary);
        addTensor(output, "model.embed_tokens.weight",
            {config.vocabulary_size, config.hidden_size}, 0.35F);
        addTensor(output, "model.norm.weight", {config.hidden_size}, 0.0F, true);
        addTensor(output, "lm_head.weight",
            {config.vocabulary_size, config.hidden_size}, 0.45F);
        std::uint32_t const kv_size = config.kv_head_count
            * config.head_dimension;
        for (std::uint32_t layer = 0; layer < config.layer_count; ++layer) {
            std::string const prefix = "model.layers."
                + std::to_string(layer) + ".";
            addTensor(output, prefix + "input_layernorm.weight",
                {config.hidden_size}, 0.0F, true);
            addTensor(output, prefix + "post_attention_layernorm.weight",
                {config.hidden_size}, 0.0F, true);
            addTensor(output, prefix + "self_attn.q_proj.weight",
                {config.hidden_size, config.hidden_size}, 0.12F);
            addTensor(output, prefix + "self_attn.k_proj.weight",
                {kv_size, config.hidden_size}, 0.12F);
            addTensor(output, prefix + "self_attn.v_proj.weight",
                {kv_size, config.hidden_size}, 0.12F);
            addTensor(output, prefix + "self_attn.o_proj.weight",
                {config.hidden_size, config.hidden_size}, 0.10F);
            addTensor(output, prefix + "mlp.gate_proj.weight",
                {config.intermediate_size, config.hidden_size}, 0.11F);
            addTensor(output, prefix + "mlp.up_proj.weight",
                {config.intermediate_size, config.hidden_size}, 0.11F);
            addTensor(output, prefix + "mlp.down_proj.weight",
                {config.hidden_size, config.intermediate_size}, 0.08F);
        }
        output.close();
        manifest.data_bytes = std::filesystem::file_size(data_path);
        for (WeightTensorDescriptor& tensor : manifest.tensors) {
            expect(computeWeightDataSha256(
                data_path.string(), tensor.offset, tensor.byte_size,
                tensor.sha256
            ).ok(), "compute model test checksum");
        }
        writeManifest();
    }
};

std::vector<float> matvec(
    std::vector<float> const& matrix,
    std::vector<float> const& input,
    std::size_t rows,
    std::size_t columns,
    bool round_output = true)
{
    std::vector<float> output(rows, 0.0F);
    for (std::size_t row = 0; row < rows; ++row) {
        float sum = 0.0F;
        for (std::size_t column = 0; column < columns; ++column) {
            sum += matrix[row * columns + column] * input[column];
        }
        output[row] = round_output ? rounded(sum) : sum;
    }
    return output;
}

std::vector<float> rmsNorm(
    std::vector<float> const& input,
    std::vector<float> const& weight,
    float epsilon)
{
    float square_sum = 0.0F;
    for (float value : input) {
        square_sum += value * value;
    }
    float const scale = 1.0F / std::sqrt(
        square_sum / static_cast<float>(input.size()) + epsilon
    );
    std::vector<float> output(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        output[index] = rounded(input[index] * scale * weight[index]);
    }
    return output;
}

void applyRope(
    std::vector<float>& values,
    std::uint32_t head_count,
    std::uint32_t head_dimension,
    std::uint32_t position,
    float theta)
{
    std::uint32_t const half = head_dimension / 2;
    for (std::uint32_t head = 0; head < head_count; ++head) {
        for (std::uint32_t dimension = 0; dimension < half; ++dimension) {
            std::size_t const first = head * head_dimension + dimension;
            std::size_t const second = first + half;
            float const angle = static_cast<float>(position) * std::pow(
                theta,
                -2.0F * static_cast<float>(dimension)
                    / static_cast<float>(head_dimension)
            );
            float const left = values[first];
            float const right = values[second];
            values[first] = rounded(
                left * std::cos(angle) - right * std::sin(angle)
            );
            values[second] = rounded(
                right * std::cos(angle) + left * std::sin(angle)
            );
        }
    }
}

struct ReferenceModel final {
    TestArchive const& archive;
    std::vector<std::vector<std::vector<float>>> keys;
    std::vector<std::vector<std::vector<float>>> values;
    std::vector<float> final_hidden{};

    explicit ReferenceModel(TestArchive const& value)
        : archive(value)
        , keys(value.config.layer_count)
        , values(value.config.layer_count)
    {
    }

    std::vector<float> forward(std::uint32_t token, std::uint32_t position)
    {
        TinyLlamaConfig const& config = archive.config;
        auto const& embedding = archive.weights.at("model.embed_tokens.weight");
        std::vector<float> hidden(
            embedding.begin() + token * config.hidden_size,
            embedding.begin() + (token + 1) * config.hidden_size
        );
        std::uint32_t const kv_size = config.kv_head_count
            * config.head_dimension;
        std::uint32_t const group_size = config.attention_head_count
            / config.kv_head_count;
        for (std::uint32_t layer = 0; layer < config.layer_count; ++layer) {
            std::string const prefix = "model.layers."
                + std::to_string(layer) + ".";
            std::vector<float> normalized = rmsNorm(
                hidden,
                archive.weights.at(prefix + "input_layernorm.weight"),
                config.rms_norm_epsilon
            );
            std::vector<float> query = matvec(
                archive.weights.at(prefix + "self_attn.q_proj.weight"),
                normalized, config.hidden_size, config.hidden_size
            );
            std::vector<float> key = matvec(
                archive.weights.at(prefix + "self_attn.k_proj.weight"),
                normalized, kv_size, config.hidden_size
            );
            std::vector<float> value = matvec(
                archive.weights.at(prefix + "self_attn.v_proj.weight"),
                normalized, kv_size, config.hidden_size
            );
            applyRope(query, config.attention_head_count,
                config.head_dimension, position, config.rope_theta);
            applyRope(key, config.kv_head_count,
                config.head_dimension, position, config.rope_theta);
            keys[layer].push_back(key);
            values[layer].push_back(value);

            std::vector<float> attention(config.hidden_size, 0.0F);
            float const scale = 1.0F / std::sqrt(
                static_cast<float>(config.head_dimension)
            );
            for (std::uint32_t query_head = 0;
                 query_head < config.attention_head_count;
                 ++query_head) {
                std::uint32_t const kv_head = query_head / group_size;
                std::vector<float> scores(keys[layer].size(), 0.0F);
                for (std::size_t past = 0; past < keys[layer].size(); ++past) {
                    for (std::uint32_t dimension = 0;
                         dimension < config.head_dimension;
                         ++dimension) {
                        scores[past] += query[
                            query_head * config.head_dimension + dimension
                        ] * keys[layer][past][
                            kv_head * config.head_dimension + dimension
                        ];
                    }
                    scores[past] *= scale;
                }
                float const maximum = *std::max_element(
                    scores.begin(), scores.end()
                );
                float denominator = 0.0F;
                for (float score : scores) {
                    denominator += std::exp(score - maximum);
                }
                for (std::uint32_t dimension = 0;
                     dimension < config.head_dimension;
                     ++dimension) {
                    float output = 0.0F;
                    for (std::size_t past = 0; past < scores.size(); ++past) {
                        output += std::exp(scores[past] - maximum)
                            / denominator * values[layer][past][
                                kv_head * config.head_dimension + dimension
                            ];
                    }
                    attention[query_head * config.head_dimension + dimension]
                        = rounded(output);
                }
            }
            std::vector<float> projected = matvec(
                archive.weights.at(prefix + "self_attn.o_proj.weight"),
                attention, config.hidden_size, config.hidden_size
            );
            for (std::size_t index = 0; index < hidden.size(); ++index) {
                hidden[index] = rounded(hidden[index] + projected[index]);
            }
            normalized = rmsNorm(
                hidden,
                archive.weights.at(
                    prefix + "post_attention_layernorm.weight"),
                config.rms_norm_epsilon
            );
            std::vector<float> gate = matvec(
                archive.weights.at(prefix + "mlp.gate_proj.weight"),
                normalized, config.intermediate_size, config.hidden_size
            );
            std::vector<float> up = matvec(
                archive.weights.at(prefix + "mlp.up_proj.weight"),
                normalized, config.intermediate_size, config.hidden_size
            );
            for (std::size_t index = 0; index < gate.size(); ++index) {
                gate[index] = rounded(
                    gate[index] / (1.0F + std::exp(-gate[index])) * up[index]
                );
            }
            projected = matvec(
                archive.weights.at(prefix + "mlp.down_proj.weight"),
                gate, config.hidden_size, config.intermediate_size
            );
            for (std::size_t index = 0; index < hidden.size(); ++index) {
                hidden[index] = rounded(hidden[index] + projected[index]);
            }
        }
        final_hidden = rmsNorm(
            hidden,
            archive.weights.at("model.norm.weight"),
            config.rms_norm_epsilon
        );
        return matvec(
            archive.weights.at("lm_head.weight"),
            final_hidden,
            config.vocabulary_size,
            config.hidden_size,
            false
        );
    }
};

std::vector<std::uint32_t> referenceGeneration(
    TestArchive const& archive,
    std::vector<std::uint32_t> const& prompt,
    std::uint32_t max_new_tokens)
{
    ReferenceModel reference(archive);
    std::uint32_t next_token = 0;
    for (std::uint32_t position = 0; position < prompt.size(); ++position) {
        std::vector<float> const logits = reference.forward(
            prompt[position], position
        );
        next_token = static_cast<std::uint32_t>(
            std::max_element(logits.begin(), logits.end()) - logits.begin()
        );
    }

    std::vector<std::uint32_t> output;
    output.reserve(max_new_tokens);
    for (std::uint32_t generated = 0;
         generated < max_new_tokens;
         ++generated) {
        output.push_back(next_token);
        if (generated + 1 != max_new_tokens) {
            std::uint32_t const position = static_cast<std::uint32_t>(
                prompt.size()
            ) + generated;
            std::vector<float> const logits = reference.forward(
                next_token, position
            );
            next_token = static_cast<std::uint32_t>(
                std::max_element(logits.begin(), logits.end()) - logits.begin()
            );
        }
    }
    return output;
}

std::uint32_t eosOutside(
    std::vector<std::uint32_t> const& tokens,
    std::uint32_t vocabulary_size)
{
    for (std::uint32_t candidate = 0;
         candidate < vocabulary_size;
         ++candidate) {
        if (std::find(tokens.begin(), tokens.end(), candidate) == tokens.end()) {
            return candidate;
        }
    }
    return vocabulary_size;
}

void testModelRunner()
{
    TestArchive archive(
        std::filesystem::temp_directory_path() / "kim_kv_model_runner_contract"
    );
    cudaStream_t stream = nullptr;
    expect(cudaStreamCreate(&stream) == cudaSuccess, "create model stream");
    std::size_t const attention_workspace =
        archive.config.attention_head_count
        * archive.config.max_position_embeddings * sizeof(float);
    EngineKvConfig const kv_config{
        KvLayout{
            archive.config.layer_count,
            archive.config.kv_head_count,
            archive.config.head_dimension,
        },
        archive.config.attention_head_count,
        attention_workspace,
    };
    std::unique_ptr<EngineKvBackend> backend =
        createHeterogeneousCudaEngineKvBackend(kv_config, 32, 4);
    expect(backend != nullptr, "create model KV backend");
    expect(backend->createRequest(41).ok(), "create model request");

    CudaModelRunnerCreateResult created = createCudaTinyLlamaModelRunner(
        archive.manifest_path.string(),
        archive.data_path.string(),
        *backend,
        reinterpret_cast<EngineStream>(stream)
    );
    expect(created.ok(), std::string("create model runner: ")
        + created.status.detail);
    if (!created.ok()) {
        static_cast<void>(cudaStreamDestroy(stream));
        return;
    }
    expect(created.runner->deviceWeightBytes() == archive.manifest.data_bytes,
        "runner reports exact weight bytes");
    expect(created.runner->deviceWorkspaceBytes() > 0,
        "runner owns preallocated workspace");

    ReferenceModel reference(archive);
    std::vector<std::uint32_t> const tokens{1, 7};
    for (std::size_t position = 0; position < tokens.size(); ++position) {
        ModelRunnerDebugCapture debug;
        debug.layer_indices = {0, 1};
        ModelTokenResult actual = created.runner->forwardToken(
            41, tokens[position], static_cast<std::uint32_t>(position), &debug
        );
        expect(actual.ok(), std::string("forward model token: ")
            + actual.status.detail);
        std::vector<float> const expected = reference.forward(
            tokens[position], static_cast<std::uint32_t>(position)
        );
        expect(actual.logits.size() == expected.size(), "logit shape matches");
        float maximum_error = 0.0F;
        for (std::size_t index = 0;
             index < std::min(actual.logits.size(), expected.size());
             ++index) {
            maximum_error = std::max(
                maximum_error, std::abs(actual.logits[index] - expected[index])
            );
        }
        expect(maximum_error < 0.035F, "logits match independent FP16 reference");
        std::uint32_t const expected_token = static_cast<std::uint32_t>(
            std::max_element(expected.begin(), expected.end()) - expected.begin()
        );
        expect(actual.greedy_token_id == expected_token,
            "greedy token matches reference");
        expect(debug.embedding.size() == archive.config.hidden_size,
            "embedding capture is populated");
        expect(debug.layer_hidden_states.size() == 2,
            "selected layer captures are populated");
        float hidden_error = 0.0F;
        for (std::size_t index = 0; index < reference.final_hidden.size(); ++index) {
            hidden_error = std::max(hidden_error, std::abs(
                fp32(debug.final_hidden_state[index])
                    - reference.final_hidden[index]
            ));
        }
        expect(hidden_error < 0.02F, "final hidden matches reference");
    }
    expect(backend->snapshot().committed_token_count == tokens.size(),
        "model tokens commit through paged KV backend");
    expect(backend->snapshot().active_transaction_count == 0,
        "no model transaction remains active");
    expect(backend->checkInvariants(), "model backend invariants hold");

    std::uint64_t const before_fault =
        backend->snapshot().committed_token_count;
    expect(injectCudaEngineFailureOnce(
        *backend, CudaFailurePoint::Submission),
        "inject model submission failure");
    ModelTokenResult const submission_failed = created.runner->forwardToken(
        41, 3, static_cast<std::uint32_t>(tokens.size())
    );
    expect(submission_failed.status.error
        == CudaModelRunnerError::SubmissionFailed,
        "model submission failure reaches caller");
    expect(backend->snapshot().committed_token_count == before_fault
        && backend->snapshot().active_transaction_count == 0,
        "model submission failure rolls back KV");

    expect(injectCudaEngineFailureOnce(
        *backend, CudaFailurePoint::Completion),
        "inject model completion failure");
    ModelTokenResult const completion_failed = created.runner->forwardToken(
        41, 3, static_cast<std::uint32_t>(tokens.size())
    );
    expect(completion_failed.status.error
        == CudaModelRunnerError::ExecutionFailed,
        "model completion failure reaches caller");
    expect(backend->snapshot().committed_token_count == before_fault
        && backend->snapshot().active_transaction_count == 0,
        "model completion failure rolls back KV");
    expect(backend->checkInvariants(), "model fault rollback invariants hold");

    ModelRunnerDebugCapture invalid_debug;
    invalid_debug.layer_indices = {0, 0};
    ModelTokenResult const invalid = created.runner->forwardToken(
        41, 3, 2, &invalid_debug
    );
    expect(invalid.status.error == CudaModelRunnerError::InvalidArgument,
        "invalid debug capture is rejected before reserve");
    expect(backend->snapshot().committed_token_count == tokens.size(),
        "invalid model call does not publish KV");

    expect(backend->releaseRequest(41).ok(), "release model request");
    expect(backend->snapshot().request_count == 0, "model request reclaimed");
    created.runner.reset();
    expect(cudaStreamDestroy(stream) == cudaSuccess, "destroy model stream");
}

void testGenerationRuntime()
{
    TestArchive archive(
        std::filesystem::temp_directory_path()
            / "kim_kv_generation_runtime_cuda_contract"
    );
    cudaStream_t stream = nullptr;
    expect(cudaStreamCreate(&stream) == cudaSuccess,
        "create generation stream");
    std::size_t const attention_workspace =
        archive.config.attention_head_count
        * archive.config.max_position_embeddings * sizeof(float);
    EngineKvConfig const kv_config{
        KvLayout{
            archive.config.layer_count,
            archive.config.kv_head_count,
            archive.config.head_dimension,
        },
        archive.config.attention_head_count,
        attention_workspace,
    };
    std::unique_ptr<EngineKvBackend> backend =
        createHeterogeneousCudaEngineKvBackend(kv_config, 64, 4);
    expect(backend != nullptr, "create generation KV backend");
    if (backend == nullptr) {
        static_cast<void>(cudaStreamDestroy(stream));
        return;
    }

    CudaModelRunnerCreateResult created = createCudaTinyLlamaModelRunner(
        archive.manifest_path.string(),
        archive.data_path.string(),
        *backend,
        reinterpret_cast<EngineStream>(stream)
    );
    expect(created.ok(), std::string("create generation model runner: ")
        + created.status.detail);
    if (!created.ok()) {
        static_cast<void>(cudaStreamDestroy(stream));
        return;
    }
    SingleRequestGenerationRuntime runtime(*backend, *created.runner);

    struct Workload final {
        RequestId request_id;
        std::uint32_t input_length;
        std::uint32_t output_length;
    };
    std::vector<Workload> const workloads{{51, 32, 1}, {52, 128, 32}};
    for (Workload const& workload : workloads) {
        std::vector<std::uint32_t> input(workload.input_length);
        for (std::uint32_t index = 0; index < workload.input_length; ++index) {
            input[index] = (index * 5 + 1) % archive.config.vocabulary_size;
        }
        std::vector<std::uint32_t> const expected = referenceGeneration(
            archive, input, workload.output_length
        );
        std::uint32_t const eos = eosOutside(
            expected, archive.config.vocabulary_size
        );
        expect(eos < archive.config.vocabulary_size,
            "generation reference leaves an EOS sentinel");
        if (eos >= archive.config.vocabulary_size) {
            continue;
        }

        GenerationTerminal terminal = runtime.generate(GenerationRequest{
            workload.request_id,
            input,
            SamplingConfig{workload.output_length, eos},
            {},
        });
        expect(terminal.ok(), std::string("CUDA generation succeeds: ")
            + terminal.detail);
        expect(terminal.reason == GenerationTerminalReason::MaxNewTokens,
            "CUDA generation reaches the length terminal");
        expect(terminal.output_token_ids == expected,
            "CUDA generation tokens match independent CPU FP16 reference");
        expect(terminal.usage.prompt_tokens == workload.input_length
            && terminal.usage.completion_tokens == workload.output_length,
            "CUDA generation reports prompt and completion usage");
        expect(terminal.metrics.e2e_ns >= terminal.metrics.ttft_ns,
            "CUDA generation E2E includes TTFT");
        EngineKvBackendSnapshot const snapshot = backend->snapshot();
        expect(snapshot.request_count == 0
            && snapshot.active_transaction_count == 0
            && snapshot.committed_token_count == 0,
            "CUDA generation releases request, transaction, and committed KV");
        if (workload.input_length >= kExtentPageTokenCapacity) {
            expect(snapshot.automatic_promotion_successes >= 2
                    && snapshot.automatic_promotion_failures == 0,
                "long generation automatically exercises Extent pages");
        }
        expect(backend->checkInvariants(),
            "CUDA generation preserves backend invariants");
    }

    created.runner.reset();
    expect(cudaStreamDestroy(stream) == cudaSuccess,
        "destroy generation stream");
}

void testIterationSchedulerRuntime()
{
    TestArchive archive(
        std::filesystem::temp_directory_path()
            / "kim_kv_iteration_scheduler_cuda_contract"
    );
    cudaStream_t stream = nullptr;
    expect(cudaStreamCreate(&stream) == cudaSuccess,
        "create scheduler stream");
    std::size_t const attention_workspace =
        archive.config.attention_head_count
        * archive.config.max_position_embeddings * sizeof(float);
    EngineKvConfig const kv_config{
        KvLayout{
            archive.config.layer_count,
            archive.config.kv_head_count,
            archive.config.head_dimension,
        },
        archive.config.attention_head_count,
        attention_workspace,
    };

    for (std::uint32_t concurrency : {1U, 2U, 4U}) {
        std::unique_ptr<EngineKvBackend> backend =
            createHeterogeneousCudaEngineKvBackend(kv_config, 128, 8);
        expect(backend != nullptr, "create scheduler KV backend");
        if (backend == nullptr) {
            continue;
        }
        CudaModelRunnerCreateResult created = createCudaTinyLlamaModelRunner(
            archive.manifest_path.string(),
            archive.data_path.string(),
            *backend,
            reinterpret_cast<EngineStream>(stream)
        );
        expect(created.ok(), std::string("create scheduler model runner: ")
            + created.status.detail);
        if (!created.ok()) {
            continue;
        }

        IterationSchedulerRuntime scheduler(
            *backend,
            *created.runner,
            {concurrency, concurrency * 2, 1024, 2}
        );
        std::unordered_map<RequestId, std::vector<std::uint32_t>> expected;
        std::uint64_t expected_prefill_tokens = 0;
        for (std::uint32_t index = 0; index < concurrency; ++index) {
            RequestId const request_id = 100 + concurrency * 10 + index;
            std::vector<std::uint32_t> input(2 + index);
            for (std::uint32_t position = 0;
                 position < input.size();
                 ++position) {
                input[position] = (position * 5 + index + 1)
                    % archive.config.vocabulary_size;
            }
            std::vector<std::uint32_t> output = referenceGeneration(
                archive, input, 3
            );
            expected_prefill_tokens += input.size();
            std::uint32_t const eos = eosOutside(
                output, archive.config.vocabulary_size
            );
            expect(eos < archive.config.vocabulary_size,
                "scheduler reference leaves EOS sentinel");
            expected.emplace(request_id, output);
            expect(scheduler.submit(GenerationRequest{
                request_id,
                std::move(input),
                SamplingConfig{3, eos},
                {},
            }).ok(), "CUDA scheduler request accepted");
        }

        std::vector<GenerationTerminal> terminals = scheduler.drain();
        expect(terminals.size() == concurrency,
            "CUDA c1/c2/c4 produces one terminal per request");
        for (GenerationTerminal const& terminal : terminals) {
            expect(terminal.ok(), std::string("CUDA scheduler succeeds: ")
                + terminal.detail);
            auto const found = expected.find(terminal.request_id);
            expect(found != expected.end()
                && terminal.output_token_ids == found->second,
                "CUDA c1/c2/c4 matches independent FP16 reference");
        }
        EngineKvBackendSnapshot const kv = backend->snapshot();
        IterationSchedulerSnapshot const state = scheduler.snapshot();
        expect(kv.request_count == 0
            && kv.active_transaction_count == 0
            && kv.committed_token_count == 0,
            "CUDA scheduler reclaims KV state");
        expect(state.activeCount() == 0
            && state.reserved_kv_tokens == 0,
            "CUDA scheduler reclaims admission budgets");
        expect(state.prefill_tokens == expected_prefill_tokens
            && state.decode_tokens == concurrency * 2,
            "CUDA scheduler accounts chunked prefill and decode tokens");
        expect(state.model_forward_tokens
            == state.prefill_tokens + state.decode_tokens,
            "CUDA scheduler accounts every model batch lane");
        if (concurrency > 1) {
            expect(state.model_forward_batches < state.model_forward_tokens,
                "CUDA scheduler executes multi-request dense batches");
            expect(kv.batched_kv_write_submissions != 0
                    && kv.batched_kv_write_lanes
                        > kv.batched_kv_write_submissions,
                "CUDA scheduler submits multi-lane KV-write batches");
            expect(kv.batched_attention_submissions != 0
                    && kv.batched_attention_lanes
                        > kv.batched_attention_submissions,
                "CUDA scheduler submits multi-lane paged attention batches");
        } else {
            expect(kv.batched_kv_write_submissions == 0
                    && kv.batched_kv_write_lanes == 0,
                "single-request CUDA scheduler keeps scalar KV-write path");
            expect(kv.batched_attention_submissions == 0
                    && kv.batched_attention_lanes == 0,
                "single-request CUDA scheduler keeps scalar attention path");
        }
        expect(backend->checkInvariants(),
            "CUDA scheduler preserves backend invariants");
    }

    expect(cudaStreamDestroy(stream) == cudaSuccess,
        "destroy scheduler stream");
}

} // namespace

int main()
{
    testModelRunner();
    testGenerationRuntime();
    testIterationSchedulerRuntime();
    if (failures == 0) {
        std::cout << "CUDA model runner contract passed\n";
    }
    return failures == 0 ? 0 : 1;
}
