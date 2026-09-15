#include "kim-kv/model/weight_manifest.h"
#include "support/unique_temp_directory.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace kimkvcache;
using test_support::UniqueTempDirectory;

int failures = 0;

void expect(bool condition, std::string const& message)
{
    if (!condition) {
        std::cerr << "[FAILED] " << message << '\n';
        ++failures;
    }
}

std::vector<std::pair<std::string, std::vector<std::uint32_t>>>
expectedTensors(TinyLlamaConfig const& config)
{
    using Tensor = std::pair<std::string, std::vector<std::uint32_t>>;
    std::vector<Tensor> tensors;
    tensors.emplace_back(
        "model.embed_tokens.weight",
        std::vector<std::uint32_t>{config.vocabulary_size, config.hidden_size}
    );
    tensors.emplace_back(
        "model.norm.weight",
        std::vector<std::uint32_t>{config.hidden_size}
    );
    tensors.emplace_back(
        "lm_head.weight",
        std::vector<std::uint32_t>{config.vocabulary_size, config.hidden_size}
    );
    std::uint32_t const kv_size = config.kv_head_count * config.head_dimension;
    for (std::uint32_t layer = 0; layer < config.layer_count; ++layer) {
        std::string const prefix = "model.layers."
            + std::to_string(layer) + ".";
        tensors.emplace_back(prefix + "input_layernorm.weight",
            std::vector<std::uint32_t>{config.hidden_size});
        tensors.emplace_back(prefix + "post_attention_layernorm.weight",
            std::vector<std::uint32_t>{config.hidden_size});
        tensors.emplace_back(prefix + "self_attn.q_proj.weight",
            std::vector<std::uint32_t>{config.hidden_size, config.hidden_size});
        tensors.emplace_back(prefix + "self_attn.k_proj.weight",
            std::vector<std::uint32_t>{kv_size, config.hidden_size});
        tensors.emplace_back(prefix + "self_attn.v_proj.weight",
            std::vector<std::uint32_t>{kv_size, config.hidden_size});
        tensors.emplace_back(prefix + "self_attn.o_proj.weight",
            std::vector<std::uint32_t>{config.hidden_size, config.hidden_size});
        tensors.emplace_back(prefix + "mlp.gate_proj.weight",
            std::vector<std::uint32_t>{
                config.intermediate_size, config.hidden_size});
        tensors.emplace_back(prefix + "mlp.up_proj.weight",
            std::vector<std::uint32_t>{
                config.intermediate_size, config.hidden_size});
        tensors.emplace_back(prefix + "mlp.down_proj.weight",
            std::vector<std::uint32_t>{
                config.hidden_size, config.intermediate_size});
    }
    return tensors;
}

void writeManifest(
    std::filesystem::path const& path,
    WeightManifest const& manifest)
{
    std::ofstream output(path);
    TinyLlamaConfig const& config = manifest.config;
    output << "version=" << manifest.version << '\n'
        << "checkpoint=" << manifest.checkpoint << '\n'
        << "checkpoint_revision=" << manifest.checkpoint_revision << '\n'
        << "tokenizer_revision=" << manifest.tokenizer_revision << '\n'
        << "config_sha256=" << manifest.config_sha256 << '\n'
        << "dtype=" << manifest.data_type << '\n'
        << "data_file=" << manifest.data_file << '\n'
        << "data_bytes=" << manifest.data_bytes << '\n'
        << "hidden_size=" << config.hidden_size << '\n'
        << "intermediate_size=" << config.intermediate_size << '\n'
        << "layer_count=" << config.layer_count << '\n'
        << "attention_head_count=" << config.attention_head_count << '\n'
        << "kv_head_count=" << config.kv_head_count << '\n'
        << "head_dimension=" << config.head_dimension << '\n'
        << "vocabulary_size=" << config.vocabulary_size << '\n'
        << "max_position_embeddings=" << config.max_position_embeddings << '\n'
        << "bos_token_id=" << config.bos_token_id << '\n'
        << "eos_token_id=" << config.eos_token_id << '\n'
        << "rms_norm_epsilon=" << config.rms_norm_epsilon << '\n'
        << "rope_theta=" << config.rope_theta << '\n'
        << "tied_word_embeddings="
        << (config.tied_word_embeddings ? 1 : 0) << '\n'
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

void testManifestContract()
{
    TinyLlamaConfig const config{
        8, 16, 2, 4, 2, 2, 16, 32, 1, 2, 1.0e-5F, 10000.0F, false,
    };
    expect(config.valid(), "small model config is valid");
    expect(tinyLlama11bChatConfig().valid(), "production config is valid");

    UniqueTempDirectory temporary_directory("kim_kv_manifest_contract");
    std::filesystem::path const& root = temporary_directory.path();
    std::filesystem::path const data_path = root / "weights.bin";
    std::filesystem::path const manifest_path = root / "weights.manifest";

    WeightManifest manifest;
    manifest.version = 1;
    manifest.checkpoint = "contract/tinyllama";
    manifest.checkpoint_revision = "0123456789abcdef";
    manifest.tokenizer_revision = "0123456789abcdef";
    manifest.config_sha256 = std::string(64, 'a');
    manifest.data_type = "fp16";
    manifest.data_file = data_path.filename().string();
    manifest.config = config;

    std::uint64_t offset = 0;
    for (auto const& [name, shape] : expectedTensors(config)) {
        std::uint64_t elements = 1;
        for (std::uint32_t dimension : shape) {
            elements *= dimension;
        }
        manifest.tensors.push_back(WeightTensorDescriptor{
            name, offset, elements * 2, shape, {},
        });
        offset += elements * 2;
    }
    manifest.data_bytes = offset;
    {
        std::ofstream data(data_path, std::ios::binary);
        std::vector<char> zeros(static_cast<std::size_t>(offset), 0);
        data.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
    }
    for (WeightTensorDescriptor& tensor : manifest.tensors) {
        expect(computeWeightDataSha256(
            data_path.string(), tensor.offset, tensor.byte_size, tensor.sha256
        ).ok(), "compute tensor checksum");
    }
    writeManifest(manifest_path, manifest);

    WeightManifestLoadResult loaded = loadWeightManifest(
        manifest_path.string()
    );
    expect(loaded.ok(), "load complete manifest");
    expect(loaded.manifest.tensors.size() == 21, "all expected tensors loaded");
    expect(loaded.manifest.find("model.layers.1.mlp.down_proj.weight")
        != nullptr, "find tensor by stable name");
    expect(validateWeightDataFile(
        loaded.manifest, data_path.string()).ok(), "verify archive hashes");

    {
        std::fstream data(data_path, std::ios::binary | std::ios::in
            | std::ios::out);
        char byte = 1;
        data.write(&byte, 1);
    }
    expect(validateWeightDataFile(
        loaded.manifest, data_path.string()).error
            == WeightManifestError::ChecksumMismatch,
        "corrupted tensor is rejected");

    WeightManifest missing = loaded.manifest;
    missing.tensors.pop_back();
    expect(validateWeightManifest(missing).error
        == WeightManifestError::MissingTensor,
        "missing required tensor is rejected");
    WeightManifest duplicate = loaded.manifest;
    duplicate.tensors.push_back(duplicate.tensors.front());
    expect(validateWeightManifest(duplicate).error
        == WeightManifestError::DuplicateTensor,
        "duplicate tensor is rejected");
    expect(!temporary_directory.cleanup(), "remove unique temporary directory");
}

} // namespace

int main()
{
    testManifestContract();
    if (failures == 0) {
        std::cout << "weight manifest contract passed\n";
    }
    return failures == 0 ? 0 : 1;
}
