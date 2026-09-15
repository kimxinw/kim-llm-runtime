#pragma once

#include <cstdint>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace kimkvcache::test_support {

class UniqueTempDirectory final {
public:
    explicit UniqueTempDirectory(std::string const& prefix)
    {
        std::filesystem::path const parent =
            std::filesystem::temp_directory_path();
        std::mt19937_64 random(std::random_device{}());
        std::uniform_int_distribution<std::uint64_t> suffix;

        constexpr unsigned int kMaximumAttempts = 128;
        for (unsigned int attempt = 0; attempt < kMaximumAttempts; ++attempt) {
            std::filesystem::path candidate = parent
                / (prefix + "_" + std::to_string(suffix(random)));
            std::error_code error;
            if (std::filesystem::create_directory(candidate, error)) {
                path_ = std::move(candidate);
                return;
            }
            if (error) {
                throw std::filesystem::filesystem_error(
                    "create unique temporary directory", candidate, error
                );
            }
        }

        throw std::runtime_error("exhausted temporary directory candidates");
    }

    ~UniqueTempDirectory()
    {
        static_cast<void>(cleanup());
    }

    UniqueTempDirectory(UniqueTempDirectory const&) = delete;
    UniqueTempDirectory& operator=(UniqueTempDirectory const&) = delete;

    [[nodiscard]] std::filesystem::path const& path() const noexcept
    {
        return path_;
    }

    [[nodiscard]] std::error_code cleanup() noexcept
    {
        if (path_.empty()) {
            return {};
        }

        std::error_code error;
        static_cast<void>(std::filesystem::remove_all(path_, error));
        if (!error) {
            path_.clear();
        }
        return error;
    }

private:
    std::filesystem::path path_;
};

} // namespace kimkvcache::test_support
