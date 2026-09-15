#include "kim-kv/benchmark/benchmark.h"
#include "support/unique_temp_directory.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace kimkvcache::benchmark;

int failures = 0;

void expect(bool condition, std::string const& message)
{
    if (!condition) {
        std::cerr << "[FAILED] " << message << '\n';
        ++failures;
    }
}

std::size_t countOccurrences(
    std::string_view text,
    std::string_view needle)
{
    if (needle.empty()) {
        return 0;
    }

    std::size_t count = 0;
    std::size_t position = 0;
    while ((position = text.find(needle, position))
           != std::string_view::npos) {
        ++count;
        position += needle.size();
    }
    return count;
}

CapacityResult const* findCapacity(
    std::vector<CapacityResult> const& results,
    BaselineKind baseline)
{
    for (CapacityResult const& result : results) {
        if (result.baseline == baseline) {
            return &result;
        }
    }
    return nullptr;
}

BaselineKind fixedBaseline(std::uint16_t tokens_per_page)
{
    switch (tokens_per_page) {
    case 8:
        return BaselineKind::Fixed8;
    case 16:
        return BaselineKind::Fixed16;
    case 32:
        return BaselineKind::Fixed32;
    case 64:
        return BaselineKind::Fixed64;
    }
    throw std::invalid_argument("unsupported fixed page size in test");
}

void testWorkloadGeneration()
{
    BenchmarkConfig config{};
    config.request_count = 256;
    config.seed = 1234567;

    for (WorkloadKind const workload : allWorkloads()) {
        WorkloadTrace const first = generateWorkload(workload, config);
        WorkloadTrace const second = generateWorkload(workload, config);

        expect(first.seed == config.seed, "trace must record its seed");
        expect(
            first.sequence_lengths.size() == config.request_count,
            "trace must contain the configured request count"
        );
        expect(
            first.sequence_lengths == second.sequence_lengths,
            "same workload and seed must reproduce the exact trace"
        );
        for (std::uint32_t const length : first.sequence_lengths) {
            expect(length != 0, "generated sequence length must be non-zero");
            expect(
                length <= config.maximum_sequence_length,
                "generated sequence length must respect the maximum"
            );
        }
    }

    WorkloadTrace const short_trace = generateWorkload(
        WorkloadKind::Short,
        config
    );
    for (std::uint32_t const length : short_trace.sequence_lengths) {
        expect(length <= 63, "Short workload must stay within [1,63]");
    }

    WorkloadTrace const adversarial = generateWorkload(
        WorkloadKind::Adversarial,
        config
    );
    for (std::uint32_t const length : adversarial.sequence_lengths) {
        expect(
            length % 64 == 1,
            "Adversarial lengths must be exactly 64n+1"
        );
    }

    WorkloadTrace const shared = generateWorkload(
        WorkloadKind::SharedPrompt,
        config
    );
    expect(
        shared.reused_tokens != 0,
        "Shared Prompt trace must record reused prefix tokens"
    );
}

void testCapacityAccounting()
{
    BenchmarkConfig config{};
    config.request_count = 2;
    config.concurrency = 2;
    config.maximum_sequence_length = 544;
    config.capacity_budget_bytes = 1ULL << 40U;

    WorkloadTrace trace{};
    trace.workload = WorkloadKind::Adversarial;
    trace.seed = config.seed;
    trace.sequence_lengths = {1, 65};

    std::vector<CapacityResult> const results = calculateCapacity(
        trace,
        config
    );
    expect(results.size() == 7, "all seven baselines must be reported");

    CapacityResult const* contiguous = findCapacity(
        results,
        BaselineKind::ContiguousMax
    );
    expect(contiguous != nullptr, "Contiguous Max result must exist");
    if (contiguous != nullptr) {
        expect(contiguous->used_tokens == 66, "used token accounting");
        expect(
            contiguous->reserved_tokens == 1088,
            "Contiguous Max reserves max_seq_len per request"
        );
        expect(
            contiguous->block_table_entries == 2,
            "Contiguous Max has one entry per request"
        );
    }

    CapacityResult const* fixed64 = findCapacity(
        results,
        BaselineKind::Fixed64
    );
    expect(fixed64 != nullptr, "Fixed-64 result must exist");
    if (fixed64 != nullptr) {
        expect(fixed64->reserved_tokens == 192, "Fixed-64 rounding");
        expect(
            fixed64->internal_fragmentation_tokens == 126,
            "Fixed-64 fragmentation"
        );
        expect(fixed64->block_table_entries == 3, "Fixed-64 entries");
    }

    CapacityResult const* hetero = findCapacity(
        results,
        BaselineKind::HeteroWithPromotion
    );
    expect(hetero != nullptr, "promoted Hetero result must exist");
    if (hetero != nullptr) {
        expect(hetero->reserved_tokens == 80, "Hetero reserved tokens");
        expect(
            hetero->peak_reserved_tokens == 144,
            "Promotion target must count toward peak reservation"
        );
        expect(
            hetero->internal_fragmentation_tokens == 14,
            "Hetero fragmentation"
        );
        expect(hetero->block_table_entries == 3, "Hetero entries");
        expect(hetero->micro_pages == 2, "Hetero Micro page count");
        expect(hetero->extent_pages == 1, "Hetero Extent page count");
        expect(
            hetero->used_bytes == 66ULL * 22'528ULL,
            "capacity rows must include used bytes"
        );
        expect(
            std::fabs(hetero->utilization - 0.825) < 1.0e-12,
            "Hetero utilization"
        );
    }
}

void testLatencySummary()
{
    std::vector<OperationSample> samples;
    for (std::uint64_t const duration : {10, 20, 30, 40, 50}) {
        samples.push_back(OperationSample{
            OperationKind::Append,
            samples.size(),
            duration,
            0,
            0,
            duration != 40,
            {},
        });
    }
    samples.push_back(OperationSample{
        OperationKind::Release,
        samples.size(),
        100,
        0,
        0,
        true,
        {},
    });

    LatencySummary const summary = summarizeLatency(
        OperationKind::Append,
        samples
    );
    expect(summary.sample_count == 5, "summary filters by operation");
    expect(summary.failure_count == 1, "summary counts failures");
    expect(summary.minimum_ns == 10, "summary minimum");
    expect(summary.maximum_ns == 50, "summary maximum");
    expect(summary.mean_ns == 30, "summary arithmetic mean");
    expect(summary.p50_ns == 30, "nearest-rank p50");
    expect(summary.p95_ns == 50, "nearest-rank p95");
    expect(summary.p99_ns == 50, "nearest-rank p99");
}

void testCpuHarnessAndReports()
{
    BenchmarkConfig config{};
    config.request_count = 24;
    config.concurrency = 4;
    config.seed = 99;
    config.git_commit = "benchmark-contract";

    std::vector<WorkloadKind> const workloads{
        WorkloadKind::Short,
        WorkloadKind::SharedPrompt,
        WorkloadKind::ForkCow,
        WorkloadKind::Fault,
    };
    BenchmarkReport const report = runCpuBenchmark(config, workloads);
    expect(report.successful(), "CPU smoke benchmark must succeed");
    expect(report.workloads.size() == 4, "four workloads must run");
    expect(report.capacity.size() == 28, "seven baselines per workload");
    for (WorkloadResult const& result : report.workloads) {
        expect(result.invariants_ok, "workload invariants must hold");
        expect(result.resources_released, "workload resources must return");
        expect(
            result.failed_operations == 0,
            "workload must not contain unexpected failures"
        );
        expect(!result.samples.empty(), "workload must retain raw samples");
    }

    kimkvcache::test_support::UniqueTempDirectory temporary_directory(
        "kim_kv_benchmark_contract"
    );
    std::filesystem::path const& directory = temporary_directory.path();
    std::filesystem::path const json = directory / "result.json";
    std::filesystem::path const csv = directory / "result.csv";
    writeJsonReport(report, json.string());
    writeCsvReport(report, csv.string());

    std::ifstream json_input(json);
    std::ifstream csv_input(csv);
    std::string const json_text{
        std::istreambuf_iterator<char>(json_input),
        std::istreambuf_iterator<char>()
    };
    std::string const csv_text{
        std::istreambuf_iterator<char>(csv_input),
        std::istreambuf_iterator<char>()
    };
    expect(
        json_text.find("\"schema_version\": 1") != std::string::npos,
        "JSON schema version must be present"
    );
    expect(
        countOccurrences(json_text, "\"seed\":") == 1,
        "JSON config keys must not contain a duplicate seed"
    );
    expect(
        json_text.find("benchmark-contract") != std::string::npos,
        "JSON must record the Git revision"
    );
    expect(
        csv_text.find("record_type,suite,workload") == 0,
        "CSV header must be stable"
    );
    expect(
        csv_text.find("sample,cpu_metadata") != std::string::npos,
        "CSV must include raw operation samples"
    );
    json_input.close();
    csv_input.close();
    expect(!temporary_directory.cleanup(),
        "remove unique benchmark directory");
}

void testFixedPageHarnessAndCommandLine()
{
    BenchmarkConfig config{};
    config.request_count = 24;
    config.concurrency = 4;
    config.seed = 20260828;
    config.git_commit = "fixed-page-contract";
    config.capacity_budget_bytes = 1ULL << 30U;

    std::vector<WorkloadKind> const workloads{
        WorkloadKind::Short,
        WorkloadKind::SharedPrompt,
        WorkloadKind::ForkCow,
    };

    for (std::uint16_t const tokens_per_page : {8, 16, 32, 64}) {
        BenchmarkReport const report = runCpuFixedBenchmark(
            config,
            workloads,
            tokens_per_page
        );

        expect(report.successful(), "Fixed CPU smoke benchmark must succeed");
        expect(
            report.suite == "cpu_metadata_fixed_"
                + std::to_string(tokens_per_page),
            "Fixed report suite must encode its page size"
        );
        expect(report.workloads.size() == 3, "three Fixed workloads run");
        expect(
            report.capacity.size() == 3,
            "one Fixed capacity row per workload"
        );

        for (std::size_t index = 0; index < report.capacity.size(); ++index) {
            CapacityResult const& capacity = report.capacity[index];
            expect(
                capacity.micro_pages == 0 && capacity.extent_pages == 0,
                "Fixed capacity rows must not masquerade as Hetero pages"
            );
            expect(
                capacity.block_table_entries != 0,
                "Fixed capacity rows expose page count through entries"
            );

            WorkloadTrace const trace = generateWorkload(
                workloads[index],
                config
            );
            std::vector<CapacityResult> const analytic = calculateCapacity(
                trace,
                config
            );
            CapacityResult const* expected = findCapacity(
                analytic,
                fixedBaseline(tokens_per_page)
            );
            expect(expected != nullptr, "matching analytic Fixed row exists");
            if (expected != nullptr) {
                expect(
                    capacity.used_tokens == expected->used_tokens
                        && capacity.reserved_tokens
                            == expected->reserved_tokens
                        && capacity.block_table_entries
                            == expected->block_table_entries,
                    "executable Fixed probe must match analytic accounting"
                );
            }
        }

        for (WorkloadResult const& result : report.workloads) {
            expect(result.invariants_ok, "Fixed workload invariants hold");
            expect(result.resources_released, "Fixed workload pool drains");
            expect(
                result.failed_operations == 0,
                "Fixed workload has no unexpected failure"
            );
            expect(
                result.completed_requests == config.request_count,
                "Fixed workload completes the full trace"
            );
            expect(
                result.requests_per_second > 0.0,
                "Fixed workload reports non-zero throughput"
            );
        }
    }

    bool rejected_fault = false;
    try {
        static_cast<void>(runCpuFixedBenchmark(
            config,
            {WorkloadKind::Fault},
            8
        ));
    } catch (std::invalid_argument const&) {
        rejected_fault = true;
    }
    expect(rejected_fault, "Fixed runner must reject Fault workload");

    char const* valid_argv[]{
        "kim_kv_cpu_benchmark",
        "--fixed-page-tokens",
        "16",
        "--workload",
        "short",
    };
    CommandLineOptions valid_options{};
    std::string valid_error;
    expect(
        parseCommandLine(5, valid_argv, valid_options, valid_error),
        "Fixed CLI option must parse"
    );
    expect(
        valid_options.fixed_page_tokens == 16
            && valid_options.workloads.size() == 1
            && valid_options.workloads.front() == WorkloadKind::Short,
        "Fixed CLI must retain page size and explicit workload"
    );

    char const* default_argv[]{
        "kim_kv_cpu_benchmark",
        "--fixed-page-tokens",
        "8",
    };
    CommandLineOptions default_options{};
    std::string default_error;
    expect(
        parseCommandLine(3, default_argv, default_options, default_error),
        "Fixed CLI defaults must parse"
    );
    expect(
        default_options.workloads.size() == 6,
        "Fixed CLI defaults must select six comparable workloads"
    );
    expect(
        std::find(
            default_options.workloads.begin(),
            default_options.workloads.end(),
            WorkloadKind::Fault
        ) == default_options.workloads.end(),
        "Fixed CLI defaults must exclude Fault"
    );

    char const* invalid_argv[]{
        "kim_kv_cpu_benchmark",
        "--fixed-page-tokens",
        "12",
    };
    CommandLineOptions invalid_options{};
    std::string invalid_error;
    expect(
        !parseCommandLine(3, invalid_argv, invalid_options, invalid_error),
        "unsupported Fixed page size must be rejected by CLI"
    );
}

} // namespace

int main()
{
    testWorkloadGeneration();
    testCapacityAccounting();
    testLatencySummary();
    testCpuHarnessAndReports();
    testFixedPageHarnessAndCommandLine();

    if (failures != 0) {
        std::cerr << failures << " benchmark contract(s) failed\n";
        return 1;
    }

    std::cout << "All K5/K6 benchmark contracts passed\n";
    return 0;
}
