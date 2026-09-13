function(add_kim_kv_contract target_name test_name source_file timeout_seconds)
    add_executable(
        ${target_name}
        ${source_file}
    )

    target_link_libraries(
        ${target_name}
        PRIVATE
            kim_kv_warnings
            ${ARGN}
    )

    add_test(
        NAME ${test_name}
        COMMAND ${target_name}
    )

    set_tests_properties(
        ${test_name}
        PROPERTIES
            TIMEOUT ${timeout_seconds}
    )
endfunction()

add_kim_kv_contract(
    public_headers_test
    public_headers_contract
    api/public_headers_test.cpp
    10
    kim_kv
)

add_kim_kv_contract(
    page_pool_test
    page_pool_contract
    core/page_pool_test.cpp
    10
    kim_kv
)

add_kim_kv_contract(
    block_table_test
    block_table_contract
    runtime/block_table_test.cpp
    10
    kim_kv
)

add_kim_kv_contract(
    engine_kv_contract_test
    engine_kv_contract
    engine/engine_kv_contract_test.cpp
    10
    kim_kv
)

add_kim_kv_contract(
    generation_runtime_test
    generation_runtime_contract
    engine/generation_runtime_test.cpp
    10
    kim_kv
)

add_kim_kv_contract(
    iteration_scheduler_test
    iteration_scheduler_contract
    engine/iteration_scheduler_test.cpp
    10
    kim_kv
)

add_kim_kv_contract(
    weight_manifest_test
    weight_manifest_contract
    model/weight_manifest_test.cpp
    30
    kim_kv
)

add_kim_kv_contract(
    token_reservation_test
    token_reservation_contract
    runtime/token_reservation_test.cpp
    10
    kim_kv
)

add_kim_kv_contract(
    fixed_page_test
    fixed_page_contract
    runtime/fixed_page_test.cpp
    10
    kim_kv
)

add_kim_kv_contract(
    promotion_test
    promotion_contract
    runtime/promotion_test.cpp
    10
    kim_kv
)

add_kim_kv_contract(
    page_lease_test
    page_lease_contract
    runtime/page_lease_test.cpp
    10
    kim_kv
)

add_kim_kv_contract(
    kv_reference_test
    kv_reference_contract
    reference/kv_reference_test.cpp
    10
    kim_kv
)

if(KIM_KV_ENABLE_CUDA)
    add_kim_kv_contract(
        paged_attention_fusion_test
        paged_attention_fusion_contract
        cuda/paged_attention_fusion_test.cpp
        120
        kim_kv_cuda
        CUDA::cudart
    )
    target_include_directories(
        paged_attention_fusion_test
        PRIVATE "${PROJECT_SOURCE_DIR}/src/cuda/kv"
    )

    add_kim_kv_contract(
        cuda_kv_test
        cuda_kv_contract
        cuda/cuda_kv_test.cpp
        60
        kim_kv_cuda
        CUDA::cudart
    )
    target_include_directories(
        cuda_kv_test
        PRIVATE
            "${PROJECT_SOURCE_DIR}/src/cuda/kv"
    )

    add_kim_kv_contract(
        cuda_engine_kv_backend_test
        cuda_engine_kv_backend_contract
        cuda/cuda_engine_kv_backend_test.cpp
        60
        kim_kv_cuda
        CUDA::cudart
    )
    target_include_directories(
        cuda_engine_kv_backend_test
        PRIVATE
            "${PROJECT_SOURCE_DIR}/src/cuda/kv"
    )

    add_kim_kv_contract(
        cuda_model_runner_test
        cuda_model_runner_contract
        cuda/cuda_model_runner_test.cpp
        120
        kim_kv_cuda
        CUDA::cudart
    )

    if(KIM_KV_BUILD_BENCHMARKS)
        foreach(fixed_page_tokens IN ITEMS 8 16 32 64)
            add_test(
                NAME fixed_cuda_benchmark_smoke_${fixed_page_tokens}
                COMMAND
                    kim_kv_cuda_benchmark
                    --workload short
                    --fixed-page-tokens ${fixed_page_tokens}
                    --warmup 0
                    --iterations 1
                    --max-sequence-length 65
                    --layers 2
                    --kv-heads 2
                    --head-dimension 8
                    --output-dir
                    ${CMAKE_CURRENT_BINARY_DIR}/fixed_cuda_${fixed_page_tokens}
            )
            set_tests_properties(
                fixed_cuda_benchmark_smoke_${fixed_page_tokens}
                PROPERTIES
                    TIMEOUT 30
            )
        endforeach()
    endif()
endif()

if(KIM_KV_BUILD_BENCHMARKS)
    add_kim_kv_contract(
        benchmark_test
        benchmark_contract
        benchmark/benchmark_test.cpp
        60
        kim_kv_benchmark
    )
endif()
