add_library(
    kim_kv_cuda
    STATIC
        attention/paged_decode_kernels.cu
        attention/reference_attention.cpp
        attention/reference_attention_kernels.cu
        common/submission.cpp
        engine/backend.cpp
        engine/transaction.cpp
        kv/append.cpp
        kv/cache.cpp
        kv/fixed_cache.cpp
        kv/gather.cpp
        kv/page_kernels.cu
        kv/promotion.cpp
        kv/storage.cpp
        model/kernels.cu
        model/runner.cpp
        model/runner_factory.cpp
        model/runner_support.cpp
)

set_target_properties(
    kim_kv_cuda
    PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)

target_include_directories(
    kim_kv_cuda
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
)

if(KIM_KV_ENABLE_FUSED_ATTENTION)
    target_compile_definitions(kim_kv_cuda PRIVATE KIM_KV_ENABLE_FUSED_ATTENTION=1)
endif()

target_link_libraries(
    kim_kv_cuda
    PRIVATE
        kim_kv_warnings
    PUBLIC
        kim_kv
        CUDA::cudart
        CUDA::cublas
)
