add_executable(
    kim_kv_tinyllama_validate
    tinyllama_validate.cpp
)

target_link_libraries(
    kim_kv_tinyllama_validate
    PRIVATE
        kim_kv_warnings
        kim_kv_cuda
        CUDA::cudart
)

add_executable(
    kim_kv_tinyllama_generate
    tinyllama_generate.cpp
)

target_link_libraries(
    kim_kv_tinyllama_generate
    PRIVATE
        kim_kv_warnings
        kim_kv_cuda
        CUDA::cudart
)

add_executable(
    kim_kv_tinyllama_e2e_benchmark
    tinyllama_e2e_benchmark.cpp
)

target_link_libraries(
    kim_kv_tinyllama_e2e_benchmark
    PRIVATE
        kim_kv_warnings
        kim_kv_cuda
        CUDA::cudart
)

file(
    GLOB
    kim_kv_nvtx_include_hints
    "/opt/nvidia/nsight-systems-cli/*/target-linux-x64/nvtx/include"
    "${CUDAToolkit_LIBRARY_ROOT}/nsight-compute-*/host/target-linux-x64/nvtx/include"
)
find_path(
    KIM_KV_NVTX_INCLUDE_DIR
    NAMES nvtx3/nvToolsExt.h
    HINTS
        ${CUDAToolkit_INCLUDE_DIRS}
        ${kim_kv_nvtx_include_hints}
)
if(KIM_KV_NVTX_INCLUDE_DIR)
    target_include_directories(
        kim_kv_tinyllama_e2e_benchmark
        PRIVATE
            "${KIM_KV_NVTX_INCLUDE_DIR}"
    )
    target_compile_definitions(
        kim_kv_tinyllama_e2e_benchmark
        PRIVATE
            KIM_KV_HAS_NVTX=1
    )
    target_link_libraries(
        kim_kv_tinyllama_e2e_benchmark
        PRIVATE
            ${CMAKE_DL_LIBS}
    )
else()
    message(
        STATUS
        "NVTX headers were not found; E2E --profile mode will be unavailable"
    )
endif()
