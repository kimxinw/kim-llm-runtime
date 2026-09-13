option(
    KIM_KV_ENABLE_ASAN
    "Enable AddressSanitizer"
    OFF
)

option(
    KIM_KV_ENABLE_UBSAN
    "Enable UndefinedBehaviorSanitizer"
    OFF
)

option(
    KIM_KV_ENABLE_TSAN
    "Enable ThreadSanitizer"
    OFF
)

option(
    KIM_KV_ENABLE_CUDA
    "Build the CUDA storage and correctness contracts"
    OFF
)

option(
    KIM_KV_BUILD_BENCHMARKS
    "Build deterministic benchmark harnesses"
    ON
)

option(
    KIM_KV_ENABLE_FUSED_ATTENTION
    "Fuse paged attention score, softmax and output for head dimensions <=128"
    OFF
)

if(
    KIM_KV_ENABLE_TSAN
    AND
    (
        KIM_KV_ENABLE_ASAN
        OR
        KIM_KV_ENABLE_UBSAN
    )
)
    message(
        FATAL_ERROR
        "TSan must use a separate build from ASan/UBSan"
    )
endif()
