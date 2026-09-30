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

# Cursor 默认跟随 Fusion：首次配置时 Fusion ON 则默认 ON。已有构建目录切换
# Fusion 后不会自动改写该缓存值，预设中显式指定两者。
if(KIM_KV_ENABLE_FUSED_ATTENTION)
    set(kim_kv_descriptor_cursor_default ON)
else()
    set(kim_kv_descriptor_cursor_default OFF)
endif()

option(
    KIM_KV_ENABLE_DESCRIPTOR_CURSOR
    "Reuse the current descriptor and page while fused attention walks tokens"
    ${kim_kv_descriptor_cursor_default}
)

if(
    KIM_KV_ENABLE_DESCRIPTOR_CURSOR
    AND
    NOT KIM_KV_ENABLE_FUSED_ATTENTION
)
    message(
        FATAL_ERROR
        "KIM_KV_ENABLE_DESCRIPTOR_CURSOR requires KIM_KV_ENABLE_FUSED_ATTENTION"
    )
endif()

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
