# E5 TinyLlama 端到端证据

## 验证结论

| 项目 | 结果 |
|---|---|
| 五种 Page 策略配置一致 | PASS |
| CUDA KV Storage 字节预算一致 | PASS |
| 跨策略输出 Token 一致 | PASS |
| Transformers FP16 独立 Reference | PASS（8 个唯一 Prompt） |
| 每组测量轮数 | 3 或以上 |
| 结果类型 | 真实 TinyLlama 1.1B FP16 端到端 Generation |

## Fixed-8 与 Hetero-8/64

正值表示 Hetero 指标高于 Fixed-8；对于延迟，正值代表更慢。

| Workload | E2E p50 差值 | TPOT p50 差值 | Output tokens/s 差值 |
|---|---:|---:|---:|
| decode_short_c1 | -6.31% | -6.51% | +6.74% |
| decode_long_prompt_c1 | -2.96% | -3.58% | +3.10% |
| decode_short_c2 | +2.80% | +2.96% | -2.52% |
| decode_long_prompt_c2 | -2.16% | -1.95% | +2.10% |
| decode_short_c4 | +4.70% | +4.71% | -4.23% |
| decode_long_prompt_c4 | -0.57% | -0.91% | +0.56% |
| mixed_c4 | +0.32% | -1.26% | -0.22% |

## 绝对结果

| Variant | Workload | E2E p50 (ms) | TPOT p50 (ms) | Output tokens/s |
|---|---|---:|---:|---:|
| fixed_8 | decode_short_c1 | 279.281 | 8.399 | 114.606 |
| fixed_8 | decode_short_c4 | 309.224 | 8.704 | 413.931 |
| fixed_8 | decode_long_prompt_c4 | 484.254 | 9.621 | 264.486 |
| fixed_8 | mixed_c4 | 411.670 | 9.962 | 272.837 |
| fixed_16 | decode_short_c1 | 254.282 | 7.639 | 125.755 |
| fixed_16 | decode_short_c4 | 325.727 | 9.287 | 391.708 |
| fixed_16 | decode_long_prompt_c4 | 483.304 | 9.633 | 265.001 |
| fixed_16 | mixed_c4 | 412.551 | 9.924 | 272.242 |
| fixed_32 | decode_short_c1 | 264.848 | 7.965 | 120.803 |
| fixed_32 | decode_short_c4 | 315.699 | 8.899 | 405.251 |
| fixed_32 | decode_long_prompt_c4 | 471.432 | 9.413 | 271.509 |
| fixed_32 | mixed_c4 | 405.436 | 9.793 | 276.846 |
| fixed_64 | decode_short_c1 | 266.802 | 8.017 | 120.088 |
| fixed_64 | decode_short_c4 | 314.274 | 8.847 | 407.123 |
| fixed_64 | decode_long_prompt_c4 | 475.552 | 9.547 | 270.689 |
| fixed_64 | mixed_c4 | 412.465 | 9.910 | 272.337 |
| hetero_8_64 | decode_short_c1 | 261.654 | 7.852 | 122.330 |
| hetero_8_64 | decode_short_c4 | 323.752 | 9.114 | 396.414 |
| hetero_8_64 | decode_long_prompt_c4 | 481.483 | 9.533 | 265.968 |
| hetero_8_64 | mixed_c4 | 412.985 | 9.837 | 272.236 |

## Batch 执行证据

| Workload | Model Batch Size | Attention Batch Size | Attention Submissions/Run |
|---|---:|---:|---:|
| decode_short_c1 | 1.91 | 0.00 | 0 |
| decode_short_c2 | 3.82 | 2.00 | 682 |
| decode_short_c4 | 7.20 | 4.00 | 682 |
| decode_long_prompt_c4 | 13.53 | 4.00 | 682 |
| mixed_c4 | 9.45 | 3.61 | 682 |

## Capacity 与故障隔离

| Variant | Capacity 完成/失败/拒绝 | Fault 完成/失败 | Peak fragmentation tokens |
|---|---:|---:|---:|
| fixed_8 | 48/0/3 | 9/3 | 0 |
| fixed_16 | 48/0/3 | 9/3 | 0 |
| fixed_32 | 48/0/3 | 9/3 | 256 |
| fixed_64 | 24/24/3 | 9/3 | 384 |
| hetero_8_64 | 42/6/3 | 9/3 | 0 |

## 边界说明

- Hetero 的 Extent Page 分配次数为 `18`。大于零表示 Generation 自动 Promotion 已在本轮 E2E Workload 中生效，长序列实际使用了 Extent Page。
- Dense GEMM 已按总 Token 数动态 Batch 执行；单 Token 多请求使用 Batch KV Write 与 Paged Decode Attention。多 Token Chunk 已走真实因果 Prefill，但包含多 Token Chunk 的混合 Batch 仍按请求提交 KV/Attention。
- Reference 的 Prefill Attention Scores 与 Softmax/Value Output 为两个 Kernel；Fused 在 head dimension ≤128 时使用在线 Softmax 单 Kernel，>128 回退 Reference。两条路径仍保留 Score Workspace 接口。
- Microbenchmark 与本报告的 E2E 结果分开；K6 的 Gather/Promotion 收益不能直接替代模型端到端收益。
- Nsight Systems GPU Activity Timeline 受当前 WSL2/CUPTI 环境限制，本报告不以 CUDA API Duration 冒充 Kernel Timeline。
- 未与 vLLM 或 TensorRT-LLM 比较峰值性能。
