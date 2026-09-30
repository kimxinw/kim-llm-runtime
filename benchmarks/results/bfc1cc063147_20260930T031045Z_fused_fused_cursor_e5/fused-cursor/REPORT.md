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
| decode_short_c1 | -1.00% | -1.00% | +0.84% |
| decode_long_prompt_c1 | +4.76% | +4.26% | -3.41% |
| decode_short_c2 | -1.68% | -1.73% | +1.77% |
| decode_long_prompt_c2 | +0.52% | -0.46% | +2.11% |
| decode_short_c4 | +0.05% | -0.22% | -0.00% |
| decode_long_prompt_c4 | +0.39% | +0.13% | -0.51% |
| mixed_c4 | -5.23% | -5.13% | +5.80% |

## 绝对结果

| Variant | Workload | E2E p50 (ms) | TPOT p50 (ms) | Output tokens/s |
|---|---|---:|---:|---:|
| fixed_8 | decode_short_c1 | 279.381 | 8.399 | 114.831 |
| fixed_8 | decode_short_c4 | 335.008 | 9.462 | 381.765 |
| fixed_8 | decode_long_prompt_c4 | 490.560 | 9.826 | 261.246 |
| fixed_8 | mixed_c4 | 422.328 | 10.135 | 266.027 |
| fixed_16 | decode_short_c1 | 275.053 | 8.252 | 116.831 |
| fixed_16 | decode_short_c4 | 332.239 | 9.351 | 385.314 |
| fixed_16 | decode_long_prompt_c4 | 486.698 | 9.772 | 263.175 |
| fixed_16 | mixed_c4 | 421.551 | 10.128 | 266.812 |
| fixed_32 | decode_short_c1 | 277.625 | 8.337 | 115.363 |
| fixed_32 | decode_short_c4 | 331.822 | 9.349 | 385.945 |
| fixed_32 | decode_long_prompt_c4 | 489.613 | 9.850 | 262.813 |
| fixed_32 | mixed_c4 | 405.394 | 9.728 | 277.244 |
| fixed_64 | decode_short_c1 | 273.195 | 8.204 | 117.298 |
| fixed_64 | decode_short_c4 | 316.797 | 8.923 | 404.377 |
| fixed_64 | decode_long_prompt_c4 | 491.130 | 9.871 | 263.705 |
| fixed_64 | mixed_c4 | 425.332 | 10.197 | 264.432 |
| hetero_8_64 | decode_short_c1 | 276.598 | 8.314 | 115.797 |
| hetero_8_64 | decode_short_c4 | 335.176 | 9.441 | 381.750 |
| hetero_8_64 | decode_long_prompt_c4 | 492.472 | 9.839 | 259.918 |
| hetero_8_64 | mixed_c4 | 400.234 | 9.615 | 281.459 |

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
