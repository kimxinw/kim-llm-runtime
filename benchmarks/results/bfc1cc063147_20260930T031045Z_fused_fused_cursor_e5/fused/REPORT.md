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
| decode_short_c1 | +0.86% | +0.87% | -0.88% |
| decode_long_prompt_c1 | -12.79% | -13.79% | +14.74% |
| decode_short_c2 | -8.95% | -8.91% | +9.75% |
| decode_long_prompt_c2 | -3.22% | -3.28% | +4.06% |
| decode_short_c4 | +0.64% | +0.82% | -0.74% |
| decode_long_prompt_c4 | -6.05% | -5.13% | +4.68% |
| mixed_c4 | -3.82% | -6.75% | +0.24% |

## 绝对结果

| Variant | Workload | E2E p50 (ms) | TPOT p50 (ms) | Output tokens/s |
|---|---|---:|---:|---:|
| fixed_8 | decode_short_c1 | 277.261 | 8.323 | 115.434 |
| fixed_8 | decode_short_c4 | 334.875 | 9.428 | 382.388 |
| fixed_8 | decode_long_prompt_c4 | 525.200 | 10.447 | 246.522 |
| fixed_8 | mixed_c4 | 438.903 | 10.827 | 265.477 |
| fixed_16 | decode_short_c1 | 275.129 | 8.271 | 116.174 |
| fixed_16 | decode_short_c4 | 308.428 | 8.699 | 415.200 |
| fixed_16 | decode_long_prompt_c4 | 509.494 | 10.192 | 251.264 |
| fixed_16 | mixed_c4 | 436.569 | 10.499 | 257.359 |
| fixed_32 | decode_short_c1 | 276.352 | 8.296 | 115.813 |
| fixed_32 | decode_short_c4 | 329.293 | 9.306 | 388.313 |
| fixed_32 | decode_long_prompt_c4 | 495.059 | 9.985 | 257.795 |
| fixed_32 | mixed_c4 | 425.872 | 10.218 | 264.045 |
| fixed_64 | decode_short_c1 | 252.605 | 7.596 | 126.693 |
| fixed_64 | decode_short_c4 | 332.171 | 9.362 | 385.397 |
| fixed_64 | decode_long_prompt_c4 | 490.611 | 9.850 | 260.562 |
| fixed_64 | mixed_c4 | 426.081 | 10.233 | 264.312 |
| hetero_8_64 | decode_short_c1 | 279.642 | 8.396 | 114.421 |
| hetero_8_64 | decode_short_c4 | 337.017 | 9.505 | 379.563 |
| hetero_8_64 | decode_long_prompt_c4 | 493.437 | 9.911 | 258.069 |
| hetero_8_64 | mixed_c4 | 422.139 | 10.096 | 266.101 |

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
