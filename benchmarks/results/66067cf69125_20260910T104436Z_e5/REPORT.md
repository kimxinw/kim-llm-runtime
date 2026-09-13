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
| decode_short_c1 | -0.48% | -0.46% | +2.54% |
| decode_long_prompt_c1 | -17.13% | -20.41% | +20.73% |
| decode_short_c2 | -3.04% | -3.10% | +3.20% |
| decode_long_prompt_c2 | -14.56% | -17.88% | +15.89% |
| decode_short_c4 | +0.53% | +0.73% | -0.57% |
| decode_long_prompt_c4 | -11.85% | -17.72% | +13.42% |
| mixed_c4 | -3.79% | -11.36% | +4.63% |

## 绝对结果

| Variant | Workload | E2E p50 (ms) | TPOT p50 (ms) | Output tokens/s |
|---|---|---:|---:|---:|
| fixed_8 | decode_short_c1 | 317.914 | 8.981 | 98.585 |
| fixed_8 | decode_short_c4 | 496.317 | 11.278 | 257.891 |
| fixed_8 | decode_long_prompt_c4 | 1556.971 | 19.182 | 82.222 |
| fixed_8 | mixed_c4 | 1143.995 | 18.178 | 97.511 |
| fixed_16 | decode_short_c1 | 310.782 | 8.771 | 102.767 |
| fixed_16 | decode_short_c4 | 495.900 | 11.450 | 252.794 |
| fixed_16 | decode_long_prompt_c4 | 1435.680 | 17.232 | 89.190 |
| fixed_16 | mixed_c4 | 1033.808 | 15.983 | 105.983 |
| fixed_32 | decode_short_c1 | 309.000 | 8.698 | 103.587 |
| fixed_32 | decode_short_c4 | 486.696 | 11.018 | 262.742 |
| fixed_32 | decode_long_prompt_c4 | 1370.966 | 16.179 | 93.371 |
| fixed_32 | mixed_c4 | 1088.401 | 16.341 | 103.276 |
| fixed_64 | decode_short_c1 | 310.971 | 8.749 | 102.889 |
| fixed_64 | decode_short_c4 | 485.257 | 10.946 | 264.026 |
| fixed_64 | decode_long_prompt_c4 | 1341.937 | 15.714 | 95.379 |
| fixed_64 | mixed_c4 | 978.630 | 14.521 | 112.144 |
| hetero_8_64 | decode_short_c1 | 316.403 | 8.940 | 101.090 |
| hetero_8_64 | decode_short_c4 | 498.947 | 11.360 | 256.420 |
| hetero_8_64 | decode_long_prompt_c4 | 1372.421 | 15.784 | 93.260 |
| hetero_8_64 | mixed_c4 | 1100.593 | 16.114 | 102.025 |

## Batch 执行证据

| Workload | Model Batch Size | Attention Batch Size | Attention Submissions/Run |
|---|---:|---:|---:|
| decode_short_c1 | 1.80 | 0.00 | 0 |
| decode_short_c2 | 3.23 | 2.00 | 682 |
| decode_short_c4 | 5.36 | 4.00 | 682 |
| decode_long_prompt_c4 | 6.69 | 4.00 | 682 |
| mixed_c4 | 4.67 | 3.23 | 682 |

## Capacity 与故障隔离

| Variant | Capacity 完成/失败/拒绝 | Fault 完成/失败 | Peak fragmentation tokens |
|---|---:|---:|---:|
| fixed_8 | 48/0/3 | 9/3 | 0 |
| fixed_16 | 48/0/3 | 9/3 | 128 |
| fixed_32 | 48/0/3 | 9/3 | 384 |
| fixed_64 | 24/24/3 | 9/3 | 448 |
| hetero_8_64 | 30/18/3 | 9/3 | 0 |

## 边界说明

- Hetero 的 Extent Page 分配次数为 `18`。大于零表示 Generation 自动 Promotion 已在本轮 E2E Workload 中生效，长序列实际使用了 Extent Page。
- Dense GEMM 已按总 Token 数动态 Batch 执行；单 Token 多请求使用 Batch KV Write 与 Paged Decode Attention。多 Token Chunk 已走真实因果 Prefill，但包含多 Token Chunk 的混合 Batch 仍按请求提交 KV/Attention。
- Prefill Attention 的 Scores 与 Softmax/Value Output 仍为两个 Kernel，Score Workspace 随 `chunk_tokens × query_heads × sequence_tokens` 增长。
- Microbenchmark 与本报告的 E2E 结果分开；K6 的 Gather/Promotion 收益不能直接替代模型端到端收益。
- Nsight Systems GPU Activity Timeline 受当前 WSL2/CUPTI 环境限制，本报告不以 CUDA API Duration 冒充 Kernel Timeline。
- 未与 vLLM 或 TensorRT-LLM 比较峰值性能。
