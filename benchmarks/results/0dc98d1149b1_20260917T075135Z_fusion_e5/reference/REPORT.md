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
| decode_short_c1 | -8.36% | -8.34% | +9.15% |
| decode_long_prompt_c1 | -15.50% | -17.31% | +18.21% |
| decode_short_c2 | -0.21% | -0.27% | +0.34% |
| decode_long_prompt_c2 | -21.68% | -22.43% | +27.55% |
| decode_short_c4 | +0.19% | +0.12% | -0.13% |
| decode_long_prompt_c4 | -8.11% | -10.38% | +8.72% |
| mixed_c4 | -4.51% | -7.77% | +5.72% |

## 绝对结果

| Variant | Workload | E2E p50 (ms) | TPOT p50 (ms) | Output tokens/s |
|---|---|---:|---:|---:|
| fixed_8 | decode_short_c1 | 326.470 | 9.751 | 97.960 |
| fixed_8 | decode_short_c4 | 398.733 | 10.981 | 320.831 |
| fixed_8 | decode_long_prompt_c4 | 1117.905 | 18.382 | 114.526 |
| fixed_8 | mixed_c4 | 781.717 | 17.404 | 145.254 |
| fixed_16 | decode_short_c1 | 293.764 | 8.772 | 108.899 |
| fixed_16 | decode_short_c4 | 391.824 | 10.768 | 326.866 |
| fixed_16 | decode_long_prompt_c4 | 1002.192 | 16.517 | 127.738 |
| fixed_16 | mixed_c4 | 714.325 | 15.742 | 159.364 |
| fixed_32 | decode_short_c1 | 318.894 | 9.518 | 100.339 |
| fixed_32 | decode_short_c4 | 423.589 | 12.197 | 295.004 |
| fixed_32 | decode_long_prompt_c4 | 1036.953 | 17.195 | 123.074 |
| fixed_32 | mixed_c4 | 744.847 | 16.302 | 152.999 |
| fixed_64 | decode_short_c1 | 290.084 | 8.658 | 110.228 |
| fixed_64 | decode_short_c4 | 419.372 | 11.503 | 305.377 |
| fixed_64 | decode_long_prompt_c4 | 1008.630 | 16.698 | 127.429 |
| fixed_64 | mixed_c4 | 735.331 | 16.048 | 155.033 |
| hetero_8_64 | decode_short_c1 | 299.161 | 8.938 | 106.927 |
| hetero_8_64 | decode_short_c4 | 399.486 | 10.995 | 320.411 |
| hetero_8_64 | decode_long_prompt_c4 | 1027.283 | 16.473 | 124.512 |
| hetero_8_64 | mixed_c4 | 746.491 | 16.052 | 153.559 |

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
