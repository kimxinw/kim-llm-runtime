# Fused/Fused+Cursor TinyLlama E5 正式矩阵

## 验证结论

| 项目 | 结果 |
|---|---|
| 两套实现绑定同一提交 | PASS |
| 配置、模型与 Storage Budget 一致 | PASS |
| Case 集合一致 | PASS |
| 跨实现 Token 一致 | PASS（35 个策略/Case） |
| 故障与容量结果一致 | PASS（10 个策略/Case） |
| Transformers FP16 Reference | PASS（8 个唯一 Prompt） |
| 测量 Run 总数 | 270 |

## Fused+Cursor 相对 Fused

延迟差值为负表示 Fused+Cursor 更快；吞吐差值为正表示 Fused+Cursor 更快。

| Page 策略 | Workload | E2E p50（Fused → Fused+Cursor） | E2E 差值 | TPOT 差值 | Output tokens/s 差值 |
|---|---|---:|---:|---:|---:|
| fixed_8 | decode_short_c1 | 277.261 → 279.381 ms | +0.76% | +0.91% | -0.52% |
| fixed_8 | decode_long_prompt_c1 | 372.021 → 332.692 ms | -10.57% | -10.94% | +11.88% |
| fixed_8 | decode_short_c2 | 318.889 → 299.655 ms | -6.03% | -6.01% | +6.29% |
| fixed_8 | decode_long_prompt_c2 | 418.052 → 397.016 ms | -5.03% | -3.65% | +4.66% |
| fixed_8 | decode_short_c4 | 334.875 → 335.008 ms | +0.04% | +0.36% | -0.16% |
| fixed_8 | decode_long_prompt_c4 | 525.200 → 490.560 ms | -6.60% | -5.94% | +5.97% |
| fixed_8 | mixed_c4 | 438.903 → 422.328 ms | -3.78% | -6.39% | +0.21% |
| fixed_16 | decode_short_c1 | 275.129 → 275.053 ms | -0.03% | -0.23% | +0.57% |
| fixed_16 | decode_long_prompt_c1 | 357.778 → 345.953 ms | -3.31% | -4.37% | +3.55% |
| fixed_16 | decode_short_c2 | 315.707 → 313.979 ms | -0.55% | -0.62% | +0.66% |
| fixed_16 | decode_long_prompt_c2 | 408.814 → 398.028 ms | -2.64% | -1.97% | +0.63% |
| fixed_16 | decode_short_c4 | 308.428 → 332.239 ms | +7.72% | +7.49% | -7.20% |
| fixed_16 | decode_long_prompt_c4 | 509.494 → 486.698 ms | -4.47% | -4.12% | +4.74% |
| fixed_16 | mixed_c4 | 436.569 → 421.551 ms | -3.44% | -3.53% | +3.67% |
| fixed_32 | decode_short_c1 | 276.352 → 277.625 ms | +0.46% | +0.50% | -0.39% |
| fixed_32 | decode_long_prompt_c1 | 348.540 → 346.013 ms | -0.72% | -0.96% | +0.79% |
| fixed_32 | decode_short_c2 | 309.509 → 310.267 ms | +0.24% | +0.40% | -0.23% |
| fixed_32 | decode_long_prompt_c2 | 397.715 → 394.411 ms | -0.83% | -0.75% | +0.67% |
| fixed_32 | decode_short_c4 | 329.293 → 331.822 ms | +0.77% | +0.47% | -0.61% |
| fixed_32 | decode_long_prompt_c4 | 495.059 → 489.613 ms | -1.10% | -1.36% | +1.95% |
| fixed_32 | mixed_c4 | 425.872 → 405.394 ms | -4.81% | -4.79% | +5.00% |
| fixed_64 | decode_short_c1 | 252.605 → 273.195 ms | +8.15% | +8.01% | -7.42% |
| fixed_64 | decode_long_prompt_c1 | 322.208 → 343.610 ms | +6.64% | +6.58% | -6.24% |
| fixed_64 | decode_short_c2 | 288.421 → 314.297 ms | +8.97% | +9.04% | -8.23% |
| fixed_64 | decode_long_prompt_c2 | 400.308 → 398.820 ms | -0.37% | +0.17% | +1.47% |
| fixed_64 | decode_short_c4 | 332.171 → 316.797 ms | -4.63% | -4.69% | +4.92% |
| fixed_64 | decode_long_prompt_c4 | 490.611 → 491.130 ms | +0.11% | +0.21% | +1.21% |
| fixed_64 | mixed_c4 | 426.081 → 425.332 ms | -0.18% | -0.36% | +0.05% |
| hetero_8_64 | decode_short_c1 | 279.642 → 276.598 ms | -1.09% | -0.97% | +1.20% |
| hetero_8_64 | decode_long_prompt_c1 | 324.433 → 348.527 ms | +7.43% | +7.71% | -5.82% |
| hetero_8_64 | decode_short_c2 | 290.338 → 294.628 ms | +1.48% | +1.40% | -1.45% |
| hetero_8_64 | decode_long_prompt_c2 | 404.584 → 399.087 ms | -1.36% | -0.84% | +2.71% |
| hetero_8_64 | decode_short_c4 | 337.017 → 335.176 ms | -0.55% | -0.68% | +0.58% |
| hetero_8_64 | decode_long_prompt_c4 | 493.437 → 492.472 ms | -0.20% | -0.73% | +0.72% |
| hetero_8_64 | mixed_c4 | 422.139 → 400.234 ms | -5.19% | -4.76% | +5.77% |

## 结论边界

- Fused（Fusion ON / Cursor OFF）与 Fused+Cursor（Fusion ON / Cursor ON）仅编译开关不同；模型、请求、分页策略、容量、Warmup、迭代数和计时边界保持一致。
- 每个实现包含五种分页策略和九类 Case；性能比较排除故障注入与容量 Case。
- 本结果仅适用于记录的 TinyLlama FP16、RTX 3060 和当前 CUDA/Driver 环境，不能外推到其他模型或硬件。
- 实现间收益与 Heterogeneous/Fixed Page 策略收益分别统计，不能相互替代。
