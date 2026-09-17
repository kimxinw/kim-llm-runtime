# Reference/Fused TinyLlama E5 正式矩阵

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

## Fused 相对 Reference

延迟差值为负表示 Fused 更快；吞吐差值为正表示 Fused 更快。

| Page 策略 | Workload | E2E p50（Reference → Fused） | E2E 差值 | TPOT 差值 | Output tokens/s 差值 |
|---|---|---:|---:|---:|---:|
| fixed_8 | decode_short_c1 | 326.470 → 279.281 ms | -14.45% | -13.87% | +16.99% |
| fixed_8 | decode_long_prompt_c1 | 597.242 → 341.479 ms | -42.82% | -38.61% | +74.72% |
| fixed_8 | decode_short_c2 | 345.959 → 289.295 ms | -16.38% | -14.95% | +19.49% |
| fixed_8 | decode_long_prompt_c2 | 826.787 → 389.096 ms | -52.94% | -44.14% | +112.63% |
| fixed_8 | decode_short_c4 | 398.733 → 309.224 ms | -22.45% | -20.73% | +29.02% |
| fixed_8 | decode_long_prompt_c4 | 1117.905 → 484.254 ms | -56.68% | -47.66% | +130.94% |
| fixed_8 | mixed_c4 | 781.717 → 411.670 ms | -47.34% | -42.76% | +87.83% |
| fixed_16 | decode_short_c1 | 293.764 → 254.282 ms | -13.44% | -12.92% | +15.48% |
| fixed_16 | decode_long_prompt_c1 | 538.897 → 329.852 ms | -38.79% | -33.96% | +63.34% |
| fixed_16 | decode_short_c2 | 338.862 → 290.200 ms | -14.36% | -12.71% | +16.93% |
| fixed_16 | decode_long_prompt_c2 | 687.898 → 387.693 ms | -43.64% | -32.51% | +75.13% |
| fixed_16 | decode_short_c4 | 391.824 → 325.727 ms | -16.87% | -13.75% | +19.84% |
| fixed_16 | decode_long_prompt_c4 | 1002.192 → 483.304 ms | -51.78% | -41.68% | +107.46% |
| fixed_16 | mixed_c4 | 714.325 → 412.551 ms | -42.25% | -36.96% | +70.83% |
| fixed_32 | decode_short_c1 | 318.894 → 264.848 ms | -16.95% | -16.32% | +20.39% |
| fixed_32 | decode_long_prompt_c1 | 558.186 → 339.236 ms | -39.23% | -34.34% | +64.61% |
| fixed_32 | decode_short_c2 | 367.704 → 299.301 ms | -18.60% | -17.09% | +22.91% |
| fixed_32 | decode_long_prompt_c2 | 709.918 → 381.859 ms | -46.21% | -36.28% | +85.99% |
| fixed_32 | decode_short_c4 | 423.589 → 315.699 ms | -25.47% | -27.04% | +37.37% |
| fixed_32 | decode_long_prompt_c4 | 1036.953 → 471.432 ms | -54.54% | -45.26% | +120.61% |
| fixed_32 | mixed_c4 | 744.847 → 405.436 ms | -45.57% | -39.93% | +80.95% |
| fixed_64 | decode_short_c1 | 290.084 → 266.802 ms | -8.03% | -7.41% | +8.94% |
| fixed_64 | decode_long_prompt_c1 | 497.067 → 330.121 ms | -33.59% | -28.26% | +50.46% |
| fixed_64 | decode_short_c2 | 365.674 → 296.043 ms | -19.04% | -17.49% | +23.68% |
| fixed_64 | decode_long_prompt_c2 | 686.144 → 379.007 ms | -44.76% | -34.51% | +81.21% |
| fixed_64 | decode_short_c4 | 419.372 → 314.274 ms | -25.06% | -23.09% | +33.32% |
| fixed_64 | decode_long_prompt_c4 | 1008.630 → 475.552 ms | -52.85% | -42.83% | +112.42% |
| fixed_64 | mixed_c4 | 735.331 → 412.465 ms | -43.91% | -38.25% | +75.66% |
| hetero_8_64 | decode_short_c1 | 299.161 → 261.654 ms | -12.54% | -12.15% | +14.41% |
| hetero_8_64 | decode_long_prompt_c1 | 504.663 → 331.371 ms | -34.34% | -28.41% | +52.37% |
| hetero_8_64 | decode_short_c2 | 345.236 → 297.388 ms | -13.86% | -12.19% | +16.09% |
| hetero_8_64 | decode_long_prompt_c2 | 647.503 → 380.691 ms | -41.21% | -29.40% | +70.21% |
| hetero_8_64 | decode_short_c4 | 399.486 → 323.752 ms | -18.96% | -17.11% | +23.72% |
| hetero_8_64 | decode_long_prompt_c4 | 1027.283 → 481.483 ms | -53.13% | -42.13% | +113.61% |
| hetero_8_64 | mixed_c4 | 746.491 → 412.985 ms | -44.68% | -38.72% | +77.28% |

## 结论边界

- Reference/Fused 仅改变 Attention Fusion 编译开关；模型、请求、分页策略、容量、Warmup、迭代数和计时边界保持一致。
- 每个实现包含五种分页策略和九类 Case；性能比较排除故障注入与容量 Case。
- 本结果仅适用于记录的 TinyLlama FP16、RTX 3060 和当前 CUDA/Driver 环境，不能外推到其他模型或硬件。
- Fusion 收益与 Heterogeneous/Fixed Page 策略收益分别统计，不能相互替代。
