# Fused (No Cursor)/Fused TinyLlama E5 正式矩阵

## 验证结论

| 项目 | 结果 |
|---|---|
| 两套实现绑定同一提交 | PASS |
| 配置、模型与 Storage Budget 一致 | PASS |
| Case 集合一致 | PASS |
| 跨实现 Token 一致 | PASS（20 个策略/Case） |
| 故障与容量结果一致 | 不适用（该套件不含故障注入与容量 Case） |
| Transformers FP16 Reference | PASS（10 个唯一 Prompt） |
| 测量 Run 总数 | 120 |

## Fused 相对 Fused (No Cursor)

延迟差值为负表示 Fused 更快；吞吐差值为正表示 Fused 更快。

| Page 策略 | Workload | E2E p50（Fused (No Cursor) → Fused） | E2E 差值 | TPOT 差值 | Output tokens/s 差值 |
|---|---|---:|---:|---:|---:|
| fixed_8 | long_prompt512_c4 | 2478.940 → 1348.621 ms | -45.60% | -41.63% | +83.73% |
| fixed_8 | long_prompt1024_c1 | 4109.574 → 1343.652 ms | -67.30% | -71.34% | +205.91% |
| fixed_8 | long_prompt1024_c4 | 11576.891 → 3237.880 ms | -72.03% | -69.51% | +260.19% |
| fixed_8 | long_mixed1024_c4 | 1350.307 → 892.029 ms | -33.94% | -65.60% | +208.40% |
| fixed_16 | long_prompt512_c4 | 1917.875 → 1346.361 ms | -29.80% | -26.65% | +43.51% |
| fixed_16 | long_prompt1024_c1 | 2676.775 → 1315.946 ms | -50.84% | -55.40% | +105.55% |
| fixed_16 | long_prompt1024_c4 | 7567.085 → 3172.512 ms | -58.07% | -58.77% | +140.31% |
| fixed_16 | long_mixed1024_c4 | 1139.395 → 897.077 ms | -21.27% | -53.19% | +111.68% |
| fixed_32 | long_prompt512_c4 | 1684.777 → 1305.640 ms | -22.50% | -24.45% | +31.02% |
| fixed_32 | long_prompt1024_c1 | 1883.740 → 1261.783 ms | -33.02% | -39.93% | +54.04% |
| fixed_32 | long_prompt1024_c4 | 5568.158 → 3105.268 ms | -44.23% | -39.92% | +78.08% |
| fixed_32 | long_mixed1024_c4 | 1014.790 → 892.484 ms | -12.05% | -37.25% | +58.06% |
| fixed_64 | long_prompt512_c4 | 1501.336 → 1296.207 ms | -13.66% | -11.26% | +16.16% |
| fixed_64 | long_prompt1024_c1 | 1562.368 → 1279.601 ms | -18.10% | -20.58% | +24.01% |
| fixed_64 | long_prompt1024_c4 | 4466.933 → 3129.803 ms | -29.93% | -26.71% | +42.45% |
| fixed_64 | long_mixed1024_c4 | 948.928 → 890.132 ms | -6.20% | -24.33% | +29.77% |
| hetero_8_64 | long_prompt512_c4 | 1509.929 → 1318.800 ms | -12.66% | -8.13% | +15.03% |
| hetero_8_64 | long_prompt1024_c1 | 1698.726 → 1301.390 ms | -23.39% | -25.06% | +29.93% |
| hetero_8_64 | long_prompt1024_c4 | 4513.469 → 3153.543 ms | -30.13% | -24.80% | +41.67% |
| hetero_8_64 | long_mixed1024_c4 | 958.633 → 883.362 ms | -7.85% | -26.06% | +32.39% |

## 结论边界

- Fused (No Cursor)（Fusion ON / Cursor OFF）与 Fused（Fusion ON / Cursor ON）仅编译开关不同；模型、请求、分页策略、容量、Warmup、迭代数和计时边界保持一致。
- 每个实现包含五种分页策略和 4 类 Case（long 长上下文套件）；性能比较排除故障注入与容量 Case。
- 本结果仅适用于记录的 TinyLlama FP16、RTX 3060 和当前 CUDA/Driver 环境，不能外推到其他模型或硬件。
- 实现间收益与 Heterogeneous/Fixed Page 策略收益分别统计，不能相互替代。
