# 优秀实践

- [FlashAttention算子性能调优案例](flash_attention_tuning.md)：本案例中的算子FlashAttentionScoreGrad，用于训练场景下计算注意力的反向输出，即FlashAttentionScore算子的反向计算。
- [GroupedMatmul算子性能调优案例](grouped_matmul_tuning.md)：本案例对分组Matmul即GroupedMatmul算子的per-token量化场景进行性能分析和优化，GroupedMatmul算子计算过程（通过python代码表达）为：
- [matmul tuning cases](matmul_tuning_cases/index.md)：共 14 张知识卡（如 AIV核上的ND2NZ格式转换、Matmul高阶API开启IBShare模板共享A和B矩阵数据、Matmul高阶API开启IBShare模板共享B矩阵数据 等）。
- [MC²算子性能调优案例](mc_operator_tuning.md)：MC2通算融合算子性能调优案例，通过数据切分实现计算与通信流水掩盖，介绍切分策略制定与优化方法。
- [SIMT实现InsertHashTable算子案例](simt_insert_hash_table.md)：本案例介绍InsertHashTable算子，展示基于SIMT实现的大规模线程并发访问Global Memory内存的算子实践。
