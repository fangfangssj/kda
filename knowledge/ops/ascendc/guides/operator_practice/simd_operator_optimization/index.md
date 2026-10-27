# SIMD算子性能优化

- [矩阵计算](matrix_compute/index.md)：共 6 张知识卡（如 通过BT Buffer实现高效的bias计算、通过FP Buffer存放量化参数实现高效随路量化、使用hifloat8_t实现Matmul性能优化 等）。
- [内存访问](memory_access/index.md)：共 14 张知识卡（如 避免同地址访问、EfficientTransferApi、GM地址尽量512B对齐 等）。
- [优化建议总览表（SIMD算子优化）](optimization_summary.md)：汇总 SIMD 算子的 Tiling、启动开销、流水编排、内存访问、矢量计算和矩阵计算优化建议。
- [头尾开销优化](overhead_optimization/index.md)：共 5 张知识卡（如 避免TPipe在对象内创建和初始化、设置合适的核数和算子Kernel类型、设置DCI编译选项来减少算子尾开销 等）。
- [流水编排](pipeline_scheduling/index.md)：共 2 张知识卡（如 开启DoubleBuffer、启用Iterate或IterateAll异步接口避免AIC/AIV同步依赖）。
- [Tiling策略](tiling_strategy/index.md)：通过调整拖尾核位置实现核间负载均衡，避免部分核计算拖尾时其他核空闲，优化多核整体性能。
- [矢量计算](vector_compute/index.md)：共 8 张知识卡（如 选择低延迟指令，优化归约操作性能、基于全局掩码复用的计算性能优化、通过Unified Buffer融合实现连续vector计算 等）。
