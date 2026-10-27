# SIMD与SIMT混合算子性能优化

- [compute optimization](compute_optimization/index.md)：>该性能优化建议适用于如下型号：
- [execution control](execution_control/index.md)：>该性能优化建议适用于如下型号：
- [instruction optimization](instruction_optimization/index.md)：>该性能优化建议适用于如下型号：
- [memory access](memory_access/index.md)：>该性能优化建议适用于如下型号：
- [SIMD与SIMT混合算子性能优化概述](overview.md)：AI Core上SIMD与SIMT混合编程结合了SIMD多数据并行计算能力与SIMT离散访存、复杂分支处理能力，可以在同一算子中以Vector Function（VF）为基本调度单位灵活组织SIMD和SIMT执行逻辑。SIMD编程适合处理连续、规整的向量或矩阵计算，SIMT编程适合处理分支判断、索引映射、离散访存等不规则逻辑，在程序中结合使用两种编程方式，可以兼顾硬件高吞吐和编程灵活性。
