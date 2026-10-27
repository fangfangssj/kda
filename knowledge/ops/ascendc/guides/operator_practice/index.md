# operator practice

- [优秀实践](best_practices/index.md)：共 18 张知识卡（如 FlashAttention算子性能调优案例、GroupedMatmul算子性能调优案例、MC²算子性能调优案例 等）。
- [本文档组织结构（算子实践）](document_structure.md)：本文档主要用于指导开发者基于Ascend C编程语言在昇腾AI处理器上开发高性能的算子。在阅读此文档时，需要开发者具备如下能力：
- [功能调试](functional_debug/index.md)：共 3 张知识卡（如 精度正常（功能调试）、算子调试（功能调试）、运行正常（功能调试））。
- [Host Tiling 单元测试开发](host_tiling_ut.md)：面向 ACLNN 等注册式算子包的 Host Tiling 单元测试，涵盖独立 oracle、TilingContext 构造、平台资源 mock、注册调用、隔离构建，以及测试先行和已有实现的回归验证。
- [异构计算（算子实践）](heterogeneous_computing.md)：使用Ascend C进行编程时，会涉及到在两个不同的平台（Host、Device）上开发代码。本章简单介绍Host、Device之间的差异，便于开发者宏观地了解这个异构系统；同时给出算子相关的数据流，结合异构架构的特点，开发者可以进一步了解应该如何合理安排算子代码的执行位置，便于得到更好的性能。
- [性能分析](performance_analysis/index.md)：共 2 张知识卡（如 分析性能数据（性能分析）、获取性能数据（性能分析））。
- [SIMD算子实现](simd_operator_impl/index.md)：Ascend C的算子实现主要包含两个部分：（共 37 张知识卡，分布在 4 个子目录）
- [SIMD算子性能优化](simd_operator_optimization/index.md)：共 37 张知识卡（如 优化建议总览表（SIMD算子优化） 等）。
- [SIMD与SIMT混合算子实现](simd_simt_hybrid_operator_impl/index.md)：共 2 张知识卡（如 基础知识（SIMD与SIMT混合算子实现）、算子实现（SIMD与SIMT混合算子实现））。
- [SIMD与SIMT混合算子性能优化](simd_simt_hybrid_optimization/index.md)：AI Core上SIMD与SIMT混合编程结合了SIMD多数据并行计算能力与SIMT离散访存、复杂分支处理能力，可以在同一算子中以Vector Function（VF）为基本调度单位灵活组织SIMD和SIMT执行逻辑。SIMD编程适合处理连续、规整的向量或矩阵计算，SIMT编程适合处理分支判断、索引映射、离散访存等不规则逻辑，在程序中结合使用两种编程方式，可以兼顾硬件高吞吐和编程灵活性。（共 5 张知识卡，分布在 4 个子目录）
- [SIMT算子实现](simt_operator_impl/index.md)：共 2 张知识卡（如 基础知识（SIMT算子实现）、算子实现（SIMT算子实现））。
- [SIMT算子性能优化](simt_operator_optimization/index.md)：SIMT编程模式以线程（Thread）为基本编程单位，由开发者编写单个线程的执行逻辑，再通过>>配置将大量线程组织为线程块（线程块）与线程网格（Grid），调度到硬件的Vector Core（物理核，即AIV核）上以Warp为单位并行执行。（共 12 张知识卡，分布在 4 个子目录）
- [典型算子实践案例](typical_operator_cases/index.md)：共 12 张知识卡，分布在 1 个子目录（simd_operator_impl）。
