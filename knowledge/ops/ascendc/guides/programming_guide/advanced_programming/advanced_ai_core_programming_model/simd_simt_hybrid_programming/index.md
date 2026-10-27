# simd simt hybrid programming

- [抽象硬件架构（SIMD与SIMT混合编程）](abstract_hardware_architecture.md)：Ascend C支持SIMD与SIMT混合编程，实现向量级并行与线程级并行的高效协同。AI Core包含AIC和AIV，其中AIV核支持SIMT、SIMD两种工作模式，因此可以实现AIV内SIMD与SIMT的混合编程，也支持AIV的SIMT与AIC的SIMD融合编程，整体硬件架构如下图所示：
- [核函数与VF函数（SIMD与SIMT混合编程）](kernel_and_vf_functions.md)：SIMD与SIMT混合编程涉及多种类型的函数，它们之间遵循严格的调用关系和层级约束。本章节将介绍整体函数类型和调用关系，再详细展开每种函数的定义、调用语法及约束。
- [内存层级（SIMD与SIMT混合编程）](memory_hierarchy.md)：在SIMD与SIMT混合编程场景下，整体内存资源如下图所示。核外的Global Memory是所有核的全局内存，容量最大，但访问效率最低。AIC的L1和AIV的UB是单核内的共享内存，容量较小，但访问效率较高。私有内存层最靠近计算单元，容量最小，访问效率最高。
- [SIMD与SIMT混合编程概述](overview.md)：为提升向量计算在复杂控制流、离散访存等场景下的编程灵活性，在SIMD与SIMT深度融合的混合编程架构中：矩阵计算单元延续SIMD设计，向量计算单元则在SIMD基础上引入SIMT能力。这一设计确立了以SIMD为主、SIMT为辅的混合编程模型。
- [编程示例（SIMD与SIMT混合编程）](programming_examples.md)：本节以gather & adds混合编程样例为例，展示SIMD与SIMT混合编程的关键语法。
