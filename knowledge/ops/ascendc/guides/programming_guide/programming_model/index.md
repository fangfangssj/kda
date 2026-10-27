# 编程模型

- [AI-Core-SIMD编程](ai_core_simd_programming/index.md)：SIMD并行机制是AI Core的核心算力支撑，承担了整机90%以上的计算吞吐量。因此，本章将以SIMD编程模型为核心主线，遵循「宏观架构—硬件调度—代码落地」的递进逻辑，完整覆盖SIMD算子开发全链路：阐述多AI Core集群的并行任务分发机制，拆解单AI Core内部异构计算单元的任务调度逻辑，并逐层讲解如何基于差异化的C/C++编程接口，完成算子Kernel计算逻辑的开发。（共 14 张知识卡：抽象硬件架构（AI Core SIMD编程）、核函数（AI Core SIMD编程） 等）
- [AI-Core-SIMT编程](ai_core_simt_programming/index.md)：SIMT编程模型适用于离散数据访问和复杂控制流场景，基于Grid-Block-Thread三级线程层级在AIV核上实现线程级并行计算。（共 8 张知识卡：抽象硬件架构（AI Core SIMT编程）、原子操作（AI Core SIMT编程）、核函数（AI Core SIMT编程） 等）
- [AI CPU编程（编程模型）](ai_cpu_programming.md)：AI CPU是位于Device侧ARM64架构的处理器，其具备与AI Core相同的内存访问能力，可直接访问Device侧内存资源；也可以与Host侧的CPU一样，进行类似的数据计算，通常作为AI Core的补充，主要承担非矩阵类、逻辑比较复杂的分支密集型计算。AI CPU的运行环境为基础的Linux环境，编程时可使用libc库，C++标准库，STL模板库等。其硬件架构图如下所示：
- [异构系统（编程模型）](heterogeneous_system.md)：一个基于昇腾处理器的异构系统通常包含CPU与昇腾NPU。其中，CPU及其内存称为Host与Host Memory；NPU及其内存称为Device与Device Memory。
- [编程模型概述](programming_model_overview.md)：介绍Ascend C编程模型，涵盖SIMD与SIMT并行执行模型、AI Core硬件基础及编程路径选择。
