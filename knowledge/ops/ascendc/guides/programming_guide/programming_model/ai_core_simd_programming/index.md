# AI-Core-SIMD编程

- [抽象硬件架构（AI Core SIMD编程）](abstract_hardware_architecture.md)：AI Core是AI处理器的计算核心，AI处理器内部有多个AI Core。本章节将介绍AI Core的并行计算架构抽象，该抽象架构屏蔽了不同硬件之间的差异。使用Ascend C进行编程时，基于抽象硬件架构，可以简化硬件细节，显著降低开发门槛。如需了解更详细的硬件架构信息或者原理，请参考硬件实现。
- [c pointer programming](c_pointer_programming/index.md)：共 4 张知识卡（如 C语言编程概述、Cube矩阵计算编程、Memory矢量计算编程 等）。
- [cpp tensor programming](cpp_tensor_programming/index.md)：共 5 张知识卡（如 CppTensorProgrammingOverview、Cube矩阵计算编程、MemoryVectorComputation 等）。
- [核函数（AI Core SIMD编程）](kernel_function.md)：核函数（Kernel Function）是在设备侧并行执行的C/C++函数。它由主机端代码调用，调用时被实例化为多份并行副本，分别调度到多个AI Core上运行（每个AI Core被抽象为一个Block）。这些Block利用AI Core的计算能力，协同完成某一算子的计算。
- [AI Core SIMD编程模型概述](overview.md)：SIMD并行机制是AI Core的核心算力支撑，承担了整机90%以上的计算吞吐量。因此，本章将以SIMD编程模型为核心主线，遵循「宏观架构—硬件调度—代码落地」的递进逻辑，完整覆盖SIMD算子开发全链路：阐述多AI Core集群的并行任务分发机制，拆解单AI Core内部异构计算单元的任务调度逻辑，并逐层讲解如何基于差异化的C/C++编程接口，完成算子Kernel计算逻辑的开发。
- [tpipe tque programming](tpipe_tque_programming/index.md)：共 2 张知识卡（如 TPipe和TQue编程范式、TPipe-TQue编程原理介绍）。
