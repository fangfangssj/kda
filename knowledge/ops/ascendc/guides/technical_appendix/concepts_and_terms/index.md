# concepts and terms

- [技术附录术语表](glossary.md)：AscendC::TPosition::A1代表设备上用于矩阵计算的逻辑内存，用于存放左矩阵，物理存储对应AI Core的L1 Buffer。
- [memory access](memory_access/index.md)：Scalar计算单元仅支持读写Global Memory和Unified Buffer，介绍其读写方式及同步机制。
- [neural networks and operators](neural_networks_and_operators/index.md)：共 2 张知识卡（如 数据排布格式、算子基本概念）。
- [performance optimization](performance_optimization/index.md)：执行于AI Core上的指令队列主要包括如下几类，即Vector指令队列、Cube指令队列和MTE指令队列。不同指令队列间的相互独立性和可并行执行性，是DoubleBuffer优化机制的基石。
