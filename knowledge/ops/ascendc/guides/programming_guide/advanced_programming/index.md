# 高级编程

- [aclnn operator development](aclnn_operator_development/index.md)：本文是基础内容，说明Aclnn算子工程化开发这组文档的范围、阅读顺序和前置条件。建议先确认环境满足要求，再按快速入门跑通AddCustom示例，跑通后回到对应章节逐步补齐细节。（共 25 张知识卡：Aclnn算子工程化开发快速入门 等）
- [高级AI-Core编程模型](advanced_ai_core_programming_model/index.md)：共 5 张知识卡，分布在 1 个子目录（simd_simt_hybrid_programming）。
- [ai framework adaptation](ai_framework_adaptation/index.md)：本章节内容介绍AI框架调用自定义算子的方法。如下图所示，Pytorch支持单算子和图模式两种，TensorFlow仅支持图模式。（共 5 张知识卡：PyTorch框架、TensorFlow框架 等）
- [AOT编译优化（高级编程）](aot_compilation_optimization.md)：AOT（Ahead-of-Time，提前编译）编译优化是一种将运行时参数提前到编译阶段常量化的性能优化技术。通过在编译阶段对Kernel进行额外的常量化编译，在运行时根据Tiling参数自动匹配预编译的特化版本，让编译器能够进行更深度的优化（如循环展开、死代码消除、常量传播等），从而获得超越通用版本的执行性能。
- [硬件实现](hardware_implementation/index.md)：共 8 张知识卡（如 基本架构（硬件实现） 等）。
- [内存模型](memory_model/index.md)：共 2 张知识卡（如 缓存一致性、内存一致性）。
- [operator graph development](operator_graph_development/index.md)：图模式是神经网络模型的一种运行模式，在图模式下用户首先将模型的计算过程构造成一张图，然后通过GE将图下发到昇腾硬件执行。相对于单个算子依次下发的方式，图模式下，GE可以通过计算图优化、多流并行、内存复用、模型下沉等技术手段，加速模型执行效率，减少模型内存占用。（共 4 张知识卡：基本开发流程、开启Tiling下沉、图编译和图执行）
- [SIMT协作组简介（高级编程）](simt_cooperative_groups.md)：协作组（cooperative_groups）是Ascend C SIMT编程模型的扩展，用于组织协作线程组。协作组使开发者能够控制线程协作的粒度，实现更丰富、更高效的并行处理逻辑。
- [super kernel](super_kernel/index.md)：共 4 张知识卡（如 核函数直调算子额外适配说明、算子适配说明、算子自验证说明 等）。

- [advanced kernel programming](advanced_kernel_programming/index.md)：共 1 张知识卡，分布在 1 个子目录（intra_core_sync）。
- [inter core communication](inter_core_communication/index.md)：本文介绍SSBuffer的存储范围、地址空间、访问方式以及配合核间同步进行消息交换的编程方法。消息队列的具体设计、Host侧校验和完整算子实现请参见基于SSBuffer的核间通信特性。
