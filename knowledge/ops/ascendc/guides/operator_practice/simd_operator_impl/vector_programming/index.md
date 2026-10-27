# 矢量编程

- [基础矢量算子](basic_vector_operator.md)：基于Ascend C方式实现基础矢量算子核函数的流程如下图所示。
- [Broadcast场景](broadcast_scenario.md)：在某些场景下，可能会存在两个输入shape不相同的情况。由于Add接口只支持对shape相同的输入进行计算，因此需要先对输入进行shape变换，再进行Add计算。本节将对满足Broadcast条件的输入在算子实现中的Broadcast处理进行介绍，其他场景可以参考本章节中提供的思路。
- [DoubleBuffer场景](double_buffer_scenario.md)：因存在算子中多次搬入搬出数据的场景，为充分利用硬件资源，实现多流水并行，引入DoubleBuffer机制。DoubleBuffer是通过将输入数据分成大小相等的两块，充分利用AI Core的硬件资源，实现数据搬入、计算、数据搬出的并行执行方式。下面以“核间不均分，核内不均分”的样例为例，介绍算子中DoubleBuffer的实现。
- [multi core tiling](multi_core_tiling/index.md)：Ascend C核函数是在Device侧执行的并行函数。为了提高算子的执行效率，通常采用多核并行计算，将输入数据切分后分配到不同的逻辑核上处理；同时，由于单个核的Local Memory容量有限，无法一次完整容纳算子的输入和输出数据，需要分批搬运数据进行计算。这种对输入数据进行切分、分块计算的过程称为Tiling。（共 5 张知识卡：SIMD算子实现主块均分、SIMD算子实现尾块均分、TailCoreSplit 等）
- [SIMD算子实现概述](overview.md)：本节将以Add算子为例，带您快速构建Ascend C矢量算子程序，并学习矢量算子开发的典型场景以及处理方式。涉及的场景包括：
- [TBuf的使用](tbuf_usage.md)：在大多数算子开发时，核函数计算过程需要使用临时内存来存储运算的中间结果，这些中间结果以临时变量表示，临时变量占用的内存可以使用TBuf数据结构来管理，具体介绍请参考TBuf。下文将以输入的数据类型为bfloat16_t、在单核上运行的Add算子为例，介绍TBuf的使用方式。本文中介绍的算子完整代码请参见使用临时内存的Add算子样例。
- [非对齐场景](unaligned_scenario.md)：本节介绍非32字节对齐数据的更多处理方式，包括数据搬入、计算和搬出的处理。用户在实际算子开发中，可以参考如下方案介绍和算子样例灵活地处理非对齐场景。
