# 概览

- [数据搬运概念](data_move_concept.md)：数据搬运接口覆盖从连续复制到多维寻址、格式转换和量化激活等能力。选择接口时，先确定源、目的存储单元和执行核，再判断数据是否连续、是否对齐以及是否需要随路处理。不同接口中的同名参数可能采用不同单位，调用前应核对对应接口文档。
- [数据通路](data_path.md)：AI Core采用分层存储架构。计算单元包括Cube、Vector和Scalar；存储单元包括Global Memory（GM）、L1 Buffer、L0A/L0B/L0C Buffer、Unified Buffer（UB）、BiasTable Buffer、Fixpipe Buffer和SIMD Register File；MTE1、MTE2、MTE3和Fixpipe负责在不同存储单元之间传输数据。
- [总体约束说明](overall_constraints.md)：数据搬运接口共享地址对齐、存储容量、参数单位和同步等通用规则。调用前应先完成本节检查；具体接口文档给出的要求更严格时，以具体接口为准。
- [概览](overview.md)：数据搬运是算子开发中最基础且最关键的操作之一。在AI Core的分层存储架构中，输入数据需要从Global Memory（GM）搬入片上存储，计算结果也需要从片上存储搬回GM；矩阵计算和矢量计算之间交换中间结果时，还会涉及L1 Buffer、L0 Buffer、Unified Buffer（UB）和寄存器之间的数据流转。
