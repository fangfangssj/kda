# Tensor API

- [Algorithm](algorithm.md)：Tensor API的Algorithm层为基于Tensor的数据搬运和矩阵计算提供统一的调用入口。其中，copy和mmad接口分别用于执行数据搬运和矩阵计算，make_copy和make_mmad接口用于构造对应的Atom原子对象。
- [数据搬运](data_move/index.md)：Tensor API通过copy统一执行不同存储位置之间的数据搬运。Tensor表达不同存储位置和类型的数据，Layout表达Tensor的尺寸大小和排列规则。
- [废弃接口](deprecated_interface/index.md)：共 16 张知识卡（如 layout::Capacity（废弃）、layout::Get（废弃）、layout::Rank（废弃） 等）。
- [layout](layout/index.md)：layout数据结构是描述多维张量内存布局的基础模板类，通过编译期的形状和步长信息，实现逻辑坐标空间到一维内存地址空间的映射。借助模板元编程技术，该类在编译期完成计算和代码生成，从而降低运行时开销。（共 23 张知识卡：capacity、coord、coshape 等）
- [矩阵计算（Tensor API）](matrix_compute/index.md)：矩阵计算由数据搬入、mmad计算和结果搬出组成。各阶段通过Tensor的存储位置和Layout选择对应Arch通路，通过Atom配置Trait及运行时Params。（共 27 张知识卡：mmad计算辅助配置接口、概述 等）
- [reg矢量计算（Tensor API）](reg_vector_compute/index.md)：共 66 张知识卡，分布在 15 个子目录（basic_arithmetic、compare_and_select、composite_compute 等）。
- [tensor](tensor/index.md)：头文件路径为：tensor_api/tensor.h。（共 7 张知识卡：make_mem_ptr、make_tensor、pointer 等）
- [Tensor API](tensor_api.md)：Tensor API是一套面向Ascend C算子开发的基础编程接口，使用Tensor和Layout统一描述数据的存储位置、数据类型、尺寸大小和排列规则，并基于这些信息完成数据搬运、矩阵计算和矢量计算，减少开发者手工组织硬件指令参数和数据排布的工作量。
- [utils](utils/index.md)：共 6 张知识卡（如 编译期整型别名、维度保留标记、get_layout_pattern 等）。
