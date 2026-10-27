# 寄存器类型

- [Reg数据类型定义](data_type_definition.md)：矢量数据寄存器用于存储矢量数据，其长度为VL（Vector Length）。在Ascend 950PR/Ascend 950DT产品中，VL为256字节，可存储VL / sizeof(dtype)个dtype类型的元素。例如，vector_float可存储256 / sizeof(float) = 64个float类型的元素。

- [寄存器类型](reg_data_types.md)：Ascend 950PR&950DT系列产品（NPU架构版本3510）的C API提供矢量数据寄存器、掩码寄存器、非对齐搬运寄存器和地址寄存器类型。寄存器类型主要在AIV的Vector Function（__simd_vf__标记的函数）内使用，最细粒度公共头文件为c_api/defs/type.h。
