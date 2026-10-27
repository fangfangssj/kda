# API列表

- [原子操作](atomic/index.md)：共 20 张知识卡（如 原子操作概述、关键特性说明 等）。
- [API列表](c_api.md)：C API文档按最细粒度公共头文件归类。除试验接口外，整体使用时可以包含c_api/asc_simd.h。
- [缓存控制](cache_ctrl/index.md)：共 9 张知识卡（如 asc_datacache_preload、AscDcci、asc_dci（缓存控制） 等）。
- [矩阵计算](cube_compute/index.md)：共 19 张知识卡，分布在 5 个子目录（cube_compute_fractal_intro、mmad_compute、mmad_compute_aux_config 等）。
- [矩阵计算搬运](cube_datamove/index.md)：共 76 张知识卡，分布在 5 个子目录（cube_compute_load、cube_compute_store、cube_load_aux_config 等）。
- [基础定义](defs/index.md)：共 25 张知识卡，分布在 5 个子目录（constant、enum、macro 等）。
- [废弃接口](deprecated_interface/index.md)：废弃接口索引页，汇总矢量计算asc_abs_sync与原子操作asc_set_atomic_add_int等已废弃接口。（共 236 张知识卡：AscAbsSync、asc_add_relu_sync（废弃）、asc_add_scalar_sync（废弃） 等）
- [其他操作](experimental/index.md)：本接口为试验接口，在后续版本中可能会调整或改进，不保证后续兼容性。请开发者在使用过程中关注后续版本更新。
- [通用说明和约束（C API）](general_description_and_constraints.md)：AI Core内外包含多级存储单元，各存储单元的主要用途如下：
- [Reg矢量计算](reg_compute/index.md)：Reg矢量计算C API提供面向Reg矢量计算架构的寄存器级矢量计算能力，用户可以通过该API直接对芯片中涉及Vector计算的寄存器进行操作，实现更大的灵活性和更好的性能。Reg矢量计算C API与Memory矢量计算C API功能相似，但与Memory矢量计算C API的输入和输出数据通常位于Unified Buffer（UB）不同，Reg矢量计算C API的输入或输出数据可使用矢量数据寄存器vector_（dtype为占位符，表示half、float等C API支持的数据类型）和掩码寄存器vector_bool。对于计算类API，其功能是从给定的寄存器获取数据，进行计算，并将结果保存在给定的寄存器。对于搬运类API，其功能是实现UB和寄存器的数据搬运。由此可见，Reg矢量计算C API相较于Memory矢量计算C API，将数据搬运和Reg计算过程交给用户自主控制，从而实现更大的开发自由度。（共 194 张知识卡：关键特性说明 等）
- [标量计算](scalar_compute/index.md)：共 12 张知识卡，分布在 4 个子目录（scalar_bit、scalar_convert、scalar_load 等）。
- [特殊寄存器访问](spr/index.md)：共 7 张知识卡（如 asc_get_ctrl、asc_get_saturation_flag、asc_get_saturation_strategy 等）。
- [同步控制](sync/index.md)：共 22 张知识卡（如 系统同步能力概述 等）。
- [工具接口](utils/index.md)：共 18 张知识卡，分布在 3 个子目录（sys_init、sys_misc、sys_var）。
- [Memory矢量计算](vector_compute/index.md)：共 100 张知识卡，分布在 11 个子目录（vector_arith、vector_broadcast、vector_compare 等）。
- [矢量数据搬运](vector_datamove/index.md)：共 23 张知识卡（如 asc_copy、asc_copy_ub2l1、asc_copy_ub2ub 等）。

- [C API](c_api_reference.md)：C API开放芯片完备编程能力，支持以数组形式分配内存，一般基于指针编程。提供与业界一致的C语言编程体验。

- [数据搬运导览](data_move_guide/index.md)：本章介绍如何根据源、目的存储单元、数据排布和随路处理需求选择C API数据搬运接口，并以NPU架构版本3510的数据通路和接口能力为例进行说明。其他架构的产品支持情况和接口能力以具体接口文档为准。开发者需要显式管理地址、搬运参数、配置状态和流水同步。使用本章介绍的接口时，除接口文档另有说明外，均可包含头文件c_api/asc_simd.h。（共 13 张知识卡：GM与L1/L0数据搬运、GM与UB数据搬运、L1到BiasTable数据搬运 等）
