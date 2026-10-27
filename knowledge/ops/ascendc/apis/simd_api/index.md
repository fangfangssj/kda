# SIMD API

- [高阶API](adv_api/index.md)：共 498 张知识卡，分布在 15 个子目录（activation_functions、convolution_compute、cube_compute 等）。
- [基础API](basic_api/index.md)：共 580 张知识卡，分布在 17 个子目录（atomic_operations、aux_data_structures、cache_control 等）。
- [API列表](c_api/index.md)：C API文档按最细粒度公共头文件归类。除试验接口外，整体使用时可以包含c_api/asc_simd.h。（共 778 张知识卡：C API、通用说明和约束（C API） 等）
- [通用说明和约束（SIMD API）](general_description_and_constraints.md)：说明 SIMD 基础与高阶 API 的头文件、链接方式、存储单元对齐要求及 TPosition 内存映射。
- [Tensor API](tensor_api/index.md)：Tensor API是一套面向Ascend C算子开发的基础编程接口，使用Tensor和Layout统一描述数据的存储位置、数据类型、尺寸大小和排列规则，并基于这些信息完成数据搬运、矩阵计算和矢量计算，减少开发者手工组织硬件指令参数和数据排布的工作量。（共 148 张知识卡：Algorithm 等）
