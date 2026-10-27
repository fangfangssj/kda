# Utils API

- [C++标准库](cpp_stdlib/index.md)：共 35 张知识卡，分布在 5 个子目录（algorithms、container_functions、general_utils 等）。
- [log](log/index.md)：提供Host侧打印Log功能，支持在TilingFunc中输出日志；Tiling下沉场景下必须使用本接口，日志需Dump落盘解析。
- [平台信息获取](platform_info/index.md)：共 17 张知识卡（如 PlatformAscendCManager 等）。
- [原型注册与管理](prototype_register_management/index.md)：共 79 张知识卡（如 原型注册接口（OP_ADD）、OpAICoreConfig注册接口（REGISTER_OP_AICORE_CONFIG） 等）。
- [RTC](rtc/index.md)：共 11 张知识卡（如 aclrtcAddNameExpr、aclrtcCompileProg、aclrtcCreateProg 等）。
- [SuperKernel](super_kernel/index.md)：本接口为算子SuperKernel场景提供绑定原核函数和SK子函数的能力。核函数直调算子的完整适配方法请参见核函数直调算子额外适配说明。
- [Tiling数据结构注册](tiling_data_structure_register/index.md)：共 2 张知识卡（如 TilingData结构定义、TilingData结构注册）。
- [Tiling下沉](tiling_sink/index.md)：在Tiling下沉场景中，该宏定义用于生成Tiling下沉的注册类，再通过调用注册类的成员函数来注册需要下沉的Tiling函数。
- [Tiling模板编程](tiling_template_programming/index.md)：共 3 张知识卡（如 ASCENDC_TPL_SEL_PARAM、GET_TPL_TILING_KEY、模板参数定义）。
- [Tiling调测](tiling_tuning/index.md)：共 22 张知识卡，分布在 2 个子目录（context_builder、op_tiling_registry）。
- [调测接口](tuning_interface/index.md)：共 11 张知识卡（如 __trap、asc_dump、asc_mark_stamp 等）。
