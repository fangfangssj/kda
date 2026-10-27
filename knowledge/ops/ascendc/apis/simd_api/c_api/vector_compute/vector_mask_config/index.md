# 掩码操作

- [asc_set_mask_count](asc_set_mask_count.md)：设置Mask模式为Counter模式。该模式下，不需要开发者去感知迭代次数、处理非对齐的尾块等操作，可直接传入数据计算数量，实际迭代次数由Vector计算单元自动推断。在前n个数据计算类的接口内部会自动设置该模式，不需要自行设置。
- [asc_set_mask_norm](asc_set_mask_norm.md)：设置Mask模式为Normal模式，该模式为系统默认模式。该模式下，Mask参数可控制高维切分计算过程中单次迭代内参与计算的元素。
- [asc_set_vector_mask](asc_set_vector_mask.md)：本API用于设置Mask值。某些矢量计算接口需要提前设置Mask，用于指示哪些数据参与计算。 单独使用该API时可引入c_api/vector_compute/compute/vector_mask_config.h，整体使用C API时可引入c_api/asc_simd.h。
