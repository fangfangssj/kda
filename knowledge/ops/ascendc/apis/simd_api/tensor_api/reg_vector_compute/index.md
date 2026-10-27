# reg矢量计算（Tensor API）

- [基础算术](basic_arithmetic/index.md)：共 20 张知识卡（如 abs、add_carry、operator/ 等）。
- [概述](compare_and_select/index.md)：共 2 张知识卡（如 operator_compare、select）。
- [数据填充](data_padding/index.md)：头文件路径："include/tensor_api/experimental/arch/vector/compare_and_select.h"。
- [数据重排](data_reorder/index.md)：共 2 张知识卡（如 deinterleave、interleave）。
- [逻辑计算](logical_compute/index.md)：共 6 张知识卡（如 and、not、or 等）。
- [mask寄存器计算](mask_reg_compute/index.md)：共 7 张知识卡（如 all_mask、deinterleave(掩码解交织)、interleave（掩码交织） 等）。
- [寄存器数据搬入](reg_data_load/index.md)：共 4 张知识卡（如 load、load_broadcast、load_gather 等）。
- [寄存器数据搬出](reg_data_store/index.md)：共 3 张知识卡（如 store、store_scatter、unalign_store）。
- [reg_tensor](reg_tensor/index.md)：头文件路径为："tensor_api/arch/vector/experimental/reg_tensor.h"，命名空间为asc::te::experimental。（共 2 张知识卡：reg_pair）
- [类型转换](type_conversion/index.md)：共 2 张知识卡（如 cast、trunc）。
- [矢量计算的搬入](vector_compute_load/index.md)：共 3 张知识卡（如 copy（Global Memory到Unified Buffer的数据搬运）、copy（Unified Buffer到L1 Buffer的数据搬运）、copy（Unified Buffer到Unified Buffer的数据搬运））。
- [矢量计算的搬出](vector_compute_store/index.md)：头文件路径为：tensor_api/tensor.h，命名空间为asc::te。

- [复合计算](composite_compute/index.md)：本节介绍将多个计算步骤融合为一次调用的寄存器计算接口。（共 8 张知识卡：abs_diff、axpy、exp_diff 等）
- [直方图计算](histogram_compute/index.md)：本节介绍寄存器直方图统计接口。（共 2 张知识卡：histograms）
- [reduction compute](reduction_compute/index.md)：共 3 张知识卡（如 reduce_max、reduce_min、reduce_sum）。
