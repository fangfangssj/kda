# 高阶API

- [激活函数](activation_functions/index.md)：共 35 张知识卡，分布在 9 个子目录（ge_glu_interface、gelu_interface、log_soft_max_interface 等）。
- [卷积计算](convolution_compute/index.md)：共 71 张知识卡，分布在 6 个子目录（conv3_d_backprop_filter_kernel、conv3_d_backprop_filter_tiling、conv3_d_backprop_input_kernel 等）。
- [矩阵计算](cube_compute/index.md)：共 96 张知识卡，分布在 3 个子目录（get_matmul_compute_space、matmul_kernel、matmul_tiling）。
- [数据过滤](data_filter/index.md)：共 4 张知识卡（如 DropOut、GetDropOutMaxMinTmpSize、GetSelectMaxMinTmpSize 等）。
- [数据结构](data_structures/index.md)：汇总高阶API在Host侧Tiling接口中使用的Ascend C公共类型，包含TensorShape、TensorDataType等数据结构定义。（共 3 张知识卡：TensorDataType、TensorShape）
- [HCCL通信类](hccl_communication/index.md)：共 57 张知识卡，分布在 4 个子目录（hccl_context、hccl_kernel、hccl_mc2 等）。
- [索引计算](index_compute/index.md)：共 2 张知识卡（如 Arange（索引计算）、GetArangeMaxMinTmpSize）。
- [数学计算](math_compute/index.md)：共 144 张知识卡（如 更多样例（数学计算） 等）。
- [归一化操作](normalization/index.md)：共 20 张知识卡（如 BatchNorm、BatchNorm Tiling、DeepNorm 等）。
- [量化操作](quantization/index.md)：共 18 张知识卡（如 AntiQuantize、AscendAntiQuant、AscendDequant 等）。
- [随机函数](random_functions/index.md)：基于Philox随机数生成算法，给定随机数种子，生成若干的随机数。 Philox随机数生成的核心算法是一个基于记数的伪随机数生成算法，输入为一个128bit的记数器C，两个32bit的key（k0和k1），输出为4个32bit的整数。
- [归约操作](reduction_operations/index.md)：共 21 张知识卡，分布在 10 个子目录（mean_interface、reduce_all_interface、reduce_any_interface 等）。
- [排序操作](sort_operations/index.md)：共 11 张知识卡（如 Concat（排序操作）、Extract（排序操作）、GetConcatTmpSize 等）。
- [张量变换](tensor_transform/index.md)：共 11 张知识卡（如 Broadcast（张量变换）、Fill（张量变换）、GetBroadCastMaxMinTmpSize 等）。

- [experimental](experimental/index.md)：共 4 张知识卡，分布在 2 个子目录（bessel_i0_interface、ndtri_interface）。
