# Memory矢量计算

- [基础算术](basic_arithmetic/index.md)：共 29 张知识卡（如 Abs、Add、Adds 等）。
- [比较与选择](compare_and_select/index.md)：Compare、Compares称之为比较接口，用于逐元素比较Tensor或Scalar的大小，并根据比较结果生成掩码。Select、GatherMask称之为选择接口，用于根据掩码条件从源操作数中选取元素输出到目的操作数。推荐使用场景如下：（共 10 张知识卡：Compare、CompareStoreToRegister、Compares 等）
- [复合计算](composite_compute/index.md)：复合计算通常指的是在数学或编程中对复合函数或运算的处理。（共 15 张知识卡：AbsSub、AddDeqRelu、AddRelu 等）
- [数据排布转换](data_layout_conversion/index.md)：共 2 张知识卡（如 TransDataTo5HD、Transpose）。
- [数据搬运](data_move/index.md)：共 12 张知识卡（如 Copy、Copy、DataCopy 等）。
- [数据搬运辅助配置接口](data_move_aux_config/index.md)：共 4 张知识卡（如 NdDmaDci、ResetLoopModePara、SetLoopModePara 等）。
- [数据填充](data_padding/index.md)：共 4 张知识卡（如 Brcb、CreateVecIndex、Duplicate 等）。
- [数据重排（ISASI）](data_rearrange_isasi/index.md)：共 2 张知识卡（如 DeInterleave、Interleave）。
- [逻辑计算](logical_compute/index.md)：共 9 张知识卡（如 And、Ands、Not 等）。
- [掩码操作](mask_operations/index.md)：共 4 张知识卡（如 ResetMask、SetMaskCount、SetMaskNorm 等）。
- [归约计算](reduction_compute/index.md)：共 7 张知识卡（如 归约计算概述、ReduceDataBlock、ReduceMax 等）。
- [归约计算辅助配置接口](reduction_compute_aux_config/index.md)：共 2 张知识卡（如 GetReduceRepeatMaxMinSpr、GetReduceRepeatSumSpr）。
- [离散与聚合](scatter_gather/index.md)：共 3 张知识卡（如 Gather、Gatherb、Scatter）。
- [SIMD计算说明](simd_compute/index.md)：矢量计算通过Vector计算单元完成，矢量计算的源操作数和目的操作数均通过Unified Buffer（UB）来进行存储。（共 4 张知识卡：连续计算（SIMD计算）、高维切分（SIMD计算）、SIMD计算掩码）
- [排序组合（ISASI）](sort_and_merge_isasi/index.md)：Region Proposal（简称RP）是目标检测算法中用于定位潜在目标区域的候选框。Region Proposal采用特定的数据结构进行存储和管理，以支持高效的排序和筛选操作。在业务场景下，为了从大量Region Proposals中筛选出最有可能包含目标的区域，需要根据每个RP的置信度score进行排序。（共 8 张知识卡：GetMrgSortResult、MrgSort、MrgSort4（排序与合并） 等）
- [类型转换](type_conversion/index.md)：共 2 张知识卡（如 Cast、Truncate）。
- [类型转换辅助配置接口](type_conversion_aux_config/index.md)：设置DEQSCALE寄存器的值，用于Vector计算单元上的量化计算，支持不同场景的量化参数配置。
- [Vector逻辑架构](vector_logical_arch/index.md)：Vector计算单元专用于执行向量计算。如下图所示，高亮部分展示了Vector计算单元及其关联的存储单元。
