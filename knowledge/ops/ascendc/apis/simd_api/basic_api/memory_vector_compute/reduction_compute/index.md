# 归约计算

- [归约计算概述](introduction.md)：概述 Tensor 归约计算及 ReduceSum、ReduceMax、ReduceMin、ReducePairElem 等接口能力。
- [ReduceDataBlock](reduce_data_block.md)：对输入数据以DataBlock为单位进行归约操作，根据reduceType对每个DataBlock内数据求和、求最大值或求最小值。
- [ReduceMax](reduce_max.md)：从所有输入数据中找出最大值，可选返回最大值索引。
- [ReduceMin](reduce_min.md)：从所有输入数据中找出最小值，可选返回最小值索引。
- [ReducePairElem](reduce_pair_elem.md)：对相邻两个元素进行归约求和操作，结果按顺序写入目标地址。
- [ReduceRepeat](reduce_repeat.md)：对每个repeat内所有数据进行归约操作，支持求和、求最大值和求最小值，可返回最值索引。
- [ReduceSum](reduce_sum.md)：对所有输入数据进行求和操作。
