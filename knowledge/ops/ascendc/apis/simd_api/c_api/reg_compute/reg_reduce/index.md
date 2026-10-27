# 归约计算

- [asc_pair_reduce_sum](asc_pair_reduce_sum.md)：根据mask将源操作数src中每两个相邻元素相加，得到计算结果。结果连续保存在目的操作数dst的低半部分，dst的高半部分置0。以half数据类型为例，相加示意图如下：
- [asc_reduce_max](asc_reduce_max.md)：根据mask对源操作数src进行归约最大值操作，得到归约结果。结果保存在dst的第0个元素，最大值在src中的索引原始位模式保存在dst的第1个元素，dst中的其他元素置0。如果存在多个最大值，则保留最小的索引。计算公式如下：
- [asc_reduce_max_datablock](asc_reduce_max_datablock.md)：根据mask对每个DataBlock（32B）内的元素求最大值，得到各DataBlock的归约结果。结果依次保存在dst的低位。计算公式如下：
- [asc_reduce_min](asc_reduce_min.md)：根据mask对源操作数src进行归约最小值操作，得到归约结果。结果保存在dst的第0个元素，最小值在src中的索引原始位模式保存在dst的第1个元素，dst中的其他元素置0。如果存在多个最小值，则保留最小的索引。计算公式如下：
- [asc_reduce_min_datablock](asc_reduce_min_datablock.md)：根据mask对每个DataBlock（32B）内的元素求最小值，得到各DataBlock的归约结果。结果依次保存在dst的低位。计算公式如下：
- [asc_reduce_sum](asc_reduce_sum.md)：根据mask对源操作数src进行归约求和操作，用于将src中的所有参与计算的元素求和，得到归约结果。结果保存在dst中的0位置。参考伪代码：
- [asc_reduce_sum_datablock](asc_reduce_sum_datablock.md)：根据mask将每个DataBlock（32B）中参与计算的元素相加，得到各DataBlock的归约结果。结果依次保存在dst的最低位。参考伪代码：
