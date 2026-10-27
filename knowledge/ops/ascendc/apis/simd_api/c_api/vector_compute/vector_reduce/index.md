# 归约计算

- [asc_datablock_reduce_max](asc_datablock_reduce_max.md)：执行数据块内的求最大值规约（Reduce Max）操作。
- [asc_datablock_reduce_min](asc_datablock_reduce_min.md)：执行数据块内的求最小值规约（Reduce Min）操作。
- [asc_datablock_reduce_sum](asc_datablock_reduce_sum.md)：对每个DataBlock内所有元素求和。
- [asc_get_reduce_max_cnt](asc_get_reduce_max_cnt.md)：此接口用于获取执行asc_repeat_reduce_max操作后的最大值，以及第一个最大值时的索引。
- [asc_get_reduce_min_cnt](asc_get_reduce_min_cnt.md)：此接口用于获取执行asc_repeat_reduce_min操作后的最小值，以及第一个最小值时的索引。
- [asc_get_rsvd_count](asc_get_rsvd_count.md)：此接口用于获取执行asc_reduce操作后剩余的元素数量。
- [asc_pair_reduce_sum](asc_pair_reduce_sum.md)：对输入数据做归约操作，得到数据总和。相邻两个（奇偶）元素求和，例如（a1，a2，a3，a4，a5，a6…），相邻两个数据求和为（a1+a2，a3+a4，a5+a6…）。
- [asc_reduce](asc_reduce.md)：以内置固定模式对应的二进制或者用户自定义输入的数值对应的gather mask（数据收集的掩码），从源操作数中选取元素写入目的操作数中。
- [asc_repeat_reduce_max](asc_repeat_reduce_max.md)：对每个Repeat内所有元素求最大值，以及Repeat内的首个最大值元素在当前Repeat中的索引index。一次Repeat计算256字节。
- [asc_repeat_reduce_min](asc_repeat_reduce_min.md)：对每个Repeat内所有元素求最小值，以及Repeat内的首个最小值元素在当前Repeat中的索引index。一次Repeat计算256字节。
- [asc_repeat_reduce_sum](asc_repeat_reduce_sum.md)：对每个Repeat内所有元素求和。
